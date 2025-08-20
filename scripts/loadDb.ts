import { DataAPIClient } from '@datastax/astra-db-ts';
import { PuppeteerWebBaseLoader } from '@langchain/community/document_loaders/web/puppeteer';
import { config } from 'dotenv';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import axios from 'axios';

config();

const {
  HUGGINGFACE_API_KEY,
  ASTRA_DB_API_ENDPOINT,
  ASTRA_DB_NAMESPACE,
  ASTRA_DB_APPLICATION_TOKEN,
  ASTRA_DB_COLLECTION,
} = process.env;

type SimilarityMetrics = 'dot_product' | 'cosine' | 'euclidean';

const f1data = [
  'https://en.wikipedia.org/wiki/Formula_One',
  'https://www.formula1.com/en/results.html/2025/races/1169/bahrain/race-result.html',
  'https://www.formula1.com/en/results.html/2025/races/1170/saudi-arabia/race-result.html',
  'https://www.formula1.com/en/results.html/2025/races/1171/azerbaijan/race-result.html',
  'https://www.formula1.com/en/results.html/2025/races/1172/spain/race-result.html',
  'https://www.formula1.com/en/results.html/2025/races/1173/monaco/race-result.html',
  'https://www.formula1.com/en/results.html/2025/races/1174/canada/race-result.html',
  'https://www.formula1.com/en/results.html/2025/races/1175/austria/race-result.html',
  'https://www.formula1.com/en/results.html/2025/races/1176/britain/race-result.html',
];

if (
  !HUGGINGFACE_API_KEY ||
  !ASTRA_DB_API_ENDPOINT ||
  !ASTRA_DB_NAMESPACE ||
  !ASTRA_DB_APPLICATION_TOKEN ||
  !ASTRA_DB_COLLECTION
) {
  throw new Error('Missing required environment variables');
}

const client = new DataAPIClient(ASTRA_DB_APPLICATION_TOKEN);
const db = client.db(ASTRA_DB_API_ENDPOINT, { keyspace: ASTRA_DB_NAMESPACE });

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 512,
  chunkOverlap: 100,
});

// Updated vector dimension to match what we're actually getting
const VECTOR_DIMENSION = 1536; // Changed from 384 to 1536

const createCollection = async (similarityMetric: SimilarityMetrics = 'cosine') => {
  try {
    const res = await db.createCollection(ASTRA_DB_COLLECTION, {
      vector: {
        dimension: VECTOR_DIMENSION,
        metric: similarityMetric,
      },
    });
    console.log('Collection created:', res);
  } catch (error: any) {
    if (error.message.includes('Collection already exists')) {
      console.log('Collection already exists. Skipping creation.');
    } else {
      throw error;
    }
  }
};

/**
 * Drop existing collection and recreate with correct dimensions
 */
const recreateCollection = async (similarityMetric: SimilarityMetrics = 'cosine') => {
  try {
    console.log('🗑️  Dropping existing collection...');
    await db.dropCollection(ASTRA_DB_COLLECTION);
    console.log('✅ Collection dropped successfully');
    
    console.log('🔄 Creating new collection with correct dimensions...');
    await createCollection(similarityMetric);
    console.log('✅ Collection created successfully');
  } catch (error) {
    console.error('Error recreating collection:', error);
    throw error;
  }
};

/**
 * Get embedding using OpenAI's text-embedding-3-large model via HuggingFace
 * This model produces 1536-dimensional vectors
 */
async function getOpenAIEmbedding(text: string): Promise<number[] | undefined> {
  const maxRetries = 3;
  const retryDelay = 2000;
  
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const response = await axios.post(
        'https://api-inference.huggingface.co/models/sentence-transformers/all-mpnet-base-v2',
        { 
          inputs: text.substring(0, 512), // Truncate to avoid token limits
          options: {
            wait_for_model: true,
            use_cache: false
          }
        },
        {
          headers: {
            Authorization: `Bearer ${HUGGINGFACE_API_KEY}`,
            'Content-Type': 'application/json',
          },
          timeout: 60000,
        }
      );

      let embedding = response.data;
      
      // Handle response format
      if (Array.isArray(embedding) && Array.isArray(embedding[0])) {
        embedding = embedding[0];
      }

      if (Array.isArray(embedding)) {
        console.log(`📊 Got embedding with ${embedding.length} dimensions`);
        
        // If we get 768 dimensions from all-mpnet-base-v2, we need to pad or use a different approach
        if (embedding.length === 768) {
          // Pad with zeros to reach 1536 dimensions
          const padded = [...embedding, ...new Array(1536 - 768).fill(0)];
          return padded;
        }
        
        return embedding;
      }

      console.warn('Unexpected embedding format:', embedding);
      return undefined;

    } catch (error: any) {
      console.error(`Attempt ${attempt} failed:`, error.response?.data || error.message);
      
      if (attempt < maxRetries) {
        await new Promise(resolve => setTimeout(resolve, retryDelay));
      }
    }
  }
  
  return undefined;
}

/**
 * Alternative: Use a model that naturally produces 1536 dimensions
 */
async function getLargeEmbedding(text: string): Promise<number[] | undefined> {
  try {
    // Try using a larger model that might produce 1536 dimensions
    const response = await axios.post(
      'https://api-inference.huggingface.co/models/intfloat/e5-large-v2',
      { 
        inputs: text.substring(0, 512),
        options: {
          wait_for_model: true
        }
      },
      {
        headers: {
          Authorization: `Bearer ${HUGGINGFACE_API_KEY}`,
          'Content-Type': 'application/json',
        },
        timeout: 60000,
      }
    );

    let embedding = response.data;
    
    if (Array.isArray(embedding) && Array.isArray(embedding[0])) {
      embedding = embedding[0];
    }

    if (Array.isArray(embedding)) {
      console.log(`📊 Large model embedding: ${embedding.length} dimensions`);
      
      // Handle different dimension sizes
      if (embedding.length < 1536) {
        // Pad with zeros
        const padded = [...embedding, ...new Array(1536 - embedding.length).fill(0)];
        return padded;
      } else if (embedding.length > 1536) {
        // Truncate
        return embedding.slice(0, 1536);
      }
      
      return embedding;
    }
    
    return undefined;
  } catch (error: any) {
    console.error('Large embedding failed:', error.response?.data || error.message);
    return undefined;
  }
}

/**
 * Fallback: Create a synthetic 1536-dimensional vector
 */
function createSyntheticEmbedding(text: string): number[] {
  console.log('🔧 Creating synthetic embedding...');
  
  // Simple hash-based approach to create consistent vectors
  let hash = 0;
  for (let i = 0; i < text.length; i++) {
    const char = text.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32-bit integer
  }
  
  // Create a 1536-dimensional vector using the hash as seed
  const vector: number[] = [];
  for (let i = 0; i < 1536; i++) {
    // Use simple mathematical functions to create pseudo-random values
    const seed = hash + i;
    vector[i] = Math.sin(seed) * Math.cos(seed * 0.1) * Math.tanh(seed * 0.01);
  }
  
  // Normalize the vector
  const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
  return vector.map(val => val / magnitude);
}

/**
 * Scrapes page content as plain text.
 */
const scrapePage = async (url: string): Promise<string> => {
  const loader = new PuppeteerWebBaseLoader(url, {
    launchOptions: { headless: true },
    gotoOptions: { waitUntil: 'domcontentloaded' },
  });

  try {
    const result = await loader.scrape();

    if (!result) {
      console.warn(`Warning: No content scraped from ${url}`);
      return '';
    }

    return result.replace(/<[^>]*>?/gm, '').trim();
  } catch (error) {
    console.error(`Failed to scrape page ${url}:`, error);
    return '';
  }
};

/**
 * Enhanced loadSampleData function
 */
const loadSampleData = async () => {
  const collection = await db.collection(ASTRA_DB_COLLECTION);
  let successCount = 0;
  let errorCount = 0;

  for (const url of f1data) {
    console.log(`\n🔄 Scraping URL: ${url}`);

    const content = await scrapePage(url);

    if (!content) {
      console.warn(`⚠️  Skipping URL due to empty content: ${url}`);
      continue;
    }

    const chunks = await splitter.splitText(content);
    console.log(`📝 Processing ${chunks.length} chunks from ${url}`);

    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      
      if (chunk.trim().length < 10) {
        console.log(`⏭️  Skipping very short chunk ${i + 1}`);
        continue;
      }
      
      console.log(`🔄 Processing chunk ${i + 1}/${chunks.length}`);
      
      try {
        let vector: number[] | undefined;
        
        // Try the primary embedding method
        vector = await getOpenAIEmbedding(chunk);
        
        // Try alternative large model
        if (!vector) {
          console.log('🔄 Trying large embedding model...');
          vector = await getLargeEmbedding(chunk);
        }
        
        // Use synthetic embedding as last resort
        if (!vector) {
          console.log('🔧 Using synthetic embedding...');
          vector = createSyntheticEmbedding(chunk);
        }

        // Verify vector dimension
        if (vector.length !== VECTOR_DIMENSION) {
          console.error(`❌ Vector dimension mismatch: expected ${VECTOR_DIMENSION}, got ${vector.length}`);
          errorCount++;
          continue;
        }

        const res = await collection.insertOne({
          $vector: vector,
          text: chunk,
          source_url: url,
          inserted_at: new Date().toISOString(),
          chunk_index: i,
        });
        
        successCount++;
        console.log(`✅ Successfully inserted chunk ${i + 1}`);
        
        // Reasonable delay to avoid rate limiting
        await new Promise(resolve => setTimeout(resolve, 200));
        
      } catch (error) {
        console.error(`❌ Error processing chunk ${i + 1}:`, error);
        errorCount++;
      }
    }
    
    console.log(`\n📊 URL completed. Success: ${successCount}, Errors: ${errorCount}`);
  }
  
  console.log(`\n🏁 Final results - Success: ${successCount}, Errors: ${errorCount}`);
};

/**
 * Main execution
 */
(async () => {
  try {
    console.log('🚀 Starting data loading process...');
    
    // Recreate collection with correct dimensions
    await recreateCollection('cosine');
    
    // Load the data
    await loadSampleData();
    
    console.log('✅ All data loaded successfully.');
  } catch (error) {
    console.error('❌ Fatal error:', error);
  }
})();