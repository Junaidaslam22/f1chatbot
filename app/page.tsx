'use client';

import Image from 'next/image';
import f1gpt from './assets/f1gpt.jpg';
// import { useChat } from 'ai/react'; // From `ai` package
import { useChat } from 'ai/react'; // ✅ Note the '/react'

import {  } from 'ai/react'; // ✅ Note the '/react'



export default function Home() {
  const {
    messages,
    input,
    handleInputChange,
    handleSubmit,
    isLoading,
  } = useChat();

  return (
    <main>
      <Image src={f1gpt} width={250} alt="f1gpt logo" />

      <form onSubmit={handleSubmit}>
        <input
          className='question-box'
          onChange={handleInputChange}
          value={input}
          placeholder='Ask me something...'
        />
        <button type='submit' disabled={isLoading}>
          Send
        </button>
      </form>

      <div>
        {messages.map((msg, idx) => (
          <p key={idx}>
            <strong>{msg.role}:</strong> {msg.content}
          </p>
        ))}
      </div>
    </main>
  );
}
