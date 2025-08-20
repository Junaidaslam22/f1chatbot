// app/layout.tsx
import './global.css';
import { ReactNode } from 'react';

export const metadata = {
  title: 'F2GPT',
  description: 'Your all questions for F1 GP racing',
};

const RootLayout = ({ children }: { children: ReactNode }) => {
  return (
    <html lang="en">
      <body>
        {children}
      </body>
    </html>
  );
};

export default RootLayout;
