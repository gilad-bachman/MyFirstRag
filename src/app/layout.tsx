import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "bachrag",
  description: "A quiet research agent for Gilad Bachman.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
