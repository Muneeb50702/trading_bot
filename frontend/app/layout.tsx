import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "AI Futures Trading Bot",
  description: "Probability-based crypto futures signals with confidence scoring",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
