import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "F1 Live Dashboard",
  description: "Formula 1 live telemetry dashboard with cockpit view",
  icons: { icon: "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🏎️</text></svg>" },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-f1-dark text-white min-h-screen antialiased">
        {children}
      </body>
    </html>
  );
}
