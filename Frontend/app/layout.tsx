import "./globals.css";
import { ThemeProvider } from "./providers";
import { Navbar } from "@/components/Navbar";

export const metadata = {
  title: "SS Argitech - Farmer Portal",
  description: "Empowering farmers with knowledge and resources",
  icons: {
    icon: "/ai-logo.png",
    shortcut: "/ai-logo.png",
    apple: "/ai-logo.png",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              try {
                const theme = localStorage.getItem('theme') || 
                  (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
                if (theme === 'dark') {
                  document.documentElement.classList.add('dark');
                } else {
                  document.documentElement.classList.remove('dark');
                }
              } catch (e) {}
            `,
          }}
        />
      </head>
      <body suppressHydrationWarning>
        <ThemeProvider>
          <Navbar />
          {children}
        </ThemeProvider>
      </body>
    </html>
  );
}
