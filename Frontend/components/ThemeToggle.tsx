"use client";

import { useTheme } from "@/app/providers";

export function ThemeToggle() {
  const { theme, toggleTheme } = useTheme();

  return (
    <button
      onClick={toggleTheme}
      className="rounded-xl bg-gray-200 p-2 text-gray-800 transition-colors duration-300 hover:bg-gray-300 dark:bg-neutral-900 dark:text-gray-200 dark:hover:bg-neutral-800"
      aria-label="Toggle theme"
    >
      {theme === "light" ? (
        <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
          <path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z" />
        </svg>
      ) : (
        <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
          <path
            fillRule="evenodd"
            d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414L13.536 9.172a1 1 0 11-1.414-1.414l1.293-1.293a1 1 0 011.414 0zM9 15a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zm7-4a1 1 0 11-2 0 1 1 0 012 0zM9 4.5a1 1 0 100-2 1 1 0 000 2z"
            clipRule="evenodd"
          />
        </svg>
      )}
    </button>
  );
}
