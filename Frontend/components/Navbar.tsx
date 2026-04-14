"use client";

import Link from "next/link";
import { useState, useEffect } from "react";
import { usePathname } from "next/navigation";
import { ThemeToggle } from "./ThemeToggle";

const AI_HISTORY_TOGGLE_EVENT = "ssgrow-ai-history-toggle";

export function Navbar() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const pathname = usePathname();
  const isAIPage = pathname.startsWith("/ai-grow");

  useEffect(() => {
    // Check if user is authenticated
    const token = localStorage.getItem("token");
    setIsAuthenticated(!!token);
  }, []);

  // Prevent body scroll when mobile menu is open
  useEffect(() => {
    if (isMobileMenuOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "unset";
    }
    return () => {
      document.body.style.overflow = "unset";
    };
  }, [isMobileMenuOpen]);

  const handleLogout = () => {
    localStorage.removeItem("token");
    setIsAuthenticated(false);
    window.location.href = "/";
  };

  const closeMobileMenu = () => {
    setIsMobileMenuOpen(false);
  };

  const profileButton = isAuthenticated ? (
    <Link href="/profile" aria-label="Open profile">
      <div className="flex h-11 w-11 items-center justify-center rounded-full bg-gradient-to-br from-emerald-500 to-teal-600 shadow-sm transition-all duration-200 hover:ring-2 hover:ring-emerald-400 hover:ring-offset-2 dark:hover:ring-offset-black">
        <svg
          className="h-6 w-6 text-white"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
          />
        </svg>
      </div>
    </Link>
  ) : (
    <Link
      href="/login"
      className="rounded-full bg-emerald-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-emerald-700"
    >
      Login
    </Link>
  );

  if (isAIPage) {
    return (
      <nav className="sticky top-0 z-50 border-b border-black/6 bg-[#f4f6f4]/92 py-2 backdrop-blur-xl dark:border-white/8 dark:bg-[#101010]/95">
        <div className="mx-auto flex h-16 w-full items-center justify-between px-4 sm:px-6 lg:px-8">
          <button
            type="button"
            onClick={() => window.dispatchEvent(new Event(AI_HISTORY_TOGGLE_EVENT))}
            aria-label="Open chat history"
            className="flex items-center gap-3 rounded-2xl px-1 py-1 text-left transition-colors hover:bg-black/[0.03] dark:hover:bg-white/[0.04]"
          >
            <img
              src="/logo.png"
              alt="SSGrow logo"
              className="h-auto w-28 object-contain sm:w-32 md:w-36"
            />
          </button>

          <div className="flex items-center gap-3">
            <div className="hidden sm:block">
              <ThemeToggle />
            </div>
            {profileButton}
          </div>
        </div>
      </nav>
    );
  }

  return (
    <nav className="py-2 sticky top-0 z-50 glass-effect dark:bg-black dark:backdrop-blur">
      <div className="w-full mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link href="/">
            <img
              src="/logo.png"
              alt="SS Argitech Logo"
              className="w-32 sm:w-40 md:w-44 h-auto cursor-pointer"
            />
          </Link>

          {/* Desktop Search Bar - Hidden on mobile */}
          <div className="hidden lg:flex flex-1 max-w-xl mx-4">
            <div className="relative group w-full">
              <input
                type="text"
                placeholder="Search laws, regulations, farming tips..."
                className="w-full px-5 py-3 pl-12 rounded-full bg-white dark:bg-slate-700 border-2 border-gray-200 dark:border-gray-600 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 shadow-md hover:shadow-lg hover:border-emerald-300 dark:hover:border-emerald-400 focus:border-emerald-500 dark:focus:border-emerald-400 focus:ring-2 focus:ring-emerald-200 dark:focus:ring-emerald-900 focus:outline-none transition-all duration-300"
              />
              <svg
                className="absolute left-4 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400 dark:text-gray-500 group-hover:text-emerald-600 dark:group-hover:text-emerald-400 transition-colors duration-300"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                />
              </svg>
            </div>
          </div>

          {/* Desktop AI Grow Button - Hidden on mobile */}
          <Link href="/ai-grow" className="hidden md:block">
            <div className="border border-emerald-200 dark:border-emerald-700 rounded-full p-2 flex justify-between items-center gap-2 hover:bg-emerald-50 dark:hover:bg-emerald-900/30 transition-colors duration-200 cursor-pointer">
              <span>
                <img className="w-8" src="/ai-logo.png" alt="ai-grow" />
              </span>
              <button className="text-emerald-700 dark:text-emerald-400 font-medium hover:text-emerald-900 dark:hover:text-emerald-300 transition-colors duration-200 whitespace-nowrap">
                SupalaAI
              </button>
            </div>
          </Link>

          {/* Desktop Right Menu - Hidden on mobile */}
          <div className="hidden md:flex items-center gap-3">
            <ThemeToggle />
            {!isAuthenticated ? (
              <>
                <Link href="/login">
                  <span className="text-gray-700 dark:text-gray-300 hover:text-emerald-600 dark:hover:text-emerald-400 font-medium transition-colors duration-200 cursor-pointer">
                    Login
                  </span>
                </Link>
                <Link href="/register" className="btn-primary btn-small">
                  Register
                </Link>
              </>
            ) : (
              <>
                <Link href="/dashboard">
                  <div className="w-10 h-10 rounded-full bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center cursor-pointer hover:ring-2 hover:ring-emerald-400 hover:ring-offset-2 dark:hover:ring-offset-slate-900 transition-all duration-200">
                    <svg
                      className="w-6 h-6 text-white"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                      />
                    </svg>
                  </div>
                </Link>
                <button
                  onClick={handleLogout}
                  className="btn-primary btn-small bg-red-600 hover:bg-red-700 dark:bg-red-700 dark:hover:bg-red-800"
                >
                  Logout
                </button>
              </>
            )}
          </div>

          {/* Mobile Menu Button */}
          <div className="flex md:hidden items-center gap-2">
            <ThemeToggle />
            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="p-2 rounded-lg text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-slate-700 transition-colors"
              aria-label="Toggle menu"
            >
              {isMobileMenuOpen ? (
                <svg
                  className="w-6 h-6"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              ) : (
                <svg
                  className="w-6 h-6"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 6h16M4 12h16M4 18h16"
                  />
                </svg>
              )}
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        {isMobileMenuOpen && (
          <div className="md:hidden mt-4 pb-4 border-t border-gray-200 dark:border-gray-700">
            {/* Mobile Search */}
            <div className="relative group mt-4">
              <input
                type="text"
                placeholder="Search..."
                className="w-full px-4 py-2 pl-10 rounded-lg bg-white dark:bg-slate-700 border-2 border-gray-200 dark:border-gray-600 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:border-emerald-500 dark:focus:border-emerald-400 focus:ring-2 focus:ring-emerald-200 dark:focus:ring-emerald-900 focus:outline-none transition-all"
              />
              <svg
                className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                />
              </svg>
            </div>

            {/* Mobile AI Grow Link */}
            <Link href="/ai-grow" onClick={closeMobileMenu}>
              <div className="mt-3 p-3 border border-emerald-200 dark:border-emerald-700 rounded-lg flex items-center gap-2 hover:bg-emerald-50 dark:hover:bg-emerald-900/30 transition-colors">
                <img className="w-6 h-6" src="/ai-logo.png" alt="ai-grow" />
                <span className="text-emerald-700 dark:text-emerald-400 font-medium">
                  SupalaAI
                </span>
              </div>
            </Link>

            {/* Mobile Auth Links */}
            <div className="mt-4 space-y-3">
              {!isAuthenticated ? (
                <>
                  <Link href="/login" onClick={closeMobileMenu}>
                    <div className="w-full text-center py-2 px-4 rounded-lg text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-slate-700 font-medium transition-colors">
                      Login
                    </div>
                  </Link>
                  <Link href="/register" onClick={closeMobileMenu}>
                    <button className="w-full btn-primary btn-small">
                      Register
                    </button>
                  </Link>
                </>
              ) : (
                <>
                  <Link href="/dashboard" onClick={closeMobileMenu}>
                    <div className="w-full flex items-center justify-center gap-2 py-2 px-4 rounded-lg bg-emerald-50 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-400 font-medium transition-colors">
                      <svg
                        className="w-5 h-5"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                        />
                      </svg>
                      Dashboard
                    </div>
                  </Link>
                  <button
                    onClick={() => {
                      handleLogout();
                      closeMobileMenu();
                    }}
                    className="w-full btn-primary btn-small bg-red-600 hover:bg-red-700 dark:bg-red-700 dark:hover:bg-red-800"
                  >
                    Logout
                  </button>
                </>
              )}
            </div>
          </div>
        )}
      </div>
    </nav>
  );
}
