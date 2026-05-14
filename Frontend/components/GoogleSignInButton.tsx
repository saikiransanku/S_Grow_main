"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { apiClient } from "@/lib/api";
import { consumePostAuthRedirect } from "@/lib/suphalaAI";

declare global {
  interface Window {
    google?: {
      accounts: {
        id: {
          initialize: (options: {
            client_id: string;
            callback: (response: { credential: string }) => void;
            auto_select?: boolean;
            cancel_on_tap_outside?: boolean;
            ux_mode?: "popup" | "redirect";
          }) => void;
          renderButton: (
            element: HTMLElement,
            options: {
              theme?: "outline" | "filled_blue" | "filled_black";
              size?: "large" | "medium" | "small";
              text?:
                | "signin_with"
                | "signup_with"
                | "continue_with"
                | "signin";
              shape?: "rectangular" | "pill" | "circle" | "square";
              width?: number;
              logo_alignment?: "left" | "center";
            },
          ) => void;
        };
      };
    };
  }
}

const GOOGLE_SCRIPT_SRC = "https://accounts.google.com/gsi/client";
const DEFAULT_GOOGLE_CLIENT_ID =
  "389731022642-pf5bg3jqevju3b2vh0gkbm03a1gkrce3.apps.googleusercontent.com";
let googleScriptPromise: Promise<void> | null = null;

function loadGoogleScript(): Promise<void> {
  if (typeof window === "undefined") {
    return Promise.reject(new Error("Google Sign-In requires a browser"));
  }

  if (window.google?.accounts?.id) {
    return Promise.resolve();
  }

  if (!googleScriptPromise) {
    googleScriptPromise = new Promise<void>((resolve, reject) => {
      const existing = document.querySelector<HTMLScriptElement>(
        `script[src="${GOOGLE_SCRIPT_SRC}"]`,
      );

      if (existing) {
        existing.addEventListener("load", () => resolve(), { once: true });
        existing.addEventListener(
          "error",
          () => reject(new Error("Failed to load Google Sign-In script")),
          { once: true },
        );
        return;
      }

      const script = document.createElement("script");
      script.src = GOOGLE_SCRIPT_SRC;
      script.async = true;
      script.defer = true;
      script.onload = () => resolve();
      script.onerror = () =>
        reject(new Error("Failed to load Google Sign-In script"));
      document.head.appendChild(script);
    });
  }

  return googleScriptPromise;
}

interface GoogleSignInButtonProps {
  onError: (message: string) => void;
  redirectTo?: string;
  text?: "signin_with" | "signup_with" | "continue_with" | "signin";
}

export default function GoogleSignInButton({
  onError,
  redirectTo = "/dashboard",
  text = "continue_with",
}: GoogleSignInButtonProps) {
  const router = useRouter();
  const containerRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLDivElement>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    const clientId =
      process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID?.trim() ||
      DEFAULT_GOOGLE_CLIENT_ID;

    if (!clientId) {
      onError("Google Sign-In is not configured in the frontend.");
      return;
    }

    let active = true;

    const renderGoogleButton = async () => {
      try {
        onError("");
        await loadGoogleScript();
        if (!active || !window.google?.accounts?.id || !buttonRef.current) {
          return;
        }

        window.google.accounts.id.initialize({
          client_id: clientId,
          ux_mode: "popup",
          cancel_on_tap_outside: true,
          callback: async ({ credential }) => {
            setIsSubmitting(true);
            onError("");

            try {
              const response = await apiClient.post("/auth/google", {
                credential,
              });
              const token = response.data?.data?.token;

              if (!token) {
                throw new Error("Google login returned no token.");
              }

              localStorage.setItem("token", token);
              router.push(consumePostAuthRedirect() || redirectTo);
            } catch (error: any) {
              const message =
                error?.response?.data?.message ||
                error?.response?.data?.error ||
                error?.message ||
                "Google login failed";
              onError(message);
              setIsSubmitting(false);
            }
          },
        });

        buttonRef.current.innerHTML = "";
        window.google.accounts.id.renderButton(buttonRef.current, {
          theme: "outline",
          size: "large",
          text,
          shape: "rectangular",
          width: containerRef.current?.offsetWidth || 240,
          logo_alignment: "left",
        });
        setIsReady(true);
      } catch (error: any) {
        setIsReady(false);
        const message = error?.message || "Failed to initialize Google Sign-In";
        onError(message);
      }
    };

    void renderGoogleButton();

    return () => {
      active = false;
    };
  }, [onError, redirectTo, router, text]);

  return (
    <div className="w-full">
      <div ref={containerRef} className="relative w-full">
        <div
          className={`flex min-h-[44px] w-full items-center justify-center gap-2 rounded-lg border-2 border-gray-200 px-4 py-2 text-sm font-medium text-gray-700 transition-colors dark:border-gray-600 dark:text-gray-200 ${
            isSubmitting
              ? "cursor-wait bg-gray-50 opacity-70 dark:bg-slate-600"
              : "cursor-pointer hover:bg-gray-50 dark:hover:bg-slate-600"
          }`}
          onClick={() => {
            if (!isReady && !isSubmitting) {
              onError("Google Sign-In is still loading. Try again in a moment.");
            }
          }}
          role="button"
          tabIndex={0}
          onKeyDown={(event) => {
            if ((event.key === "Enter" || event.key === " ") && !isReady) {
              event.preventDefault();
              onError("Google Sign-In is still loading. Try again in a moment.");
            }
          }}
          aria-label="Continue with Google"
        >
          <svg viewBox="0 0 24 24" className="h-5 w-5" aria-hidden="true">
            <path
              fill="#EA4335"
              d="M12 10.2v3.9h5.5c-.2 1.3-1.5 3.9-5.5 3.9-3.3 0-6-2.7-6-6s2.7-6 6-6c1.9 0 3.1.8 3.8 1.4l2.6-2.5C16.8 3.4 14.6 2.5 12 2.5A9.5 9.5 0 1 0 12 21.5c5.5 0 9.1-3.9 9.1-9.3 0-.6-.1-1.1-.2-1.6H12Z"
            />
            <path
              fill="#34A853"
              d="M3.9 7.5 7.1 9.8c.9-2 2.8-3.4 4.9-3.4 1.9 0 3.1.8 3.8 1.4l2.6-2.5C16.8 3.4 14.6 2.5 12 2.5c-3.6 0-6.9 2-8.5 5Z"
            />
            <path
              fill="#4A90E2"
              d="M12 21.5c2.5 0 4.7-.8 6.3-2.3l-2.9-2.4c-.8.6-1.9 1.1-3.4 1.1-3.9 0-5.2-2.6-5.5-3.9l-3.1 2.4c1.6 3.1 4.8 5.1 8.6 5.1Z"
            />
            <path
              fill="#FBBC05"
              d="M6.5 14c-.2-.6-.4-1.2-.4-2s.1-1.4.4-2L3.4 7.6A9.5 9.5 0 0 0 2.5 12c0 1.6.4 3.1.9 4.4L6.5 14Z"
            />
          </svg>
          <span>{isSubmitting ? "Signing in..." : "Google"}</span>
        </div>
        <div
          ref={buttonRef}
          className={isReady ? "absolute inset-0 overflow-hidden rounded-lg opacity-0" : "hidden"}
          aria-hidden="true"
        />
      </div>
      {isSubmitting ? (
        <p className="mt-2 text-center text-xs text-gray-500 dark:text-gray-400">
          Signing in with Google...
        </p>
      ) : null}
    </div>
  );
}
