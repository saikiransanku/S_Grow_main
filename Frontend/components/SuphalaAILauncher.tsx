"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import {
  classifySuphalaIntent,
  routeForSuphalaIntent,
  savePendingSuphalaIntent,
  setPostAuthRedirect,
  type SuphalaIntentKind,
} from "@/lib/suphalaAI";

type SuphalaAILauncherVariant = "nav" | "hero" | "floating";

interface SuphalaAILauncherProps {
  variant?: SuphalaAILauncherVariant;
}

const QUICK_START_PROMPTS = [
  "Suggest the best crop for red soil this season",
  "What crop should I grow after cotton on the same land?",
  "My leaf has yellow spots and curling",
];

const buildTriggerClassName = (variant: SuphalaAILauncherVariant) => {
  if (variant === "hero") {
    return "inline-flex items-center justify-center gap-3 rounded-2xl bg-gradient-to-r from-emerald-500 to-teal-600 px-8 py-4 text-base font-semibold text-white shadow-2xl transition-all duration-300 hover:scale-[1.02] hover:shadow-emerald-500/40";
  }

  if (variant === "floating") {
    return "fixed bottom-5 right-4 z-40 inline-flex items-center gap-3 rounded-full bg-neutral-950 px-4 py-3 text-sm font-semibold text-white shadow-[0_22px_50px_-22px_rgba(15,23,42,0.85)] transition-transform duration-200 hover:-translate-y-0.5 md:hidden";
  }

  return "inline-flex items-center gap-3 rounded-full border border-emerald-200 bg-white px-3 py-2 text-sm font-semibold text-emerald-900 shadow-sm transition-colors hover:bg-emerald-50 dark:border-emerald-400/20 dark:bg-white/[0.04] dark:text-emerald-100 dark:hover:bg-emerald-400/10";
};

const getIntentSummary = (
  kind: SuphalaIntentKind,
  hasImages: boolean,
  hasPrompt: boolean,
) => {
  if (hasImages) {
    return "Leaf image detected. Suphala AI will open the Disease Predictor.";
  }

  if (!hasPrompt) {
    return "Type crop-planning keywords or upload a leaf image.";
  }

  return kind === "disease"
    ? "This looks like a disease question. Suphala AI will open the Disease Predictor."
    : "This looks like a crop-planning question. Suphala AI will open Crop Suggestions.";
};

export function SuphalaAILauncher({
  variant = "nav",
}: SuphalaAILauncherProps) {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const [isOpen, setIsOpen] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [error, setError] = useState("");

  const hasImages = selectedFiles.length > 0;
  const hasPrompt = Boolean(prompt.trim());
  const nextIntent = classifySuphalaIntent(prompt, hasImages);

  useEffect(() => {
    if (!isOpen) return;

    const previousBodyOverflow = document.body.style.overflow;
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsOpen(false);
      }
    };

    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", handleEscape);

    return () => {
      document.body.style.overflow = previousBodyOverflow;
      window.removeEventListener("keydown", handleEscape);
    };
  }, [isOpen]);

  const resetLauncher = () => {
    setPrompt("");
    setSelectedFiles([]);
    setError("");
  };

  const closeLauncher = () => {
    setIsOpen(false);
    resetLauncher();
  };

  const handleFiles = (files: FileList | File[]) => {
    const nextFiles = Array.from(files).filter((file) =>
      file.type.startsWith("image/"),
    );

    if (!nextFiles.length) {
      setError("Upload a leaf image to open the disease predictor.");
      return;
    }

    setSelectedFiles((prev) => [...prev, ...nextFiles].slice(0, 3));
    setError("");
  };

  const openLauncher = () => {
    setIsOpen(true);
    setError("");
  };

  const launchSuphalaAI = async () => {
    const cleanPrompt = prompt.replace(/\s+/g, " ").trim();

    if (!cleanPrompt && selectedFiles.length === 0) {
      setError("Enter a crop question or upload a leaf image first.");
      return;
    }

    const intentKind = classifySuphalaIntent(cleanPrompt, selectedFiles.length > 0);
    const nextPath = routeForSuphalaIntent(intentKind);

    savePendingSuphalaIntent({
      kind: intentKind,
      prompt: cleanPrompt,
      images: selectedFiles,
      imageCount: selectedFiles.length,
      createdAt: Date.now(),
    });

    closeLauncher();

    const hasToken =
      typeof window !== "undefined" && Boolean(window.localStorage.getItem("token"));

    if (!hasToken) {
      setPostAuthRedirect(nextPath);
      router.push("/login");
      return;
    }

    router.push(nextPath);
  };

  return (
    <>
      <button
        type="button"
        onClick={openLauncher}
        className={buildTriggerClassName(variant)}
        aria-label="Open Suphala AI"
      >
        <span className="flex h-9 w-9 items-center justify-center rounded-full bg-white/15 ring-1 ring-white/20 dark:bg-white/10">
          <img src="/ai-logo.png" alt="Suphala AI" className="h-5 w-5 object-contain" />
        </span>
        <span>{variant === "hero" ? "Open Suphala AI" : "Suphala AI"}</span>
      </button>

      {isOpen ? (
        <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/55 p-4 backdrop-blur-sm">
          <button
            type="button"
            className="absolute inset-0 cursor-default"
            aria-label="Close Suphala AI"
            onClick={closeLauncher}
          />

          <div className="relative z-10 w-full max-w-2xl rounded-[32px] border border-white/20 bg-[linear-gradient(145deg,rgba(250,255,252,0.98),rgba(236,252,243,0.96))] p-6 shadow-[0_35px_90px_-40px_rgba(15,23,42,0.8)] dark:border-white/10 dark:bg-[linear-gradient(145deg,rgba(16,24,20,0.98),rgba(17,24,39,0.98))]">
            <div className="flex items-start justify-between gap-4">
              <div className="max-w-xl">
                <p className="text-xs font-semibold uppercase tracking-[0.22em] text-emerald-700 dark:text-emerald-300">
                  Suphala AI
                </p>
                <h2 className="mt-3 text-2xl font-semibold tracking-[-0.04em] text-neutral-950 dark:text-white md:text-3xl">
                  One AI entry for crop advice and disease prediction
                </h2>
                <p className="mt-3 text-sm leading-7 text-neutral-600 dark:text-neutral-300">
                  Type crop-planning keywords for suggestions, or upload a leaf image
                  for disease analysis. Suphala AI will route you to the right
                  workspace.
                </p>
              </div>

              <button
                type="button"
                onClick={closeLauncher}
                className="flex h-11 w-11 items-center justify-center rounded-2xl bg-black/[0.04] text-neutral-700 transition-colors hover:bg-black/[0.08] hover:text-neutral-950 dark:bg-white/[0.06] dark:text-neutral-200 dark:hover:bg-white/[0.12] dark:hover:text-white"
                aria-label="Close launcher"
              >
                <svg
                  className="h-5 w-5"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M6 18 18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </div>

            <div className="mt-6 grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
              <div className="space-y-4">
                <div className="rounded-[28px] border border-emerald-200/70 bg-white/85 p-4 shadow-sm dark:border-white/10 dark:bg-white/[0.04]">
                  <label className="block text-sm font-semibold text-neutral-900 dark:text-white">
                    Ask Suphala AI
                  </label>
                  <textarea
                    value={prompt}
                    onChange={(event) => {
                      setPrompt(event.target.value);
                      setError("");
                    }}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && !event.shiftKey) {
                        event.preventDefault();
                        void launchSuphalaAI();
                      }
                    }}
                    rows={4}
                    placeholder="Example: suggest the best crop for black soil this season"
                    className="mt-3 w-full resize-none rounded-[24px] border border-black/8 bg-white px-4 py-3 text-sm leading-7 text-neutral-900 outline-none transition-colors focus:border-emerald-300 dark:border-white/10 dark:bg-black/20 dark:text-white dark:focus:border-emerald-300/40"
                  />

                  <div className="mt-4 flex flex-wrap gap-2">
                    {QUICK_START_PROMPTS.map((quickPrompt) => (
                      <button
                        key={quickPrompt}
                        type="button"
                        onClick={() => {
                          setPrompt(quickPrompt);
                          setError("");
                        }}
                        className="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-2 text-xs font-medium text-emerald-900 transition-colors hover:border-emerald-300 hover:bg-emerald-100 dark:border-emerald-400/15 dark:bg-emerald-400/10 dark:text-emerald-100 dark:hover:bg-emerald-400/15"
                      >
                        {quickPrompt}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="rounded-[28px] border border-dashed border-amber-300/80 bg-white/80 p-4 shadow-sm dark:border-amber-300/25 dark:bg-white/[0.04]">
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                    <div>
                      <p className="text-sm font-semibold text-neutral-900 dark:text-white">
                        Upload a leaf image
                      </p>
                      <p className="mt-1 text-sm text-neutral-600 dark:text-neutral-300">
                        Image uploads go straight to the disease predictor.
                      </p>
                    </div>

                    <button
                      type="button"
                      onClick={() => fileInputRef.current?.click()}
                      className="inline-flex items-center justify-center rounded-full bg-amber-400 px-4 py-2.5 text-sm font-semibold text-neutral-950 transition-colors hover:bg-amber-300"
                    >
                      Choose image
                    </button>
                  </div>

                  {selectedFiles.length ? (
                    <div className="mt-4 flex flex-wrap gap-2">
                      {selectedFiles.map((file, index) => (
                        <div
                          key={`${file.name}-${file.lastModified}-${index}`}
                          className="inline-flex items-center gap-2 rounded-full border border-black/8 bg-white px-3 py-2 text-xs font-medium text-neutral-700 dark:border-white/10 dark:bg-white/[0.05] dark:text-neutral-200"
                        >
                          <span className="max-w-[12rem] truncate">{file.name}</span>
                          <button
                            type="button"
                            onClick={() =>
                              setSelectedFiles((prev) =>
                                prev.filter((_, fileIndex) => fileIndex !== index),
                              )
                            }
                            className="text-neutral-400 transition-colors hover:text-red-500 dark:hover:text-red-300"
                            aria-label={`Remove ${file.name}`}
                          >
                            x
                          </button>
                        </div>
                      ))}
                    </div>
                  ) : null}
                </div>
              </div>

              <div className="rounded-[28px] border border-black/6 bg-[linear-gradient(180deg,rgba(255,255,255,0.98),rgba(240,253,248,0.92))] p-5 shadow-sm dark:border-white/10 dark:bg-[linear-gradient(180deg,rgba(15,23,42,0.92),rgba(15,23,42,0.98))]">
                <p className="text-xs font-semibold uppercase tracking-[0.22em] text-neutral-500 dark:text-neutral-400">
                  Smart Routing
                </p>

                <div className="mt-4 rounded-[24px] bg-emerald-50/90 p-4 dark:bg-emerald-400/10">
                  <p className="text-sm font-semibold text-emerald-900 dark:text-emerald-100">
                    {nextIntent === "disease"
                      ? "Destination: Disease Predictor"
                      : "Destination: Crop Suggestions"}
                  </p>
                  <p className="mt-2 text-sm leading-7 text-neutral-700 dark:text-neutral-200">
                    {getIntentSummary(nextIntent, hasImages, hasPrompt)}
                  </p>
                </div>

                <div className="mt-4 space-y-3 text-sm leading-7 text-neutral-600 dark:text-neutral-300">
                  <p>
                    Crop keywords like <span className="font-semibold text-neutral-900 dark:text-white">best crop</span>, <span className="font-semibold text-neutral-900 dark:text-white">intercrop</span>, and <span className="font-semibold text-neutral-900 dark:text-white">soil</span> open the crop advisor.
                  </p>
                  <p>
                    Leaf photos and disease keywords like <span className="font-semibold text-neutral-900 dark:text-white">spots</span>, <span className="font-semibold text-neutral-900 dark:text-white">yellow leaves</span>, and <span className="font-semibold text-neutral-900 dark:text-white">blight</span> open the disease predictor.
                  </p>
                </div>

                {error ? (
                  <p className="mt-4 rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-400/20 dark:bg-red-400/10 dark:text-red-200">
                    {error}
                  </p>
                ) : null}

                <button
                  type="button"
                  onClick={() => void launchSuphalaAI()}
                  className="mt-5 inline-flex min-h-12 w-full items-center justify-center rounded-full bg-neutral-950 px-5 text-sm font-semibold text-white transition-colors hover:bg-neutral-800 dark:bg-white dark:text-neutral-950 dark:hover:bg-neutral-100"
                >
                  Continue with Suphala AI
                </button>

                <p className="mt-3 text-xs text-neutral-500 dark:text-neutral-400">
                  If you are not signed in, we will take you to login first.
                </p>
              </div>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              multiple
              onChange={(event) => {
                handleFiles(event.target.files || []);
                event.target.value = "";
              }}
              className="hidden"
            />
          </div>
        </div>
      ) : null}
    </>
  );
}
