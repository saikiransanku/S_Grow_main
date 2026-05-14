"use client";

export type SuphalaIntentKind = "suggestion" | "disease";

export interface PendingSuphalaIntent {
  kind: SuphalaIntentKind;
  prompt: string;
  imageCount: number;
  images: File[];
  createdAt: number;
}

interface StoredPendingSuphalaIntent {
  kind: SuphalaIntentKind;
  prompt: string;
  imageCount: number;
  createdAt: number;
}

const PENDING_INTENT_STORAGE_KEY = "ssgrow-suphala-pending-intent";
const POST_AUTH_REDIRECT_STORAGE_KEY = "ssgrow-suphala-post-auth-redirect";
const PENDING_INTENT_TTL_MS = 30 * 60 * 1000;

const SUGGESTION_KEYWORDS = [
  "best crop",
  "best crops",
  "crop suggestion",
  "crop suggestions",
  "crop recommendation",
  "crop recommendations",
  "recommend crop",
  "recommend crops",
  "suggest crop",
  "suggest crops",
  "which crop",
  "what crop",
  "grow after",
  "same land",
  "intercrop",
  "intercropping",
  "crop rotation",
  "farm plan",
  "land",
  "soil",
  "water",
  "season",
  "profit",
  "yield",
  "suitable crop",
];

const DISEASE_KEYWORDS = [
  "leaf",
  "leaf spot",
  "disease",
  "diseased",
  "disease predictor",
  "diagnose",
  "diagnosis",
  "symptom",
  "symptoms",
  "yellow leaf",
  "yellow leaves",
  "blight",
  "rust",
  "mildew",
  "rot",
  "wilt",
  "wilting",
  "curl",
  "fungus",
  "fungal",
  "infection",
  "infected",
  "spots",
  "patches",
  "photo",
  "image",
  "upload",
  "camera",
  "pest",
];

let pendingIntentMemory: PendingSuphalaIntent | null = null;

const normalizePrompt = (value: string) =>
  value.replace(/\s+/g, " ").trim().toLowerCase();

const isClient = () => typeof window !== "undefined";

const isFreshIntent = (createdAt: number) =>
  Date.now() - createdAt <= PENDING_INTENT_TTL_MS;

const scoreKeywords = (text: string, keywords: string[]) =>
  keywords.reduce((score, keyword) => {
    if (!text.includes(keyword)) return score;
    return score + (keyword.includes(" ") ? 2 : 1);
  }, 0);

const clearPendingIntentStorage = () => {
  pendingIntentMemory = null;
  if (!isClient()) return;
  window.sessionStorage.removeItem(PENDING_INTENT_STORAGE_KEY);
};

export const classifySuphalaIntent = (
  prompt: string,
  hasImages = false,
): SuphalaIntentKind => {
  if (hasImages) return "disease";

  const normalizedPrompt = normalizePrompt(prompt);
  if (!normalizedPrompt) return "suggestion";

  const suggestionScore = scoreKeywords(normalizedPrompt, SUGGESTION_KEYWORDS);
  const diseaseScore = scoreKeywords(normalizedPrompt, DISEASE_KEYWORDS);

  if (diseaseScore > suggestionScore) {
    return "disease";
  }

  return "suggestion";
};

export const routeForSuphalaIntent = (kind: SuphalaIntentKind) =>
  kind === "disease" ? "/ai-grow" : "/suggestion-ai";

export const savePendingSuphalaIntent = (intent: PendingSuphalaIntent) => {
  pendingIntentMemory = intent;

  if (!isClient()) return;

  const storedIntent: StoredPendingSuphalaIntent = {
    kind: intent.kind,
    prompt: intent.prompt,
    imageCount: intent.imageCount,
    createdAt: intent.createdAt,
  };

  try {
    window.sessionStorage.setItem(
      PENDING_INTENT_STORAGE_KEY,
      JSON.stringify(storedIntent),
    );
  } catch {
    // The in-memory fallback still works for same-session client navigation.
  }
};

export const consumePendingSuphalaIntent = (
  expectedKind?: SuphalaIntentKind,
): PendingSuphalaIntent | null => {
  const memoryIntent = pendingIntentMemory;
  if (memoryIntent) {
    if (expectedKind && memoryIntent.kind !== expectedKind) {
      return null;
    }
    if (!isFreshIntent(memoryIntent.createdAt)) {
      clearPendingIntentStorage();
      return null;
    }

    clearPendingIntentStorage();
    return memoryIntent;
  }

  if (!isClient()) return null;

  try {
    const rawIntent = window.sessionStorage.getItem(PENDING_INTENT_STORAGE_KEY);
    if (!rawIntent) return null;

    const storedIntent = JSON.parse(rawIntent) as StoredPendingSuphalaIntent;
    if (expectedKind && storedIntent.kind !== expectedKind) {
      return null;
    }
    if (!isFreshIntent(storedIntent.createdAt)) {
      clearPendingIntentStorage();
      return null;
    }

    clearPendingIntentStorage();
    return {
      ...storedIntent,
      images: [],
    };
  } catch {
    clearPendingIntentStorage();
    return null;
  }
};

export const setPostAuthRedirect = (path: string) => {
  if (!isClient()) return;
  window.sessionStorage.setItem(POST_AUTH_REDIRECT_STORAGE_KEY, path);
};

export const getPostAuthRedirect = () => {
  if (!isClient()) return null;
  return window.sessionStorage.getItem(POST_AUTH_REDIRECT_STORAGE_KEY);
};

export const consumePostAuthRedirect = () => {
  if (!isClient()) return null;
  const redirectPath = window.sessionStorage.getItem(
    POST_AUTH_REDIRECT_STORAGE_KEY,
  );
  window.sessionStorage.removeItem(POST_AUTH_REDIRECT_STORAGE_KEY);
  return redirectPath;
};
