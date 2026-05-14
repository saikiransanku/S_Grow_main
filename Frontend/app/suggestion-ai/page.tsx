"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import {
  buildAdvisorProfileContext,
  buildProfileName,
  fetchCurrentUserProfile,
} from "@/lib/currentUser";
import { useAuth } from "@/lib/useAuth";

const AI_HISTORY_TOGGLE_EVENT = "ssgrow-ai-history-toggle";
const AI_API_BASE =
  process.env.NEXT_PUBLIC_AI_API_URL || "http://localhost:8000/api/ai";

const QUICK_PROMPTS = [
  "Suggest the best crops for my saved land this season.",
  "Recommend low-risk crops for red soil with low water.",
  "Give me intercropping options for my farm.",
  "What crop should I grow after cotton on the same land?",
];

interface Message {
  id: string;
  type: "user" | "ai";
  content: string;
}

interface Chat {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: Message[];
  advisorContext: any | null;
}

const createRequestId = () => {
  if (
    typeof crypto !== "undefined" &&
    typeof crypto.randomUUID === "function"
  ) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
};

const createTimestamp = () => new Date().toISOString();

const normalizeWhitespace = (value: string) =>
  value.replace(/\s+/g, " ").trim();

const trimTitle = (value: string, maxLength = 38) => {
  const clean = normalizeWhitespace(value);
  if (!clean) return "New Suggestion";
  if (clean.length <= maxLength) return clean;
  return `${clean.slice(0, maxLength - 1).trimEnd()}...`;
};

const relativeHistoryLabel = (dateText: string) => {
  const current = new Date();
  const value = new Date(dateText);

  if (Number.isNaN(value.getTime())) return "Earlier";

  const currentDay = new Date(
    current.getFullYear(),
    current.getMonth(),
    current.getDate(),
  );
  const valueDay = new Date(
    value.getFullYear(),
    value.getMonth(),
    value.getDate(),
  );

  const difference = Math.round(
    (currentDay.getTime() - valueDay.getTime()) / (1000 * 60 * 60 * 24),
  );

  if (difference <= 0) return "Today";
  if (difference === 1) return "Yesterday";
  if (difference <= 7) return "Last 7 Days";
  return "Earlier";
};

const buildEmptyChat = (title = "New Suggestion"): Chat => ({
  id: createRequestId(),
  title,
  createdAt: createTimestamp(),
  updatedAt: createTimestamp(),
  messages: [],
  advisorContext: null,
});

const orderChats = (items: Chat[]) =>
  [...items].sort(
    (left, right) =>
      new Date(right.updatedAt).getTime() - new Date(left.updatedAt).getTime(),
  );

const historyGroups = (items: Chat[]) => {
  const groups = new Map<string, Chat[]>();

  orderChats(items).forEach((chat) => {
    const label = relativeHistoryLabel(chat.updatedAt);
    const current = groups.get(label) ?? [];
    current.push(chat);
    groups.set(label, current);
  });

  return Array.from(groups.entries());
};

const buildConversationHistory = (messages: Message[]) =>
  messages
    .slice(-8)
    .filter((message) => Boolean(message.content?.trim()))
    .map((message) => ({
      role: message.type === "user" ? "user" : "assistant",
      content: message.content.trim().slice(0, 4000),
    }));

const formatThinkingTime = (startedAtMs: number) => {
  const elapsedMs = Math.max(0, performance.now() - startedAtMs);
  return `${(elapsedMs / 1000).toFixed(2)}s`;
};

const prependThinkingTime = (thinkingTime: string, body: string) =>
  `Thinking Time: ${thinkingTime}\n\n${body}`.trim();

const fetchJson = async (
  input: RequestInfo | URL,
  init?: RequestInit,
  timeoutMs = 120000,
) => {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(input, {
      ...init,
      signal: controller.signal,
    });
    const rawBody = await response.text();
    const contentType = response.headers.get("content-type") || "";
    let data: any = null;

    if (rawBody && contentType.includes("application/json")) {
      try {
        data = JSON.parse(rawBody);
      } catch {
        throw new Error("AI server returned invalid JSON.");
      }
    }

    if (!response.ok) {
      const errorMessage =
        data?.error ||
        data?.detail ||
        rawBody ||
        `AI request failed with status ${response.status}.`;
      throw new Error(errorMessage);
    }

    if (!data) {
      throw new Error(rawBody || "AI server returned an empty response.");
    }

    return data;
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error(
        `AI server timed out after ${Math.round(timeoutMs / 1000)} seconds.`,
      );
    }
    throw error;
  } finally {
    window.clearTimeout(timeoutId);
  }
};

function SidebarIcon({
  children,
  compact = false,
}: {
  children: React.ReactNode;
  compact?: boolean;
}) {
  return (
    <span
      className={`flex items-center justify-center rounded-2xl bg-black/5 text-neutral-700 dark:bg-white/6 dark:text-neutral-200 ${
        compact ? "h-11 w-11" : "h-10 w-10"
      }`}
    >
      {children}
    </span>
  );
}

function SidebarMenuButton({
  label,
  compact = false,
  active = false,
  onClick,
  children,
}: {
  label: string;
  compact?: boolean;
  active?: boolean;
  onClick?: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={label}
      className={`flex w-full items-center gap-3 rounded-2xl px-2 py-1.5 text-sm transition-colors ${
        active
          ? "border border-sky-200/90 bg-white text-sky-950 shadow-sm dark:border-sky-400/20 dark:bg-white/10 dark:text-white"
          : "border border-transparent text-neutral-700 hover:bg-white/85 hover:text-sky-950 dark:text-neutral-200/92 dark:hover:bg-white/8 dark:hover:text-white"
      } ${compact ? "justify-center px-0" : ""}`}
    >
      <SidebarIcon compact={compact}>{children}</SidebarIcon>
      {compact ? null : <span className="truncate text-[15px]">{label}</span>}
    </button>
  );
}

function ProfilePill({ label, value }: { label: string; value: string }) {
  if (!value || value === "Unknown") return null;

  return (
    <div className="rounded-full border border-sky-200/80 bg-white px-3 py-1.5 text-xs font-medium text-sky-900 shadow-sm dark:border-sky-300/15 dark:bg-white/[0.05] dark:text-sky-100">
      <span className="text-sky-600 dark:text-sky-300">{label}:</span> {value}
    </div>
  );
}

function QuickPromptButton({
  prompt,
  onClick,
}: {
  prompt: string;
  onClick: (prompt: string) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onClick(prompt)}
      className="rounded-2xl border border-sky-200/80 bg-white/90 px-4 py-3 text-left text-sm leading-6 text-neutral-700 shadow-sm transition-colors hover:border-sky-300 hover:text-sky-950 dark:border-white/10 dark:bg-white/[0.04] dark:text-neutral-200 dark:hover:border-sky-300/30 dark:hover:text-white"
    >
      {prompt}
    </button>
  );
}

function EmptySuggestionState({
  onPromptSelect,
  profileContext,
}: {
  onPromptSelect: (prompt: string) => void;
  profileContext: any | null;
}) {
  const hasProfile =
    profileContext &&
    Object.values(profileContext).some(
      (value) => value !== "" && value !== null && value !== undefined,
    );

  return (
    <div className="mx-auto flex h-full w-full max-w-[960px] items-center justify-center px-4">
      <div className="w-full space-y-8">
        <div className="text-center">
          <p className="text-sm font-medium tracking-[0.24em] text-sky-700 dark:text-sky-300">
            SUGGESTION AI
          </p>
          <h1 className="mt-5 text-3xl font-medium tracking-[-0.04em] text-neutral-900 dark:text-neutral-100 md:text-5xl">
            Plan your next crop with
            <span className="bg-gradient-to-r from-sky-500 via-cyan-500 to-emerald-500 bg-clip-text text-transparent">
              {" "}
              Suggestion AI
            </span>
          </h1>
          <p className="mx-auto mt-4 max-w-2xl text-sm leading-7 text-neutral-500 dark:text-neutral-400 md:text-base">
            Ask for crop recommendation, same-land planning, or intercropping options.
            Disease-image diagnosis stays in Disease AI.
          </p>
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          {QUICK_PROMPTS.map((prompt) => (
            <QuickPromptButton
              key={prompt}
              prompt={prompt}
              onClick={onPromptSelect}
            />
          ))}
        </div>

        <div className="rounded-[30px] border border-sky-200/70 bg-[linear-gradient(135deg,rgba(255,255,255,0.98),rgba(232,247,255,0.95))] p-6 shadow-[0_24px_80px_-42px_rgba(14,116,144,0.45)] dark:border-white/10 dark:bg-[linear-gradient(135deg,rgba(17,24,39,0.95),rgba(15,23,42,0.96))]">
          <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
            <div className="max-w-2xl">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-sky-700 dark:text-sky-300">
                Recommendation Context
              </p>
              <h2 className="mt-2 text-xl font-semibold text-neutral-950 dark:text-white">
                {hasProfile
                  ? "Your saved farm profile is ready to guide crop suggestions."
                  : "Add farm details in your profile for more accurate recommendations."}
              </h2>
              <p className="mt-2 text-sm leading-7 text-neutral-600 dark:text-neutral-300">
                Suggestion AI uses land, soil, water, season, and previous crop details when
                they are available. For leaf-image disease analysis, open{" "}
                <Link
                  href="/ai-grow"
                  className="font-semibold text-emerald-700 underline decoration-emerald-300 underline-offset-4 dark:text-emerald-300"
                >
                  Disease AI
                </Link>
                .
              </p>
            </div>

            <div className="flex flex-wrap gap-2 md:max-w-sm md:justify-end">
              <ProfilePill
                label="Soil"
                value={String(profileContext?.soil_type || "")}
              />
              <ProfilePill
                label="Water"
                value={
                  String(profileContext?.water_source || "") ||
                  String(profileContext?.irrigation_level || "")
                }
              />
              <ProfilePill
                label="Season"
                value={String(profileContext?.season || "")}
              />
              <ProfilePill
                label="Goal"
                value={
                  String(profileContext?.crop_purpose || "") ||
                  String(profileContext?.cropping_preference || "")
                }
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function SuggestionAIPage() {
  const { isAuthenticated, isLoading } = useAuth();

  const [chats, setChats] = useState<Chat[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [profileName, setProfileName] = useState("");
  const [profileContext, setProfileContext] = useState<any | null>(null);
  const [desktopSidebarOpen, setDesktopSidebarOpen] = useState(true);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const currentChat =
    chats.find((chat) => chat.id === currentChatId) || null;
  const groupedChats = historyGroups(chats);
  const sidebarCompact = !desktopSidebarOpen;

  const updateChat = (chatId: string, updater: (chat: Chat) => Chat) => {
    setChats((prev) =>
      prev.map((chat) => (chat.id === chatId ? updater(chat) : chat)),
    );
  };

  const appendMessage = (
    chatId: string,
    message: Message,
    nextTitle?: string,
  ) => {
    updateChat(chatId, (chat) => ({
      ...chat,
      title:
        chat.messages.length === 0 && nextTitle ? trimTitle(nextTitle) : chat.title,
      updatedAt: createTimestamp(),
      messages: [...chat.messages, message],
    }));
  };

  const addAiMessage = (chatId: string, message: Message) => {
    updateChat(chatId, (chat) => ({
      ...chat,
      updatedAt: createTimestamp(),
      messages: [...chat.messages, message],
    }));
  };

  const setAdvisorContext = (chatId: string, context: any) => {
    updateChat(chatId, (chat) => ({
      ...chat,
      updatedAt: createTimestamp(),
      advisorContext: context,
    }));
  };

  const startNewChat = () => {
    setCurrentChatId(null);
    setInput("");
    setMobileSidebarOpen(false);
  };

  const selectChat = (chatId: string) => {
    setCurrentChatId(chatId);
    setMobileSidebarOpen(false);
  };

  useEffect(() => {
    const handleHistoryToggle = () => {
      if (window.innerWidth >= 768) {
        setDesktopSidebarOpen((prev) => !prev);
        return;
      }
      setMobileSidebarOpen((prev) => !prev);
    };

    window.addEventListener(AI_HISTORY_TOGGLE_EVENT, handleHistoryToggle);
    return () =>
      window.removeEventListener(AI_HISTORY_TOGGLE_EVENT, handleHistoryToggle);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [currentChat?.messages, isSending]);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = "0px";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 160)}px`;
  }, [input]);

  useEffect(() => {
    const loadProfileDetails = async () => {
      try {
        const user = await fetchCurrentUserProfile();
        setProfileName(buildProfileName(user));
        setProfileContext(buildAdvisorProfileContext(user));
      } catch {
        setProfileName("");
        setProfileContext(null);
      }
    };

    loadProfileDetails();
  }, []);

  const submitPrompt = async (promptOverride?: string) => {
    if (isSending) return;

    const prompt = normalizeWhitespace(promptOverride ?? input);
    if (!prompt) return;

    const existingChat =
      chats.find((chat) => chat.id === currentChatId) || currentChat || null;
    const chat =
      existingChat ||
      (() => {
        const nextChat = buildEmptyChat(trimTitle(prompt));
        setChats((prev) => [nextChat, ...prev]);
        setCurrentChatId(nextChat.id);
        return nextChat;
      })();
    const chatId = chat.id;

    const userMessage: Message = {
      id: createRequestId(),
      type: "user",
      content: prompt,
    };

    appendMessage(chatId, userMessage, prompt);
    setInput("");
    setIsSending(true);

    const startedAt = performance.now();
    const nextHistory = buildConversationHistory([...chat.messages, userMessage]);

    try {
      const data = await fetchJson(
        `${AI_API_BASE}/chat`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            message: prompt,
            context: null,
            profile_name: profileName,
            profile_context: profileContext,
            advisor_context: chat.advisorContext,
            conversation_history: nextHistory,
          }),
        },
        90000,
      );

      if (Object.prototype.hasOwnProperty.call(data || {}, "advisor_context")) {
        setAdvisorContext(chatId, data?.advisor_context ?? null);
      }

      addAiMessage(chatId, {
        id: createRequestId(),
        type: "ai",
        content: prependThinkingTime(
          formatThinkingTime(startedAt),
          data?.reply || "Unable to generate a response right now.",
        ),
      });
    } catch (error) {
      addAiMessage(chatId, {
        id: createRequestId(),
        type: "ai",
        content: prependThinkingTime(
          formatThinkingTime(startedAt),
          error instanceof Error
            ? error.message
            : "Error connecting to the AI assistant.",
        ),
      });
    } finally {
      setIsSending(false);
    }
  };

  if (isLoading) {
    return (
      <div className="flex h-[calc(100dvh-5rem)] items-center justify-center bg-white text-neutral-900 dark:bg-black dark:text-white">
        <div className="space-y-4 text-center">
          <div className="mx-auto h-12 w-12 animate-spin rounded-full border-b-2 border-sky-500" />
          <p className="text-neutral-600 dark:text-neutral-400">Loading...</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <div className="flex h-[calc(100dvh-5rem)] items-center justify-center bg-white px-6 text-neutral-900 dark:bg-black dark:text-white">
        <div className="mx-auto max-w-md space-y-6 rounded-[28px] border border-black/10 bg-white/90 p-8 text-center shadow-lg dark:border-white/10 dark:bg-neutral-950/90">
          <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-sky-600 text-white">
            <svg
              className="h-8 w-8"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.8"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M8 10V7a4 4 0 1 1 8 0v3m-9 0h10a1 1 0 0 1 1 1v8a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1v-8a1 1 0 0 1 1-1Z"
              />
            </svg>
          </div>
          <h1 className="text-3xl font-bold text-neutral-950 dark:text-white">
            Authentication Required
          </h1>
          <p className="text-lg text-neutral-600 dark:text-neutral-300">
            Please sign in to access Suggestion AI for crop recommendations.
          </p>
          <Link
            href="/login"
            className="inline-flex w-full items-center justify-center rounded-2xl bg-sky-600 px-5 py-3 font-semibold text-white transition-colors hover:bg-sky-700"
          >
            Sign In to Continue
          </Link>
          <p className="text-sm text-neutral-500 dark:text-neutral-400">
            Do not have an account?{" "}
            <Link
              href="/register"
              className="font-medium text-sky-600 hover:text-sky-700 dark:text-sky-400 dark:hover:text-sky-300"
            >
              Create one here
            </Link>
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="relative flex h-[calc(100dvh-5rem)] overflow-hidden bg-[radial-gradient(circle_at_top_left,rgba(125,211,252,0.16),transparent_28%),radial-gradient(circle_at_top_right,rgba(16,185,129,0.14),transparent_24%),linear-gradient(180deg,#f4fbff_0%,#eef7f6_100%)] text-neutral-900 dark:bg-[radial-gradient(circle_at_top_left,rgba(14,116,144,0.18),transparent_25%),radial-gradient(circle_at_top_right,rgba(6,95,70,0.16),transparent_22%),linear-gradient(180deg,#0f172a_0%,#111827_100%)] dark:text-white">
      {mobileSidebarOpen ? (
        <button
          type="button"
          aria-label="Close suggestion AI history"
          onClick={() => setMobileSidebarOpen(false)}
          className="absolute inset-0 z-20 bg-black/30 backdrop-blur-[1px] md:hidden"
        />
      ) : null}

      <aside
        className={`absolute inset-y-0 left-0 z-30 flex w-[88vw] max-w-[320px] flex-col overflow-y-auto overflow-x-hidden border-r border-sky-100/90 bg-[linear-gradient(180deg,rgba(247,252,255,0.98),rgba(235,248,250,0.96))] text-neutral-900 shadow-2xl dark:border-white/10 dark:bg-[#101827] dark:text-neutral-100 md:static md:z-0 md:max-w-none md:shadow-none ${
          mobileSidebarOpen ? "translate-x-0" : "-translate-x-full"
        } transition-transform duration-300 md:translate-x-0 ${
          desktopSidebarOpen ? "md:w-[18rem]" : "md:w-[5.5rem]"
        }`}
      >
        <div className="sticky top-0 z-10 border-b border-black/5 bg-[linear-gradient(180deg,rgba(247,252,255,0.98),rgba(235,248,250,0.96))] px-4 pb-3 pt-4 backdrop-blur-xl dark:border-white/8 dark:bg-[#101827]/95">
          <button
            type="button"
            onClick={() =>
              window.innerWidth >= 768
                ? setDesktopSidebarOpen((prev) => !prev)
                : setMobileSidebarOpen((prev) => !prev)
            }
            className={`flex items-center gap-3 rounded-2xl transition-colors hover:bg-black/[0.03] dark:hover:bg-white/[0.04] ${
              sidebarCompact ? "justify-center px-0" : "w-full px-1 py-1"
            }`}
            title={sidebarCompact ? "Open history" : "Collapse history"}
          >
            <span className="flex h-11 w-11 items-center justify-center rounded-2xl bg-sky-100 text-sky-700 ring-1 ring-sky-200/70 dark:bg-sky-400/10 dark:text-sky-200 dark:ring-sky-300/10">
              <svg
                className="h-6 w-6"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.8"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M7 20h10M6 4h12a1 1 0 0 1 1 1v10.5a1 1 0 0 1-.4.8l-4.8 3.6a1 1 0 0 1-1.2 0l-4.8-3.6a1 1 0 0 1-.4-.8V5a1 1 0 0 1 1-1Z"
                />
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M9 9h6M9 12h6"
                />
              </svg>
            </span>
            {sidebarCompact ? null : (
              <div className="min-w-0 flex-1 text-left">
                <span className="block truncate text-[17px] font-medium tracking-[-0.03em] text-neutral-900 dark:text-neutral-100">
                  Suggestion AI
                </span>
                <span className="block truncate text-xs text-neutral-500 dark:text-neutral-400">
                  Crop recommendation workspace
                </span>
              </div>
            )}
          </button>
        </div>

        <div className="space-y-3 px-3 py-3">
          <SidebarMenuButton
            label="New chat"
            compact={sidebarCompact}
            onClick={startNewChat}
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
                d="M12 5v14m7-7H5"
              />
            </svg>
          </SidebarMenuButton>

          {sidebarCompact ? null : (
            <Link
              href="/ai-grow"
              className="flex items-center gap-3 rounded-2xl border border-emerald-200/80 bg-white/85 px-3 py-2.5 text-sm text-emerald-900 shadow-sm transition-colors hover:bg-emerald-50 dark:border-emerald-400/20 dark:bg-white/[0.04] dark:text-emerald-100 dark:hover:bg-emerald-400/10"
            >
              <span className="flex h-10 w-10 items-center justify-center rounded-2xl bg-emerald-100 text-emerald-700 dark:bg-emerald-500/12 dark:text-emerald-300">
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
                    d="M5 19c6 0 14-4 14-14C9 5 5 13 5 19Zm0 0 6-6"
                  />
                </svg>
              </span>
              <span className="min-w-0 flex-1">
                <span className="block font-medium">Open Disease AI</span>
                <span className="block text-xs text-neutral-500 dark:text-neutral-400">
                  For image-based disease diagnosis
                </span>
              </span>
            </Link>
          )}
        </div>

        <div className="flex-1 px-3 pb-4">
          {sidebarCompact ? null : (
            <div className="mb-3 px-2">
              <p className="text-[12px] font-medium uppercase tracking-[0.2em] text-sky-700/90 dark:text-sky-300/85">
                Your Chats
              </p>
            </div>
          )}

          {groupedChats.length ? (
            <div className="space-y-4">
              {groupedChats.map(([label, items]) => (
                <div
                  key={label}
                  className={sidebarCompact ? "md:flex md:flex-col md:items-center md:gap-2" : ""}
                >
                  {sidebarCompact ? null : (
                    <p className="mb-2 px-2 text-[11px] font-semibold uppercase tracking-[0.16em] text-neutral-500 dark:text-neutral-400">
                      {label}
                    </p>
                  )}
                  <div className={`space-y-1 ${sidebarCompact ? "md:space-y-2" : ""}`}>
                    {items.map((chat) => (
                      <button
                        key={chat.id}
                        type="button"
                        onClick={() => selectChat(chat.id)}
                        title={chat.title}
                        className={`group flex w-full items-center gap-3 rounded-2xl px-2 py-2.5 text-left text-sm transition-colors ${
                          currentChat?.id === chat.id
                            ? "border border-sky-200/90 bg-white text-sky-950 shadow-sm dark:border-sky-400/20 dark:bg-white/10 dark:text-white"
                            : "border border-transparent text-neutral-700 hover:bg-white/85 hover:text-sky-950 dark:text-neutral-200/92 dark:hover:bg-white/[0.06] dark:hover:text-white"
                        } ${sidebarCompact ? "justify-center px-0 md:h-11 md:w-11" : ""}`}
                      >
                        {sidebarCompact ? (
                          <span className="text-xs font-semibold">
                            {chat.title.slice(0, 1).toUpperCase() || "S"}
                          </span>
                        ) : (
                          <>
                            <span
                              className={`mt-1 h-2 w-2 flex-shrink-0 rounded-full ${
                                currentChat?.id === chat.id
                                  ? "bg-sky-500 dark:bg-sky-300"
                                  : "bg-neutral-300 dark:bg-neutral-600"
                              }`}
                            />
                            <span className="min-w-0 flex-1 truncate text-[15px] font-medium">
                              {chat.title}
                            </span>
                          </>
                        )}
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div
              className={`rounded-[24px] border border-dashed border-sky-200 bg-white/78 px-4 py-8 text-center text-sm text-neutral-500 dark:border-white/10 dark:bg-white/[0.03] dark:text-neutral-400 ${
                sidebarCompact ? "hidden md:block md:px-2 md:py-3 md:text-[11px]" : ""
              }`}
            >
              No suggestion chats yet.
            </div>
          )}
        </div>
      </aside>

      <section className="relative flex min-w-0 flex-1 flex-col overflow-hidden">
        <div className="border-b border-black/5 bg-white/55 px-4 py-3 backdrop-blur-xl dark:border-white/8 dark:bg-black/10 md:px-6">
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-sky-700 dark:text-sky-300">
                Crop Recommendation Engine
              </p>
              <h1 className="mt-1 text-lg font-semibold text-neutral-950 dark:text-white">
                Ask about crop selection, same-land planning, or intercropping.
              </h1>
            </div>

            <div className="flex flex-wrap gap-2">
              <ProfilePill
                label="Profile"
                value={profileName || "Connected"}
              />
              <ProfilePill
                label="Saved soil"
                value={String(profileContext?.soil_type || "")}
              />
              <ProfilePill
                label="Saved water"
                value={
                  String(profileContext?.water_source || "") ||
                  String(profileContext?.irrigation_level || "")
                }
              />
            </div>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto px-4 pb-[12.5rem] pt-6 md:px-6 md:pb-[13.5rem] md:pt-8">
          {currentChat?.messages.length ? (
            <div className="mx-auto flex w-full max-w-[920px] flex-col gap-6">
              <div className="rounded-[28px] border border-sky-200/70 bg-white/80 px-5 py-4 text-sm leading-7 text-neutral-600 shadow-sm dark:border-white/10 dark:bg-white/[0.04] dark:text-neutral-300">
                Need leaf-image diagnosis instead? Open{" "}
                <Link
                  href="/ai-grow"
                  className="font-semibold text-emerald-700 underline decoration-emerald-300 underline-offset-4 dark:text-emerald-300"
                >
                  Disease AI
                </Link>
                .
              </div>

              {currentChat.messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${
                    message.type === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  <div
                    className={`max-w-[82%] rounded-[28px] px-5 py-4 ${
                      message.type === "user"
                        ? "border border-sky-200/80 bg-white text-neutral-900 shadow-sm dark:border-white/10 dark:bg-[#1f2937] dark:text-white"
                        : "border border-black/8 bg-[linear-gradient(180deg,rgba(255,255,255,0.88),rgba(235,248,250,0.82))] text-neutral-800 shadow-sm dark:border-white/10 dark:bg-[linear-gradient(180deg,rgba(15,23,42,0.9),rgba(17,24,39,0.94))] dark:text-neutral-100"
                    }`}
                  >
                    <p className="whitespace-pre-wrap text-sm leading-7 md:text-[15px]">
                      {message.content}
                    </p>
                  </div>
                </div>
              ))}

              {isSending ? (
                <div className="flex justify-start">
                  <div className="w-full max-w-3xl rounded-[28px] border border-black/8 bg-white/78 p-5 shadow-sm dark:border-white/10 dark:bg-[#1f2937]/95">
                    <div className="space-y-3">
                      <div className="h-5 w-44 rounded-full bg-sky-100 dark:bg-neutral-800" />
                      <div className="h-4 w-full rounded-full bg-neutral-200 dark:bg-neutral-800" />
                      <div className="h-4 w-5/6 rounded-full bg-neutral-200 dark:bg-neutral-800" />
                      <div className="h-4 w-2/3 rounded-full bg-neutral-200 dark:bg-neutral-800" />
                    </div>
                  </div>
                </div>
              ) : null}

              <div ref={messagesEndRef} />
            </div>
          ) : (
            <EmptySuggestionState
              onPromptSelect={submitPrompt}
              profileContext={profileContext}
            />
          )}
        </div>

        <div className="pointer-events-none absolute inset-x-0 bottom-0 z-20 px-3 pb-3 md:px-6 md:pb-6">
          <div className="mx-auto w-full max-w-[920px] pointer-events-auto">
            <div className="mb-3 flex flex-wrap gap-2">
              {QUICK_PROMPTS.slice(0, 3).map((prompt) => (
                <button
                  key={prompt}
                  type="button"
                  onClick={() => submitPrompt(prompt)}
                  disabled={isSending}
                  className="rounded-full border border-sky-200/70 bg-white/92 px-3 py-2 text-xs font-medium text-sky-900 shadow-sm transition-colors hover:border-sky-300 hover:bg-sky-50 disabled:cursor-not-allowed disabled:opacity-60 dark:border-white/10 dark:bg-white/[0.05] dark:text-sky-100 dark:hover:bg-white/[0.08]"
                >
                  {prompt}
                </button>
              ))}
            </div>

            <div className="rounded-[32px] border border-black/8 bg-white/82 shadow-[0_24px_50px_-30px_rgba(15,23,42,0.35)] backdrop-blur-2xl dark:border-white/10 dark:bg-[#0f172a]/94 dark:shadow-[0_24px_60px_-32px_rgba(0,0,0,0.75)]">
              <div className="px-3 py-3 md:px-4 md:py-4">
                <textarea
                  ref={textareaRef}
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault();
                      submitPrompt();
                    }
                  }}
                  placeholder="Ask for crop recommendation, same-land planning, or intercropping..."
                  rows={1}
                  className="max-h-40 min-h-[28px] w-full resize-none bg-transparent px-2 py-1 text-[15px] leading-7 text-neutral-900 outline-none placeholder:text-neutral-500 dark:text-white dark:placeholder:text-neutral-400 md:text-base"
                />

                <div className="mt-3 flex flex-wrap items-center justify-between gap-3 border-t border-black/8 pt-3 dark:border-white/8">
                  <p className="text-xs text-neutral-500 dark:text-neutral-400">
                    Suggestion AI uses your saved profile when it is available.
                  </p>

                  <button
                    type="button"
                    onClick={() => submitPrompt()}
                    disabled={isSending || !input.trim()}
                    className="flex min-h-11 items-center justify-center rounded-full bg-sky-500 px-5 text-sm font-semibold text-white shadow-[0_10px_28px_-16px_rgba(14,165,233,0.9)] transition-colors hover:bg-sky-400 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {isSending ? "Thinking..." : "Send"}
                  </button>
                </div>
              </div>
            </div>

            <p className="mt-3 text-center text-xs text-neutral-500 dark:text-neutral-400">
              Recommendations are AI-generated. Recheck before making field decisions.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
