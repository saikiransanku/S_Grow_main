"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";

import {
  buildAdvisorProfileContext,
  buildProfileName,
  fetchCurrentUserProfile,
} from "@/lib/currentUser";
import { consumePendingSuphalaIntent } from "@/lib/suphalaAI";
import { useAuth } from "@/lib/useAuth";

const AI_HISTORY_TOGGLE_EVENT = "ssgrow-ai-history-toggle";

type SeasonOption = "auto" | "kharif" | "rabi" | "all_season";
type UrgencyLevel = "low" | "medium" | "high";
type QueuedImageSource = "camera" | "upload";

const SEASON_OPTIONS: Array<{ value: SeasonOption; label: string }> = [
  { value: "auto", label: "Auto Detect" },
  { value: "rabi", label: "Rabi" },
  { value: "kharif", label: "Kharif" },
  { value: "all_season", label: "All Season" },
];

interface Message {
  id: string;
  type: "user" | "ai";
  content: string;
  images?: string[];
  analysisCards?: AnalysisCard[];
  analysisName?: string;
}

interface QueuedImage {
  id: string;
  file: File;
  preview: string;
  sourceType: QueuedImageSource;
}

interface UrgencyState {
  level: UrgencyLevel;
  label: string;
  note: string;
}

interface AnalysisCard {
  id: string;
  title: string;
  imagePreview?: string;
  crop: string;
  disease: string;
  confidenceText: string;
  symptoms: string[];
  cause: string;
  treatments: string[];
  prevention: string[];
  urgency: UrgencyState;
  modelNote: string;
  verificationNote: string;
}

interface Chat {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: Message[];
  lastPredictionContext: any | null;
  advisorContext: any | null;
}

const SUMMARY_CARD_LABEL = "Crop Disease Diagnosis";

const createTimestamp = () => new Date().toISOString();

const normalizeWhitespace = (value: string) =>
  value.replace(/\s+/g, " ").trim();

const humanizeLabel = (value: any, fallback = "Unknown") => {
  const text = normalizeWhitespace(
    String(value ?? "")
      .replace(/[_,-]+/g, " ")
      .replace(/\s+/g, " "),
  );

  if (!text) return fallback;

  return text
    .split(" ")
    .map((part) =>
      part ? `${part.charAt(0).toUpperCase()}${part.slice(1).toLowerCase()}` : "",
    )
    .join(" ");
};

const toSentence = (value: string) => {
  const clean = normalizeWhitespace(
    value.replace(/[*_`>#-]+/g, " ").replace(/\s+/g, " "),
  );
  if (!clean) return "";
  const trimmed = clean.replace(/[.;,:-]+$/, "");
  return `${trimmed.charAt(0).toUpperCase()}${trimmed.slice(1)}.`;
};

const firstSentence = (value: string) => {
  const clean = normalizeWhitespace(
    value.replace(/[*_`>#]+/g, " ").replace(/\s+/g, " "),
  );
  if (!clean) return "";
  const match = clean.match(/.+?[.!?](?:\s|$)/);
  return normalizeWhitespace(match ? match[0] : clean);
};

const getDiagnosisStatus = (result: any) =>
  normalizeWhitespace(String(result?.diagnosis_status ?? ""))
    .toLowerCase()
    .replace(/\s+/g, "_");

const isInvalidLeafOrFruitResult = (result: any) =>
  getDiagnosisStatus(result) === "invalid_leaf_or_fruit";

const trimTitle = (value: string, maxLength = 36) => {
  const clean = normalizeWhitespace(value);
  if (!clean) return "New Chat";
  if (clean.length <= maxLength) return clean;
  return `${clean.slice(0, maxLength - 1).trimEnd()}...`;
};

const formatSeasonLabel = (value: string) => {
  if (!value) return "Auto";
  if (value === "all_season") return "All Season";
  return humanizeLabel(value);
};

const isGreetingMessage = (value: string) => {
  const normalized = normalizeWhitespace(value.replace(/[^a-zA-Z ]+/g, " "))
    .toLowerCase();
  if (!normalized) return false;

  const baseGreetings = new Set([
    "hi",
    "hii",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
  ]);
  if (baseGreetings.has(normalized)) return true;

  const tokens = normalized.split(" ").filter(Boolean);
  if (!tokens.length) return false;
  if (
    tokens.includes("good") &&
    tokens.some((token) =>
      ["morning", "mrng", "afternoon", "aftn", "evening", "evng"].includes(token),
    )
  ) {
    return true;
  }
  return ["hi", "hii", "hello", "hey"].includes(tokens[0]) && tokens.length <= 4;
};

const toConfidenceNumber = (value: any) => {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number.parseFloat(value.replace("%", "").trim());
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
};

const formatConfidenceText = (value: any) => {
  const parsed = toConfidenceNumber(value);
  if (parsed === null) return "Not available";
  return `${parsed.toFixed(1).replace(/\.0$/, "")}%`;
};

const uniqueItems = (items: string[]) => {
  const seen = new Set<string>();
  return items.filter((item) => {
    const key = item.toLowerCase();
    if (!item || seen.has(key)) return false;
    seen.add(key);
    return true;
  });
};

const createRequestId = () => {
  if (
    typeof crypto !== "undefined" &&
    typeof crypto.randomUUID === "function"
  ) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
};

const createAnalysisName = (imageCount: number) => {
  const now = new Date();
  const datePart = now.toISOString().slice(0, 10);
  const timePart = now.toTimeString().slice(0, 8).replace(/:/g, "-");
  return `Leaf Analysis ${datePart} ${timePart} (${imageCount} image${imageCount > 1 ? "s" : ""})`;
};

const formatThinkingTime = (startedAtMs: number) => {
  const elapsedMs = Math.max(0, performance.now() - startedAtMs);
  return `${(elapsedMs / 1000).toFixed(2)}s`;
};

const prependThinkingTime = (thinkingTime: string, body: string) => {
  return `Thinking Time: ${thinkingTime}\n\n${body}`.trim();
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

const buildEmptyChat = (title = "New Chat"): Chat => ({
  id: createRequestId(),
  title,
  createdAt: createTimestamp(),
  updatedAt: createTimestamp(),
  messages: [],
  lastPredictionContext: null,
  advisorContext: null,
});

const buildConversationHistory = (chat: Chat | null) => {
  if (!chat) return [];

  return chat.messages
    .slice(-8)
    .filter((message) => Boolean(message.content?.trim()))
    .map((message) => ({
      role: message.type === "user" ? "user" : "assistant",
      content: message.content.trim().slice(0, 4000),
    }));
};

const getDiseasePlaybook = (diseaseName: string) => {
  const token = diseaseName.toLowerCase().replace(/\s+/g, "_");

  if (!token || token.includes("healthy")) {
    return {
      cause:
        "No strong disease pattern was visible in the uploaded leaf image.",
      symptoms: [
        "Leaf tissue looks mostly stable without strong spreading lesions.",
        "No major warning pattern was clearly detected.",
        "Continue routine field monitoring for any new change.",
      ],
      prevention: [
        "Keep regular crop monitoring in place.",
        "Avoid unnecessary sprays on healthy leaves.",
        "Maintain balanced watering and field hygiene.",
      ],
    };
  }

  if (
    token.includes("blight") ||
    token.includes("blast") ||
    token.includes("mildew") ||
    token.includes("rust") ||
    token.includes("mould") ||
    token.includes("mold") ||
    token.includes("spot") ||
    token.includes("anthracnose")
  ) {
    return {
      cause:
        "This pattern usually points to a fungal infection that spreads faster in humid conditions.",
      symptoms: [
        "Brown or dark spots are visible on the leaf surface.",
        "Yellowing tissue can form around infected areas.",
        "Affected parts may dry out as the disease spreads.",
      ],
      prevention: [
        "Keep foliage dry and avoid overhead irrigation.",
        "Increase spacing and airflow around plants.",
        "Remove infected leaves before the spread becomes wider.",
      ],
    };
  }

  if (
    token.includes("bacterial") ||
    token.includes("canker") ||
    token.includes("speck")
  ) {
    return {
      cause:
        "This looks closer to a bacterial issue that often spreads through splash water, tools, or contact.",
      symptoms: [
        "Water-soaked or dark lesions may form on the leaf.",
        "Spots can expand quickly after rain or irrigation splash.",
        "Leaf edges may weaken or tear around infected tissue.",
      ],
      prevention: [
        "Avoid working on wet plants.",
        "Clean tools regularly between plants.",
        "Reduce leaf wetness and improve field sanitation.",
      ],
    };
  }

  if (
    token.includes("mosaic") ||
    token.includes("virus") ||
    token.includes("curl")
  ) {
    return {
      cause:
        "This looks closer to a viral issue, which is often spread by insect vectors such as whiteflies or aphids.",
      symptoms: [
        "Leaf color may look uneven or mottled.",
        "Curling or distorted growth can appear on young leaves.",
        "Plant vigor can reduce as the infection progresses.",
      ],
      prevention: [
        "Control insect vectors early.",
        "Remove heavily affected leaves or plants promptly.",
        "Use clean planting material and resistant varieties when available.",
      ],
    };
  }

  if (token.includes("wilt") || token.includes("rot")) {
    return {
      cause:
        "This kind of damage is often linked to an aggressive infection affecting moisture flow through the plant.",
      symptoms: [
        "Leaves may lose firmness or dry rapidly.",
        "Damage can spread from one area to nearby tissue.",
        "The plant may weaken quickly if untreated.",
      ],
      prevention: [
        "Improve drainage and avoid overwatering.",
        "Remove infected plant material quickly.",
        "Rotate crops and keep the field clean between cycles.",
      ],
    };
  }

  return {
    cause:
      "The uploaded image shows a disease-like stress pattern on the leaf surface.",
    symptoms: [
      "Visible leaf damage is present on the uploaded image.",
      "The pattern suggests a plant health problem rather than normal growth.",
      "Further spread can happen if conditions stay favorable.",
    ],
    prevention: [
      "Inspect nearby leaves for similar symptoms.",
      "Keep field sanitation and airflow strong.",
      "Avoid overwatering and monitor the plant closely.",
    ],
  };
};

const buildSymptoms = (result: any, diseaseName: string) => {
  if (isInvalidLeafOrFruitResult(result)) {
    return uniqueItems(
      [
        firstSentence(String(result?.status_message ?? "")) ||
          "The uploaded image does not look like a leaf or fruit.",
        "Upload one clear close image of a single leaf or fruit.",
        "Keep the subject centered, in focus, and in good light.",
      ]
        .map((item) => toSentence(item))
        .filter(Boolean),
    ).slice(0, 3);
  }

  const playbook = getDiseasePlaybook(diseaseName);
  const anomalies =
    result?.leaf_visual_analysis?.anomalies_textures &&
    typeof result.leaf_visual_analysis.anomalies_textures === "object"
      ? result.leaf_visual_analysis.anomalies_textures
      : {};

  const items: string[] = [];
  const lesionSummary = toSentence(String(anomalies?.lesion_summary ?? ""));
  const chlorosis = String(anomalies?.chlorosis_halo ?? "");
  const lesionCount =
    typeof anomalies?.lesions_detected === "number"
      ? anomalies.lesions_detected
      : null;

  if (lesionSummary) items.push(lesionSummary);
  if (chlorosis && !/not|unclear|none/i.test(chlorosis)) {
    items.push(toSentence(chlorosis));
  }
  if (lesionCount && lesionCount > 0) {
    items.push(
      `${lesionCount} visible lesion${lesionCount > 1 ? "s were" : " was"} detected on the uploaded leaf.`,
    );
  }

  return uniqueItems(
    [...items, ...playbook.symptoms.map((item) => toSentence(item))].filter(
      Boolean,
    ),
  ).slice(0, 3);
};

const buildCause = (result: any, diseaseName: string) => {
  const reportReason = firstSentence(
    String(result?.farmer_report?.reason_for_prediction ?? ""),
  );
  const verificationReason = firstSentence(
    String(result?.verification_reason ?? ""),
  );

  if (isInvalidLeafOrFruitResult(result)) {
    return (
      reportReason ||
      verificationReason ||
      "The uploaded image does not match a supported leaf or fruit input for disease analysis."
    );
  }

  const playbook = getDiseasePlaybook(diseaseName);

  if (!reportReason && !verificationReason) return playbook.cause;
  return normalizeWhitespace(
    `${playbook.cause} ${reportReason || verificationReason}`,
  );
};

const buildTreatments = (result: any, diseaseName: string) => {
  if (isInvalidLeafOrFruitResult(result)) {
    return uniqueItems(
      [
        "No pesticide recommendation is given for this image.",
        "Upload a fresh image showing one clear leaf or fruit.",
        "If symptoms are still spreading, confirm with a local agricultural expert before spraying.",
      ]
        .map((item) => toSentence(item))
        .filter(Boolean),
    ).slice(0, 3);
  }

  const healthy = diseaseName.toLowerCase().includes("healthy");
  const report =
    result?.farmer_report && typeof result.farmer_report === "object"
      ? result.farmer_report
      : {};
  const organic = Array.isArray(report.organic_recommendations)
    ? report.organic_recommendations
    : [];
  const chemical =
    report.chemical_recommendation &&
    typeof report.chemical_recommendation === "object"
      ? report.chemical_recommendation
      : {};

  const items: string[] = healthy
    ? [
        "Keep monitoring the plant instead of spraying immediately.",
        "Continue balanced watering and regular leaf inspection.",
      ]
    : [
        "Remove heavily infected leaves and keep field sanitation strong.",
        "Improve airflow and avoid overhead watering to slow spread.",
      ];

  const firstOrganic = organic.find(
    (item: any) => item && typeof item === "object",
  );
  if (firstOrganic?.name) {
    items.push(
      normalizeWhitespace(
        `Try ${humanizeLabel(firstOrganic.name)} ${firstOrganic.use_case ? `for ${String(firstOrganic.use_case).replace(/[.]+$/, "")}` : "as an early treatment option"}.`,
      ),
    );
  }

  if (!healthy && chemical?.name) {
    items.push(
      normalizeWhitespace(
        `If the disease keeps spreading, use ${humanizeLabel(chemical.name)}. ${String(chemical.usage_note || result?.usage_note || "Follow local label guidance.")}`,
      ),
    );
  }

  return uniqueItems(items.map((item) => toSentence(item)).filter(Boolean)).slice(
    0,
    3,
  );
};

const buildPreventionTips = (diseaseName: string, result: any) => {
  if (isInvalidLeafOrFruitResult(result)) {
    return uniqueItems(
      [
        "Capture only the leaf or fruit, not the whole scene.",
        "Use even lighting and avoid blur or glare.",
        "If possible, upload a close front and back view of the leaf.",
      ]
        .map((item) => toSentence(item))
        .filter(Boolean),
    ).slice(0, 3);
  }

  const playbook = getDiseasePlaybook(diseaseName);
  const tips = [...playbook.prevention];

  if (result?.season_used) {
    tips.push(
      `Monitor the crop closely during ${formatSeasonLabel(String(result.season_used)).toLowerCase()} conditions when the disease pressure can change quickly.`,
    );
  }

  return uniqueItems(tips.map((item) => toSentence(item)).filter(Boolean)).slice(
    0,
    3,
  );
};

const deriveUrgency = (result: any, diseaseName: string): UrgencyState => {
  const token = diseaseName.toLowerCase().replace(/\s+/g, "_");
  const confidence = toConfidenceNumber(result?.confidence_score);
  const diagnosisStatus = getDiagnosisStatus(result);
  const hasSevereKeyword =
    token.includes("blight") ||
    token.includes("blast") ||
    token.includes("rot") ||
    token.includes("wilt") ||
    token.includes("canker");

  if (diagnosisStatus === "invalid_leaf_or_fruit") {
    return {
      level: "low",
      label: "Upload Again",
      note: "This image was rejected before analysis because it was not recognized as a leaf or fruit.",
    };
  }

  if (!token || token.includes("healthy")) {
    return {
      level: "low",
      label: "Low Risk",
      note: "Continue normal monitoring and avoid unnecessary treatment unless new symptoms appear.",
    };
  }

  if (diagnosisStatus === "manual_review_required") {
    return {
      level: "medium",
      label: "Review Soon",
      note: "Use safe interim steps now and confirm the diagnosis before any strong spray decision.",
    };
  }

  if ((confidence ?? 0) >= 90 && hasSevereKeyword) {
    return {
      level: "high",
      label: "High Risk",
      note: "Treat quickly and inspect nearby plants today to reduce further spread.",
    };
  }

  return {
    level: "medium",
    label: "Medium Risk",
    note: "Treat within the next 2 to 3 days and keep the affected area under close watch.",
  };
};

const buildModelNote = (result: any) => {
  if (isInvalidLeafOrFruitResult(result)) {
    return "Upload was rejected before disease prediction because the image was not recognized as a leaf or fruit.";
  }

  const seasonUsed = formatSeasonLabel(String(result?.season_used ?? ""));
  const requestedSeason = formatSeasonLabel(String(result?.requested_season ?? ""));

  return `Model season used: ${seasonUsed}. Requested season: ${requestedSeason}.`;
};

const buildVerificationNote = (result: any) => {
  const statusMessage = firstSentence(String(result?.status_message ?? ""));
  const verificationReason = firstSentence(
    String(result?.verification_reason ?? ""),
  );
  return statusMessage || verificationReason || "AI-based result. Please verify before major treatment.";
};

const buildAnalysisCards = (
  results: any[],
  queuedImages: QueuedImage[],
): AnalysisCard[] => {
  return results.map((result, index) => {
    const disease = humanizeLabel(
      result?.farmer_report?.type_of_disease || result?.disease_type,
      "Unknown Disease",
    );
    const crop = humanizeLabel(
      result?.farmer_report?.leaf_name || result?.crop_detected,
      "Unknown Crop",
    );

    return {
      id: `${result?.image_hash || createRequestId()}-${index}`,
      title:
        results.length > 1
          ? trimTitle(result?.file_name || `Image ${index + 1}`, 48)
          : SUMMARY_CARD_LABEL,
      imagePreview: queuedImages[index]?.preview || queuedImages[0]?.preview,
      crop,
      disease,
      confidenceText: formatConfidenceText(result?.confidence_score),
      symptoms: buildSymptoms(result, disease),
      cause: buildCause(result, disease),
      treatments: buildTreatments(result, disease),
      prevention: buildPreventionTips(disease, result),
      urgency: deriveUrgency(result, disease),
      modelNote: buildModelNote(result),
      verificationNote: buildVerificationNote(result),
    };
  });
};

const getAnalysisTitle = (cards: AnalysisCard[], fallback: string) => {
  if (!cards.length) return fallback;
  return trimTitle(`${cards[0].crop} ${cards[0].disease}`, 42);
};

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

const readFileAsDataUrl = (file: File) =>
  new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error("Unable to read file"));
    reader.readAsDataURL(file);
  });

const isImageFile = (file: File) => {
  if (file.type.toLowerCase().startsWith("image/")) return true;
  return /\.(png|jpe?g|webp|gif|bmp|avif|heic|heif|tiff?)$/i.test(file.name);
};

const filterImageFiles = (files: Iterable<File>) =>
  Array.from(files).filter((file) => isImageFile(file));

const hasDraggedFiles = (dataTransfer: DataTransfer | null) => {
  if (!dataTransfer) return false;
  if (dataTransfer.files.length > 0) return true;
  if (Array.from(dataTransfer.items || []).some((item) => item.kind === "file")) {
    return true;
  }
  return Array.from(dataTransfer.types || []).includes("Files");
};

const clipboardToImageFiles = (clipboardData: DataTransfer | null) => {
  if (!clipboardData) return [] as File[];

  const filesFromItems = Array.from(clipboardData.items || [])
    .filter((item) => item.kind === "file")
    .map((item) => item.getAsFile())
    .filter((file): file is File => file instanceof File)
    .filter((file) => isImageFile(file));

  if (filesFromItems.length) {
    return filesFromItems;
  }

  return filterImageFiles(clipboardData.files);
};

const clipboardHasText = (clipboardData: DataTransfer | null) => {
  if (!clipboardData) return false;
  return Boolean(clipboardData.getData("text/plain"));
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
          ? "border border-emerald-200/90 bg-white text-emerald-950 shadow-sm dark:border-emerald-400/20 dark:bg-white/10 dark:text-white"
          : "border border-transparent text-neutral-700 hover:bg-white/85 hover:text-emerald-950 dark:text-neutral-200/92 dark:hover:bg-white/8 dark:hover:text-white"
      } ${compact ? "justify-center px-0" : ""}`}
    >
      <SidebarIcon compact={compact}>{children}</SidebarIcon>
      {compact ? null : <span className="truncate text-[15px]">{label}</span>}
    </button>
  );
}

function ComposerIconButton({
  onClick,
  disabled,
  title,
  children,
}: {
  onClick?: () => void;
  disabled?: boolean;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      className="flex h-11 w-11 items-center justify-center rounded-full border border-black/8 bg-black/[0.03] text-neutral-700 transition-colors hover:bg-black/[0.06] hover:text-neutral-950 disabled:cursor-not-allowed disabled:opacity-50 dark:border-white/10 dark:bg-white/[0.045] dark:text-neutral-200 dark:hover:bg-white/[0.09] dark:hover:text-white"
    >
      {children}
    </button>
  );
}

function EmptyChatState() {
  return (
    <div className="mx-auto flex h-full w-full max-w-[900px] items-center justify-center px-4">
      <div className="text-center">
        <p className="text-sm font-medium tracking-[0.24em] text-emerald-600 dark:text-emerald-300">
          DISEASE AI
        </p>
        <h1 className="mt-5 text-3xl font-medium tracking-[-0.04em] text-neutral-900 dark:text-neutral-100 md:text-5xl">
          How can{" "}
          <span className="bg-gradient-to-r from-emerald-200 via-white to-emerald-300 bg-clip-text text-transparent dark:from-emerald-300 dark:via-white dark:to-emerald-200">
            Suphala AI
          </span>{" "}
          help you today?
        </h1>
        <p className="mt-4 text-sm leading-7 text-neutral-500 dark:text-neutral-400 md:text-base">
          Upload a diseased leaf image for analysis, or ask follow-up questions about the result.
        </p>
      </div>
    </div>
  );
}

function SectionCard({
  title,
  tone,
  body,
  items,
}: {
  title: string;
  tone: "amber" | "rose" | "emerald" | "sky";
  body?: string;
  items?: string[];
}) {
  const palette = {
    amber: {
      header: "bg-amber-500 dark:bg-amber-600",
      body: "bg-amber-50 dark:bg-amber-950/40",
      dot: "bg-amber-500",
    },
    rose: {
      header: "bg-rose-500 dark:bg-rose-600",
      body: "bg-rose-50 dark:bg-rose-950/35",
      dot: "bg-rose-500",
    },
    emerald: {
      header: "bg-emerald-600 dark:bg-emerald-700",
      body: "bg-emerald-50 dark:bg-emerald-950/35",
      dot: "bg-emerald-500",
    },
    sky: {
      header: "bg-sky-600 dark:bg-sky-700",
      body: "bg-sky-50 dark:bg-sky-950/35",
      dot: "bg-sky-500",
    },
  }[tone];

  return (
    <section className="overflow-hidden rounded-[24px] border border-black/10 bg-white/80 shadow-sm dark:border-white/10 dark:bg-neutral-950/70">
      <div
        className={`flex items-center gap-3 px-4 py-3 text-sm font-semibold text-white ${palette.header}`}
      >
        <span className="block h-3 w-3 rounded-full bg-white/90" />
        <span>{title}</span>
      </div>
      <div className={`px-5 py-4 ${palette.body}`}>
        {body ? (
          <p className="text-sm leading-7 text-neutral-800 dark:text-neutral-100">
            {body}
          </p>
        ) : null}
        {items?.length ? (
          <ul className="space-y-2.5">
            {items.map((item, index) => (
              <li
                key={`${title}-${index}`}
                className="flex items-start gap-3 text-sm leading-7 text-neutral-800 dark:text-neutral-100"
              >
                <span
                  className={`mt-2 block h-2.5 w-2.5 rounded-full ${palette.dot}`}
                />
                <span>{item}</span>
              </li>
            ))}
          </ul>
        ) : null}
      </div>
    </section>
  );
}

function AnalysisResultCard({ card }: { card: AnalysisCard }) {
  const urgencyPalette = {
    low: {
      wrapper:
        "border-emerald-200 bg-emerald-50 text-emerald-900 dark:border-emerald-700/60 dark:bg-emerald-950/45 dark:text-emerald-100",
      badge:
        "bg-emerald-600 text-white dark:bg-emerald-500 dark:text-black",
    },
    medium: {
      wrapper:
        "border-amber-200 bg-amber-50 text-amber-950 dark:border-amber-700/60 dark:bg-amber-950/45 dark:text-amber-50",
      badge: "bg-amber-500 text-white dark:bg-amber-500 dark:text-black",
    },
    high: {
      wrapper:
        "border-red-200 bg-red-50 text-red-950 dark:border-red-700/60 dark:bg-red-950/45 dark:text-red-50",
      badge: "bg-red-600 text-white dark:bg-red-500 dark:text-white",
    },
  }[card.urgency.level];

  return (
    <article className="overflow-hidden rounded-[30px] border border-black/10 bg-stone-50/95 shadow-[0_24px_80px_-32px_rgba(15,23,42,0.45)] dark:border-white/10 dark:bg-neutral-900/95">
      <div className="border-b border-black/10 px-5 py-5 dark:border-white/10 md:px-6">
        <div className="flex items-center gap-4">
          <div className="flex h-12 w-12 items-center justify-center rounded-[18px] bg-emerald-100 text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-300">
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
                d="M5 19c6 0 14-4 14-14C9 5 5 13 5 19Zm0 0 6-6"
              />
            </svg>
          </div>
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-neutral-500 dark:text-neutral-400">
              {card.title}
            </p>
            <h2 className="text-2xl font-semibold text-neutral-950 dark:text-neutral-50">
              {SUMMARY_CARD_LABEL}
            </h2>
          </div>
        </div>
      </div>

      <div className="p-5 md:p-6">
        <div className="grid gap-5 lg:grid-cols-[240px_minmax(0,1fr)]">
          <div className="overflow-hidden rounded-[26px] border border-black/10 bg-white shadow-sm dark:border-white/10 dark:bg-neutral-950">
            {card.imagePreview ? (
              <img
                src={card.imagePreview}
                alt={`${card.crop} leaf`}
                className="h-full min-h-[220px] w-full object-cover"
              />
            ) : (
              <div className="flex min-h-[220px] items-center justify-center bg-neutral-100 text-sm text-neutral-500 dark:bg-neutral-900 dark:text-neutral-400">
                No image preview
              </div>
            )}
          </div>

          <div className="rounded-[26px] border border-black/10 bg-white/90 p-5 shadow-sm dark:border-white/10 dark:bg-neutral-950/70">
            <div className="grid gap-3">
              <div className="border-b border-black/10 pb-3 dark:border-white/10">
                <span className="text-sm font-medium text-neutral-500 dark:text-neutral-400">
                  Crop
                </span>
                <p className="mt-1 text-2xl font-semibold text-neutral-950 dark:text-neutral-50">
                  {card.crop}
                </p>
              </div>
              <div className="border-b border-black/10 pb-3 dark:border-white/10">
                <span className="text-sm font-medium text-neutral-500 dark:text-neutral-400">
                  Disease
                </span>
                <p className="mt-1 text-2xl font-semibold text-neutral-950 dark:text-neutral-50">
                  {card.disease}
                </p>
              </div>
              <div className="border-b border-black/10 pb-3 dark:border-white/10">
                <span className="text-sm font-medium text-neutral-500 dark:text-neutral-400">
                  Confidence
                </span>
                <p className="mt-1 text-2xl font-semibold text-neutral-950 dark:text-neutral-50">
                  {card.confidenceText}
                </p>
              </div>
              <p className="text-sm leading-6 text-neutral-600 dark:text-neutral-300">
                {card.modelNote}
              </p>
              <p className="text-sm leading-6 text-neutral-600 dark:text-neutral-300">
                {card.verificationNote}
              </p>
            </div>
          </div>
        </div>

        <div className="mt-6 grid gap-4">
          <SectionCard title="Symptoms Detected" tone="amber" items={card.symptoms} />
          <SectionCard title="Cause of Disease" tone="rose" body={card.cause} />
          <SectionCard
            title="Treatment Advice"
            tone="emerald"
            items={card.treatments}
          />
          <SectionCard
            title="Prevention Tips"
            tone="sky"
            items={card.prevention}
          />
        </div>

        <div
          className={`mt-6 rounded-[24px] border px-5 py-4 shadow-sm ${urgencyPalette.wrapper}`}
        >
          <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
            <div>
              <p className="text-sm font-semibold uppercase tracking-[0.18em]">
                Urgency Level
              </p>
              <p className="mt-1 text-2xl font-semibold">{card.urgency.label}</p>
            </div>
            <span
              className={`inline-flex w-fit rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] ${urgencyPalette.badge}`}
            >
              {card.urgency.level === "high" ? "Urgent" : "Action Needed"}
            </span>
          </div>
          <p className="mt-3 text-sm leading-7 md:text-base">
            {card.urgency.note}
          </p>
        </div>
      </div>
    </article>
  );
}

export default function AIGrow() {
  const { isAuthenticated, isLoading } = useAuth();
  const AI_API_BASE =
    process.env.NEXT_PUBLIC_AI_API_URL || "http://localhost:8000/api/ai";

  const [chats, setChats] = useState<Chat[]>([]);
  const [currentChatId, setCurrentChatId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [selectedImages, setSelectedImages] = useState<QueuedImage[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [streamingContent, setStreamingContent] = useState("");
  const [streamingChatId, setStreamingChatId] = useState<string | null>(null);
  const [profileName, setProfileName] = useState("");
  const [profileContext, setProfileContext] = useState<any | null>(null);
  const [desktopSidebarOpen, setDesktopSidebarOpen] = useState(true);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [selectedSeason, setSelectedSeason] = useState<SeasonOption>("auto");
  const [seasonMenuOpen, setSeasonMenuOpen] = useState(false);
  const [speechSupported, setSpeechSupported] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [speechError, setSpeechError] = useState("");
  const [isDragActive, setIsDragActive] = useState(false);
  const [cameraDialogOpen, setCameraDialogOpen] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [isRequestingCamera, setIsRequestingCamera] = useState(false);

  const activeImageRequestIdRef = useRef<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const seasonMenuRef = useRef<HTMLDivElement>(null);
  const recognitionRef = useRef<any>(null);
  const speechBaseInputRef = useRef("");
  const dragDepthRef = useRef(0);
  const cameraVideoRef = useRef<HTMLVideoElement>(null);
  const cameraStreamRef = useRef<MediaStream | null>(null);
  const pendingLauncherIntentHandledRef = useRef(false);

  const currentChat =
    chats.find((chat) => chat.id === currentChatId) ?? null;
  const messages = currentChat?.messages ?? [];
  const orderedChats = orderChats(chats);
  const selectedSeasonLabel =
    SEASON_OPTIONS.find((option) => option.value === selectedSeason)?.label ??
    "Auto Detect";

  const updateChat = (
    chatId: string,
    updater: (chat: Chat) => Chat,
  ) => {
    setChats((prev) =>
      prev.map((chat) => (chat.id === chatId ? updater(chat) : chat)),
    );
  };

  const ensureActiveChat = (suggestedTitle?: string) => {
    if (currentChatId && chats.some((chat) => chat.id === currentChatId)) {
      return currentChatId;
    }

    const newChat = buildEmptyChat(trimTitle(suggestedTitle || "New Chat"));
    setChats((prev) => [newChat, ...prev]);
    setCurrentChatId(newChat.id);
    return newChat.id;
  };

  const renameChat = (chatId: string, title: string) => {
    updateChat(chatId, (chat) => ({
      ...chat,
      title: trimTitle(title),
      updatedAt: createTimestamp(),
    }));
  };

  const appendMessage = (
    chatId: string,
    message: Message,
    preferredTitle?: string,
  ) => {
    updateChat(chatId, (chat) => ({
      ...chat,
      title:
        preferredTitle && preferredTitle !== "New Chat"
          ? trimTitle(preferredTitle)
          : chat.messages.length === 0 && message.type === "user"
            ? trimTitle(message.content || chat.title)
            : chat.title,
      updatedAt: createTimestamp(),
      messages: [...chat.messages, message],
    }));
  };

  const setPredictionContext = (chatId: string, context: any) => {
    updateChat(chatId, (chat) => ({
      ...chat,
      updatedAt: createTimestamp(),
      lastPredictionContext: context,
    }));
  };

  const setAdvisorContext = (chatId: string, context: any) => {
    updateChat(chatId, (chat) => ({
      ...chat,
      updatedAt: createTimestamp(),
      advisorContext: context,
    }));
  };

  const selectChat = (chatId: string) => {
    setCurrentChatId(chatId);
    setMobileSidebarOpen(false);
  };

  const startNewChat = () => {
    const newChat = buildEmptyChat("New Chat");
    setChats((prev) => [newChat, ...prev]);
    setCurrentChatId(newChat.id);
    setSelectedImages([]);
    setInput("");
    setMobileSidebarOpen(false);
  };

  const clearQueuedImages = () => {
    setSelectedImages([]);
  };

  const stopCameraStream = () => {
    const stream = cameraStreamRef.current;
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      cameraStreamRef.current = null;
    }

    if (cameraVideoRef.current) {
      cameraVideoRef.current.srcObject = null;
    }
  };

  const closeCameraDialog = () => {
    stopCameraStream();
    setCameraDialogOpen(false);
    setIsRequestingCamera(false);
  };

  useEffect(() => {
    if (currentChatId) return;
    if (!chats.length) return;
    setCurrentChatId(chats[0].id);
  }, [chats, currentChatId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingContent, currentChatId]);

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

  useEffect(() => {
    const toggleSidebar = () => {
      if (window.innerWidth >= 768) {
        setDesktopSidebarOpen((prev) => !prev);
        return;
      }
      setMobileSidebarOpen((prev) => !prev);
    };

    window.addEventListener(AI_HISTORY_TOGGLE_EVENT, toggleSidebar);
    return () => {
      window.removeEventListener(AI_HISTORY_TOGGLE_EVENT, toggleSidebar);
    };
  }, []);

  useEffect(() => {
    const previousBodyOverflow = document.body.style.overflow;
    const previousHtmlOverflow = document.documentElement.style.overflow;
    document.body.style.overflow = "hidden";
    document.documentElement.style.overflow = "hidden";

    return () => {
      document.body.style.overflow = previousBodyOverflow;
      document.documentElement.style.overflow = previousHtmlOverflow;
    };
  }, []);

  useEffect(() => {
    const recognitionCtor =
      typeof window !== "undefined"
        ? (window as any).SpeechRecognition ||
          (window as any).webkitSpeechRecognition
        : null;
    setSpeechSupported(Boolean(recognitionCtor));

    return () => {
      recognitionRef.current?.stop?.();
      recognitionRef.current = null;
    };
  }, []);

  useEffect(() => {
    const preventBrowserFileDrop = (event: DragEvent) => {
      if (!hasDraggedFiles(event.dataTransfer)) return;
      event.preventDefault();
    };

    const resetDragState = () => {
      dragDepthRef.current = 0;
      setIsDragActive(false);
    };

    window.addEventListener("dragover", preventBrowserFileDrop);
    window.addEventListener("drop", preventBrowserFileDrop);
    window.addEventListener("drop", resetDragState);
    window.addEventListener("dragend", resetDragState);
    return () => {
      window.removeEventListener("dragover", preventBrowserFileDrop);
      window.removeEventListener("drop", preventBrowserFileDrop);
      window.removeEventListener("drop", resetDragState);
      window.removeEventListener("dragend", resetDragState);
    };
  }, []);

  useEffect(() => {
    if (!cameraDialogOpen || !cameraVideoRef.current || !cameraStreamRef.current) {
      return;
    }

    cameraVideoRef.current.srcObject = cameraStreamRef.current;
    cameraVideoRef.current.play().catch(() => {
      setCameraError("Unable to start the camera preview.");
    });
  }, [cameraDialogOpen]);

  useEffect(() => {
    return () => {
      stopCameraStream();
    };
  }, []);

  useEffect(() => {
    if (!seasonMenuOpen) return;

    const handlePointerDown = (event: PointerEvent) => {
      if (!seasonMenuRef.current?.contains(event.target as Node)) {
        setSeasonMenuOpen(false);
      }
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setSeasonMenuOpen(false);
      }
    };

    window.addEventListener("pointerdown", handlePointerDown);
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("pointerdown", handlePointerDown);
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [seasonMenuOpen]);

  useEffect(() => {
    if (isAnalyzing) {
      setSeasonMenuOpen(false);
    }
  }, [isAnalyzing]);

  const streamResponse = (fullText: string, chatId: string, messageId: string) => {
    let charIndex = 0;
    setStreamingChatId(chatId);
    setStreamingContent("");

    const streamChar = () => {
      if (charIndex < fullText.length) {
        setStreamingContent((prev) => prev + fullText[charIndex]);
        charIndex += 1;
        window.setTimeout(streamChar, 15);
        return;
      }

      appendMessage(
        chatId,
        {
          id: messageId,
          type: "ai",
          content: fullText,
        },
      );
      setStreamingContent("");
      setStreamingChatId(null);
      setIsAnalyzing(false);
    };

    streamChar();
  };

  const addAiMessage = (
    chatId: string,
    message: Message,
    preferredTitle?: string,
  ) => {
    appendMessage(chatId, message, preferredTitle);
    setStreamingContent("");
    setStreamingChatId(null);
    setIsAnalyzing(false);
  };

  const queueSelectedFiles = async (
    files: File[],
    resetInput?: HTMLInputElement | null,
    sourceType: QueuedImageSource = "upload",
  ) => {
    if (isAnalyzing) return;
    if (!files.length) return;

    try {
      const queued = await Promise.all(
        files.map(async (file) => ({
          id: createRequestId(),
          file,
          preview: await readFileAsDataUrl(file),
          sourceType,
        })),
      );
      setSelectedImages((prev) => [...prev, ...queued]);
      setCameraError("");
    } catch {
      const chatId = ensureActiveChat("Leaf Analysis");
      addAiMessage(chatId, {
        id: createRequestId(),
        type: "ai",
        content: "Unable to read the selected image files.",
      });
    } finally {
      if (resetInput) resetInput.value = "";
    }
  };

  const handleImageUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    await queueSelectedFiles(
      Array.from(event.target.files || []),
      fileInputRef.current,
      "upload",
    );
  };

  const handleCameraCapture = async (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    await queueSelectedFiles(
      Array.from(event.target.files || []),
      cameraInputRef.current,
      "camera",
    );
  };

  const handleComposerPaste = async (
    event: React.ClipboardEvent<HTMLTextAreaElement>,
  ) => {
    const pastedImages = clipboardToImageFiles(event.clipboardData);
    if (!pastedImages.length) return;

    if (!clipboardHasText(event.clipboardData)) {
      event.preventDefault();
    }

    await queueSelectedFiles(pastedImages, null, "upload");
  };

  const openUploadLibrary = () => {
    if (cameraDialogOpen) {
      closeCameraDialog();
    }
    fileInputRef.current?.click();
  };

  const handlePageDragEnter = (event: React.DragEvent<HTMLDivElement>) => {
    if (!hasDraggedFiles(event.dataTransfer)) return;
    event.preventDefault();
    event.stopPropagation();
    dragDepthRef.current += 1;
    setIsDragActive(true);
    setCameraError("");
  };

  const handlePageDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    if (!hasDraggedFiles(event.dataTransfer)) return;
    event.preventDefault();
    event.stopPropagation();
    event.dataTransfer.dropEffect = "copy";
    if (!isDragActive) {
      setIsDragActive(true);
    }
  };

  const handlePageDragLeave = (event: React.DragEvent<HTMLDivElement>) => {
    if (!hasDraggedFiles(event.dataTransfer)) return;
    event.preventDefault();
    event.stopPropagation();
    dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);
    if (dragDepthRef.current === 0) {
      setIsDragActive(false);
    }
  };

  const handlePageDrop = async (event: React.DragEvent<HTMLDivElement>) => {
    if (!hasDraggedFiles(event.dataTransfer)) return;
    event.preventDefault();
    event.stopPropagation();
    dragDepthRef.current = 0;
    setIsDragActive(false);

    const droppedImages = filterImageFiles(event.dataTransfer.files);
    if (!droppedImages.length) {
      const chatId = ensureActiveChat("Leaf Analysis");
      addAiMessage(chatId, {
        id: createRequestId(),
        type: "ai",
        content: "Drop one or more image files here to analyze them.",
      });
      return;
    }

    await queueSelectedFiles(droppedImages, null, "upload");
  };

  useEffect(() => {
    const handleWindowPaste = async (event: ClipboardEvent) => {
      if (event.target instanceof HTMLTextAreaElement) {
        return;
      }

      const pastedImages = clipboardToImageFiles(event.clipboardData);
      if (!pastedImages.length) return;

      event.preventDefault();
      await queueSelectedFiles(pastedImages, null, "upload");
    };

    window.addEventListener("paste", handleWindowPaste);
    return () => {
      window.removeEventListener("paste", handleWindowPaste);
    };
  }, [queueSelectedFiles]);

  const requestCameraCapture = async () => {
    setCameraError("");

    if (
      typeof navigator === "undefined" ||
      !navigator.mediaDevices ||
      typeof navigator.mediaDevices.getUserMedia !== "function"
    ) {
      cameraInputRef.current?.click();
      return;
    }

    setIsRequestingCamera(true);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: "environment" },
        },
        audio: false,
      });

      stopCameraStream();
      cameraStreamRef.current = stream;
      setCameraDialogOpen(true);
    } catch (error: any) {
      const permissionError =
        error?.name === "NotAllowedError" ||
        error?.name === "PermissionDeniedError";
      setCameraError(
        permissionError
          ? "Camera permission was denied. Choose a photo instead or allow camera access."
          : "Unable to open the camera right now. Choose a photo instead.",
      );
    } finally {
      setIsRequestingCamera(false);
    }
  };

  const captureCameraPhoto = async () => {
    const video = cameraVideoRef.current;
    if (!video) {
      setCameraError("Camera preview is not ready yet.");
      return;
    }

    const width = video.videoWidth || 1280;
    const height = video.videoHeight || 720;
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext("2d");

    if (!context) {
      setCameraError("Unable to capture the photo right now.");
      return;
    }

    context.drawImage(video, 0, 0, width, height);

    const blob = await new Promise<Blob | null>((resolve) => {
      canvas.toBlob(resolve, "image/jpeg", 0.92);
    });

    if (!blob) {
      setCameraError("Unable to save the captured photo.");
      return;
    }

    const photoFile = new File([blob], `camera-capture-${Date.now()}.jpg`, {
      type: "image/jpeg",
      lastModified: Date.now(),
    });

    await queueSelectedFiles([photoFile], null, "camera");
    closeCameraDialog();
  };

  const toggleSpeechInput = () => {
    if (!speechSupported) {
      setSpeechError("Speech input is not supported in this browser.");
      return;
    }

    if (isListening) {
      recognitionRef.current?.stop?.();
      return;
    }

    const SpeechRecognitionCtor =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition;

    if (!SpeechRecognitionCtor) {
      setSpeechError("Speech input is not supported in this browser.");
      return;
    }

    const recognition = new SpeechRecognitionCtor();
    speechBaseInputRef.current = input.trim();
    recognition.lang = "en-US";
    recognition.interimResults = true;
    recognition.continuous = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
      setSpeechError("");
      setIsListening(true);
    };
    recognition.onresult = (event: any) => {
      const transcript = Array.from(event.results || [])
        .map((result: any) => result?.[0]?.transcript || "")
        .join(" ");
      const nextText = normalizeWhitespace(
        `${speechBaseInputRef.current} ${transcript}`,
      );
      setInput(nextText);
    };
    recognition.onerror = (event: any) => {
      const message =
        event?.error === "not-allowed"
          ? "Microphone permission was denied."
          : "Speech input failed. Please try again.";
      setSpeechError(message);
    };
    recognition.onend = () => {
      setIsListening(false);
      recognitionRef.current = null;
    };

    recognitionRef.current = recognition;
    recognition.start();
  };

  const analyzeImageBatch = async (
    queuedImages: QueuedImage[],
    userText: string,
  ) => {
    if (isAnalyzing) return;

    const prompt = userText.trim();
    const imageCount = queuedImages.length;
    const analysisName = createAnalysisName(imageCount);
    const requestId = createRequestId();
    const chatId = ensureActiveChat(prompt || "Leaf Analysis");
    const userContent =
      prompt || `Analyze ${imageCount} image${imageCount > 1 ? "s" : ""} for crop disease.`;

    activeImageRequestIdRef.current = requestId;

    appendMessage(
      chatId,
      {
        id: createRequestId(),
        type: "user",
        content: userContent,
        images: queuedImages.map((image) => image.preview),
      },
      prompt || "Leaf Analysis",
    );

    clearQueuedImages();
    setInput("");
    setIsAnalyzing(true);

    const startedAt = performance.now();

    try {
      const formData = new FormData();
      queuedImages.forEach((image) => {
        formData.append("files", image.file, image.file.name || "leaf.jpg");
        formData.append("file_sources", image.sourceType);
      });
      formData.append("season", selectedSeason);
      formData.append("request_id", requestId);
      formData.append("request_name", analysisName);
      if (prompt) {
        formData.append("message", prompt);
      }

      const data = await fetchJson(
        `${AI_API_BASE}/predict`,
        {
          method: "POST",
          body: formData,
        },
        180000,
      );

      if (
        activeImageRequestIdRef.current !== requestId ||
        data?.request_id !== requestId
      ) {
        return;
      }

      const predictionRows =
        Array.isArray(data?.results) && data.results.length > 0
          ? data.results
          : [data];

      const cards = buildAnalysisCards(predictionRows, queuedImages);
      const thinkingTime = formatThinkingTime(startedAt);
      const messageBody = [
        `Analysis Name: ${data?.request_name || analysisName}`,
        "",
        ...cards.map(
          (card) =>
            `${card.crop}: ${card.disease} (${card.confidenceText}) - ${card.urgency.label}`,
        ),
      ].join("\n");

      addAiMessage(
        chatId,
        {
          id: createRequestId(),
          type: "ai",
          content: prependThinkingTime(thinkingTime, messageBody),
          analysisCards: cards,
          analysisName: data?.request_name || analysisName,
        },
        getAnalysisTitle(cards, analysisName),
      );

      setPredictionContext(chatId, data);
    } catch (error) {
      if (activeImageRequestIdRef.current === requestId) {
        const thinkingTime = formatThinkingTime(startedAt);
        addAiMessage(chatId, {
          id: createRequestId(),
          type: "ai",
          content: prependThinkingTime(
            thinkingTime,
            error instanceof Error
              ? error.message
              : "Error connecting to the AI server.",
          ),
        });
      }
    } finally {
      if (activeImageRequestIdRef.current === requestId) {
        activeImageRequestIdRef.current = null;
      }
    }
  };

  const sendTextMessage = async (prompt: string) => {
    if (!prompt) return;

    const isGreetingPrompt = isGreetingMessage(prompt);
    const chatId = ensureActiveChat(prompt);
    const startedAt = performance.now();

    appendMessage(chatId, {
      id: createRequestId(),
      type: "user",
      content: prompt,
    });
    setInput("");
    setIsAnalyzing(true);

    try {
      const predictionContext =
        chats.find((chat) => chat.id === chatId)?.lastPredictionContext ||
        (chatId === currentChatId ? currentChat?.lastPredictionContext : null) ||
        null;
      const currentAdvisorContext =
        chats.find((chat) => chat.id === chatId)?.advisorContext ||
        (chatId === currentChatId ? currentChat?.advisorContext : null) ||
        null;
      const history =
        buildConversationHistory(
          chats.find((chat) => chat.id === chatId) ||
            (chatId === currentChatId ? currentChat : null) ||
            null,
        ) || [];

      const data = await fetchJson(
        `${AI_API_BASE}/chat`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            message: prompt,
            context: predictionContext,
            profile_name: profileName,
            profile_context: profileContext,
            advisor_context: currentAdvisorContext,
            conversation_history: history,
          }),
        },
        90000,
      );

      const reply = data?.reply || "Unable to generate a response right now.";
      if (Object.prototype.hasOwnProperty.call(data || {}, "advisor_context")) {
        setAdvisorContext(chatId, data?.advisor_context ?? null);
      }
      const thinkingTime = formatThinkingTime(startedAt);
      streamResponse(
        isGreetingPrompt ? reply : prependThinkingTime(thinkingTime, reply),
        chatId,
        createRequestId(),
      );
    } catch (error) {
      const thinkingTime = formatThinkingTime(startedAt);
      streamResponse(
        prependThinkingTime(
          thinkingTime,
          error instanceof Error
            ? error.message
            : "Error connecting to the AI assistant.",
        ),
        chatId,
        createRequestId(),
      );
    }
  };

  const handleSendMessage = async (
    promptOverride?: string,
    queuedImagesOverride?: QueuedImage[],
  ) => {
    if (isAnalyzing) return;

    const prompt = (promptOverride ?? input).trim();
    const queuedImages = queuedImagesOverride ?? selectedImages;
    const hasImages = queuedImages.length > 0;
    if (!prompt && !hasImages) return;

    if (hasImages) {
      await analyzeImageBatch(queuedImages, prompt);
      return;
    }

    await sendTextMessage(prompt);
  };

  useEffect(() => {
    if (pendingLauncherIntentHandledRef.current) return;
    if (isLoading || !isAuthenticated || isAnalyzing) return;

    pendingLauncherIntentHandledRef.current = true;
    const pendingIntent = consumePendingSuphalaIntent("disease");
    if (!pendingIntent) return;

    const launchPendingIntent = async () => {
      const cleanPrompt = normalizeWhitespace(pendingIntent.prompt || "");

      if (pendingIntent.images.length > 0) {
        try {
          const queuedImages = await Promise.all(
            pendingIntent.images.map(async (file) => ({
              id: createRequestId(),
              file,
              preview: await readFileAsDataUrl(file),
              sourceType: "upload" as QueuedImageSource,
            })),
          );

          await handleSendMessage(cleanPrompt, queuedImages);
        } catch {
          const chatId = ensureActiveChat("Leaf Analysis");
          addAiMessage(chatId, {
            id: createRequestId(),
            type: "ai",
            content:
              "Unable to reopen the image selected from Suphala AI. Please upload the leaf image again.",
          });
        }
        return;
      }

      if (pendingIntent.imageCount > 0) {
        const chatId = ensureActiveChat("Leaf Analysis");
        addAiMessage(chatId, {
          id: createRequestId(),
          type: "ai",
          content:
            "Your previous Suphala AI image selection expired before we could analyze it. Please upload the leaf image again.",
        });
        return;
      }

      if (cleanPrompt) {
        await handleSendMessage(cleanPrompt);
      }
    };

    void launchPendingIntent();
  }, [isAuthenticated, isAnalyzing, isLoading, handleSendMessage]);

  if (isLoading) {
    return (
      <div className="flex h-[calc(100dvh-5rem)] items-center justify-center bg-white text-neutral-900 dark:bg-black dark:text-white">
        <div className="space-y-4 text-center">
          <div className="mx-auto h-12 w-12 animate-spin rounded-full border-b-2 border-emerald-500" />
          <p className="text-neutral-600 dark:text-neutral-400">Loading...</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <div className="flex h-[calc(100dvh-5rem)] items-center justify-center bg-white px-6 text-neutral-900 dark:bg-black dark:text-white">
        <div className="mx-auto max-w-md space-y-6 rounded-[28px] border border-black/10 bg-white/90 p-8 text-center shadow-lg dark:border-white/10 dark:bg-neutral-950/90">
          <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-emerald-500 text-white">
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
            Please sign in to access the AI crop disease assistant.
          </p>
          <Link
            href="/login"
            className="inline-flex w-full items-center justify-center rounded-2xl bg-emerald-600 px-5 py-3 font-semibold text-white transition-colors hover:bg-emerald-700"
          >
            Sign In to Continue
          </Link>
          <p className="text-sm text-neutral-500 dark:text-neutral-400">
            Do not have an account?{" "}
            <Link
              href="/register"
              className="font-medium text-emerald-600 hover:text-emerald-700 dark:text-emerald-400 dark:hover:text-emerald-300"
            >
              Create one here
            </Link>
          </p>
        </div>
      </div>
    );
  }

  const sidebarCompact = !desktopSidebarOpen;

  return (
    <div
      className="relative flex h-[calc(100dvh-5rem)] overflow-hidden bg-transparent text-neutral-900 dark:text-neutral-100"
      onDragEnter={handlePageDragEnter}
      onDragOver={handlePageDragOver}
      onDragLeave={handlePageDragLeave}
      onDrop={handlePageDrop}
    >
      {mobileSidebarOpen ? (
        <button
          type="button"
          aria-label="Close history panel"
          onClick={() => setMobileSidebarOpen(false)}
          className="absolute inset-0 z-20 bg-black/45 md:hidden"
        />
      ) : null}

      {isDragActive ? (
        <div className="pointer-events-none absolute inset-0 z-[35] flex items-center justify-center bg-emerald-950/18 p-4 backdrop-blur-sm">
          <div className="w-full max-w-xl rounded-[32px] border-2 border-dashed border-emerald-400 bg-white/92 px-6 py-10 text-center shadow-2xl dark:bg-neutral-950/92">
            <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-[22px] bg-emerald-100 text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-300">
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
                  d="M12 16V6m0 0-4 4m4-4 4 4M5 16v1a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-1"
                />
              </svg>
            </div>
            <p className="mt-5 text-sm font-semibold uppercase tracking-[0.2em] text-emerald-700 dark:text-emerald-300">
              Drop Images
            </p>
            <p className="mt-3 text-2xl font-semibold text-neutral-950 dark:text-white">
              Drop leaf images here to upload
            </p>
            <p className="mt-3 text-sm leading-7 text-neutral-600 dark:text-neutral-300">
              PNG, JPG, WEBP, HEIC, and other image files are accepted.
            </p>
          </div>
        </div>
      ) : null}

      <aside
        className={`absolute inset-y-0 left-0 z-30 flex w-[88vw] max-w-[320px] flex-col overflow-y-auto overflow-x-hidden border-r border-emerald-100/90 bg-[linear-gradient(180deg,rgba(248,251,248,0.98),rgba(236,246,239,0.96))] text-neutral-900 shadow-2xl dark:border-white/10 dark:bg-[#111111] dark:text-neutral-100 md:static md:z-0 md:max-w-none md:shadow-none ${
          mobileSidebarOpen ? "translate-x-0" : "-translate-x-full"
        } transition-transform duration-300 md:translate-x-0 ${
          desktopSidebarOpen ? "md:w-[18rem]" : "md:w-[5.5rem]"
        }`}
      >
        <div className="sticky top-0 z-10 border-b border-black/5 bg-[linear-gradient(180deg,rgba(248,251,248,0.98),rgba(236,246,239,0.96))] px-4 pb-3 pt-4 backdrop-blur-xl dark:border-white/8 dark:bg-[#111111]/95">
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
            <span className="flex h-11 w-11 items-center justify-center rounded-2xl bg-black/[0.03] ring-1 ring-black/5 dark:bg-white/[0.04] dark:ring-white/10">
              <img
                src="/ai-logo.png"
                alt="Suphala AI logo"
                className="h-7 w-7 object-contain"
              />
            </span>
            {sidebarCompact ? null : (
              <div className="min-w-0 flex-1 text-left">
                <span className="block truncate text-[17px] font-medium tracking-[-0.03em] text-neutral-900 dark:text-neutral-100">
                  Suphala AI
                </span>
              </div>
            )}
          </button>
        </div>

        <div className="px-3 py-3">
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
        </div>

        <div className="flex-1 px-3 pb-4">
          {sidebarCompact ? null : (
            <div className="mb-3 px-2">
              <p className="text-[12px] font-medium uppercase tracking-[0.2em] text-emerald-700/90 dark:text-emerald-300/85">
                Your Chats
              </p>
            </div>
          )}

          {orderedChats.length ? (
            <div className={`space-y-1 ${sidebarCompact ? "md:flex md:flex-col md:items-center md:gap-2 md:space-y-0" : ""}`}>
              {orderedChats.map((chat) => (
                <button
                  key={chat.id}
                  type="button"
                  onClick={() => selectChat(chat.id)}
                  title={chat.title}
                  className={`group flex w-full items-center gap-3 rounded-2xl px-2 py-2.5 text-left text-sm transition-colors ${
                    currentChatId === chat.id
                      ? "border border-emerald-200/90 bg-white text-emerald-950 shadow-sm dark:border-emerald-400/20 dark:bg-white/10 dark:text-white"
                      : "border border-transparent text-neutral-700 hover:bg-white/85 hover:text-emerald-950 dark:text-neutral-200/92 dark:hover:bg-white/[0.06] dark:hover:text-white"
                  } ${sidebarCompact ? "justify-center px-0 md:h-11 md:w-11" : ""}`}
                >
                  {sidebarCompact ? (
                    <span className="text-xs font-semibold">
                      {chat.title.slice(0, 1).toUpperCase() || "C"}
                    </span>
                  ) : (
                    <>
                      <span
                        className={`mt-1 h-2 w-2 flex-shrink-0 rounded-full ${
                          currentChatId === chat.id
                            ? "bg-emerald-500 dark:bg-emerald-300"
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
          ) : (
            <div className={`rounded-[24px] border border-dashed border-emerald-200 bg-white/78 px-4 py-8 text-center text-sm text-neutral-500 dark:border-white/10 dark:bg-white/[0.03] dark:text-neutral-400 ${sidebarCompact ? "hidden md:block md:px-2 md:py-3 md:text-[11px]" : ""}`}>
              No chats yet.
            </div>
          )}
        </div>
      </aside>

      <section className="relative flex min-w-0 flex-1 flex-col overflow-hidden bg-[#f6faf7] dark:bg-[#1b1b1b]">
        <div className="flex-1 overflow-y-auto px-4 pb-[13.5rem] pt-6 md:px-6 md:pb-[15rem] md:pt-8">
          {messages.length === 0 ? (
            <EmptyChatState />
          ) : (
            <div className="mx-auto w-full max-w-[900px] space-y-6 pb-4">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${
                    message.type === "user" ? "justify-end" : "justify-start"
                  }`}
                >
                  {message.analysisCards?.length ? (
                    <div className="w-full space-y-5">
                      {message.analysisName ? (
                        <div className="rounded-[22px] border border-emerald-200/70 bg-white/78 px-4 py-3 text-sm font-medium text-emerald-900 shadow-sm dark:border-emerald-400/15 dark:bg-white/[0.04] dark:text-emerald-200">
                          {message.analysisName}
                        </div>
                      ) : null}
                      {message.analysisCards.map((card) => (
                        <AnalysisResultCard key={card.id} card={card} />
                      ))}
                    </div>
                  ) : (
                    <div
                      className={`max-w-[78%] rounded-[28px] px-5 py-4 ${
                        message.type === "user"
                          ? "border border-emerald-200/80 bg-white text-neutral-900 shadow-sm dark:border-white/10 dark:bg-[#2b2b2b] dark:text-white"
                          : "text-neutral-800 dark:text-neutral-100"
                      }`}
                    >
                      {message.images?.length ? (
                        <div className="mb-4 grid grid-cols-2 gap-3">
                          {message.images.map((image, index) => (
                            <img
                              key={`${message.id}-image-${index}`}
                              src={image}
                              alt={`Uploaded leaf ${index + 1}`}
                              className="h-32 w-full rounded-2xl object-cover"
                            />
                          ))}
                        </div>
                      ) : null}
                      <p className="whitespace-pre-wrap text-sm leading-7 md:text-[15px]">
                        {message.content}
                      </p>
                    </div>
                  )}
                </div>
              ))}

              {streamingContent && streamingChatId === currentChatId ? (
                <div className="flex justify-start">
                  <div className="max-w-[78%] text-sm leading-7 text-neutral-800 dark:text-neutral-100 md:text-[15px]">
                    <p className="whitespace-pre-wrap">
                      {streamingContent}
                      <span className="animate-pulse text-emerald-500 dark:text-emerald-300">
                        |
                      </span>
                    </p>
                  </div>
                </div>
              ) : null}

              {isAnalyzing && !streamingContent ? (
                <div className="flex justify-start">
                  <div className="w-full max-w-3xl rounded-[28px] border border-black/8 bg-white/78 p-5 shadow-sm dark:border-white/10 dark:bg-[#242424]">
                    <div className="space-y-3">
                      <div className="h-6 w-52 rounded-full bg-emerald-100 dark:bg-neutral-800" />
                      <div className="h-4 w-full rounded-full bg-neutral-200 dark:bg-neutral-800" />
                      <div className="h-4 w-5/6 rounded-full bg-neutral-200 dark:bg-neutral-800" />
                      <div className="mt-4 h-32 rounded-[24px] bg-neutral-200 dark:bg-neutral-800" />
                    </div>
                  </div>
                </div>
              ) : null}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        <div className="pointer-events-none absolute inset-x-0 bottom-0 z-20 px-3 pb-3 md:px-6 md:pb-6">
          <div className="mx-auto w-full max-w-[900px] pointer-events-auto">
            {selectedImages.length ? (
              <div className="mb-3 rounded-[26px] border border-emerald-200/70 bg-white/78 p-3 shadow-sm dark:border-white/10 dark:bg-[#242424]/96">
                <div className="mb-3 flex items-center justify-between gap-3">
                  <p className="text-xs font-medium uppercase tracking-[0.18em] text-emerald-700 dark:text-emerald-300">
                    {selectedImages.length} image
                    {selectedImages.length > 1 ? "s" : ""} ready
                  </p>
                  <button
                    type="button"
                    onClick={clearQueuedImages}
                    className="text-xs font-semibold uppercase tracking-[0.16em] text-neutral-500 transition-colors hover:text-red-600 dark:text-neutral-400 dark:hover:text-red-300"
                  >
                    Clear
                  </button>
                </div>

                <div className="flex flex-wrap gap-3">
                  {selectedImages.map((item) => (
                    <div key={item.id} className="relative">
                      <img
                        src={item.preview}
                        alt={item.file.name || "Selected leaf"}
                        className="h-16 w-16 rounded-2xl border border-black/10 object-cover dark:border-white/10"
                      />
                      <button
                        type="button"
                        onClick={() =>
                          setSelectedImages((prev) =>
                            prev.filter((image) => image.id !== item.id),
                          )
                        }
                        className="absolute -right-1 -top-1 flex h-5 w-5 items-center justify-center rounded-full bg-red-600 text-[10px] font-bold text-white transition-colors hover:bg-red-700"
                        aria-label="Remove selected image"
                      >
                        x
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}

            <div className="rounded-[32px] border border-black/8 bg-white/76 shadow-[0_24px_50px_-30px_rgba(15,23,42,0.35)] backdrop-blur-2xl dark:border-white/10 dark:bg-[#242424]/96 dark:shadow-[0_24px_60px_-32px_rgba(0,0,0,0.75)]">
              <div className="px-3 py-3 md:px-4 md:py-4">
                <textarea
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  onPaste={handleComposerPaste}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault();
                      handleSendMessage();
                    }
                  }}
                  placeholder="Ask anything"
                  rows={1}
                  className="max-h-40 min-h-[28px] w-full resize-none bg-transparent px-2 py-1 text-[15px] leading-7 text-neutral-900 outline-none placeholder:text-neutral-500 dark:text-white dark:placeholder:text-neutral-400 md:text-base"
                />

                <div className="mt-3 flex flex-wrap items-center justify-between gap-3 border-t border-black/8 pt-3 dark:border-white/8">
                  <div className="flex items-center gap-2">
                    <ComposerIconButton
                      onClick={openUploadLibrary}
                      disabled={isAnalyzing}
                      title="Upload image"
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
                          d="M4 16.5 8.5 12a2 2 0 0 1 2.828 0L16 16.5m-2-2 1.5-1.5a2 2 0 0 1 2.828 0L20 14.5M7 20h10a3 3 0 0 0 3-3V7a3 3 0 0 0-3-3H7a3 3 0 0 0-3 3v10a3 3 0 0 0 3 3Zm0-11.25h.01"
                        />
                      </svg>
                    </ComposerIconButton>

                    <ComposerIconButton
                      onClick={requestCameraCapture}
                      disabled={isAnalyzing || isRequestingCamera}
                      title={
                        isRequestingCamera
                          ? "Requesting camera access"
                          : "Open camera"
                      }
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
                          d="M4 8.5A2.5 2.5 0 0 1 6.5 6H9l1.2-1.6A2 2 0 0 1 11.8 3.5h.4a2 2 0 0 1 1.6.9L15 6h2.5A2.5 2.5 0 0 1 20 8.5v8A2.5 2.5 0 0 1 17.5 19h-11A2.5 2.5 0 0 1 4 16.5v-8Z"
                        />
                        <circle cx="12" cy="12.5" r="3.25" />
                      </svg>
                    </ComposerIconButton>
                  </div>

                  <div className="flex items-center gap-2">
                    <div ref={seasonMenuRef} className="relative">
                      <button
                        type="button"
                        onClick={() => setSeasonMenuOpen((prev) => !prev)}
                        disabled={isAnalyzing}
                        className={`inline-flex min-h-11 items-center gap-2 rounded-full px-3 text-sm font-medium text-neutral-600 outline-none transition-colors hover:text-neutral-950 focus-visible:text-neutral-950 disabled:cursor-not-allowed disabled:opacity-50 dark:text-neutral-300 dark:hover:text-white dark:focus-visible:text-white ${
                          seasonMenuOpen ? "text-neutral-950 dark:text-white" : ""
                        }`}
                        aria-haspopup="listbox"
                        aria-expanded={seasonMenuOpen}
                        aria-label="Select season"
                      >
                        <span className="max-w-[8.5rem] truncate">
                          {selectedSeasonLabel}
                        </span>
                        <svg
                          className={`h-4 w-4 text-neutral-500 transition-transform dark:text-neutral-400 ${
                            seasonMenuOpen ? "rotate-180 text-neutral-700 dark:text-neutral-200" : ""
                          }`}
                          viewBox="0 0 20 20"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="1.8"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="m5 7.5 5 5 5-5"
                          />
                        </svg>
                      </button>

                      <div
                        className={`absolute bottom-full right-0 z-30 mb-2 w-[min(12rem,calc(100vw-3rem))] overflow-hidden rounded-[24px] border border-black/10 bg-[linear-gradient(180deg,rgba(255,255,255,0.98),rgba(239,251,243,0.96))] p-2 shadow-[0_24px_60px_-32px_rgba(15,23,42,0.35)] backdrop-blur-xl transition-all dark:border-white/10 dark:bg-[linear-gradient(180deg,rgba(47,47,47,0.98),rgba(34,34,34,0.98))] dark:shadow-[0_24px_60px_-32px_rgba(0,0,0,0.85)] ${
                          seasonMenuOpen
                            ? "pointer-events-auto translate-y-0 opacity-100"
                            : "pointer-events-none translate-y-1 opacity-0"
                        }`}
                        role="listbox"
                        aria-label="Season options"
                      >
                        {SEASON_OPTIONS.map((option) => {
                          const isSelected = option.value === selectedSeason;

                          return (
                            <button
                              key={option.value}
                              type="button"
                              onClick={() => {
                                setSelectedSeason(option.value);
                                setSeasonMenuOpen(false);
                              }}
                              className={`flex w-full items-center justify-between rounded-2xl px-3 py-2.5 text-left text-sm transition-colors ${
                                isSelected
                                  ? "bg-emerald-100 text-emerald-950 dark:bg-white/10 dark:text-white"
                                  : "text-neutral-700 hover:bg-white/72 hover:text-emerald-950 dark:text-neutral-300 dark:hover:bg-white/6 dark:hover:text-white"
                              }`}
                              role="option"
                              aria-selected={isSelected}
                            >
                              <span>{option.label}</span>
                              <svg
                                className={`h-4 w-4 transition-opacity ${
                                  isSelected
                                    ? "opacity-100 text-emerald-500 dark:text-emerald-300"
                                    : "opacity-0"
                                }`}
                                viewBox="0 0 20 20"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth="2"
                              >
                                <path
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                  d="m5 10 3 3 7-7"
                                />
                              </svg>
                            </button>
                          );
                        })}
                      </div>
                    </div>

                    <ComposerIconButton
                      onClick={toggleSpeechInput}
                      disabled={!speechSupported}
                      title={isListening ? "Stop speech input" : "Start speech input"}
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
                          d="M12 14a3 3 0 0 0 3-3V7a3 3 0 1 0-6 0v4a3 3 0 0 0 3 3Z"
                        />
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          d="M19 11a7 7 0 0 1-14 0m7 7v3m-4 0h8"
                        />
                      </svg>
                    </ComposerIconButton>

                    <button
                      type="button"
                      onClick={() => void handleSendMessage()}
                      disabled={
                        isAnalyzing || (!input.trim() && selectedImages.length === 0)
                      }
                      className="flex h-11 w-11 items-center justify-center rounded-full bg-amber-400 text-neutral-950 shadow-[0_10px_28px_-16px_rgba(245,158,11,0.8)] transition-colors hover:bg-amber-300 disabled:cursor-not-allowed disabled:opacity-50"
                      title="Send message"
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
                          d="m5 12 14-7-4 7 4 7-14-7Z"
                        />
                      </svg>
                    </button>
                  </div>
                </div>
              </div>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              multiple
              onChange={handleImageUpload}
              disabled={isAnalyzing}
              className="hidden"
            />

            <input
              ref={cameraInputRef}
              type="file"
              accept="image/*"
              capture="environment"
              onChange={handleCameraCapture}
              disabled={isAnalyzing}
              className="hidden"
            />

            {cameraError ? (
              <p className="mt-3 text-center text-xs text-amber-600 dark:text-amber-300">
                {cameraError}
              </p>
            ) : null}

            {speechError ? (
              <p className="mt-2 text-center text-xs text-amber-600 dark:text-amber-300">
                {speechError}
              </p>
            ) : null}

            <p className="mt-3 text-center text-xs text-neutral-500 dark:text-neutral-400">
              AI may make mistakes. Recheck the response.
            </p>
          </div>
        </div>
      </section>

      {cameraDialogOpen ? (
        <div className="absolute inset-0 z-40 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm">
          <div className="w-full max-w-md rounded-[30px] border border-white/10 bg-neutral-950/95 p-4 text-white shadow-2xl">
            <div className="flex items-center justify-between gap-4">
              <div>
                <p className="text-sm font-semibold uppercase tracking-[0.18em] text-sky-300">
                  Camera
                </p>
                <p className="mt-1 text-sm text-neutral-300">
                  Allow camera access, frame the leaf clearly, then capture.
                </p>
              </div>
              <button
                type="button"
                onClick={closeCameraDialog}
                className="flex h-10 w-10 items-center justify-center rounded-2xl bg-white/6 text-white transition-colors hover:bg-white/12"
                aria-label="Close camera"
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

            <div className="mt-4 overflow-hidden rounded-[24px] border border-white/10 bg-black">
              <video
                ref={cameraVideoRef}
                autoPlay
                muted
                playsInline
                className="h-[360px] w-full object-cover"
              />
            </div>

            <div className="mt-4 flex items-center justify-end gap-3">
              <button
                type="button"
                onClick={openUploadLibrary}
                className="rounded-2xl border border-white/10 px-4 py-3 text-sm font-medium text-neutral-200 transition-colors hover:bg-white/6"
              >
                Choose photo
              </button>
              <button
                type="button"
                onClick={captureCameraPhoto}
                className="rounded-2xl bg-emerald-600 px-5 py-3 text-sm font-semibold text-white transition-colors hover:bg-emerald-700"
              >
                Capture photo
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
