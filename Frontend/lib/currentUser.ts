import { apiClient } from "@/lib/api";

export interface RecommendationProfile {
  farmSize?: number | null;
  cropType?: string | null;
  bio?: string | null;
  district?: string | null;
  mandalVillage?: string | null;
  soilType?: string | null;
  waterSource?: string | null;
  irrigationLevel?: string | null;
  seasonPreference?: string | null;
  cropPurpose?: string | null;
  previousCrop?: string | null;
  budget?: string | null;
  marketPreference?: string | null;
  riskPreference?: string | null;
  croppingPreference?: string | null;
}

export interface CurrentUserProfile {
  id: string;
  email: string;
  firstName: string;
  lastName: string;
  phone?: string | null;
  address?: string | null;
  city?: string | null;
  state?: string | null;
  pincode?: string | null;
  profile?: RecommendationProfile | null;
}

const parseJwtPayload = (token: string | null): Record<string, unknown> | null => {
  if (!token) return null;

  const parts = token.split(".");
  if (parts.length < 2) return null;

  try {
    const normalized = parts[1].replace(/-/g, "+").replace(/_/g, "/");
    const padded = normalized.padEnd(
      normalized.length + ((4 - (normalized.length % 4)) % 4),
      "=",
    );
    const decoded =
      typeof window !== "undefined"
        ? window.atob(padded)
        : Buffer.from(padded, "base64").toString("utf-8");
    return JSON.parse(decoded) as Record<string, unknown>;
  } catch {
    return null;
  }
};

export const getAuthenticatedUserId = (): string | null => {
  if (typeof window === "undefined") return null;
  const token = localStorage.getItem("token");
  const payload = parseJwtPayload(token);
  const userId = payload?.id;
  return typeof userId === "string" && userId.trim() ? userId : null;
};

export const buildProfileName = (user: CurrentUserProfile | null): string => {
  if (!user) return "";
  const fullName = `${user.firstName || ""} ${user.lastName || ""}`.trim();
  return fullName || user.firstName || user.email || "";
};

export const buildAdvisorProfileContext = (user: CurrentUserProfile | null) => {
  if (!user) return null;

  const profile = user.profile || {};
  const locationParts = [
    profile.mandalVillage,
    profile.district,
    user.city,
    user.state,
  ].filter(Boolean);

  return {
    user_id: user.id,
    full_name: buildProfileName(user),
    district: profile.district || "",
    mandal_village: profile.mandalVillage || "",
    soil_type: profile.soilType || "",
    water_source: profile.waterSource || "",
    irrigation_level: profile.irrigationLevel || "",
    season: profile.seasonPreference || "",
    crop_purpose: profile.cropPurpose || "",
    land_size: profile.farmSize ?? null,
    previous_crop: profile.previousCrop || "",
    budget: profile.budget || "",
    market_preference: profile.marketPreference || "",
    risk_preference: profile.riskPreference || "",
    cropping_preference: profile.croppingPreference || "",
    location_label: locationParts.join(", "),
    state: user.state || "",
  };
};

export const fetchCurrentUserProfile = async (): Promise<CurrentUserProfile | null> => {
  try {
    const response = await apiClient.get("/users/me");
    return (response.data || null) as CurrentUserProfile | null;
  } catch {
    const userId = getAuthenticatedUserId();
    if (!userId) return null;
    const response = await apiClient.get(`/users/${userId}`);
    return (response.data || null) as CurrentUserProfile | null;
  }
};
