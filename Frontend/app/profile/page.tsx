"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { apiClient } from "@/lib/api";
import {
  CurrentUserProfile,
  fetchCurrentUserProfile,
} from "@/lib/currentUser";

type ProfileFormState = {
  firstName: string;
  lastName: string;
  email: string;
  phone: string;
  address: string;
  city: string;
  state: string;
  pincode: string;
  profile: {
    farmSize: string;
    district: string;
    mandalVillage: string;
    soilType: string;
    waterSource: string;
    irrigationLevel: string;
    seasonPreference: string;
    cropPurpose: string;
    previousCrop: string;
    budget: string;
    marketPreference: string;
    riskPreference: string;
    croppingPreference: string;
    cropType: string;
    bio: string;
  };
};

const SOIL_TYPES = [
  "Unknown",
  "Black soil",
  "Red soil",
  "Sandy soil",
  "Clay soil",
  "Loamy soil",
  "Mixed soil",
];

const WATER_SOURCES = [
  "Unknown",
  "Rainfed",
  "Borewell",
  "Canal",
  "Tank",
  "Drip irrigation",
  "Sprinkler irrigation",
];

const IRRIGATION_LEVELS = ["Unknown", "Low", "Medium", "High"];
const SEASONS = ["Unknown", "Kharif", "Rabi", "Summer"];
const CROP_PURPOSES = [
  "Unknown",
  "Food crop",
  "Cash crop",
  "Fodder",
  "Short-duration crop",
  "Long-duration crop",
  "Multi-cropping",
  "Intercropping",
];
const BUDGET_LEVELS = ["Unknown", "Low", "Medium", "High"];
const RISK_LEVELS = ["Unknown", "Low", "Medium", "High"];
const CROPPING_PREFERENCES = [
  "Unknown",
  "Single crop",
  "Multi-cropping",
  "Intercropping",
];

const createEmptyForm = (): ProfileFormState => ({
  firstName: "",
  lastName: "",
  email: "",
  phone: "",
  address: "",
  city: "",
  state: "Telangana",
  pincode: "",
  profile: {
    farmSize: "",
    district: "",
    mandalVillage: "",
    soilType: "Unknown",
    waterSource: "Unknown",
    irrigationLevel: "Unknown",
    seasonPreference: "Unknown",
    cropPurpose: "Unknown",
    previousCrop: "",
    budget: "Unknown",
    marketPreference: "",
    riskPreference: "Unknown",
    croppingPreference: "Unknown",
    cropType: "",
    bio: "",
  },
});

const toFormState = (user: CurrentUserProfile | null): ProfileFormState => {
  const empty = createEmptyForm();
  if (!user) return empty;

  return {
    firstName: user.firstName || "",
    lastName: user.lastName || "",
    email: user.email || "",
    phone: user.phone || "",
    address: user.address || "",
    city: user.city || "",
    state: user.state || "Telangana",
    pincode: user.pincode || "",
    profile: {
      farmSize:
        user.profile?.farmSize !== null &&
        user.profile?.farmSize !== undefined
          ? String(user.profile.farmSize)
          : "",
      district: user.profile?.district || "",
      mandalVillage: user.profile?.mandalVillage || "",
      soilType: user.profile?.soilType || "Unknown",
      waterSource: user.profile?.waterSource || "Unknown",
      irrigationLevel: user.profile?.irrigationLevel || "Unknown",
      seasonPreference: user.profile?.seasonPreference || "Unknown",
      cropPurpose: user.profile?.cropPurpose || "Unknown",
      previousCrop: user.profile?.previousCrop || "",
      budget: user.profile?.budget || "Unknown",
      marketPreference: user.profile?.marketPreference || "",
      riskPreference: user.profile?.riskPreference || "Unknown",
      croppingPreference: user.profile?.croppingPreference || "Unknown",
      cropType: user.profile?.cropType || "",
      bio: user.profile?.bio || "",
    },
  };
};

const formatLabelValue = (value: string | null | undefined, fallback = "Not set") =>
  value && value.trim() ? value : fallback;

export default function ProfilePage() {
  const [user, setUser] = useState<CurrentUserProfile | null>(null);
  const [formData, setFormData] = useState<ProfileFormState>(createEmptyForm);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [success, setSuccess] = useState("");
  const [error, setError] = useState("");

  const loadProfile = async () => {
    try {
      const currentUser = await fetchCurrentUserProfile();
      setUser(currentUser);
      setFormData(toFormState(currentUser));
      setError("");
    } catch (loadError) {
      console.error("Failed to fetch profile", loadError);
      setError("Failed to load profile.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadProfile();
  }, []);

  const handleChange = (
    event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>,
  ) => {
    const { name, value } = event.target;
    setFormData((current) => ({
      ...current,
      [name]: value,
    }));
  };

  const handleProfileChange = (
    event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>,
  ) => {
    const { name, value } = event.target;
    setFormData((current) => ({
      ...current,
      profile: {
        ...current.profile,
        [name]: value,
      },
    }));
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    setSuccess("");
    setError("");

    try {
      const payload = {
        firstName: formData.firstName.trim(),
        lastName: formData.lastName.trim(),
        phone: formData.phone.trim(),
        address: formData.address.trim(),
        city: formData.city.trim(),
        state: formData.state.trim(),
        pincode: formData.pincode.trim(),
        profile: {
          farmSize: formData.profile.farmSize.trim()
            ? Number.parseFloat(formData.profile.farmSize)
            : null,
          district: formData.profile.district.trim(),
          mandalVillage: formData.profile.mandalVillage.trim(),
          soilType: formData.profile.soilType,
          waterSource: formData.profile.waterSource,
          irrigationLevel: formData.profile.irrigationLevel,
          seasonPreference: formData.profile.seasonPreference,
          cropPurpose: formData.profile.cropPurpose,
          previousCrop: formData.profile.previousCrop.trim(),
          budget: formData.profile.budget,
          marketPreference: formData.profile.marketPreference.trim(),
          riskPreference: formData.profile.riskPreference,
          croppingPreference: formData.profile.croppingPreference,
          cropType: formData.profile.cropType.trim(),
          bio: formData.profile.bio.trim(),
        },
      };

      const response = await apiClient.put("/users/me", payload);
      const nextUser = response.data as CurrentUserProfile;
      setUser(nextUser);
      setFormData(toFormState(nextUser));
      setIsEditing(false);
      setSuccess("Profile updated successfully.");
      window.setTimeout(() => setSuccess(""), 3000);
    } catch (saveError) {
      console.error("Failed to update profile", saveError);
      setError("Failed to update profile. Please try again.");
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-slate-50 via-white to-emerald-50">
        <div className="space-y-4 text-center">
          <div className="mx-auto h-16 w-16 animate-spin rounded-full border-4 border-emerald-200 border-t-emerald-600" />
          <p className="text-lg font-semibold text-emerald-700">Loading profile...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-emerald-50 px-4 py-12">
      <div className="mx-auto max-w-5xl">
        <div className="mb-8">
          <Link
            href="/dashboard"
            className="mb-6 inline-flex items-center gap-2 font-semibold text-emerald-600 hover:text-emerald-700"
          >
            Back to Dashboard
          </Link>
          <h1 className="text-4xl font-bold text-gray-900">My Profile</h1>
          <p className="mt-2 text-gray-600">
            Save your farm details here so the crop recommendation engine can use
            them directly before asking for extra inputs.
          </p>
        </div>

        {success ? (
          <div className="mb-6 rounded-lg border-l-4 border-emerald-500 bg-emerald-50 p-4">
            <p className="font-medium text-emerald-700">{success}</p>
          </div>
        ) : null}

        {error ? (
          <div className="mb-6 rounded-lg border-l-4 border-red-500 bg-red-50 p-4">
            <p className="font-medium text-red-700">{error}</p>
          </div>
        ) : null}

        <div className="overflow-hidden rounded-[28px] border border-emerald-100 bg-white shadow-sm">
          <div className="flex flex-col gap-4 border-b border-gray-100 bg-gradient-to-r from-emerald-50 to-teal-50 px-6 py-6 md:flex-row md:items-center md:justify-between">
            <div>
              <h2 className="text-2xl font-bold text-gray-900">
                {isEditing ? "Edit Profile" : "Farmer Profile"}
              </h2>
              <p className="mt-1 text-sm text-gray-600">
                Keep this updated for better Telangana crop planning.
              </p>
            </div>
            <button
              type="button"
              onClick={() => {
                setIsEditing((current) => !current);
                setFormData(toFormState(user));
                setError("");
              }}
              className={
                isEditing ? "btn-secondary btn-small" : "btn-primary btn-small"
              }
            >
              {isEditing ? "Cancel" : "Edit Profile"}
            </button>
          </div>

          <div className="p-6 lg:p-8">
            {isEditing ? (
              <form onSubmit={handleSubmit} className="space-y-10">
                <section className="space-y-6">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">
                      Personal Details
                    </h3>
                    <p className="mt-1 text-sm text-gray-600">
                      These fields help us identify your account and location.
                    </p>
                  </div>

                  <div className="grid gap-6 md:grid-cols-2">
                    <label className="form-group">
                      <span className="mb-2 block text-sm font-semibold text-gray-700">
                        First Name
                      </span>
                      <input
                        type="text"
                        name="firstName"
                        value={formData.firstName}
                        onChange={handleChange}
                        className="input-field"
                        required
                      />
                    </label>

                    <label className="form-group">
                      <span className="mb-2 block text-sm font-semibold text-gray-700">
                        Last Name
                      </span>
                      <input
                        type="text"
                        name="lastName"
                        value={formData.lastName}
                        onChange={handleChange}
                        className="input-field"
                        required
                      />
                    </label>
                  </div>

                  <div className="grid gap-6 md:grid-cols-2">
                    <label className="form-group">
                      <span className="mb-2 block text-sm font-semibold text-gray-700">
                        Email Address
                      </span>
                      <input
                        type="email"
                        value={formData.email}
                        disabled
                        className="input-field cursor-not-allowed bg-gray-100 opacity-75"
                      />
                    </label>

                    <label className="form-group">
                      <span className="mb-2 block text-sm font-semibold text-gray-700">
                        Phone Number
                      </span>
                      <input
                        type="tel"
                        name="phone"
                        value={formData.phone}
                        onChange={handleChange}
                        placeholder="+91 98765 43210"
                        className="input-field"
                      />
                    </label>
                  </div>

                  <div className="grid gap-6 md:grid-cols-2">
                    <label className="form-group">
                      <span className="mb-2 block text-sm font-semibold text-gray-700">
                        Address
                      </span>
                      <input
                        type="text"
                        name="address"
                        value={formData.address}
                        onChange={handleChange}
                        placeholder="Village or street address"
                        className="input-field"
                      />
                    </label>

                    <label className="form-group">
                      <span className="mb-2 block text-sm font-semibold text-gray-700">
                        City or Town
                      </span>
                      <input
                        type="text"
                        name="city"
                        value={formData.city}
                        onChange={handleChange}
                        placeholder="Town or city"
                        className="input-field"
                      />
                    </label>
                  </div>

                  <div className="grid gap-6 md:grid-cols-2">
                    <label className="form-group">
                      <span className="mb-2 block text-sm font-semibold text-gray-700">
                        State
                      </span>
                      <input
                        type="text"
                        name="state"
                        value={formData.state}
                        onChange={handleChange}
                        placeholder="Telangana"
                        className="input-field"
                      />
                    </label>

                    <label className="form-group">
                      <span className="mb-2 block text-sm font-semibold text-gray-700">
                        Postal Code
                      </span>
                      <input
                        type="text"
                        name="pincode"
                        value={formData.pincode}
                        onChange={handleChange}
                        placeholder="500001"
                        className="input-field"
                      />
                    </label>
                  </div>
                </section>

                <section className="space-y-6 border-t border-gray-100 pt-8">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">
                      Crop Recommendation Profile
                    </h3>
                    <p className="mt-1 text-sm text-gray-600">
                      These are the fields the agriculture recommendation engine
                      will read first.
                    </p>
                  </div>

                  <div className="grid gap-6 md:grid-cols-3">
                    <label className="form-group">
                      <span className="mb-2 block text-sm font-semibold text-gray-700">
                        District
                      </span>
                      <input
                        type="text"
                        name="district"
                        value={formData.profile.district}
                        onChange={handleProfileChange}
                        placeholder="Warangal"
                        className="input-field"
                      />
                    </label>

                    <label className="form-group">
                      <span className="mb-2 block text-sm font-semibold text-gray-700">
                        Mandal or Village
                      </span>
                      <input
                        type="text"
                        name="mandalVillage"
                        value={formData.profile.mandalVillage}
                        onChange={handleProfileChange}
                        placeholder="Parvathagiri"
                        className="input-field"
                      />
                    </label>

                    <label className="form-group">
                      <span className="mb-2 block text-sm font-semibold text-gray-700">
                        Land Size (acres)
                      </span>
                      <input
                        type="number"
                        min="0"
                        step="0.1"
                        name="farmSize"
                        value={formData.profile.farmSize}
                        onChange={handleProfileChange}
                        placeholder="2.5"
                        className="input-field"
                      />
                    </label>
                  </div>

                  <div className="grid gap-6 md:grid-cols-3">
                    <label className="form-group">
                      <span className="mb-2 block text-sm font-semibold text-gray-700">
                        Soil Type
                      </span>
                      <select
                        name="soilType"
                        value={formData.profile.soilType}
                        onChange={handleProfileChange}
                        className="input-field"
                      >
                        {SOIL_TYPES.map((option) => (
                          <option key={option} value={option}>
                            {option}
                          </option>
                        ))}
                      </select>
                    </label>

                    <label className="form-group">
                      <span className="mb-2 block text-sm font-semibold text-gray-700">
                        Water Source
                      </span>
                      <select
                        name="waterSource"
                        value={formData.profile.waterSource}
                        onChange={handleProfileChange}
                        className="input-field"
                      >
                        {WATER_SOURCES.map((option) => (
                          <option key={option} value={option}>
                            {option}
                          </option>
                        ))}
                      </select>
                    </label>

                    <label className="form-group">
                      <span className="mb-2 block text-sm font-semibold text-gray-700">
                        Irrigation Level
                      </span>
                      <select
                        name="irrigationLevel"
                        value={formData.profile.irrigationLevel}
                        onChange={handleProfileChange}
                        className="input-field"
                      >
                        {IRRIGATION_LEVELS.map((option) => (
                          <option key={option} value={option}>
                            {option}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>

                  <div className="grid gap-6 md:grid-cols-3">
                    <label className="form-group">
                      <span className="mb-2 block text-sm font-semibold text-gray-700">
                        Preferred Season
                      </span>
                      <select
                        name="seasonPreference"
                        value={formData.profile.seasonPreference}
                        onChange={handleProfileChange}
                        className="input-field"
                      >
                        {SEASONS.map((option) => (
                          <option key={option} value={option}>
                            {option}
                          </option>
                        ))}
                      </select>
                    </label>

                    <label className="form-group">
                      <span className="mb-2 block text-sm font-semibold text-gray-700">
                        Crop Purpose
                      </span>
                      <select
                        name="cropPurpose"
                        value={formData.profile.cropPurpose}
                        onChange={handleProfileChange}
                        className="input-field"
                      >
                        {CROP_PURPOSES.map((option) => (
                          <option key={option} value={option}>
                            {option}
                          </option>
                        ))}
                      </select>
                    </label>

                    <label className="form-group">
                      <span className="mb-2 block text-sm font-semibold text-gray-700">
                        Single or Multi-Cropping
                      </span>
                      <select
                        name="croppingPreference"
                        value={formData.profile.croppingPreference}
                        onChange={handleProfileChange}
                        className="input-field"
                      >
                        {CROPPING_PREFERENCES.map((option) => (
                          <option key={option} value={option}>
                            {option}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>

                  <div className="grid gap-6 md:grid-cols-3">
                    <label className="form-group">
                      <span className="mb-2 block text-sm font-semibold text-gray-700">
                        Previous Crop
                      </span>
                      <input
                        type="text"
                        name="previousCrop"
                        value={formData.profile.previousCrop}
                        onChange={handleProfileChange}
                        placeholder="Cotton"
                        className="input-field"
                      />
                    </label>

                    <label className="form-group">
                      <span className="mb-2 block text-sm font-semibold text-gray-700">
                        Budget
                      </span>
                      <select
                        name="budget"
                        value={formData.profile.budget}
                        onChange={handleProfileChange}
                        className="input-field"
                      >
                        {BUDGET_LEVELS.map((option) => (
                          <option key={option} value={option}>
                            {option}
                          </option>
                        ))}
                      </select>
                    </label>

                    <label className="form-group">
                      <span className="mb-2 block text-sm font-semibold text-gray-700">
                        Risk Preference
                      </span>
                      <select
                        name="riskPreference"
                        value={formData.profile.riskPreference}
                        onChange={handleProfileChange}
                        className="input-field"
                      >
                        {RISK_LEVELS.map((option) => (
                          <option key={option} value={option}>
                            {option}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>

                  <div className="grid gap-6 md:grid-cols-2">
                    <label className="form-group">
                      <span className="mb-2 block text-sm font-semibold text-gray-700">
                        Market Preference
                      </span>
                      <input
                        type="text"
                        name="marketPreference"
                        value={formData.profile.marketPreference}
                        onChange={handleProfileChange}
                        placeholder="Local mandi, millers, vegetables market"
                        className="input-field"
                      />
                    </label>

                    <label className="form-group">
                      <span className="mb-2 block text-sm font-semibold text-gray-700">
                        Main Crop Type
                      </span>
                      <input
                        type="text"
                        name="cropType"
                        value={formData.profile.cropType}
                        onChange={handleProfileChange}
                        placeholder="Cotton, paddy, maize"
                        className="input-field"
                      />
                    </label>
                  </div>

                  <div className="grid gap-6 md:grid-cols-2">
                    <label className="form-group">
                      <span className="mb-2 block text-sm font-semibold text-gray-700">
                        Farm Notes
                      </span>
                      <textarea
                        name="bio"
                        value={formData.profile.bio}
                        onChange={handleProfileChange}
                        rows={4}
                        placeholder="Anything important about the land, labour, or irrigation"
                        className="input-field min-h-[120px]"
                      />
                    </label>

                    <div className="rounded-2xl border border-emerald-100 bg-emerald-50/70 p-5">
                      <h4 className="text-base font-semibold text-emerald-900">
                        Used by the AI advisor
                      </h4>
                      <p className="mt-3 text-sm leading-7 text-emerald-800">
                        The AI will first read these saved land details and ask
                        whether you want to use the same land before giving crop
                        recommendations or intercropping plans.
                      </p>
                    </div>
                  </div>
                </section>

                <div className="flex flex-col gap-3 border-t border-gray-100 pt-6 md:flex-row">
                  <button
                    type="submit"
                    disabled={saving}
                    className="btn-primary flex-1"
                  >
                    {saving ? "Saving..." : "Save Changes"}
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setIsEditing(false);
                      setFormData(toFormState(user));
                      setError("");
                    }}
                    className="btn-secondary"
                  >
                    Cancel
                  </button>
                </div>
              </form>
            ) : (
              <div className="space-y-8">
                <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">
                  <div className="rounded-2xl border border-emerald-100 bg-gradient-to-br from-emerald-50 to-teal-50 p-5">
                    <p className="text-sm font-medium text-gray-600">Farmer</p>
                    <p className="mt-2 text-2xl font-bold text-gray-900">
                      {formatLabelValue(
                        `${user?.firstName || ""} ${user?.lastName || ""}`.trim(),
                      )}
                    </p>
                  </div>

                  <div className="rounded-2xl border border-teal-100 bg-gradient-to-br from-teal-50 to-cyan-50 p-5">
                    <p className="text-sm font-medium text-gray-600">Location</p>
                    <p className="mt-2 text-lg font-semibold text-gray-900">
                      {formatLabelValue(
                        [user?.profile?.mandalVillage, user?.profile?.district, user?.state]
                          .filter(Boolean)
                          .join(", "),
                      )}
                    </p>
                  </div>

                  <div className="rounded-2xl border border-amber-100 bg-gradient-to-br from-amber-50 to-orange-50 p-5">
                    <p className="text-sm font-medium text-gray-600">Land Size</p>
                    <p className="mt-2 text-lg font-semibold text-gray-900">
                      {user?.profile?.farmSize ? `${user.profile.farmSize} acres` : "Not set"}
                    </p>
                  </div>
                </div>

                <div className="grid gap-6 md:grid-cols-2">
                  <div className="rounded-2xl border border-gray-100 bg-gray-50 p-6">
                    <h3 className="text-lg font-semibold text-gray-900">
                      Contact and Address
                    </h3>
                    <div className="mt-4 space-y-3 text-sm leading-7 text-gray-700">
                      <p>
                        <span className="font-semibold text-gray-900">Email:</span>{" "}
                        {formatLabelValue(user?.email)}
                      </p>
                      <p>
                        <span className="font-semibold text-gray-900">Phone:</span>{" "}
                        {formatLabelValue(user?.phone)}
                      </p>
                      <p>
                        <span className="font-semibold text-gray-900">Address:</span>{" "}
                        {formatLabelValue(user?.address)}
                      </p>
                      <p>
                        <span className="font-semibold text-gray-900">City:</span>{" "}
                        {formatLabelValue(user?.city)}
                      </p>
                      <p>
                        <span className="font-semibold text-gray-900">State:</span>{" "}
                        {formatLabelValue(user?.state)}
                      </p>
                    </div>
                  </div>

                  <div className="rounded-2xl border border-emerald-100 bg-emerald-50/70 p-6">
                    <h3 className="text-lg font-semibold text-gray-900">
                      Recommendation Inputs
                    </h3>
                    <div className="mt-4 grid gap-3 text-sm leading-7 text-gray-700 md:grid-cols-2">
                      <p>
                        <span className="font-semibold text-gray-900">Soil:</span>{" "}
                        {formatLabelValue(user?.profile?.soilType)}
                      </p>
                      <p>
                        <span className="font-semibold text-gray-900">Water:</span>{" "}
                        {formatLabelValue(user?.profile?.waterSource)}
                      </p>
                      <p>
                        <span className="font-semibold text-gray-900">
                          Irrigation:
                        </span>{" "}
                        {formatLabelValue(user?.profile?.irrigationLevel)}
                      </p>
                      <p>
                        <span className="font-semibold text-gray-900">Season:</span>{" "}
                        {formatLabelValue(user?.profile?.seasonPreference)}
                      </p>
                      <p>
                        <span className="font-semibold text-gray-900">Goal:</span>{" "}
                        {formatLabelValue(user?.profile?.cropPurpose)}
                      </p>
                      <p>
                        <span className="font-semibold text-gray-900">
                          Previous crop:
                        </span>{" "}
                        {formatLabelValue(user?.profile?.previousCrop)}
                      </p>
                      <p>
                        <span className="font-semibold text-gray-900">Budget:</span>{" "}
                        {formatLabelValue(user?.profile?.budget)}
                      </p>
                      <p>
                        <span className="font-semibold text-gray-900">Risk:</span>{" "}
                        {formatLabelValue(user?.profile?.riskPreference)}
                      </p>
                      <p className="md:col-span-2">
                        <span className="font-semibold text-gray-900">
                          Cropping preference:
                        </span>{" "}
                        {formatLabelValue(user?.profile?.croppingPreference)}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="mt-8 rounded-2xl border border-blue-100 bg-gradient-to-r from-blue-50 to-cyan-50 p-6">
          <p className="mb-2 font-semibold text-gray-900">Need help?</p>
          <p className="text-gray-700">
            Keep your district, soil, water source, season, and previous crop
            updated here. That helps the recommendation engine avoid generic
            answers and ask fewer follow-up questions.
          </p>
        </div>
      </div>
    </div>
  );
}
