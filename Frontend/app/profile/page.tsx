// Profile page component
"use client";

import { useState, useEffect } from "react";
import { apiClient } from "@/lib/api";
import Link from "next/link";

interface UserProfile {
  id: string;
  email: string;
  firstName: string;
  lastName: string;
  phone?: string;
  address?: string;
  city?: string;
  state?: string;
  pincode?: string;
  profile?: {
    farmSize?: number;
    cropType?: string;
    bio?: string;
  };
}

export default function Profile() {
  const [user, setUser] = useState<UserProfile | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [success, setSuccess] = useState("");
  const [error, setError] = useState("");
  const [formData, setFormData] = useState<Partial<UserProfile>>({});

  useEffect(() => {
    fetchProfile();
  }, []);

  const fetchProfile = async () => {
    try {
      const response = await apiClient.get("/users");
      if (Array.isArray(response.data) && response.data.length > 0) {
        setUser(response.data[0]);
        setFormData(response.data[0]);
      }
    } catch (error) {
      console.error("Failed to fetch profile", error);
      setError("Failed to load profile");
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>,
  ) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setError("");
    setSuccess("");

    try {
      const response = await apiClient.put(`/users/${user?.id}`, formData);
      setUser(response.data);
      setFormData(response.data);
      setIsEditing(false);
      setSuccess("Profile updated successfully!");
      setTimeout(() => setSuccess(""), 3000);
    } catch (error) {
      console.error("Failed to update profile", error);
      setError("Failed to update profile. Please try again.");
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-emerald-50 flex items-center justify-center">
        <div className="text-center space-y-4">
          <div className="w-16 h-16 border-4 border-emerald-200 border-t-emerald-600 rounded-full animate-spin mx-auto"></div>
          <p className="text-lg font-semibold text-emerald-700">
            Loading profile...
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-emerald-50 py-12 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <Link
            href="/dashboard"
            className="text-emerald-600 hover:text-emerald-700 font-semibold flex items-center gap-2 mb-6"
          >
            ← Back to Dashboard
          </Link>
          <h1 className="text-4xl font-bold text-gray-900">My Profile</h1>
          <p className="text-gray-600 mt-2">
            Manage your farm information and personal details
          </p>
        </div>

        {/* Success Message */}
        {success && (
          <div className="mb-6 bg-emerald-50 border-l-4 border-emerald-500 p-4 rounded-lg animate-in fade-in">
            <p className="text-emerald-700 font-medium">✓ {success}</p>
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="mb-6 bg-red-50 border-l-4 border-red-500 p-4 rounded-lg">
            <p className="text-red-700 font-medium">✗ {error}</p>
          </div>
        )}

        {/* Profile Card */}
        <div className="card overflow-hidden">
          {/* Header with Actions */}
          <div className="px-6 lg:px-8 py-6 bg-gradient-to-r from-emerald-50 to-teal-50 border-b border-gray-100 flex justify-between items-center">
            <div>
              <h2 className="text-2xl font-bold text-gray-900">
                {isEditing ? "Edit Profile" : "Your Information"}
              </h2>
            </div>
            <button
              onClick={() => {
                setIsEditing(!isEditing);
                if (isEditing) setFormData(user || {});
                setError("");
              }}
              className={
                isEditing ? "btn-secondary btn-small" : "btn-primary btn-small"
              }
            >
              {isEditing ? "Cancel" : "Edit Profile"}
            </button>
          </div>

          {/* Content */}
          <div className="p-6 lg:p-8">
            {isEditing ? (
              // Edit Form
              <form onSubmit={handleSubmit} className="space-y-6">
                {/* Name Row */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="form-group">
                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                      First Name
                    </label>
                    <input
                      type="text"
                      name="firstName"
                      value={formData.firstName || ""}
                      onChange={handleChange}
                      placeholder="John"
                      className="input-field"
                      required
                    />
                  </div>
                  <div className="form-group">
                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                      Last Name
                    </label>
                    <input
                      type="text"
                      name="lastName"
                      value={formData.lastName || ""}
                      onChange={handleChange}
                      placeholder="Doe"
                      className="input-field"
                      required
                    />
                  </div>
                </div>

                {/* Email (Disabled) */}
                <div className="form-group">
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    Email Address
                  </label>
                  <input
                    type="email"
                    value={formData.email || ""}
                    disabled
                    className="input-field bg-gray-100 cursor-not-allowed opacity-75"
                  />
                  <p className="text-xs text-gray-500 mt-2">
                    Email cannot be changed
                  </p>
                </div>

                {/* Contact Info */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="form-group">
                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                      Phone Number
                    </label>
                    <input
                      type="tel"
                      name="phone"
                      value={formData.phone || ""}
                      onChange={handleChange}
                      placeholder="+91 98765 43210"
                      className="input-field"
                    />
                  </div>
                  <div className="form-group">
                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                      Address
                    </label>
                    <input
                      type="text"
                      name="address"
                      value={formData.address || ""}
                      onChange={handleChange}
                      placeholder="Farmer lane, Village"
                      className="input-field"
                    />
                  </div>
                </div>

                {/* Location */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="form-group">
                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                      City
                    </label>
                    <input
                      type="text"
                      name="city"
                      value={formData.city || ""}
                      onChange={handleChange}
                      placeholder="Mumbai"
                      className="input-field"
                    />
                  </div>
                  <div className="form-group">
                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                      State
                    </label>
                    <input
                      type="text"
                      name="state"
                      value={formData.state || ""}
                      onChange={handleChange}
                      placeholder="Maharashtra"
                      className="input-field"
                    />
                  </div>
                  <div className="form-group">
                    <label className="block text-sm font-semibold text-gray-700 mb-2">
                      Postal Code
                    </label>
                    <input
                      type="text"
                      name="pincode"
                      value={formData.pincode || ""}
                      onChange={handleChange}
                      placeholder="400001"
                      className="input-field"
                    />
                  </div>
                </div>

                {/* Submit Buttons */}
                <div className="flex gap-4 pt-4">
                  <button
                    type="submit"
                    disabled={saving}
                    className="btn-primary flex-1"
                  >
                    {saving ? (
                      <span className="flex items-center justify-center gap-2">
                        <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
                        Saving...
                      </span>
                    ) : (
                      "Save Changes"
                    )}
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setIsEditing(false);
                      setFormData(user || {});
                      setError("");
                    }}
                    className="btn-secondary"
                  >
                    Cancel
                  </button>
                </div>
              </form>
            ) : (
              // View Mode
              <div className="space-y-6">
                {/* Personal Info */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                  {/* Name */}
                  <div className="p-4 bg-gradient-to-br from-emerald-50 to-teal-50 rounded-lg border border-emerald-100">
                    <p className="text-sm font-medium text-gray-600 mb-1">
                      Full Name
                    </p>
                    <p className="text-2xl font-bold text-gray-900">
                      {user?.firstName} {user?.lastName}
                    </p>
                  </div>

                  {/* Email */}
                  <div className="p-4 bg-gradient-to-br from-teal-50 to-cyan-50 rounded-lg border border-teal-100">
                    <p className="text-sm font-medium text-gray-600 mb-1">
                      Email Address
                    </p>
                    <p className="text-lg font-semibold text-gray-900">
                      {user?.email}
                    </p>
                  </div>

                  {/* Phone */}
                  {user?.phone && (
                    <div className="p-4 bg-gradient-to-br from-cyan-50 to-blue-50 rounded-lg border border-cyan-100">
                      <p className="text-sm font-medium text-gray-600 mb-1">
                        Phone Number
                      </p>
                      <p className="text-lg font-semibold text-gray-900">
                        {user.phone}
                      </p>
                    </div>
                  )}

                  {/* Address */}
                  {user?.address && (
                    <div className="p-4 bg-gradient-to-br from-emerald-50 to-green-50 rounded-lg border border-emerald-100">
                      <p className="text-sm font-medium text-gray-600 mb-1">
                        Address
                      </p>
                      <p className="text-lg font-semibold text-gray-900">
                        {user.address}
                      </p>
                    </div>
                  )}

                  {/* Location */}
                  {(user?.city || user?.state || user?.pincode) && (
                    <div className="p-4 bg-gradient-to-br from-teal-50 to-cyan-50 rounded-lg border border-teal-100 md:col-span-2">
                      <p className="text-sm font-medium text-gray-600 mb-2">
                        Location
                      </p>
                      <p className="text-lg font-semibold text-gray-900">
                        {[user?.city, user?.state, user?.pincode]
                          .filter(Boolean)
                          .join(", ")}
                      </p>
                    </div>
                  )}
                </div>

                {/* Account Status */}
                <div className="border-t border-gray-100 pt-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-gray-600">
                        Account Status
                      </p>
                      <p className="text-lg font-semibold text-gray-900 mt-1">
                        Active
                      </p>
                    </div>
                    <div className="badge badge-success">✓ Verified</div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Help Section */}
        <div className="mt-8 card p-6 bg-gradient-to-r from-blue-50 to-cyan-50 border-l-4 border-blue-500">
          <p className="font-semibold text-gray-900 mb-2">💡 Need help?</p>
          <p className="text-gray-700">
            For account security, email address cannot be changed. Contact
            support if you need to update it.
          </p>
        </div>
      </div>
    </div>
  );
}
