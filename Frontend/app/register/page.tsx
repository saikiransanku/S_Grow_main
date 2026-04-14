"use client";

import { useState } from "react";
import axios from "axios";
import Link from "next/link";
import GoogleSignInButton from "@/components/GoogleSignInButton";

export default function Register() {
  const [formData, setFormData] = useState({
    email: "",
    password: "",
    firstName: "",
    lastName: "",
  });
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSuccess("");

    if (formData.password !== confirmPassword) {
      setError("Passwords do not match");
      return;
    }

    setLoading(true);

    try {
      const response = await axios.post(
        `${process.env.NEXT_PUBLIC_API_URL}/auth/register`,
        formData,
      );

      const token = response.data?.data?.token;
      if (!token) {
        throw new Error("Registration response missing token");
      }
      localStorage.setItem("token", token);
      setSuccess("Account created successfully! Redirecting...");
      setTimeout(() => {
        window.location.href = "/dashboard";
      }, 1500);
    } catch (err: any) {
      setError(
        err.response?.data?.message ||
          err.response?.data?.error ||
          err.message ||
          "Registration failed",
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-emerald-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 flex items-center justify-center py-12 px-4">
      <div className="w-full max-w-md">
        {/* Card */}
        <div className="card p-8 lg:p-10 space-y-8">
          {/* Header */}
          <div className="text-center space-y-3">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-2xl">
              <span className="text-3xl">🌱</span>
            </div>
            <h1 className="text-3xl font-bold text-transparent bg-gradient-to-r from-emerald-600 to-teal-600 bg-clip-text">
              Join SS Argitech
            </h1>
            <p className="text-gray-600 dark:text-gray-400">
              Create your account in seconds
            </p>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-5">
            {/* Name Fields */}
            <div className="grid grid-cols-2 gap-4">
              <div className="form-group">
                <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                  First Name
                </label>
                <input
                  type="text"
                  name="firstName"
                  placeholder="John"
                  value={formData.firstName}
                  onChange={handleChange}
                  className="input-field"
                  required
                />
              </div>
              <div className="form-group">
                <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                  Last Name
                </label>
                <input
                  type="text"
                  name="lastName"
                  placeholder="Doe"
                  value={formData.lastName}
                  onChange={handleChange}
                  className="input-field"
                  required
                />
              </div>
            </div>

            {/* Email Field */}
            <div className="form-group">
              <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                Email Address
              </label>
              <input
                type="email"
                name="email"
                placeholder="farmer@example.com"
                value={formData.email}
                onChange={handleChange}
                className="input-field"
                required
              />
            </div>

            {/* Password Field */}
            <div className="form-group">
              <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                Password
              </label>
              <div className="relative">
                <input
                  type={showPassword ? "text" : "password"}
                  name="password"
                  placeholder="Create a strong password (min 6 chars)"
                  value={formData.password}
                  onChange={handleChange}
                  className="input-field pr-12"
                  minLength={6}
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 transition-colors"
                  aria-label="Toggle password visibility"
                >
                  {showPassword ? (
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
                        d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21"
                      />
                    </svg>
                  ) : (
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
                        d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                      />
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                      />
                    </svg>
                  )}
                </button>
              </div>
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
                Min 6 characters, with numbers and symbols recommended
              </p>
            </div>

            {/* Confirm Password Field */}
            <div className="form-group">
              <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-2">
                Confirm Password
              </label>
              <div className="relative">
                <input
                  type={showConfirmPassword ? "text" : "password"}
                  name="confirmPassword"
                  placeholder="Re-enter your password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className="input-field pr-12"
                  minLength={6}
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 transition-colors"
                  aria-label="Toggle confirm password visibility"
                >
                  {showConfirmPassword ? (
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
                        d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21"
                      />
                    </svg>
                  ) : (
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
                        d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                      />
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                      />
                    </svg>
                  )}
                </button>
              </div>
            </div>

            {/* Error Message */}
            {error && (
              <div className="bg-red-50 dark:bg-red-900/20 border-l-4 border-red-500 dark:border-red-600 p-4 rounded-lg">
                <p className="text-red-700 dark:text-red-400 text-sm font-medium">
                  {error}
                </p>
              </div>
            )}

            {/* Success Message */}
            {success && (
              <div className="bg-emerald-50 dark:bg-emerald-900/20 border-l-4 border-emerald-500 dark:border-emerald-600 p-4 rounded-lg">
                <p className="text-emerald-700 dark:text-emerald-400 text-sm font-medium">
                  {success}
                </p>
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading}
              className="btn-primary w-full"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></span>
                  Creating account...
                </span>
              ) : (
                "Create Account"
              )}
            </button>

            {/* Terms */}
            <p className="text-center text-xs text-gray-600 dark:text-gray-400">
              By registering, you agree to our{" "}
              <a
                href="#"
                className="text-emerald-600 dark:text-emerald-400 hover:text-emerald-700 dark:hover:text-emerald-300 font-semibold"
              >
                Terms of Service
              </a>{" "}
              and{" "}
              <a
                href="#"
                className="text-emerald-600 dark:text-emerald-400 hover:text-emerald-700 dark:hover:text-emerald-300 font-semibold"
              >
                Privacy Policy
              </a>
            </p>
          </form>

          {/* Divider */}
          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-gray-200 dark:border-gray-600"></div>
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="px-2 bg-white dark:bg-slate-700 text-gray-500 dark:text-gray-400">
                Already farming with us?
              </span>
            </div>
          </div>

          {/* Social Buttons */}
          <div className="grid grid-cols-2 gap-3">
            <GoogleSignInButton
              onError={setError}
              redirectTo="/dashboard"
              text="signup_with"
            />
            <button className="flex items-center justify-center gap-2 px-4 py-2 border-2 border-gray-200 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-slate-600 transition-colors">
              <span>
                <img src="/smartphone.svg" alt="phone" className="w-5 h-5" />
              </span>
              <span className="text-sm font-medium">Phone</span>
            </button>
          </div>

          {/* Footer */}
          <div className="text-center pt-4 border-t border-gray-100 dark:border-gray-600">
            <p className="text-gray-600 dark:text-gray-400 text-sm">
              Already have an account?{" "}
              <Link
                href="/login"
                className="text-emerald-600 dark:text-emerald-400 hover:text-emerald-700 dark:hover:text-emerald-300 font-bold transition"
              >
                Sign in here
              </Link>
            </p>
          </div>
        </div>

        {/* Features */}
        <div className="mt-8 grid grid-cols-3 gap-4 text-center text-sm text-gray-600 dark:text-gray-400">
          <div className="text-2xl mb-1">📚</div>
          <div className="text-2xl mb-1">🔒</div>
          <div className="text-2xl mb-1">📱</div>
          <div>Laws</div>
          <div>Secure</div>
          <div>Mobile</div>
        </div>
      </div>
    </main>
  );
}
