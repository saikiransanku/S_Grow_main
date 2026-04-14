"use client";

import { useState, useEffect } from "react";
import axios from "axios";
import Link from "next/link";

export default function Dashboard() {
  const [user, setUser] = useState<any>(null);
  const [laws, setLaws] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const token = localStorage.getItem("token");
      if (!token) {
        window.location.href = "/login";
        return;
      }

      const apiUrl = process.env.NEXT_PUBLIC_API_URL;
      const headers = { Authorization: `Bearer ${token}` };

      const lawsResponse = await axios.get(`${apiUrl}/laws`, { headers });
      setLaws(lawsResponse.data);
    } catch (error) {
      console.error("Failed to fetch data", error);
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    window.location.href = "/login";
  };

  const filteredLaws = laws.filter(
    (law) =>
      law.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      law.description.toLowerCase().includes(searchTerm.toLowerCase()),
  );

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-emerald-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 flex items-center justify-center">
        <div className="text-center space-y-4">
          <div className="w-16 h-16 border-4 border-emerald-200 dark:border-emerald-700 border-t-emerald-600 dark:border-t-emerald-400 rounded-full animate-spin mx-auto"></div>
          <p className="text-lg font-semibold text-emerald-700 dark:text-emerald-400">
            Loading dashboard...
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-emerald-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
      {/* Header */}
      {/* <header className="sticky top-0 z-40 bg-white border-b border-gray-100 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-emerald-500 to-teal-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold">🌾</span>
              </div>
              <h1 className="text-2xl font-bold text-transparent bg-gradient-to-r from-emerald-600 to-teal-600 bg-clip-text">
                Dashboard
              </h1>
            </div>
            <div className="flex items-center gap-4">
              <Link
                href="/profile"
                className="text-gray-600 hover:text-emerald-600 font-medium transition-colors"
              >
                👤 Profile
              </Link>
              <button
                onClick={handleLogout}
                className="bg-gradient-to-r from-red-500 to-red-600 text-white px-4 py-2 rounded-lg font-semibold hover:shadow-lg transition-all duration-200"
              >
                Logout
              </button>
            </div>
          </div>
        </div>
      </header> */}

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 lg:py-12">
        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-10">
          {/* Card 1 - Total Laws */}
          <div className="card-hover p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                  Total Laws
                </p>
                <p className="text-4xl font-bold text-transparent bg-gradient-to-r from-emerald-600 to-teal-600 bg-clip-text">
                  {laws.length}
                </p>
              </div>
              <div className="text-4xl">📚</div>
            </div>
          </div>

          {/* Card 2 - Profile */}
          <Link
            href="/profile"
            className="card-hover p-6 hover:border-emerald-300"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                  Your Profile
                </p>
                <p className="font-semibold text-gray-900 dark:text-gray-100">
                  View & Edit
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  Update your info
                </p>
              </div>
              <div className="text-4xl">👤</div>
            </div>
          </Link>

          {/* Card 3 - Categories */}
          <div className="card-hover p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                  Categories
                </p>
                <p className="text-4xl font-bold text-transparent bg-gradient-to-r from-teal-600 to-cyan-600 bg-clip-text">
                  12+
                </p>
              </div>
              <div className="text-4xl">🏷️</div>
            </div>
          </div>

          {/* Card 4 - History */}
          <div className="card-hover p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                  Activity
                </p>
                <p className="font-semibold text-gray-900 dark:text-gray-100">
                  Track Usage
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  View history
                </p>
              </div>
              <div className="text-4xl">📊</div>
            </div>
          </div>
        </div>

        {/* Laws Section */}
        <div className="card overflow-hidden">
          {/* Header */}
          <div className="px-6 lg:px-8 py-6 bg-gradient-to-r from-emerald-50 to-teal-50 dark:from-emerald-900/20 dark:to-teal-900/20 border-b border-gray-100 dark:border-gray-700">
            <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4">
              <div>
                <h2 className="text-2xl lg:text-3xl font-bold text-gray-900 dark:text-gray-100">
                  Farmer Laws & Regulations
                </h2>
                <p className="text-gray-600 dark:text-gray-400 text-sm mt-1">
                  Browse and search agricultural laws
                </p>
              </div>
              <div className="w-full lg:w-64">
                <input
                  type="text"
                  placeholder="Search laws..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="input-field text-sm"
                />
              </div>
            </div>
          </div>

          {/* Content */}
          <div className="p-6 lg:p-8">
            {filteredLaws.length === 0 ? (
              <div className="text-center py-12">
                <div className="text-5xl mb-4">🔍</div>
                <p className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  {searchTerm ? "No laws found" : "No laws available yet"}
                </p>
                <p className="text-gray-600 dark:text-gray-400 mt-2">
                  {searchTerm
                    ? "Try adjusting your search terms"
                    : "Check back soon for more content"}
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {filteredLaws.map((law, index) => (
                  <div
                    key={law.id}
                    className="group card p-6 hover:border-emerald-300 dark:hover:border-emerald-600 border-2 cursor-pointer transition-all"
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-2">
                          <span className="badge badge-success">
                            {law.category}
                          </span>
                          <span className="text-xs text-gray-500 dark:text-gray-400">
                            Law #{index + 1}
                          </span>
                        </div>
                        <h3 className="text-lg font-bold text-gray-900 dark:text-gray-100 group-hover:text-emerald-600 dark:group-hover:text-emerald-400 transition-colors">
                          {law.title}
                        </h3>
                        <p className="text-gray-700 dark:text-gray-300 mt-2 line-clamp-2">
                          {law.description}
                        </p>
                        {law.source && (
                          <p className="text-xs text-gray-500 dark:text-gray-400 mt-3">
                            📌 Source: {law.source}
                          </p>
                        )}
                      </div>
                      <div className="text-2xl">📋</div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Footer */}
          {filteredLaws.length > 0 && (
            <div className="px-6 lg:px-8 py-4 bg-gray-50 dark:bg-slate-800/50 border-t border-gray-100 dark:border-gray-700 text-center">
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Showing {filteredLaws.length} of {laws.length} laws
              </p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
