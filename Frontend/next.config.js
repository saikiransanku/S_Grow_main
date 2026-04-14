/** @type {import('next').NextConfig} */
const defaultGoogleClientId =
  "389731022642-pf5bg3jqevju3b2vh0gkbm03a1gkrce3.apps.googleusercontent.com";

const nextConfig = {
  reactStrictMode: true,
  env: {
    NEXT_PUBLIC_API_URL:
      process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000/api",
    NEXT_PUBLIC_AI_API_URL:
      process.env.NEXT_PUBLIC_AI_API_URL || "http://localhost:8000/api/ai",
    NEXT_PUBLIC_GOOGLE_CLIENT_ID:
      process.env.NEXT_PUBLIC_GOOGLE_CLIENT_ID || defaultGoogleClientId,
  },
};

module.exports = nextConfig;
