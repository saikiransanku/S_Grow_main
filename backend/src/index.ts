import express, { Express, Request, Response } from "express";
import cors from "cors";
import helmet from "helmet";
import dotenv from "dotenv";
import cookieParser from "cookie-parser";
import authRoutes from "./routes/auth";
import userRoutes from "./routes/users";
import lawRoutes from "./routes/laws";
import historyRoutes from "./routes/history";

dotenv.config();

const app: Express = express();
const port = process.env.PORT || 5000;

// ============ SECURITY MIDDLEWARE ============

// Helmet for additional security headers
app.use(
  helmet({
    contentSecurityPolicy: {
      directives: {
        defaultSrc: ["'self'"],
        styleSrc: ["'self'", "'unsafe-inline'"],
        scriptSrc: ["'self'"],
        imgSrc: ["'self'", "data:", "https:"],
      },
    },
    hsts: {
      maxAge: 31536000, // 1 year
      includeSubDomains: true,
      preload: true,
    },
    frameguard: {
      action: "deny",
    },
    noSniff: true,
    xssFilter: true,
    referrerPolicy: {
      policy: "strict-origin-when-cross-origin",
    },
  }),
);

// CORS configuration
app.use(
  cors({
    origin: process.env.CORS_ORIGIN || "http://localhost:3000",
    credentials: true,
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"],
    maxAge: 86400, // 24 hours
  }),
);

// Body parser with size limits
app.use(express.json({ limit: "10kb" }));
app.use(express.urlencoded({ extended: true, limit: "10kb" }));

// Cookie parser for secure cookie handling
app.use(cookieParser(process.env.COOKIE_SECRET || "your-secret-key"));

// ============ SECURITY LOGGING ============
// Log all requests for security audit
app.use((req: Request, res: Response, next) => {
  const start = Date.now();
  res.on("finish", () => {
    const duration = Date.now() - start;
    const log = {
      timestamp: new Date().toISOString(),
      method: req.method,
      path: req.path,
      status: res.statusCode,
      duration: `${duration}ms`,
      ip: req.ip,
      userAgent: req.get("user-agent"),
    };
    if (res.statusCode >= 400) {
      console.log("[REQUEST LOG]", JSON.stringify(log));
    }
  });
  next();
});

// ============ INPUT VALIDATION MIDDLEWARE ============
// Prevent NoSQL injection and parameter pollution
app.use((req: Request, res: Response, next) => {
  // Check for suspicious patterns
  const checkForSuspiciousPatterns = (obj: any): boolean => {
    if (!obj) return false;
    for (const key in obj) {
      if (typeof key === "string") {
        // Check for injection patterns
        if (key.includes("$") || key.includes(".")) {
          return true;
        }
      }
      if (typeof obj[key] === "object") {
        if (checkForSuspiciousPatterns(obj[key])) {
          return true;
        }
      }
    }
    return false;
  };

  if (checkForSuspiciousPatterns(req.body)) {
    return res.status(400).json({
      error: "Invalid request parameters",
    });
  }
  next();
});

// ============ ROUTES ============

// Health Check Route
app.get("/api/health", (req: Request, res: Response) => {
  res.json({ status: "OK", timestamp: new Date().toISOString() });
});

// API Routes
app.use("/api/auth", authRoutes);
app.use("/api/users", userRoutes);
app.use("/api/laws", lawRoutes);
app.use("/api/history", historyRoutes);

// ============ ERROR HANDLING ============

// 404 Handler
app.use((req: Request, res: Response) => {
  res.status(404).json({ error: "Route not found" });
});

// Global Error Handler
app.use((err: any, req: Request, res: Response) => {
  console.error("[ERROR]", {
    message: err.message,
    stack: err.stack,
    timestamp: new Date().toISOString(),
  });

  // Don't expose internal error details
  res.status(err.status || 500).json({
    error: "Internal server error",
    ...(process.env.NODE_ENV === "development" && { details: err.message }),
  });
});

app.listen(port, () => {
  console.log(`✓ Server running on http://localhost:${port}`);
  console.log(`✓ Environment: ${process.env.NODE_ENV || "development"}`);
});
