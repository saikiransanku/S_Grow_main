// ! refactor
import { Router, Request, Response, NextFunction } from "express";
import { randomBytes } from "crypto";
import bcrypt from "bcryptjs";
import { OAuth2Client } from "google-auth-library";
import jwt from "jsonwebtoken";
import { body, validationResult, check } from "express-validator";
import rateLimit from "express-rate-limit";
import xss from "xss";
import { prisma } from "../lib/prisma";

const router = Router();

// Ensure JWT secret exists
if (!process.env.JWT_SECRET) {
  throw new Error("JWT_SECRET is not defined in environment variables");
}

const JWT_SECRET = process.env.JWT_SECRET;
const GOOGLE_CLIENT_ID = process.env.GOOGLE_CLIENT_ID || "";
const GOOGLE_CLIENT_SECRET = process.env.GOOGLE_CLIENT_SECRET || "";
const googleClient = GOOGLE_CLIENT_ID
  ? new OAuth2Client(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET || undefined)
  : null;

// ================= SECURITY CONFIG =================

// Rate limiting - prevent brute force attacks
const registerLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 5,
  message: "Too many registration attempts. Please try again later.",
  standardHeaders: true,
  legacyHeaders: false,
});

const loginLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 10,
  message: "Too many login attempts. Please try again later.",
  standardHeaders: true,
  legacyHeaders: false,
  skipSuccessfulRequests: true,
});

// ================= VALIDATION =================

// Password validation (no sanitization)
const passwordRules = body("password")
  .trim()
  .isLength({ min: 12 })
  .withMessage("Password must be at least 12 characters long")
  .matches(/[A-Z]/)
  .withMessage("Must contain an uppercase letter")
  .matches(/[a-z]/)
  .withMessage("Must contain a lowercase letter")
  .matches(/\d/)
  .withMessage("Must contain a number")
  .matches(/[@$!%*?&]/)
  .withMessage("Must contain a special character");

// Email validation
const emailRules = body("email")
  .trim()
  .toLowerCase()
  .isEmail()
  .withMessage("Invalid email format")
  .isLength({ max: 255 })
  .withMessage("Email too long")
  .normalizeEmail();

// Name validation
const nameRules = (fieldName: string) =>
  body(fieldName)
    .trim()
    .isLength({ min: 1, max: 100 })
    .withMessage(`${fieldName} must be between 1 and 100 characters`)
    .matches(/^[a-zA-Z\s'-]+$/)
    .withMessage(`${fieldName} contains invalid characters`)
    .customSanitizer((value) => {
      return xss(value, { whiteList: {}, stripIgnoreTag: true });
    });

const requireOneOfFields = (fields: string[], label: string) =>
  body().custom((_, { req }) => {
    const hasValue = fields.some((field) => {
      const value = req.body?.[field];
      return typeof value === "string" && value.trim().length > 0;
    });

    if (!hasValue) {
      throw new Error(`${label} required`);
    }

    return true;
  });

const sanitizeProfileName = (value: string | undefined, fallback: string) =>
  xss((value || fallback).trim(), { whiteList: {}, stripIgnoreTag: true })
    .slice(0, 100)
    .trim() || fallback;

const buildAuthResponse = (
  user: {
    id: string;
    email: string;
    firstName: string;
    lastName: string;
    createdAt?: Date;
  },
  message: string,
) => {
  const token = jwt.sign({ id: user.id, email: user.email }, JWT_SECRET, {
    expiresIn: "7d",
    algorithm: "HS256",
    issuer: "your-app",
  });

  return {
    success: true,
    message,
    data: {
      token,
      user: {
        id: user.id,
        email: user.email,
        firstName: user.firstName,
        lastName: user.lastName,
        ...(user.createdAt ? { createdAt: user.createdAt } : {}),
      },
    },
  };
};

// Content type validation
const validateContentType = (
  req: Request,
  res: Response,
  next: NextFunction,
) => {
  if (req.method === "POST" && !req.is("application/json")) {
    return res.status(415).json({
      success: false,
      message: "Content-Type must be application/json",
    });
  }
  next();
};

// ================= REGISTER =================

router.post(
  "/register",
  validateContentType,
  registerLimiter,
  [
    emailRules,
    passwordRules,
    nameRules("firstName").optional(),
    nameRules("lastName").optional(),
    nameRules("firstname").optional(),
    nameRules("lastname").optional(),
    requireOneOfFields(["firstName", "firstname"], "First name"),
    requireOneOfFields(["lastName", "lastname"], "Last name"),
  ],
  async (req: Request, res: Response) => {
    const errors = validationResult(req);

    if (!errors.isEmpty()) {
      console.log("[DEBUG] /register validation errors:", errors.array());

      return res.status(422).json({
        success: false,
        message: "Validation failed",
        errors: errors.array(),
      });
    }

    try {
      const { email, password } = req.body;
      const firstName = req.body.firstName ?? req.body.firstname;
      const lastName = req.body.lastName ?? req.body.lastname;

      const sanitizedFirstname = xss(firstName);
      const sanitizedLastname = xss(lastName);

      // Check if user exists
      const existingUser = await prisma.user.findUnique({
        where: { email },
      });

      if (existingUser) {
        return res.status(409).json({
          success: false,
          message:
            "Registration failed. Please verify your details and try again.",
        });
      }

      // Hash password
      const hashedPassword = await bcrypt.hash(password, 12);

      // Create user
      const user = await prisma.user.create({
        data: {
          email,
          password: hashedPassword,
          firstName: sanitizedFirstname,
          lastName: sanitizedLastname,
        },
        select: {
          id: true,
          email: true,
          firstName: true,
          lastName: true,
          createdAt: true,
        },
      });

      console.log(
        `[SECURITY] New user registered at ${new Date().toISOString()}`,
      );

      res
        .status(201)
        .json(buildAuthResponse(user, "User registered successfully"));
    } catch (error) {
      console.error("[ERROR] Registration error:", error);

      res.status(500).json({
        success: false,
        message: "Registration failed. Please try again later.",
      });
    }
  },
);

// ================= LOGIN =================

router.post(
  "/login",
  validateContentType,
  loginLimiter,
  [
    emailRules.bail(),
    body("password")
      .trim()
      .notEmpty()
      .withMessage("Password is required")
      .isLength({ max: 1000 })
      .withMessage("Invalid password format"),
  ],
  async (req: Request, res: Response) => {
    const errors = validationResult(req);

    if (!errors.isEmpty()) {
      console.log("[DEBUG] /login validation errors:", errors.array());

      return res.status(422).json({
        success: false,
        message: "Validation failed",
        errors: errors.array(),
        debug: {
          contentType: req.get("content-type"),
          bodyKeys: Object.keys(req.body || {}),
          hasEmail: Boolean(req.body?.email),
          hasPassword: Boolean(req.body?.password),
        },
      });
    }

    try {
      const { email, password } = req.body;

      // Find user
      const user = await prisma.user.findUnique({
        where: { email },
      });

      if (!user) {
        console.warn(
          `[SECURITY] Login attempt for unknown user at ${new Date().toISOString()}`,
        );

        return res.status(401).json({
          success: false,
          message: "Invalid email or password",
        });
      }

      // Compare password
      const passwordMatch = await bcrypt.compare(password, user.password);

      if (!passwordMatch) {
        console.warn(
          `[SECURITY] Failed login attempt at ${new Date().toISOString()}`,
        );

        return res.status(401).json({
          success: false,
          message: "Invalid email or password",
        });
      }

      console.log(`[SECURITY] Successful login at ${new Date().toISOString()}`);

      res.status(200).json(
        buildAuthResponse(
          {
            id: user.id,
            email: user.email,
            firstName: user.firstName,
            lastName: user.lastName,
          },
          "Login successful",
        ),
      );
    } catch (error) {
      console.error("[ERROR] Login error:", error);

      res.status(500).json({
        success: false,
        message: "Login failed. Please try again later.",
      });
    }
  },
);

// ================= GOOGLE LOGIN =================

router.post(
  "/google",
  validateContentType,
  loginLimiter,
  [
    body("credential")
      .trim()
      .notEmpty()
      .withMessage("Google credential is required")
      .isLength({ max: 5000 })
      .withMessage("Google credential is too large"),
  ],
  async (req: Request, res: Response) => {
    const errors = validationResult(req);

    if (!errors.isEmpty()) {
      return res.status(422).json({
        success: false,
        message: "Validation failed",
        errors: errors.array(),
      });
    }

    if (!GOOGLE_CLIENT_ID || !googleClient) {
      return res.status(500).json({
        success: false,
        message: "Google login is not configured on the server",
      });
    }

    try {
      const credential = String(req.body.credential || "").trim();
      const ticket = await googleClient.verifyIdToken({
        idToken: credential,
        audience: GOOGLE_CLIENT_ID,
      });
      const payload = ticket.getPayload();

      if (!payload?.email || payload.email_verified !== true) {
        return res.status(401).json({
          success: false,
          message: "Google account email is not verified",
        });
      }

      const email = payload.email.trim().toLowerCase();
      const firstName = sanitizeProfileName(payload.given_name, "Google");
      const lastName = sanitizeProfileName(payload.family_name, "User");

      let user = await prisma.user.findUnique({
        where: { email },
      });

      if (!user) {
        const generatedPassword = randomBytes(32).toString("hex");
        const hashedPassword = await bcrypt.hash(generatedPassword, 12);

        user = await prisma.user.create({
          data: {
            email,
            password: hashedPassword,
            firstName,
            lastName,
          },
        });

        console.log(
          `[SECURITY] Google signup completed at ${new Date().toISOString()}`,
        );

        return res
          .status(201)
          .json(buildAuthResponse(user, "Google account linked successfully"));
      }

      console.log(
        `[SECURITY] Google login completed at ${new Date().toISOString()}`,
      );

      return res
        .status(200)
        .json(buildAuthResponse(user, "Google login successful"));
    } catch (error) {
      console.error("[ERROR] Google login error:", error);

      return res.status(401).json({
        success: false,
        message: "Google authentication failed",
      });
    }
  },
);

// ================= LOGOUT =================

router.post("/logout", (req: Request, res: Response) => {
  try {
    // Optional: implement Redis token blacklist
    console.log(`[SECURITY] User logged out at ${new Date().toISOString()}`);

    res.json({
      success: true,
      message: "Logged out successfully",
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      message: "Logout failed",
    });
  }
});

export default router;
