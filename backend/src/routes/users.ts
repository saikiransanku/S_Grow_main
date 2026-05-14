import { Router, Request, Response } from "express";
import { authMiddleware, AuthRequest } from "../middleware/auth";
import { prisma } from "../lib/prisma";

const router = Router();

const USER_UPDATE_FIELDS = [
  "firstName",
  "lastName",
  "phone",
  "address",
  "city",
  "state",
  "pincode",
] as const;

const PROFILE_UPDATE_FIELDS = [
  "farmSize",
  "cropType",
  "bio",
  "avatar",
  "district",
  "mandalVillage",
  "soilType",
  "waterSource",
  "irrigationLevel",
  "seasonPreference",
  "cropPurpose",
  "previousCrop",
  "budget",
  "marketPreference",
  "riskPreference",
  "croppingPreference",
] as const;

const USER_RESPONSE_SELECT = {
  id: true,
  email: true,
  firstName: true,
  lastName: true,
  phone: true,
  address: true,
  city: true,
  state: true,
  pincode: true,
  profile: true,
  createdAt: true,
  updatedAt: true,
} as const;

const normalizeOptionalString = (value: unknown) => {
  if (typeof value !== "string") return undefined;
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
};

const normalizeOptionalNumber = (value: unknown) => {
  if (value === "" || value === null) return null;
  if (typeof value === "number") {
    return Number.isFinite(value) ? value : undefined;
  }
  if (typeof value === "string") {
    const parsed = Number.parseFloat(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
};

const pickUserUpdateData = (body: Record<string, unknown>) => {
  const data: Record<string, string | null> = {};

  USER_UPDATE_FIELDS.forEach((field) => {
    const normalized = normalizeOptionalString(body[field]);
    if (normalized !== undefined) {
      data[field] = normalized;
    }
  });

  return data;
};

const pickProfileUpdateData = (body: Record<string, unknown>) => {
  const nestedProfile =
    body.profile && typeof body.profile === "object"
      ? (body.profile as Record<string, unknown>)
      : {};
  const source = { ...body, ...nestedProfile };
  const data: Record<string, string | number | null> = {};

  PROFILE_UPDATE_FIELDS.forEach((field) => {
    if (field === "farmSize") {
      const normalized = normalizeOptionalNumber(source[field]);
      if (normalized !== undefined) {
        data[field] = normalized;
      }
      return;
    }

    const normalized = normalizeOptionalString(source[field]);
    if (normalized !== undefined) {
      data[field] = normalized;
    }
  });

  return data;
};

const updateUserProfile = async (
  userId: string,
  body: Record<string, unknown>,
) => {
  const userData = pickUserUpdateData(body);
  const profileData = pickProfileUpdateData(body);
  const hasProfileUpdate = Object.keys(profileData).length > 0;

  return prisma.user.update({
    where: { id: userId },
    data: {
      ...userData,
      ...(hasProfileUpdate
        ? {
            profile: {
              upsert: {
                update: profileData,
                create: profileData,
              },
            },
          }
        : {}),
    },
    select: USER_RESPONSE_SELECT,
  });
};

router.get("/me", authMiddleware, async (req: AuthRequest, res: Response) => {
  try {
    if (!req.userId) {
      return res.status(401).json({ error: "Unauthorized" });
    }

    const user = await prisma.user.findUnique({
      where: { id: req.userId },
      select: USER_RESPONSE_SELECT,
    });

    if (!user) {
      return res.status(404).json({ error: "User not found" });
    }

    return res.json(user);
  } catch (error) {
    return res.status(500).json({ error: "Failed to fetch user" });
  }
});

// Get all users
router.get("/", async (req: Request, res: Response) => {
  try {
    const users = await prisma.user.findMany({
      select: USER_RESPONSE_SELECT,
    });
    res.json(users);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch users" });
  }
});

// Get user by ID
router.get("/:id", async (req: Request, res: Response) => {
  try {
    const user = await prisma.user.findUnique({
      where: { id: req.params.id },
      select: USER_RESPONSE_SELECT,
    });
    if (!user) {
      return res.status(404).json({ error: "User not found" });
    }
    res.json(user);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch user" });
  }
});

router.put("/me", authMiddleware, async (req: AuthRequest, res: Response) => {
  try {
    if (!req.userId) {
      return res.status(401).json({ error: "Unauthorized" });
    }

    const user = await updateUserProfile(
      req.userId,
      req.body as Record<string, unknown>,
    );
    return res.json(user);
  } catch (error) {
    return res.status(500).json({ error: "Failed to update user" });
  }
});

// Update user profile
router.put("/:id", async (req: Request, res: Response) => {
  try {
    const user = await updateUserProfile(
      req.params.id,
      req.body as Record<string, unknown>,
    );
    res.json(user);
  } catch (error) {
    res.status(500).json({ error: "Failed to update user" });
  }
});

export default router;
