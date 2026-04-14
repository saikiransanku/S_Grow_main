import { Router, Request, Response } from "express";
import { PrismaClient } from "@prisma/client";

const router = Router();
const prisma = new PrismaClient();

// Get all users
router.get("/", async (req: Request, res: Response) => {
  try {
    const users = await prisma.user.findMany({
      select: {
        id: true,
        email: true,
        firstName: true,
        lastName: true,
        phone: true,
        profile: true,
      },
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
      include: { profile: true },
    });
    if (!user) {
      return res.status(404).json({ error: "User not found" });
    }
    res.json(user);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch user" });
  }
});

// Update user profile
router.put("/:id", async (req: Request, res: Response) => {
  try {
    const user = await prisma.user.update({
      where: { id: req.params.id },
      data: req.body,
      include: { profile: true },
    });
    res.json(user);
  } catch (error) {
    res.status(500).json({ error: "Failed to update user" });
  }
});

export default router;
