import { Router, Request, Response } from "express";
import { PrismaClient } from "@prisma/client";

const router = Router();
const prisma = new PrismaClient();

// Get usage history for a user
router.get("/user/:userId", async (req: Request, res: Response) => {
  try {
    const history = await prisma.usageHistory.findMany({
      where: { userId: req.params.userId },
      orderBy: { createdAt: "desc" },
    });
    res.json(history);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch history" });
  }
});

// Log user action
router.post("/", async (req: Request, res: Response) => {
  try {
    const { userId, action, data } = req.body;
    const record = await prisma.usageHistory.create({
      data: { userId, action, data },
    });
    res.status(201).json(record);
  } catch (error) {
    res.status(500).json({ error: "Failed to log action" });
  }
});

export default router;
