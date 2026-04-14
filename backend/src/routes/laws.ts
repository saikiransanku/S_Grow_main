import { Router, Request, Response } from "express";
import { PrismaClient } from "@prisma/client";

const router = Router();
const prisma = new PrismaClient();

// Get all farmer laws
router.get("/", async (req: Request, res: Response) => {
  try {
    const laws = await prisma.farmerLaw.findMany();
    res.json(laws);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch laws" });
  }
});

// Get laws by category
router.get("/category/:category", async (req: Request, res: Response) => {
  try {
    const laws = await prisma.farmerLaw.findMany({
      where: { category: req.params.category },
    });
    res.json(laws);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch laws" });
  }
});

// Get law by ID
router.get("/:id", async (req: Request, res: Response) => {
  try {
    const law = await prisma.farmerLaw.findUnique({
      where: { id: req.params.id },
    });
    if (!law) {
      return res.status(404).json({ error: "Law not found" });
    }
    res.json(law);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch law" });
  }
});

// Create new farmer law (admin only)
// router.post("/", async (req: Request, res: Response) => {
//   try {
//     const law = await prisma.farmerLaw.create({
//       data: req.body,
//     });
//     res.status(201).json(law);
//   } catch (error) {
//     res.status(500).json({ error: "Failed to create law" });
//   }
// });

export default router;
