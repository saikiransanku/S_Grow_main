import { Request, Response, NextFunction } from "express";
import jwt from "jsonwebtoken";

export interface AuthRequest extends Request {
  userId?: string;
}

export const authMiddleware = (
  req: AuthRequest,
  res: Response,
  next: NextFunction,
) => {
  try {
    const token = req.headers.authorization?.split(" ")[1];

    if (!token) {
      return res.status(401).json({ error: "No token provided" });
    }

    const decoded: any = jwt.verify(
      token,
      process.env.JWT_SECRET || "secret-key",
    );
    req.userId = decoded.id;
    next();
  } catch (error) {
    res.status(401).json({ error: "Invalid token" });
  }
};
