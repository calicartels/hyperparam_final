import { pgTable, text, serial, integer, boolean, jsonb } from "drizzle-orm/pg-core";
import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

export const users = pgTable("users", {
  id: serial("id").primaryKey(),
  username: text("username").notNull().unique(),
  password: text("password").notNull(),
});

export const insertUserSchema = createInsertSchema(users).pick({
  username: true,
  password: true,
});

export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = typeof users.$inferSelect;

// Hyperparameter schemas
export const hyperparameters = pgTable("hyperparameters", {
  id: serial("id").primaryKey(),
  name: text("name").notNull(),
  paramKey: text("param_key").notNull().unique(),
  description: text("description").notNull(),
  framework: text("framework").notNull(),
  impact: text("impact").notNull(),
  defaultValue: text("default_value"),
  alternatives: jsonb("alternatives"),
});

export const insertHyperparameterSchema = createInsertSchema(hyperparameters).omit({
  id: true,
});

export type InsertHyperparameter = z.infer<typeof insertHyperparameterSchema>;
export type Hyperparameter = typeof hyperparameters.$inferSelect;

// Frameworks schema
export const frameworks = pgTable("frameworks", {
  id: serial("id").primaryKey(),
  name: text("name").notNull().unique(),
  language: text("language").notNull(),
  logoUrl: text("logo_url"),
});

export const insertFrameworkSchema = createInsertSchema(frameworks).omit({
  id: true,
});

export type InsertFramework = z.infer<typeof insertFrameworkSchema>;
export type Framework = typeof frameworks.$inferSelect;
