import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

const writing = defineCollection({
  loader: glob({
    base: "./src/content/writing",
    pattern: "**/*.{md,mdx}",
  }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    publishedAt: z.coerce.date(),
    updatedAt: z.coerce.date().optional(),
    retrospectiveDate: z.coerce.date().optional(),
    number: z.number().int().positive(),
    readingTime: z.string(),
    tags: z.array(z.string()),
    draft: z.boolean().default(false),
  }),
});

export const collections = { writing };
