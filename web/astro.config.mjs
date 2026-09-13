import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";
import { unified } from "@astrojs/markdown-remark";

const base = process.env.SITE_BASE ?? "/MoE-Scope";
// Markdown links need the same deployment prefix as Astro templates.
function remarkBaseLinks() {
  return (tree) => {
    const visit = (node) => {
      if (typeof node.url === "string" && node.url.startsWith("/") && !node.url.startsWith("//")) {
        node.url = `${base.replace(/\/$/, "")}${node.url}`;
      }
      node.children?.forEach(visit);
    };
    visit(tree);
  };
}

export default defineConfig({
  integrations: [mdx()],
  markdown: { processor: unified({ remarkPlugins: [remarkBaseLinks] }) },
  output: "static",
  prefetch: true,
  site: process.env.SITE_ORIGIN ?? "https://KingReaper6940.github.io",
  base,
  trailingSlash: "always",
});
