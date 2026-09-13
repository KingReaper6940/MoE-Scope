/** Keep navigation and artifact fetches valid under a GitHub Pages project path. */
export function sitePath(path: string): string {
  return `${import.meta.env.BASE_URL.replace(/\/$/, "")}/${path.replace(/^\//, "")}`;
}
