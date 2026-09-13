/** Keep navigation and artifact fetches valid at the configured deployment root. */
export function sitePath(path: string): string {
  return `${import.meta.env.BASE_URL.replace(/\/$/, "")}/${path.replace(/^\//, "")}`;
}
