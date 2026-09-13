export interface Timing { median_ms: number; p95_ms: number; samples_ms: number[]; }
export interface Case {
  id: string; trace_kind: string; distribution: string; tokens: number; experts: number;
  top_k: number; active_experts: number; hidden_size: number; intermediate_size: number;
  histogram: number[]; trace_sha256: string;
  timings: Record<string, Timing>;
  correctness: Record<string, { passed: boolean; max_abs_error: number }>;
  workloads: Record<string, { padding_fraction: number; output_tiles: number; estimated_waves: number }>;
}
export interface Report {
  complete: boolean; created_at: string;
  environment: { gpu: string; torch: string; triton: string; cuda: string; num_sms: number };
  cases: Case[]; edge_checks: { passed: boolean }[];
}
export function bestBackend(item: Case) {
  return Object.entries(item.timings).filter(([name]) => name.startsWith("triton")).sort((a, b) => a[1].median_ms - b[1].median_ms)[0];
}
