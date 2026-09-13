export interface Trace {
  schema_version: string;
  trace_kind: string;
  model_name: string;
  model_revision: string;
  num_routed_experts: number;
  num_shared_experts: number;
  routed_experts_per_token: number;
  tokens: { token_index: number; expert_ids: number[]; weights: number[] }[];
}
export interface Config {
  hidden_size: number; intermediate_size: number; block_m: number;
  block_n: number; block_k: number; num_sms: number; ctas_per_sm: number;
}
export function compile(trace: Trace, config: Config) {
  for (const value of Object.values(config)) {
    if (!Number.isSafeInteger(value) || value <= 0) throw new Error("Configuration values must be positive integers.");
  }
  const histogram = Array<number>(trace.num_routed_experts).fill(0);
  for (const token of trace.tokens) for (const e of token.expert_ids) histogram[e]++;
  const groups = [
    ...histogram.map((m, expert) => ({ kind: "routed", expert, m })),
    ...Array.from({ length: trace.num_shared_experts }, (_, expert) => ({ kind: "shared", expert, m: trace.tokens.length })),
  ];
  const ceil = (n: number, d: number) => Math.ceil(n / d);
  const projections = [
    { name: "gate", n: config.intermediate_size, k: config.hidden_size },
    { name: "up", n: config.intermediate_size, k: config.hidden_size },
    { name: "down", n: config.hidden_size, k: config.intermediate_size },
  ].map(({ name, n, k }) => {
    const gemms = groups.map(g => ({ ...g, n, k,
      tiles: ceil(g.m, config.block_m) * ceil(n, config.block_n),
      useful: 2 * g.m * n * k,
      padded: 2 * ceil(g.m, config.block_m) * config.block_m * ceil(n, config.block_n) * config.block_n * ceil(k, config.block_k) * config.block_k,
    }));
    const tiles = gemms.reduce((sum, g) => sum + g.tiles, 0);
    return { name, gemms, tiles, waves: ceil(tiles, config.num_sms * config.ctas_per_sm) };
  });
  const useful = projections.flatMap(p => p.gemms).reduce((sum, g) => sum + g.useful, 0);
  const padded = projections.flatMap(p => p.gemms).reduce((sum, g) => sum + g.padded, 0);
  return { histogram, groups, projections, useful, padded, padding: 1 - useful / padded,
    tiles: projections.reduce((sum, p) => sum + p.tiles, 0),
    waves: projections.reduce((sum, p) => sum + p.waves, 0) };
}
