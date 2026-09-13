import { readFileSync } from "node:fs";
import assert from "node:assert/strict";
import { test } from "node:test";
import { compile } from "../src/lib/workload.ts";

const checks = JSON.parse(readFileSync(new URL("../src/data/compiler-checks.json", import.meta.url), "utf8"));
for (const item of checks) {
  test(`browser compiler agrees with Python: ${item.trace.model_name} layer ${item.trace.layer_index} H=${item.workload.config.hidden_size}`, () => {
    const actual = compile(item.trace, item.workload.config);
    const expected = item.workload;
    assert.deepEqual(actual.histogram, expected.routing.routed_histogram);
    assert.equal(actual.useful, expected.totals.useful_flops);
    assert.equal(actual.padded, expected.totals.padded_flops);
    assert.equal(actual.tiles, expected.totals.output_tiles);
    assert.equal(actual.waves, expected.totals.estimated_waves);
    assert.equal(actual.padding, expected.totals.padding_fraction);
    for (let i = 0; i < 3; i++) {
      assert.equal(actual.projections[i].tiles, expected.projections[i].output_tiles);
      assert.equal(actual.projections[i].waves, expected.projections[i].estimated_waves);
    }
  });
}
