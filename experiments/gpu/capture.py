"""Capture real pretrained OLMoE routing, with prompt and environment provenance."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch
import transformers
from huggingface_hub import model_info
from transformers import AutoTokenizer, OlmoeForCausalLM

from moescope.trace import RoutingTrace, RoutingRule, TokenRoute, TRACE_SCHEMA_VERSION
from moescope.workloads import artifact_hash

PROMPTS = {
    "systems": "Explain how a database uses a write ahead log to recover after a crash. Describe the order of writes, durable storage, and replay.",
    "code": "Write a Python function that merges two sorted lists. Explain the time complexity and handle empty inputs.\n\ndef merge_sorted(left, right):",
    "math": "A rectangle has a perimeter of 34 meters and a length of 12 meters. Calculate its width and area, explaining each step.",
    "science": "Explain why the seasons change on Earth. Discuss axial tilt, sunlight angles, day length, and the difference between the two hemispheres.",
    "narrative": "The old observatory had been closed for years. On a rainy evening, two friends discovered a notebook beside the telescope. The first page said",
    "long_systems": ("A mixture of experts model routes each token to a subset of experts. The selected tokens are gathered into groups, multiplied by expert weights, and scattered back. " * 12),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/capture"))
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    torch.manual_seed(0)
    args.output.mkdir(parents=True, exist_ok=True)
    model_id = "allenai/OLMoE-1B-7B-0924"
    revision = model_info(model_id, revision=args.revision).sha
    print(f"Loading {model_id}@{revision}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = OlmoeForCausalLM.from_pretrained(model_id, revision=revision,
                                            torch_dtype=torch.bfloat16,
                                            device_map="cuda", attn_implementation="sdpa").eval()
    config = model.config
    current = {}
    events, checks = [], []
    handles = []
    seen = {}
    # Independent observation: count the rows actually delivered to every expert.
    for layer, decoder in enumerate(model.model.layers):
        for expert, module in enumerate(decoder.mlp.experts):
            def expert_hook(module, inputs, layer=layer, expert=expert):
                seen[(layer, expert)] = inputs[0].shape[0]
            handles.append(module.register_forward_pre_hook(expert_hook))
        def block_hook(module, inputs, outputs, layer=layer):
            logits = outputs[1]
            probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
            weights, selected = torch.topk(probs, module.top_k, dim=-1)
            if module.norm_topk_prob:
                weights = weights / weights.sum(dim=-1, keepdim=True)
            weights = weights.to(inputs[0].dtype).float().cpu().tolist()
            ids = selected.cpu().tolist()
            counts = torch.bincount(selected.flatten(), minlength=module.num_experts).cpu().tolist()
            observed = [seen.get((layer, expert), 0) for expert in range(module.num_experts)]
            if counts != observed:
                raise RuntimeError(f"expert dispatch mismatch in layer {layer}")
            trace = RoutingTrace(TRACE_SCHEMA_VERSION, "captured", model_id, revision,
                                 current["phase"], layer, current["step"], module.num_experts, 0,
                                 module.top_k, RoutingRule("softmax", "top_k",
                                 "sum_to_one_before_dtype_cast" if module.norm_topk_prob else "not_normalized"),
                                 tuple(TokenRoute(current["offset"] + i, tuple(es), tuple(ws))
                                       for i, (es, ws) in enumerate(zip(ids, weights))))
            name = f'{current["prompt"]}-{current["phase"]}-{current["step"]:02d}-layer-{layer:02d}.json'
            (args.output / name).write_text(trace.to_json() + "\n")
            events.append({"file": name, "sha256": artifact_hash(trace.to_dict()),
                           "prompt_id": current["prompt"], "num_tokens": trace.num_tokens,
                           "layer": layer, "phase": current["phase"], "step": current["step"],
                           "histogram": counts, "dispatch_counts_verified": True})
            # Verify one complete learned expert block per prompt against an explicit oracle.
            if layer == 0 and current["phase"] == "prefill":
                hidden = inputs[0].reshape(-1, config.hidden_size)
                expected = torch.zeros_like(hidden)
                selected_weights = torch.tensor(weights, device=hidden.device, dtype=hidden.dtype)
                for expert in range(module.num_experts):
                    row, slot = torch.where(selected == expert)
                    if row.numel():
                        values = module.experts[expert](hidden[row]) * selected_weights[row, slot, None]
                        expected.index_add_(0, row, values)
                actual = outputs[0].reshape_as(expected)
                torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
                checks.append({"prompt_id": current["prompt"], "layer": layer,
                               "max_abs_error": float((actual - expected).abs().max()),
                               "atol": .02, "rtol": .02, "passed": True})
        handles.append(decoder.mlp.register_forward_hook(block_hook))
    prompts = []
    try:
        with torch.inference_mode():
            for name, prompt in PROMPTS.items():
                tokens = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to("cuda")
                current.update(prompt=name, phase="prefill", step=0, offset=0)
                seen.clear()
                output = model(**tokens, use_cache=True)
                token_ids = tokens.input_ids[0].cpu().tolist()
                generated = []
                for step in range(1, 5):
                    next_id = output.logits[:, -1:].argmax(dim=-1)
                    generated.append(int(next_id.item()))
                    current.update(phase="decode", step=step, offset=len(token_ids) + step - 1)
                    seen.clear()
                    output = model(input_ids=next_id, past_key_values=output.past_key_values, use_cache=True)
                prompts.append({"id": name, "text": prompt, "input_ids": token_ids,
                                "generated_token_ids": generated,
                                "prompt_sha256": artifact_hash(prompt)})
                print(f"Captured {name}: {len(token_ids)} prefill tokens + 4 decode steps", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    manifest = {"schema_version": "1.0.0", "artifact_kind": "model_capture_manifest",
                "created_at": datetime.now(timezone.utc).isoformat(), "model": model_id,
                "revision": revision, "torch": torch.__version__, "transformers": transformers.__version__,
                "device": torch.cuda.get_device_name(), "dtype": "bfloat16", "seed": 0,
                "hidden_size": config.hidden_size, "intermediate_size": config.intermediate_size,
                "num_experts": config.num_experts, "top_k": config.num_experts_per_tok,
                "num_layers": config.num_hidden_layers, "shared_experts": 0,
                "capture_method": "MLP forward hooks; top-k reconstructed using the pinned model rule; expert input counts checked independently",
                "limitations": "six authored prompts; greedy decode; batch one; not a representative production dataset",
                "prompts": prompts, "events": events, "learned_block_checks": checks}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {len(events)} verified routing events", flush=True)


if __name__ == "__main__":
    main()
