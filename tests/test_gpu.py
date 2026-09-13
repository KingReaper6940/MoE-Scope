"""Optional hardware tests: pytest tests/test_gpu.py on the pinned GPU image."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA device required", allow_module_level=True)

from moescope.triton_backend import PreparedGroup


@pytest.mark.parametrize("m,n,k", [(0, 31, 17), (1, 33, 15), (31, 63, 47), (128, 256, 64)])
def test_masked_grouped_gemm(m, n, k):
    torch.manual_seed(17)
    a = torch.randn((m, k), device="cuda", dtype=torch.float16) * .1
    b = torch.randn((k, n), device="cuda", dtype=torch.float16) * .1
    expected = (a.float() @ b.float()).half()
    group = PreparedGroup([a], [b])
    torch.testing.assert_close(group.triton()[0], expected, atol=.002, rtol=.01)


def test_python_oracle_matches_independent_torch_expression():
    from moescope.routing import generate_trace
    from moescope.workloads import WorkloadConfig
    from moescope.replay import make_inputs, reference_replay
    trace = generate_trace(tokens=5, experts=4, shared_experts=1, distribution="random")
    inputs = make_inputs(trace, WorkloadConfig(hidden_size=3, intermediate_size=7), 17)
    x = torch.tensor(inputs["x"], dtype=torch.float64, device="cuda")
    out = torch.zeros_like(x)
    for e, expert in enumerate(inputs["experts"]):
        gate, up, down = [torch.tensor(expert[key], dtype=torch.float64, device="cuda") for key in ("gate", "up", "down")]
        y = (torch.nn.functional.silu(x @ gate) * (x @ up)) @ down
        weights = [1. if e == trace.num_routed_experts else
                   (t.weights[t.expert_ids.index(e)] if e in t.expert_ids else 0.) for t in trace.tokens]
        out += y * torch.tensor(weights, dtype=torch.float64, device="cuda")[:, None]
    torch.testing.assert_close(out.cpu(), torch.tensor(reference_replay(trace, inputs), dtype=torch.float64), atol=1e-12, rtol=1e-10)
