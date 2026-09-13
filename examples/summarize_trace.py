"""Load and summarize the hand-checkable four-token trace."""

from pathlib import Path

from moescope.trace import RoutingTrace, summarize_routing

TRACE_PATH = Path(__file__).parent / "traces" / "four-token-top2.json"


def main() -> None:
    trace = RoutingTrace.from_json(TRACE_PATH.read_text())
    print(summarize_routing(trace).to_json())


if __name__ == "__main__":
    main()
