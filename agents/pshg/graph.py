"""
PSHG Agent — 그래프 빌더 & CLI 진입점

StateGraph 구성 (선형 흐름):
  parse → chem_context → generate → analyze → hypothesis → END
"""

from __future__ import annotations

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from agents.pshg.state import PSHGState
from agents.pshg.nodes import (
    node_parse,
    node_chemical_context,
    node_generate_sequences,
    node_analyze_candidates,
    node_hypothesis,
)

load_dotenv()


def build_graph():
    """StateGraph를 구성하고 컴파일된 그래프를 반환합니다."""

    g = StateGraph(PSHGState)

    # ── 노드 등록 ──
    g.add_node("parse", node_parse)
    g.add_node("chem_context", node_chemical_context)
    g.add_node("generate", node_generate_sequences)
    g.add_node("analyze", node_analyze_candidates)
    g.add_node("hypothesis", node_hypothesis)

    # ── 엣지 (선형 흐름) ──
    g.set_entry_point("parse")
    g.add_edge("parse", "chem_context")
    g.add_edge("chem_context", "generate")
    g.add_edge("generate", "analyze")
    g.add_edge("analyze", "hypothesis")
    g.add_edge("hypothesis", END)

    return g.compile()


# ─────────────────────────────────────────────
# CLI 진입점
# ─────────────────────────────────────────────

def main():
    graph = build_graph()
    user_prompt = input(
        "프롬프트 입력 (예: 기판 물질: SiO2, target 물질: biotin, "
        "펩타이드 길이: 20, 생성 샘플링 횟수: 3, seed: 7)\n> "
    )
    out = graph.invoke({"user_prompt": user_prompt})

    print("\n=== RUN DIR ===")
    print(out.get("run_dir"))

    print("\n=== Candidates ===")
    cand = out.get("candidates", [])
    print(f"{len(cand)} candidates")

    print("\n=== Generated Hypotheses ===")
    for h in out.get("hypotheses", []):
        print(f"\n[Candidate {h['rank']}]")
        if "error" in h:
            print(f"Error: {h['error']}")
        else:
            print(h["hypothesis"][:1000] + ("..." if len(h["hypothesis"]) > 1000 else ""))


if __name__ == "__main__":
    main()
