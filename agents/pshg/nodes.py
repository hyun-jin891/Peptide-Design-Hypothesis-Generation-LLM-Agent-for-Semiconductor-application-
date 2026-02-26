"""
PSHG Agent — 그래프 노드 함수

각 노드는 PSHGState를 받아 변경할 필드만 dict로 반환합니다.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from agents.pshg.state import PSHGState
from agents.pshg.tools import parse_prompt, compute_binding_similarity_context
from agents.pshg.chains import build_llm_prompt, call_hypothesis_llm

from chemical_retriever import ChemicalRetriever
from protGPT2_inference import inference as protgpt2_generate
from esm_inference import inference as esmfold_infer
from biopython_analysis_function import sequence_analysis, compute_global_properties_from_pdb
from propka_context_generation import run_propka, get_propka_context


# ─────────────────────────────────────────────
# 1. 프롬프트 파싱
# ─────────────────────────────────────────────

def node_parse(state: PSHGState) -> PSHGState:
    cfg = parse_prompt(state["user_prompt"])
    run_dir = Path("runs") / f"{cfg.target}_{cfg.substrate}_L{cfg.peptide_length}_N{cfg.num_samples}_seed{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "cfg": cfg.__dict__,
        "run_dir": str(run_dir),
        "hypotheses": [],
    }


# ─────────────────────────────────────────────
# 2. 화학 컨텍스트 생성
# ─────────────────────────────────────────────

def node_chemical_context(state: PSHGState) -> PSHGState:
    cfg = state["cfg"]
    chem = ChemicalRetriever()
    substrate_data = chem.search_molecule(cfg["substrate"])
    target_data = chem.search_molecule(cfg["target"])
    chemical_context = (
        "[Chemical Context]\n"
        + chem.generate_rag_context(substrate_data)
        + "\n\n"
        + chem.generate_rag_context(target_data)
    )
    return {"chemical_context": chemical_context}


# ─────────────────────────────────────────────
# 3. 펩타이드 서열 생성
# ─────────────────────────────────────────────

def node_generate_sequences(state: PSHGState) -> PSHGState:
    cfg = state["cfg"]
    run_dir = Path(state["run_dir"])

    # output_csv는 로컬 변수로만 사용 (state에 저장하지 않음)
    out_csv = run_dir / "output.csv"
    df = protgpt2_generate(
        sequence_length=int(cfg["peptide_length"]),
        num_sequences=int(cfg["num_samples"]),
        output_path=str(out_csv),
        seed=int(cfg["seed"]),
    )
    sequences = df["Sequence"].astype(str).tolist() if "Sequence" in df.columns else []
    return {"sequences": sequences}


# ─────────────────────────────────────────────
# 4. 후보 분석
# ─────────────────────────────────────────────

def node_analyze_candidates(state: PSHGState) -> PSHGState:
    cfg = state["cfg"]
    run_dir = Path(state["run_dir"])
    seqs = state.get("sequences", [])
    results: List[Dict[str, Any]] = []

    for i, seq in enumerate(seqs, start=1):
        item: Dict[str, Any] = {"rank": i, "sequence": seq}

        pdb_path = run_dir / f"cand_{i:03d}.pdb"
        try:
            esmfold_infer(seq, str(pdb_path))
            item["pdb_path"] = str(pdb_path)
        except Exception as e:
            item["pdb_error"] = f"{type(e).__name__}: {e}"
            results.append(item)
            continue

        try:
            item["sequence_props"] = sequence_analysis(seq)
        except Exception as e:
            item["sequence_props_error"] = f"{type(e).__name__}: {e}"

        try:
            item["structure_props"] = compute_global_properties_from_pdb(str(pdb_path))
        except Exception as e:
            item["structure_props_error"] = f"{type(e).__name__}: {e}"

        try:
            run_propka(str(pdb_path))
            pka_path = str(pdb_path).replace(".pdb", ".pka")
            if os.path.exists(pka_path):
                item["propka_context"] = get_propka_context(pka_path)
            else:
                item["propka_context_error"] = f".pka not found: {pka_path}"
        except Exception as e:
            item["propka_context_error"] = f"{type(e).__name__}: {e}"

        try:
            score, ctx = compute_binding_similarity_context(
                target_name=str(cfg["target"]),
                generated_seq=seq,
                target_mapping_path="target_mapping.json",
                mimoset_path="mimoset.csv",
                topk=5,
            )
            item["binding_score"] = score
            item["binding_context"] = ctx
        except Exception as e:
            item["binding_context_error"] = f"{type(e).__name__}: {e}"

        results.append(item)

    return {"candidates": results}


# ─────────────────────────────────────────────
# 5. 가설 생성 (build_llm + generate_hypothesis 통합)
# ─────────────────────────────────────────────

def node_hypothesis(state: PSHGState) -> PSHGState:
    """LLM 프롬프트 빌드 → OpenAI 호출까지 한 노드에서 처리.

    기존에는 build_llm → generate_hypothesis 로 나뉘어
    llm_inputs를 state에 저장했으나, 이제 내부 변수로 처리합니다.
    """
    chemical_context = state.get("chemical_context", "")
    candidates = state.get("candidates", [])
    hypotheses: List[Dict[str, Any]] = []

    print(f"\n[Hypothesis Generation] Generating hypotheses for {len(candidates)} candidates...")

    for i, cand in enumerate(candidates, start=1):
        prompt = build_llm_prompt(cand, chemical_context)
        try:
            text = call_hypothesis_llm(prompt)
            hypotheses.append({"rank": i, "hypothesis": text})
        except Exception as e:
            hypotheses.append({"rank": i, "error": f"LLM Error: {e}"})

    return {"hypotheses": hypotheses}
