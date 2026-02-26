"""
PSHG Agent — Tool / 유틸리티 함수

LLM과 직접 관련 없는 도구 함수들:
  - 프롬프트 파싱
  - 타겟 매핑 / MiMoset 참조 서열 추출
  - ESM-2 임베딩 기반 바인딩 유사도 계산
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F

from agents.pshg.state import AgentInput
from esm_embedding import esm2_embed


# ─────────────────────────────────────────────
# 프롬프트 파싱
# ─────────────────────────────────────────────

def parse_prompt(text: str) -> AgentInput:
    """사용자 텍스트에서 substrate, target, length, num_samples, seed를 추출."""

    def pick(patterns: List[str], default: Optional[str] = None) -> Optional[str]:
        for p in patterns:
            m = re.search(p, text, flags=re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return default

    substrate = pick([r"기판\s*물질\s*[:=]\s*([^\n,]+)", r"substrate\s*[:=]\s*([^\n,]+)"])
    target = pick([r"target\s*물질\s*[:=]\s*([^\n,]+)", r"타겟\s*물질\s*[:=]\s*([^\n,]+)", r"target\s*[:=]\s*([^\n,]+)"])
    length_str = pick([r"펩타이드\s*길이\s*[:=]\s*(\d+)", r"length\s*[:=]\s*(\d+)"])
    n_str = pick([r"생성\s*샘플링\s*횟수\s*[:=]\s*(\d+)", r"num_samples\s*[:=]\s*(\d+)", r"samples\s*[:=]\s*(\d+)"])
    seed_str = pick([r"seed\s*[:=]\s*(\d+)"], default="42")

    if substrate is None or target is None or length_str is None or n_str is None:
        raise ValueError("프롬프트에서 substrate/target/length/num_samples를 못 찾았습니다. 예시 형식으로 입력해 주세요.")

    return AgentInput(
        substrate=substrate,
        target=target,
        peptide_length=int(length_str),
        num_samples=int(n_str),
        seed=int(seed_str) if seed_str else 42,
    )


# ─────────────────────────────────────────────
# 타겟 매핑 & 참조 서열
# ─────────────────────────────────────────────

def load_target_mapping(path: str = "target_mapping.json") -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_target_id(target_name: str, mapping: Dict[str, Any]) -> Optional[str]:
    lowered = {str(k).lower(): str(v) for k, v in mapping.items()}
    return lowered.get(target_name.lower())


def extract_reference_seqs_from_mimoset(
    target_id: int,
    mimoset_path: str = "mimoset.csv",
) -> List[str]:
    df = pd.read_csv(mimoset_path)
    rows = df[df["TargetID"] == int(target_id)]
    if rows.empty:
        return []
    seqs: List[str] = []
    for cell in rows["Sequences"].astype(str).tolist():
        found = re.findall(r"^[A-Z]+", cell, flags=re.MULTILINE)
        seqs.extend(found)
    return list(dict.fromkeys(seqs))


# ─────────────────────────────────────────────
# 바인딩 유사도 계산
# ─────────────────────────────────────────────

@torch.no_grad()
def compute_binding_similarity_context(
    target_name: str,
    generated_seq: str,
    target_mapping_path: str = "target_mapping.json",
    mimoset_path: str = "mimoset.csv",
    topk: int = 5,
) -> Tuple[Optional[float], str]:
    mapping = load_target_mapping(target_mapping_path)
    target_id = get_target_id(target_name, mapping)
    if target_id is None:
        return None, f"[Binding Similarity] target '{target_name}'를 target_mapping에서 찾지 못했습니다."

    ref_seqs = extract_reference_seqs_from_mimoset(int(target_id), mimoset_path=mimoset_path)
    if not ref_seqs:
        return None, f"[Binding Similarity] target '{target_name}'(TargetID={target_id}) reference 서열을 mimoset.csv에서 찾지 못했습니다."

    seqs = ref_seqs + [generated_seq]
    embs = esm2_embed(seqs, pooling="mean")["sequence_embeddings"]
    if embs is None:
        return None, "[Binding Similarity] 임베딩 생성 실패."

    ref_embs = embs[:-1]
    cand_emb = embs[-1].unsqueeze(0)
    sims = F.cosine_similarity(ref_embs, cand_emb, dim=1, eps=1e-8)

    k = min(topk, sims.numel())
    top_vals, top_idx = torch.topk(sims, k=k, largest=True)
    top_vals_list = [float(v) for v in top_vals.cpu()]
    top_seqs = [ref_seqs[int(i)] for i in top_idx.cpu()]

    binding_score = float(sum(top_vals_list) / max(len(top_vals_list), 1))

    context = (
        "The context below provides evidence of binding plausibility based on ESM-2 embedding similarity.\n"
        f"[Target] {target_name} (TargetID={target_id})\n"
        f"[Binding score] top-{k} mean cosine similarity = {binding_score:.4f}\n"
        f"[Top similar reference sequences]\n"
        + json.dumps({s: v for s, v in zip(top_seqs, top_vals_list)}, ensure_ascii=False)
    )
    return binding_score, context
