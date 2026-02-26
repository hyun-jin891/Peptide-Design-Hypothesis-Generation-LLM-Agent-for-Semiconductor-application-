"""
PSHG Agent — LLM Chain 로직

LLM 프롬프트 구성 및 OpenAI API 호출을 담당합니다.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from openai import OpenAI


# ─────────────────────────────────────────────
# 프롬프트 빌드
# ─────────────────────────────────────────────

def build_llm_prompt(candidate: Dict[str, Any], chemical_context: str) -> str:
    """후보 분석 결과 + 화학 컨텍스트 → LLM 프롬프트 문자열."""

    blocks: List[str] = []
    blocks.append(f"[Candidate]\nsequence={candidate.get('sequence')}\npdb={candidate.get('pdb_path', 'N/A')}")
    blocks.append(chemical_context)

    if "sequence_props" in candidate:
        blocks.append("[Sequence Properties]\n" + json.dumps(candidate["sequence_props"], ensure_ascii=False))
    if "structure_props" in candidate:
        blocks.append("[Structure Properties]\n" + json.dumps(candidate["structure_props"], ensure_ascii=False))
    if "propka_context" in candidate:
        blocks.append(candidate["propka_context"])
    if "binding_context" in candidate:
        blocks.append(candidate["binding_context"])

    task = (
        "Task: Based on the contexts above, generate a scientifically cautious hypothesis about whether "
        "this peptide could plausibly bind the target molecule while being compatible with the substrate. "
        "Do NOT claim it is proven; present it as a hypothesis and suggest what should be validated next (e.g., MD/docking/assay)."
    )

    return "\n\n".join(blocks) + "\n\n" + task


# ─────────────────────────────────────────────
# LLM 호출
# ─────────────────────────────────────────────

SYSTEM_MSG = "You are a specialized AI assistant in peptide design and semiconductor materials science."


def call_hypothesis_llm(prompt: str, model: str = "gpt-4o", temperature: float = 0.7) -> str:
    """OpenAI API를 호출하여 가설 텍스트를 반환합니다."""

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content
