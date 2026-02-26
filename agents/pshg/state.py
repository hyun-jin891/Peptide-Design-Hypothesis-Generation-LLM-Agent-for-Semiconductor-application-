"""
PSHG Agent — State 정의

State는 그래프 전체에서 공유되는 최소한의 필드만 포함합니다.
노드 하나에서만 쓰이는 중간값(output_csv, llm_inputs 등)은 포함하지 않습니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict


@dataclass
class AgentInput:
    """사용자 프롬프트에서 파싱된 실행 설정."""

    substrate: str
    target: str
    peptide_length: int
    num_samples: int
    seed: int = 42


class PSHGState(TypedDict, total=False):
    """LangGraph State — 노드 간 공유 데이터.

    제거된 필드:
      - output_csv: 어떤 노드에서도 읽지 않음 (로컬 변수로 충분)
      - llm_inputs: build_llm → generate_hypothesis 연속 노드에서만 사용
                     → hypothesis 노드 하나로 통합하여 내부 처리
      - retries: 재생성 루프 제거 (점수와 무관하게 LLM에 전달)
    """

    user_prompt: str                    # 그래프 입력 (원본 프롬프트)
    cfg: Dict[str, Any]                 # parse_prompt 결과 (AgentInput.__dict__)
    run_dir: str                        # 실행별 출력 디렉토리 경로
    chemical_context: str               # 화학 RAG 컨텍스트
    sequences: List[str]                # ProtGPT2가 생성한 펩타이드 서열
    candidates: List[Dict[str, Any]]    # 구조/속성 분석 결과
    hypotheses: List[Dict[str, Any]]    # 최종 LLM 가설
