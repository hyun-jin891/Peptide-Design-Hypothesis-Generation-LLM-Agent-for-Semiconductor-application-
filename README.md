# PSHGagent: Protein & Semiconductor Hypothesis Generation Agent

**PSHGagent**는 기판(substrate)과 타겟(target) 분자를 입력받아, 해당 조건에 맞는 펩타이드 서열을 생성하고, 구조·물성·결합 유사도를 분석한 뒤 **LLM으로 바인딩 가설**을 생성하는 파이프라인 에이전트입니다. 반도체/바이오 소재 연구에서 “이 펩타이드가 타겟에 붙을 가능성이 있는가?”를 과학적으로 신중한 가설로 정리해 줍니다.

---

## 목차

- [개요](#개요)
- [시스템 요구사항](#시스템-요구사항)
- [설치](#설치)
- [데이터 파일](#데이터-파일)
- [사용법](#사용법)
- [파이프라인 구조](#파이프라인-구조)
- [가설 생성 시 주입되는 Context](#가설-생성-시-주입되는-context)
- [모듈 설명](#모듈-설명)
- [환경 변수](#환경-변수)
- [출력 구조](#출력-구조)

---

## 개요

- **입력**: 기판 물질명, 타겟 물질명, 펩타이드 길이, 샘플링 횟수, 시드 등이 포함된 자연어 프롬프트
- **처리**: LangGraph 기반 선형 파이프라인  
  `parse → chem_context → generate → analyze → hypothesis → END`
- **출력**: 실행별 `runs/` 디렉터리(서열·PDB·분석 결과)와 각 후보에 대한 **가설 텍스트** (OpenAI GPT-4o)

주요 단계:
1. **화학 컨텍스트**: PubChem으로 기판/타겟 분자 정보 조회 → RAG용 텍스트 생성  
2. **서열 생성**: ProtGPT2로 펩타이드 서열 샘플링  
3. **구조·물성 분석**: ESMFold로 3D 구조 예측 → Biopython·ProPKa·ESM-2 기반 속성 계산 및 바인딩 유사도 컨텍스트 생성  
4. **가설 생성**: 위 context를 모두 모아 LLM에 넣고, “타겟 결합 가능성 + 기판 호환성”에 대한 **가설**과 검증 제안(MD/도킹/실험 등)을 생성

---

## 시스템 요구사항

- **Python**: 3.10+
- **GPU**: CUDA 지원 GPU (ProtGPT2, ESMFold, ESM-2 임베딩 사용)
- **외부 도구**: **ProPKa 3** (`propka3` 명령어로 PATH에 설치되어 있어야 함)
- **네트워크**: PubChem API 접근, OpenAI API 호출, Hugging Face 모델 다운로드

---

## 설치

1. 저장소 클론 후 의존성 설치:

```bash
cd PSHGagent
pip install -r requirements.txt
```

2. **추가 패키지** (requirements.txt에 없을 수 있음):

```bash
pip install langgraph python-dotenv openai pubchempy
```

3. **ProPKa 3** 설치 및 `propka3` PATH 설정  
   - [ProPKa](https://github.com/jensengroup/propka) 참고

4. **OpenAI API 키** 설정 (가설 생성용):

```bash
# .env 파일 생성
echo "OPENAI_API_KEY=your-api-key" > .env
```

---

## 데이터 파일

| 파일 | 설명 |
|------|------|
| `target_mapping.json` | 타겟 이름 → TargetID 매핑 (MiMoSet 기준). 바인딩 유사도 컨텍스트에 사용 |
| `mimoset.csv` | 타겟별 참조 펩타이드 서열. ESM-2 임베딩 유사도 계산 시 참조 |

- **기판/타겟 이름**: PubChem에서 검색 가능한 영문 이름 사용 (예: `SiO2`, `biotin`).
- **바인딩 유사도**: 사용하는 타겟이 `target_mapping.json` 및 `mimoset.csv`에 있어야 정상 동작합니다.

---

## 사용법

### 프롬프트 형식

다음 항목을 포함한 한 줄(또는 여러 줄) 텍스트를 입력합니다. 키워드는 대소문자 구분 없이 매칭됩니다.

| 항목 | 예시 키워드 | 예시 값 |
|------|-------------|---------|
| 기판 물질 | `기판 물질`, `substrate` | `SiO2` |
| 타겟 물질 | `target 물질`, `타겟 물질`, `target` | `biotin` |
| 펩타이드 길이 | `펩타이드 길이`, `length` | `20` |
| 생성 샘플링 횟수 | `생성 샘플링 횟수`, `num_samples`, `samples` | `3` |
| 시드 | `seed` (선택, 기본 42) | `7` |

**예시 프롬프트:**

```
기판 물질: SiO2, target 물질: biotin, 펩타이드 길이: 20, 생성 샘플링 횟수: 3, seed: 7
```

### 로컬 실행 (대화형)

프롬프트를 한 줄로 입력하면 됩니다.

```bash
python -m agents.pshg.graph
# 프롬프트 입력 (예: 기판 물질: SiO2, target 물질: biotin, ...)
```

### 파이프 입력

```bash
echo "기판 물질: SiO2, target 물질: biotin, 펩타이드 길이: 20, 생성 샘플링 횟수: 3, seed: 7" | python -m agents.pshg.graph
```

### SLURM 배치 실행 (GPU 클러스터)

제출 스크립트 예시는 `test.sh`입니다. **반드시 프로젝트 루트(PSHGagent)에서** `sbatch test.sh`로 제출하세요.

```bash
cd /path/to/PSHGagent

# 기본 프롬프트 사용 (스크립트 내 기본값)
sbatch test.sh

# 다른 프롬프트로 실행
PSHG_PROMPT="기판 물질: pentacene, target 물질: biotin, 펩타이드 길이: 15, 생성 샘플링 횟수: 2, seed: 42" sbatch test.sh
```

- `test.sh`는 `SLURM_SUBMIT_DIR`로 작업 디렉터리를 맞추므로, 제출 디렉터리가 `PSHGagent`여야 합니다.
- 스크립트가 Windows에서 편집되었다면 줄바꿈을 Unix 형식으로 바꿔야 합니다:  
  `sed -i 's/\r$//' test.sh` 또는 `dos2unix test.sh`

---

## 파이프라인 구조

```
parse → chem_context → generate → analyze → hypothesis → END
```

| 단계 | 노드 | 설명 |
|------|------|------|
| 1 | `parse` | 사용자 프롬프트에서 substrate, target, length, num_samples, seed 파싱. `run_dir` 생성 (`runs/{target}_{substrate}_L{length}_N{num}_seed{seed}`) |
| 2 | `chem_context` | ChemicalRetriever로 기판/타겟 PubChem 조회 → RAG용 `chemical_context` 문자열 생성 |
| 3 | `generate` | ProtGPT2로 펩타이드 서열 샘플링 → `run_dir/output.csv` 저장, state에 `sequences` 저장 |
| 4 | `analyze` | 각 서열에 대해: ESMFold 구조 예측 → PDB 저장, Biopython 서열/구조 속성, ProPKa 전하·pKa, ESM-2 바인딩 유사도 컨텍스트 계산 → `candidates` 리스트 구성 |
| 5 | `hypothesis` | 각 후보별로 context를 모아 `build_llm_prompt`로 프롬프트 구성 → OpenAI API로 가설 생성 → `hypotheses` 리스트 반환 |

State는 `agents/pshg/state.py`의 `PSHGState`에 정의되어 있으며, `user_prompt`, `cfg`, `run_dir`, `chemical_context`, `sequences`, `candidates`, `hypotheses` 등만 그래프 전체에서 공유합니다.

---

## 가설 생성 시 주입되는 Context

가설 생성 단계에서는 **아래 순서**로 블록이 하나의 유저 메시지로 이어져 LLM에 전달됩니다. (코드: `agents/pshg/chains.py` → `build_llm_prompt`)

1. **\[Candidate]**  
   - 해당 후보의 `sequence`, `pdb_path`

2. **chemical_context**  
   - 기판·타겟에 대한 PubChem RAG 텍스트 (분자명, CID, IUPAC, 분자식, SMILES, 분자량, LogP 등)

3. **\[Sequence Properties]** (있을 경우)  
   - Biopython `ProteinAnalysis`: 서열, 분자량, 아미노산 조성, pI, aromaticity, instability index, GRAVY

4. **\[Structure Properties]** (있을 경우)  
   - PDB 기반 전역 물성: Rg, Dmax, asphericity, SASA·양/음전하 잔기 SASA 비율, CA contact pairs, contact density, SASA 파라미터 등

5. **propka_context** (있을 경우)  
   - ProPKa 결과: pI (folded/unfolded), pH 7.4에서 Q, sensitive_pKa 등

6. **binding_context** (있을 경우)  
   - ESM-2 임베딩 기반: 타겟명·TargetID, top-k 참조 서열과의 코사인 유사도 점수 및 바인딩 점수

7. **Task 문구**  
   - “위 context를 바탕으로, 이 펩타이드가 타겟에 결합할 가능성과 기판과의 호환성에 대한 **가설**을 신중하게 제안하고, MD/도킹/실험 등 검증 방안을 제시하라”는 지시

---

## 모듈 설명

### 에이전트 코어 (`agents/pshg/`)

| 파일 | 역할 |
|------|------|
| `state.py` | `AgentInput`, `PSHGState` 정의 (파싱 설정, run_dir, chemical_context, sequences, candidates, hypotheses) |
| `graph.py` | LangGraph `StateGraph` 구성 및 CLI 진입점 (`python -m agents.pshg.graph`) |
| `nodes.py` | 파이프라인 노드: parse, chem_context, generate, analyze, hypothesis |
| `chains.py` | `build_llm_prompt`(후보 + chemical_context → 프롬프트 문자열), `call_hypothesis_llm`(OpenAI API 호출) |
| `tools.py` | `parse_prompt`, `load_target_mapping`, `get_target_id`, `extract_reference_seqs_from_mimoset`, `compute_binding_similarity_context` |

### 화학·서열·구조·분석

| 파일 | 주요 함수/클래스 | 설명 |
|------|------------------|------|
| `chemical_retriever.py` | `ChemicalRetriever.search_molecule`, `generate_rag_context` | PubChem API로 분자 검색, RAG용 텍스트 생성 |
| `protGPT2_inference.py` | `inference(...)` | ProtGPT2로 펩타이드 서열 생성, CSV 저장 |
| `esm_inference.py` | `inference(sequence, out_file_path_name)` | ESMFold로 3D 구조 예측, PDB 저장 |
| `esm_embedding.py` | `esm2_embed(sequences, pooling)` | ESM-2 임베딩 (바인딩 유사도용) |
| `biopython_analysis_function.py` | `sequence_analysis(seq)`, `compute_global_properties_from_pdb(pdb_path)` | 서열 물성(분자량, pI, GRAVY 등), PDB 기반 형태·SASA·접촉 등 |
| `propka_context_generation.py` | `run_propka(pdb_path)`, `get_propka_context(propka_path)` | ProPKa 실행, .pka → LLM용 전기적 특성 컨텍스트 |
| `binding_context_retriever.py` | `get_target_mapping`, `get_target_id`, `get_binding_context` | 타겟 매핑 및 (레거시) 바인딩 컨텍스트; 파이프라인에서는 `agents/pshg/tools.py`의 `compute_binding_similarity_context` 사용 |

### `sequence_analysis` 반환값

- Sequence, molecular weight, amino acid composition, **pI**, **aromaticity**, **instability index**, **GRAVY**

### `compute_global_properties_from_pdb` 반환값

- `n_residues`, `n_atoms`
- **shape**: `Rg_Angstrom`, `Dmax_Angstrom`, `asphericity`
- **exposure**: `total_SASA_A2`, `pos_residue_SASA_fraction`, `neg_residue_SASA_fraction`
- **contacts**: `CA_contact_pairs_within_cutoff`, `CA_contact_cutoff_A`, `seq_sep_exclude`, `contact_density_pairs_per_residue`
- **sasa_params**: `probe_radius_A`, `n_points`

### ProPKa context 예시

- pI (folded/unfolded), Q at pH 7.4 (folded/unfolded), **sensitive_pKa** (리간드 결합 등으로 전하 변화 가능 잔기)  
- `[ProPKa Context]` 아래 JSON

### Binding similarity context (ESM-2)

- Target 이름, TargetID  
- top-k 참조 서열과 생성 서열 간 코사인 유사도 점수  
- “binding plausibility evidence” 설명 문구

---

## 환경 변수

| 변수 | 용도 |
|------|------|
| `OPENAI_API_KEY` | 가설 생성용 OpenAI API 키 (`.env` 또는 환경에 설정) |
| `PSHG_PROMPT` | `test.sh`에서 기본 프롬프트 대신 사용할 문자열 (선택) |

---

## 출력 구조

- **실행별 디렉터리**: `runs/{target}_{substrate}_L{length}_N{num}_seed{seed}/`
  - `output.csv`: ProtGPT2 생성 서열 목록
  - `cand_001.pdb`, `cand_002.pdb`, ... : ESMFold 예측 구조
  - 해당 디렉터리 또는 작업 디렉터리에 `cand_*.pka` (ProPKa 결과)가 생성될 수 있음

- **State 출력**: `run_dir`, `candidates`, `hypotheses`  
  - `hypotheses`: 각 후보에 대한 `{ "rank": i, "hypothesis": "..." }` 또는 에러 시 `{ "rank": i, "error": "..." }`

- **SLURM**: `sbatch test.sh` 사용 시 `pshg_%j.out`, `pshg_%j.err`에 로그가 기록됩니다.

---

## 요약

- **PSHGagent**는 “기판 + 타겟” 조건에 맞는 펩타이드를 ProtGPT2로 생성하고, ESMFold·Biopython·ProPKa·ESM-2로 분석한 뒤, 그 결과를 **하나의 긴 context**로 모아 LLM이 **결합 가능성 가설**을 쓰도록 유도합니다.
- 코드 수정 없이 사용하려면: 올바른 프롬프트 형식, `target_mapping.json`/`mimoset.csv`, ProPKa 설치, `OPENAI_API_KEY` 설정만 확인하면 됩니다.
