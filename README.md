# Peptide-Design-Hypothesis-Generation-LLM-Agent-for-Semiconductor-application

# protGPT2_inference.py

* inference 함수 사용
* sequence_length: 원하는 아미노산 서열의 길이
* num_sequences: 샘플링 횟수
* seed: 랜덤 시드
* output.csv: 샘플링된 아미노산 서열 목록 저장 


# esm_inference.py
* inference 함수 사용
* sequence: 입력 서열
* out_file_path_name: 결과 pdb파일 이름 지정
* 결과 구조 정보는 pdb 파일로 저장


# biopython_analysis_function.py
* 두 함수를 사용
    * sequence_analysis: 서열을 입력받아서 여러 물성 계산
    * compute_global_properties_from_pdb: pdb파일을 입력받아서 여러 물성 계산
* sequence_analysis 반환하는 값
    * 분자량
    * 서열 내 아미노산 조성 비율
    * 등전점(pI)
    * Aromaticity
    * Instability
    * GRAVY
* compute_global_properties_from_pdb 반환하는 값

```json
{
        "n_residues": n_res,   # 아미노산 개수
        "n_atoms": len(atoms),  # 총 원자 수
        "shape": {
            "Rg_Angstrom": rg,   # 구조가 얼마나 퍼져있는가 (작음: compact, 큼: 늘어짐/펼쳐짐)
            "Dmax_Angstrom": dmax, # 구조 내 가장 먼 두 원자 사이 거리(얼마나 구조가 길쭉한가)
            "asphericity": asp,   # 0에 가까울수록 구형
        },
        "exposure": {
            "total_SASA_A2": total_sasa,   # 펩타이드의 전체 용매 노출 면적
            "pos_residue_SASA_fraction": pos_frac,   # 양전하 아미노산의 노출 면적 비율
            "neg_residue_SASA_fraction": neg_frac,      #음전하 아미노산의 노출 면적 비율
        },
        "contacts": {
            "CA_contact_pairs_within_cutoff": contact_pairs,   # 거리기준 내 접촉쌍 개수
            "CA_contact_cutoff_A": ca_contact_cutoff_A,   # 잡은 거리 기준
            "seq_sep_exclude": seq_sep_exclude,   # 서열자체가 가까운 경우는 카운트 x를 위한 제외할 서열 내 거리 기준
            "contact_density_pairs_per_residue": contact_density,   # 잔기 하나당 내부 접촉 횟수
        },      # contact_density가 높을수록 내부적으로 뭉치는 구조
        "sasa_params": {
            "probe_radius_A": probe_radius, # SASA 계산할 때 용매가 이 정도의 크기를 가진다고 가정
            "n_points": n_points,   # 원자 표면에 뿌리는 용매 분자의 개수를 가정
        },
    }






