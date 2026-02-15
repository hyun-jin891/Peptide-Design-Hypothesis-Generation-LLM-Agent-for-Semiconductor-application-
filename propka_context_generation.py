import subprocess
import re
import json
from pathlib import Path

def run_propka(pdb_path):
  subprocess.run(f"propka3 {pdb_path}", shell=True)

def load_propka_res(propka_path):
  text = Path(propka_path).read_text(errors="ignore")
  return text

def propka_stdout_to_minjson(propka_text: str, operating_pH: float = 7.4, window: float = 1.0):
  m_pi = re.search(r"The pI is\s+([0-9]+\.[0-9]+)\s+\(folded\)\s+and\s+([0-9]+\.[0-9]+)\s+\(unfolded\)", propka_text)
  pI_folded = float(m_pi.group(1)) if m_pi else None
  pI_unfolded = float(m_pi.group(2)) if m_pi else None
  
  ph_pat = rf"^\s*{operating_pH:.2f}\s+([-\d\.]+)\s+([-\d\.]+)\s*$"
  m_q = re.search(ph_pat, propka_text, flags=re.MULTILINE)
  q_unfolded = float(m_q.group(1)) if m_q else None
  q_folded   = float(m_q.group(2)) if m_q else None
  
  residues = []
  m_block = re.search(r"SUMMARY OF THIS PREDICTION(.*?)^-{10,}\s*$", propka_text, flags=re.DOTALL | re.MULTILINE)
  
  if m_block:
    for line in m_block.group(1).splitlines():
      m = re.match(r"^\s*([A-Z0-9\+\-]{2,3})\s+(\d+)\s+([A-Z])\s+([-\d\.]+)\s+", line)
      if m:
        res, rid, chain, pka = m.group(1), int(m.group(2)), m.group(3), float(m.group(4))
        residues.append({"res": res, "id": rid, "chain": chain, "pKa": pka})
        
  sensitive_pKa = [r for r in residues if r["pKa"] is not None and (operating_pH - window) <= r["pKa"] <= (operating_pH + window)]
  
  return {
        "propka_res": {
            "pI_folded": pI_folded,
            "pI_unfoled": pI_unfolded,
            f"Q_folded_pH{operating_pH}": q_folded,
            f"Q_unfolded_pH{operating_pH}": q_unfolded,
            "sensitive_pKa": sensitive_pKa
        }
    }

def get_propka_context(propka_path, operating_pH=7.4):
  text = load_propka_res(propka_path)
  
  propka_res = propka_stdout_to_minjson(text, operating_pH)
  
  context = f"""The context below represents the electrical property of generated peptide sequence measured by ProPKa tools
  
  pI_folded: pI value of folded peptide
  pI_unfoled: pI value of unfolded peptide
  Q_foled_pH{operating_pH}: charge of folded peptide at pH{operating_pH}
  Q_unfoled_pH{operating_pH}: charge of unfolded peptide at pH{operating_pH}
  sensitive_pKa: Residue that can induce charge change for various reason like ligand binding
  
  [ProPKa Context]
  {json.dumps(propka_res, ensure_ascii=False)}
  
  """
  
  return context
  
  
  
if __name__ == "__main__":
    print(get_propka_context("sample_structure.pka"))  
  
  
  
  
  
  
  
  
  
  
  