from esm_embedding import esm2_embed
import xml.etree.ElementTree as ET
import json
import pandas as pd
import re
import torch
import torch.nn.functional as F


def get_all_records(xml_path="target.xml"):
  tree = ET.parse(xml_path)
  root = tree.getroot()

  mapping = {}
  
  for record in root.findall("RECORD"):
    target_id = record.findtext("TargetID")
    target_name = record.findtext("TargetName")

    if target_name:
      mapping[target_name.strip()] = target_id.strip()
  
  return mapping

def get_target_mapping():
  with open("target_mapping.json", "r", encoding="utf-8") as f:
        return json.load(f)


def get_target_id(target_name, mapping, case_insensitive=True):
  if case_insensitive:
    lowered = {k.lower(): v for k, v in mapping.items()}
    return lowered.get(target_name.lower(), None)
  else:
    return mapping.get(target_name, None)


def get_binding_context(target_name, target_id, generated_seq):
  if target_id == None:
    return
  
  id = int(target_id)
  df = pd.read_csv("mimoset.csv")
  
  input_seqs = df[df["TargetID"] == id]["Sequences"].item()
  input_seqs = re.findall(r'^[A-Z]+', input_seqs, re.MULTILINE)
  
  input_seqs.append(generated_seq)
  
  seq_embeddings = esm2_embed(input_seqs)["sequence_embeddings"]
  
  ref_seq = input_seqs[-1]
  ref_emb = seq_embeddings[-1]
  
  other_seqs = input_seqs[:-1]
  other_embs = seq_embeddings[:-1]
  
  cos = F.cosine_similarity(other_embs, ref_emb.unsqueeze(0), dim=1, eps=1e-8)
  
  sims = {seq: float(score) for seq, score in zip(other_seqs, cos.cpu())}
  
  context = f"""The context below explains whether the generated amino acid sequence is similar to other sequences that can bind to the target.
  
  [Target]
  {target_name}

  
  [Similarity context]
  {json.dumps(sims, ensure_ascii=False)}
  """
  
  return context
  
  


if __name__ == "__main__":
    mapping = get_target_mapping()
    
    id = get_target_id("biotin", mapping)
    print(get_binding_context("biotin", id, "MTAAADEVRHRDDSIAQDEL"))

