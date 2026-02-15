import torch
from transformers import AutoTokenizer, AutoModel



@torch.no_grad()
def esm2_embed(
    sequences,
    pooling: str = "mean",   # "mean" | "cls" | "none"
):
  
  MODEL_NAME = "facebook/esm2_t48_15B_UR50D" 
  device = "cuda" if torch.cuda.is_available() else "cpu"

  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
  model = AutoModel.from_pretrained(MODEL_NAME).to(device)
  model.eval()
  
  
  if isinstance(sequences, str):
        sequences = [sequences]
  
  seqs_for_tokenizer = sequences
  
  enc = tokenizer(
        seqs_for_tokenizer,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)
  
  out = model(**enc, output_hidden_states=False, return_dict=True)
  h = out.last_hidden_state
  
  mask = enc["attention_mask"].bool()
  
  residue_embeddings = []
  seq_embeddings = []
  
  for i in range(h.size(0)):
    hi = h[i]              
    mi = mask[i]
    
    valid = hi[mi]
    if valid.size(0) >= 3:
      residues = valid[1:-1]
    else:
      residues = valid
    
    residue_embeddings.append(residues.cpu())
    
    if pooling == "mean":
      if residues.numel() == 0:
        seq_embeddings.append(torch.zeros(h.size(-1), device="cpu"))
      else:
        seq_embeddings.append(residues.mean(dim=0).cpu())
    
    elif pooling == "cls":
      seq_embeddings.append(valid[0].cpu())
    elif pooling == "none":
      pass
    else:
      raise ValueError("pooling must be one of: 'mean', 'cls', 'none'")
    
  
  if pooling == "none":
    seq_embeddings_tensor = None
  else:
    seq_embeddings_tensor = torch.stack(seq_embeddings, dim=0)  # [B, D]

  return {
        "residue_embeddings": residue_embeddings,
        "sequence_embeddings": seq_embeddings_tensor                    # residue_embeddings: Embedding of each residue
                                                                        # sequence_embeddings: CLS token embedding or Mean pooled embedding
        }
    
    
    
if __name__ == "__main__":
    seqs = [
        "MKTFFVLLLFLTLATYAFSPVQA",
        "GHHHHHHSSGLVPRGSHMASMTGGQQMGRGSEF",
    ]

    embs = esm2_embed(seqs, pooling="mean")
    print("Batch sequence embeddings:", embs["sequence_embeddings"].shape)  
    print("First residue embedding:", embs["residue_embeddings"][0].shape)  
    
    
    
    
    
    
    
    
    
        