from __future__ import annotations
from Bio.SeqUtils.ProtParam import ProteinAnalysis


from typing import Optional, Dict, Any, Tuple, List
import numpy as np

from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.SASA import ShrakeRupley

def sequence_analysis(seq):
  pa = ProteinAnalysis(seq)
  molecular_weight = pa.molecular_weight()
  fractions = pa.get_amino_acids_percent()
  pI = pa.isoelectric_point()
  arom = pa.aromaticity()
  instab = pa.instability_index()
  gravy = pa.gravy()
  
  return {"Sequence":seq, "molecular weight":molecular_weight, "amino acid composition":fractions, "pI":pI, "aromaticity":arom, "instability":instab, "GRAVY":gravy}
  
  
def _get_model(structure, model_id: int = 0):
    return structure[model_id]


def _protein_atoms_and_residues(model, chain_id: Optional[str] = None):
    atoms = []
    residues = []
    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for res in chain:
            if is_aa(res, standard=False):
                residues.append(res)
                atoms.extend(list(res.get_atoms()))
    return atoms, residues


def _coords(atoms) -> np.ndarray:
    return np.asarray([a.coord for a in atoms], dtype=float)


def radius_of_gyration(coords: np.ndarray) -> float:
    com = coords.mean(axis=0)
    rg2 = np.mean(np.sum((coords - com) ** 2, axis=1))
    return float(np.sqrt(rg2))


def max_pairwise_distance(coords: np.ndarray) -> float:
    n = coords.shape[0]
    if n <= 1:
        return 0.0
    dmax = 0.0
    for i in range(n - 1):
        diff = coords[i + 1 :] - coords[i]
        dist = np.sqrt(np.sum(diff * diff, axis=1)).max()
        if dist > dmax:
            dmax = float(dist)
    return dmax


def asphericity(coords: np.ndarray) -> float:
    x = coords - coords.mean(axis=0)
    cov = np.cov(x.T)
    evals = np.linalg.eigvalsh(cov)
    l1, l2, l3 = np.sort(evals)  # l1 <= l2 <= l3
    s = float(l1 + l2 + l3)
    if s <= 0.0:
        return 0.0
    return float((l3 - 0.5 * (l1 + l2)) / s)


def compute_global_properties_from_pdb(
    pdb_path: str,
    chain_id: Optional[str] = None,
    model_id: int = 0,
    probe_radius: float = 1.4,
    n_points: int = 100,
    ca_contact_cutoff_A: float = 8.0,
    seq_sep_exclude: int = 2,
) -> Dict[str, Any]:

    structure = PDBParser(QUIET=True).get_structure("x", pdb_path)
    model = _get_model(structure, model_id=model_id)

    atoms, residues = _protein_atoms_and_residues(model, chain_id=chain_id)
    if not atoms:
        raise ValueError("No protein atoms found. Check chain_id/model_id or PDB content.")

    coords = _coords(atoms)


    rg = radius_of_gyration(coords)
    dmax = max_pairwise_distance(coords)
    asp = asphericity(coords)


    sr = ShrakeRupley(probe_radius=probe_radius, n_points=n_points)
    sr.compute(structure, level="A")

    total_sasa = 0.0


    POS_RES = {"LYS", "ARG", "HIS"}
    NEG_RES = {"ASP", "GLU"}
    pos_sasa = 0.0
    neg_sasa = 0.0

    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for res in chain:
            if not is_aa(res, standard=False):
                continue

            res_sasa = 0.0
            for a in res.get_atoms():
                s = getattr(a, "sasa", None)
                if s is None:
                    continue
                res_sasa += float(s)

            total_sasa += res_sasa
            if res.resname in POS_RES:
                pos_sasa += res_sasa
            elif res.resname in NEG_RES:
                neg_sasa += res_sasa

    pos_frac = (pos_sasa / total_sasa) if total_sasa > 0 else 0.0
    neg_frac = (neg_sasa / total_sasa) if total_sasa > 0 else 0.0


    ca_atoms: List[Any] = []
    ca_keys: List[Tuple[str, int, str]] = []

    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for res in chain:
            if not is_aa(res, standard=False):
                continue
            if "CA" not in res:
                continue
            het, resseq, icode = res.id
            ca_atoms.append(res["CA"])
            ca_keys.append((chain.id, int(resseq), str(icode).strip()))

    contact_pairs = 0
    if len(ca_atoms) >= 2:
        ns = NeighborSearch(ca_atoms)

        atom2meta = {}
        for i, (a, key) in enumerate(zip(ca_atoms, ca_keys)):
            atom2meta[id(a)] = (key[0], i)

        seen = set()
        for a in ca_atoms:
            chain_a, idx_a = atom2meta[id(a)]
            for b in ns.search(a.coord, ca_contact_cutoff_A, level="A"):
                if a is b:
                    continue
                chain_b, idx_b = atom2meta[id(b)]

                if chain_a == chain_b and abs(idx_a - idx_b) <= seq_sep_exclude:
                    continue

                pair = tuple(sorted((id(a), id(b))))
                seen.add(pair)

        contact_pairs = len(seen)

    n_res = len(residues)
    contact_density = contact_pairs / max(n_res, 1)

    return {
        "n_residues": n_res,
        "n_atoms": len(atoms),
        "shape": {
            "Rg_Angstrom": rg,
            "Dmax_Angstrom": dmax,
            "asphericity": asp,
        },
        "exposure": {
            "total_SASA_A2": total_sasa,
            "pos_residue_SASA_fraction": pos_frac,
            "neg_residue_SASA_fraction": neg_frac,
        },
        "contacts": {
            "CA_contact_pairs_within_cutoff": contact_pairs,
            "CA_contact_cutoff_A": ca_contact_cutoff_A,
            "seq_sep_exclude": seq_sep_exclude,
            "contact_density_pairs_per_residue": contact_density,
        },
        "sasa_params": {
            "probe_radius_A": probe_radius,
            "n_points": n_points,
        },
    }


def main():
  sequence = "MTAAADEVRHRDDSIAQDEL"
  print(sequence_analysis(sequence))
  
  print(compute_global_properties_from_pdb("sample_structure.pdb"))





if __name__ == "__main__":
  main()













  
  
  