"""
FastAPI backend for Self-Healing Molecule Generator
Base model: ncfrey/ChemGPT-4.7M + LoRA adapter (PEFT)

Metric calculation matches SelfHealing_MotifEval.ipynb exactly.
Requires: selfies==1.0.3 (v2.x breaks ChemGPT vocab)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
import torch
import numpy as np
import selfies as sf
import os
import sys

# ── Verify SELFIES version ────────────────────────────────────────────
assert sf.__version__.startswith('1.0'), (
    f'SELFIES v2.x detected ({sf.__version__}) — breaks ChemGPT vocab. '
    f'Run: pip install selfies==1.0.3'
)
print(f'selfies version OK: {sf.__version__}')

# ── RDKit imports ──────────────────────────────────────────────────────
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs, rdMolDescriptors

# ── SA Scorer setup (same as Colab Section B) ─────────────────────────
SA_SCORER_AVAILABLE = False
try:
    from rdkit.Chem import RDConfig
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
    SA_SCORER_AVAILABLE = True
    print("SA scorer loaded via RDConfig.")
except Exception as e1:
    print(f"SA scorer unavailable ({e1}) — SA metrics will be N/A")

# ── Self-healing SMARTS patterns (Colab Section D) ────────────────────
SH_SMARTS = {
    'disulfide':     '[#16X2]-[#16X2]',       # S-S dynamic covalent — most important
    'urea':          '[NX3]C(=O)[NX3]',        # H-bond donor+acceptor pair — most important
    'amide_hbond':   '[NX3;H1,H2]C(=O)',       # amide H-bond donor
    'furan':         'c1ccoc1',                 # Diels-Alder partner
    'imine':         '[CX3]=[NX2]',            # dynamic covalent
    'boronic_ester': 'B(O)O',                   # dynamic covalent
    'da_adduct':     'C1CC2CCC1O2',            # DA oxanorbornene adduct
}

SH_PATTERNS = {}
for name, smarts in SH_SMARTS.items():
    pat = Chem.MolFromSmarts(smarts)
    if pat:
        SH_PATTERNS[name] = pat
        print(f"  {name:15s}: {smarts}")
    else:
        print(f"  WARNING: invalid SMARTS for {name}")

print(f"{len(SH_PATTERNS)} motif patterns compiled.")


# ── FastAPI app ────────────────────────────────────────────────────────
app = FastAPI(title="Self-Healing Molecule Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model at startup ─────────────────────────────────────────────
MODEL_NAME = "ncfrey/ChemGPT-4.7M"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")
print(f"Loading base model: {MODEL_NAME} ...")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

print("Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained("./model")

# Resize embeddings if tokenizer vocab != model vocab (same as Colab Section G)
if len(tokenizer) != base_model.config.vocab_size:
    base_model.resize_token_embeddings(len(tokenizer))

print("Loading LoRA adapter ...")
model = PeftModel.from_pretrained(base_model, "./model").merge_and_unload().to(device)
del base_model  # free memory
model.eval()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Vocab size: {len(tokenizer)}")
print("✅ Model ready!")


# ── Generation (matches Colab Section E: generate_polymer_smiles) ─────
def generate_molecules(
    seed_smiles: str,
    n: int = 3,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
) -> list[dict]:
    """
    Generate n SMILES continuations from a seed.
    Strips seed tokens before decoding — metrics measure only model-generated content.
    Matches Colab generate_polymer_smiles() exactly.
    """
    # Encode seed: replace * with [Xe], convert to SELFIES
    seed_clean = seed_smiles.replace('*', '[Xe]')
    selfies_seed = sf.encoder(seed_clean)
    if selfies_seed is None:
        raise ValueError(f'Could not encode seed "{seed_smiles}" to SELFIES')

    input_ids = tokenizer.encode(selfies_seed, return_tensors='pt').to(device)
    attention_mask = torch.ones_like(input_ids)
    seed_len = input_ids.shape[1]

    gen_config = GenerationConfig(
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
    )

    results = []
    with torch.no_grad():
        for _ in range(n):
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                generation_config=gen_config,
            )
            # Strip seed tokens (same as Colab line 217)
            generated_ids = output[0][seed_len:]
            raw = tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Strip spaces before sf.decoder (same as Colab line 219)
            selfies_out = raw.replace(' ', '')

            smiles_out = None
            try:
                smiles_out = sf.decoder(selfies_out)
                if smiles_out:
                    smiles_out = smiles_out.replace('[Xe]', '*')
            except Exception:
                pass

            results.append({
                "selfies": raw,
                "smiles": smiles_out,
            })

    return results


# ── Metric calculation (matches Colab Section E: compute_metrics_with_motifs) ──
def compute_metrics_with_motifs(smiles_list: list) -> dict:
    """
    Extended metric computation: polymer-ness + self-healing motifs + SA score.
    Matches Colab compute_metrics_with_motifs() exactly.
    """
    total = len([s for s in smiles_list if s is not None])
    if total == 0:
        base = {
            'validity_%': 0, 'continuation_polymer_%': 0,
            'uniqueness_%': 0, 'diversity': 'N/A',
            'self_healing_motif_%': 0, 'strict_sh_%': 0,
            'mean_sa_score': 'N/A', 'synthesisable_%': 'N/A',
            'n_valid': 0, 'n_total': 0,
        }
        return {**base, **{f'{k}_%': 0 for k in SH_PATTERNS}}

    valid_smiles, valid_mols = [], []
    for s in smiles_list:
        if s is None:
            continue
        mol = Chem.MolFromSmiles(s.replace('*', '[Xe]'), sanitize=False)
        if mol:
            try:
                mol.UpdatePropertyCache(strict=False)
            except Exception:
                pass
            canon = Chem.MolToSmiles(mol).replace('[Xe]', '*')
            valid_smiles.append(canon)
            valid_mols.append(mol)

    n_valid = len(valid_smiles)
    validity_pct = 100 * n_valid / total

    # Continuation polymer %
    polymer_valid = [s for s in valid_smiles if '*' in s]
    continuation_polymer_pct = 100 * len(polymer_valid) / max(n_valid, 1)

    uniqueness_pct = 100 * len(set(valid_smiles)) / max(n_valid, 1)

    # Diversity — Tanimoto dissimilarity on Morgan FP
    diversity = None
    if len(valid_mols) >= 2:
        fps = []
        for m in valid_mols:
            try:
                Chem.FastFindRings(m)
                fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)
                fps.append(fp)
            except Exception:
                pass
        if len(fps) >= 2:
            sims = [
                DataStructs.TanimotoSimilarity(fps[i], fps[j])
                for i in range(len(fps))
                for j in range(i + 1, min(len(fps), i + 20))
            ]
            diversity = round(1.0 - float(np.mean(sims)), 4)

    # Self-healing motif detection
    motif_counts = {k: 0 for k in SH_PATTERNS}
    any_motif = 0
    strict_sh = 0
    for mol in valid_mols:
        mol_has_any = False
        mol_has_strict = False
        for name, pat in SH_PATTERNS.items():
            try:
                if mol.HasSubstructMatch(pat):
                    motif_counts[name] += 1
                    mol_has_any = True
                    if name in ('disulfide', 'urea'):
                        mol_has_strict = True
            except Exception:
                pass
        if mol_has_any:
            any_motif += 1
        if mol_has_strict:
            strict_sh += 1

    self_healing_pct = 100 * any_motif / max(n_valid, 1)
    per_motif_pcts = {
        f'{k}_%': round(100 * v / max(n_valid, 1), 1)
        for k, v in motif_counts.items()
    }

    # SA score
    sa_scores = []
    if SA_SCORER_AVAILABLE:
        for mol in valid_mols:
            try:
                Chem.FastFindRings(mol)
                score = sascorer.calculateScore(mol)
                sa_scores.append(score)
            except Exception:
                pass

    mean_sa = round(float(np.mean(sa_scores)), 3) if sa_scores else 'N/A'
    synth_pct = round(
        100 * sum(1 for s in sa_scores if s <= 6) / max(len(sa_scores), 1), 1
    ) if sa_scores else 'N/A'

    return {
        'validity_%': round(validity_pct, 1),
        'continuation_polymer_%': round(continuation_polymer_pct, 1),
        'uniqueness_%': round(uniqueness_pct, 1),
        'diversity': diversity if diversity is not None else 'N/A',
        'self_healing_motif_%': round(self_healing_pct, 1),
        'strict_sh_%': round(100 * strict_sh / max(n_valid, 1), 1),
        'mean_sa_score': mean_sa,
        'synthesisable_%': synth_pct,
        'n_valid': n_valid,
        'n_total': total,
        **per_motif_pcts,
    }


# ── Per-molecule metrics (for individual candidate cards) ─────────────
def compute_single_molecule_metrics(smiles: str) -> dict:
    """Compute metrics for a single molecule (for frontend display)."""
    if smiles is None:
        return {"valid": False}

    mol = Chem.MolFromSmiles(smiles.replace('*', '[Xe]'), sanitize=False)
    if mol is None:
        return {"valid": False}

    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass

    canon = Chem.MolToSmiles(mol).replace('[Xe]', '*')

    # Motif detection
    motifs_found = []
    has_strict_sh = False
    for name, pat in SH_PATTERNS.items():
        try:
            if mol.HasSubstructMatch(pat):
                motifs_found.append(name)
                if name in ('disulfide', 'urea'):
                    has_strict_sh = True
        except Exception:
            pass

    # SA score
    sa_score = None
    if SA_SCORER_AVAILABLE:
        try:
            Chem.FastFindRings(mol)
            sa_score = round(sascorer.calculateScore(mol), 3)
        except Exception:
            pass

    # Molecular descriptors
    try:
        Chem.FastFindRings(mol)
        num_rings = rdMolDescriptors.CalcNumRings(mol)
    except Exception:
        num_rings = 0

    return {
        "valid": True,
        "canonical_smiles": canon,
        "is_polymer": '*' in canon,
        "molecular_weight": round(Descriptors.MolWt(mol), 2),
        "num_heavy_atoms": mol.GetNumHeavyAtoms(),
        "num_rings": num_rings,
        "motifs_found": motifs_found,
        "has_any_sh_motif": len(motifs_found) > 0,
        "has_strict_sh": has_strict_sh,
        "sa_score": sa_score,
        "synthesisable": sa_score <= 6 if sa_score is not None else None,
    }


# ── Request schema ────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    seed: str = "*CC(=O)O*"       # Seed SMILES (use * for polymer endpoints)
    num_molecules: int = 3        # Number of candidates to generate
    max_new_tokens: int = 80      # Max new tokens (not total length)
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95


# ── Endpoints ─────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "ChemGPT-4.7M + LoRA (self-healing)",
        "device": device,
        "selfies_version": sf.__version__,
        "sa_scorer": SA_SCORER_AVAILABLE,
    }


@app.post("/generate")
def generate(req: GenerateRequest):
    try:
        # 1. Generate molecules (matches Colab generate_polymer_smiles)
        raw_results = generate_molecules(
            seed_smiles=req.seed,
            n=req.num_molecules,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
        )

        # 2. Compute per-molecule metrics (for candidate cards)
        molecules = []
        smiles_for_batch = []
        for r in raw_results:
            per_mol = compute_single_molecule_metrics(r["smiles"])
            molecules.append({
                "selfies": r["selfies"],
                "smiles": r["smiles"],
                **per_mol,
            })
            smiles_for_batch.append(r["smiles"])

        # 3. Compute batch metrics (matches Colab compute_metrics_with_motifs)
        batch_metrics = compute_metrics_with_motifs(smiles_for_batch)

        return {
            "seed": req.seed,
            "molecules": molecules,
            "batch_metrics": batch_metrics,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)