"""
FastAPI backend for Self-Healing Molecule Generator
Base model: ncfrey/ChemGPT-4.7M + LoRA adapter (PEFT)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

app = FastAPI(title="Self-Healing Molecule Generator")

# Allow your frontend to connect (adjust ports as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model at startup ──────────────────────────────────────────────
print("Loading base model: ncfrey/ChemGPT-4.7M ...")
base_model = AutoModelForCausalLM.from_pretrained("ncfrey/ChemGPT-4.7M")

print("Loading LoRA adapter ...")
model = PeftModel.from_pretrained(base_model, "./model")
model.eval()

print("Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained("./model")

# Set pad token if not defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("✅ Model ready!")


# ── Request/Response schemas ───────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str = ""              # Starting SMILES fragment or empty for unconditional
    max_length: int = 128         # Max tokens to generate
    temperature: float = 1.0      # Higher = more creative/diverse molecules
    top_k: int = 50               # Top-k sampling
    top_p: float = 0.95           # Nucleus sampling
    num_molecules: int = 1        # How many molecules to generate


class GenerateResponse(BaseModel):
    molecules: list[str]
    prompt_used: str


# ── Endpoints ──────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": "ChemGPT-4.7M + LoRA (self-healing)"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    try:
        # Tokenize input (or start from scratch if no prompt)
        if req.prompt.strip():
            inputs = tokenizer(req.prompt, return_tensors="pt")
        else:
            # Unconditional generation — start with BOS or pad token
            bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.pad_token_id
            inputs = {"input_ids": torch.tensor([[bos]])}

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                max_length=req.max_length,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                do_sample=True,
                num_return_sequences=req.num_molecules,
                pad_token_id=tokenizer.pad_token_id,
            )

        molecules = [
            tokenizer.decode(seq, skip_special_tokens=True).strip()
            for seq in outputs
        ]

        return GenerateResponse(
            molecules=molecules,
            prompt_used=req.prompt or "(unconditional)",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)