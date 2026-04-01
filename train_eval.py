"""
SyllabusQA Fine-tuning & Evaluation Pipeline — with RAG
Model  : Qwen2.5-3B-Instruct + QLoRA (4-bit)
GPU    : RTX 2080 Ti (11 GB VRAM)
Purpose: Analyze LLM limits on course-logistics QA using RAG retrieval
"""

import json
import os
import time
import warnings

warnings.filterwarnings("ignore")

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ─────────────────────────────────────────────
# Paths  (adjust BASE_DIR if needed)
# ─────────────────────────────────────────────
BASE_DIR    = Path(".")
DATA_DIR    = BASE_DIR / "data" / "dataset_split"
SYLLABI_DIR = BASE_DIR / "syllabi" / "syllabi_redacted" / "text"
OUTPUT_DIR  = BASE_DIR / "outputs" / datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SYLLABUS = "Syllabus_Marine_Microbiome_2022_redacted"
MODEL_NAME      = "Qwen/Qwen2.5-3B-Instruct"

# RAG settings
CHUNK_SIZE    = 400   # words per chunk
CHUNK_OVERLAP = 80    # overlapping words between adjacent chunks
TOP_K         = 3     # number of chunks to retrieve per question

print("=" * 60)
print("  SyllabusQA Pipeline  (Fine-tuning + RAG)")
print(f"  Model   : {MODEL_NAME}")
print(f"  Syllabus: {TARGET_SYLLABUS}")
print(f"  RAG     : chunk={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, top_k={TOP_K}")
print(f"  Output  : {OUTPUT_DIR}")
print("=" * 60)


# ─────────────────────────────────────────────
# 1. Data loading
# ─────────────────────────────────────────────

def load_syllabus_text(name: str) -> str:
    path = SYLLABI_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Syllabus not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_and_filter(name: str) -> dict:
    splits = {}
    for split in ["train", "val", "test"]:
        df = pd.read_csv(DATA_DIR / f"{split}.csv")
        filtered = df[df["syllabus_name"] == name].reset_index(drop=True)
        splits[split] = filtered
        print(f"  [{split}] total {len(df)} → filtered {len(filtered)}")
    return splits


# ─────────────────────────────────────────────
# 2. RAG — chunking & retrieval
# ─────────────────────────────────────────────

class SyllabusRAG:
    """
    Lightweight in-memory RAG for a single syllabus document.

    Steps
    -----
    1. Chunk   : split the full syllabus text into overlapping word-level chunks
    2. Embed   : encode every chunk with a small sentence-transformer (CPU)
    3. Retrieve: at query time embed the question, then return top-k chunks
                 by cosine similarity
    """

    def __init__(self, syllabus_text: str):
        from sentence_transformers import SentenceTransformer

        # 'all-MiniLM-L6-v2' is ~90 MB and runs well on CPU,
        # leaving the full GPU budget for the LLM.
        print("  [RAG] Loading embedding model (CPU)...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

        self.chunks = self._chunk(syllabus_text)
        print(f"  [RAG] {len(self.chunks)} chunks created")

        print("  [RAG] Encoding chunks...")
        self.chunk_embeddings = self.embedder.encode(
            self.chunks,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-normalise → dot == cosine similarity
        )

    def _chunk(self, text: str) -> list:
        """Sliding-window word-level chunking."""
        words  = text.split()
        step   = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
        chunks = []
        for start in range(0, len(words), step):
            chunk = " ".join(words[start : start + CHUNK_SIZE])
            if chunk:
                chunks.append(chunk)
        return chunks

    def retrieve(self, question: str, top_k: int = TOP_K) -> str:
        """Return top-k chunks most relevant to the question."""
        q_emb = self.embedder.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        scores  = (self.chunk_embeddings @ q_emb.T).squeeze()
        top_idx = np.argsort(scores)[::-1][:top_k]

        # Keep original document order so the context reads naturally
        retrieved = [self.chunks[i] for i in sorted(top_idx)]
        return "\n\n---\n\n".join(retrieved)


# ─────────────────────────────────────────────
# 3. Prompt builder
# ─────────────────────────────────────────────

def build_prompt(retrieved_context: str, question: str):
    """Build system/user messages from RAG-retrieved context only."""
    system_msg = (
        "You are a helpful teaching assistant. "
        "Answer the student's question using ONLY the syllabus excerpts provided. "
        "Be concise and accurate. "
        "If the answer cannot be found in the excerpts, "
        "say \"Not specified in the syllabus.\""
    )
    user_msg = (
        f"Relevant syllabus excerpts:\n"
        f"\"\"\"\n{retrieved_context}\n\"\"\"\n\n"
        f"Student question: {question}"
    )
    return system_msg, user_msg


def format_for_training(row, rag: SyllabusRAG, tokenizer) -> dict:
    """Format one training example as a complete chat string for SFT."""
    context              = rag.retrieve(row["question"])
    system_msg, user_msg = build_prompt(context, row["question"])

    messages = [
        {"role": "system",    "content": system_msg},
        {"role": "user",      "content": user_msg},
        {"role": "assistant", "content": str(row["answer"])},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


# ─────────────────────────────────────────────
# 4. Model & LoRA
# ─────────────────────────────────────────────

def load_model_and_tokenizer():
    from peft import (LoraConfig, get_peft_model,
                      prepare_model_for_kbit_training)
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    print("\n[1/5] Loading model with 4-bit quantization...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


# ─────────────────────────────────────────────
# 5. Training
# ─────────────────────────────────────────────

def train(model, tokenizer, splits, rag: SyllabusRAG):
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    print("\n[2/5] Preparing training data with RAG context...")

    def df_to_dataset(df):
        rows = [format_for_training(row, rag, tokenizer)
                for _, row in df.iterrows()]
        return Dataset.from_list(rows)

    train_ds = df_to_dataset(splits["train"])
    val_ds   = df_to_dataset(splits["val"])
    print(f"  train: {len(train_ds)}  |  val: {len(val_ds)}")

    sft_config = SFTConfig(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,   # effective batch size = 16
        warmup_ratio=0.05,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        fp16=True,                        # 2080 Ti has no bf16 support
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_seq_length=1024,
        dataset_text_field="text",
        report_to="none",
        dataloader_pin_memory=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )

    print("\n[3/5] Training...")
    t0 = time.time()
    trainer.train()
    print(f"  Done ({(time.time()-t0)/3600:.1f} h)")

    adapter_path = OUTPUT_DIR / "lora_adapter"
    trainer.model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"  LoRA adapter saved → {adapter_path}")

    return trainer.model


# ─────────────────────────────────────────────
# 6. Evaluation
# ─────────────────────────────────────────────

def generate_answers(model, tokenizer, df, rag: SyllabusRAG, batch_size=4) -> list:
    model.eval()
    all_preds = []

    for i in range(0, len(df), batch_size):
        batch   = df.iloc[i : i + batch_size]
        prompts = []
        for _, row in batch.iterrows():
            context              = rag.retrieve(row["question"])
            system_msg, user_msg = build_prompt(context, row["question"])
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(text)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        for out in outputs:
            pred = tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
            all_preds.append(pred)

        if (i // batch_size) % 5 == 0:
            print(f"  generating... {min(i+batch_size, len(df))}/{len(df)}")

    return all_preds


def compute_metrics(preds, refs, qtypes):
    from bert_score import score as bert_score
    from rouge_score import rouge_scorer as rs_mod

    print("  Computing ROUGE...")
    scorer = rs_mod.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rL = [], [], []
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)
        r1.append(s["rouge1"].fmeasure)
        r2.append(s["rouge2"].fmeasure)
        rL.append(s["rougeL"].fmeasure)

    print("  Computing BERTScore...")
    _, _, F1 = bert_score(
        preds, refs,
        lang="en",
        model_type="distilbert-base-uncased",
        verbose=False,
    )
    bf = F1.tolist()

    overall = dict(rouge1=np.mean(r1), rouge2=np.mean(r2),
                   rougeL=np.mean(rL), bert_f1=np.mean(bf),
                   n_samples=len(preds))

    by_type = {}
    for qt in set(qtypes):
        idx = [i for i, t in enumerate(qtypes) if t == qt]
        by_type[qt] = dict(
            rouge1=np.mean([r1[i] for i in idx]),
            rouge2=np.mean([r2[i] for i in idx]),
            rougeL=np.mean([rL[i] for i in idx]),
            bert_f1=np.mean([bf[i] for i in idx]),
            n_samples=len(idx),
        )

    return overall, by_type, r1, bf


def save_results(splits, preds, refs, qtypes, overall, by_type, r1, bf, tag):
    df = splits["test"].copy()
    df["predicted_answer"] = preds
    df["rouge1"]           = r1
    df["bert_f1"]          = bf
    df.to_csv(OUTPUT_DIR / f"test_predictions_{tag}.csv", index=False)

    summary = dict(
        model=MODEL_NAME, syllabus=TARGET_SYLLABUS,
        tag=tag, timestamp=datetime.now().isoformat(),
        overall={k: round(v, 4) if isinstance(v, float) else v
                 for k, v in overall.items()},
        by_question_type={
            qt: {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}
            for qt, m in by_type.items()
        },
    )
    with open(OUTPUT_DIR / f"metrics_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  ── {tag} ──")
    print(f"  ROUGE-1  : {overall['rouge1']:.4f}")
    print(f"  ROUGE-2  : {overall['rouge2']:.4f}")
    print(f"  ROUGE-L  : {overall['rougeL']:.4f}")
    print(f"  BERTScore: {overall['bert_f1']:.4f}  (n={overall['n_samples']})")
    print(f"\n  by question_type:")
    for qt, m in sorted(by_type.items()):
        print(f"    [{qt}] n={m['n_samples']}  "
              f"R1={m['rouge1']:.3f}  RL={m['rougeL']:.3f}  BERT={m['bert_f1']:.3f}")


# ─────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────

def main():
    t_total = time.time()

    # Load data
    print("\n[Data]")
    syllabus_text = load_syllabus_text(TARGET_SYLLABUS)
    print(f"  Syllabus length: {len(syllabus_text):,} chars")
    splits = load_and_filter(TARGET_SYLLABUS)
    for name, df in splits.items():
        if len(df) == 0:
            raise ValueError(f"No rows for '{TARGET_SYLLABUS}' in {name}.csv")

    # Build RAG index (once, reused for train / eval)
    print("\n[RAG index]")
    rag = SyllabusRAG(syllabus_text)

    # Load model
    model, tokenizer = load_model_and_tokenizer()

    # Baseline evaluation (before fine-tuning)
    print("\n[4/5] Baseline evaluation (before fine-tuning)...")
    base_preds = generate_answers(model, tokenizer, splits["test"], rag)
    base_overall, base_by_type, base_r1, base_bf = compute_metrics(
        base_preds,
        splits["test"]["answer"].tolist(),
        splits["test"]["question_type"].tolist(),
    )
    save_results(splits, base_preds,
                 splits["test"]["answer"].tolist(),
                 splits["test"]["question_type"].tolist(),
                 base_overall, base_by_type, base_r1, base_bf,
                 tag="baseline")

    # Fine-tuning
    model = train(model, tokenizer, splits, rag)

    # Post fine-tuning evaluation
    print("\n[5/5] Evaluation after fine-tuning...")
    ft_preds = generate_answers(model, tokenizer, splits["test"], rag)
    ft_overall, ft_by_type, ft_r1, ft_bf = compute_metrics(
        ft_preds,
        splits["test"]["answer"].tolist(),
        splits["test"]["question_type"].tolist(),
    )
    save_results(splits, ft_preds,
                 splits["test"]["answer"].tolist(),
                 splits["test"]["question_type"].tolist(),
                 ft_overall, ft_by_type, ft_r1, ft_bf,
                 tag="finetuned")

    # Before / after comparison JSON
    compare = {
        "baseline":  {k: round(v, 4) if isinstance(v, float) else v
                      for k, v in base_overall.items()},
        "finetuned": {k: round(v, 4) if isinstance(v, float) else v
                      for k, v in ft_overall.items()},
    }
    with open(OUTPUT_DIR / "comparison.json", "w") as f:
        json.dump(compare, f, indent=2)

    print("\n" + "=" * 50)
    print("  Before vs After")
    print("=" * 50)
    for m in ["rouge1", "rougeL", "bert_f1"]:
        b, a = base_overall[m], ft_overall[m]
        sign = "+" if a >= b else ""
        print(f"  {m:10s}: {b:.4f} → {a:.4f}  ({sign}{a-b:.4f})")

    print(f"\n  Total time: {(time.time()-t_total)/3600:.2f} h")
    print(f"  Results → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()