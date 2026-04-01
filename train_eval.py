"""
SyllabusQA Fine-tuning & Evaluation Pipeline
모델: Qwen2.5-3B-Instruct + QLoRA (4-bit)
GPU: RTX 2080 Ti (11GB VRAM) 기준 설계
목적: LLM의 강좌 안내 Q&A 한계 분석
"""

import os
import json
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────
# 경로 설정 (프로젝트 루트 기준으로 수정하세요)
# ─────────────────────────────────────────────
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data" / "dataset_split"
SYLLABI_DIR = BASE_DIR / "syllabi" / "syllabi_redacted" / "text"
OUTPUT_DIR = BASE_DIR / "outputs" / datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 사용할 Syllabus (하나만 사용 – 첫 실험 권장)
TARGET_SYLLABUS = "Syllabus_Marine_Microbiome_2022_redacted"

# 모델 설정
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

print("=" * 60)
print("  SyllabusQA Fine-tuning Pipeline")
print(f"  모델: {MODEL_NAME}")
print(f"  대상 Syllabus: {TARGET_SYLLABUS}")
print(f"  출력 경로: {OUTPUT_DIR}")
print("=" * 60)


# ─────────────────────────────────────────────
# 1. 데이터 로드 및 필터링
# ─────────────────────────────────────────────

def load_syllabus_text(syllabus_name: str) -> str:
    """syllabus 텍스트 파일을 읽어 반환"""
    path = SYLLABI_DIR / f"{syllabus_name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Syllabus 파일을 찾을 수 없습니다: {path}")
    return path.read_text(encoding="utf-8").strip()


def load_and_filter_data(syllabus_name: str):
    """CSV 로드 후 특정 syllabus만 필터링"""
    splits = {}
    for split in ["train", "val", "test"]:
        df = pd.read_csv(DATA_DIR / f"{split}.csv")
        filtered = df[df["syllabus_name"] == syllabus_name].reset_index(drop=True)
        splits[split] = filtered
        print(f"  [{split}] 전체: {len(df)}행 → 필터링 후: {len(filtered)}행")

    return splits


def build_prompt(syllabus_text: str, question: str) -> str:
    """
    모델 입력 프롬프트 생성.
    Qwen2.5의 chat template 형식을 따릅니다.
    """
    system_msg = (
        "You are a helpful teaching assistant. "
        "Answer the student's question based ONLY on the provided course syllabus. "
        "Be concise and accurate. If the answer is not in the syllabus, say so."
    )
    user_msg = f"""Course Syllabus:
\"\"\"
{syllabus_text[:3000]}
\"\"\"

Student Question: {question}"""
    return system_msg, user_msg


def format_for_training(row, syllabus_text: str, tokenizer) -> dict:
    """
    SFT(지도 파인튜닝)를 위한 chat template 포맷 변환.
    Qwen2.5의 apply_chat_template을 사용합니다.
    """
    system_msg, user_msg = build_prompt(syllabus_text, row["question"])

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": str(row["answer"])},
    ]

    # apply_chat_template: 모델이 기대하는 특수 토큰 포함 문자열 생성
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


# ─────────────────────────────────────────────
# 2. 모델 & LoRA 설정
# ─────────────────────────────────────────────

def load_model_and_tokenizer():
    """4-bit QLoRA 설정으로 모델 로드"""
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    print("\n[1/5] 모델 로딩 중...")

    # 4-bit 양자화 설정
    # → VRAM을 약 1/4로 줄여줍니다 (3B 모델 기준 ~2GB로 압축)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,   # 이중 양자화로 추가 압축
        bnb_4bit_quant_type="nf4",         # NF4: 4-bit 중 가장 정밀도 높음
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # 학습 시 우측 패딩

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # k-bit 학습을 위한 준비 (gradient checkpointing 포함)
    model = prepare_model_for_kbit_training(model)

    # LoRA 설정
    # r=16: 랭크(rank). 작을수록 빠르고 메모리↓, 클수록 표현력↑
    # alpha=32: 학습률 스케일. 보통 r*2 권장
    # target_modules: 어떤 레이어에 LoRA를 붙일지
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
# 3. 학습
# ─────────────────────────────────────────────

def train(model, tokenizer, splits, syllabus_text):
    """SFT(Supervised Fine-Tuning) 학습"""
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    print("\n[2/5] 학습 데이터 준비 중...")

    # DataFrame → HuggingFace Dataset 변환 후 프롬프트 포맷 적용
    def df_to_hf_dataset(df):
        rows = [format_for_training(row, syllabus_text, tokenizer)
                for _, row in df.iterrows()]
        return Dataset.from_list(rows)

    train_dataset = df_to_hf_dataset(splits["train"])
    val_dataset = df_to_hf_dataset(splits["val"])

    print(f"  학습 샘플: {len(train_dataset)}개")
    print(f"  검증 샘플: {len(val_dataset)}개")

    # 학습 설정
    # ── 핵심 하이퍼파라미터 설명 ──
    # num_train_epochs=5: 전체 데이터를 5번 반복 학습
    # per_device_train_batch_size=2: GPU당 한 번에 2개 샘플
    # gradient_accumulation_steps=8: 8번 모아서 한 번 업데이트
    #   → 실질적 배치 크기 = 2×8 = 16
    # learning_rate=2e-4: LoRA에서 검증된 표준 학습률
    # max_seq_length=1024: 입력 최대 토큰 수 (VRAM 절약)
    sft_config = SFTConfig(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        num_train_epochs=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_ratio=0.05,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        fp16=True,                    # 2080 Ti는 bf16 미지원 → fp16 사용
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_seq_length=1024,
        dataset_text_field="text",
        report_to="none",             # wandb 등 비활성화
        dataloader_pin_memory=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    print("\n[3/5] 학습 시작...")
    train_start = time.time()
    trainer.train()
    train_time = time.time() - train_start
    print(f"  ✓ 학습 완료 ({train_time/3600:.1f}시간)")

    # 최종 모델 저장
    adapter_path = OUTPUT_DIR / "lora_adapter"
    trainer.model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"  ✓ LoRA 어댑터 저장: {adapter_path}")

    return trainer.model


# ─────────────────────────────────────────────
# 4. 평가
# ─────────────────────────────────────────────

def generate_answers(model, tokenizer, df, syllabus_text, batch_size=4) -> list:
    """테스트셋에 대한 답변 생성"""
    model.eval()
    all_preds = []

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i: i + batch_size]
        prompts = []
        for _, row in batch.iterrows():
            system_msg, user_msg = build_prompt(syllabus_text, row["question"])
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,  # 답변 생성 유도
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
                do_sample=False,        # 재현성을 위해 greedy decoding
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        # 입력 부분 제거 후 답변만 디코딩
        input_len = inputs["input_ids"].shape[1]
        for output in outputs:
            pred = tokenizer.decode(
                output[input_len:], skip_special_tokens=True
            ).strip()
            all_preds.append(pred)

        if (i // batch_size) % 5 == 0:
            print(f"  생성 중... {min(i+batch_size, len(df))}/{len(df)}")

    return all_preds


def compute_metrics(predictions: list, references: list, question_types: list) -> dict:
    """ROUGE + BERTScore + question_type별 분석"""
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score

    print("  ROUGE 계산 중...")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rouge1_list, rouge2_list, rougeL_list = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_list.append(scores["rouge1"].fmeasure)
        rouge2_list.append(scores["rouge2"].fmeasure)
        rougeL_list.append(scores["rougeL"].fmeasure)

    print("  BERTScore 계산 중... (시간이 걸릴 수 있습니다)")
    P, R, F1 = bert_score(
        predictions, references,
        lang="en",
        model_type="distilbert-base-uncased",  # 경량 모델로 속도 향상
        verbose=False,
    )
    bert_f1_list = F1.tolist()

    # 전체 평균
    overall = {
        "rouge1": np.mean(rouge1_list),
        "rouge2": np.mean(rouge2_list),
        "rougeL": np.mean(rougeL_list),
        "bert_f1": np.mean(bert_f1_list),
        "n_samples": len(predictions),
    }

    # question_type별 분석
    by_type = {}
    unique_types = set(question_types)
    for qt in unique_types:
        indices = [i for i, t in enumerate(question_types) if t == qt]
        by_type[qt] = {
            "rouge1": np.mean([rouge1_list[i] for i in indices]),
            "rouge2": np.mean([rouge2_list[i] for i in indices]),
            "rougeL": np.mean([rougeL_list[i] for i in indices]),
            "bert_f1": np.mean([bert_f1_list[i] for i in indices]),
            "n_samples": len(indices),
        }

    return overall, by_type, rouge1_list, bert_f1_list


# ─────────────────────────────────────────────
# 5. 결과 저장
# ─────────────────────────────────────────────

def save_results(
    splits, predictions, references, question_types,
    overall, by_type, rouge1_list, bert_f1_list
):
    """예측 결과 및 지표를 CSV/JSON으로 저장"""

    # 예측 결과 상세 CSV
    test_df = splits["test"].copy()
    test_df["predicted_answer"] = predictions
    test_df["rouge1"] = rouge1_list
    test_df["bert_f1"] = bert_f1_list
    result_csv = OUTPUT_DIR / "test_predictions.csv"
    test_df.to_csv(result_csv, index=False)
    print(f"  ✓ 예측 결과: {result_csv}")

    # 지표 요약 JSON
    summary = {
        "model": MODEL_NAME,
        "syllabus": TARGET_SYLLABUS,
        "timestamp": datetime.now().isoformat(),
        "overall_metrics": {k: round(v, 4) if isinstance(v, float) else v
                            for k, v in overall.items()},
        "by_question_type": {
            qt: {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}
            for qt, m in by_type.items()
        },
    }
    summary_json = OUTPUT_DIR / "metrics_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  ✓ 지표 요약: {summary_json}")

    # 콘솔 출력
    print("\n" + "=" * 50)
    print("  📊 전체 평가 결과")
    print("=" * 50)
    print(f"  ROUGE-1 :  {overall['rouge1']:.4f}")
    print(f"  ROUGE-2 :  {overall['rouge2']:.4f}")
    print(f"  ROUGE-L :  {overall['rougeL']:.4f}")
    print(f"  BERTScore: {overall['bert_f1']:.4f}")
    print(f"  샘플 수 :  {overall['n_samples']}개\n")
    print("  📊 Question Type별 결과")
    print("=" * 50)
    for qt, m in sorted(by_type.items()):
        print(f"  [{qt}] (n={m['n_samples']})")
        print(f"    ROUGE-1: {m['rouge1']:.4f} | ROUGE-L: {m['rougeL']:.4f} | BERT: {m['bert_f1']:.4f}")
    print("=" * 50)


# ─────────────────────────────────────────────
# 메인 실행
# ─────────────────────────────────────────────

def main():
    total_start = time.time()

    # 데이터 로드
    print("\n[데이터 로드]")
    syllabus_text = load_syllabus_text(TARGET_SYLLABUS)
    print(f"  Syllabus 길이: {len(syllabus_text)}자")
    splits = load_and_filter_data(TARGET_SYLLABUS)

    # 데이터가 없으면 종료
    for split_name, df in splits.items():
        if len(df) == 0:
            raise ValueError(
                f"'{TARGET_SYLLABUS}'에 해당하는 {split_name} 데이터가 없습니다. "
                "syllabus_name 컬럼값을 확인해주세요."
            )

    # 모델 로드
    model, tokenizer = load_model_and_tokenizer()

    # ── 파인튜닝 전 베이스라인 평가 ──
    print("\n[4/5] 파인튜닝 前 베이스라인 평가...")
    baseline_preds = generate_answers(
        model, tokenizer, splits["test"], syllabus_text
    )
    baseline_overall, baseline_by_type, baseline_r1, baseline_bert = compute_metrics(
        baseline_preds,
        splits["test"]["answer"].tolist(),
        splits["test"]["question_type"].tolist(),
    )

    baseline_results = {
        "predictions": baseline_preds,
        "overall": baseline_overall,
        "by_type": baseline_by_type,
        "rouge1_list": baseline_r1,
        "bert_f1_list": baseline_bert,
    }

    # ── 파인튜닝 ──
    model = train(model, tokenizer, splits, syllabus_text)

    # ── 파인튜닝 후 평가 ──
    print("\n[5/5] 파인튜닝 後 평가...")
    ft_preds = generate_answers(model, tokenizer, splits["test"], syllabus_text)
    ft_overall, ft_by_type, ft_r1, ft_bert = compute_metrics(
        ft_preds,
        splits["test"]["answer"].tolist(),
        splits["test"]["question_type"].tolist(),
    )

    # 결과 저장
    print("\n[결과 저장]")
    save_results(
        splits, ft_preds,
        splits["test"]["answer"].tolist(),
        splits["test"]["question_type"].tolist(),
        ft_overall, ft_by_type, ft_r1, ft_bert,
    )

    # 베이스라인 vs 파인튜닝 비교
    compare = {
        "baseline": {k: round(v, 4) if isinstance(v, float) else v
                     for k, v in baseline_overall.items()},
        "finetuned": {k: round(v, 4) if isinstance(v, float) else v
                      for k, v in ft_overall.items()},
    }
    with open(OUTPUT_DIR / "baseline_vs_finetuned.json", "w") as f:
        json.dump(compare, f, indent=2)

    print(f"\n🎉 전체 완료! 총 소요 시간: {(time.time()-total_start)/3600:.2f}시간")
    print(f"   결과 폴더: {OUTPUT_DIR}")

    # 개선 요약
    print("\n  📈 파인튜닝 전후 비교")
    for metric in ["rouge1", "rougeL", "bert_f1"]:
        before = baseline_overall[metric]
        after = ft_overall[metric]
        delta = after - before
        sign = "+" if delta >= 0 else ""
        print(f"  {metric:10s}: {before:.4f} → {after:.4f}  ({sign}{delta:.4f})")


if __name__ == "__main__":
    main()