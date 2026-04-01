# SyllabusQA Fine-tuning

```
root/
├── data/
│   └── dataset_split/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── syllabi/
│   └── syllabi_redacted/
│       └── text/
│           └── Syllabus_Marine_Microbiome_2022_redacted.txt
├── train_eval.py
└── requirements.txt
```

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-finetuning.txt
pip install --user --upgrade pillow
pip install --user --upgrade jinja2
```

```bash
python train_eval.py
```

## Outputs
- `outputs/{timestamp}/test_predictions.csv`   ← predictions
- `outputs/{timestamp}/metrics_summary.json`   ← metrics summary
- `outputs/{timestamp}/baseline_vs_finetuned.json` ← baseline vs finetuned
- `outputs/{timestamp}/lora_adapter/`          ← lora adapter
- 