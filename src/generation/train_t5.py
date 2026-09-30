"""
T5-Small Fine-Tuning Pipeline for Customer Support Response Generation.

This script fine-tunes google-t5/t5-small using Hugging Face Transformers and PyTorch.
It supports:
- Data loading from JSON (input prompt -> gold standard response)
- Tokenization with max_source_length=512 and max_target_length=128
- Seq2SeqTrainingArguments configuration (learning rate, warmup, batch size, epochs)
- Model checkpointing, evaluation, and export for inference.

Usage:
    python src/generation/train_t5.py --data_path data/synthetic_training_data.json --epochs 5 --batch_size 4 --output_dir models/t5_support_finetuned
"""

import argparse
import json
import os
import sys

# Prevent transformers from trying to load TensorFlow/Keras 3 backend
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune T5-small for Customer Email Response Generation")
    parser.add_argument("--data_path", type=str, default="data/synthetic_training_data.json", help="Path to JSON dataset")
    parser.add_argument("--base_model", type=str, default="google-t5/t5-small", help="Hugging Face model identifier")
    parser.add_argument("--output_dir", type=str, default="models/t5_support_finetuned", help="Directory to save fine-tuned model")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Peak learning rate")
    parser.add_argument("--max_source_length", type=int, default=512, help="Max token length for input prompt")
    parser.add_argument("--max_target_length", type=int, default=128, help="Max token length for generated response")
    parser.add_argument("--dry_run", action="store_true", help="Validate dataset and configuration without running training")
    return parser.parse_args()


def train():
    args = parse_args()
    print("=" * 70)
    print("       T5-SMALL FINE-TUNING FOR CUSTOMER SUPPORT RESPONSES")
    print("=" * 70)
    print(f"Base Model:       {args.base_model}")
    print(f"Dataset:          {args.data_path}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Epochs:           {args.epochs}")
    print(f"Batch Size:       {args.batch_size}")
    print(f"Learning Rate:    {args.learning_rate}")
    print("=" * 70)

    # 1. Validate dataset existence
    if not os.path.exists(args.data_path):
        print(f"[ERROR] Training data not found at: {args.data_path}")
        sys.exit(1)

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[OK] Successfully loaded {len(data)} training pairs.")

    # 2. Check dependencies
    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSeq2SeqLM,
            Seq2SeqTrainingArguments,
            Seq2SeqTrainer,
            DataCollatorForSeq2Seq,
        )
        from datasets import Dataset
    except Exception as e:
        print("\n[DEPENDENCY NOTE] PyTorch/Transformers training environment requires compatible packages.")
        print(f"Details: {e}")
        print("\nTo train on your machine or GPU:")
        print("    pip install torch transformers datasets accelerate")
        print("The pipeline's inference engine operates with the built-in Contextual Policy Engine fallback.")
        return

    if args.dry_run:
        print("[DRY-RUN] Validation complete. Ready for full training execution.")
        return

    # 3. Prepare Dataset
    print("\n[1/4] Preparing dataset formatting and splits...")
    inputs = [item["input_text"] for item in data]
    targets = [item["target_text"] for item in data]

    raw_dataset = Dataset.from_dict({
        "input_text": inputs,
        "target_text": targets,
    })

    # Train / Validation split (80/20 or fallback if small dataset)
    if len(raw_dataset) > 4:
        split_dataset = raw_dataset.train_test_split(test_size=0.2, seed=42)
        train_ds = split_dataset["train"]
        eval_ds = split_dataset["test"]
    else:
        train_ds = raw_dataset
        eval_ds = raw_dataset

    # 4. Tokenizer & Preprocessing
    print(f"\n[2/4] Loading tokenizer '{args.base_model}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    def preprocess_function(examples):
        model_inputs = tokenizer(
            examples["input_text"],
            max_length=args.max_source_length,
            truncation=True,
            padding="max_length",
        )
        labels = tokenizer(
            text_target=examples["target_text"],
            max_length=args.max_target_length,
            truncation=True,
            padding="max_length",
        )
        # Replace padding token id's with -100 so cross-entropy ignores them
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing datasets...")
    tokenized_train = train_ds.map(preprocess_function, batched=True)
    tokenized_eval = eval_ds.map(preprocess_function, batched=True)

    # 5. Model Loading
    print(f"\n[3/4] Loading base model '{args.base_model}'...")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    # 6. Training Configuration
    print("\n[4/4] Initializing Seq2SeqTrainer...")
    os.makedirs(args.output_dir, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch" if len(tokenized_eval) > 1 else "no",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("\nStarting fine-tuning...")
    trainer.train()

    print(f"\n[SUCCESS] Saving fine-tuned model and tokenizer to: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Fine-tuning completed successfully!")


if __name__ == "__main__":
    train()
