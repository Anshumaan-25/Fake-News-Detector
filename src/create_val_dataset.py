# src/create_val_dataset.py
import argparse
from pathlib import Path
import random

import pandas as pd
from sklearn.model_selection import train_test_split  # stratified split [web:665]


def build_synthetic(n_rows: int, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    spam_phrases = [
        "Free entry to win now!", "Congratulations, you've won a prize", "Claim your reward today",
        "Limited-time offer", "Act now, click link", "Winner selected! Reply YES",
        "You have been selected", "Exclusive deal just for you", "Lowest price guaranteed",
        "Hot singles near you", "Urgent: verify account", "Get cash fast", "Pre-approved loan",
        "Make money quickly", "Gift card awaiting"
    ]
    ham_phrases = [
        "See you at 7pm", "Meeting moved to 3", "Lunch tomorrow?", "Call when free",
        "On my way", "Thanks for the help", "Let’s catch up", "Happy birthday!",
        "Where are you?", "Got it, thanks", "Send the file please", "Can we reschedule?",
        "Stuck in traffic", "Back home now", "Talk later"
    ]
    half = max(1, n_rows // 2)
    spam = [{"text": spam_phrases[i % len(spam_phrases)], "label": "spam"} for i in range(half)]
    ham = [{"text": ham_phrases(i % len(ham_phrases)) if callable(ham_phrases) else ham_phrases[i % len(ham_phrases)], "label": "ham"} for i in range(n_rows - half)]
    data = spam + ham
    random.shuffle(data)
    return pd.DataFrame(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_csv", type=str, default=None, help="Existing labeled CSV with columns text/label")
    ap.add_argument("--text_col", type=str, default="text")
    ap.add_argument("--label_col", type=str, default="label")
    ap.add_argument("--val_csv", type=str, default="data/processed/val_labeled.csv")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--synth_rows", type=int, default=200, help="Rows for synthetic dataset if no source_csv")
    args = ap.parse_args()

    Path(args.val_csv).parent.mkdir(parents=True, exist_ok=True)

    if args.source_csv:
        # Split an existing labeled dataset with stratification
        df = pd.read_csv(args.source_csv)  # read labeled data [web:430]
        if args.text_col not in df.columns or args.label_col not in df.columns:
            raise ValueError(f"Expected columns '{args.text_col}' and '{args.label_col}', found {list(df.columns)}")
        train_df, val_df = train_test_split(
            df,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=df[args.label_col],  # keep class balance [web:665]
        )
        val_df.to_csv(args.val_csv, index=False)  # write CSV [web:439]
        print(f"[split] wrote {len(val_df)} rows -> {args.val_csv}")
    else:
        # Build a synthetic balanced dataset if no labeled file is provided
        val_df = build_synthetic(args.synth_rows, seed=args.seed)
        val_df.to_csv(args.val_csv, index=False)  # write CSV [web:439]
        print(f"[synthetic] wrote {len(val_df)} rows -> {args.val_csv}")


if __name__ == "__main__":
    main()
