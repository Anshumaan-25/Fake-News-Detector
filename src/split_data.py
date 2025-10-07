import os
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

RAW_PATH = "../data/raw/SMSSpamCollection"
OUT_DIR = "../data/processed"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(RAW_PATH, sep="\t", header=None, names=["label", "text"])
    train_pool, test_df = train_test_split(df, test_size=0.20, stratify=df["label"], random_state=SEED)
    train_df, val_df  = train_test_split(
        train_pool, test_size=0.125, stratify=train_pool["label"], random_state=SEED)
    train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)
    print("Splits made:", train_df.shape, val_df.shape, test_df.shape)

if __name__ == "__main__":
    main()
