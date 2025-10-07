import pandas as pd

# Load SMS Spam data (tab-separated, no header)
df = pd.read_csv("../data/raw/SMSSpamCollection", sep="\t", header=None, names=["label", "text"])

print("Shape:", df.shape)
print(df.head(5))
print("Label counts:\n", df["label"].value_counts())
