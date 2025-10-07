# src/set_threshold.py
import argparse, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--value", type=float, required=True, help="Operational threshold for spam (0..1)")
    ap.add_argument("--out", type=str, default="models/threshold.json")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"threshold": float(args.value)}, f, indent=2)
    print(f"Wrote threshold={args.value} to {args.out}")

if __name__ == "__main__":
    main()
