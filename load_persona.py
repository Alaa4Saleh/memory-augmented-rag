from datasets import load_dataset
from pathlib import Path

def save_split(name: str, split: str, out_dir: Path):
    ds = load_dataset(name, split=split)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Parquet is compact and fast
    out_path = out_dir / f"{split}.parquet"
    ds.to_parquet(str(out_path))
    print(f"Saved {split} → {out_path} ({len(ds)} rows)")

if __name__ == "__main__":
    base = Path("data/persona_chat")
    dataset_name = "AlekseyKorshuk/persona-chat"  # v3-friendly mirror
    save_split(dataset_name, "train", base)
    save_split(dataset_name, "validation", base)  # sometimes called 'valid'
