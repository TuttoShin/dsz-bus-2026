# parquet를 반입용 csv로 변환
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")

for name in ["bus_long_9401", "stop_master_9401", "load_profile_9401"]:
    df = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
    df.to_csv(PROCESSED_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    print(f"{name}: {len(df):,} rows -> {name}.csv")