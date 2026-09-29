"""
_make_dummy_raw.py (data-engineering/ 밖, 예: scripts/ 또는 data-engineering/dev/ 에 둘 것)

실제 stop_master_9401.parquet의 좌표 주변에 가짜 SKT 유동인구를 뿌리고,
정의서와 동일한 컬럼명으로 가짜 KEC 차량통행지표/사고데이터를 만든다.
04~07을 안심구역 방문 전에 리허설하기 위한 용도 — 결과 숫자는 의미 없고,
파이프라인이 안 깨지고 끝까지 도는지만 확인하는 게 목적.

실행:
  python _make_dummy_raw.py
  python 04_skt_catchment.py --flow-file dummy_flow_time.csv
  python 05_kec_route_index.py --road-indicator-file dummy_road_indicator.csv --road-names DUMMY_ROAD --sigungu-names DUMMY_SIGUNGU
  python 06_build_features.py
  python 07_rule_based_score.py --diagnose-only
  python 07_rule_based_score.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

rng = np.random.default_rng(42)


def make_dummy_flow(route: str = "9401") -> None:
    stops = pd.read_parquet(PROCESSED_DIR / f"stop_master_{route}.parquet")
    seoul = stops.dropna(subset=["utmk_x", "utmk_y"])[["utmk_x", "utmk_y"]].drop_duplicates()

    rows = []
    for _, s in seoul.iterrows():
        # 정류장 하나당 3x3 셀 격자(50m 간격)를 주변에 흩뿌림
        for dx in (-50, 0, 50):
            for dy in (-50, 0, 50):
                row = {
                    "STD_YM": "202508",
                    "X_COORD": str(s["utmk_x"] + dx),
                    "Y_COORD": str(s["utmk_y"] + dy),
                }
                for h in range(24):
                    # 출퇴근 시간대(8,18시)에 더 높게, 나머지는 낮게 - 리허설용 가짜 패턴
                    base = 50 if h in (8, 18) else 15
                    row[f"TMST_{h:02d}"] = float(rng.poisson(base))
                rows.append(row)

    pd.DataFrame(rows).to_csv(RAW_DIR / "dummy_flow_time.csv", sep="|", index=False, encoding="utf-8")
    print(f"dummy flow saved: {len(rows)} rows")


def make_dummy_road_indicator() -> None:
    rows = []
    for i in range(5):
        rows.append({
            "LEVEL5_5_LINKID": 1000 + i,
            "ROAD_RANK": 1,
            "ROAD_LENGTH": float(rng.uniform(200, 800)),
            "ROAD_NAME": "DUMMY_ROAD",
            "SIDO_CODE": 41, "SIGUNGU_CODE": 4113, "EMD_CODE": 4113510,
            "SIDO_NAME": "경기도", "SIGUNGU_NAME": "DUMMY_SIGUNGU", "EMD_NAME": "정자동",
            "velocity_AVRG": float(rng.uniform(30, 80)),
            "velocity_AVRG_NRMLT": float(rng.uniform(60, 90)),
            "velocity_CG_AVRG": float(rng.uniform(10, 40)),
            "TI_CG": float(rng.uniform(0, 100)),
            "FRIN_CG": float(rng.uniform(0, 100)),
            "ALL_AADT": int(rng.integers(10000, 50000)),
            "PSCR_AADT": int(rng.integers(8000, 40000)),
            "BUS_AADT": int(rng.integers(100, 2000)),
            "FGCR_AADT": int(rng.integers(500, 5000)),
            "ALL_VKT": int(rng.integers(1000, 9000)),
            "CO": 0, "CO2": 0, "PM": 0, "NOx": 0, "VOCs": 0,
        })
    pd.DataFrame(rows).to_csv(RAW_DIR / "dummy_road_indicator.csv", index=False, encoding="utf-8")
    print(f"dummy road indicator saved: {len(rows)} rows")


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    make_dummy_flow()
    make_dummy_road_indicator()