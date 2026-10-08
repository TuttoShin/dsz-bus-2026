"""
MakeDummyRaw.py (data-engineering/ 밖, 예: scripts/ 또는 data-engineering/dev/ 에 둘 것)

실제 stop_master_9401.parquet의 좌표 주변에 가짜 SKT 유동인구를 뿌리고,
정의서와 동일한 컬럼명으로 가짜 KEC 차량통행지표/사고데이터를 만든다.
04~07을 안심구역 방문 전에 리허설하기 위한 용도 — 결과 숫자는 의미 없고,
파이프라인이 안 깨지고 끝까지 도는지만 확인하는 게 목적.

실행:
  python data-engineering/MakeDummyRaw.py
  python RunAll.py --dummy      # 04~10 전체 리허설 -> export_dummy/
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


def make_dummy_wkdy(route: str = "9401") -> None:
    """SKT_002 요일별 유동인구 가짜 데이터. 정류장마다 평일/주말 차이를 다르게 줘서 통근 비율이 정류장마다 다르게."""
    stops = pd.read_parquet(PROCESSED_DIR / f"stop_master_{route}.parquet")
    seoul = stops.dropna(subset=["utmk_x", "utmk_y"])[["utmk_x", "utmk_y"]].drop_duplicates()
    days = ["MON", "TUS", "WED", "THU", "FRI", "SAT", "SUN"]

    rows = []
    for _, s in seoul.iterrows():
        weekend_ratio = float(rng.uniform(0.3, 1.1))  # 주말/평일 비율 (작을수록 통근 동네)
        for dx in (-50, 0, 50):
            for dy in (-50, 0, 50):
                row = {"STD_YM": "202508", "X_COORD": str(s["utmk_x"] + dx), "Y_COORD": str(s["utmk_y"] + dy)}
                weekday = float(rng.poisson(400))
                for d in days:
                    row[f"FLOW_POP_CNT_{d}"] = weekday * (weekend_ratio if d in ("SAT", "SUN") else 1.0)
                rows.append(row)

    pd.DataFrame(rows).to_csv(RAW_DIR / "dummy_flow_wkdy.csv", sep="|", index=False, encoding="utf-8")
    print(f"dummy wkdy saved: {len(rows)} rows")


def make_dummy_road_indicator() -> None:
    """05번 ROUTE_SECTIONS와 같은 도로명으로 가짜 링크 생성.
    대전 경부고속도로, 부산 중구 링크도 섞어둠 -> 05번 필터가 이걸 제대로 빼는지 확인용."""
    links = [
        ("경부고속도로", "경기도", "성남시분당구", 4), ("경부고속도로", "서울특별시", "서초구", 3),
        ("한남대로", "서울특별시", "용산구", 2), ("을지로", "서울특별시", "중구", 2), ("종로", "서울특별시", "종로구", 2),
        ("세종대로", "서울특별시", "중구", 2), ("퇴계로", "서울특별시", "중구", 2),
        ("경부고속도로", "대전광역시", "대덕구", 3), ("중앙대로", "부산광역시", "중구", 2),  # 미끼
    ]
    # 02b번 결과가 있으면 정류장이 있는 동마다 도로 2개씩 추가 (정류장별 지수 리허설용)
    # 마지막 동은 일부러 빼서 "동 이름 불일치 -> 시군구로 대신" 경로도 확인
    dong_path = PROCESSED_DIR / "stop_dong_9401.parquet"
    emd_of = {}
    if dong_path.exists():
        dongs = pd.read_parquet(dong_path).dropna(subset=["dong_nm"]).drop_duplicates("dong_nm")
        for d in dongs.iloc[:-1].itertuples():
            links.append((f"{d.dong_nm}로", d.sido_nm, d.sgg_nm, 2))
            emd_of[f"{d.dong_nm}로"] = d.dong_nm

    rows, link_id = [], 1000
    for road, sido, sigungu, n in links:
        for _ in range(n):
            v_n = float(rng.uniform(60, 90)) if "고속" in road else float(rng.uniform(25, 40))
            rows.append({
                "LEVEL5_5_LINKID": link_id,
                "ROAD_RANK": 1 if "고속" in road else 3,
                "ROAD_LENGTH": float(rng.uniform(300, 1500)),
                "ROAD_NAME": road,
                "SIDO_CODE": 0, "SIGUNGU_CODE": 0, "EMD_CODE": 0,
                "SIDO_NAME": sido, "SIGUNGU_NAME": sigungu, "EMD_NAME": emd_of.get(road, "DUMMY"),
                "velocity_AVRG": v_n * 0.8,
                "velocity_AVRG_NRMLT": v_n,
                "velocity_CG_AVRG": v_n * float(rng.uniform(0.3, 0.7)),
                "TI_CG": float(rng.uniform(0, 100)),
                "FRIN_CG": float(rng.uniform(0, 100)),
                "ALL_AADT": int(rng.integers(10000, 50000)),
                "PSCR_AADT": int(rng.integers(8000, 40000)),
                "BUS_AADT": int(rng.integers(100, 2000)),
                "FGCR_AADT": int(rng.integers(500, 5000)),
                "ALL_VKT": int(rng.integers(1000, 9000)),
                "CO": 0, "CO2": 0, "PM": 0, "NOx": 0, "VOCs": 0,
            })
            link_id += 1
    pd.DataFrame(rows).to_csv(RAW_DIR / "dummy_road_indicator.csv", index=False, encoding="utf-8")
    print(f"dummy road indicator saved: {len(rows)} rows")


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    make_dummy_flow()
    make_dummy_road_indicator()