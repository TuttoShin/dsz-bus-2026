# 서울시 버스정류장 좌표 변환

"""
02StopMaster.py

01IngestBus.py의 결과(표준버스정류장ID / 버스정류장ARS번호가 있는 테이블)에
정류장 위도/경도 좌표를 덧붙여 정류장 테이블을 만든다.

좌표 출처: 서울 열린데이터광장 서울시 버스정류소 위치정보
컬럼: NODE_ID, ARS_ID, 정류소명, X좌표(경도), Y좌표(위도), 정류소타입

- 조인 키는 반드시 NODE_ID(표준버스정류장ID)를 쓴다.
  ARS_ID는 지자체별로 독립 부여되는 번호라 다른 지역과 번호가 우연히 겹칠 수 있음!

좌표계는 위도/경도를 UTM-K로 변환한다.
SKT 유동인구 데이터가 UTM-K 격자를 쓰기 때문에 이 스크립트에서 미리 맞춰둔다.

CLI: python data-engineering/02StopMaster.py --route 9401
"""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path

import pandas as pd
from pyproj import Transformer

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# WGS84(위도/경도) -> UTM-K 변환
_TRANSFORMER = Transformer.from_crs(4326, 5179, always_xy=True)

# 이름 유사도가 이 값보다 낮으면 경고
NAME_SIMILARITY_WARN_THRESHOLD = 0.3


# %%
def load_bus_stops(bus_long_path: Path) -> pd.DataFrame:
    """01번 산출물에서 정류장 단위로 unique 목록만 뽑는다."""
    df = pd.read_parquet(bus_long_path)
    cols = ["표준버스정류장ID", "버스정류장ARS번호", "stop_name", "seq", "노선번호"]
    stops = (
        df[cols]
        .drop_duplicates(subset=["표준버스정류장ID", "노선번호"])
        .reset_index(drop=True)
    )
    return stops


# %%
def load_coord_source(coord_path: Path, sheet_name: str = "Data") -> pd.DataFrame:
    """서울시 버스정류소 위치정보 xlsx (Data 시트) 로드"""
    raw = pd.read_excel(coord_path, sheet_name=sheet_name)
    expected = {"NODE_ID", "ARS_ID", "정류소명", "X좌표", "Y좌표"}
    missing = expected - set(raw.columns)
    if missing:
        raise KeyError(f"좌표 파일에 없는 컬럼: {missing}. 실제 컬럼명: {list(raw.columns)}")

    coords = raw.rename(
        columns={
            "NODE_ID": "coord_id",
            "정류소명": "coord_name",
            "X좌표": "lon",
            "Y좌표": "lat",
        }
    )[["coord_id", "ARS_ID", "coord_name", "lat", "lon"]]
    coords["coord_id"] = coords["coord_id"].astype(str).str.strip()
    return coords.dropna(subset=["lat", "lon"])


# %%
def add_utmk(df: pd.DataFrame) -> pd.DataFrame:
    """위도/경도 -> UTM-K x, y 컬럼 추가."""
    df = df.copy()
    x, y = _TRANSFORMER.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
    df["utmk_x"] = x
    df["utmk_y"] = y
    return df


def _name_similarity(a: str, b: str) -> float:
    if pd.isna(a) or pd.isna(b):
        return 0.0
    return difflib.SequenceMatcher(None, str(a), str(b)).ratio()


# %%
def build_stop_master(stops: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    """NODE_ID(표준버스정류장ID) 기준. ARS_ID는 검증용!!"""
    stops = stops.copy()
    stops["표준버스정류장ID"] = stops["표준버스정류장ID"].astype(str).str.strip()

    merged = stops.merge(
        coords, left_on="표준버스정류장ID", right_on="coord_id", how="left"
    ).drop(columns=["coord_id"])

    matched = merged["lat"].notna()
    print(f"좌표 매칭(NODE_ID 기준): {matched.sum()}/{len(merged)} ({matched.mean():.1%})")

    unmatched = merged[~matched]
    if not unmatched.empty:
        prefix_counts = unmatched["표준버스정류장ID"].str[0].value_counts()
        print(
            "매칭 실패 정류장 (표준버스정류장ID 앞자리별 개수):\n"
            f"{prefix_counts.to_string()}\n"
            "  -> '2'로 시작하면 대개 서울 외 구역이라 데이터셋 범위 밖일 가능성이 높음.\n"
        )
        print(unmatched[["표준버스정류장ID", "버스정류장ARS번호", "stop_name"]].drop_duplicates().to_string(index=False))

    # 매칭된 것 중에서도 이름이 너무 다르면 ARS 재사용 같은 오매칭 가능성 -> 경고만 출력
    merged["name_similarity"] = merged.apply(
        lambda r: _name_similarity(r["stop_name"], r["coord_name"]) if pd.notna(r["coord_name"]) else None,
        axis=1,
    )
    suspicious = merged[matched & (merged["name_similarity"] < NAME_SIMILARITY_WARN_THRESHOLD)]
    if not suspicious.empty:
        print("\n매칭은 됐지만 정류장명이 많이 달라 오매칭 의심:")
        print(
            suspicious[["표준버스정류장ID", "stop_name", "coord_name", "name_similarity"]]
            .drop_duplicates()
            .to_string(index=False)
        )

    return add_utmk(merged)


# %%
def main() -> None:
    parser = argparse.ArgumentParser(description="정류장 좌표 마스터 테이블 생성 (NODE_ID 조인)")
    parser.add_argument("--route", type=str, default="9401", help="bus_long_<route>.parquet 파일을 찾음")
    parser.add_argument(
        "--coord-file",
        type=str,
        default="서울시버스정류소위치정보(20260902).xlsx",
        help="data/raw/ 아래 좌표 xlsx 파일명",
    )
    args = parser.parse_args()

    bus_long_path = PROCESSED_DIR / f"bus_long_{args.route}.parquet"
    coord_path = RAW_DIR / args.coord_file

    stops = load_bus_stops(bus_long_path)
    coords = load_coord_source(coord_path)
    stop_master = build_stop_master(stops, coords)

    out_path = PROCESSED_DIR / f"stop_master_{args.route}.parquet"
    stop_master.to_parquet(out_path, index=False)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()