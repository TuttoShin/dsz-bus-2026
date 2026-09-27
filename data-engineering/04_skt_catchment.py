# 정류장 반경 100~150m 내 유동인구 셀 합산, 정류장x시간대 유동인구 테이블
# 안심구역에서....

"""
04_skt_catchment.py

SKT 시간대별 유동인구(seoul_flow_time.csv, 50m*50m cell 단위, 서울시 전역)를
정류장 반경 버퍼로 묶어서, 정류장별/시간대별 주변 유동인구 테이블을 만든다.

배경)
버스 승하차 데이터는 탑승한 사람만 기록한다. 못 탄 사람은 데이터에 존재하지 않는다.
반면 이 정류장x시간대 유동인구 테이블은, '그 정류장 주변에 실제로 체류/이동한
사람이 몇 명이었는가'라는 별개의 축을 제공한다.

03_load_profile.py의 정류장x시간대 승하차(boarding)와 이 파일의 정류장x시간대 유동인구(flow_pop)를 
나중에 같은 (표준버스정류장ID, hour) 키로 merge하면,
주변엔 사람이 많았는데 승차는 안 늘어난 정류장/시간대를 잔차로 잡아낼 수 있다
-> 미승차 수요 추정의 재료

데이터 정의서 기준 스펙 (SKT_003, seoul_flow_time.csv)
- 구분자: 파이프(|), 인코딩: utf-8
- PK: STD_YM(기준년월, YYYYMM) + X_COORD + Y_COORD (둘 다 UTM-K, 문자열로 제공됨)
- TMST_00 ~ TMST_23: 해당 시(hour)의 유동인구 추정치(명, 소수점 있음)
- 공간 단위: 50m*50m cell (즉 각 좌표는 셀의 대표점)

반경 파라미터)
정류장 반경 100~150m를 기본 탐색 범위로 잡는다(50m cell 한 칸~두 칸 정도 커버)
너무 좁으면(50m 이하) 셀이 안 걸리는 정류장이 생기고, 너무 넓으면(300m+) 
옆 정류장 수요까지 섞여서 그 정류장 고유의 주변 유동인구라는 의미가 흐려진다.
--radius-m으로 조정하면서 실제 분포를 보고 정할 것.

CLI: python data-engineering/04_skt_catchment.py --route 9401 --radius-m 150
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

FLOW_HOURS = [f"TMST_{h:02d}" for h in range(24)]


# %%
def load_seoul_stops(route: str) -> pd.DataFrame:
    """02_stop_master.py 산출물에서 서울 좌표가 매칭된 정류장만 남긴다.

    SKT 유동인구는 서울시만 커버하므로, 좌표가 없는(NODE_ID 매칭 실패) 정류장은 애초에 분석 대상 X
    """
    stops = pd.read_parquet(PROCESSED_DIR / f"stop_master_{route}.parquet")
    seoul_stops = stops.dropna(subset=["utmk_x", "utmk_y"])[
        ["표준버스정류장ID", "seq", "stop_name", "utmk_x", "utmk_y"]
    ].drop_duplicates()

    dropped = stops["표준버스정류장ID"].nunique() - seoul_stops["표준버스정류장ID"].nunique()
    if dropped > 0:
        print(f"SKT 유동인구 대상 제외(서울 좌표 없음): {dropped}개 정류장 (성남 구간 등)")
    return seoul_stops.reset_index(drop=True)


# %%
def load_flow_population(flow_path: Path, std_ym: str | None = None) -> pd.DataFrame:
    """SKT_003 시간대별 유동인구 CSV를 로드한다.

    실제 컬럼명: STD_YM, X_COORD, Y_COORD, TMST_00 ~ TMST_23
    X_COORD/Y_COORD는 정의서상 VARCHAR(UTM-K 문자열)로 내려오므로 float 변환 필요.
    """
    df = pd.read_csv(flow_path, sep="|", encoding="utf-8", dtype={"STD_YM": str})
    if std_ym is not None:
        df = df[df["STD_YM"] == std_ym]

    df["utmk_x"] = df["X_COORD"].astype(float)
    df["utmk_y"] = df["Y_COORD"].astype(float)
    return df


# %%
def join_flow_to_stops(
    stops: pd.DataFrame, flow: pd.DataFrame, radius_m: float = 150.0
) -> pd.DataFrame:
    """정류장별로 반경 radius_m 안의 유동인구 셀을 모두 찾아 시간대별로 합산한다.

    한 셀이 여러 정류장의 버퍼에 동시에 걸리면 양쪽에 모두 카운팅된다(중복 허용)
    정류장끼리 셀을 나눠 갖는 게 아니라 그 정류장 주변엔 이만큼의 유동인구가
    있었다는 정류장 각자의 관점이라, 정의서상 유동인구 자체도 원래 중복 카운팅을
    허용하는 지표라는 점과 맞는다.

    numpy 브로드캐스팅으로 정류장×셀 거리 행렬을 한 번에 계산한다(정류장·셀 수가 각각 수백~수천 단위)
    if. 실제 셀 개수가 훨씬 크면(수만+) 안심구역에서 scipy.spatial.cKDTree로 바꿔서 반경 검색만 빠르게
    """
    stop_xy = stops[["utmk_x", "utmk_y"]].to_numpy()
    flow_xy = flow[["utmk_x", "utmk_y"]].to_numpy()
    flow_vals = flow[FLOW_HOURS].to_numpy()

    # (n_stops, n_cells) 거리 행렬
    dist = np.sqrt(
        ((stop_xy[:, None, :] - flow_xy[None, :, :]) ** 2).sum(axis=2)
    )
    within = dist <= radius_m  # (n_stops, n_cells)

    n_cells_matched = within.sum(axis=1)
    # within(bool) x flow_vals(n_cells, 24) -> (n_stops, 24) 합산
    hourly_sum = within.astype(float) @ flow_vals

    result = stops[["표준버스정류장ID", "seq", "stop_name"]].copy()
    result["n_cells_matched"] = n_cells_matched
    for h in range(24):
        result[f"flow_hour_{h:02d}"] = hourly_sum[:, h]

    unmatched = result[result["n_cells_matched"] == 0]
    if not unmatched.empty:
        print(
            f"반경 {radius_m}m 안에 유동인구 셀이 하나도 안 걸린 정류장 "
            f"{len(unmatched)}개 (반경을 넓혀볼 것):\n"
            f"{unmatched[['stop_name', 'seq']].to_string(index=False)}"
        )
    return result


# %%
def to_long(flow_by_stop: pd.DataFrame) -> pd.DataFrame:
    """wide(flow_hour_00~23) -> long(hour, flow_pop) 변환. load_profile과 merge하기 쉬운 형태로."""
    id_cols = ["표준버스정류장ID", "seq", "stop_name", "n_cells_matched"]
    hour_cols = [c for c in flow_by_stop.columns if c.startswith("flow_hour_")]
    long_df = flow_by_stop.melt(
        id_vars=id_cols, value_vars=hour_cols, var_name="hour_col", value_name="flow_pop"
    )
    long_df["hour"] = long_df["hour_col"].str.extract(r"(\d+)$").astype(int)
    return (
        long_df.drop(columns=["hour_col"])
        .sort_values(["seq", "hour"])
        .reset_index(drop=True)
    )


# %%
def build_flow_profile(
    route: str, flow_file: str, radius_m: float, std_ym: str | None = None
) -> pd.DataFrame:
    stops = load_seoul_stops(route)
    flow = load_flow_population(RAW_DIR / flow_file, std_ym=std_ym)
    flow_by_stop = join_flow_to_stops(stops, flow, radius_m=radius_m)
    return to_long(flow_by_stop)


# %%
def main() -> None:
    parser = argparse.ArgumentParser(description="정류장 반경 내 SKT 유동인구 집계 (시간대별)")
    parser.add_argument("--route", type=str, default="9401")
    parser.add_argument(
        "--flow-file", type=str, default="seoul_flow_time.csv",
        help="data/raw/ 아래 SKT_003(시간대별 유동인구) 파일명",
    )
    parser.add_argument(
        "--radius-m", type=float, default=150.0,
        help="정류장 기준 버퍼 반경(m). 100~150 권장 범위",
    )
    parser.add_argument("--std-ym", type=str, default=None, help="기준년월(YYYYMM) 필터, 미지정시 전체")
    args = parser.parse_args()

    profile = build_flow_profile(args.route, args.flow_file, args.radius_m, args.std_ym)

    out_path = PROCESSED_DIR / f"flow_profile_{args.route}.parquet"
    profile.to_parquet(out_path, index=False)

    print(f"\nrows: {len(profile):,}")
    print(f"stops covered: {profile['표준버스정류장ID'].nunique()}")
    peak = profile.loc[profile.groupby("표준버스정류장ID")["flow_pop"].idxmax()]
    print("\n정류장별 최고 유동인구 시간대 (상위 5):")
    print(
        peak.nlargest(5, "flow_pop")[["stop_name", "seq", "hour", "flow_pop"]]
        .to_string(index=False)
    )
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()