"""
06BuildFeatures.py

아래 네 파일의 결과를 정류장x시간대 표 하나로 합친다.
  03LoadProfile.py   : 승하차, 재차
  03bRouteShare.py   : 9401 분담률
  04SktCatchment.py  : 주변 유동인구 (시간대별 + 평일/주말)
  05KecRouteIndex.py: 혼잡 노출 지수 (노선 단위 + 정류장별)

분석 범위)
SKT 유동인구는 서울만 있으므로 서울 도심 구간 정류장만 남긴다.
성남시 분당구 정류장은 여기서 뺀다.
(단, 03번 재차 계산에는 성남 구간이 필요함: 서울에 들어올 때 이미 타고 있는 사람 수)

혼잡 노출 지수를 어떻게 붙이는가)
- congestion_exposure_index: 노선 단위(경부선 구간) 숫자 하나 -> 모든 행에 같은 값
- stop_congestion_index    : 정류장이 있는 동의 도로 기준 -> 정류장마다 다른 값 (seq로 붙임)
둘 다 시간대 구분이 없는 데이터라 24시간 같은 값이다.

CLI 예시:
  python  data-engineering/06BuildFeatures.py --route 9401
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

ANALYSIS_SEGMENTS = ["서울_도심루프"]


# %%
def load_inputs(route: str) -> dict[str, pd.DataFrame | None]:
    share_path = PROCESSED_DIR / f"route_share_{route}.parquet"
    stop_path = PROCESSED_DIR / "kec_stop_index_9401.parquet"
    return {
        "load_profile": pd.read_parquet(PROCESSED_DIR / f"load_profile_{route}.parquet"),
        "flow_profile": pd.read_parquet(PROCESSED_DIR / f"flow_profile_{route}.parquet"),
        "kec_overall": pd.read_parquet(PROCESSED_DIR / "kec_route_index_9401.parquet"),
        "route_share": pd.read_parquet(share_path) if share_path.exists() else None,
        "kec_stop": pd.read_parquet(stop_path) if stop_path.exists() else None,
    }


# %%
def merge_boarding_and_flow(load_profile: pd.DataFrame, flow_profile: pd.DataFrame) -> pd.DataFrame:
    """seq+hour 기준 left join. 04번은 서울 좌표 있는 정류장만 커버하므로
    성남 구간 정류장은 flow_pop이 NaN으로 남는다 — 여기서 드롭하지 않고
    남겨서, 이후 단계(modeling/)가 "왜 이 정류장은 값이 없는지" 명시적으로
    알 수 있게 한다."""
    before = load_profile["seq"].nunique()
    merged = load_profile.merge(
        flow_profile[["seq", "hour", "flow_pop", "n_cells_matched"]],
        on=["seq", "hour"],
        how="left",
    )
    missing = merged.loc[merged["flow_pop"].isna(), "seq"].nunique()
    if missing:
        print(f"flow_pop 없음(성남 구간 등, SKT 커버리지 밖): {missing}개 정류장")
    print(f"merge 후 정류장 수: {before} -> {merged['seq'].nunique()} (변화 없어야 정상, left join이므로)")
    return merged


# %%
def merge_route_share(df: pd.DataFrame, route_share: pd.DataFrame | None) -> pd.DataFrame:
    """03b 노선 분담률을 (표준버스정류장ID, hour)로 붙인다. 파일이 없으면 NaN 컬럼만 만든다."""
    df = df.copy()
    if route_share is None:
        print("route_share 없음 -> 분담률 컬럼 NaN (09번은 분담률 없는 신호만 사용)")
        df["share_all_stop"] = np.nan
        df["share_all"] = np.nan
        df["share_express"] = np.nan
        return df
    rs = route_share[["표준버스정류장ID", "hour", "share_all_stop", "share_all", "share_express", "n_routes"]].copy()
    rs["표준버스정류장ID"] = rs["표준버스정류장ID"].astype(str)
    if "표준버스정류장ID" not in df.columns:
        raise KeyError("load_profile에 표준버스정류장ID가 없습니다. 03LoadProfile.py 산출물을 확인하세요.")
    df["표준버스정류장ID"] = df["표준버스정류장ID"].astype(str)
    return df.merge(rs, on=["표준버스정류장ID", "hour"], how="left")


# %%
def attach_congestion_index(df: pd.DataFrame, kec_overall: pd.DataFrame) -> pd.DataFrame:
    """노선 단위 혼잡 노출 지수(05번)를 모든 행에 같은 값으로 붙인다."""
    df = df.copy()
    df["congestion_exposure_index"] = float(kec_overall["congestion_exposure_index"].iloc[0])
    return df


def attach_weekday_profile(df: pd.DataFrame, wkdy_profile: pd.DataFrame | None) -> pd.DataFrame:
    """정류장별 평일/주말 유동인구(04번)를 seq로 붙인다. 없으면 빈 값."""
    if wkdy_profile is None:
        print("요일별 유동인구 없음 -> flow_weekday/flow_weekend 빈 값 (09번 통근 보정 꺼짐)")
        return df.assign(flow_weekday=np.nan, flow_weekend=np.nan)
    return df.merge(wkdy_profile[["seq", "flow_weekday", "flow_weekend"]], on="seq", how="left")


def attach_stop_congestion_index(df: pd.DataFrame, kec_stop: pd.DataFrame | None) -> pd.DataFrame:
    """정류장별 혼잡 지수(05번 --by-stop)를 seq로 붙인다. 없으면 빈 값."""
    if kec_stop is None:
        print("kec_stop 없음 -> stop_congestion_index 빈 값")
        return df.assign(stop_congestion_index=np.nan)
    return df.merge(kec_stop[["seq", "stop_congestion_index"]], on="seq", how="left")


# %%
def merge_all(
    load_profile: pd.DataFrame,
    flow_profile: pd.DataFrame,
    kec_overall: pd.DataFrame,
    route_share: pd.DataFrame | None,
    kec_stop: pd.DataFrame | None = None,
    wkdy_profile: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """파일을 읽고 쓰지 않고 합치기만 한다. (RunAll.py가 반경별로 여러 번 부름)"""
    merged = merge_boarding_and_flow(load_profile, flow_profile)
    before = merged["seq"].nunique()
    merged = merged[merged["segment"].isin(ANALYSIS_SEGMENTS)].reset_index(drop=True)
    print(f"분석 범위 {ANALYSIS_SEGMENTS}로 한정: 정류장 {before} -> {merged['seq'].nunique()}")
    merged = merge_route_share(merged, route_share)
    merged = attach_congestion_index(merged, kec_overall)
    merged = attach_stop_congestion_index(merged, kec_stop)
    merged = attach_weekday_profile(merged, wkdy_profile)
    return merged


def build_features(route: str) -> pd.DataFrame:
    return merge_all(**load_inputs(route))


# %%
def main() -> None:
    parser = argparse.ArgumentParser(description="03+04+05 산출물을 feature 테이블로 병합 (Phase 1 경계선)")
    parser.add_argument("--route", type=str, default="9401")
    args = parser.parse_args()
    features = build_features(args.route)

    out_path = PROCESSED_DIR / "features.parquet"
    features.to_parquet(out_path, index=False)

    print(f"\nrows: {len(features):,}")
    print(f"columns: {list(features.columns)}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()