# 복수 노선 정류장에서 9401 몫(분담률) 계산 — 공개 데이터라 안심구역 밖에서 실행
"""
03bRouteShare.py

정류장 주변 유동인구(04번)는 '그 정류장에서 어떤 버스를 기다리는지' 구분하지 않는다.
서울역버스환승센터처럼 수십 개 노선이 서는 정류장이면 유동인구를 그대로 9401 대기 수요로
보면 과대 추정이 된다

그래서 같은 정류장·시간대의 서울 버스 전체 승차 중 9401이 차지하는 비중(분담률)을
공개 승하차 데이터로 계산해둔다. 09번 모델에서 유동인구 x 분담률을 '9401 대기 수요 신호'로 쓴다.

  share_all     = 9401 승차 / 그 정류장·시간대 서울 버스 전체 승차
  share_express = 9401 승차 / 그 정류장·시간대 광역버스 승차

한계)
서울시 데이터라 같은 정류장에 서는 경기도 버스 승차는 분모에 들어가지 않는다 -> 분담률 과대 추정 가능.
9401 승차 자체가 좌석 제약으로 잘린 값이라 분담률도 첨두시간엔 낮게 나올 수 있다(보수적 방향).

입력: data/processed/bus_long_all.parquet  (01IngestBus.py를 --route 없이 실행한 결과)
CLI: python data-engineering/03bRouteShare.py --route 9401
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

EXPRESS_TYPE_KEYWORD = "광역"


# %%
def build_route_share(bus_long_all: pd.DataFrame, route: str) -> pd.DataFrame:
    df = bus_long_all.copy()
    df["표준버스정류장ID"] = df["표준버스정류장ID"].astype(str)
    df["노선번호"] = df["노선번호"].astype(str)

    route_rows = df[df["노선번호"] == route]
    if route_rows.empty:
        raise ValueError(f"bus_long_all에 노선 {route}가 없습니다.")
    stop_ids = route_rows["표준버스정류장ID"].unique()
    at_stops = df[df["표준버스정류장ID"].isin(stop_ids)]

    key = ["표준버스정류장ID", "hour"]
    # 같은 정류장을 한 노선이 두 번 지나는 경우(순환 구간)도 있어 노선 승차는 정류장ID 기준으로 합산
    board_route = route_rows.groupby(key)["boarding"].sum().rename("board_route")
    board_all = at_stops.groupby(key)["boarding"].sum().rename("board_all")
    is_express = at_stops["교통수단타입명"].astype(str).str.contains(EXPRESS_TYPE_KEYWORD)
    board_express = at_stops[is_express].groupby(key)["boarding"].sum().rename("board_express")
    n_routes = at_stops.groupby("표준버스정류장ID")["노선번호"].nunique().rename("n_routes")

    out = pd.concat([board_route, board_all, board_express], axis=1).reset_index()
    out = out.merge(n_routes.reset_index(), on="표준버스정류장ID", how="left")
    out["board_express"] = out["board_express"].fillna(0)
    # 분모가 0이면 분담률 정의 불가 -> NaN (09번에서 정류장 평균 분담률로 대체)
    out["share_all"] = out["board_route"] / out["board_all"].where(out["board_all"] > 0)
    out["share_express"] = out["board_route"] / out["board_express"].where(out["board_express"] > 0)
    return out.sort_values(key).reset_index(drop=True)


# %%
def main() -> None:
    parser = argparse.ArgumentParser(description="정류장x시간대 노선 분담률 (공개 데이터)")
    parser.add_argument("--route", type=str, default="9401")
    args = parser.parse_args()

    all_path = PROCESSED_DIR / "bus_long_all.parquet"
    if not all_path.exists():
        raise FileNotFoundError(
            f"{all_path} 없음. 먼저 python data-engineering/01IngestBus.py (--route 없이) 실행"
        )
    share = build_route_share(pd.read_parquet(all_path), args.route)

    out_path = PROCESSED_DIR / f"route_share_{args.route}.parquet"
    share.to_parquet(out_path, index=False)

    summary = share.groupby("표준버스정류장ID").agg(
        n_routes=("n_routes", "first"), share_all_mean=("share_all", "mean")
    )
    print(f"rows: {len(share):,}, stops: {share['표준버스정류장ID'].nunique()}")
    print(f"정차 노선 수 분포:\n{summary['n_routes'].describe().to_string()}")
    print(f"9401 평균 분담률(정류장별 평균의 분포):\n{summary['share_all_mean'].describe().to_string()}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
