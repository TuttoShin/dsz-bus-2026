"""
07_rule_based_score.py

data_engineering/06_build_features.py 산출물(data/processed/features.parquet)에서
재차(03) / 유동인구(04) / 혼잡지수(05) 세 신호를 정규화 후 가중합해서
"탑승 여유 지수(boarding_margin_index)"를 산출한다.

여기서부터는 가중치/정규화 방식 적용. features.parquet만 있으면 안심구역 밖에서 얼마든지 반복 실험 가능

로직
----
1. load_ratio = per_bus_avg / SEAT_CAPACITY
   재차(이미 버스에 몇 명이 타고 있었는가)를 좌석수 대비 비율로. per_bus_avg_low/high
   범위 중 중간값(per_bus_avg_mid)을 쓴다. 1을 넘으면(추정 재차가 좌석수 초과) 1로 clip
   -> "이미 꽉 찼다"는 신호로만 쓰고, 초과분 크기 차이는 더 반영하지 않는다.

2. flow_norm = min-max(flow_pop)  (전체 정류장×시간대 기준)
   정류장 주변 수요 압력. 절대값이 아니라 "이 데이터셋 안에서 상대적으로 얼마나
   높은가"이므로, 데이터가 바뀌면(다른 달/다른 노선) 재계산해야 한다.

3. congestion_norm = min-max(congestion_exposure_index)
   고속도로 구간(성남_상행/하행)에만 값이 있고 도심 구간은 NaN. 결측 그대로 둔다.

4. burden_index = 가중합. congestion_norm이 NaN인 행(도심 구간)은 그 항을 빼고
   나머지 가중치를 합이 1이 되도록 재정규화해서 계산한다. 그냥 0으로 채우면
   "그 구간은 혼잡이 전혀 없다"는 잘못된 신호를 주게 되므로, 아예 그 신호가
   없다는 걸 가중치 재분배로 반영한다.

5. boarding_margin_index = (1 - burden_index) * 100
   0~100, 높을수록 "여유", 낮을수록 "탑승 실패 위험".

SEAT_CAPACITY는 9401 실제 좌석수 41석!

CLI 예시:
  python modeling/07_rule_based_score.py --route 9401
  python modeling/07_rule_based_score.py --route 9401 --seat-capacity 45 --w-load 0.5 --w-flow 0.3 --w-congestion 0.2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# TODO(안심구역 전): 9401 실제 좌석수로 교체
SEAT_CAPACITY_DEFAULT = 41

SCORE_THRESHOLDS = {"여유": 70, "보통": 40}  # margin_index >= 70: 여유, 40~70: 보통, <40: 위험


# %%
def load_features(route: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "features.parquet")


# %%
def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        print(f"'{s.name}' 값이 전부 동일하거나 결측이라 min-max 정규화 불가 (NaN 처리됨)")
        return pd.Series(np.nan, index=s.index)
    return (s - lo) / (hi - lo)


# %%
def add_normalized_signals(df: pd.DataFrame, seat_capacity: float) -> pd.DataFrame:
    df = df.copy()
    df["per_bus_avg_mid"] = (df["per_bus_avg_low"] + df["per_bus_avg_high"]) / 2
    df["load_ratio"] = (df["per_bus_avg_mid"] / seat_capacity).clip(upper=1.3)
    df["flow_norm"] = _minmax(df["flow_pop"])
    # congestion_exposure_index는 05번에서 이미 0~100으로 정규화된 지수이자
    # 세그먼트당 상수값이라 min-max 대상이 아님, 스케일만 0~1로 맞춘다.
    df["congestion_norm"] = df["congestion_exposure_index"] / 100.0
    return df


# %%
def compute_burden_index(df: pd.DataFrame, w_load: float, w_flow: float, w_congestion: float) -> pd.Series:
    """결측(congestion_norm이 NaN인 도심 구간)은 그 항을 빼고 나머지 가중치를
    재정규화해서 계산한다."""
    weights = pd.DataFrame({
        "load_ratio": w_load,
        "flow_norm": w_flow,
        "congestion_norm": w_congestion,
    }, index=df.index)

    values = df[["load_ratio", "flow_norm", "congestion_norm"]]
    available = values.notna()
    weights = weights.where(available, 0.0)

    weight_sum = weights.sum(axis=1)
    if (weight_sum == 0).any():
        raise ValueError("모든 신호가 결측인 행이 있습니다 — load_ratio/flow_norm 계산을 먼저 점검하세요.")

    weighted = (values.fillna(0) * weights).sum(axis=1)
    return weighted / weight_sum


# %%
def compute_margin_score(
    df: pd.DataFrame, seat_capacity: float, w_load: float, w_flow: float, w_congestion: float
) -> pd.DataFrame:
    df = add_normalized_signals(df, seat_capacity)
    df["burden_index"] = compute_burden_index(df, w_load, w_flow, w_congestion)
    df["boarding_margin_index"] = ((1 - df["burden_index"]) * 100).clip(0, 100)

    def _label(score: float) -> str:
        if pd.isna(score):
            return "산정불가"
        if score >= SCORE_THRESHOLDS["여유"]:
            return "여유"
        if score >= SCORE_THRESHOLDS["보통"]:
            return "보통"
        return "위험"

    df["margin_label"] = df["boarding_margin_index"].apply(_label)
    return df


# %%
def lowest_margin(df: pd.DataFrame, segment: str | None = None, top_n: int = 10) -> pd.DataFrame:
    """탑승 여유가 가장 적은(=탑승 실패 위험이 가장 큰) 정류장×시간대 top_n."""
    d = df if segment is None else df[df["segment"] == segment]
    cols = [
        "segment", "stop_label", "seq", "hour", "per_bus_avg_mid", "flow_pop",
        "congestion_exposure_index", "load_ratio", "flow_norm", "congestion_norm",
        "boarding_margin_index", "margin_label",
    ]
    return d.nsmallest(top_n, "boarding_margin_index")[cols]


# %%
def main() -> None:
    parser = argparse.ArgumentParser(description="재차/유동인구/혼잡지수 가중합 -> 탑승 여유 지수")
    parser.add_argument("--route", type=str, default="9401")
    parser.add_argument("--seat-capacity", type=float, default=SEAT_CAPACITY_DEFAULT)
    parser.add_argument("--w-load", type=float, default=0.35)
    parser.add_argument("--w-flow", type=float, default=0.4)
    parser.add_argument("--w-congestion", type=float, default=0.25)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    w_total = args.w_load + args.w_flow + args.w_congestion
    if abs(w_total - 1.0) > 1e-6:
        print(f"가중치 합이 {w_total}입니다(1.0 아님). 상대적 비중으로만 쓰이니 계속 진행은 합니다.")

    features = load_features(args.route)
    scored = compute_margin_score(features, args.seat_capacity, args.w_load, args.w_flow, args.w_congestion)

    out_path = PROCESSED_DIR / "scores.parquet"
    scored.to_parquet(out_path, index=False)

    print(f"rows: {len(scored):,}")
    print(f"\n[전체] 탑승 여유 지수 최하위 {args.top_n} (=탑승 실패 위험 최상위)")
    print(lowest_margin(scored, top_n=args.top_n).to_string(index=False))

    print(f"\n[서울 도심 구간] 탑승 여유 지수 최하위 {args.top_n}")
    print(lowest_margin(scored, segment="서울_도심루프", top_n=args.top_n).to_string(index=False))

    print(f"\nmargin_label 분포:\n{scored['margin_label'].value_counts().to_string()}")
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()