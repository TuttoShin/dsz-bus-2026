# 도로명/행정구역으로 경부선 구간 필터, 노선 단위 혼잡 노출 지수
# 안심구역에서...

"""
05_kec_route_index.py

한국도로공사/KOTI 차량통행지표에서 9401이 지나는 경부선 구간만
도로명+행정구역으로 필터링하고, 노선 단위 혼잡 노출 지수로 압축한다.

왜 도로명만으로는 안 되는가)
"경부고속도로"라는 ROAD_NAME은 전국(서울~부산)에 걸쳐 있다. 도로명만으로
필터링하면 9401과 무관한 구간(대전, 부산 등)의 링크까지 섞여 들어와서 혼잡 지수가 왜곡된다. 
그래서 ROAD_NAME과 SIGUNGU_NAME(또는 SIDO_NAME) 두 조건을 AND로 걸어서, 
9401이 실제로 주행하는 시군구 구간의 링크만 남긴다.

SIGUNGU_NAME_CANDIDATES는 placeholder

노선 단위 지수를 만드는 방식)
매칭된 링크마다 ROAD_LENGTH(길이)로 가중 평균해서 지표를 합친다 
-> 링크 하나가 튀는 값이어도 그 링크가 짧으면 전체 지수에 크게 영향을 주지 않는다.

구성 지표:
  - speed_drop_ratio = 1 - velocity_CG_AVRG/velocity_AVRG_NRMLT
    (혼잡시 평균속도가 정상시 대비 얼마나 떨어지는가. 0=안 막힘, 1에 가까울수록 거의 서있는 수준)
  - TI_CG(혼잡시간강도), FRIN_CG(혼잡빈도강도): 정의서상 절대 스케일을 문서에서 명시하지 않아, 매칭된 링크 집합 내에서 min-max 정규화 후 사용
  - congestion_exposure_index = 위 세 요소의 동일 가중 평균 * 100 (0~100)
    -> 절대적인 "혼잡도 %"로 과대 해석하지 말 것. 
    어디까지나 '이 매칭 구간이 다른 구간 대비 상대적으로 얼마나 막히는가'의 상대 지수.

SIGUNGU_NAME 단위로도 별도 breakdown을 남겨서(그룹별 표), 나중에 03번의 성남_상행/성남_하행 세그먼트에 매핑할 실마리를 만들어둔다 
정확한 1:1 매핑(어느 시군구가 상행이고 어느 게 하행인지)은 안심구역에서 실제 행정구역 순서를 보고 사람이 판단해야 한다.

CLI 예시:
  python data-engineering/05_kec_route_index.py --list-only
  python data-engineering/05_kec_route_index.py --road-names 경부고속도로 --sigungu-names 성남시분당구 서초구
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# TODO(안심구역): 9401이 실제로 타는 경부선 구간의 도로명/행정구역으로 교체.
ROAD_NAME_CANDIDATES: list[str] = ["경부고속도로"]
SIGUNGU_NAME_CANDIDATES: list[str] = ["성남시분당구", "서초구"]


# %%
def load_road_indicator(path: Path) -> pd.DataFrame:
    """EX_011 차량통행지표 로드"""
    return pd.read_csv(path, encoding="utf-8")


# %%
def list_unique_road_names(road_df: pd.DataFrame, top_n: int = 50) -> pd.Series:
    return road_df["ROAD_NAME"].value_counts().head(top_n)


# %%
def list_unique_sigungu_on_road(road_df: pd.DataFrame, road_names: list[str], top_n: int = 50) -> pd.Series:
    """특정 도로명(예: 경부고속도로)이 지나는 시군구 목록. 9401의 실제 경유 구간을 좁히기 위해 안심구역에서 가장 먼저 돌려볼 것
    -> 여기서 나온 값 중 9401이 실제로 지나는 구간만 골라 SIGUNGU_NAME_CANDIDATES에 채운다."""
    on_road = road_df[road_df["ROAD_NAME"].isin(road_names)]
    return on_road["SIGUNGU_NAME"].value_counts().head(top_n)


# %%
def filter_route_segments(
    road_df: pd.DataFrame, road_names: list[str], sigungu_names: list[str]
) -> pd.DataFrame:
    """도로명 + 행정구역 이중 필터로 9401 경유 구간만 남긴다."""
    filtered = road_df[road_df["ROAD_NAME"].isin(road_names)]
    if sigungu_names:
        filtered = filtered[filtered["SIGUNGU_NAME"].isin(sigungu_names)]
    return filtered.copy()


# %%
def _minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


# %%
def compute_congestion_exposure_index(segments: pd.DataFrame) -> dict:
    """매칭된 링크들을 도로 길이로 가중 평균해서 노선 단위 지수 하나로 압축.

    빈 매칭(segments가 비어있음)이면 필터 조건이 잘못됐다는 신호이므로 명시적으로
    에러를 낸다 — 조용히 NaN을 리턴하면 "지수가 0이라 안 막힌다"로 오해할 수 있다.
    """
    if segments.empty:
        raise ValueError(
            "필터링된 링크가 0건입니다. --road-names/--sigungu-names 조합을 확인하세요 "
            "(list_unique_sigungu_on_road()로 먼저 실제 표기를 확인할 것)."
        )

    w = segments["ROAD_LENGTH"]
    w_sum = w.sum()

    speed_drop_ratio = 1 - (segments["velocity_CG_AVRG"] / segments["velocity_AVRG_NRMLT"])
    ti_norm = _minmax(segments["TI_CG"])
    frin_norm = _minmax(segments["FRIN_CG"])

    def wavg(x: pd.Series) -> float:
        return float((x * w).sum() / w_sum)

    composite = wavg((speed_drop_ratio + ti_norm + frin_norm) / 3) * 100

    return {
        "n_links": int(segments["LEVEL5_5_LINKID"].nunique()),
        "road_length_sum_m": float(w_sum),
        "velocity_avg_wavg": wavg(segments["velocity_AVRG"]),
        "velocity_congested_avg_wavg": wavg(segments["velocity_CG_AVRG"]),
        "speed_drop_ratio_wavg": wavg(speed_drop_ratio),
        "ti_cg_wavg": wavg(segments["TI_CG"]),
        "frin_cg_wavg": wavg(segments["FRIN_CG"]),
        "congestion_exposure_index": composite,
    }


# %%
def breakdown_by_sigungu(segments: pd.DataFrame) -> pd.DataFrame:
    """시군구별 서브 지수. 03번 segment(성남_상행/하행)에 나중에 매핑할 실마리용."""
    rows = []
    for name, grp in segments.groupby("SIGUNGU_NAME"):
        idx = compute_congestion_exposure_index(grp)
        idx["SIGUNGU_NAME"] = name
        rows.append(idx)
    return pd.DataFrame(rows).sort_values("congestion_exposure_index", ascending=False)


# %%
def main() -> None:
    parser = argparse.ArgumentParser(description="9401 경유 경부선 구간 혼잡 노출 지수")
    parser.add_argument("--road-indicator-file", type=str, default="TB_KOTI_ROAD TRAFFIC INDICATOR.csv")
    parser.add_argument("--road-names", type=str, nargs="+", default=ROAD_NAME_CANDIDATES)
    parser.add_argument("--sigungu-names", type=str, nargs="*", default=SIGUNGU_NAME_CANDIDATES)
    parser.add_argument(
        "--list-only", action="store_true",
        help="필터 없이 ROAD_NAME과 (있다면) 그 도로가 지나는 SIGUNGU_NAME 목록만 출력",
    )
    args = parser.parse_args()

    road_df = load_road_indicator(RAW_DIR / args.road_indicator_file)

    if args.list_only:
        print("[ROAD_NAME 상위 빈도]")
        print(list_unique_road_names(road_df).to_string())
        if args.road_names:
            print(f"\n[{args.road_names}가 지나는 SIGUNGU_NAME 상위 빈도]")
            print(list_unique_sigungu_on_road(road_df, args.road_names).to_string())
        return

    segments = filter_route_segments(road_df, args.road_names, args.sigungu_names)

    overall = compute_congestion_exposure_index(segments)
    print(f"[9401 경유 경부선 구간 혼잡 노출 지수]")
    for k, v in overall.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    sub = breakdown_by_sigungu(segments)
    print("\n[시군구별 breakdown]")
    print(sub.to_string(index=False))

    out_path = PROCESSED_DIR / "kec_route_index_9401.parquet"
    pd.DataFrame([overall]).to_parquet(out_path, index=False)
    sub.to_parquet(PROCESSED_DIR / "kec_route_index_9401_by_sigungu.parquet", index=False)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()