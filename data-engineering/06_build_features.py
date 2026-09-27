"""
06_build_features.py

03_load_profile.py(승하차) + 04_skt_catchment.py(유동인구) +
05_kec_route_index.py(혼잡 노출 지수)를 정류장×시간대 단위 feature 테이블 하나로 병합한다.

순수 병합 원칙)
데이터 엔지니어링 단계는 같은 입력이면 항상 같은 출력이어야 하고,
가중치/임계값 같은 통계적 판단이 들어가면 안 된다. 그래서 이 파일은
boarding/flow_pop 같은 원자료를 그대로 합치는 것까지만 하고, 
capture_rate 같은 파생 비율도 만들지 않는다

혼잡 노출 지수를 어디에 붙이는가)
05번은 정류장×시간대 단위가 아니라 노선 전체(경부선 구간) 단위의
값 하나다. 그래서 congestion_exposure_index는 segment가 "성남_상행"/"성남_하행"인
행에만 채우고, 도심 구간은 NaN으로 비워둔다 
이 구분을 안 하면 "도심구간도 고속도로만큼 막힌다"는 잘못된 신호를 모델에 주게 된다.

시군구별 breakdown(05번의 kec_route_index_9401_by_sigungu.parquet)을
"성남_상행" vs "성남_하행"에 각각 다르게 매핑하고 싶다면
 --segment-sigungu-map으로 수동 매핑을 넘길 수 있게 해뒀다. 
안 넘기면 전체 평균(overall index) 하나를 양쪽에 동일하게 쓴다

CLI 예시:
  python  data-engineering/06_build_features.py --route 9401
  python  data-engineering/06_build_features.py --route 9401 --segment-sigungu-map 성남_상행:성남시분당구 성남_하행:서초구
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

HIGHWAY_SEGMENTS = ["성남_상행", "성남_하행"]


# %%
def load_inputs(route: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    load_profile = pd.read_parquet(PROCESSED_DIR / f"load_profile_{route}.parquet")
    flow_profile = pd.read_parquet(PROCESSED_DIR / f"flow_profile_{route}.parquet")
    kec_overall = pd.read_parquet(PROCESSED_DIR / "kec_route_index_9401.parquet")
    kec_by_sigungu = pd.read_parquet(PROCESSED_DIR / "kec_route_index_9401_by_sigungu.parquet")
    return load_profile, flow_profile, kec_overall, kec_by_sigungu


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
def _parse_segment_sigungu_map(pairs: list[str] | None) -> dict[str, str]:
    """--segment-sigungu-map 성남_상행:성남시분당구 성남_하행:서초구 형태 파싱."""
    if not pairs:
        return {}
    out = {}
    for p in pairs:
        seg, sigungu = p.split(":", 1)
        out[seg] = sigungu
    return out


# %%
def attach_congestion_index(
    df: pd.DataFrame,
    kec_overall: pd.DataFrame,
    kec_by_sigungu: pd.DataFrame,
    segment_sigungu_map: dict[str, str],
) -> pd.DataFrame:
    """혼잡 노출 지수를 고속도로 구간(성남_상행/하행)에만 broadcast.
    서울 도심 구간은 고속도로 지표가 의미 없으므로 NaN으로 남긴다."""
    df = df.copy()
    df["congestion_exposure_index"] = np.nan

    overall_index = kec_overall["congestion_exposure_index"].iloc[0]

    for seg in HIGHWAY_SEGMENTS:
        sigungu = segment_sigungu_map.get(seg)
        if sigungu is not None:
            match = kec_by_sigungu[kec_by_sigungu["SIGUNGU_NAME"] == sigungu]
            value = match["congestion_exposure_index"].iloc[0] if not match.empty else overall_index
            if match.empty:
                print(f"⚠ {seg} <- {sigungu} 매핑 실패, 전체 평균 지수로 대체")
        else:
            value = overall_index
        df.loc[df["segment"] == seg, "congestion_exposure_index"] = value

    return df


# %%
def build_features(route: str, segment_sigungu_map: dict[str, str]) -> pd.DataFrame:
    load_profile, flow_profile, kec_overall, kec_by_sigungu = load_inputs(route)
    merged = merge_boarding_and_flow(load_profile, flow_profile)
    merged = attach_congestion_index(merged, kec_overall, kec_by_sigungu, segment_sigungu_map)
    return merged


# %%
def main() -> None:
    parser = argparse.ArgumentParser(description="03+04+05 산출물을 feature 테이블로 병합 (Phase 1 경계선)")
    parser.add_argument("--route", type=str, default="9401")
    parser.add_argument(
        "--segment-sigungu-map", type=str, nargs="*", default=None,
        help="예: 성남_상행:성남시분당구 성남_하행:서초구 (미지정시 전체 평균 지수 사용)",
    )
    args = parser.parse_args()

    segment_sigungu_map = _parse_segment_sigungu_map(args.segment_sigungu_map)
    features = build_features(args.route, segment_sigungu_map)

    out_path = PROCESSED_DIR / "features.parquet"
    features.to_parquet(out_path, index=False)

    print(f"\nrows: {len(features):,}")
    print(f"columns: {list(features.columns)}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()