# 도로명/행정구역으로 경부선 구간 필터, 노선 단위 혼잡 노출 지수
# 안심구역에서...

"""
05KecRouteIndex.py

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
  python data-engineering/05KecRouteIndex.py --list-only
  python data-engineering/05KecRouteIndex.py --road-names 경부고속도로 --sigungu-names 성남시분당구 서초구
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# TODO(안심구역): 9401이 실제로 타는 경부선 구간의 도로명/행정구역으로 교체.
ROAD_NAME_CANDIDATES: list[str] = ["경부고속도로"]
SIGUNGU_NAME_CANDIDATES: list[str] = ["성남시분당구", "서초구"]


# %%
def load_road_indicator(path: Path) -> pd.DataFrame:
    """EX_011 차량통행지표 로드"""
    df = pd.read_csv(path, encoding="utf-8")
    rename_map = {
        "road_name": "ROAD_NAME",
        "velocity_AVRG_CG": "velocity_CG_AVRG",
        "sigungu_name": "SIGUNGU_NAME",
        "road_length": "ROAD_LENGTH",
        "lev5_5_link_id": "LEVEL5_5_LINKID",
    }
    df = df.rename(columns=rename_map)
    return df


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
# ---------------------------------------------------------------------
# 정류장별 혼잡 지수
# 노선 단위 지수는 숫자 하나라 정류장끼리 비교가 안 된다.
# 그래서 정류장이 있는 동(02b번 결과)의 도로만 골라 같은 방식으로 정류장마다 지수를 만든다.
#
# 1. 정류장마다 도로 고르기: 시도 + 시군구 + 읍면동(EMD_NAME)이 모두 같은 링크
#    동 이름 표기가 달라서 하나도 안 걸리면 -> 그 시군구 전체 도로로 대신 (match_level에 표시)
# 2. TI_CG, FRIN_CG 0~1 변환은 11개 정류장 도로를 모두 합친 범위로 한다
#    (정류장마다 따로 하면 정류장끼리 비교가 안 됨)
# 3. stop_congestion_index = 세 지표 평균을 도로 길이로 가중평균 x 100  (노선 단위와 같은 공식)
# ---------------------------------------------------------------------
def _norm_name(s: pd.Series) -> pd.Series:
    """동 이름 비교용: 공백, 가운뎃점, 마침표 제거 ("종로1·2·3·4가동" = "종로1.2.3.4가동")."""
    return s.astype(str).str.replace(r"[\s·.,ㆍ]", "", regex=True)


def select_stop_links(
    road_df: pd.DataFrame, stop: pd.Series, emd_override: dict | None = None
) -> tuple[pd.DataFrame, str]:
    """한 정류장 주변 도로 링크와 매칭 수준.

    순서대로 시도해서 먼저 걸리는 걸 쓴다
    1. stop_emd_override에 직접 적은 EMD_NAME
    2. 동 코드: EMD_CODE == 행정동 코드(10자리 또는 8자리)
    3. 동 이름: EMD_NAME == 행정동 이름
    4. 다 안 걸리면 그 시군구 전체 도로
    """
    in_sgg = road_df[(road_df["SIDO_NAME"] == stop["sido_nm"]) & (road_df["SIGUNGU_NAME"] == stop["sgg_nm"])]
    override = (emd_override or {}).get(int(stop["seq"]))
    if override:
        hit = in_sgg[_norm_name(in_sgg["EMD_NAME"]).isin(_norm_name(pd.Series(override)))]
        if not hit.empty:
            return hit, "읍면동(직접 지정)"
    codes = {str(stop.get("adm_cd") or ""), str(stop.get("adm_cd") or "")[:8]} - {""}
    hit = in_sgg[in_sgg["EMD_CODE"].astype(str).isin(codes)]
    if not hit.empty:
        return hit, "읍면동(코드)"
    hit = in_sgg[_norm_name(in_sgg["EMD_NAME"]) == _norm_name(pd.Series([stop["dong_nm"]]))[0]]
    if not hit.empty:
        return hit, "읍면동(이름)"
    return in_sgg, "시군구(동 불일치)"


def compute_stop_congestion_index(
    road_df: pd.DataFrame, stop_dong: pd.DataFrame, emd_override: dict | None = None
) -> pd.DataFrame:
    selected = {}
    for _, stop in stop_dong.iterrows():
        links, level = select_stop_links(road_df, stop, emd_override)
        if not level.startswith("읍면동"):
            print(f"⚠ seq{stop['seq']} {stop['stop_name']}: '{stop['dong_nm']}' 도로 없음 -> {stop['sgg_nm']} 전체로 대신")
        selected[stop["seq"]] = (links, level)

    # 2. 11개 정류장 도로를 합친 범위로 TI_CG, FRIN_CG를 0~1로
    pool = pd.concat([l for l, _ in selected.values()]).drop_duplicates("LEVEL5_5_LINKID")
    ti_norm = dict(zip(pool["LEVEL5_5_LINKID"], _minmax(pool["TI_CG"])))
    frin_norm = dict(zip(pool["LEVEL5_5_LINKID"], _minmax(pool["FRIN_CG"])))

    rows = []
    for _, stop in stop_dong.iterrows():
        links, level = selected[stop["seq"]]
        row = {"seq": stop["seq"], "stop_name": stop["stop_name"], "sgg_nm": stop["sgg_nm"],
               "dong_nm": stop["dong_nm"], "match_level": level, "n_links": len(links)}
        if links.empty:
            row.update({"road_length_sum_m": np.nan, "speed_drop_ratio_wavg": np.nan, "stop_congestion_index": np.nan})
        else:
            w = links["ROAD_LENGTH"]
            sdr = 1 - links["velocity_CG_AVRG"] / links["velocity_AVRG_NRMLT"]
            comp = (sdr + links["LEVEL5_5_LINKID"].map(ti_norm) + links["LEVEL5_5_LINKID"].map(frin_norm)) / 3
            row.update({
                "road_length_sum_m": float(w.sum()),
                "speed_drop_ratio_wavg": float((sdr * w).sum() / w.sum()),
                "stop_congestion_index": float((comp * w).sum() / w.sum() * 100),
            })
        rows.append(row)
    return pd.DataFrame(rows)


# %%
def list_emd_names(road_df: pd.DataFrame, stop_dong: pd.DataFrame) -> pd.DataFrame:
    """정류장이 있는 시군구들의 EMD_NAME 목록. 동 이름 표기가 다를 때 RunAll.py의 stop_emd_override를 채우는 용도."""
    sgg = stop_dong["sgg_nm"].dropna().unique()
    d = road_df[road_df["SIGUNGU_NAME"].isin(sgg)]
    return d.groupby(["SIGUNGU_NAME", "EMD_NAME"]).size().rename("n_links").reset_index()


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
    parser.add_argument("--by-stop", action="store_true", help="정류장별 혼잡 지수 (02b번 stop_dong 필요)")
    parser.add_argument("--list-emd", action="store_true", help="정류장 시군구의 EMD_NAME 목록 출력")
    args = parser.parse_args()

    road_df = load_road_indicator(RAW_DIR / args.road_indicator_file)

    if args.by_stop or args.list_emd:
        stop_dong = pd.read_parquet(PROCESSED_DIR / "stop_dong_9401.parquet")
        if args.list_emd:
            print(list_emd_names(road_df, stop_dong).to_string(index=False))
            return
        by_stop = compute_stop_congestion_index(road_df, stop_dong)
        print(by_stop.round(3).to_string(index=False))
        by_stop.to_parquet(PROCESSED_DIR / "kec_stop_index_9401.parquet", index=False)
        return

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