# 정류장 반경 안 SKT 유동인구 셀 합산 -> 정류장x시간대 유동인구
"""
05SktCatchment.py

SKT 유동인구(50m x 50m 셀, 서울 전역)를 정류장 반경 안에서 합산한다.

왜 필요한가)
교통카드는 탄 사람만 기록한다. 버스가 꽉 차서 못 탄 사람은 기록이 없다.
정류장 주변 유동인구는 "그 시간에 정류장 근처에 있던 사람 수"라는 별개의 축이라,
07번에서 원래 타려던 사람 수(수요)를 추정하는 재료로 쓴다.

1. 시간대별 (SKT_003: STD_YM, X_COORD, Y_COORD, TMST_00 ~ TMST_23)
   -> 정류장x시간대 flow_pop
2. 요일별 (SKT_002: FLOW_POP_CNT_MON ~ SUN, 화요일만 TUS)
   -> 정류장별 flow_weekday(월~금 평균), flow_weekend(토·일 평균), flow_mon ~ flow_sun
   07번 그 요일 유동인구 = 시간대별 x (그 요일 / 7개 요일 평균)에 씀

반경: 100~200m (CONFIG radii). 한 셀이 여러 정류장 반경에 걸리면 양쪽에 다 더한다
(정류장 각자의 "주변에 이만큼 있었다"라는 관점. SKT 유동인구 자체도 중복 집계를 허용하는 지표)
좌표: SKT X/Y는 UTM-K(m). 정류장 좌표도 01번에서 UTM-K로 바꿔 둠.
SKT는 서울만 있으므로 서울 정류장(01번 is_seoul)만 계산한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FLOW_HOURS = [f"TMST_{h:02d}" for h in range(24)]
WKDY_COLS = {  # 정의서 표기 그대로 (화요일은 TUS)
    "MON": "FLOW_POP_CNT_MON", "TUE": "FLOW_POP_CNT_TUS", "WED": "FLOW_POP_CNT_WED",
    "THU": "FLOW_POP_CNT_THU", "FRI": "FLOW_POP_CNT_FRI", "SAT": "FLOW_POP_CNT_SAT", "SUN": "FLOW_POP_CNT_SUN",
}

_PRINTED: set[str] = set()


def _print_once(msg: str) -> None:
    if msg not in _PRINTED:
        _PRINTED.add(msg)
        print(msg)


def seoul_stops(stops: pd.DataFrame) -> pd.DataFrame:
    """좌표가 있는 서울 정류장만."""
    s = stops[stops["is_seoul"] & stops["utmk_x"].notna()][["seq", "sttn_id", "stop_name", "utmk_x", "utmk_y"]]
    dropped = stops["seq"].nunique() - s["seq"].nunique()
    if dropped:
        _print_once(f"SKT 유동인구 대상 제외(서울 밖이거나 좌표 없음): {dropped}개 정류장 (성남 구간 등)")
    return s.drop_duplicates("seq").reset_index(drop=True)


# %%
def load_skt(path: Path, std_ym: str | None = None) -> pd.DataFrame:
    """SKT 파일 (구분자 |). X_COORD/Y_COORD는 문자로 와서 숫자로 바꿈."""
    df = pd.read_csv(path, sep="|", encoding="utf-8", dtype={"STD_YM": str})
    if std_ym is not None:
        df = df[df["STD_YM"] == std_ym]
        if df.empty:
            raise ValueError(f"{path.name}에 STD_YM={std_ym} 행이 없습니다.")
    df["utmk_x"] = df["X_COORD"].astype(float)
    df["utmk_y"] = df["Y_COORD"].astype(float)
    return df


def _sum_within_radius(stops: pd.DataFrame, cells: pd.DataFrame, value_cols: list[str], radius_m: float):
    """정류장마다 반경 안 셀들의 값 합계. 반환: (정류장별 셀 개수, 정류장 x 컬럼 합계)"""
    pad = radius_m + 1  # 정류장 주변 사각형 밖의 셀은 미리 제외 (계산 시간 단축)
    in_box = (cells["utmk_x"].between(stops["utmk_x"].min() - pad, stops["utmk_x"].max() + pad)
              & cells["utmk_y"].between(stops["utmk_y"].min() - pad, stops["utmk_y"].max() + pad))
    cells = cells[in_box]
    stop_xy = stops[["utmk_x", "utmk_y"]].to_numpy()
    cell_xy = cells[["utmk_x", "utmk_y"]].to_numpy()
    within = np.sqrt(((stop_xy[:, None, :] - cell_xy[None, :, :]) ** 2).sum(axis=2)) <= radius_m
    return within.sum(axis=1), within.astype(float) @ cells[value_cols].to_numpy(float)


# %%
def build_flow_profile(stops: pd.DataFrame, flow: pd.DataFrame, radius_m: float) -> pd.DataFrame:
    """정류장x시간대 주변 유동인구 (long: seq, hour, flow_pop, n_cells_matched)."""
    s = seoul_stops(stops)
    n_cells, sums = _sum_within_radius(s, flow, FLOW_HOURS, radius_m)
    none = s.loc[n_cells == 0, "stop_name"].tolist()
    if none:
        print(f"반경 {radius_m}m 안에 유동인구 셀이 없는 정류장 {len(none)}개: {none}")
    wide = pd.DataFrame(sums, columns=range(24))
    wide["seq"] = s["seq"].to_numpy()
    wide["n_cells_matched"] = n_cells
    long = wide.melt(id_vars=["seq", "n_cells_matched"], var_name="hour", value_name="flow_pop")
    long["hour"] = long["hour"].astype(int)
    return long.sort_values(["seq", "hour"]).reset_index(drop=True)


def build_weekday_profile(stops: pd.DataFrame, wkdy: pd.DataFrame, radius_m: float) -> pd.DataFrame:
    """정류장별 평일 평균 / 주말 평균 / 요일별 유동인구."""
    s = seoul_stops(stops)
    _, sums = _sum_within_radius(s, wkdy, list(WKDY_COLS.values()), radius_m)
    by_day = pd.DataFrame(sums, columns=list(WKDY_COLS))
    out = pd.DataFrame({"seq": s["seq"].to_numpy()})
    out["flow_weekday"] = by_day[["MON", "TUE", "WED", "THU", "FRI"]].mean(axis=1).to_numpy()
    out["flow_weekend"] = by_day[["SAT", "SUN"]].mean(axis=1).to_numpy()
    for k in WKDY_COLS:
        out[f"flow_{k.lower()}"] = by_day[k].to_numpy()
    return out
