# 정류장x시간대 표 하나로 합치기 (04 승하차·재차·실제 운행 + 05 유동인구 + 9401 분담률)
"""
06BuildFeatures.py

아래 결과를 정류장x시간대 표 하나로 합친다. (RunAll.py가 반경별·요일별로 여러 번 부름)
  04LoadProfile.py  : 승하차, 재차, 실제 운행 대수·공백·몰림 (교통카드)
  04LoadProfile.py  : 9401 분담률 (교통카드, 정류장별 전체 노선 승차)
  05SktCatchment.py : 주변 유동인구 (시간대별 + 요일별)

분석 범위)
SKT 유동인구는 서울만 있으므로 서울 도심 구간 정류장만 남긴다.
(성남 구간은 03번 재차 계산에만 필요: 서울에 들어올 때 이미 타고 있는 사람 수)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANALYSIS_SEGMENTS = ["서울_도심루프"]

_PRINTED: set[str] = set()


def _print_once(msg: str) -> None:
    if msg not in _PRINTED:
        _PRINTED.add(msg)
        print(msg)


def merge_all(
    load_profile: pd.DataFrame,
    flow_profile: pd.DataFrame,
    share: pd.DataFrame | None = None,
    wkdy_profile: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """seq + hour로 합친다. 파일을 읽고 쓰지 않음."""
    df = load_profile.merge(flow_profile[["seq", "hour", "flow_pop", "n_cells_matched"]], on=["seq", "hour"], how="left")
    before = df["seq"].nunique()
    df = df[df["segment"].isin(ANALYSIS_SEGMENTS)].reset_index(drop=True)
    _print_once(f"분석 범위 {ANALYSIS_SEGMENTS}로 한정: 정류장 {before} -> {df['seq'].nunique()}")
    if share is None:
        df["share_all_stop"], df["share_all"] = np.nan, np.nan
    else:
        df = df.merge(share[["seq", "hour", "share_all_stop", "share_all"]], on=["seq", "hour"], how="left")
    if wkdy_profile is None:
        _print_once("요일별 유동인구 없음 -> 07번 통근 보정 꺼짐")
        df["flow_weekday"], df["flow_weekend"] = np.nan, np.nan
    else:
        cols = ["seq"] + [c for c in wkdy_profile.columns if c.startswith("flow_")]
        df = df.merge(wkdy_profile[cols], on="seq", how="left")
    missing = df.loc[df["flow_pop"].isna(), "seq"].nunique()
    if missing:
        _print_once(f"⚠ 서울 구간인데 유동인구가 없는 정류장 {missing}개 (좌표 확인)")
    return df
