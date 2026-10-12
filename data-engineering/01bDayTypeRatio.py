# 정류장별 평일/주말 승하차 비율 (공개 데이터. RunAll.py가 안심구역 안에서 실행)
"""
01bDayTypeRatio.py

01번 승하차 데이터(시간대별)는 한 달 치 합계라서 평일과 주말이 섞여 있다.
그래서 서울시 "버스노선별 정류장별 승하차 인원 정보"(일별, 시간대 없음)로
정류장마다 주말에 평일보다 얼마나 적게/많이 타는지 비율을 구한다.
03번이 이 비율로 시간대별 월 합계를 평일 하루 평균 / 주말 하루 평균으로 나눈다.

1. 일별 파일에서 9401만 남긴다
2. 날짜마다 평일 / 주말(토·일·공휴일)로 구분한다
3. 정류장마다
     평일 하루 평균 승차 = 평일 승차 합계 / 평일 일수
     주말 하루 평균 승차 = 주말 승차 합계 / 주말 일수
     r_board = 주말 하루 평균 / 평일 하루 평균   (하차도 같은 방식으로 r_alight)
4. 요일별(월~일)로도 같은 방식으로 하루 평균을 남긴다 (board_mon ~ board_sun)
   공휴일(8/17 대체공휴일)은 일요일과 같이 묶는다 ("sun" = 일요일·공휴일)

출처: BusStationBoardingMonth202608.csv

CLI: python data-engineering/01bDayTypeRatio.py --route 9401
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# 2026년 8월 공휴일 중 평일에 있는 날 (8/15 광복절은 토요일 -> 8/17 월요일 대체공휴일)
HOLIDAYS_DEFAULT = ["20260817"]
DOW_KEYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]  # sun = 일요일·공휴일
DOW_KR = {"mon": "월", "tue": "화", "wed": "수", "thu": "목", "fri": "금", "sat": "토", "sun": "일·공휴일"}

# 일별 파일 컬럼 (파일마다 표기가 조금 다를 수 있어 후보를 여러 개 둠)
COLUMN_CANDIDATES = {
    "date": ["사용일자", "USE_YMD", "USE_DT"],
    "route": ["노선번호", "RTE_NO", "BUS_ROUTE_NO"],
    "stop_id": ["표준버스정류장ID", "STOPS_ID", "BUS_STA_ID"],
    "board": ["승차총승객수", "GTON_TNOPE", "RIDE_PASGR_NUM"],
    "alight": ["하차총승객수", "GTOFF_TNOPE", "ALIGHT_PASGR_NUM"],
}


# %%
def load_daily(path: Path) -> pd.DataFrame:
    """일별 승하차 파일 로드. 인코딩이 utf-8 / cp949 중 무엇인지 몰라서 차례로 시도."""
    for enc in ("utf-8-sig", "cp949"):
        try:
            raw = pd.read_csv(path, encoding=enc, dtype=str, low_memory=False)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError("utf-8/cp949", b"", 0, 1, f"{path} 인코딩을 알 수 없습니다.")

    rename = {}
    for key, cands in COLUMN_CANDIDATES.items():
        hit = next((c for c in cands if c in raw.columns), None)
        if hit is None:
            raise KeyError(f"일별 파일에 '{key}' 컬럼이 없습니다. 후보 {cands} / 실제 컬럼: {list(raw.columns)}")
        rename[hit] = key
    df = raw.rename(columns=rename)[list(COLUMN_CANDIDATES)]
    df["date"] = df["date"].str.replace("-", "").str[:8]
    for c in ("board", "alight"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


# %%
def build_daytype_ratio(daily: pd.DataFrame, route: str, holidays: list[str] | None = None) -> pd.DataFrame:
    """정류장별 평일/주말 하루 평균 승하차와 비율."""
    holidays = set(holidays or HOLIDAYS_DEFAULT)
    d = daily[daily["route"].astype(str) == route].copy()
    if d.empty:
        raise ValueError(f"일별 파일에 노선 {route}가 없습니다.")
    dates = pd.to_datetime(d["date"], format="%Y%m%d")
    d["daytype"] = (dates.dt.dayofweek >= 5) | d["date"].isin(holidays)
    d["daytype"] = d["daytype"].map({True: "weekend", False: "weekday"})

    # 일수는 노선 전체 기준 (그 정류장에 아무도 안 탄 날도 하루로 셈)
    n_days = d.groupby("daytype")["date"].nunique()
    sums = d.groupby(["stop_id", "daytype"])[["board", "alight"]].sum().unstack("daytype", fill_value=0)

    out = pd.DataFrame(index=sums.index)
    for kind in ("board", "alight"):
        wd = sums[(kind, "weekday")] / n_days["weekday"]
        we = sums[(kind, "weekend")] / n_days["weekend"]
        out[f"{kind}_weekday"] = wd
        out[f"{kind}_weekend"] = we
        out[f"r_{kind}"] = we / wd.where(wd > 0)
    out["n_weekday_days"] = int(n_days["weekday"])
    out["n_weekend_days"] = int(n_days["weekend"])

    # 요일별 하루 평균 (공휴일은 일요일로)
    d["dow"] = dates.dt.dayofweek.map(dict(enumerate(DOW_KEYS)))
    d.loc[d["date"].isin(holidays), "dow"] = "sun"
    n_dow = d.groupby("dow")["date"].nunique()
    sums_dow = d.groupby(["stop_id", "dow"])[["board", "alight"]].sum().unstack("dow", fill_value=0)
    for kind in ("board", "alight"):
        for k in DOW_KEYS:
            col = (kind, k)
            out[f"{kind}_{k}"] = (sums_dow[col] if col in sums_dow else 0) / max(int(n_dow.get(k, 0)), 1)
    for k in DOW_KEYS:
        out[f"n_{k}_days"] = int(n_dow.get(k, 0))
    return out.reset_index().rename(columns={"stop_id": "표준버스정류장ID"})


# %%
def main() -> None:
    parser = argparse.ArgumentParser(description="정류장별 평일/주말 승하차 비율")
    parser.add_argument("--route", type=str, default="9401")
    parser.add_argument("--daily-file", type=str, default="BusStationBoardingMonth202608.csv")
    args = parser.parse_args()

    ratio = build_daytype_ratio(load_daily(RAW_DIR / args.daily_file), args.route)
    out_path = PROCESSED_DIR / f"daytype_ratio_{args.route}.parquet"
    ratio.to_parquet(out_path, index=False)
    print(f"평일 {ratio['n_weekday_days'].iloc[0]}일, 주말·공휴일 {ratio['n_weekend_days'].iloc[0]}일")
    print(ratio[["표준버스정류장ID", "r_board", "r_alight"]].describe().round(3).to_string())
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
