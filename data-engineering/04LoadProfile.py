# 정류장x시간대 하루 평균 승하차·재차·실제 운행 (평일/주말/요일별) + 9401 분담률
"""
04LoadProfile.py

03번 버스별 기록을 정류장x시간대 하루 평균으로 묶는다. 평일 / 주말 / 요일(월~일)별로 따로.
  boarding, alighting   : 하루 평균 승차·하차 인원
  load_before_board     : 그 시간대에 온 버스들의 도착 시 재차 합 (하루 평균)
  load_lower            : 떠날 때 재차 합 (하루 평균)
  n_buses               : 하루 평균 실제 운행 대수
  headway_obs           : 실제 평균 배차간격(분) = 60 / n_buses
  gap_mean, gap_cv      : 앞차와의 실제 공백 평균, 몰림 정도(공백 표준편차 / 평균)
  full_bus_share        : 떠날 때 거의 만석(좌석 - 1명 이상)이었던 버스 비율

9401 분담률 (02번 정류장별 전체 노선 승차 집계)
  share_all_stop = 그 정류장에서 9401 승차 / 모든 노선 승차   (기간 전체, 정류장마다 값 하나)
  share_all      = 같은 값을 시간대별로 (비교용)
  share_weekday  = 시간대별, 평일(공휴일 제외)만 (장표 17쪽 "평일 18시" 그림용)
07번 수요신호 = 주변 유동인구 x 9401 분담률

정류장 사이 실제 차내 이동 시간(ride_times)은 09번 서비스 추천에 쓴다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_days(ev: pd.DataFrame, key: str) -> pd.DataFrame:
    """key: weekday / weekend / mon~sun / all"""
    if key == "all":
        return ev
    if key in ("weekday", "weekend"):
        return ev[ev["daytype"] == key]
    return ev[ev["dow"] == key]


def build_profile(ev: pd.DataFrame, stops: pd.DataFrame, key: str, seat_capacity: float = 41) -> pd.DataFrame:
    """정류장x시간대 하루 평균 (모든 정류장 x 24시간, 버스가 안 온 칸은 n_buses = 0)."""
    e = select_days(ev, key).copy()
    n_days = max(e["date"].nunique(), 1)
    e["load_after"] = e["load_before"] + e["boarding"]
    e["full"] = e["load_after"] >= seat_capacity - 1
    g = e.groupby(["seq", "hour"])
    prof = pd.DataFrame({
        "boarding": g["boarding"].sum() / n_days,
        "alighting": g["alighting"].sum() / n_days,
        "load_before_board": g["load_before"].sum() / n_days,
        "load_lower": g["load_after"].sum() / n_days,
        "n_buses": g.size() / n_days,
        "gap_mean": g["gap_min"].mean(),
        "gap_cv": g["gap_min"].std() / g["gap_min"].mean(),
        "full_bus_share": g["full"].mean(),
    })
    grid = pd.MultiIndex.from_product([sorted(stops["seq"].unique()), range(24)], names=["seq", "hour"])
    prof = prof.reindex(grid)
    for col in ("boarding", "alighting", "load_before_board", "load_lower", "n_buses"):
        prof[col] = prof[col].fillna(0.0)
    prof = prof.reset_index()
    prof["net"] = prof["boarding"] - prof["alighting"]
    prof["route_max_load_hour"] = prof.groupby("hour")["load_lower"].transform("max")
    prof["headway_obs"] = 60.0 / prof["n_buses"].where(prof["n_buses"] > 0)
    prof["per_bus_load"] = prof["load_lower"] / prof["n_buses"].where(prof["n_buses"] > 0)
    info = stops[["seq", "sttn_id", "stop_name", "stop_label", "segment"]].drop_duplicates("seq")
    prof = prof.merge(info, on="seq", how="left")
    cols = [
        "sttn_id", "seq", "stop_name", "stop_label", "segment", "hour",
        "boarding", "alighting", "net", "load_before_board", "load_lower", "route_max_load_hour",
        "per_bus_load", "n_buses", "headway_obs", "gap_mean", "gap_cv", "full_bus_share",
    ]
    return prof[cols].sort_values(["hour", "seq"]).reset_index(drop=True)


# %%
def route_share(stop_board: pd.DataFrame, stops: pd.DataFrame, stop_routes: pd.Series | None = None) -> pd.DataFrame | None:
    """정류장별 9401 분담률. stop_board = 02번 (stop, hour, is_route, users), stop_routes = 02번 정류장별 노선 수."""
    if stop_board is None or stop_board.empty:
        print("⚠ 정류장별 전체 노선 승차 집계 없음 -> 분담률 빈 값 (07번은 유동인구만 수요신호로)")
        return None
    sb = stop_board.copy()
    tot = sb.groupby(["stop", "hour"])["users"].sum()
    mine = sb[sb["is_route"]].groupby(["stop", "hour"])["users"].sum().reindex(tot.index, fill_value=0)
    hourly = (mine / tot.where(tot > 0)).rename("share_all").reset_index()
    if "weekday" in sb:
        w = sb[sb["weekday"].astype(bool)]
        tot_w = w.groupby(["stop", "hour"])["users"].sum()
        mine_w = w[w["is_route"]].groupby(["stop", "hour"])["users"].sum().reindex(tot_w.index, fill_value=0)
        hourly = hourly.merge((mine_w / tot_w.where(tot_w > 0)).rename("share_weekday").reset_index(), on=["stop", "hour"], how="left")
    else:
        hourly["share_weekday"] = np.nan
    stop_tot = sb.groupby("stop")["users"].sum()
    stop_mine = sb[sb["is_route"]].groupby("stop")["users"].sum().reindex(stop_tot.index, fill_value=0)
    stop_share = (stop_mine / stop_tot.where(stop_tot > 0)).rename("share_all_stop")
    n_routes = sb[~sb["is_route"]].groupby("stop").size()  # 참고: 다른 노선 기록이 있는 시간대 수
    info = stops[["seq", "sttn_id"]].drop_duplicates("seq")
    grid = info.merge(pd.DataFrame({"hour": range(24)}), how="cross")
    out = grid.merge(hourly.rename(columns={"stop": "sttn_id"}), on=["sttn_id", "hour"], how="left")
    out["share_all_stop"] = out["sttn_id"].map(stop_share)
    out["n_other_route_hours"] = out["sttn_id"].map(n_routes).fillna(0)
    out["n_routes"] = out["sttn_id"].map(stop_routes) if stop_routes is not None and len(stop_routes) else np.nan
    return out[["seq", "hour", "share_all_stop", "share_all", "share_weekday", "n_other_route_hours", "n_routes"]]


def ride_times(ev: pd.DataFrame, key: str = "weekday", min_n: int = 3) -> dict:
    """이웃 정류장 사이 실제 차내 이동 시간(분) 중앙값 {(앞 seq, 뒤 seq, 시간대): 분}. 정차 시간 포함."""
    e = select_days(ev, key).sort_values(["trip_key", "seq"])
    nxt = e.groupby("trip_key")[["seq", "arr_min", "arr_observed"]].shift(-1)
    d = pd.DataFrame({
        "a": e["seq"], "b": nxt["seq"], "hour": e["hour"],
        "mins": nxt["arr_min"] - e["arr_min"],
        "ok": e["arr_observed"] & nxt["arr_observed"].fillna(False).astype(bool),
    }).dropna(subset=["b"])
    d = d[d["ok"] & (d["mins"] >= 0) & (d["mins"] < 60)]
    agg = d.groupby(["a", "b", "hour"])["mins"].agg(["median", "size"])
    agg = agg[agg["size"] >= min_n]
    return {(int(a), int(b), int(h)): float(v) for (a, b, h), v in agg["median"].items()}


# %%
OTHER_BUS_MIN_PER_HOUR = 1.0  # 그 구간 다른 버스가 시간당 1대도 없으면 선택지에서 뺌
OTHER_BUS_MAX_WAIT = 15.0  # 다른 버스 기다리는 시간 상한(분)


def other_bus_times(pair_trips: pd.DataFrame, stops: pd.DataFrame, route_id: str, holidays: list[str]) -> dict:
    """서울 정류장 A -> B를 9401이 아닌 다른 버스로 갈 때 (차내 시간, 기다리는 시간). 09번 추천 서비스용.

    차내 시간   = A에서 타서 B에서 내린 승객들의 실제 이동 시간 평균 (교통카드, 모든 노선)
    기다리는 시간 = 배차/2,  배차 = 60 / (그 구간을 실제로 다닌 버스 대수/시간)
                  (버스 대수 = 그 시간 A에서 타서 B에서 내린 사람이 있는 노선·차량 수, 하루 평균)
    반환 {(평일/주말, A seq, B seq, 시간대): (차내 분, 대기 분)}
    """
    if pair_trips is None or pair_trips.empty:
        return {}
    p = pair_trips[pair_trips["route"] != route_id].copy()
    d = pd.to_datetime(p["date"].str[:8], format="%Y%m%d", errors="coerce")
    weekend = (d.dt.dayofweek >= 5) | p["date"].str[:8].isin(set(holidays))
    p["daytype"] = np.where(weekend, "weekend", "weekday")
    n_days = p.groupby("daytype")["date"].nunique()
    g = p.groupby(["daytype", "a", "b", "hour"])
    agg = pd.DataFrame({
        "ride": g["ride_sum"].sum() / g["n"].sum(),
        "buses_per_hour": g.size() / n_days.reindex(g.size().index.get_level_values(0)).to_numpy(),
    })
    agg = agg[agg["buses_per_hour"] >= OTHER_BUS_MIN_PER_HOUR]
    agg["wait"] = (30.0 / agg["buses_per_hour"]).clip(upper=OTHER_BUS_MAX_WAIT)
    seqs_of = stops.groupby("sttn_id")["seq"].apply(list).to_dict()  # 같은 정류장ID가 두 seq일 수 있음
    out = {}
    for (dt, a, b, h), row in agg.iterrows():
        for sa in seqs_of.get(a, []):
            for sb in seqs_of.get(b, []):
                out[(dt, int(sa), int(sb), int(h))] = (float(row["ride"]), float(row["wait"]))
    return out
