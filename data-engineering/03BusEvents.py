# 교통카드로 버스 한 대씩 정류장 도착 시각·재차·앞차와의 공백
"""
03BusEvents.py

교통카드 이용내역에는 차량ID와 초 단위 승하차 시각이 있어서, 버스 한 대씩 실제 값을 계산할 수 있다.
결과 한 행 = 버스 한 대가 정류장 하나에 도착한 것.

1. 정류장ID -> seq (01번 stop_master)
   같은 정류장ID가 노선에 두 번 나오면(루프 시작과 끝 등) 그 차량의 바로 앞 기록 seq와 가까운 쪽으로
2. 운행 나누기: 같은 차량의 기록을 시간순으로 놓고
     정류장 순서가 10칸 넘게 뒤로 돌아가면 다음 운행 (차고지에서 바로 다시 출발해도 구분됨)
     90분 넘게 끊겨도 다음 운행
3. 하차 기록이 없는 승객: 같은 정류장에서 탄 다른 승객들이 내린 정류장 비율대로 나눔
4. 정류장 도착 시각 = 그 운행에서 그 정류장의 첫 승차·하차 태그 시각
   아무도 안 타고 안 내린 정류장은 앞뒤 정류장 시각 + 보통 걸리는 시간으로 메움
   (만차라서 서지 않고 지나간 정류장도 여기에 해당 -> 빠뜨리면 안 됨)
5. 재차 = max(0, 이전 재차 - 하차) + 승차   (내리는 사람이 먼저, 0 밑으로 안 내려감)
6. 공백 = 같은 정류장에 앞 버스가 온 시각과의 차이(분). 90분 넘으면 그날 첫차로 보고 비움
7. 평일/주말/요일: 운행일자 기준. 평일 공휴일은 주말·일요일로

결과(버스별 행)는 안심구역 안에서만 쓰고 반출하지 않는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRIP_GAP_MIN = 90  # 같은 차량 기록이 90분 넘게 끊기면 다음 운행
TRIP_BACK_SEQ = 10  # 정류장 순서가 10칸 넘게 뒤로 가면 다음 운행
GAP_MAX_MIN = 90  # 앞차와 90분 넘게 벌어지면 그날 첫차로 봄
DOW_KEYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]  # sun = 일요일·공휴일
DOW_KR = {"mon": "월", "tue": "화", "wed": "수", "thu": "목", "fri": "금", "sat": "토", "sun": "일·공휴일"}


def _norm_id(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lstrip("0")


def day_keys(dates: pd.Series, holidays: list[str]) -> tuple[pd.Series, pd.Series]:
    """날짜(YYYYMMDD) -> (weekday/weekend, mon~sun). 공휴일은 주말·일요일로."""
    d = pd.to_datetime(dates.str[:8], format="%Y%m%d", errors="coerce")
    dow = d.dt.dayofweek.map(dict(enumerate(DOW_KEYS)))
    dow = dow.where(~dates.str[:8].isin(set(holidays)), "sun")
    daytype = pd.Series(np.where(dow.isin(["sat", "sun"]), "weekend", "weekday"), index=dates.index)
    return daytype, dow


# %%
def assign_seq(card: pd.DataFrame, stops: pd.DataFrame) -> pd.DataFrame:
    """승차·하차 정류장ID -> seq. 같은 ID가 두 번 나오는 정류장은 차량의 앞 기록 seq와 가까운 쪽."""
    cand = stops.groupby("sttn_id")["seq"].apply(lambda s: sorted(s)).to_dict()
    c = card.copy()
    g, o = _norm_id(c["gin_stop"]), _norm_id(c["gff_stop"])
    c["seq"] = g.map(lambda k: cand[k][0] if k in cand else np.nan)
    amb = g.map(lambda k: k in cand and len(cand[k]) > 1)
    if amb.any():
        print(f"같은 정류장ID가 노선에 두 번 나오는 승차 기록 {amb.mean():.1%} -> 앞 기록 순서로 구분")
        c = c.sort_values(["car", "gin_time"])
        seqs, gs, cars = c["seq"].to_numpy(float), g.loc[c.index].to_numpy(), c["car"].to_numpy()
        prev, prev_car = np.nan, None
        for i in range(len(c)):
            if cars[i] != prev_car:
                prev, prev_car = np.nan, cars[i]
            opts = cand.get(gs[i])
            if opts and len(opts) > 1:
                if np.isnan(prev):
                    seqs[i] = opts[0]
                else:  # 앞 기록보다 같거나 뒤인 것 중 가장 가까운 것, 없으면 처음 것(다음 운행)
                    ahead = [s for s in opts if s >= prev]
                    seqs[i] = ahead[0] if ahead else opts[0]
            if not np.isnan(seqs[i]):
                prev = seqs[i]
        c["seq"] = seqs
    # 하차: 탄 정류장보다 뒤인 후보 중 가장 가까운 것
    o = o.loc[c.index]
    c["off"] = [
        next((s for s in cand[k] if s > b), np.nan) if (k in cand and not np.isnan(b)) else np.nan
        for k, b in zip(o, c["seq"].to_numpy(float))
    ]
    return c


def split_trips(c: pd.DataFrame) -> pd.DataFrame:
    """같은 차량 기록을 운행 한 번씩 나눈다 (seq가 크게 뒤로 가거나 오래 끊기면 다음 운행)."""
    c = c.dropna(subset=["gin_time", "seq"]).sort_values(["car", "gin_time"]).copy()
    gap = c.groupby("car")["gin_time"].diff().dt.total_seconds().div(60)
    back = c.groupby("car")["seq"].diff() < -TRIP_BACK_SEQ
    new = gap.isna() | (gap > TRIP_GAP_MIN) | back
    c["trip"] = new.groupby(c["car"]).cumsum()
    c["trip_key"] = c["car"] + "-" + c["trip"].astype(str)
    return c


# %%
def _typical_offsets(obs: np.ndarray) -> np.ndarray:
    """정류장마다 출발 후 보통 몇 분 뒤에 도착하는지 (관측된 이웃 정류장 사이 시간의 중앙값을 최소제곱으로 이음)."""
    T, S = obs.shape
    rows = []
    for t in range(T):
        idx = np.where(~np.isnan(obs[t]))[0]
        if len(idx) > 1:
            rows.append(np.column_stack([idx[:-1], idx[1:], np.diff(obs[t, idx])]))
    if not rows:
        return np.arange(S, dtype=float)
    d = pd.DataFrame(np.vstack(rows), columns=["i", "j", "dt"])
    d = d[d["dt"] >= 0].groupby(["i", "j"])["dt"].agg(["median", "size"]).reset_index()
    A = np.zeros((len(d) + 1, S))
    w = np.sqrt(d["size"].to_numpy(float))
    A[np.arange(len(d)), d["i"].astype(int)] = -w
    A[np.arange(len(d)), d["j"].astype(int)] = w
    A[-1, 0] = 1.0  # 첫 정류장 = 0분
    y = np.append(d["median"].to_numpy(float) * w, 0.0)
    typ, *_ = np.linalg.lstsq(A, y, rcond=None)
    seen = np.zeros(S, bool)
    seen[d["i"].astype(int)] = True
    seen[d["j"].astype(int)] = True
    seen[0] = True
    if (~seen).any() and seen.sum() > 1:  # 정보가 없는 정류장은 앞뒤 사이 직선으로
        typ[~seen] = np.interp(np.where(~seen)[0], np.where(seen)[0], typ[seen])
    return np.maximum.accumulate(typ)


def _min_at(a: np.ndarray, rows: np.ndarray, cols: np.ndarray, vals: np.ndarray) -> None:
    ok = ~np.isnan(vals)
    np.fmin.at(a, (rows[ok], cols[ok]), vals[ok])


def build_bus_events(card: pd.DataFrame, stops: pd.DataFrame, holidays: list[str]) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """버스 한 대가 정류장 하나에 도착한 것 = 한 행.

    반환: (행 표, 품질 숫자, 승차 정류장 x 하차 정류장 인원표(하차 기록 있는 승객, 10번 E3 그림용))
    """
    seqs = np.array(sorted(stops["seq"].unique()))
    pos_of = {s: i for i, s in enumerate(seqs)}
    S = len(seqs)

    c = assign_seq(card, stops)
    q = {"승차정류장_연결비율": float(c["seq"].notna().mean())}
    c = c.dropna(subset=["seq", "gin_time"])
    c["seq"] = c["seq"].astype(int)
    bad = c["off"].isna() | c["gff_time"].isna()
    q["하차정류장_추정비율"] = float((c["users"] * bad).sum() / max(c["users"].sum(), 1))

    t = split_trips(c)
    day = t["date"].where(t["date"] != "", t["gin_dt"].str[:8])
    t["svc_date"] = day.groupby(t["trip_key"]).transform("first")  # 자정 넘어 끝나는 운행도 출발한 날로
    trips = t["trip_key"].unique()
    T = len(trips)
    ti = t["trip_key"].map({k: i for i, k in enumerate(trips)}).to_numpy()
    pi = t["seq"].map(pos_of).to_numpy()
    good = ~bad.loc[t.index].to_numpy()
    oi = t["off"].map(pos_of).to_numpy()
    users = t["users"].to_numpy(float)

    board = np.zeros((T, S))
    np.add.at(board, (ti, pi), users)
    alight = np.zeros((T, S))
    np.add.at(alight, (ti[good], oi[good].astype(int)), users[good])
    # 하차 기록 없는 승객: 같은 정류장 승차 승객의 하차 정류장 비율대로 (정보가 없으면 뒤쪽 하차 비중대로)
    od = np.zeros((S, S))
    np.add.at(od, (pi[good], oi[good].astype(int)), users[good])
    od_table = pd.DataFrame(od.copy(), index=seqs, columns=seqs)
    later_w = alight.sum(axis=0) + 1e-9
    for i in range(S):
        if od[i].sum() == 0:
            od[i, i + 1:] = later_w[i + 1:]
    od = od / np.maximum(od.sum(axis=1, keepdims=True), 1e-12)
    miss = np.zeros((T, S))
    np.add.at(miss, (ti[~good], pi[~good]), users[~good])
    alight += miss @ od

    # 도착 시각(분): 승차 태그와 (하차 기록 있는) 하차 태그 중 가장 이른 시각
    epoch = pd.Timestamp("2000-01-01")
    obs = np.full((T, S), np.nan)
    _min_at(obs, ti, pi, ((t["gin_time"] - epoch).dt.total_seconds() / 60).to_numpy())
    gff_m = ((t["gff_time"] - epoch).dt.total_seconds() / 60).to_numpy()
    _min_at(obs, ti[good], oi[good].astype(int), gff_m[good])
    has = ~np.isnan(obs)
    first = np.where(has.any(axis=1), has.argmax(axis=1), 0)
    last = np.where(has.any(axis=1), S - 1 - has[:, ::-1].argmax(axis=1), -1)
    cover = (np.arange(S)[None, :] >= first[:, None]) & (np.arange(S)[None, :] <= last[:, None])
    typ = _typical_offsets(obs)
    x = pd.DataFrame(obs - typ[None, :]).ffill(axis=1).bfill(axis=1).to_numpy()
    arr = np.where(cover, typ[None, :] + x, np.nan)

    before = np.zeros((T, S))
    load = np.zeros(T)
    for j in range(S):
        load = np.maximum(0.0, load - np.where(cover[:, j], alight[:, j], 0.0))
        before[:, j] = load
        load = load + np.where(cover[:, j], board[:, j], 0.0)

    info = t.drop_duplicates("trip_key").set_index("trip_key")[["car", "svc_date"]]
    tt, ss = np.where(cover)
    ev = pd.DataFrame({
        "trip_key": trips[tt], "seq": seqs[ss], "arr_min": arr[tt, ss], "arr_observed": has[tt, ss],
        "load_before": before[tt, ss], "boarding": board[tt, ss], "alighting": alight[tt, ss],
    })
    ev["car"] = ev["trip_key"].map(info["car"])
    ev["date"] = ev["trip_key"].map(info["svc_date"])
    ev["arr_time"] = epoch + pd.to_timedelta(ev["arr_min"], unit="m")
    ev["hour"] = ev["arr_time"].dt.hour
    ev = ev.sort_values(["seq", "arr_min"]).reset_index(drop=True)
    gap = ev.groupby("seq")["arr_min"].diff()
    ev["gap_min"] = gap.where((gap > 0) & (gap <= GAP_MAX_MIN))
    ev["daytype"], ev["dow"] = day_keys(ev["date"], holidays)

    n_days = max(ev["date"].nunique(), 1)
    trip_max = (ev["load_before"] + ev["boarding"]).groupby(ev["trip_key"]).max()
    gaps = ev["gap_min"].dropna()
    q.update({
        "하루_운행횟수": round(T / n_days, 1),
        "도착시각_실측비율": float(ev["arr_observed"].mean()),
        "정류장순서_일치도": _order_check(t),
        # 점검용: 같은 버스가 두 번 잡히면(차량ID 중복) 30초 이내 연속 도착이 비정상적으로 많아짐
        "30초이내_연속도착_비율": float((gaps <= 0.5).mean()) if len(gaps) else float("nan"),
        # 점검용: 41명을 자주 넘으면 입석이 있거나 좌석이 더 많은 차량(2층 버스 등)이 섞인 것
        "운행별_최대재차_95백분위": round(float(trip_max.quantile(0.95)), 1),
        "재차_41명초과_운행비율": float((trip_max > 41).mean()),
    })
    return ev, q, od_table


def _order_check(t: pd.DataFrame) -> float:
    """운행 시작 후 경과 시간 순서가 seq 순서와 맞는가 (스피어만, 1에 가까울수록 정류장 연결·운행 구분이 맞음)."""
    start = t.groupby("trip_key")["gin_time"].transform("min")
    m = ((t["gin_time"] - start).dt.total_seconds() / 60).groupby(t["seq"]).median()
    return round(float(pd.Series(m.index, index=m.index).rank().corr(m.rank())), 3)
