"""
09ServiceRecommend.py

07번 결과를 경로 안내 서비스로 바꾼다.
"가장 가까운 정류장"이 아니라 "9401에 타서 서울을 빠져나가기까지 가장 빠른 정류장"을 안내한다.
모든 계산은 07번 10분 단위 결과로 한다 ("18:20에 도착하면 어디로").

1. 기대 대기 시간 = 07번 줄 서기 결과 (실제 버스 순서·남은 좌석, 만석 버스를 보낸 시간 포함)

2. 방향 (01번 direction): 9401은 서울역버스환승센터에서 회차한다.
     서울역 방향 : 서울 첫 정류장 ~ 회차 전 정류장
     성남 방향   : 회차 정류장 ~ 서울 마지막 정류장
   추천은 같은 방향 정류장끼리만 비교한다.

3. 정류장 추천 (recommend): 같은 방향 9401 정류장 u마다
     총 시간(u) = 지금 위치 -> u 이동 + u에서 기대 대기 + u -> 그 방향 끝까지 9401 차내 시간
   끝 지점을 맞춰야 앞 정류장과 뒤 정류장을 공정하게 비교할 수 있다.
   이동: 정류장에 있을 때는 다른 버스(교통카드로 잰 차내 시간 + 대기, 04번 other_bus_times)와
         도보 중 빠른 쪽. 다른 버스는 길 건너 짝 정류장(반대 방향, cross_street_m 안)에서 타거나 내려도 된다.
         예: 순천향대(38) -> 길 건너 순천향대(28)에서 시내버스 -> 국가인권위(29) 하차 -> 길 건너 명동성당(37)
   9401 차내 시간 = 교통카드로 잰 이웃 정류장 사이 실제 이동 시간 (04번 ride_times)
   추천 = 총 시간이 가장 짧은 u, 지금 정류장보다 min_saving_min 이상 빠를 때만.
   제외: 이름이 같은 반대 방향 정류장, 거꾸로 돌아가 타는 차내 시간이 max_ride_back_min을 넘는 경우

4. 위치 기반 추천 (recommend_from_points): 정류장이 아닌 곳(지하철역, 도심 격자)에서 성남으로 가는 사람.
   걸어갈 수 있는 성남 방향 정류장 중 총 시간이 가장 짧은 곳.

5. 회피 가능한 탑승 실패 (avoidable_failures)
   대상: 평일 출퇴근에 처음 온 9401을 만석으로 놓친 사람 (10분 칸 도착 인원 x (1 - 첫 버스 탑승확률))
   회피 가능: S분 이상 빨라지는 선택지가 하나라도 있었던 사람
     정류장 변경    : 3번 정류장 추천으로 서울을 빠져나가기까지 S분 이상 단축
     출발 시각 조정 : 같은 정류장 ±W분 안 다른 10분 칸의 기대 대기가 S분 이상 짧음
   S(3·5·10분) x W(±10·20·30분) 9가지 가정의 범위로 제시한다.
   한 사람이 선택을 바꿀 때 기준이며, 여러 사람이 동시에 옮겨 생기는 혼잡은 반영하지 않는다.

6. 대기 시간 부담 (wait_burden)
   하루 초과 대기(인·시간) = 시간대별 sum(대기 수요 x 추가 대기) / 60, 연간 = 하루 x 연간 평일 수
   추천 이용 시 절감 상한 = sum(추천이 있는 칸의 대기 수요 x 단축 시간) / 60 (전원이 추천을 따를 때)

서비스 기준값(보행 속도 등)은 SERVICE_DEFAULTS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Transformer

SERVICE_DEFAULTS = {
    "walk_speed_kmh": 4.5,  # 보행 속도
    "detour": 1.3,  # 직선거리 -> 실제 이동거리 우회계수
    "max_walk_m": 500,  # 정류장끼리 걸어서 옮기는 최대 직선거리
    "max_walk_point_m": 1200,  # 정류장이 아닌 곳에서 정류장까지 걸어갈 최대 직선거리
    "bus_speed_kmh": 15.0,  # 9401 차내 기록이 없는 구간만 쓰는 도심 버스 평균 속도
    "min_saving_min": 3.0,  # 이만큼은 빨라야 추천
    "max_ride_back_min": 15.0,  # 거꾸로 돌아가서 탈 때 9401 차내 시간 상한
    "workdays_per_year": 250,  # 연간 평일 수
    "cross_street_m": 100.0,  # 길 건너 짝 정류장(반대 방향) 인정 거리
    "cross_street_min": 2.0,  # 길 건너는 시간 (신호 대기 포함)
}
_TF = Transformer.from_crs(4326, 5179, always_xy=True)


# %%
def add_expected_wait(scored: pd.DataFrame) -> pd.DataFrame:
    """정류장·시간대별 9401 기대 대기 시간.

    07번 줄 서기 결과(wait_total_min, 도착부터 실제로 탈 때까지)가 있으면 그대로 쓴다.
    없으면 근사식: 배차/2 x (1 + 몰림^2) + 추가 대기
    """
    df = scored.copy()
    if "wait_total_min" in df:
        df["wait_base_min"] = df["wait_next_min"]
        return df
    cv = df["gap_cv"].fillna(0.0)
    df["wait_base_min"] = df["headway_obs"] / 2 * (1 + cv ** 2)
    df["wait_total_min"] = df["wait_base_min"] + df["extra_wait_min_mid"]
    return df


def stop_geometry(stops: pd.DataFrame) -> pd.DataFrame:
    """서울 도심 구간 정류장 좌표(UTM-K, m), 노선 순서대로."""
    s = stops[(stops["segment"] == "서울_도심루프") & stops["utmk_x"].notna()]
    return s.drop_duplicates("seq").set_index("seq")[
        ["stop_name", "stop_label", "direction", "lat", "lon", "utmk_x", "utmk_y"]].sort_index()


def direction_stops(g: pd.DataFrame, direction: str) -> tuple[list, int]:
    """그 방향에서 탈 수 있는 정류장들과 그 방향의 끝 정류장.

    서울역 방향: 회차 전 정류장들에서 타고, 끝 = 회차 정류장(서울역)
    성남 방향  : 회차 정류장부터 타고, 끝 = 서울 마지막 정류장
    """
    seqs = [q for q in g.index if g.at[q, "direction"] == direction]
    if direction == "in":
        out = [q for q in g.index if g.at[q, "direction"] == "out"]
        end = out[0] if out else seqs[-1]
    else:
        end = seqs[-1]
    return seqs, end


def _walk_min(d_m: float, sv: dict) -> float:
    return d_m * sv["detour"] / 1000 / sv["walk_speed_kmh"] * 60


def ride_to_end(g: pd.DataFrame, hour: int, sv: dict, ride_tbl: dict | None, end: int | None = None) -> dict:
    """각 정류장에서 9401을 타고 end(기본 = 서울 마지막 정류장)까지 차내 시간 {seq: 분}."""
    seqs = [q for q in g.index if end is None or q <= end]
    out = {seqs[-1]: 0.0}
    for a, b in zip(seqs[-2::-1], seqs[:0:-1]):  # 뒤에서부터 누적
        v = (ride_tbl or {}).get((a, b, hour))
        if v is None:
            d = float(np.hypot(g.at[a, "utmk_x"] - g.at[b, "utmk_x"], g.at[a, "utmk_y"] - g.at[b, "utmk_y"]))
            v = d * sv["detour"] / 1000 / sv["bus_speed_kmh"] * 60
        out[a] = out[b] + v
    return out


def twin_stops(g: pd.DataFrame, max_m: float) -> dict:
    """정류장마다 길 건너 짝 정류장(반대 방향, max_m 안, 가까운 순). 다른 버스를 타고 내리는 곳 후보."""
    out = {}
    for s in g.index:
        opp = g[g["direction"] != g.at[s, "direction"]]
        d = np.hypot(opp["utmk_x"] - g.at[s, "utmk_x"], opp["utmk_y"] - g.at[s, "utmk_y"])
        out[s] = [int(q) for q in d[d <= max_m].sort_values().index]
    return out


def _bus_access(other_bus: dict, bday: str, s: int, u: int, hour: int, twins: dict, cross_min: float):
    """s -> u 를 다른 버스로: 그대로 / 길 건너 타기 / 내려서 길 건너기 / 둘 다 중 가장 빠른 것. (분, 수단) 또는 None."""
    best = None
    for a, ca in [(s, 0.0)] + [(t, cross_min) for t in twins.get(s, [])]:
        for b, cb in [(u, 0.0)] + [(t, cross_min) for t in twins.get(u, [])]:
            ob = other_bus.get((bday, a, b, hour))
            if ob is None or a == b:
                continue
            t = ob[0] + ob[1] + ca + cb
            if best is None or t < best[0]:
                best = (t, "버스" if ca + cb == 0 else "버스(길 건너)")
    return best


def _bus_day(day: str) -> str:
    return "weekend" if day in ("weekend", "sat", "sun") else "weekday"


# %%
def _waits(src: pd.DataFrame, hours: list[int], g: pd.DataFrame) -> pd.DataFrame:
    """대기 시간 출처. 07번 10분 단위 표(slot 컬럼)가 오면 10분 단위로, 시간대 표가 오면 시간대로.

    반환 컬럼: seq, key(10분 칸 또는 시간), hour, time("18:20"), wait_total_min
    """
    if "slot" in src.columns:
        w = src[src["hour"].isin(hours) & src["seq"].isin(g.index)].copy()
        w["key"] = w["slot"]
    else:
        w = add_expected_wait(src)
        w = w[w["has_skt"] & w["hour"].isin(hours) & w["seq"].isin(g.index)].copy()
        w["key"] = w["hour"]
        w["time"] = w["hour"].map(lambda h: f"{int(h):02d}:00")
    return w


def _options(access: dict, wait: dict, to_end: dict) -> list:
    """선택지별 총 시간. access = {u: (이동 분, 이동 수단)}"""
    rows = []
    for u, (acc, mode) in access.items():
        if u in wait and np.isfinite(wait[u]):
            rows.append((u, acc, mode, wait[u], to_end[u], acc + wait[u] + to_end[u]))
    return rows


def recommend(scored: pd.DataFrame, g: pd.DataFrame, hours: list[int], sv: dict | None = None,
              ride_tbl: dict | None = None, other_bus: dict | None = None, day: str = "weekday") -> pd.DataFrame:
    """정류장에 있는 사람: 지금 정류장에서 기다리기 vs 다른 9401 정류장으로 (버스·도보) 옮겨서 타기.

    scored 자리에 07번 10분 단위 표(diag["slots"])를 넘기면 10분 단위로 계산 ("18:20에 도착하면")
    """
    sv = {**SERVICE_DEFAULTS, **(sv or {})}
    w = _waits(scored, hours, g)
    bday = _bus_day(day)
    rows = []
    dirs = {d: direction_stops(g, d) for d in ("in", "out") if (g["direction"] == d).any()}
    twins = twin_stops(g, sv["cross_street_m"])
    for key, wh in w.groupby("key"):
        hour, time = int(wh["hour"].iloc[0]), wh["time"].iloc[0]
        wait = wh.set_index("seq")["wait_total_min"].to_dict()
        ends = {d: ride_to_end(g, int(hour), sv, ride_tbl, end) for d, (_, end) in dirs.items()}
        for s in g.index:
            if s not in wait or not np.isfinite(wait[s]):
                continue
            d_s = g.at[s, "direction"]
            same_dir, _ = dirs[d_s]
            to_end = ends[d_s]
            access = {s: (0.0, "그대로")}
            for u in same_dir:  # 같은 방향 정류장만
                if u == s or g.at[u, "stop_name"] == g.at[s, "stop_name"]:
                    continue
                if u < s and to_end[u] - to_end[s] > sv["max_ride_back_min"]:
                    continue  # 도심을 거의 한 바퀴 돌아오는 안내는 제외
                cands = []
                d = float(np.hypot(g.at[s, "utmk_x"] - g.at[u, "utmk_x"], g.at[s, "utmk_y"] - g.at[u, "utmk_y"]))
                if d <= sv["max_walk_m"]:
                    cands.append((_walk_min(d, sv), "도보"))
                ob = _bus_access(other_bus or {}, bday, s, u, int(hour), twins, sv["cross_street_min"])
                if ob is not None:
                    cands.append(ob)
                if cands:
                    access[u] = min(cands)
            opts = _options(access, wait, to_end)
            here = next(o for o in opts if o[0] == s)
            best = min(opts, key=lambda o: o[5])
            ok = best[0] != s and here[5] - best[5] >= sv["min_saving_min"]
            rows.append({
                "seq": s, "direction": d_s, "hour": hour, "slot": key, "time": time,
                "wait_here_min": here[3], "total_here_min": here[5],
                "alt_seq": best[0] if ok else np.nan, "alt_mode": best[2] if ok else "",
                "alt_access_min": best[1] if ok else np.nan, "alt_wait_min": best[3] if ok else np.nan,
                "alt_extra_ride_min": (best[4] - here[4]) if ok else np.nan,
                "alt_total_min": best[5] if ok else np.nan,
                "saving_min": (here[5] - best[5]) if ok else 0.0,
            })
    return pd.DataFrame(rows)


def points_utmk(points: list) -> pd.DataFrame:
    """[(이름, 위도, 경도)] -> UTM-K"""
    df = pd.DataFrame(points, columns=["name", "lat", "lon"])
    x, y = _TF.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
    return df.assign(utmk_x=x, utmk_y=y)


def recommend_from_points(scored: pd.DataFrame, g: pd.DataFrame, pts: pd.DataFrame, hours: list[int],
                          sv: dict | None = None, ride_tbl: dict | None = None) -> pd.DataFrame:
    """정류장이 아닌 곳(지하철역, 격자점 등)에서 성남으로 가는 사람: 걸어서 갈 수 있는 성남 방향 9401 정류장 중 가장 빠른 곳.

    기준 = 가장 가까운 성남 방향 정류장으로 가서 타는 경우
    """
    sv = {**SERVICE_DEFAULTS, **(sv or {})}
    if not (g["direction"] == "out").any():
        print("⚠ 성남 방향 정류장이 없어 위치 기반 추천을 건너뜀 (회차 정류장 확인)")
        return pd.DataFrame(columns=["name", "x", "y", "hour", "slot", "time", "near_seq", "near_walk_min", "near_wait_min", "near_total_min",
                                     "best_seq", "best_walk_min", "best_wait_min", "best_total_min", "saving_min"])
    seqs, end = direction_stops(g, "out")  # 성남으로 가는 사람 -> 성남 방향 정류장만
    g = g.loc[seqs]
    w = _waits(scored, hours, g)
    gx, gy = g["utmk_x"].to_numpy(), g["utmk_y"].to_numpy()
    rows = []
    for key, wh in w.groupby("key"):
        hour, time = int(wh["hour"].iloc[0]), wh["time"].iloc[0]
        wait = wh.set_index("seq")["wait_total_min"].to_dict()
        to_end = ride_to_end(g, int(hour), sv, ride_tbl, end)
        for p in pts.itertuples():
            d = np.hypot(gx - p.utmk_x, gy - p.utmk_y)
            near = g.index[int(np.argmin(d))]
            access = {u: (_walk_min(dd, sv), "도보") for u, dd in zip(g.index, d) if dd <= sv["max_walk_point_m"]}
            access.setdefault(near, (_walk_min(float(d.min()), sv), "도보"))
            opts = _options(access, wait, to_end)
            if not opts:
                continue
            base = next((o for o in opts if o[0] == near), None)
            best = min(opts, key=lambda o: o[5])
            if base is None:
                continue
            ok = best[0] != near and base[5] - best[5] >= sv["min_saving_min"]
            rows.append({
                "name": p.name, "x": p.utmk_x, "y": p.utmk_y, "hour": hour, "slot": key, "time": time,
                "near_seq": near, "near_walk_min": base[1], "near_wait_min": base[3], "near_total_min": base[5],
                "best_seq": best[0] if ok else near, "best_walk_min": best[1] if ok else base[1],
                "best_wait_min": best[3] if ok else base[3], "best_total_min": best[5] if ok else base[5],
                "saving_min": (base[5] - best[5]) if ok else 0.0,
            })
    return pd.DataFrame(rows)


def grid_points(g: pd.DataFrame, spacing: float = 100.0, pad: float = 500.0) -> pd.DataFrame:
    """서울 도심 정류장 주변 격자 (지도 그림용)."""
    xs = np.arange(g["utmk_x"].min() - pad, g["utmk_x"].max() + pad + 1, spacing)
    ys = np.arange(g["utmk_y"].min() - pad, g["utmk_y"].max() + pad + 1, spacing)
    xx, yy = np.meshgrid(xs, ys)
    return pd.DataFrame({"name": [f"g{i}" for i in range(xx.size)], "utmk_x": xx.ravel(), "utmk_y": yy.ravel()})


# %%
def wait_burden(scored: pd.DataFrame, rec: pd.DataFrame | None, sv: dict | None = None) -> pd.DataFrame:
    """시간대별 하루 초과 대기(인·시간)와 연간 환산, 추천 이용 시 절감 상한."""
    sv = {**SERVICE_DEFAULTS, **(sv or {})}
    d = scored[scored["has_skt"]]
    excess = (d["latent_daily"] * d["extra_wait_min_mid"]).groupby(d["hour"]).sum() / 60
    out = pd.DataFrame({"excess_wait_person_hours_day": excess})
    out["saving_upper_person_hours_day"] = 0.0
    if rec is not None and not rec.empty:
        sv_h = rec.groupby(["seq", "hour"])["saving_min"].mean().reset_index()  # 10분 칸이면 시간대 안 평균
        r = sv_h[sv_h["saving_min"] > 0].merge(d[["seq", "hour", "latent_daily"]], on=["seq", "hour"])
        saved = (r["latent_daily"] * r["saving_min"]).groupby(r["hour"]).sum() / 60
        out["saving_upper_person_hours_day"] = saved.reindex(out.index).fillna(0)
    out["excess_wait_person_hours_year"] = out["excess_wait_person_hours_day"] * sv["workdays_per_year"]
    out.index.name = "hour"
    return out.reset_index()


# %%
SLOT_MIN = 10  # 07번 10분 칸 (slot 번호 1 = 10분)
AVOID_SAVINGS = (3.0, 5.0, 10.0)  # 회피 가능 기준: 이만큼(분) 이상 빨라져야
AVOID_WINDOWS = (10, 20, 30)  # 출발 시각을 앞뒤로 이만큼(분)까지 바꿀 수 있다고 봄
AVOID_BASE = (5.0, 30)  # 기본 가정 (S, W)


def _wmedian(x: pd.Series, w: pd.Series) -> float:
    m = x.notna() & (w > 0)
    if not m.any():
        return np.nan
    o = np.argsort(x[m].to_numpy())
    xs, cw = x[m].to_numpy()[o], np.cumsum(w[m].to_numpy()[o])
    return float(xs[min(np.searchsorted(cw, cw[-1] / 2), len(xs) - 1)])


def avoidable_failures(slots: pd.DataFrame, rec: pd.DataFrame, g: pd.DataFrame, hours: list[int],
                       savings=AVOID_SAVINGS, windows=AVOID_WINDOWS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """처음 온 9401을 만석으로 놓친 사람 중 더 빨리 타는 선택지가 있었던 사람 비율 (모듈 설명 5번).

    slots = 07번 10분 단위 표 (arrivals_daily, p_board, wait_total_min), rec = recommend() 10분 단위 결과
    반환: (가정 S x W별 비율·단축 시간 표, 10분 칸별 상세(그림용))
    """
    s = slots[slots["seq"].isin(g.index) & slots["hour"].isin(hours)].copy()
    s["failed"] = s["arrivals_daily"] * (1 - s["p_board"].clip(0, 1))
    s = s[s["failed"] > 1e-9].reset_index(drop=True)
    idx = pd.MultiIndex.from_frame(s[["seq", "slot"]])
    sv = rec.set_index(["seq", "slot"])["saving_min"] if len(rec) else pd.Series(dtype=float)
    s["stop_saving"] = sv.reindex(idx).fillna(0.0).to_numpy()
    s["stop_mode"] = (rec.set_index(["seq", "slot"])["alt_mode"].reindex(idx).fillna("").to_numpy() if len(rec) else "")
    # 같은 정류장 앞뒤 W분 안 다른 10분 칸 중 기대 대기가 가장 짧은 칸 (출퇴근 밖 시간대로 옮기는 것도 허용)
    wait = slots.set_index(["seq", "slot"])["wait_total_min"]
    for W in windows:
        k = int(W // SLOT_MIN)
        alt = np.full(len(s), np.inf)
        for d in [d for d in range(-k, k + 1) if d != 0]:
            v = wait.reindex(pd.MultiIndex.from_arrays([s["seq"], (s["slot"] + d) % (1440 // SLOT_MIN)])).to_numpy(float)
            alt = np.fmin(alt, np.where(np.isfinite(v), v, np.inf))
        s[f"time_saving_{W}"] = np.where(np.isfinite(alt), (s["wait_total_min"] - alt).clip(lower=0), 0.0)
    tot = s["failed"].sum()
    rows = []
    for S in savings:
        for W in windows:
            stop_ok = s["stop_saving"] >= S
            time_ok = s[f"time_saving_{W}"] >= S
            share = lambda m: float(s.loc[m, "failed"].sum() / tot) if tot > 0 else np.nan
            rows.append({
                "단축기준_분(S)": S, "시각조정범위_분(W)": W,
                "회피가능_비율": share(stop_ok | time_ok),
                "정류장변경_가능_비율": share(stop_ok), "시각조정_가능_비율": share(time_ok),
                "둘다_비율": share(stop_ok & time_ok),
                "정류장변경_단축_중앙값_분": _wmedian(s.loc[stop_ok, "stop_saving"], s.loc[stop_ok, "failed"]),
                "시각조정_대기감소_중앙값_분": _wmedian(s.loc[time_ok, f"time_saving_{W}"], s.loc[time_ok, "failed"]),
                "기본가정": (S, W) == AVOID_BASE,
            })
    return pd.DataFrame(rows), s
