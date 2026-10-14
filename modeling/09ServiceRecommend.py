"""
09ServiceRecommend.py

07번 탑승가능확률을 경로 안내 서비스로 바꾼다. (PPT 22~24쪽, 31쪽)
"가장 가까운 정류장"이 아니라 "9401에 올라타서 서울을 빠져나가기까지 가장 빠른 정류장"을 안내한다.

1. 정류장·시간대별 9401 기대 대기 시간 (분)
     기본 대기 = 실제 평균 배차간격 / 2 x (1 + 몰림 정도^2)   (교통카드 실측. 버스가 몰려 오면 평균 대기도 길어짐)
     추가 대기 = 실제 평균 배차간격 x (1/탑승가능확률 - 1)    (만차로 보내는 버스 수만큼, 07번 extra_wait)
     기대 대기 = 기본 대기 + 추가 대기

2. 방향 (01번 direction)
   9401은 서울역버스환승센터에서 회차한다.
     서울역 방향(진입) : 서울 첫 정류장 ~ 회차 전 정류장. 여기서 타는 사람은 서울역 쪽으로 간다
     성남 방향(진출)   : 회차 정류장 ~ 서울 마지막 정류장. 여기서 타는 사람은 성남으로 간다
   추천은 같은 방향 정류장끼리만 비교한다 (서울역 방향 승객에게 성남 방향 정류장을 추천하면 안 됨)

3. 선택지 u (같은 방향 9401 정류장)마다 총 시간
     총 시간(u) = 지금 위치 -> u 이동 + u에서 9401 기대 대기 + u -> 그 방향 끝까지 9401 차내 시간
       그 방향 끝 = 서울역 방향이면 회차 정류장(서울역), 성남 방향이면 서울 마지막 정류장
   마지막 항이 있어야 공정하다: 서울역에서 타면 그 버스가 명동·순천향대를 지나 성남으로 가니까
   (도착 지점을 맞춰야 앞 정류장과 뒤 정류장을 비교할 수 있음)

   지금 위치 -> u 이동
     정류장에 있을 때 : 다른 버스 타고 (교통카드로 잰 그 구간 차내 시간 + 그 구간 버스 대기, 04번 other_bus_times)
                        또는 걸어서 (가까우면) 중 빠른 쪽
     정류장이 아닌 곳 : 걸어서 (직선거리 x 우회계수 / 도보속도)
   9401 차내 시간 = 교통카드로 잰 이웃 정류장 사이 실제 이동 시간 (04번 ride_times)

   추천 = 총 시간이 가장 짧은 u. 지금 정류장(정류장이 아닌 곳이면 가장 가까운 정류장)보다
          min_saving_min(3분) 이상 빨라야 추천 (조금 빠른 건 안내할 가치가 없음)
   제외: 이름이 같은 정류장(루프 처음·끝 순천향대학병원처럼 길 건너 반대 방향)
         거꾸로 돌아가서 타는 경우 9401 차내 시간이 max_ride_back_min(15분) 넘으면 (도심을 한 바퀴 도는 안내 방지)

4. 위치 기반 (정류장에 없어도)
   성남으로 가는 사람을 위한 안내 -> 성남 방향(회차 정류장 ~ 서울 마지막 정류장)만 후보
   예시 위치(지하철역 등, CONFIG service_points)와 도심 격자(100m 간격)마다 같은 식으로 계산
   -> "을지로입구역에서는 가까운 명동성당보다 서울역환승센터로 가는 게 빠르다" 같은 안내

5. 대기 시간 부담 (기대효과용)
     하루 초과 대기 (인·시간) = 시간대별 sum(원래 타려던 인원 x 추가 대기) / 60
     연간 = 하루 x 연간 평일 수
     추천 이용 시 절감 상한 = sum(추천이 있는 칸의 타려던 인원 x 단축 시간) / 60   (전원이 추천을 따른다고 할 때)

서비스 기준값(도보속도 등)은 SERVICE_DEFAULTS. 모델 가정이 아니라 안내 계산용 일반 기준값.
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
}
_TF = Transformer.from_crs(4326, 5179, always_xy=True)


# %%
def add_expected_wait(scored: pd.DataFrame) -> pd.DataFrame:
    """정류장·시간대별 9401 기대 대기 시간."""
    df = scored.copy()
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


def _bus_day(day: str) -> str:
    return "weekend" if day in ("weekend", "sat", "sun") else "weekday"


# %%
def _options(access: dict, wait: dict, to_end: dict) -> list:
    """선택지별 총 시간. access = {u: (이동 분, 이동 수단)}"""
    rows = []
    for u, (acc, mode) in access.items():
        if u in wait and np.isfinite(wait[u]):
            rows.append((u, acc, mode, wait[u], to_end[u], acc + wait[u] + to_end[u]))
    return rows


def recommend(scored: pd.DataFrame, g: pd.DataFrame, hours: list[int], sv: dict | None = None,
              ride_tbl: dict | None = None, other_bus: dict | None = None, day: str = "weekday") -> pd.DataFrame:
    """정류장에 있는 사람: 지금 정류장에서 기다리기 vs 다른 9401 정류장으로 (버스·도보) 옮겨서 타기."""
    sv = {**SERVICE_DEFAULTS, **(sv or {})}
    w = add_expected_wait(scored)
    w = w[w["has_skt"] & w["hour"].isin(hours) & w["seq"].isin(g.index)]
    bday = _bus_day(day)
    rows = []
    dirs = {d: direction_stops(g, d) for d in ("in", "out") if (g["direction"] == d).any()}
    for hour, wh in w.groupby("hour"):
        wait = wh.set_index("seq")["wait_total_min"].to_dict()
        ends = {d: ride_to_end(g, int(hour), sv, ride_tbl, end) for d, (_, end) in dirs.items()}
        for s in g.index:
            if s not in wait or not np.isfinite(wait[s]):
                continue
            d_s = g.at[s, "direction"]
            same_dir, _ = dirs[d_s]
            to_end = ends[d_s]
            access = {s: (0.0, "그대로")}
            for u in same_dir:  # 같은 방향 정류장만 (서울역 방향 승객에게 성남 방향 정류장 추천 X)
                if u == s or g.at[u, "stop_name"] == g.at[s, "stop_name"]:
                    continue
                if u < s and to_end[u] - to_end[s] > sv["max_ride_back_min"]:
                    continue  # 도심을 거의 한 바퀴 돌아오는 안내는 빼기
                cands = []
                d = float(np.hypot(g.at[s, "utmk_x"] - g.at[u, "utmk_x"], g.at[s, "utmk_y"] - g.at[u, "utmk_y"]))
                if d <= sv["max_walk_m"]:
                    cands.append((_walk_min(d, sv), "도보"))
                ob = (other_bus or {}).get((bday, s, u, int(hour)))
                if ob is not None:
                    cands.append((ob[0] + ob[1], "버스"))
                if cands:
                    access[u] = min(cands)
            opts = _options(access, wait, to_end)
            here = next(o for o in opts if o[0] == s)
            best = min(opts, key=lambda o: o[5])
            ok = best[0] != s and here[5] - best[5] >= sv["min_saving_min"]
            rows.append({
                "seq": s, "direction": d_s, "hour": int(hour), "wait_here_min": here[3], "total_here_min": here[5],
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
        return pd.DataFrame(columns=["name", "x", "y", "hour", "near_seq", "near_walk_min", "near_wait_min", "near_total_min",
                                     "best_seq", "best_walk_min", "best_wait_min", "best_total_min", "saving_min"])
    seqs, end = direction_stops(g, "out")  # 성남으로 가는 사람 -> 성남 방향 정류장만
    g = g.loc[seqs]
    w = add_expected_wait(scored)
    w = w[w["has_skt"] & w["hour"].isin(hours) & w["seq"].isin(g.index)]
    gx, gy = g["utmk_x"].to_numpy(), g["utmk_y"].to_numpy()
    rows = []
    for hour, wh in w.groupby("hour"):
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
                "name": p.name, "x": p.utmk_x, "y": p.utmk_y, "hour": int(hour),
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
        r = rec[rec["saving_min"] > 0].merge(d[["seq", "hour", "latent_daily"]], on=["seq", "hour"])
        saved = (r["latent_daily"] * r["saving_min"]).groupby(r["hour"]).sum() / 60
        out["saving_upper_person_hours_day"] = saved.reindex(out.index).fillna(0)
    out["excess_wait_person_hours_year"] = out["excess_wait_person_hours_day"] * sv["workdays_per_year"]
    out.index.name = "hour"
    return out.reset_index()
