"""
11ServiceRecommend.py

09번 탑승가능확률을 경로 안내 서비스로 바꾼다. (PPT 22~24쪽, 31쪽)
"가장 가까운 정류장"이 아니라 "가장 빨리 버스에 올라타는 정류장"을 안내한다.

1. 정류장·시간대별 기대 대기 시간 (분)
     기본 대기 = 배차간격 / 2 x (1 + 몰림 정도^2)        (버스가 몰려 오면 평균 대기도 길어짐)
     추가 대기 = 배차간격 x (1/탑승가능확률 - 1)          (만차로 보내는 버스 수만큼, 09번 extra_wait)
     기대 대기 = 기본 대기 + 추가 대기

2. 상류 정류장 추천
   지금 정류장 s에 있는 승객이 상류 정류장 u로 걸어가서 타는 경우
     총 소요 = 도보 접근(s -> u) + u의 기대 대기 + 차내 이동(u -> s)
   s에서 기다리는 것(= s의 기대 대기)보다 빠르면 u를 추천한다
     도보 접근 = 직선거리 x 우회계수 / 도보속도
     차내 이동 = 정류장 사이 직선거리 x 우회계수 / 도로 평균속도 + 정차 시간
                 도로 평균속도 = 한국도로공사 차량통행지표, 정류장 주변 도로 (출퇴근 첨두는 혼잡시 속도)

3. 대기 시간 부담 (기대효과용)
     하루 초과 대기 (인·시간) = 시간대별 sum(원래 타려던 인원 x 추가 대기) / 60
     연간 = 하루 x 연간 평일 수
     추천 이용 시 절감 상한 = sum(추천이 있는 칸의 타려던 인원 x 단축 시간) / 60   (전원이 추천을 따른다고 할 때)

서비스 기준값(도보속도 등)은 SERVICE_DEFAULTS. 모델 가정이 아니라 안내 계산용 일반 기준값.
RunAll.py가 09번 다음에 호출한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SERVICE_DEFAULTS = {
    "walk_speed_kmh": 4.5,  # 보행 속도
    "detour": 1.3,  # 직선거리 -> 실제 이동거리 우회계수
    "max_walk_m": 800,  # 상류 정류장으로 걸어갈 수 있는 최대 직선거리
    "dwell_min": 0.5,  # 중간 정류장 1곳당 정차 시간
    "workdays_per_year": 250,  # 연간 평일 수
}
PEAK_HOURS = [7, 8, 9, 17, 18, 19]  # 혼잡시 속도를 쓰는 출퇴근 첨두


# %%
def add_expected_wait(scored: pd.DataFrame) -> pd.DataFrame:
    """정류장·시간대별 기대 대기 시간 (배차 중앙 시나리오)."""
    df = scored.copy()
    h = df["headway_eff_mid"]
    cv = df["bunching_cv_stop"] if "bunching_cv_stop" in df else 0.0
    df["wait_base_min"] = h / 2 * (1 + cv ** 2)
    df["wait_total_min"] = df["wait_base_min"] + df["extra_wait_min_mid"]
    return df


def stop_geometry(stop_master: pd.DataFrame, kec_stop: pd.DataFrame | None) -> pd.DataFrame:
    """서울 정류장 좌표(UTM-K, m)와 주변 도로 평균속도."""
    g = stop_master.dropna(subset=["utmk_x"]).drop_duplicates("seq").set_index("seq")[["stop_name", "utmk_x", "utmk_y"]]
    if kec_stop is not None and {"velocity_avg_wavg", "velocity_cg_wavg"} <= set(kec_stop.columns):
        g = g.join(kec_stop.set_index("seq")[["velocity_avg_wavg", "velocity_cg_wavg"]])
    else:
        g["velocity_avg_wavg"] = np.nan
        g["velocity_cg_wavg"] = np.nan
    return g.sort_index()


def _dist(g: pd.DataFrame, a: int, b: int) -> float:
    return float(np.hypot(g.at[a, "utmk_x"] - g.at[b, "utmk_x"], g.at[a, "utmk_y"] - g.at[b, "utmk_y"]))


def ride_minutes(g: pd.DataFrame, u: int, s: int, hour: int, sv: dict) -> float:
    """정류장 u에서 타서 s까지 가는 차내 시간 (u < s, 노선 순서대로)."""
    seqs = [q for q in g.index if u <= q <= s]
    minutes = 0.0
    for a, b in zip(seqs[:-1], seqs[1:]):
        col = "velocity_cg_wavg" if hour in PEAK_HOURS else "velocity_avg_wavg"
        speed = g.at[b, col]
        if not np.isfinite(speed) or speed <= 0:
            speed = 15.0  # 도로 속도가 없으면 도심 버스 평균 수준으로
        minutes += _dist(g, a, b) * sv["detour"] / 1000 / speed * 60
    return minutes + sv["dwell_min"] * max(len(seqs) - 2, 0)


# %%
def recommend(scored: pd.DataFrame, g: pd.DataFrame, hours: list[int], sv: dict | None = None) -> pd.DataFrame:
    """정류장·시간대마다 걸어갈 수 있는 상류 정류장 중 가장 빨리 타는 곳."""
    sv = {**SERVICE_DEFAULTS, **(sv or {})}
    df = add_expected_wait(scored)
    df = df[df["has_skt"] & df["hour"].isin(hours)]
    wait = df.set_index(["seq", "hour"])["wait_total_min"]
    rows = []
    for (s, hour), w_s in wait.items():
        best = None
        for u in g.index:
            if u >= s or (u, hour) not in wait.index:
                continue
            walk_m = _dist(g, s, u) * sv["detour"]
            if _dist(g, s, u) > sv["max_walk_m"]:
                continue
            walk = walk_m / 1000 / sv["walk_speed_kmh"] * 60
            total = walk + wait[(u, hour)] + ride_minutes(g, u, s, hour, sv)
            if best is None or total < best["total"]:
                best = {"u": u, "walk": walk, "wait_u": wait[(u, hour)], "ride": total - walk - wait[(u, hour)], "total": total}
        rows.append({
            "seq": s, "hour": hour, "wait_here_min": w_s,
            "alt_seq": best["u"] if best else np.nan,
            "alt_walk_min": best["walk"] if best else np.nan,
            "alt_wait_min": best["wait_u"] if best else np.nan,
            "alt_ride_min": best["ride"] if best else np.nan,
            "alt_total_min": best["total"] if best else np.nan,
            "saving_min": (w_s - best["total"]) if best else np.nan,
        })
    return pd.DataFrame(rows)


def wait_burden(scored: pd.DataFrame, rec: pd.DataFrame | None, sv: dict | None = None) -> pd.DataFrame:
    """시간대별 하루 초과 대기(인·시간)와 연간 환산, 추천 이용 시 절감 상한."""
    sv = {**SERVICE_DEFAULTS, **(sv or {})}
    d = scored[scored["has_skt"]]
    excess = (d["latent_daily"] * d["extra_wait_min_mid"]).groupby(d["hour"]).sum() / 60
    out = pd.DataFrame({"excess_wait_person_hours_day": excess})
    if rec is not None and not rec.empty:
        r = rec[rec["saving_min"] > 0].merge(d[["seq", "hour", "latent_daily"]], on=["seq", "hour"])
        saved = (r["latent_daily"] * r["saving_min"]).groupby(r["hour"]).sum() / 60
        out["saving_upper_person_hours_day"] = saved.reindex(out.index).fillna(0)
    out["excess_wait_person_hours_year"] = out["excess_wait_person_hours_day"] * sv["workdays_per_year"]
    out.index.name = "hour"
    return out.reset_index()
