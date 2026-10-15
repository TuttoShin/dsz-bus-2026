"""
08Validate.py

07번 결과를 믿을 수 있는지 검증한다.
"못 탄 사람"은 데이터에 없어서 정답과 비교할 수가 없다. 대신 아래를 확인한다.
결과는 전부 요약표라 반출 가능 (10번이 export/ 폴더로 옮김).

1. cv_ablation: SKT 데이터와 체류 목적 보정을 넣으면 정류장 대기 수요 예측이 더 정확해지는가
   - 9401 자리가 넉넉했던 칸(정류장 전체 노선 승차 = 대기 수요)으로 학습하고 맞춰보기 (정류장 x 요일 x 시간대)
       무작위 5-fold   : 칸을 무작위로 5등분. 이미 본 정류장의 다른 시간대를 맞히는 시험
       정류장 제외(LOSO): 정류장 하나를 통째로 빼고 맞히는 시험 (노선을 넓힐 때 쓸 수 있는가)
   - no_skt보다 오차가 작으면 = SKT 데이터가 도움이 된다
   - flow_only(유동인구 x 평균 대기 비율)보다 stay(체류 목적 보정)가 작으면 = 정류장·요일·시간 배수가 도움이 된다

2. saturation_curve: 버스가 찰수록 승차가 줄어드는가
   - 실제 승차 / 모델 예측 수요를 버스 도착 시 재차율 구간별로
   - 버스가 꽉 찰수록 1보다 떨어지면 = 자리가 없어서 못 탔다는 증거

3. sensitivity: 설정을 바꿔도 결과가 비슷한가
   - 반경(100/150/200m) x tau(0.5/0.6/0.7) x 수요신호 종류
   - 기본 설정과의 순위 상관(Spearman), 위험 top10이 얼마나 겹치는지

4. bunching_effect: 버스 몰림을 무시하면 결과가 얼마나 달라지는가
   - 실측(버스별 실제 공백·재차) vs 모든 버스가 평균 간격·평균 재차로 고르게 왔다고 칠 때

5. measured_headway: 교통카드로 잰 정류장별 실제 배차 공백과 몰림 정도 (출근 / 퇴근)

6. 장표용 (18쪽 모델 신뢰성 검증)
   ① reproduction_error: 자리 넉넉 칸에서 예측 수요 vs 실제 승차 오차 (WAPE)
   ② time_split: 1~3주로 만든 위험 칸이 4주차 실제 만석 칸을 얼마나 맞히는가
   ③ demand_sensitivity: 수요 ±20%에도 위험 top10이 유지되는가
   ④ skt_contribution: SKT 없이 돌릴 때보다 위험 칸을 몇 개 더 잡는가
   bus_profile: 정류장·시간대별 남은 좌석, 만석 도착, 도착 간격·몰림 비율 (13·14쪽 그림)

CLI 없음 (RunAll.py에서 호출)
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from Loader import load_module  # noqa: E402

m07 = load_module("modeling/07BoardingProbability.py")

# 재차율(재차/좌석) 구간. 0.6 이상은 칸이 몇 개 안 돼서 하나로 합침
SATURATION_BINS = [0, 0.2, 0.4, 0.6, np.inf]
PEAK_HOURS = [7, 8, 9, 17, 18, 19]  # 출퇴근 첨두


def _poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    mu = np.maximum(mu, 1e-9)
    term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return float(2 * np.mean(term - (y - mu)))


# %%
def cv_ablation(stay_features: dict, cfg, k: int = 5, seed: int = 42) -> pd.DataFrame:
    """두 가지 나눔(무작위 5-fold / 정류장 제외)으로 수요 모델 변형별 정류장 대기 수요 예측 오차 비교.

    칸 = 정류장 x 요일 x 시간대 중 9401 자리가 넉넉했던 칸 (정류장 전체 노선 승차 = 대기 수요)
    mae_improvement_vs_no_skt = 1 - (MAE / 같은 방식 no_skt의 MAE)   (0.2면 SKT 없을 때보다 오차 20% 감소)
    mae_peak_hours: 출퇴근 첨두(7~9시, 17~19시) 칸만의 MAE
    """
    cells = m07.stay_cells(stay_features, cfg)
    pool = cells[cells["unconstrained"] & (cells["flow_day"] > 0)].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    splits = {
        "무작위 5-fold": rng.integers(0, k, size=len(pool)),
        "정류장 제외(LOSO)": pool["seq"].to_numpy(),
    }
    return pd.concat([_cv_once(pool, folds).assign(cv_type=name) for name, folds in splits.items()], ignore_index=True)


def _cv_once(pool: pd.DataFrame, folds: np.ndarray) -> pd.DataFrame:
    rows = []
    for v in m07.SIGNAL_VARIANTS:
        y_all, pred_all, hour_all = [], [], []
        for f in np.unique(folds):
            train, test = pool[folds != f], pool[folds == f]
            if test.empty or train.empty:
                continue
            model = m07.fit_stay_model(train, v)
            pred_all.append(model.predict_cells(test.reset_index(drop=True)))
            y_all.append(test["station_board"].to_numpy(float))
            hour_all.append(test["hour"].to_numpy())
        y, pred = np.concatenate(y_all), np.concatenate(pred_all)
        peak = np.isin(np.concatenate(hour_all), PEAK_HOURS)
        ss_res = np.sum((np.log1p(y) - np.log1p(pred)) ** 2)
        ss_tot = np.sum((np.log1p(y) - np.log1p(y).mean()) ** 2)
        rows.append({
            "signal_variant": v,
            "n_cells": len(y),
            "mae_daily_boarding": float(np.mean(np.abs(y - pred))),
            "n_cells_peak": int(peak.sum()),
            "mae_peak_hours": float(np.mean(np.abs(y[peak] - pred[peak]))) if peak.any() else np.nan,
            "rmse_daily_boarding": float(np.sqrt(np.mean((y - pred) ** 2))),
            "poisson_deviance": _poisson_deviance(y, pred),
            "r2_log": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        })
    out = pd.DataFrame(rows)
    if "no_skt" in set(out["signal_variant"]):
        ref = out.loc[out["signal_variant"] == "no_skt", "mae_daily_boarding"].iloc[0]
        out["mae_improvement_vs_no_skt"] = 1 - out["mae_daily_boarding"] / ref
        ref_peak = out.loc[out["signal_variant"] == "no_skt", "mae_peak_hours"].iloc[0]
        out["mae_peak_improvement_vs_no_skt"] = 1 - out["mae_peak_hours"] / ref_peak
    return out


# %%
def saturation_curve(scored: pd.DataFrame) -> pd.DataFrame:
    """버스 도착 시 재차율 구간별 상대 승차율 = 실제 승차 / 모델 예측 수요.

    모델 예측에 정류장·시간대 차이가 이미 들어 있어서 "출퇴근이라 원래 많이 탄다"는 효과가 빠진다.
    '떠날 때' 재차율은 그 정류장 승차가 포함돼서 '도착할 때' 재차율로 구간을 나눈다.
    """
    d = scored[scored["has_skt"] & (scored["demand_pred"] > 0)].copy()
    d["rel_capture"] = d["daily_boarding"] / d["demand_pred"]
    d = d[np.isfinite(d["rel_capture"])]
    d["load_bin"] = pd.cut(d["load_before_ratio"], SATURATION_BINS, right=False)
    out = d.groupby("load_bin", observed=False).agg(
        n_cells=("rel_capture", "size"),
        rel_capture_mean=("rel_capture", "mean"),
        rel_capture_median=("rel_capture", "median"),
    ).reset_index()
    out["load_bin"] = out["load_bin"].astype(str)
    return out


# %%
def sensitivity(features_by_radius: dict, stay_by_radius: dict, cfg, ref_radius: float, taus: list[float],
                variants: list[str], top_n: int = 10) -> pd.DataFrame:
    """모든 설정 조합을 돌려서 기본 설정과 비교 (순위 상관, 위험 top10 겹침). 반경마다 체류 목적 보정도 다시 학습."""
    results = {}
    for radius, feats in features_by_radius.items():
        for tau in taus:
            for v in variants:
                try:
                    scored, _, diag = m07.run_model(feats, replace(cfg, tau=tau, signal=v, stay_features=stay_by_radius[radius]))
                except ValueError as e:
                    print(f"sensitivity 생략 (r={radius}, tau={tau}, {v}): {e}")
                    continue
                p = scored[scored["has_skt"]].set_index(["seq", "hour"])["p_board_mid"].dropna()
                results[(radius, tau, v)] = (p, scored, diag)
    ref_key = (ref_radius, cfg.tau, cfg.signal)
    if ref_key not in results:
        raise ValueError(f"기준 설정 {ref_key} 결과가 없습니다.")
    ref = results[ref_key][0]
    ref_top = set(ref.nsmallest(top_n).index)
    rows = []
    for (radius, tau, v), (p, scored, diag) in results.items():
        common = ref.index.intersection(p.index)
        skt = scored[scored["has_skt"]]
        rows.append({
            "radius_m": radius, "tau": tau, "signal_variant": v,
            "is_reference": (radius, tau, v) == ref_key,
            "spearman_vs_ref": float(ref.loc[common].rank().corr(p.loc[common].rank())),
            f"top{top_n}_overlap_vs_ref": len(ref_top & set(p.nsmallest(top_n).index)) / top_n,
            "share_grade_위험": float((skt["grade"] == "위험").mean()),
            "p_board_mid_mean": float(p.mean()),
            "n_train_cells": diag["n_train_cells"],
        })
    return pd.DataFrame(rows).sort_values(["signal_variant", "radius_m", "tau"]).reset_index(drop=True)


# %%
def bunching_effect(features: pd.DataFrame, cfg, model=None, hours: list[int] | None = None) -> pd.DataFrame:
    """실측(버스별 공백·재차) vs 고르게 왔다고 칠 때. 차이 = 버스 몰림 때문에 생기는 탑승 실패."""
    rows = []
    for name, c in (("실측(버스별 공백·재차)", cfg), ("몰림 무시(평균 간격·평균 재차)", replace(cfg, uniform=True))):
        scored, _, _ = m07.run_model(features, c, model=model)
        d = scored[scored["has_skt"] & scored["p_board_mid"].notna()]
        if hours is not None:
            d = d[d["hour"].isin(hours)]
        rows.append({
            "방식": name,
            "p_board_mid_mean": float(d["p_board_mid"].mean()),
            "p_board_mid_min": float(d["p_board_mid"].min()),
            "n_cells_위험": int((d["grade"] == "위험").sum()),
            "n_cells_보통": int((d["grade"] == "보통").sum()),
            "추가대기_분_평균": float(d["extra_wait_min_mid"].mean()),
        })
    return pd.DataFrame(rows)


def measured_headway(ev: pd.DataFrame, features: pd.DataFrame, daytype: str, hours: list[int]) -> pd.DataFrame:
    """교통카드로 잰 정류장별 실제 배차 공백과 몰림 정도 (출근 / 퇴근). 반출용 집계."""
    e = m07.events_for(ev, daytype)
    e = e[e["hour"].isin(hours) & e["seq"].isin(features["seq"].unique())].copy()
    e["band"] = np.where(e["hour"] < 12, "출근", "퇴근")
    n_days = max(e["date"].nunique(), 1)
    # 몰림 비율: 앞차와 간격이 그 정류장·시간대 평균 간격의 50% 미만 (장표 14쪽 정의)
    e["bunch"] = (e["gap_min"] < BUNCH_RATIO * e.groupby(["seq", "hour"])["gap_min"].transform("mean")).astype(float)
    e["bunch"] = e["bunch"].where(e["gap_min"].notna())
    g = e.groupby(["seq", "band"])
    gap = g["gap_min"]
    return pd.DataFrame({
        "하루_운행대수": g.size() / n_days,
        "평균공백_분": gap.mean(),
        "공백표준편차_분": gap.std(),
        "몰림정도_CV": gap.std() / gap.mean(),
        "몰림비율(평균간격50%미만)": g["bunch"].mean(),
        "1분이내_연속도착_비율": gap.apply(lambda x: float((x.dropna() <= 1).mean())),
        "평균의2배넘는_공백_비율": gap.apply(lambda x: float((x.dropna() > 2 * x.mean()).mean())),
        "거의만석_도착_비율": g["load_before"].apply(lambda x: float((x >= 40).mean())),
    }).reset_index()


# %%
# ---- 장표(13~18쪽)용 집계
BUNCH_RATIO = 0.5  # 앞차와 간격이 그 정류장·시간대 평균 간격의 50% 미만이면 "몰려 온 버스"
TEST_FROM_DAY = 22  # 시간 분할 검증: 1~21일(1~3주)로 학습 -> 22일~(4주차) 시험
FULL_HIT_SHARE = 0.3  # 시간 분할 검증: 그 칸 버스의 30% 이상이 만석으로 떠나면 "만석 칸"
DEMAND_SCALES = (0.8, 1.0, 1.2)  # 수요 ±20% 민감도


def bus_profile(ev: pd.DataFrame, seat_capacity: float, daytypes=("weekday", "weekend")) -> pd.DataFrame:
    """정류장·시간대별 버스 도착 지표 (13·14쪽). 그림·요약용 집계.

    trips_per_day      : 하루 평균 운행 횟수
    gap_mean / gap_sd  : 앞차와의 실제 도착 간격 평균·표준편차 (분), gap_cv = 표준편차 / 평균
    bunch_share        : 몰림 비율 = 간격이 그 정류장·시간대 평균 간격의 50% 미만인 버스 비율
    seats_left_mean    : 도착할 때 남은 좌석 평균 (좌석 정원 - 도착 시 재차), p25 / p75
    full_arrival_share : 만석(남은 좌석 1석 이하)으로 도착한 버스 비율
    """
    rows = []
    for dt in daytypes:
        e = m07.events_for(ev, dt).copy()
        n_days = max(e["date"].nunique(), 1)
        e["seats_left"] = (seat_capacity - e["load_before"]).clip(lower=0)
        mean = e.groupby(["seq", "hour"])["gap_min"].transform("mean")
        e["bunch"] = (e["gap_min"] < BUNCH_RATIO * mean).astype(float).where(e["gap_min"].notna())
        e["full_arr"] = (e["seats_left"] <= 1).astype(float)
        g = e.groupby(["seq", "hour"])
        rows.append(pd.DataFrame({
            "trips_per_day": g.size() / n_days,
            "gap_mean": g["gap_min"].mean(),
            "gap_sd": g["gap_min"].std(),
            "bunch_share": g["bunch"].mean(),
            "seats_left_mean": g["seats_left"].mean(),
            "seats_left_p25": g["seats_left"].quantile(0.25),
            "seats_left_p75": g["seats_left"].quantile(0.75),
            "full_arrival_share": g["full_arr"].mean(),
        }).reset_index().assign(daytype=dt))
    out = pd.concat(rows, ignore_index=True)
    out["gap_cv"] = out["gap_sd"] / out["gap_mean"]
    return out


def reproduction_error(scored: pd.DataFrame) -> float:
    """① 재현도: 자리가 넉넉했던 칸(실제 승차 = 수요)에서 모델 예측 수요와 실제 승차의 차이 (WAPE).
    WAPE = |예측 - 실제| 합 / 실제 합. 0.2 = 칸마다 평균 20% 정도 어긋남"""
    u = scored[scored["unconstrained"] & scored["has_skt"] & scored["demand_pred"].notna()]
    tot = u["daily_boarding"].sum()
    return float((u["demand_pred"] - u["daily_boarding"]).abs().sum() / tot) if tot > 0 else np.nan


def time_split(ev: pd.DataFrame, make_features, cfg, test_from_day: int = TEST_FROM_DAY) -> tuple[dict, pd.DataFrame]:
    """② 시간 분할 검증: 1~3주 데이터만으로 모델·확률을 만들고, 4주차에 실제로 만석 버스가 나온 칸을 맞혔는가.

    make_features(버스별 기록) -> (평일 features, 요일별 features) (RunAll이 넘김. 같은 SKT·분담률에 그 기간 버스·정류장 승차만)
    반환: (요약 dict, 1~3주 등급별 4주차 만석 칸 비율 표)
    """
    day = ev["date"].str[6:8].astype(int)
    ev_tr, ev_te = ev[day < test_from_day], ev[day >= test_from_day]
    if m07.events_for(ev_te, cfg.daytype)["date"].nunique() < 3:
        print(f"시간 분할 검증 생략: {test_from_day}일 이후 {cfg.daytype} 날짜가 3일 미만")
        return {}, pd.DataFrame()
    f_tr, stay_tr = make_features(ev_tr)
    f_te, stay_te = make_features(ev_te)
    sc_tr, model, _ = m07.run_model(f_tr, replace(cfg, bus_events=ev_tr, stay_features=stay_tr))
    sc_te, _, _ = m07.run_model(f_te, replace(cfg, bus_events=ev_te, stay_features=stay_te), model=model)
    key = ["seq", "hour"]
    a = sc_tr[sc_tr["has_skt"] & (sc_tr["n_buses"] > 0)][key + ["grade", "p_board_mid"]]
    b = sc_te[sc_te["has_skt"] & (sc_te["n_buses"] > 0)][key + ["full_bus_share"]]
    m = a.merge(b, on=key)
    m["full_te"] = m["full_bus_share"].fillna(0) >= FULL_HIT_SHARE
    risk = m["grade"] == "위험"
    by_grade = m.groupby("grade").agg(**{"칸수": ("full_te", "size"), "4주차_만석칸_비율": ("full_te", "mean"),
                                         "4주차_만석으로떠난버스_비율": ("full_bus_share", "mean")}).reset_index()
    summary = {
        "학습기간": f"1~{test_from_day - 1}일", "시험기간": f"{test_from_day}일~",
        "학습_위험칸수": int(risk.sum()), "시험_만석칸수": int(m["full_te"].sum()),
        "적중률(위험예측중_실제만석)": float(m.loc[risk, "full_te"].mean()) if risk.any() else np.nan,
        "포착률(실제만석중_위험예측)": float(risk[m["full_te"]].mean()) if m["full_te"].any() else np.nan,
        "시험기간_재현오차_WAPE": reproduction_error(sc_te),  # 1~3주 모델로 4주차 자리 넉넉 칸 승차 예측
    }
    return summary, by_grade


def demand_sensitivity(features: pd.DataFrame, cfg, model, scales=DEMAND_SCALES, top_n: int = 10) -> pd.DataFrame:
    """③ 수요를 ±20% 바꿔도 위험한 칸이 그대로인가 (같은 모델, 줄 서기에 넣는 수요만 바꿈)."""
    res = {}
    for s in scales:
        sc, _, _ = m07.run_model(features, replace(cfg, demand_scale=s), model=model)
        res[s] = sc[sc["has_skt"] & sc["p_board_mid"].notna()].set_index(["seq", "hour"])
    ref = res[1.0] if 1.0 in res else res[scales[len(scales) // 2]]
    ref_top = set(ref["p_board_mid"].nsmallest(top_n).index)
    return pd.DataFrame([{
        "수요배율": s,
        "위험칸수": int((d["grade"] == "위험").sum()),
        "평균_탑승확률": float(d["p_board_mid"].mean()),
        "평균_기대대기_분": float(d["wait_total_min"].mean()),
        f"위험top{top_n}_겹침": len(ref_top & set(d["p_board_mid"].nsmallest(top_n).index)) / top_n,
    } for s, d in res.items()])


def skt_contribution(features: pd.DataFrame, cfg, scored: pd.DataFrame) -> dict:
    """④ SKT 기여: SKT 없이(정류장 배수 x 요일·시간 배수만) 돌렸을 때보다 위험 칸을 몇 개 더 잡는가."""
    sc0, _, _ = m07.run_model(features, replace(cfg, signal="no_skt"))
    risk = lambda d: set(d.loc[d["has_skt"] & (d["grade"] == "위험"), ["seq", "hour"]].itertuples(index=False, name=None))
    with_skt, without = risk(scored), risk(sc0)
    return {"SKT반영_위험칸수": len(with_skt), "SKT없이_위험칸수": len(without),
            "SKT로_추가포착": len(with_skt - without), "SKT로_빠진칸": len(without - with_skt)}
