"""
08Validate.py

07번 결과를 믿을 수 있는지 검증한다.
"못 탄 사람"은 데이터에 없어서 정답과 비교할 수가 없다. 대신 아래를 확인한다.
결과는 전부 요약표라 반출 가능 (10번이 export/ 폴더로 옮김).

1. cv_ablation: SKT 데이터를 넣으면 예측이 더 정확해지는가
   - 자리가 넉넉했던 칸(실제 승차 = 수요)으로 학습하고 맞춰보기
       무작위 5-fold   : 칸을 무작위로 5등분. 이미 본 정류장의 다른 시간대를 맞히는 시험
       정류장 제외(LOSO): 정류장 하나를 통째로 빼고 맞히는 시험 (노선을 넓힐 때 쓸 수 있는가)
   - no_skt보다 오차가 작으면 = SKT 데이터가 도움이 된다
   - 통근 보정을 넣어서 출퇴근 시간 오차가 작아지면 = 직장인 비중 반영이 도움이 된다

2. saturation_curve: 버스가 찰수록 승차가 줄어드는가
   - 실제 승차 / 모델 예측 수요를 버스 도착 시 재차율 구간별로
   - 버스가 꽉 찰수록 1보다 떨어지면 = 자리가 없어서 못 탔다는 증거

3. sensitivity: 설정을 바꿔도 결과가 비슷한가
   - 반경(100/150/200m) x tau(0.5/0.6/0.7) x 수요신호 종류
   - 기본 설정과의 순위 상관(Spearman), 위험 top10이 얼마나 겹치는지

4. bunching_effect: 버스 몰림을 무시하면 결과가 얼마나 달라지는가
   - 실측(버스별 실제 공백·재차) vs 모든 버스가 평균 간격·평균 재차로 고르게 왔다고 칠 때

5. measured_headway: 교통카드로 잰 정류장별 실제 배차 공백과 몰림 정도 (출근 / 퇴근)

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
COMMUTE_PEAK_HOURS = [7, 8, 9, 17, 18, 19]  # 통근 보정이 작동하는 시간 (07번 오전첨두·오후첨두)

CV_VARIANTS = [  # (표에 쓸 이름, 수요신호, 통근 보정)
    ("no_skt", "no_skt", False),
    ("flow", "flow", False),
    ("flow_x_share_hourly", "flow_x_share_hourly", False),
    ("flow_x_share", "flow_x_share", False),
    ("flow_x_share+통근", "flow_x_share", True),
]


def _poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    mu = np.maximum(mu, 1e-9)
    term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return float(2 * np.mean(term - (y - mu)))


# %%
def cv_ablation(features: pd.DataFrame, cfg, k: int = 5, seed: int = 42) -> pd.DataFrame:
    """두 가지 나눔(무작위 5-fold / 정류장 제외)으로 수요신호별 예측 오차 비교. 모든 신호가 같은 칸·같은 나눔.

    mae_improvement_vs_no_skt = 1 - (MAE / 같은 방식 no_skt의 MAE)   (0.2면 SKT 없을 때보다 오차 20% 감소)
    mae_peak_hours: 출퇴근 첨두(7~9시, 17~19시) 칸만의 MAE. 통근 보정 효과는 여기서 봐야 함
    """
    base = m07.add_capacity_terms(m07.add_demand_signal(features, "no_skt", cfg.daytype), cfg)
    pool_idx = base.index[base["unconstrained"] & base["has_skt"]]
    rng = np.random.default_rng(seed)
    splits = {
        "무작위 5-fold": rng.integers(0, k, size=len(pool_idx)),
        "정류장 제외(LOSO)": base.loc[pool_idx, "seq"].to_numpy(),
    }
    return pd.concat([_cv_once(features, cfg, pool_idx, folds).assign(cv_type=name) for name, folds in splits.items()],
                     ignore_index=True)


def _cv_once(features: pd.DataFrame, cfg, pool_idx, folds: np.ndarray) -> pd.DataFrame:
    rows = []
    for label, v, commute in CV_VARIANTS:
        try:
            df = m07.add_capacity_terms(m07.add_demand_signal(features, v, cfg.daytype), cfg).loc[pool_idx]
        except ValueError as e:
            print(f"{v} 생략: {e}")
            continue
        y_all, pred_all, hour_all = [], [], []
        for f in np.unique(folds):
            train, test = df[folds != f], df[folds == f]
            if test.empty or train.empty:
                continue
            model = m07.fit_demand_model(train, v, commute)
            pred_all.append(model.predict(test))
            y_all.append(test["daily_boarding"].to_numpy(float))
            hour_all.append(test["hour"].to_numpy())
        y, pred = np.concatenate(y_all), np.concatenate(pred_all)
        peak = np.isin(np.concatenate(hour_all), COMMUTE_PEAK_HOURS)
        ss_res = np.sum((np.log1p(y) - np.log1p(pred)) ** 2)
        ss_tot = np.sum((np.log1p(y) - np.log1p(y).mean()) ** 2)
        rows.append({
            "signal_variant": label,
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
def sensitivity(features_by_radius: dict, cfg, ref_radius: float, taus: list[float], variants: list[str],
                top_n: int = 10) -> pd.DataFrame:
    """모든 설정 조합을 돌려서 기본 설정과 비교 (순위 상관, 위험 top10 겹침)."""
    results = {}
    for radius, feats in features_by_radius.items():
        for tau in taus:
            for v in variants:
                try:
                    scored, _, diag = m07.run_model(feats, replace(cfg, tau=tau, signal=v))
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
    g = e.groupby(["seq", "band"])
    gap = g["gap_min"]
    return pd.DataFrame({
        "하루_운행대수": g.size() / n_days,
        "평균공백_분": gap.mean(),
        "몰림정도_CV": gap.std() / gap.mean(),
        "1분이내_연속도착_비율": gap.apply(lambda x: float((x.dropna() <= 1).mean())),
        "평균의2배넘는_공백_비율": gap.apply(lambda x: float((x.dropna() > 2 * x.mean()).mean())),
        "거의만석_도착_비율": g["load_before"].apply(lambda x: float((x >= 40).mean())),
    }).reset_index()
