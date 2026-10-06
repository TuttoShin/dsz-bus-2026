"""
08Validate.py

09번 결과를 믿을 수 있는지 검증한다.
"못 탄 사람"은 데이터에 없어서 정답과 비교할 수가 없다. 대신 아래 4가지를 확인한다.
결과는 전부 요약표라 반출 가능 (10번이 export/ 폴더로 옮김).

1. cv_ablation: SKT 데이터를 넣으면 예측이 더 정확해지는가
   - 자리가 넉넉했던 칸(실제 승차 = 수요)을 5등분해서, 4개로 학습하고 1개로 맞춰보기를 5번 반복
   - 비교: no_skt / flow / flow_x_share / flow_x_share + 통근 보정
   - no_skt보다 오차가 작으면 = SKT 데이터가 도움이 된다
   - 통근 보정을 넣어서 오차가 더 작아지면 = 직장인 비중 반영이 도움이 된다 (PPT 17쪽 근거)

2. saturation_curve: 버스가 찰수록 승차가 줄어드는가
   - 상대 승차율 = (승차 인원 / 주변 유동인구) / 그 정류장의 평소 값
   - 버스가 꽉 찰수록 이 값이 1보다 떨어지면 = 자리가 없어서 못 탔다는 증거

3. sensitivity: 설정을 바꿔도 결과가 비슷한가
   - 반경(100/150/200m) x tau(0.5/0.6/0.7) x 수요신호 종류를 모두 돌려봄
   - 기본 설정과의 순위 상관(Spearman, 1에 가까울수록 같음), 위험 top10이 얼마나 겹치는지

4. rule_vs_model: 07번(가중합 점수)과 09번(확률)이 위험 순위를 비슷하게 매기는가

CLI: python modeling/08Validate.py  (반경 비교는 빠짐. 전체는 RunAll.py에서)
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from Loader import load_module  # noqa: E402

m09 = load_module("modeling/09BoardingProbability.py")

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
SATURATION_BINS = [0, 0.2, 0.4, 0.6, 0.8, 1.0, np.inf]  # 재차율(재차/좌석) 구간


# %%
def _poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    """포아송 모델용 오차 지표 (0에 가까울수록 잘 맞음)."""
    mu = np.maximum(mu, 1e-9)
    term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return float(2 * np.mean(term - (y - mu)))


CV_VARIANTS = [  # (표에 쓸 이름, 수요신호, 통근 보정)
    ("no_skt", "no_skt", False),
    ("flow", "flow", False),
    ("flow_x_share", "flow_x_share", False),
    ("flow_x_share+통근", "flow_x_share", True),
]


def cv_ablation(features: pd.DataFrame, cfg, k: int = 5, seed: int = 42) -> pd.DataFrame:
    """공정한 비교를 위해 모든 수요신호가 같은 칸, 같은 5등분을 쓴다.

    지표: MAE(평균 오차, 명), RMSE, 포아송 오차, R2(log 기준, 1에 가까울수록 좋음)
    mae_improvement_vs_no_skt = 1 - (MAE / no_skt의 MAE)   (0.2면 SKT 없을 때보다 오차 20% 감소)
    """
    base = m09.add_capacity_terms(m09.add_demand_signal(features, "no_skt"), cfg)
    pool_idx = base.index[base["unconstrained"] & base["has_skt"]]
    rng = np.random.default_rng(seed)
    folds = rng.integers(0, k, size=len(pool_idx))

    rows = []
    for label, v, commute in CV_VARIANTS:
        try:
            df = m09.add_capacity_terms(m09.add_demand_signal(features, v), cfg).loc[pool_idx]
        except ValueError as e:
            print(f"{v} 생략: {e}")
            continue
        y_all, pred_all = [], []
        for f in range(k):
            train, test = df[folds != f], df[folds == f]
            if test.empty or train.empty:
                continue
            model = m09.fit_demand_model(train, v, commute)
            pred_all.append(model.predict(test))
            y_all.append(test["daily_boarding"].to_numpy(float))
        y, pred = np.concatenate(y_all), np.concatenate(pred_all)
        ss_res = np.sum((np.log1p(y) - np.log1p(pred)) ** 2)
        ss_tot = np.sum((np.log1p(y) - np.log1p(y).mean()) ** 2)
        rows.append({
            "signal_variant": label,
            "n_cells": len(y),
            "mae_daily_boarding": float(np.mean(np.abs(y - pred))),
            "rmse_daily_boarding": float(np.sqrt(np.mean((y - pred) ** 2))),
            "poisson_deviance": _poisson_deviance(y, pred),
            "r2_log": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        })
    out = pd.DataFrame(rows)
    if "no_skt" in set(out["signal_variant"]):
        ref = out.loc[out["signal_variant"] == "no_skt", "mae_daily_boarding"].iloc[0]
        out["mae_improvement_vs_no_skt"] = 1 - out["mae_daily_boarding"] / ref
    return out


# %%
def saturation_curve(scored: pd.DataFrame) -> pd.DataFrame:
    """버스 도착 시 재차율 구간별 상대 승차율.

    상대 승차율 = 실제 승차 / 09번 모델이 예측한 수요(자리가 충분했다면 탔을 인원)
      1에 가까움   = 예측한 만큼 다 탔다
      1보다 작음   = 자리가 없어서 덜 탔다 (수요가 좌석 수에서 잘림)
    모델 예측에 정류장·시간대 차이가 이미 들어 있어서, "출퇴근 시간이라 원래 많이 탄다"는 효과가 빠진다.
    (승차/유동인구를 그대로 쓰면 붐비는 시간대 = 원래 많이 타는 시간대라 곡선이 거꾸로 올라감)
    """
    d = scored[scored["has_skt"] & (scored["demand_pred"] > 0)].copy()
    d["rel_capture"] = d["daily_boarding"] / d["demand_pred"]
    d = d[np.isfinite(d["rel_capture"])]
    # 버스가 '도착할 때' 재차율로 구간을 나눈다.
    # '떠날 때' 재차율은 그 정류장 승차가 포함돼서, 승차 많은 칸이 저절로 높은 구간에 몰림 (곡선이 거꾸로 올라감)
    d["load_bin"] = pd.cut(d["load_before_ratio_mid"], SATURATION_BINS, right=False)
    out = d.groupby("load_bin", observed=False).agg(
        n_cells=("rel_capture", "size"),
        rel_capture_mean=("rel_capture", "mean"),
        rel_capture_median=("rel_capture", "median"),
    ).reset_index()
    out["load_bin"] = out["load_bin"].astype(str)
    return out


# %%
def _cell_key(scored: pd.DataFrame) -> pd.Series:
    """(정류장, 시간대)별 탑승확률. 설정끼리 비교할 때 씀."""
    d = scored[scored["has_skt"]]
    return d.set_index(["seq", "hour"])["p_board_mid"]


def sensitivity(
    features_by_radius: dict[float, pd.DataFrame],
    cfg,
    ref_radius: float,
    taus: list[float],
    variants: list[str],
    top_n: int = 10,
) -> pd.DataFrame:
    """모든 설정 조합을 돌려서 기본 설정과 비교한다.

    spearman_vs_ref    : 기본 설정과 확률 순위가 얼마나 같은가 (1 = 완전히 같음)
    top10_overlap_vs_ref: 위험 top10 중 몇 %가 겹치는가
    """
    results = {}
    for radius, feats in features_by_radius.items():
        for tau in taus:
            for v in variants:
                try:
                    scored, _, diag = m09.run_model(feats, replace(cfg, tau=tau, signal=v))
                except ValueError as e:
                    print(f"sensitivity 생략 (r={radius}, tau={tau}, {v}): {e}")
                    continue
                results[(radius, tau, v)] = (_cell_key(scored), scored, diag)

    ref_key = (ref_radius, cfg.tau, cfg.signal)
    if ref_key not in results:
        raise ValueError(f"기준 설정 {ref_key} 결과가 없습니다.")
    ref = results[ref_key][0]
    ref_top = set(ref.nsmallest(top_n).index)

    rows = []
    for (radius, tau, v), (p, scored, diag) in results.items():
        common = ref.index.intersection(p.index)
        rho = ref.loc[common].rank().corr(p.loc[common].rank())
        skt = scored[scored["has_skt"]]
        rows.append({
            "radius_m": radius, "tau": tau, "signal_variant": v,
            "is_reference": (radius, tau, v) == ref_key,
            "spearman_vs_ref": float(rho),
            f"top{top_n}_overlap_vs_ref": len(ref_top & set(p.nsmallest(top_n).index)) / top_n,
            "share_grade_위험": float((skt["grade"] == "위험").mean()),
            "p_board_mid_mean": float(p.mean()),
            "n_train_cells": diag["n_train_cells"],
        })
    return pd.DataFrame(rows).sort_values(["signal_variant", "radius_m", "tau"]).reset_index(drop=True)


# %%
def rule_vs_model(rule_scored: pd.DataFrame, model_scored: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """07번 점수와 09번 확률의 순위 상관, 위험 top10 겹침 비율."""
    key = ["seq", "hour"]
    m = model_scored.loc[model_scored["has_skt"], key + ["p_board_mid"]].merge(
        rule_scored[key + ["boarding_margin_index"]], on=key
    )
    rows = []
    for name, sub in [("전체 시간대", m)]:
        rho = sub["p_board_mid"].rank().corr(sub["boarding_margin_index"].rank())
        top_model = set(map(tuple, sub.nsmallest(top_n, "p_board_mid")[key].to_numpy()))
        top_rule = set(map(tuple, sub.nsmallest(top_n, "boarding_margin_index")[key].to_numpy()))
        rows.append({
            "scope": name, "n_cells": len(sub), "spearman": float(rho),
            f"top{top_n}_overlap": len(top_model & top_rule) / top_n,
        })
    return pd.DataFrame(rows)


# %%
def main() -> None:
    features = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    cfg = m09.ModelConfig()
    scored, _, _ = m09.run_model(features, cfg)

    print("[1] CV ablation")
    print(cv_ablation(features, cfg).round(3).to_string(index=False))
    print("\n[2] 포화 곡선")
    print(saturation_curve(scored).round(3).to_string(index=False))
    print("\n[3] tau x 신호 sensitivity (반경 고정)")
    print(sensitivity({0: features}, cfg, 0, [0.5, 0.6, 0.7], m09.SIGNAL_VARIANTS).round(3).to_string(index=False))

    rule_path = PROCESSED_DIR / "scores.parquet"
    if rule_path.exists():
        print("\n[4] 규칙 기반(07) vs 모델(09)")
        print(rule_vs_model(pd.read_parquet(rule_path), scored).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
