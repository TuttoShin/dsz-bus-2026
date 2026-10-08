"""
09BoardingProbability.py

정류장x시간대별로 "기다리는 사람이 이번 버스를 탈 수 있을 확률"을 계산한다.

왜 필요한가)
승하차 데이터에는 버스를 탄 사람만 있다. 버스가 꽉 차서 못 탄 사람은 기록이 없다.
그래서 SKT 유동인구(정류장 주변에 있던 사람 수)로 "원래 타려던 사람 수"를 추정한다.

1. 수요 복원: 원래 타려던 사람 수 추정
   - 버스에 자리가 넉넉했던 정류장/시간대만 골라서 학습 (이때는 승차 인원 = 타려던 사람 수)
   - 학습하는 관계식:
       log(하루 평균 승차 인원) = log(수요신호) + 정류장 계수 + 시간대 계수
                                 + 오전 통근 계수 x 통근 비율 x [7~9시]
                                 + 오후 통근 계수 x 통근 비율 x [17~19시]
       수요신호 = 주변 유동인구 x 9401 분담률   (복수 노선 정류장에서 9401 몫만, 정류장마다 값 하나)
       통근 비율 = (평일 유동인구 - 주말 유동인구) / 평일 유동인구   (0~1, 요일별 데이터)
   - 버스가 꽉 찼던 칸에 이 식을 적용하면 "자리만 있었으면 탔을 인원"이 나온다
   - 못 탄 인원(unmet) = 원래 타려던 인원 - 실제 탄 인원

2. 탑승가능확률
   - 버스 1대가 왔을 때
       남은 좌석 R = 좌석수(41) - 버스 1대당 이미 타고 있는 인원
       기다리는 사람 수 D: 평균 lambda인 포아송 분포
       lambda = 하루 평균 타려던 인원 x 배차간격(분) / 60
       탑승가능확률 P = E[min(D, R)] / lambda
         (= 기다리던 사람 중 이번 버스에 탄 사람의 비율)
   - 배차간격을 정확히 몰라서(3~7분) 확률도 범위로 남긴다
       p_board_high: 3분 (버스 많음 -> 확률 높음)
       p_board_mid : 5분
       p_board_low : 7분 (버스 적음 -> 확률 낮음)

3. 추가 대기시간
   - 매 버스마다 확률 P로 탈 수 있다면, 놓치는 버스 수 평균 = 1/P - 1
   - 추가 대기시간(분) = 배차간격 x (1/P - 1)   (60분 이상은 60으로 표시)

수요신호 종류 (08번에서 서로 비교)
   no_skt       : SKT 안 씀. 정류장 계수 x 시간대 계수만 (비교 기준)
   flow         : 주변 유동인구 그대로
   flow_x_share        : 주변 유동인구 x 9401 분담률(정류장마다 하루 전체로 한 번 계산) (기본값)
   flow_x_share_hourly : 주변 유동인구 x 9401 분담률(시간대별) (비교용)
                         9401이 꽉 찬 시간대엔 못 탄 사람이 승차에 안 잡혀 분담률이 낮게 나옴
                         -> 수요를 복원해야 할 시간대에 신호가 작아지는 순환 문제가 있어서 기본값에서 뺌
   참고: 정류장마다 고정된 분담률은 정류장 계수에 흡수돼서 탑승확률은 flow와 똑같이 나온다.
         즉 분담률의 크기(서울역 = 1/20 등)가 정확하지 않아도 결과는 흔들리지 않는다.

통근 보정 (use_commute, PPT 17쪽)
   SKT의 거주/직장/방문 인구 구분은 "서비스인구" 데이터에만 있고, 우리 데이터(유동인구)에는 없다.
   그래서 요일별 유동인구로 직장인 비중을 대신 추정한다.
     출퇴근하는 사람은 평일에만 있고, 쇼핑·방문하는 사람은 주말에도 있다
     -> 평일이 주말보다 많은 만큼이 통근 인구
   정류장마다 고정된 비율을 그냥 곱하면 정류장 계수에 흡수돼서 아무 효과가 없다.
   그래서 출퇴근 시간대에만 작동하는 항목으로 넣고, 그 효과 크기(계수)는 데이터에서 추정한다.
   오전/오후를 따로 두는 이유: 도심 정류장은 아침엔 직장인이 내리는 곳, 저녁엔 타는 곳이라 방향이 반대

한계
   - 못 탄 사람이 다음 버스 대기 줄에 더해지는 건 반영 안 함 -> 실제보다 확률이 높게 나올 수 있음
   - 05번 혼잡 노출 지수는 노선 전체에 숫자 하나라 이 확률 계산에는 넣지 않는다
     (모든 정류장·시간대에 같은 값이라 확률 차이를 만들지 못함)

CLI: python modeling/09BoardingProbability.py --signal flow_x_share
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

SEAT_CAPACITY_DEFAULT = 41
DAYS_DEFAULT = 20
HEADWAY_MIN_DEFAULT = 3
HEADWAY_MAX_DEFAULT = 7
TAU_DEFAULT = 0.6
SIGNAL_DEFAULT = "flow_x_share"
SIGNAL_VARIANTS = ["no_skt", "flow", "flow_x_share", "flow_x_share_hourly"]
COMMUTE_BANDS = ["오전첨두", "오후첨두"]  # 통근 보정이 작동하는 시간대
MAX_EXTRA_WAIT_MIN = 60.0
GRADE_THRESHOLDS = {"여유": 0.9, "보통": 0.7}  # 확률 0.9 이상 여유 / 0.7~0.9 보통 / 0.7 미만 위험

# 시간대 계수를 시간 단위(24개)가 아니라 6개 묶음으로 추정 (데이터가 적어서)
HOUR_BANDS = {
    "심야": [0, 1, 2, 3, 4],
    "새벽": [5, 6],
    "오전첨두": [7, 8, 9],
    "낮": [10, 11, 12, 13, 14, 15, 16],
    "오후첨두": [17, 18, 19],
    "저녁": [20, 21, 22, 23],
}
_HOUR_TO_BAND = {h: band for band, hours in HOUR_BANDS.items() for h in hours}


@dataclass
class ModelConfig:
    seat_capacity: float = SEAT_CAPACITY_DEFAULT
    days: int = DAYS_DEFAULT
    headway_min: float = HEADWAY_MIN_DEFAULT
    headway_max: float = HEADWAY_MAX_DEFAULT
    tau: float = TAU_DEFAULT
    signal: str = SIGNAL_DEFAULT
    use_commute: bool = True  # 통근 보정 (요일별 유동인구 필요)

    @property
    def headway_mid(self) -> float:
        return (self.headway_min + self.headway_max) / 2


# %%
def add_demand_signal(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    """수요신호(signal) 컬럼을 만든다. 유동인구가 없는 칸은 비워둔다(NaN)."""
    if variant not in SIGNAL_VARIANTS:
        raise ValueError(f"signal은 {SIGNAL_VARIANTS} 중 하나")
    df = df.copy()
    df["hour_band"] = df["hour"].map(_HOUR_TO_BAND)
    df["has_skt"] = df["flow_pop"].notna()

    # 통근 비율 = (평일 - 주말) / 평일. 주말이 더 많으면 0
    if {"flow_weekday", "flow_weekend"} <= set(df.columns):
        wd, we = df["flow_weekday"], df["flow_weekend"]
        df["commute_ratio"] = ((wd - we) / wd.where(wd > 0)).clip(0, 1)
    else:
        df["commute_ratio"] = np.nan
    for band in COMMUTE_BANDS:
        df[f"commute_{band}"] = df["commute_ratio"].fillna(0) * (df["hour_band"] == band)

    if variant == "no_skt":
        sig = pd.Series(1.0, index=df.index)
    else:
        sig = df["flow_pop"].astype(float)
        if "_x_share" in variant:
            col = "share_all" if variant.endswith("_hourly") else "share_all_stop"
            if col not in df or df[col].notna().sum() == 0:
                raise ValueError(f"{col}이 전부 결측 — 03bRouteShare.py 산출물을 확인하세요.")
            # 분담률이 빈 칸은 그 정류장의 평균 분담률로, 그래도 없으면 전체 평균으로 채움
            share = df[col].fillna(df.groupby("seq")[col].transform("mean"))
            share = share.fillna(df[col].mean())
            sig = sig * share
        # 계산에 log를 쓰므로 0이 있으면 안 됨 -> 아주 작은 값(중앙값의 1%)으로 바꿔둔다
        floor = max(float(sig[sig > 0].median()) * 0.01, 1e-6) if (sig > 0).any() else 1e-6
        sig = sig.clip(lower=floor)

    df["signal"] = sig.where(df["has_skt"])
    return df


# %%
def _scenarios(cfg: ModelConfig) -> list[tuple[str, float]]:
    # (이름, 배차간격). 배차간격이 짧을수록 버스가 많아서 확률이 높다
    return [("high", cfg.headway_min), ("mid", cfg.headway_mid), ("low", cfg.headway_max)]


def add_capacity_terms(df: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    """배차간격별로 버스 1대당 남은 좌석을 계산한다.

    월간 운행 대수 = 평일 수 x (60 / 배차간격)
    버스 1대당 남은 좌석 = 좌석수 - (이 정류장 도착 시 재차 / 월간 운행 대수)

    단, 입석 금지라 1대에 41명 넘게 못 탄다.
    7분 간격으로 계산했더니 1대에 74명이 타야 하는 시간대가 있었음 -> 7분 가정이 틀렸다는 뜻
    그래서 "실제 탄 인원을 다 태우려면 최소 몇 대가 필요한가"를 먼저 구하고,
    운행 대수가 그보다 적어지지 않게 한다.
      최소 운행 대수 = 그 시간대 노선 최대 재차 / 좌석수
    """
    df = df.copy()
    df["daily_boarding"] = df["boarding"] / cfg.days
    # 최대 재차는 성남 구간까지 포함한 노선 전체 기준이어야 함 (03번에서 미리 계산해 둔 값)
    if "route_max_load_hour" in df.columns:
        route_max = df["route_max_load_hour"]
    else:
        print("⚠ route_max_load_hour 없음 -> 남은 구간 내 최대 재차로 대체 (03번 재실행 권장)")
        route_max = df.groupby("hour")["load_lower"].transform("max")
    min_deps = route_max / cfg.seat_capacity
    for tag, h in _scenarios(cfg):
        deps = np.maximum(cfg.days * 60.0 / h, min_deps)  # 월간 운행 대수
        df[f"headway_eff_{tag}"] = cfg.days * 60.0 / deps  # 보정된 배차간격(분)
        df[f"seats_avail_{tag}"] = (cfg.seat_capacity - df["load_before_board"] / deps).clip(
            0, cfg.seat_capacity
        )
        # 재차율 = 버스 1대당 재차 / 좌석수
        df[f"load_before_ratio_{tag}"] = df["load_before_board"] / deps / cfg.seat_capacity  # 도착할 때 (승차 전)
        df[f"load_after_ratio_{tag}"] = (df["load_before_board"] + df["boarding"]) / deps / cfg.seat_capacity  # 떠날 때
    # 자리가 넉넉해서 타려던 사람이 다 탔던 칸 -> 수요 복원 학습에 사용
    #   1. 버스가 가장 적게 다닌다고(7분) 가정해도, 도착할 때 좌석의 tau(60%) 이하만 차 있었고
    #   2. 승차 후에도 꽉 차지 않았던 칸 (꽉 찼으면 승차 인원이 좌석 수에서 잘린 값)
    # 1번을 '떠날 때' 재차로 판단하면 승차가 많은 칸이 저절로 빠져서 수요가 낮게 학습됨 -> '도착할 때' 기준
    df["unconstrained"] = (df["load_before_ratio_low"] <= cfg.tau) & (df["load_after_ratio_low"] < 1.0)
    return df


# %%
def fit_poisson_glm(
    X: np.ndarray, y: np.ndarray, offset: np.ndarray, ridge: float = 1e-2, max_iter: int = 100
) -> np.ndarray:
    """포아송 회귀 계수를 구한다 (scipy/sklearn 없이 numpy만으로).

    log(승차 인원) = log(수요신호) + 계수들
    데이터가 수백 행뿐이라 계수가 튀지 않게 약하게 눌러준다(ridge).
    """
    n, p = X.shape
    beta = np.zeros(p)
    beta[0] = math.log(max(y.mean(), 1e-6) / max(np.exp(offset).mean(), 1e-12))
    penalty = ridge * np.eye(p)
    penalty[0, 0] = 0.0  # 절편은 누르지 않음
    for _ in range(max_iter):
        eta = np.clip(X @ beta + offset, -30, 30)
        mu = np.exp(eta)
        z = eta - offset + (y - mu) / mu
        XtW = X.T * mu
        new = np.linalg.solve(XtW @ X + penalty, XtW @ z)
        if np.max(np.abs(new - beta)) < 1e-8:
            beta = new
            break
        beta = new
    return beta


@dataclass
class DemandModel:
    signal: str
    stops: list = field(default_factory=list)  # 계수를 따로 갖는 정류장 seq (첫 정류장은 기준이라 제외)
    bands: list = field(default_factory=list)
    use_commute: bool = False
    beta: np.ndarray | None = None
    n_train: int = 0

    def design(self, df: pd.DataFrame) -> np.ndarray:
        cols = [np.ones(len(df))]
        cols += [(df["seq"] == s).to_numpy(float) for s in self.stops]
        cols += [(df["hour_band"] == b).to_numpy(float) for b in self.bands]
        if self.use_commute:
            cols += [df[f"commute_{b}"].to_numpy(float) for b in COMMUTE_BANDS]
        return np.column_stack(cols)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """자리가 충분했다면 탔을 하루 평균 인원을 예측한다."""
        eta = self.design(df) @ self.beta + np.log(df["signal"].to_numpy(float))
        return np.exp(np.clip(eta, -30, 30))

    def coef_table(self) -> pd.DataFrame:
        names = ["(절편)"] + [f"정류장 seq{s}" for s in self.stops] + [f"시간대 {b}" for b in self.bands]
        if self.use_commute:
            names += [f"통근 비율 x {b}" for b in COMMUTE_BANDS]
        return pd.DataFrame({"term": names, "coef": self.beta, "multiplier": np.exp(self.beta)})


def fit_demand_model(train: pd.DataFrame, signal: str, use_commute: bool = False) -> DemandModel:
    train = train.dropna(subset=["signal"])
    # 통근 보정은 SKT를 쓸 때만, 그리고 정류장마다 통근 비율이 달라야 추정 가능
    if use_commute and (signal == "no_skt" or train["commute_ratio"].nunique() < 2):
        use_commute = False
    if train.empty:
        raise ValueError("학습 가능한(unconstrained & 유동인구 있는) 칸이 0개 — tau를 올려보세요.")
    stops = sorted(train["seq"].unique())
    bands = [b for b in HOUR_BANDS if b in set(train["hour_band"])]
    # 첫 정류장/첫 시간대 묶음은 기준으로 두고 나머지만 계수를 구함 (계수가 하나로 정해지게)
    model = DemandModel(signal=signal, stops=stops[1:], bands=bands[1:], use_commute=use_commute, n_train=len(train))
    X = model.design(train)
    model.beta = fit_poisson_glm(X, train["daily_boarding"].to_numpy(float), np.log(train["signal"].to_numpy(float)))
    return model


# %%
def estimate_latent_demand(df: pd.DataFrame, model: DemandModel) -> pd.DataFrame:
    df = df.copy()
    has = df["signal"].notna()
    df["demand_pred"] = np.nan
    df.loc[has, "demand_pred"] = model.predict(df[has])
    # 버스가 꽉 찼던 칸만 예측값으로 바꾼다. 자리가 넉넉했던 칸은 실제 승차 인원이 곧 수요.
    lift = has & ~df["unconstrained"] & (df["demand_pred"] > df["daily_boarding"])
    df["latent_daily"] = df["daily_boarding"].where(~lift, df["demand_pred"])
    df["unmet_daily"] = df["latent_daily"] - df["daily_boarding"]
    df["prob_basis"] = np.where(has, "skt", "boarding_only")
    return df


# %%
def poisson_expected_min(lam: np.ndarray, r: np.ndarray, kmax: int) -> np.ndarray:
    """기다리는 사람 D명(평균 lam인 포아송), 남은 좌석 R개일 때 실제로 타는 사람 수의 평균.

    E[min(D, R)] = P(D > 0) + P(D > 1) + ... + P(D > R-1)
    """
    lam = np.asarray(lam, float)[:, None]
    k = np.arange(kmax + 1)[None, :]
    log_fact = np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, kmax + 1)))])[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        logpmf = np.where(lam > 0, k * np.log(np.where(lam > 0, lam, 1.0)) - lam - log_fact, np.where(k == 0, 0.0, -np.inf))
    surv = 1.0 - np.cumsum(np.exp(logpmf), axis=1)  # P(D > k)
    mask = k < np.asarray(r, int)[:, None]
    return (np.clip(surv, 0, 1) * mask).sum(axis=1)


def add_boarding_probability(df: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    df = df.copy()
    kmax = int(cfg.seat_capacity)
    for tag, _ in _scenarios(cfg):
        h = df[f"headway_eff_{tag}"].to_numpy(float)
        lam = df["latent_daily"].to_numpy(float) * h / 60.0  # 버스 1대 앞 대기 인원
        r = np.floor(df[f"seats_avail_{tag}"].to_numpy(float))
        served = poisson_expected_min(lam, r, kmax)
        p = np.where(lam > 1e-9, served / np.maximum(lam, 1e-9), 1.0)
        df[f"p_board_{tag}"] = np.clip(p, 0, 1)
        wait = h * (1 / np.maximum(df[f"p_board_{tag}"], 1e-3) - 1)
        # 확률이 0에 가까우면 대기시간이 무한대로 커짐 -> 60분에서 자름 (60 = 사실상 탑승 불가)
        df[f"extra_wait_min_{tag}"] = np.minimum(wait, MAX_EXTRA_WAIT_MIN)

    def _grade(p: float) -> str:
        if p >= GRADE_THRESHOLDS["여유"]:
            return "여유"
        if p >= GRADE_THRESHOLDS["보통"]:
            return "보통"
        return "위험"

    df["grade"] = df["p_board_mid"].apply(_grade)
    return df


# %%
def run_model(features: pd.DataFrame, cfg: ModelConfig) -> tuple[pd.DataFrame, DemandModel, dict]:
    """위의 1~3단계를 순서대로 실행. diag는 학습에 쓴 칸 수 등 확인용 숫자."""
    df = add_demand_signal(features, cfg.signal)
    df = add_capacity_terms(df, cfg)
    train = df[df["unconstrained"] & df["signal"].notna()]
    model = fit_demand_model(train, cfg.signal, cfg.use_commute)
    df = estimate_latent_demand(df, model)
    df = add_boarding_probability(df, cfg)
    skt = df[df["has_skt"]]
    diag = {
        "signal": cfg.signal,
        "use_commute": model.use_commute,
        "tau": cfg.tau,
        "n_skt_cells": int(len(skt)),
        "n_train_cells": int(model.n_train),
        "n_constrained_skt_cells": int((~skt["unconstrained"]).sum()),
        "n_lifted_cells": int((skt["unmet_daily"] > 0).sum()),
    }
    return df, model, diag


# %%
def worst_cells(scored: pd.DataFrame, top_n: int = 10, skt_only: bool = True) -> pd.DataFrame:
    d = scored[scored["has_skt"]] if skt_only else scored
    cols = [
        "segment", "stop_label", "seq", "hour", "p_board_low", "p_board_mid", "p_board_high",
        "extra_wait_min_low", "extra_wait_min_high", "grade",
    ]
    return d.nsmallest(top_n, "p_board_mid")[cols]


def main() -> None:
    parser = argparse.ArgumentParser(description="수요 복원 + 탑승가능확률")
    parser.add_argument("--signal", type=str, default=SIGNAL_DEFAULT, choices=SIGNAL_VARIANTS)
    parser.add_argument("--seat-capacity", type=float, default=SEAT_CAPACITY_DEFAULT)
    parser.add_argument("--days", type=int, default=DAYS_DEFAULT)
    parser.add_argument("--headway-min", type=float, default=HEADWAY_MIN_DEFAULT)
    parser.add_argument("--headway-max", type=float, default=HEADWAY_MAX_DEFAULT)
    parser.add_argument("--tau", type=float, default=TAU_DEFAULT)
    parser.add_argument("--no-commute", action="store_true", help="통근 보정 끄기")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    cfg = ModelConfig(args.seat_capacity, args.days, args.headway_min, args.headway_max, args.tau, args.signal,
                      use_commute=not args.no_commute)
    features = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    scored, model, diag = run_model(features, cfg)

    # 안심구역 안에서 확인용 (행 단위 데이터라 반출 X)
    out_path = PROCESSED_DIR / "scores_model.parquet"
    scored.to_parquet(out_path, index=False)

    print(f"진단: {diag}")
    print(f"\n[모델 계수] (multiplier = exp(coef))\n{model.coef_table().round(3).to_string(index=False)}")
    print(f"\n[서울 도심 구간] 탑승가능확률 최하위 {args.top_n}")
    print(worst_cells(scored, args.top_n).round(3).to_string(index=False))
    print(f"\ngrade 분포(SKT 커버 칸):\n{scored.loc[scored['has_skt'], 'grade'].value_counts().to_string()}")
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
