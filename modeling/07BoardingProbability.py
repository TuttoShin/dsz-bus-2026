"""
07BoardingProbability.py

정류장x시간대별로 "기다리는 사람이 이번 버스를 탈 수 있을 확률"을 계산한다.

왜 필요한가)
교통카드에는 버스를 탄 사람만 있다. 버스가 꽉 차서 못 탄 사람은 기록이 없다.
그래서 SKT 유동인구(정류장 주변에 있던 사람 수)로 "원래 타려던 사람 수"를 추정한다.

1. 수요 복원: 원래 타려던 사람 수 추정
   - 버스에 자리가 넉넉했던 정류장/시간대만 골라서 학습 (이때는 승차 인원 = 타려던 사람 수)
       자리가 넉넉 = 버스 1대당 도착 시 재차가 좌석의 tau(60%) 이하
                   + 거의 만석으로 떠난 버스가 5% 미만 (평균은 괜찮아도 몇 대는 꽉 찼을 수 있어서)
   - 학습하는 관계식 (포아송 회귀):
       log(하루 평균 승차 인원) = log(수요신호) + 정류장 계수 + 시간대 계수
                                 + 오전 통근 계수 x 통근 비율 x [7~9시]
                                 + 오후 통근 계수 x 통근 비율 x [17~19시]
       수요신호 = 주변 유동인구 x 9401 분담률 (교통카드로 구한 정류장별 9401 몫)
       통근 비율 = (평일 유동인구 - 주말 유동인구) / 평일 유동인구   (0~1, SKT 요일별)
   - 버스가 꽉 찼던 칸에 이 식을 적용하면 "자리만 있었으면 탔을 인원"이 나온다
   - 못 탄 인원(unmet) = 원래 타려던 인원 - 실제 탄 인원
   - 버스가 안 다닌 시간대는 자리 문제가 아니라서 올리지 않는다

2. 탑승가능확률 (교통카드 버스별 기록, 03번)
   실제로 온 버스 한 대씩:
       기다린 사람 = 분당 수요 x 앞차와의 실제 공백 (평균이 이 값인 포아송 분포)
       남은 좌석   = 41 - 그 버스의 실제 도착 시 재차
       탄 사람     = E[min(기다린 사람, 남은 좌석)]
   탑승가능확률(p_board_mid) = 그 정류장·시간대 모든 버스에서 탄 사람 합 / 기다린 사람 합
   p_board_low / high = 날마다 계산한 확률의 하위 25% / 상위 25% (나쁜 날 / 좋은 날)
   버스가 몰려 오면 긴 공백 뒤 버스는 기다린 사람도 많고 앞에서 타고 온 사람도 많다.
   버스별 실제 (공백, 재차)를 그대로 쓰니까 몰림 효과가 가정 없이 들어간다.
   비교용 uniform=True: 모든 버스가 평균 간격·평균 재차로 고르게 왔다고 치고 계산 (몰림을 무시하면?)

3. 추가 대기시간
   - 매 버스마다 확률 P로 탈 수 있다면, 놓치는 버스 수 평균 = 1/P - 1
   - 추가 대기시간(분) = 실제 평균 배차간격 x (1/P - 1)   (60분 이상은 60으로 표시)

수요신호 종류 (08번에서 서로 비교)
   no_skt              : SKT 안 씀. 정류장 계수 x 시간대 계수만 (비교 기준)
   flow                : 주변 유동인구 그대로
   flow_x_share        : 주변 유동인구 x 9401 분담률(정류장마다 기간 전체로 한 번) (기본값)
   flow_x_share_hourly : 주변 유동인구 x 9401 분담률(시간대별) (비교용)
                         9401이 꽉 찬 시간대엔 못 탄 사람이 승차에 안 잡혀 분담률이 낮게 나옴
                         -> 수요를 복원해야 할 시간대에 신호가 작아지는 순환 문제가 있어서 기본값에서 뺌
   참고: 정류장마다 고정된 분담률은 정류장 계수에 흡수돼서 확률은 flow와 똑같이 나온다.

평일 / 주말 / 요일 (daytype)
   모델은 평일로 학습하고, 주말·요일은 같은 모델에 아래만 바꿔 적용한다.
     - 수요신호: 유동인구 x (그 날 유동인구 / 요일 평균 유동인구)   <- SKT 요일별
     - 통근 보정: 주말엔 출퇴근이 없으니 끔
     - 승하차·재차·버스별 기록: 그 날들의 교통카드
   "평일엔 위험, 주말엔 여유"라면 그 차이는 출퇴근 때문이라고 말할 수 있다.

한계
   - 못 탄 사람이 다음 버스 대기 줄에 더해지는 건 반영 안 함 -> 실제보다 확률이 높게 나올 수 있음
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SEAT_CAPACITY_DEFAULT = 41
TAU_DEFAULT = 0.6
SIGNAL_DEFAULT = "flow_x_share"
SIGNAL_VARIANTS = ["no_skt", "flow", "flow_x_share", "flow_x_share_hourly"]
COMMUTE_BANDS = ["오전첨두", "오후첨두"]  # 통근 보정이 작동하는 시간대
WEEKEND_KEYS = {"weekend", "sat", "sun"}  # 통근 보정을 끄는 날 (sun = 일요일·공휴일)
DOW_KEYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
MAX_EXTRA_WAIT_MIN = 60.0
GRADE_THRESHOLDS = {"여유": 0.9, "보통": 0.7}  # 확률 0.9 이상 여유 / 0.7~0.9 보통 / 0.7 미만 위험
FULL_BUS_SHARE_MAX = 0.05  # 거의 만석으로 떠난 버스가 5% 넘으면 "자리가 넉넉"에서 뺌
DAY_QUANTILES = (0.25, 0.75)  # 날마다 다른 확률의 하위/상위 25% (나쁜 날 / 좋은 날)

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
    tau: float = TAU_DEFAULT
    signal: str = SIGNAL_DEFAULT
    use_commute: bool = True  # 통근 보정 (요일별 유동인구 필요)
    daytype: str = "weekday"  # "weekday" / "weekend" / 요일 "mon"~"sun" / "all"
    uniform: bool = False  # 비교용: 버스가 고르게 왔다고 치고 계산 (몰림 무시)
    bus_events: pd.DataFrame | None = field(default=None, repr=False)  # 03번 버스별 기록


# %%
def add_demand_signal(df: pd.DataFrame, variant: str, daytype: str = "all") -> pd.DataFrame:
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
        on = 0.0 if daytype in WEEKEND_KEYS else 1.0  # 주말엔 출퇴근이 없으니 통근 보정 0
        df[f"commute_{band}"] = df["commute_ratio"].fillna(0) * (df["hour_band"] == band) * on

    # 그 날 유동인구 배율 = 그 날 유동인구 / 요일 평균 유동인구 (정류장별)
    day_factor = pd.Series(1.0, index=df.index)
    if daytype != "all" and f"flow_{daytype}" in df.columns and {"flow_weekday", "flow_weekend"} <= set(df.columns):
        week_avg = (5 * df["flow_weekday"] + 2 * df["flow_weekend"]) / 7
        day_factor = (df[f"flow_{daytype}"] / week_avg.where(week_avg > 0)).fillna(1.0)

    if variant == "no_skt":
        sig = pd.Series(1.0, index=df.index)
    else:
        sig = df["flow_pop"].astype(float) * day_factor
        if "_x_share" in variant:
            col = "share_all" if variant.endswith("_hourly") else "share_all_stop"
            if col not in df or df[col].notna().sum() == 0:
                raise ValueError(f"{col}이 전부 결측 — 02번 정류장별 전체 노선 승차 집계를 확인하세요.")
            share = df[col].fillna(df.groupby("seq")[col].transform("mean")).fillna(df[col].mean())
            sig = sig * share
        # 계산에 log를 쓰므로 0이 있으면 안 됨 -> 아주 작은 값(중앙값의 1%)으로
        floor = max(float(sig[sig > 0].median()) * 0.01, 1e-6) if (sig > 0).any() else 1e-6
        sig = sig.clip(lower=floor)
    df["signal"] = sig.where(df["has_skt"])
    return df


# %%
def add_capacity_terms(df: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    """실제 운행 대수로 버스 1대당 재차율, 학습용 '자리가 넉넉했던 칸' 표시."""
    df = df.copy()
    df["daily_boarding"] = df["boarding"]  # 04번이 이미 하루 평균
    deps = df["n_buses"].where(df["n_buses"] > 0)
    df["seats_avail"] = (cfg.seat_capacity - df["load_before_board"] / deps).clip(0, cfg.seat_capacity)
    df["load_before_ratio"] = df["load_before_board"] / deps / cfg.seat_capacity  # 도착할 때 (승차 전)
    df["load_after_ratio"] = (df["load_before_board"] + df["boarding"]) / deps / cfg.seat_capacity  # 떠날 때
    # '떠날 때' 재차로 판단하면 승차가 많은 칸이 저절로 빠져서 수요가 낮게 학습됨 -> '도착할 때' 기준
    df["unconstrained"] = (
        (df["load_before_ratio"] <= cfg.tau) & (df["full_bus_share"].fillna(0) < FULL_BUS_SHARE_MAX)
        & (df["n_buses"] > 0)
    )
    return df


# %%
def fit_poisson_glm(X: np.ndarray, y: np.ndarray, offset: np.ndarray, ridge: float = 1e-2, max_iter: int = 100) -> np.ndarray:
    """포아송 회귀 계수 (numpy만으로). 데이터가 수백 행뿐이라 계수가 튀지 않게 약하게 눌러준다(ridge)."""
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
    ref_stop: int | None = None  # 기준 정류장 (계수 0)
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
        """자리가 충분했다면 탔을 하루 평균 인원."""
        eta = self.design(df) @ self.beta + np.log(df["signal"].to_numpy(float))
        # 학습 때 못 본 정류장(08번 정류장 제외 검증)은 정류장 효과 평균
        unseen = ~df["seq"].isin(set(self.stops) | {self.ref_stop}).to_numpy()
        if unseen.any():
            stop_coefs = np.concatenate([[0.0], self.beta[1:1 + len(self.stops)]])
            eta = eta + unseen * stop_coefs.mean()
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
        raise ValueError("학습 가능한(자리가 넉넉 & 유동인구 있는) 칸이 0개 — CONFIG tau를 올려보세요.")
    stops = sorted(train["seq"].unique())
    bands = [b for b in HOUR_BANDS if b in set(train["hour_band"])]
    model = DemandModel(signal=signal, stops=stops[1:], bands=bands[1:], use_commute=use_commute,
                        ref_stop=stops[0], n_train=len(train))
    X = model.design(train)
    model.beta = fit_poisson_glm(X, train["daily_boarding"].to_numpy(float), np.log(train["signal"].to_numpy(float)))
    return model


def estimate_latent_demand(df: pd.DataFrame, model: DemandModel) -> pd.DataFrame:
    df = df.copy()
    has = df["signal"].notna()
    df["demand_pred"] = np.nan
    df.loc[has, "demand_pred"] = model.predict(df[has])
    # 버스가 꽉 찼던 칸만 예측값으로 바꾼다. 자리가 넉넉했던 칸은 실제 승차 인원이 곧 수요.
    # 버스가 안 다닌 시간대는 자리 문제가 아니라 운행이 없는 것이라 올리지 않음
    lift = has & ~df["unconstrained"] & (df["demand_pred"] > df["daily_boarding"]) & (df["n_buses"] > 0)
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
        logpmf = np.where(lam > 0, k * np.log(np.where(lam > 0, lam, 1.0)) - lam - log_fact,
                          np.where(k == 0, 0.0, -np.inf))
    surv = 1.0 - np.cumsum(np.exp(logpmf), axis=1)  # P(D > k)
    mask = k < np.asarray(r, int)[:, None]
    return (np.clip(surv, 0, 1) * mask).sum(axis=1)


def _grade(p: float) -> str:
    if not np.isfinite(p):
        return "운행없음"
    if p >= GRADE_THRESHOLDS["여유"]:
        return "여유"
    if p >= GRADE_THRESHOLDS["보통"]:
        return "보통"
    return "위험"


def events_for(ev: pd.DataFrame, daytype: str) -> pd.DataFrame:
    if daytype in ("weekday", "weekend"):
        return ev[ev["daytype"] == daytype]
    if daytype in DOW_KEYS:
        return ev[ev["dow"] == daytype]
    return ev


def add_boarding_probability(df: pd.DataFrame, cfg: ModelConfig) -> pd.DataFrame:
    """버스 한 대씩 계산해서 정류장·시간대로 합친다 (위 2번 설명)."""
    df = df.copy()
    if cfg.bus_events is None:
        raise ValueError("bus_events(03번 버스별 기록)가 없습니다.")
    kmax = int(cfg.seat_capacity)
    ev = events_for(cfg.bus_events, cfg.daytype)
    ev = ev[ev["seq"].isin(df["seq"].unique())][["date", "seq", "hour", "gap_min", "load_before"]].copy()
    key = df.set_index(["seq", "hour"])
    idx = pd.MultiIndex.from_frame(ev[["seq", "hour"]])
    ev["rate"] = (key["latent_daily"] / 60.0).reindex(idx).to_numpy()  # 분당 기다리러 오는 사람
    mean_gap = key["headway_obs"].reindex(idx).to_numpy()
    ev["gap"] = ev["gap_min"].fillna(pd.Series(mean_gap, index=ev.index))  # 첫차처럼 공백을 모르면 평균
    if cfg.uniform:  # 비교용: 모든 버스가 평균 간격·평균 재차
        ev["gap"] = mean_gap
        ev["load_before"] = ev.groupby(["seq", "hour"])["load_before"].transform("mean")
    ev = ev.dropna(subset=["rate", "gap"])
    lam = ev["rate"].to_numpy(float) * ev["gap"].to_numpy(float)
    r = np.floor(np.clip(cfg.seat_capacity - ev["load_before"].to_numpy(float), 0, cfg.seat_capacity))
    ev["served"] = poisson_expected_min(lam, r, kmax)
    ev["waited"] = lam

    tot = ev.groupby(["seq", "hour"])[["served", "waited"]].sum()
    p_mid = tot["served"] / tot["waited"].where(tot["waited"] > 1e-9)
    by_day = ev.groupby(["seq", "hour", "date"])[["served", "waited"]].sum()
    p_day = by_day["served"] / by_day["waited"].where(by_day["waited"] > 1e-9)
    q_lo = p_day.groupby(level=[0, 1]).quantile(DAY_QUANTILES[0])
    q_hi = p_day.groupby(level=[0, 1]).quantile(DAY_QUANTILES[1])
    cell = pd.MultiIndex.from_frame(df[["seq", "hour"]])
    no_bus = (df["n_buses"] <= 0).to_numpy()
    h = df["headway_obs"].to_numpy(float)
    for tag, s in (("mid", p_mid), ("low", q_lo), ("high", q_hi)):
        p = s.reindex(cell).to_numpy(float)
        p = np.where(np.isnan(p) & ~no_bus, 1.0, p)  # 버스는 다녔는데 기다린 사람이 없던 칸
        df[f"p_board_{tag}"] = np.clip(p, 0, 1)
        wait = h * (1 / np.maximum(df[f"p_board_{tag}"], 1e-3) - 1)
        df[f"extra_wait_min_{tag}"] = np.minimum(wait, MAX_EXTRA_WAIT_MIN)  # 60 = 사실상 탑승 불가
    df["grade"] = df["p_board_mid"].apply(_grade)
    return df


# %%
def run_model(features: pd.DataFrame, cfg: ModelConfig, model: DemandModel | None = None) -> tuple[pd.DataFrame, DemandModel, dict]:
    """1~3단계를 순서대로. model을 넘기면 새로 학습하지 않고 그대로 적용 (주말·요일 = 평일 모델)."""
    df = add_demand_signal(features, cfg.signal, cfg.daytype)
    df = add_capacity_terms(df, cfg)
    train = df[df["unconstrained"] & df["signal"].notna()]
    if model is None:
        model = fit_demand_model(train, cfg.signal, cfg.use_commute)
    df = estimate_latent_demand(df, model)
    df = add_boarding_probability(df, cfg)
    skt = df[df["has_skt"]]
    diag = {
        "daytype": cfg.daytype,
        "signal": cfg.signal,
        "use_commute": model.use_commute,  # 모델에 통근 항이 있는가 (평일 학습)
        "commute_applied": bool(model.use_commute and cfg.daytype not in WEEKEND_KEYS),  # 이번 계산에 실제로 적용했는가 (주말 X)
        "tau": cfg.tau,
        "uniform": cfg.uniform,
        "n_skt_cells": int(len(skt)),
        "n_train_cells": int(model.n_train),
        "n_constrained_skt_cells": int((~skt["unconstrained"]).sum()),
        "n_lifted_cells": int((skt["unmet_daily"] > 0).sum()),
    }
    return df, model, diag


def worst_cells(scored: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    d = scored[scored["has_skt"]]
    cols = ["stop_label", "seq", "hour", "p_board_low", "p_board_mid", "p_board_high",
            "extra_wait_min_low", "extra_wait_min_high", "headway_obs", "gap_cv", "grade"]
    return d.nsmallest(top_n, "p_board_mid")[cols]
