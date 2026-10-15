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
                                 + 오전 체류 목적 계수 x 체류 목적 비율 x [7~9시]
                                 + 오후 체류 목적 계수 x 체류 목적 비율 x [17~19시]
       수요신호 = 주변 유동인구 x 9401 분담률 (교통카드로 구한 정류장별 9401 몫)
       체류 목적 비율 = (평일 유동인구 - 주말 유동인구) / 평일 유동인구   (0~1, SKT 요일별. 평일에만 있는 인구 = 출근·업무 목적)
   - 버스가 꽉 찼던 칸에 이 식을 적용하면 "자리만 있었으면 탔을 인원"이 나온다
   - 못 탄 인원(unmet) = 원래 타려던 인원 - 실제 탄 인원
   - 버스가 안 다닌 시간대는 자리 문제가 아니라서 올리지 않는다
   - 교통카드 하한: 자리가 남은 채 떠난 버스 바로 다음 버스(앞차도 자리가 남았던 경우)의 승차는
     "그 사이 온 사람 전부"라서, 그 승차 / 그 사이 시간 = 사람이 오는 속도의 하한.
     꽉 찬 시간대에 SKT 복원값이 이보다 작으면 이 값을 쓴다 (SKT만으로 숨은 수요가 덜 잡히는 것 보완)

2. 줄 서기 계산 (교통카드 버스별 기록, 03번)
   정류장마다 날마다, 실제로 온 버스를 시간 순서대로 놓고
       버스 i의 남은 좌석 = 41 - 그 버스의 실제 도착 시 재차
       사람은 원래 타려던 인원의 속도로 꾸준히 와서 먼저 온 순서대로 줄을 선다
       버스 i는 줄 앞에서부터 남은 좌석만큼 태우고, 못 탄 사람은 그대로 다음 버스를 기다린다
       이탈 규칙은 두 가지로 돌려서 비교한다 (RunAll.py --patience)
         patience_min = None : 이탈 없음. 모두 탈 때까지 기다린다 (기본)
         patience_min = 120  : 120분을 기다려도 못 탄 사람은 줄에서 빠진다 (자리를 차지하지 않음)
   -> 사람마다 "도착한 시각"과 "실제로 탄 버스 시각(또는 이탈)"이 나온다
   대기 시간 = 탄 시각 - 도착 시각 (이탈한 사람은 patience_min, 막차까지 못 탄 사람은 막차 시각까지)
   못 타고 남은 비율(give_up_share) = 이탈한 사람 + 그날 막차까지 못 탄 사람의 비율
   사람이 오는 속도: 시간당 수요를 시간대 한가운데(예: 18:30)의 값으로 두고 이웃 시간대와 직선으로 이어서 1분마다 나눔
     (SKT는 1시간 단위라 시간대 안의 모양은 알 수 없음. 17:59 -> 18:00에 속도가 뚝 바뀌지 않게만 함)
   결과는 시간대별(표·모델용)과 10분 단위(diag["slots"], 서비스용: "18:20에 도착하면")로 모은다
   탑승가능확률(p_board_mid) = 처음 온 버스에 탄 사람 비율 (그 정류장·시간대에 도착한 사람 기준, 한 달 평균)
   p_board_low / high = 날마다 계산한 확률의 하위 25% / 상위 25% (나쁜 날 / 좋은 날)
   추가 대기시간 = 대기 시간 - 다음 버스까지 기다리는 시간 (만차로 보낸 버스 때문에 더 기다린 시간)
   버스별 실제 (도착 시각, 남은 좌석)을 그대로 쓰니까 몰림과 만차 연속이 가정 없이 들어간다.
   비교용 uniform=True: 매 시간 같은 대수가 같은 간격으로, 그 시간 평균 좌석을 갖고 왔다고 치고 계산 (몰림을 무시하면?)

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
     - 체류 목적 보정: 주말엔 출퇴근이 없으니 끔
     - 승하차·재차·버스별 기록: 그 날들의 교통카드
   "평일엔 위험, 주말엔 여유"라면 그 차이는 출퇴근 때문이라고 말할 수 있다.

한계
   - 줄 서기는 사람이 고르게 온다고 본다 (같은 시간대 안에서 몰려 오는 건 반영 안 함)
   - 이탈 없음: 다른 수단으로 가는 사람이 있으면 실제보다 줄이 길게 나올 수 있음
     이탈 120분: 120분은 가정값. 두 결과를 비교해서 장표에 쓸 쪽을 고른다
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
COMMUTE_BANDS = ["오전첨두", "오후첨두"]  # 체류 목적 보정이 작동하는 시간대
WEEKEND_KEYS = {"weekend", "sat", "sun"}  # 체류 목적 보정을 끄는 날 (sun = 일요일·공휴일)
DOW_KEYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
QUEUE_STEP_MIN = 1.0  # 줄 서기 계산에서 사람이 오는 시각 간격(분)
SLOT_MIN = 10  # 서비스용 결과 단위(분). 버스가 4~5분마다 와서 1분 단위는 칸마다 버스가 0~1대라 불안정
FLOOR_MIN_BUSES = 5  # 교통카드 수요 하한을 쓰려면 필요한 "자리 남기고 떠난 연속 버스" 수
GRADE_THRESHOLDS = {"여유": 0.9, "보통": 0.7}  # 확률 0.9 이상 여유 / 0.7~0.9 보통 / 0.7 미만 위험
FULL_BUS_SHARE_MAX = 0.05  # 거의 만석으로 떠난 버스가 5% 넘으면 "자리가 넉넉"에서 뺌
DAY_QUANTILES = (0.25, 0.75)  # 날마다 다른 확률의 하위/상위 25% (나쁜 날 / 좋은 날)
# 사람 한 명 기준 분포 (한 달 동안 그 정류장·시간대에 온 모든 사람): 대기 중앙값·상위 10%, 만석이라 보낸 버스 수
PERSON_COLS = ["wait_p50_min", "wait_p90_min", "missed_buses_mean", "missed_buses_p90"]

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
    use_commute: bool = True  # 체류 목적 보정 (요일별 유동인구 필요)
    daytype: str = "weekday"  # "weekday" / "weekend" / 요일 "mon"~"sun" / "all"
    uniform: bool = False  # 비교용: 버스가 고르게 왔다고 치고 계산 (몰림 무시)
    patience_min: float | None = None  # 줄 서기: 이만큼 기다려도 못 타면 이탈. None = 이탈 없음 (RunAll.py --patience)
    demand_scale: float = 1.0  # 검증용: 복원 수요를 몇 배로 넣을지 (08번 수요 ±20% 민감도)
    bus_events: pd.DataFrame | None = field(default=None, repr=False)  # 03번 버스별 기록


# %%
def add_demand_signal(df: pd.DataFrame, variant: str, daytype: str = "all") -> pd.DataFrame:
    """수요신호(signal) 컬럼을 만든다. 유동인구가 없는 칸은 비워둔다(NaN)."""
    if variant not in SIGNAL_VARIANTS:
        raise ValueError(f"signal은 {SIGNAL_VARIANTS} 중 하나")
    df = df.copy()
    df["hour_band"] = df["hour"].map(_HOUR_TO_BAND)
    df["has_skt"] = df["flow_pop"].notna()

    # 체류 목적 비율 = (평일 - 주말) / 평일. 주말이 더 많으면 0
    if {"flow_weekday", "flow_weekend"} <= set(df.columns):
        wd, we = df["flow_weekday"], df["flow_weekend"]
        df["commute_ratio"] = ((wd - we) / wd.where(wd > 0)).clip(0, 1)
    else:
        df["commute_ratio"] = np.nan
    for band in COMMUTE_BANDS:
        on = 0.0 if daytype in WEEKEND_KEYS else 1.0  # 주말엔 출퇴근이 없으니 체류 목적 보정 0
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
    df["flow_day"] = (df["flow_pop"].astype(float) * day_factor).where(df["has_skt"])  # 그 요일 유동인구 (그림용 비율 계산)
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
            names += [f"체류 목적 비율 x {b}" for b in COMMUTE_BANDS]
        return pd.DataFrame({"term": names, "coef": self.beta, "multiplier": np.exp(self.beta)})


def fit_demand_model(train: pd.DataFrame, signal: str, use_commute: bool = False) -> DemandModel:
    train = train.dropna(subset=["signal"])
    # 체류 목적 보정은 SKT를 쓸 때만, 그리고 정류장마다 체류 목적 비율이 달라야 추정 가능
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


def card_demand_floor(ev: pd.DataFrame, daytype: str, seat_capacity: float) -> pd.Series:
    """교통카드로 본 사람이 오는 속도의 하한 (정류장·시간대, 시간당 인원).

    앞 버스와 이 버스 모두 자리가 남은 채 떠났으면, 이 버스 승차 = 그 사이 정류장에 온 사람 전부.
    이런 버스들만 모아 승차 합 / 공백 합 x 60.
    붐비는 시간에는 자리가 남은 버스가 드물고 그때는 사람이 덜 온 때일 가능성이 커서 실제보다 작게 나옴 -> 하한
    """
    e = events_for(ev, daytype).sort_values(["seq", "date", "arr_min"]).copy()
    e["cleared"] = (e["load_before"] + e["boarding"]) <= seat_capacity - 1
    e["prev_cleared"] = e.groupby(["seq", "date"])["cleared"].shift(1).fillna(False).astype(bool)
    ok = e[e["cleared"] & e["prev_cleared"] & e["gap_min"].notna()]
    g = ok.groupby(["seq", "hour"])
    rate = g["boarding"].sum() / g["gap_min"].sum() * 60
    return rate[g.size() >= FLOOR_MIN_BUSES]


def estimate_latent_demand(df: pd.DataFrame, model: DemandModel, floor: pd.Series | None = None) -> pd.DataFrame:
    df = df.copy()
    has = df["signal"].notna()
    df["demand_pred"] = np.nan
    df.loc[has, "demand_pred"] = model.predict(df[has])
    # 버스가 꽉 찼던 칸만 예측값으로 바꾼다. 자리가 넉넉했던 칸은 실제 승차 인원이 곧 수요.
    # 버스가 안 다닌 시간대는 자리 문제가 아니라 운행이 없는 것이라 올리지 않음
    if floor is not None and len(floor):
        fl = floor.reindex(pd.MultiIndex.from_frame(df[["seq", "hour"]])).to_numpy(float)
        df["card_floor"] = fl
        # 꽉 찼던 칸: SKT 예측과 교통카드 하한 중 큰 값 (SKT가 숨은 수요를 덜 잡는 것 보완)
        df["demand_pred"] = np.where(~df["unconstrained"] & np.isfinite(fl),
                                     np.fmax(df["demand_pred"].to_numpy(float), fl), df["demand_pred"])
    lift = df["demand_pred"].notna() & ~df["unconstrained"] & (df["demand_pred"] > df["daily_boarding"]) & (df["n_buses"] > 0)
    df["latent_daily"] = df["daily_boarding"].where(~lift, df["demand_pred"])
    df["unmet_daily"] = df["latent_daily"] - df["daily_boarding"]
    # 정류장 대기 수요(모든 노선) = 9401 대기 수요 ÷ 9401 점유율. 장표 16쪽 대기 비율·17쪽 단계별 축소 그림용
    if "share_all_stop" in df and df["share_all_stop"].notna().any():
        sh = df["share_all_stop"].fillna(df.groupby("seq")["share_all_stop"].transform("mean")).fillna(df["share_all_stop"].mean())
        df["station_demand_daily"] = df["latent_daily"] / sh.where(sh > 0)
    else:
        df["station_demand_daily"] = np.nan
    df["prob_basis"] = np.where(has, "skt", "boarding_only")
    return df


# %%
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


def _rate_per_min(tau: np.ndarray, rate_by_hour: np.ndarray) -> np.ndarray:
    """분당 오는 사람. 시간대 한가운데(hh:30)를 그 시간 값으로 두고 이웃 시간대와 직선으로 잇는다 (자정 넘어 이어짐)."""
    hod = (tau / 60.0) % 24
    xs = np.concatenate([[-0.5], np.arange(24) + 0.5, [24.5]])
    ys = np.concatenate([[rate_by_hour[23]], rate_by_hour, [rate_by_hour[0]]])
    return np.interp(hod, xs, ys)


def _queue_one_day(t: np.ndarray, seats: np.ndarray, rate_by_hour: np.ndarray, patience: float | None):
    """한 정류장 하루치 줄 서기. t = 버스 도착 시각(분, 오름차순), seats = 남은 좌석, rate_by_hour[24] = 분당 오는 사람.

    1분마다 도착한 사람 묶음을 줄 끝에 세우고, 버스가 오면
      1) (patience가 있으면) patience를 넘게 기다린 묶음은 이탈하고 빠짐 (기다린 시간 = patience)
      2) 줄 앞에서부터 남은 좌석만큼 태움 (먼저 온 순서)
    반환: (시간대별, 10분 단위별, 사람별 조각) 표. 앞의 둘은 도착 인원, 처음 온 버스에 탄 인원, 기다린 시간 합,
          다음 버스까지 시간 합, 못 타고 남은 인원(이탈 + 막차까지 못 탐)
          사람별 조각 = (시간대, 인원, 기다린 분, 보낸 만석 버스 수): 대기 시간 중앙값·상위 10%, 놓치는 버스 수 계산용
    """
    start = np.floor(t[0] / 60) * 60
    tau = np.arange(start + QUEUE_STEP_MIN / 2, t[-1], QUEUE_STEP_MIN)
    if len(tau) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    hour = ((tau // 60) % 24).astype(int)
    slot = ((tau // SLOT_MIN) % (1440 // SLOT_MIN)).astype(int)  # 하루를 10분씩 (0 = 00:00~00:10)
    arrive = _rate_per_min(tau, rate_by_hour) * QUEUE_STEP_MIN  # 묶음마다 사람 수
    left = arrive.copy()  # 아직 못 탄 사람
    first = np.zeros(len(tau))
    wait_sum = np.zeros(len(tau))
    give_up = np.zeros(len(tau))
    j0 = np.searchsorted(t, tau, side="left")  # 묶음마다 다음 버스
    pc_i, pc_w, pc_wait, pc_miss = [], [], [], []  # 사람별 조각: 묶음, 인원, 기다린 분, 보낸 버스 수
    head, k = 0, 0  # 줄 맨 앞 묶음, 아직 줄에 안 선 첫 묶음
    for i in range(len(t)):
        while k < len(tau) and tau[k] < t[i]:
            k += 1
        while patience is not None and head < k and t[i] - tau[head] > patience:  # 이탈
            if left[head] > 0:
                give_up[head] += left[head]
                wait_sum[head] += left[head] * patience
                pc_i.append(head); pc_w.append(left[head]); pc_wait.append(patience); pc_miss.append(i - j0[head])
                left[head] = 0.0
            head += 1
        cap = seats[i]
        while cap > 1e-9 and head < k:  # 먼저 온 순서대로 = 줄 맨 앞부터
            take = min(cap, left[head])
            left[head] -= take
            cap -= take
            wait_sum[head] += take * (t[i] - tau[head])
            if take > 1e-9:
                pc_i.append(head); pc_w.append(take); pc_wait.append(t[i] - tau[head]); pc_miss.append(i - j0[head])
            if j0[head] == i:
                first[head] += take
            if left[head] <= 1e-9:
                left[head] = 0.0
                head += 1
    # 그날 막차까지 못 탄 사람: 막차 시각까지 기다린 것으로 (이탈 규칙이 있으면 patience까지)
    give_up += left
    waited = np.maximum(t[-1] - tau, 0)
    wait_sum += left * (waited if patience is None else np.minimum(patience, waited))
    rest = np.flatnonzero(left > 1e-9)
    pc_i += rest.tolist(); pc_w += left[rest].tolist()
    pc_wait += (waited[rest] if patience is None else np.minimum(patience, waited[rest])).tolist()
    pc_miss += (len(t) - j0[rest]).tolist()
    pc_i = np.asarray(pc_i, int)
    pieces = pd.DataFrame({"hour": hour[pc_i], "w": np.asarray(pc_w, float),
                           "wait": np.asarray(pc_wait, float), "missed": np.asarray(pc_miss, float)})
    nxt = np.where(j0 < len(t), t[np.minimum(j0, len(t) - 1)] - tau, np.nan)
    out = pd.DataFrame({"hour": hour, "slot": slot, "w": arrive, "wf": first, "ww": wait_sum,
                        "wn": arrive * np.nan_to_num(nxt), "wg": give_up, "next": nxt})

    def _agg(key):
        g = out.groupby(key)
        return pd.DataFrame({"w": g["w"].sum(), "wf": g["wf"].sum(), "ww": g["ww"].sum(), "wn": g["wn"].sum(),
                             "wg": g["wg"].sum(), "next_plain": g["next"].mean()})
    return _agg("hour"), _agg("slot"), pieces


def _weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    """인원(w)으로 가중한 분위수. 사람 한 명 한 명의 대기 시간을 줄 세운 것과 같음."""
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    if len(cw) == 0 or cw[-1] <= 0:
        return np.nan
    return float(x[o][min(np.searchsorted(cw, q * cw[-1]), len(cw) - 1)])


def _uniform_buses(e: pd.DataFrame) -> pd.DataFrame:
    """비교용: 매 시간 실제와 같은 대수가 같은 간격으로, 그 시간 평균 좌석을 갖고 온다고 침."""
    rows = []
    for (hk,), g in e.groupby([np.floor(e["arr_min"] / 60)]):
        n = len(g)
        rows.append(pd.DataFrame({"arr_min": hk * 60 + (np.arange(n) + 0.5) * 60 / n,
                                  "seats": np.full(n, g["seats"].mean())}))
    return pd.concat(rows, ignore_index=True) if rows else e[["arr_min", "seats"]]


def _slot_table(rows: list) -> pd.DataFrame:
    """10분 단위 결과 (모든 날 합산). 서비스(09번)·그림용."""
    if not rows:
        return pd.DataFrame(columns=["seq", "slot", "hour", "time", "p_board", "wait_total_min", "wait_next_min", "give_up_share"])
    q = pd.concat(rows, ignore_index=True)
    g = q.groupby(["seq", "slot"])
    tot = g[["w", "wf", "ww", "wn", "wg"]].sum()
    any_w = tot["w"] > 1e-9
    plain = g["next_plain"].mean()
    out = pd.DataFrame({
        "p_board": (tot["wf"] / tot["w"]).where(any_w, 1.0),
        "wait_total_min": (tot["ww"] / tot["w"]).where(any_w, plain),
        "wait_next_min": (tot["wn"] / tot["w"]).where(any_w, plain),
        "give_up_share": (tot["wg"] / tot["w"]).where(any_w, 0.0),
        "n_days": g["date"].nunique(),
    }).reset_index()
    out["hour"] = out["slot"] * SLOT_MIN // 60
    out["time"] = out["slot"].map(lambda k: f"{k * SLOT_MIN // 60:02d}:{k * SLOT_MIN % 60:02d}")
    return out


def add_boarding_probability(df: pd.DataFrame, cfg: ModelConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """정류장마다 날마다 줄 서기를 돌려서 정류장·시간대로 합친다 (위 2번 설명). 반환: (시간대 표, 10분 단위 표)"""
    df = df.copy()
    if cfg.bus_events is None:
        raise ValueError("bus_events(03번 버스별 기록)가 없습니다.")
    ev = events_for(cfg.bus_events, cfg.daytype)
    ev = ev[ev["seq"].isin(df["seq"].unique())][["date", "seq", "hour", "arr_min", "load_before"]].copy()
    ev["seats"] = np.clip(cfg.seat_capacity - ev["load_before"], 0, cfg.seat_capacity)
    lat = df.pivot_table(index="seq", columns="hour", values="latent_daily", aggfunc="first").reindex(columns=range(24))
    rows, slot_rows, piece_rows = [], [], []
    for (q, day), e in ev.sort_values("arr_min").groupby(["seq", "date"]):
        if q not in lat.index or len(e) < 2:
            continue
        if cfg.uniform:
            e = _uniform_buses(e)
        rate = (lat.loc[q].fillna(0).to_numpy(float)) / 60.0 * cfg.demand_scale
        r, rs, pc = _queue_one_day(e["arr_min"].to_numpy(float), np.floor(e["seats"].to_numpy(float)), rate, cfg.patience_min)
        if len(r):
            rows.append(r.assign(seq=q, date=day).reset_index())
            slot_rows.append(rs.assign(seq=q, date=day).reset_index())
            piece_rows.append(pc.assign(seq=q))
    cell = pd.MultiIndex.from_frame(df[["seq", "hour"]])
    no_bus = (df["n_buses"] <= 0).to_numpy()
    if not rows:
        for tag in ("mid", "low", "high"):
            df[f"p_board_{tag}"] = np.where(no_bus, np.nan, 1.0)
            df[f"extra_wait_min_{tag}"] = np.where(no_bus, np.nan, 0.0)
        df["wait_total_min"], df["wait_next_min"] = df["headway_obs"] / 2, df["headway_obs"] / 2
        df["give_up_share"] = np.where(no_bus, np.nan, 0.0)
        for c in PERSON_COLS:
            df[c] = np.nan
        df["grade"] = df["p_board_mid"].apply(_grade)
        return df, _slot_table([])
    q = pd.concat(rows, ignore_index=True)
    has_w = q["w"] > 1e-9
    q["p"] = np.where(has_w, q["wf"] / q["w"].where(has_w), 1.0)
    q["wait"] = np.where(has_w, q["ww"] / q["w"].where(has_w), q["next_plain"])
    q["next"] = np.where(has_w, q["wn"] / q["w"].where(has_w), q["next_plain"])
    g = q.groupby(["seq", "hour"])
    tot = g[["w", "wf", "ww", "wn", "wg"]].sum()
    any_w = tot["w"] > 1e-9
    p_mid = (tot["wf"] / tot["w"]).where(any_w, 1.0)
    wait_mid = (tot["ww"] / tot["w"]).where(any_w, g["next_plain"].mean())
    next_mid = (tot["wn"] / tot["w"]).where(any_w, g["next_plain"].mean())
    stats = {
        ("p", "mid"): p_mid, ("p", "low"): g["p"].quantile(DAY_QUANTILES[0]), ("p", "high"): g["p"].quantile(DAY_QUANTILES[1]),
        ("wait", "mid"): wait_mid, ("wait", "low"): g["wait"].quantile(DAY_QUANTILES[1]),  # 나쁜 날 = 오래 기다린 날
        ("wait", "high"): g["wait"].quantile(DAY_QUANTILES[0]),
    }
    nxt = next_mid.reindex(cell).to_numpy(float)
    for tag in ("mid", "low", "high"):
        p = stats[("p", tag)].reindex(cell).to_numpy(float)
        p = np.where(np.isnan(p) & ~no_bus, 1.0, p)
        df[f"p_board_{tag}"] = np.where(no_bus, np.nan, np.clip(p, 0, 1))
        w = stats[("wait", tag)].reindex(cell).to_numpy(float)
        extra = np.clip(w - nxt, 0, None)
        df[f"extra_wait_min_{tag}"] = np.where(no_bus, np.nan, np.nan_to_num(extra))  # 상한 없음 (줄 서기 결과 그대로)
    df["wait_total_min"] = np.where(no_bus, np.nan, wait_mid.reindex(cell).to_numpy(float))  # 다음 버스 + 만차로 보낸 시간
    df["wait_next_min"] = np.where(no_bus, np.nan, nxt)  # 다음 버스까지 (몰림 포함 실측)
    give = (tot["wg"] / tot["w"]).where(any_w, 0.0).reindex(cell).to_numpy(float)
    df["give_up_share"] = np.where(no_bus, np.nan, np.nan_to_num(give))  # 이탈 + 막차까지 못 탄 비율
    # 사람별 분포 (한 달 모든 날의 사람을 한 줄로): 대기 시간 중앙값·상위 10%, 만석이라 보낸 버스 수
    pcs = pd.concat(piece_rows, ignore_index=True)
    pcs = pcs[pcs["w"] > 1e-9]
    dist = {c: {} for c in PERSON_COLS}
    for key, d in pcs.groupby(["seq", "hour"]):
        x, w, m = d["wait"].to_numpy(), d["w"].to_numpy(), d["missed"].to_numpy()
        dist["wait_p50_min"][key] = _weighted_quantile(x, w, 0.5)
        dist["wait_p90_min"][key] = _weighted_quantile(x, w, 0.9)
        dist["missed_buses_mean"][key] = float(np.average(m, weights=w))
        dist["missed_buses_p90"][key] = _weighted_quantile(m, w, 0.9)
    for c in PERSON_COLS:
        df[c] = np.where(no_bus, np.nan, pd.Series(dist[c], dtype=float).reindex(cell).to_numpy(float))
    df["grade"] = df["p_board_mid"].apply(_grade)
    return df, _slot_table(slot_rows)


# %%
def run_model(features: pd.DataFrame, cfg: ModelConfig, model: DemandModel | None = None) -> tuple[pd.DataFrame, DemandModel, dict]:
    """1~3단계를 순서대로. model을 넘기면 새로 학습하지 않고 그대로 적용 (주말·요일 = 평일 모델)."""
    df = add_demand_signal(features, cfg.signal, cfg.daytype)
    df = add_capacity_terms(df, cfg)
    train = df[df["unconstrained"] & df["signal"].notna()]
    if model is None:
        model = fit_demand_model(train, cfg.signal, cfg.use_commute)
    floor = card_demand_floor(cfg.bus_events, cfg.daytype, cfg.seat_capacity) if cfg.bus_events is not None else None
    df = estimate_latent_demand(df, model, floor)
    df, slots = add_boarding_probability(df, cfg)
    skt = df[df["has_skt"]]
    diag = {
        "daytype": cfg.daytype,
        "signal": cfg.signal,
        "use_commute": model.use_commute,  # 모델에 체류 목적 항이 있는가 (평일 학습)
        "commute_applied": bool(model.use_commute and cfg.daytype not in WEEKEND_KEYS),  # 이번 계산에 실제로 적용했는가 (주말 X)
        "tau": cfg.tau,
        "uniform": cfg.uniform,
        "patience_min": cfg.patience_min if cfg.patience_min is not None else "이탈 없음",
        "n_skt_cells": int(len(skt)),
        "n_train_cells": int(model.n_train),
        "n_constrained_skt_cells": int((~skt["unconstrained"]).sum()),
        "n_lifted_cells": int((skt["unmet_daily"] > 0).sum()),
        "slots": slots,  # 10분 단위 줄 서기 결과 (09번 서비스·그림용, 반출 표에는 안 들어감)
        # 꽉 찼던 칸 중 교통카드 하한이 SKT 예측보다 커서 하한을 쓴 칸 수
        "n_card_floor_used": int(((skt.get("card_floor", np.nan) > skt["demand_pred"] - 1e-9)
                                  & ~skt["unconstrained"] & (skt["unmet_daily"] > 0)).sum()) if "card_floor" in skt else 0,
    }
    return df, model, diag


def worst_cells(scored: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    d = scored[scored["has_skt"]]
    cols = ["stop_label", "seq", "hour", "p_board_low", "p_board_mid", "p_board_high",
            "wait_total_min", "extra_wait_min_mid", "give_up_share", "headway_obs", "gap_cv", "grade"]
    return d.nsmallest(top_n, "p_board_mid")[cols]
