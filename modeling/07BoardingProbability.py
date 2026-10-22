"""
07BoardingProbability.py

정류장 x 시간대별 9401 탑승확률과 기대 대기 시간을 계산한다.

교통카드에는 버스를 탄 사람만 남고, 만석이라 못 탄 사람은 남지 않는다.
그래서 SKT 유동인구로 "원래 타려던 사람 수(대기 수요)"를 복원한 뒤, 실제 버스 기록으로 줄 서기를 재현한다.

1. 수요 복원
   (1) 정류장 대기 수요(모든 노선) 관측값 = 정류장·요일·시간대 전체 노선 승차 (교통카드, 하루 평균)
       대기 비율 = 정류장 대기 수요 / 반경 내 유동인구. 유동인구에는 버스를 기다리지 않는 사람(보행자,
       상점 이용객, 근무 중인 직장인 등)이 섞여 있어 대기 비율이 정류장·요일·시간대마다 다르다.
   (2) 체류 목적 보정 (포아송 회귀)
       log(정류장 대기 수요) = log(유동인구) + 기준 대기 비율 + 정류장 배수 + 요일·시간 배수
         정류장 배수    : 같은 정류장의 다른 요일·시간대에서 학습
         요일·시간 배수 : 같은 요일·시간대의 다른 정류장에서 학습 (7 x 24개)
       9401 자리가 넉넉했던 칸만 학습한다 (도착 시 재차 <= 좌석의 tau, 거의 만석으로 떠난 버스 < 5%).
       만석 도착 칸은 위 식으로 복원한다 (관측값보다 작으면 관측값).
   (3) 9401 대기 수요 = 정류장 대기 수요 x 9401 점유율 (정류장 x 요일 x 시간대, adjusted_share)
         소수 칸     : 한 달 전체 노선 승차 10명 미만 -> 같은 정류장·같은 요일 하루 점유율
         만석 도착 칸: 좌석에 막혀 점유율이 낮게 관측 -> 같은 정류장·시간대의 좌석 여유 요일 평균
       자리가 넉넉했던 칸은 실제 9401 승차가 곧 수요이고, 만석 도착 칸만 복원값으로 올린다.
   - 교통카드 하한: 앞뒤 버스 모두 자리가 남은 채 떠났다면 뒤 버스 승차 = 그 사이 온 사람 전부.
     이 승차 / 간격 = 사람이 오는 속도의 하한. 만석 도착 칸에서 복원값이 이보다 작으면 하한을 쓴다.

2. 줄 서기 (03번 버스별 기록)
   정류장·날짜마다 실제로 온 버스를 순서대로 놓고, 버스마다 남은 좌석(41 - 도착 시 재차)만큼
   먼저 온 사람부터 태운다. 못 탄 사람은 다음 버스를 기다린다.
   사람은 1분 단위로 도착한다 (시간당 수요를 각 시간대 가운데 값으로 두고 이웃 시간대와 직선으로 연결).
   이탈 규칙: patience_min = None (이탈 없음, 기본) 또는 분 단위 값 (그만큼 기다려도 못 타면 줄에서 빠짐)
   결과
     p_board_mid       처음 온 버스에 탄 사람 비율 (p_board_low / high = 날마다 계산한 값의 하위·상위 25%)
     wait_total_min    도착부터 실제로 탄 버스까지 평균 대기 (만석 버스를 보낸 시간 포함)
     wait_p50/p90_min  사람 기준 대기 시간 중앙값 / 상위 10%
     missed_buses_*    만석이라 보낸 버스 수
     give_up_share     이탈하거나 막차까지 못 탄 사람 비율
     grade             기대 대기 10분 이하 여유 / 20분 이하 보통 / 초과 위험
   시간대 단위 표와 10분 단위 표(diag["slots"], 서비스용)를 함께 만든다.
   uniform=True: 버스가 매 시간 같은 간격·평균 좌석으로 왔다고 가정한 비교 계산 (몰림 효과 확인)

수요 모델 변형 (cfg.signal, 08번 교차검증에서 비교)
   no_skt    : 정류장 배수 x 요일·시간 배수만 (SKT 미사용, 비교 기준)
   flow_only : 유동인구 x 기준 대기 비율 (유동인구를 그대로 대기 수요로 보는 경우)
   stay      : 유동인구 x 기준 대기 비율 x 정류장 배수 x 요일·시간 배수 (기본값)

평일 / 주말 / 요일 (cfg.daytype)
   체류 목적 보정은 7개 요일을 한꺼번에 학습하고(cfg.stay_features), 평일·주말은 해당 요일 배수의 평균을 쓴다.
   그 요일 유동인구 = SKT 시간대별 x (그 요일 유동인구 / 7개 요일 평균)

한계
   - 같은 시간대 안에서는 사람이 고르게 온다고 본다
   - 이탈 없음은 다른 수단으로 떠나는 사람을 반영하지 않아 대기가 길게 나올 수 있고, 이탈 시간은 가정값이다
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SEAT_CAPACITY_DEFAULT = 41
TAU_DEFAULT = 0.6
SIGNAL_DEFAULT = "stay"
SIGNAL_VARIANTS = ["no_skt", "flow_only", "stay"]
# 변형별 (유동인구를 쓰나, 정류장 배수, 요일·시간 배수)
VARIANT_TERMS = {"no_skt": (False, True, True), "flow_only": (True, False, False), "stay": (True, True, True)}
DOW_KEYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
DAYTYPE_DOWS = {"weekday": DOW_KEYS[:5], "weekend": DOW_KEYS[5:], "all": DOW_KEYS, **{k: [k] for k in DOW_KEYS}}
SHARE_MIN_USERS = 10  # 점유율 소수 칸: 한 달 전체 노선 승차가 이보다 적으면 같은 요일 하루 점유율로
STAY_RIDGE = 0.1  # 체류 목적 보정 회귀 규제 (학습 칸이 적은 배수가 튀지 않게)
QUEUE_STEP_MIN = 1.0  # 줄 서기 계산에서 사람이 오는 시각 간격(분)
SLOT_MIN = 10  # 서비스용 결과 단위(분)
FLOOR_MIN_BUSES = 5  # 교통카드 수요 하한을 쓰려면 필요한 "자리 남기고 떠난 연속 버스" 수
GRADE_THRESHOLDS = {"여유": 10.0, "보통": 20.0}  # 기대 대기 10분 이하 여유 / 10~20분 보통 / 20분 초과 위험
FULL_BUS_SHARE_MAX = 0.05  # 거의 만석으로 떠난 버스가 5% 넘으면 "자리가 넉넉"에서 뺌
DAY_QUANTILES = (0.25, 0.75)  # 날마다 다른 확률의 하위/상위 25% (나쁜 날 / 좋은 날)
# 사람 한 명 기준 분포 (한 달 동안 그 정류장·시간대에 온 모든 사람): 대기 중앙값·상위 10%, 만석이라 보낸 버스 수
PERSON_COLS = ["wait_p50_min", "wait_p90_min", "missed_buses_mean", "missed_buses_p90"]


@dataclass
class ModelConfig:
    seat_capacity: float = SEAT_CAPACITY_DEFAULT
    tau: float = TAU_DEFAULT
    signal: str = SIGNAL_DEFAULT  # 수요 모델 변형 (SIGNAL_VARIANTS)
    daytype: str = "weekday"  # "weekday" / "weekend" / 요일 "mon"~"sun" / "all"
    uniform: bool = False  # 비교용: 버스가 고르게 왔다고 치고 계산 (몰림 무시)
    patience_min: float | None = None  # 줄 서기: 이만큼 기다려도 못 타면 이탈. None = 이탈 없음 (RunAll.py --patience)
    demand_scale: float = 1.0  # 검증용: 복원 수요를 몇 배로 넣을지 (08번 수요 ±20% 민감도)
    bus_events: pd.DataFrame | None = field(default=None, repr=False)  # 03번 버스별 기록
    stay_features: dict | None = field(default=None, repr=False)  # 체류 목적 보정 학습용 {요일: 06번 features}


# %%
def add_flow_day(df: pd.DataFrame, daytype: str = "all") -> pd.DataFrame:
    """그 요일 유동인구(flow_day) = SKT 시간대별 x (그 요일 유동인구 / 7개 요일 평균). 유동인구가 없는 칸은 NaN."""
    df = df.copy()
    df["has_skt"] = df["flow_pop"].notna()
    day_factor = pd.Series(1.0, index=df.index)
    if daytype != "all" and f"flow_{daytype}" in df.columns and {"flow_weekday", "flow_weekend"} <= set(df.columns):
        week_avg = (5 * df["flow_weekday"] + 2 * df["flow_weekend"]) / 7
        day_factor = (df[f"flow_{daytype}"] / week_avg.where(week_avg > 0)).fillna(1.0)
    df["flow_day"] = (df["flow_pop"].astype(float) * day_factor).where(df["has_skt"])
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
    """포아송 회귀 계수 (numpy만으로). 칸이 적은 계수가 튀지 않게 약하게 눌러준다(ridge)."""
    n, p = X.shape
    beta = np.zeros(p)
    beta[0] = math.log(max(y.mean(), 1e-6) / max(np.exp(offset).mean(), 1e-12))
    penalty = ridge * np.eye(p)
    penalty[0, 0] = 0.0  # 절편(기준 대기 비율)은 누르지 않음
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


def stay_cells(stay_features: dict, cfg: ModelConfig) -> pd.DataFrame:
    """체류 목적 보정 학습·예측용 칸 (정류장 x 요일 x 시간대). station_board = 정류장 전체 노선 승차(하루 평균)."""
    rows = []
    for k, f in stay_features.items():
        d = add_capacity_terms(add_flow_day(f, k), cfg)
        rows.append(d.assign(dow=k))
    c = pd.concat(rows, ignore_index=True)
    c = c[c["has_skt"] & (c["n_buses"] > 0) & c["station_board"].notna()].copy()
    c["dh"] = c["dow"] + "_" + c["hour"].astype(int).astype(str)
    return c


def adjusted_share(cells: pd.DataFrame) -> pd.DataFrame:
    """9401 점유율 (정류장 x 요일 x 시간대) + 예외 처리. cells = stay_cells() (share_dh, n_total 필요).

    반환: seq, dow, hour, share_adj, share_rule("관측" / "소수 칸" / "만석 도착 칸")
    """
    c = cells.copy()
    n = c["n_total"].fillna(0)
    mine = c["share_dh"].fillna(0) * n
    key = [c["seq"], c["dow"]]
    day = mine.groupby(key).transform("sum") / n.groupby(key).transform("sum").where(lambda v: v > 0)
    few = n < SHARE_MIN_USERS
    s = c["share_dh"].where(~few, day)
    rule = pd.Series(np.where(few, "소수 칸", "관측"), index=c.index)
    # 만석 도착 칸: 같은 정류장·같은 시간대에서 좌석이 여유 있던 요일들의 점유율 평균
    free = s.where(c["unconstrained"] & ~few).groupby([c["seq"], c["hour"]]).transform("mean")
    full = ~c["unconstrained"] & free.notna()
    s = s.where(~full, free)
    rule[full] = "만석 도착 칸"
    return c.assign(share_adj=s, share_rule=rule)[["seq", "dow", "hour", "share_adj", "share_rule"]]


@dataclass
class StayModel:
    """정류장 대기 수요 = 유동인구 x 기준 대기 비율 x 정류장 배수 x 요일·시간 배수 (변형에 따라 일부만)."""
    variant: str
    base: float = 0.0  # log(기준 대기 비율)
    stop_eff: dict = field(default_factory=dict)  # seq -> log(정류장 배수). 학습에 없던 정류장 = 0 (평균)
    dh_eff: dict = field(default_factory=dict)  # (요일, 시간) -> log(요일·시간 배수)
    n_train: int = 0
    n_cells: int = 0
    n_full: int = 0

    def _dh(self, dow: str, hour: int) -> float:
        if not self.dh_eff:
            return 0.0
        if (dow, hour) in self.dh_eff:
            return self.dh_eff[(dow, hour)]
        same_hour = [v for (d, h), v in self.dh_eff.items() if h == hour]  # 학습에 없던 요일·시간: 같은 시간 다른 요일 평균
        return float(np.mean(same_hour)) if same_hour else 0.0

    def ratio(self, seq: pd.Series, hour: pd.Series, dows: list) -> np.ndarray:
        """대기 비율 (no_skt면 하루 평균 대기 수요 자체). 여러 요일이면 요일·시간 배수를 평균."""
        st = seq.map(self.stop_eff).fillna(0.0).to_numpy(float)
        mult = np.array([np.mean([math.exp(self._dh(d, int(h))) for d in dows]) for h in hour])
        return np.exp(self.base + st) * mult

    def predict_station(self, df: pd.DataFrame, dows: list) -> np.ndarray:
        """정류장 대기 수요 (하루 평균, 모든 노선)."""
        r = self.ratio(df["seq"], df["hour"], dows)
        return r * df["flow_day"].to_numpy(float) if VARIANT_TERMS[self.variant][0] else r

    def predict_cells(self, cells: pd.DataFrame) -> np.ndarray:
        """칸마다 자기 요일로 (교차검증용)."""
        out = np.empty(len(cells))
        for d, idx in cells.groupby("dow").indices.items():
            out[idx] = self.predict_station(cells.iloc[idx], [d])
        return out

    def coef_table(self) -> pd.DataFrame:
        """기준 대기 비율과 정류장 배수 (결과표 05)."""
        rows = [("(기준 대기 비율)", self.base)] + [(f"정류장 배수 seq{q}", v) for q, v in sorted(self.stop_eff.items())]
        return pd.DataFrame(rows, columns=["term", "coef"]).assign(multiplier=lambda d: np.exp(d["coef"]))

    def dow_hour_table(self) -> pd.DataFrame:
        """요일 x 시간대 배수 (결과표 05b, 7행 x 24열)."""
        return pd.DataFrame({h: [math.exp(self._dh(d, h)) for d in DOW_KEYS] for h in range(24)}, index=DOW_KEYS)


def fit_stay_model(cells: pd.DataFrame, variant: str = SIGNAL_DEFAULT) -> StayModel:
    """체류 목적 보정 학습. cells = stay_cells(). 9401 자리가 넉넉했던 칸만 학습."""
    if variant not in VARIANT_TERMS:
        raise ValueError(f"signal은 {SIGNAL_VARIANTS} 중 하나")
    use_flow, use_stop, use_dh = VARIANT_TERMS[variant]
    train = cells[cells["unconstrained"] & (cells["flow_day"] > 0)]
    if train.empty:
        raise ValueError("학습 가능한(자리가 넉넉 & 유동인구 있는) 칸이 0개 — CONFIG tau를 올려보세요.")
    stops = sorted(train["seq"].unique()) if use_stop else []
    dhs = sorted(train["dh"].unique()) if use_dh else []
    # 기준 수준 없이 모든 정류장·요일시간을 넣고 규제를 걸어 배수가 평균 1 근처에 모이게 함
    cols = [np.ones(len(train))]
    cols += [(train["seq"] == q).to_numpy(float) for q in stops]
    cols += [(train["dh"] == d).to_numpy(float) for d in dhs]
    offset = np.log(train["flow_day"].to_numpy(float)) if use_flow else np.zeros(len(train))
    beta = fit_poisson_glm(np.column_stack(cols), train["station_board"].to_numpy(float), offset, ridge=STAY_RIDGE)
    dh_key = {d: (d.split("_")[0], int(d.split("_")[1])) for d in dhs}
    return StayModel(
        variant=variant, base=float(beta[0]),
        stop_eff={int(q): float(b) for q, b in zip(stops, beta[1:1 + len(stops)])},
        dh_eff={dh_key[d]: float(b) for d, b in zip(dhs, beta[1 + len(stops):])},
        n_train=len(train), n_cells=len(cells), n_full=int((~cells["unconstrained"]).sum()),
    )


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


def estimate_latent_demand(df: pd.DataFrame, model: StayModel, floor: pd.Series | None = None,
                           daytype: str = "weekday", shares: pd.DataFrame | None = None) -> pd.DataFrame:
    """정류장 대기 수요 -> 9401 대기 수요. 자리가 넉넉했던 칸은 관측값 그대로, 만석 도착 칸만 복원값으로 올린다.
    shares = adjusted_share() (정류장 x 요일 x 시간대). 없는 칸은 정류장 한 달 점유율."""
    df = df.copy()
    has = df["has_skt"]
    if "share_all_stop" not in df or df["share_all_stop"].notna().sum() == 0:
        raise ValueError("9401 점유율(share_all_stop)이 전부 결측 — 02번 정류장별 전체 노선 승차 집계를 확인하세요.")
    stop_share = df["share_all_stop"].fillna(df.groupby("seq")["share_all_stop"].transform("mean")).fillna(df["share_all_stop"].mean())
    df["share_adj"] = np.nan
    if shares is not None and len(shares):
        sm = shares[shares["dow"].isin(DAYTYPE_DOWS.get(daytype, DOW_KEYS))].groupby(["seq", "hour"])["share_adj"].mean()
        df["share_adj"] = sm.reindex(pd.MultiIndex.from_frame(df[["seq", "hour"]])).to_numpy(float)
    share = df["share_adj"].fillna(stop_share)
    obs_station = df["station_board"] if "station_board" in df else pd.Series(np.nan, index=df.index)
    df["station_pred"] = np.nan
    df.loc[has, "station_pred"] = model.predict_station(df[has], DAYTYPE_DOWS.get(daytype, DOW_KEYS))
    # 정류장 대기 수요: 자리 넉넉 칸 = 관측(전체 노선 승차), 만석 도착 칸 = 복원값(관측보다 작으면 관측)
    restored = np.fmax(df["station_pred"].to_numpy(float), obs_station.to_numpy(float))
    df["station_demand_daily"] = np.where(df["unconstrained"] | ~has, obs_station, restored)
    df["station_demand_daily"] = df["station_demand_daily"].fillna(df["station_pred"])
    df["demand_pred"] = df["station_pred"] * share  # 9401 몫 (학습·검증 비교용 예측값)
    if floor is not None and len(floor):
        fl = floor.reindex(pd.MultiIndex.from_frame(df[["seq", "hour"]])).to_numpy(float)
        df["card_floor"] = fl
    else:
        fl = np.full(len(df), np.nan)
    # 만석 도착 칸의 9401 대기 수요 = 복원한 정류장 대기 수요 x 9401 점유율 (교통카드 하한과 비교해 큰 값)
    lifted = np.fmax((df["station_demand_daily"] * share).to_numpy(float), fl)
    lift = has & ~df["unconstrained"] & (df["n_buses"] > 0) & (lifted > df["daily_boarding"].to_numpy(float))
    df["latent_daily"] = df["daily_boarding"].where(~lift, lifted)
    df["unmet_daily"] = df["latent_daily"] - df["daily_boarding"]
    df["prob_basis"] = np.where(has, "skt", "boarding_only")
    return df


# %%
def _grade(wait: float) -> str:
    """등급 = 기대 대기 시간(정류장 도착부터 실제로 탄 버스까지, 분) 기준."""
    if not np.isfinite(wait):
        return "운행없음"
    if wait <= GRADE_THRESHOLDS["여유"]:
        return "여유"
    if wait <= GRADE_THRESHOLDS["보통"]:
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
        "arrivals_daily": tot["w"] / g["date"].nunique(),  # 그 10분에 오는 사람 (하루 평균)
    }).reset_index()
    out["hour"] = out["slot"] * SLOT_MIN // 60
    out["time"] = out["slot"].map(lambda k: f"{k * SLOT_MIN // 60:02d}:{k * SLOT_MIN % 60:02d}")
    return out


def add_boarding_probability(df: pd.DataFrame, cfg: ModelConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """정류장·날짜마다 줄 서기를 돌려 정류장·시간대로 합친다. 반환: (시간대 표, 10분 단위 표)"""
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
        df["grade"] = df["wait_total_min"].apply(_grade)
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
    df["grade"] = df["wait_total_min"].apply(_grade)
    return df, _slot_table(slot_rows)


# %%
def run_model(features: pd.DataFrame, cfg: ModelConfig, model: StayModel | None = None) -> tuple[pd.DataFrame, StayModel, dict]:
    """수요 복원 -> 줄 서기. model을 넘기면 새로 학습하지 않고 그대로 적용한다."""
    df = add_capacity_terms(add_flow_day(features, cfg.daytype), cfg)
    cells = stay_cells(cfg.stay_features, cfg) if cfg.stay_features else None
    if model is None:
        if cells is None:
            raise ValueError("체류 목적 보정 학습용 요일별 features(cfg.stay_features)가 없습니다.")
        model = fit_stay_model(cells, cfg.signal)
    shares = adjusted_share(cells) if cells is not None and "share_dh" in cells else None
    floor = card_demand_floor(cfg.bus_events, cfg.daytype, cfg.seat_capacity) if cfg.bus_events is not None else None
    df = estimate_latent_demand(df, model, floor, cfg.daytype, shares)
    df, slots = add_boarding_probability(df, cfg)
    skt = df[df["has_skt"]]
    diag = {
        "daytype": cfg.daytype,
        "signal": cfg.signal,
        "n_stay_cells": int(model.n_cells),  # 체류 목적 보정 칸 (정류장 x 요일 x 시간대, 운행·유동인구 있는 칸)
        "n_stay_full_cells": int(model.n_full),  # 그중 만석 도착 칸 (학습 제외, 복원 대상)
        # 9401 점유율 예외 처리한 칸 수 (정류장 x 요일 x 시간대)
        "n_share_cells": int(len(shares)) if shares is not None else 0,
        "n_share_few": int((shares["share_rule"] == "소수 칸").sum()) if shares is not None else 0,
        "n_share_full": int((shares["share_rule"] == "만석 도착 칸").sum()) if shares is not None else 0,
        "tau": cfg.tau,
        "uniform": cfg.uniform,
        "patience_min": cfg.patience_min if cfg.patience_min is not None else "이탈 없음",
        "n_skt_cells": int(len(skt)),
        "n_train_cells": int(model.n_train),
        "n_constrained_skt_cells": int((~skt["unconstrained"]).sum()),
        "n_lifted_cells": int((skt["unmet_daily"] > 0).sum()),
        "slots": slots,  # 10분 단위 줄 서기 결과 (09번 서비스·그림용)
        # 만석 칸 중 교통카드 하한이 복원값보다 커서 하한을 쓴 칸 수
        "n_card_floor_used": int(((skt.get("card_floor", np.nan) >= skt["latent_daily"] - 1e-9)
                                  & ~skt["unconstrained"] & (skt["unmet_daily"] > 0)).sum()) if "card_floor" in skt else 0,
    }
    return df, model, diag


def worst_cells(scored: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    d = scored[scored["has_skt"]]
    cols = ["stop_label", "seq", "hour", "p_board_low", "p_board_mid", "p_board_high",
            "wait_total_min", "extra_wait_min_mid", "give_up_share", "headway_obs", "gap_cv", "grade"]
    return d.nlargest(top_n, "wait_total_min")[cols]
