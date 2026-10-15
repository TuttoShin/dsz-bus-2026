"""
10ExportResults.py

안심구역에서 가지고 나갈 결과물만 export/ 폴더에 모은다.
1차 반출 때 features.parquet, scores.parquet이 "행 단위 데이터"라서 반려됨
-> 요약표, 통계량, 모델 성능, 그래프만 만든다.

반출 규칙 (코드에서 자동으로 검사)
1. 유동인구, 승하차 인원, 카드·차량·정류장ID, 좌표 같은 원본 값은 표에 넣지 않는다 (FORBIDDEN_COLUMNS)
2. 반올림: 확률은 0.05 단위, 대기시간은 1분 단위, 인원은 10명 단위
3. 표 하나당 최대 100행 (정류장x시간대 원본 테이블이 실수로 섞이는 것 방지)
4. 반출 신청서에 붙일 파일 목록(ExportManifest.md)도 자동으로 만든다

RunAll.py 마지막 단계에서 호출됨 (단독 실행 X)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# 이 컬럼이 표에 있으면 저장하지 않고 멈춤
FORBIDDEN_COLUMNS = {
    "flow_pop", "flow_weekday", "flow_weekend",
    "flow_mon", "flow_tue", "flow_wed", "flow_thu", "flow_fri", "flow_sat", "flow_sun", "signal", "n_cells_matched",
    "boarding", "alighting", "net", "daily_boarding", "load_lower", "load_before_board", "load_before",
    "route_max_load_hour", "per_bus_load", "latent_daily", "demand_pred", "unmet_daily", "station_demand_daily", "flow_day",
    "station_board", "station_pred", "n_total",
    "sttn_id", "ars", "lat", "lon", "utmk_x", "utmk_y", "X_COORD", "Y_COORD",
    "car", "trip_key", "gin_dt", "gff_dt", "gin_stop", "gff_stop", "users", "gap_min",
}
MAX_ROWS = 100
PIVOT_HOURS = list(range(24))  # 표(CSV)는 24시간 전체
COMMUTE_HOURS = list(range(6, 10)) + list(range(16, 24))  # 그래프는 출근 6~9시, 퇴근 16~23시만
DAYTYPE_KR = {"weekday": "평일", "weekend": "주말"}
DOW_KR = {"mon": "월", "tue": "화", "wed": "수", "thu": "목", "fri": "금", "sat": "토", "sun": "일·공휴일"}


# %%
def _round_to(x, step: float):
    return (np.round(np.asarray(x, float) / step) * step).round(6)


def _check_exportable(name: str, df: pd.DataFrame) -> None:
    """반출 규칙 1, 3번 검사. 걸리면 에러를 내고 멈춘다."""
    cols = set(map(str, df.columns)) | set(map(str, getattr(df.index, "names", []) or []))
    bad = cols & FORBIDDEN_COLUMNS
    if bad:
        raise ValueError(f"[반출 차단] {name}에 원자료 컬럼 포함: {bad}")
    if len(df) > MAX_ROWS:
        raise ValueError(f"[반출 차단] {name} 행 수 {len(df)} > {MAX_ROWS} (행 단위 데이터로 보일 수 있음)")


def _save_table(out_dir: Path, name: str, df: pd.DataFrame, manifest: list, desc: str, index: bool = False) -> None:
    _check_exportable(name, df)
    df.to_csv(out_dir / "tables" / f"{name}.csv", index=index, encoding="utf-8-sig")
    manifest.append(("표", f"tables/{name}.csv", desc, f"{len(df)}행"))


# %%
def _stop_labels(scored: pd.DataFrame) -> tuple[pd.DataFrame, list, dict]:
    order = scored.drop_duplicates("seq").sort_values("seq")[["seq", "stop_label", "segment"]]
    labels = [f"{row.seq:02d} {row.stop_label}" for row in order.itertuples()]
    return order, labels, dict(zip(order["seq"], labels))


def _daytype_tables(scored: pd.DataFrame, daytype: str, suffix: str) -> dict:
    """평일 또는 주말 하나에 대한 확률표 / 등급표 / 위험 top10."""
    kr = DAYTYPE_KR[daytype]
    _, labels, label_map = _stop_labels(scored)
    src = scored[scored["hour"].isin(PIVOT_HOURS)].copy()
    src["정류장"] = src["seq"].map(label_map)
    p_pivot = src.pivot_table(index="정류장", columns="hour", values="p_board_mid", aggfunc="first", dropna=False)
    p_pivot = p_pivot.reindex(labels).apply(lambda c: _round_to(c, 0.05))
    g_pivot = src.pivot_table(index="정류장", columns="hour", values="grade", aggfunc="first").reindex(labels)

    worst = scored[scored["has_skt"]].nsmallest(10, "p_board_mid")
    worst_tbl = pd.DataFrame({
        "정류장": worst["seq"].map(label_map).to_numpy(),
        "시간대": worst["hour"].to_numpy(),
        "탑승확률_나쁜날(하위25%)": _round_to(worst["p_board_low"], 0.05),
        "탑승확률": _round_to(worst["p_board_mid"], 0.05),
        "탑승확률_좋은날(상위25%)": _round_to(worst["p_board_high"], 0.05),
        "기대대기_분(다음버스+만차)": _round_to(worst["wait_total_min"], 1) if "wait_total_min" in worst else np.nan,
        "추가대기_분": _round_to(worst["extra_wait_min_mid"], 1),
        "못타고_남은_비율(이탈·막차놓침)": _round_to(worst["give_up_share"], 0.05) if "give_up_share" in worst else np.nan,
        "추가대기_분_나쁜날": _round_to(worst["extra_wait_min_low"], 1),
        "실제평균배차_분": _round_to(worst["headway_obs"], 0.1),
        "몰림정도_CV": _round_to(worst["gap_cv"], 0.01),
        "등급": worst["grade"].to_numpy(),
    })
    return {
        f"01{suffix}BoardingProbabilityPivot{daytype.capitalize()}": (
            p_pivot, f"[{kr}] 정류장x시간대 탑승가능확률(교통카드 버스별 실측, 5%p 반올림, 빈칸 = 운행 없음). 서비스 입력값", True),
        f"02{suffix}GradePivot{daytype.capitalize()}": (g_pivot, f"[{kr}] 정류장x시간대 등급(여유/보통/위험/운행없음)", True),
        f"03{suffix}WorstCellsTop10{daytype.capitalize()}": (
            worst_tbl, f"[{kr}] 탑승가능확률 최하위 10개 정류장·시간대(날마다 차이, 추가 대기, 실제 배차·몰림)", False),
    }


def _unmet_by_hour(scored: pd.DataFrame, kr: str, seat_capacity: float) -> pd.DataFrame:
    skt = scored[scored["has_skt"]]
    by_hour = skt.groupby("hour").agg(latent=("latent_daily", "sum"), unmet=("unmet_daily", "sum"))
    return pd.DataFrame({
        f"{kr}_추정_미승차_일평균_명(10명단위)": _round_to(by_hour["unmet"], 10),
        f"{kr}_미승차_비율_%": _round_to(100 * by_hour["unmet"] / by_hour["latent"].where(by_hour["latent"] > 0), 1),
        # 미승차 인원을 모두 태우려면 필요한 추가 운행 대수 (좌석수 기준, 증차 비용 환산용)
        f"{kr}_필요_추가운행_대수": _round_to(by_hour["unmet"] / seat_capacity, 0.1),
    }, index=by_hour.index)


def _daytype_summary(scored_by_day: dict) -> pd.DataFrame:
    """시간대별 평일 vs 주말 요약 (서울 도심 구간)."""
    cols = {}
    for daytype, scored in scored_by_day.items():
        kr = DAYTYPE_KR[daytype]
        g = scored[scored["has_skt"]].groupby("hour")
        cols[f"{kr}_평균확률"] = _round_to(g["p_board_mid"].mean(), 0.05)
        cols[f"{kr}_최저확률"] = _round_to(g["p_board_mid"].min(), 0.05)
        cols[f"{kr}_실제평균배차_분"] = _round_to(g["headway_obs"].mean(), 0.1)
        cols[f"{kr}_위험칸수"] = g["grade"].apply(lambda x: int((x == "위험").sum()))
    out = pd.DataFrame(cols)
    out.index.name = "시간대"
    return out.reset_index()


def _dow_tables(scored_by_dow: dict) -> dict:
    """요일별 정류장x시간대 확률(한 표에 요일 열로) + 시간대별 요일 요약."""
    long_rows, summary = [], {}
    for k, sc in scored_by_dow.items():
        _, _, label_map = _stop_labels(sc)
        piv = sc.pivot_table(index="seq", columns="hour", values="p_board_mid", aggfunc="first", dropna=False)
        piv = piv.apply(lambda c: _round_to(c, 0.05))
        piv.insert(0, "정류장", piv.index.map(label_map))
        piv.insert(1, "요일", DOW_KR[k])
        long_rows.append(piv.reset_index(drop=True))
        g = sc[sc["has_skt"]].groupby("hour")
        summary[f"{DOW_KR[k]}_최저확률"] = _round_to(g["p_board_mid"].min(), 0.05)
        summary[f"{DOW_KR[k]}_위험칸수"] = g["grade"].apply(lambda x: int((x == "위험").sum()))
    summ = pd.DataFrame(summary)
    summ.index.name = "시간대"
    long = pd.concat(long_rows, ignore_index=True)
    out = {}
    if len(long) <= MAX_ROWS:
        out["14BoardingProbabilityByDow"] = (long, "요일별(월~일) 정류장x시간대 탑승가능확률(교통카드 실측, 5%p 반올림)", False)
    else:  # 서울 구간 정류장이 많으면 100행을 넘어서 평일 / 주말 두 파일로
        wk = long["요일"].isin(["토", "일·공휴일"])
        out["14BoardingProbabilityByDow"] = (long[~wk], "요일별(월~금) 정류장x시간대 탑승가능확률(교통카드 실측, 5%p 반올림)", False)
        out["14bBoardingProbabilityByDowWeekend"] = (long[wk], "요일별(토, 일·공휴일) 정류장x시간대 탑승가능확률(교통카드 실측, 5%p 반올림)", False)
    out["15DowSummaryByHour"] = (summ.reset_index(), "시간대별 요일별 최저 탑승확률과 위험 칸 수", False)
    return out


def _service_tables(r: dict, m09, label_map: dict) -> dict:
    """09번 서비스 결과: 기대 대기 시간표, 상류 정류장 추천, 대기 시간 부담."""
    # 10분 단위 줄 서기 결과 (07번). 열 = 도착 시각 "18:20" (18:20~18:30에 도착한 사람)
    sl = r["slots"]["weekday"]
    src = sl[sl["hour"].isin(COMMUTE_HOURS) & sl["seq"].isin(label_map)].copy()
    src["정류장"] = src["seq"].map(label_map)
    order_t = src.drop_duplicates("slot").sort_values("slot", key=lambda k: (k * 10 // 60 < 4) * 1000 + k)["time"].tolist()
    wait_pivot = src.pivot_table(index="정류장", columns="time", values="wait_total_min", aggfunc="first")
    wait_pivot = wait_pivot.reindex(index=[label_map[s] for s in sorted(src["seq"].unique())], columns=order_t)
    wait_pivot = wait_pivot.apply(lambda c: _round_to(c, 1))

    rows = []
    for key, kr in (("weekday", "평일"), ("weekend", "주말")):
        rk = r["service_rec_by_day"].get(key)
        if rk is not None and len(rk):
            rows.append(rk[rk["saving_min"] > 0].assign(day=kr))
    rec = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=list(r["service_rec"].columns) + ["day"])
    rec = rec.sort_values("saving_min", ascending=False).head(MAX_ROWS)
    rec_tbl = pd.DataFrame({
        "구분": rec["day"].to_numpy(),
        "지금정류장": rec["seq"].map(label_map).to_numpy(),
        "도착시각(10분)": rec["time"].to_numpy(),
        "그대로_기다리면_서울빠져나가기까지_분": _round_to(rec["total_here_min"], 1),
        "추천정류장": rec["alt_seq"].astype(float).map(lambda v: label_map.get(int(v), "")).to_numpy(),
        "이동수단": rec["alt_mode"].to_numpy(),
        "이동_분": _round_to(rec["alt_access_min"], 1),
        "추천정류장_9401대기_분": _round_to(rec["alt_wait_min"], 1),
        "추가_9401차내_분": _round_to(rec["alt_extra_ride_min"], 1),
        "추천경로_총_분": _round_to(rec["alt_total_min"], 1),
        "단축_분": _round_to(rec["saving_min"], 1),
    })
    pt = r["service_points"]
    pt = pt[pt["saving_min"] > 0].sort_values("saving_min", ascending=False).head(MAX_ROWS)
    pt_tbl = pd.DataFrame({
        "구분": pt["day"].to_numpy(),
        "지금위치": pt["name"].to_numpy(),
        "도착시각(10분)": pt["time"].to_numpy(),
        "가장가까운정류장": pt["near_seq"].map(label_map).to_numpy(),
        "가까운곳_총_분(도보+대기+차내)": _round_to(pt["near_total_min"], 1),
        "추천정류장": pt["best_seq"].map(label_map).to_numpy(),
        "추천_도보_분": _round_to(pt["best_walk_min"], 1),
        "추천_9401대기_분": _round_to(pt["best_wait_min"], 1),
        "추천_총_분": _round_to(pt["best_total_min"], 1),
        "단축_분": _round_to(pt["saving_min"], 1),
    })
    b = r["wait_burden"]
    burden_tbl = pd.DataFrame({
        "시간대": b["hour"],
        "하루_초과대기_인시간(10단위)": _round_to(b["excess_wait_person_hours_day"], 10),
        "연간_초과대기_인시간(1000단위)": _round_to(b["excess_wait_person_hours_year"], 1000),
        "추천이용시_하루절감_상한_인시간(10단위)": _round_to(b["saving_upper_person_hours_day"], 10),
    })
    sv = r.get("service_cfg", {})
    return {
        "16ExpectedWaitPivot": (wait_pivot, "[평일] 출퇴근 시간 정류장 x 도착 시각(10분 단위) 기대 대기 시간(분): 실제 버스 순서·남은 좌석으로 줄 서기 계산 (만석 연속·못 탄 사람 누적 포함)", True),
        "17ServiceRecommendation": (rec_tbl, f"[평일/주말 출퇴근] 정류장에 있는 사람: 다른 9401 정류장으로 버스·도보로 옮겨 타는 게 서울을 빠져나가기까지 {sv.get('min_saving_min', 3)}분 이상 빠른 경우", False),
        "19LocationRecommendation": (pt_tbl, f"[평일/주말 출퇴근] 정류장이 아닌 곳(지하철역 예시)에 있는 사람: 가장 가까운 정류장보다 다른 정류장으로 걸어가는 게 {sv.get('min_saving_min', 3)}분 이상 빠른 경우", False),
        "18WaitBurdenByHour": (burden_tbl, f"[평일] 서울 도심 구간 시간대별 초과 대기 부담(인·시간)과 연간 환산(평일 {sv.get('workdays_per_year', 250)}일), 추천 전원 이용 시 절감 상한", False),
    }


def build_tables(r: dict, m09) -> dict[str, tuple[pd.DataFrame, str, bool]]:
    """반출할 표를 만든다. {파일명: (표, 설명, 인덱스 저장 여부)}"""
    scored, scored_we = r["model_scored"], r.get("model_scored_weekend")
    _, _, label_map = _stop_labels(scored)
    seats = r["cfg"].seat_capacity

    tables = {}
    tables.update(_daytype_tables(scored, "weekday", ""))
    if scored_we is not None:
        tables.update(_daytype_tables(scored_we, "weekend", "b"))

    unmet = _unmet_by_hour(scored, "평일", seats)
    if scored_we is not None:
        unmet = unmet.join(_unmet_by_hour(scored_we, "주말", seats))
    unmet.index.name = "시간대"

    coef = r["model"].coef_table().copy()
    coef["term"] = coef["term"].str.replace(r"seq(\d+)", lambda m: label_map.get(int(m.group(1)), m.group(0)), regex=True)
    coef[["coef", "multiplier"]] = coef[["coef", "multiplier"]].round(4)
    dh = r["model"].dow_hour_table().round(3)
    dh.index = [DOW_KR[k] for k in dh.index]
    dh.index.name = "요일"

    def _plain(d: dict) -> dict:  # 버스별 기록(DataFrame) 같은 큰 값은 빼고
        return {k: v for k, v in d.items() if not isinstance(v, (pd.DataFrame, pd.Series, dict))}

    items = _plain({**vars(r["cfg"]), **r["diag"], **r.get("run_params", {})})
    if r.get("diag_weekend"):
        items.update({f"weekend_{k}": v for k, v in r["diag_weekend"].items()})
    config_tbl = pd.DataFrame(list(items.items()), columns=["항목", "값"]).astype(str)

    q = {k: (f"{v:.1%}" if isinstance(v, float) and ("비율" in k) else v) for k, v in r["card_quality"].items()}
    quality_tbl = pd.DataFrame(list(q.items()), columns=["항목", "값"]).astype(str)

    mh = r["measured_headway"].copy()
    mh.insert(0, "정류장", mh.pop("seq").map(label_map))
    mh = mh.rename(columns={"band": "시간대", "daytype": "구분"})
    mh["하루_운행대수"] = _round_to(mh["하루_운행대수"], 1)
    mh["평균공백_분"] = _round_to(mh["평균공백_분"], 0.1)
    mh["공백표준편차_분"] = _round_to(mh["공백표준편차_분"], 0.1)
    for col in ("몰림정도_CV", "몰림비율(평균간격50%미만)", "1분이내_연속도착_비율", "평균의2배넘는_공백_비율", "거의만석_도착_비율"):
        mh[col] = _round_to(mh[col], 0.01)

    tables.update({
        "04UnmetDemandByHour": (unmet.reset_index(), "서울 도심 구간 시간대별 추정 미승차 수요(평일/주말, 구간 합계, 10명 단위)와 이를 태우는 데 필요한 추가 운행 대수", False),
        "05ModelCoefficients": (coef, "체류 목적 보정(포아송 회귀): 기준 대기 비율(정류장 대기 수요 ÷ 유동인구)과 정류장 배수 (multiplier=exp(coef))", False),
        "05bDowHourMultiplier": (dh, "체류 목적 보정: 요일 x 시간대 배수 (같은 요일·시간대의 다른 정류장에서 학습, 1 = 평균)", True),
        "06CvAblation": (r["cv"].round(4), "[평일] 교차검증(무작위 5-fold / 정류장 제외 LOSO): SKT 유동인구 신호 유무·변형별 오차 비교", False),
        "07Sensitivity": (r["sensitivity"].round(4), "[평일] 반경 x tau x 신호 변형 민감도: 기준 대비 순위 상관·top10 겹침", False),
        "08SaturationCurve": (r["saturation"].round(4), "[평일] 버스 도착 시 재차율 구간별 실제 승차 / 예측 수요 (1보다 작으면 자리가 없어 덜 탐)", False),
        "09RunConfig": (config_tbl, "실행 파라미터·진단값(데이터 없음)", False),
        "10CardDataQuality": (quality_tbl, "교통카드 데이터 처리 품질(기간, 정류장 연결률, 하차 추정 비율, 운행 횟수 등. 인원 없음)", False),
        "11MeasuredHeadwayByStop": (mh, "교통카드로 잰 정류장별 실제 운행 대수·배차 공백·몰림 정도·거의 만석 도착 비율(평일/주말, 출근 6~9시 / 퇴근 16~23시)", False),
        "12BunchingEffect": (r["bunching"].round(4), "출퇴근 시간 탑승가능확률: 실측(버스별 공백·재차) vs 몰림 무시(평균 간격·평균 재차)", False),
    })
    if scored_we is not None:
        tables["13DaytypeSummaryByHour"] = (
            _daytype_summary({"weekday": scored, "weekend": scored_we}),
            "시간대별 평일 vs 주말 탑승가능확률 요약(평균·최저 확률, 실제 배차, 위험 칸 수)", False)
    if r.get("scored_by_dow"):
        tables.update(_dow_tables(r["scored_by_dow"]))
    tables.update(_service_tables(r, m09, label_map))
    if r.get("deck"):
        tables.update(_deck_tables(r, label_map))
    return tables


# %%
def _seoul_real(r: dict) -> list:
    """서울 도심 실제 정류장(가상 제외) seq, 노선 순서대로."""
    return sorted(r["geom"].index)


def _deck_tables(r: dict, label_map: dict) -> dict:
    """장표 13~19쪽 그래프에 들어가는 숫자 (전부 비율·평균·지수, 인원은 10명 단위)."""
    deck = r["deck"]
    seoul = _seoul_real(r)
    bp = r["bus_profile"]
    bp_s = bp[bp["seq"].isin(seoul)]
    out = {}

    # 13쪽: 정류장 순서별 남은 좌석·만석 도착 (평일 퇴근 피크 / 주말 오후 피크)
    rows = []
    for dt, h, kr in (("weekday", deck["weekday_pm"], "평일"), ("weekend", deck["weekend_pm"], "주말")):
        d = bp_s[(bp_s["daytype"] == dt) & (bp_s["hour"] == h)].set_index("seq").reindex(seoul)
        rows.append(pd.DataFrame({"구분": f"{kr} {h}시", "정류장": [label_map.get(q, q) for q in seoul],
                                  "도착시_남은좌석_평균": _round_to(d["seats_left_mean"], 0.1),
                                  "만석도착_비율": _round_to(d["full_arrival_share"], 0.05)}))
    out["20SeatsAlongRoute"] = (pd.concat(rows, ignore_index=True),
                                "[장표13] 서울 도심 정류장 순서별 버스 도착 시 남은 좌석 평균·만석(1석 이하) 도착 비율 (평일 퇴근 피크 / 주말 오후 피크)", False)

    # 14쪽: 평일 vs 주말 시간대별 운행 횟수·평균 간격·편차·몰림 비율 (서울 도심 정류장 평균)
    g = bp_s[bp_s["trips_per_day"] > 0].groupby(["daytype", "hour"])
    hs = pd.DataFrame({
        "정류장당_하루운행횟수": g["trips_per_day"].mean(), "평균도착간격_분": g["gap_mean"].mean(),
        "간격표준편차_분": g["gap_sd"].mean(), "간격변동계수": g["gap_cv"].mean(),
        "몰림비율(평균간격50%미만)": g["bunch_share"].mean(),
    }).reset_index()
    for c, step in (("정류장당_하루운행횟수", 0.1), ("평균도착간격_분", 0.1), ("간격표준편차_분", 0.1),
                    ("간격변동계수", 0.01), ("몰림비율(평균간격50%미만)", 0.01)):
        hs[c] = _round_to(hs[c], step)
    hs["daytype"] = hs["daytype"].map(DAYTYPE_KR)
    out["21HourlyServiceByDaytype"] = (hs.rename(columns={"daytype": "구분", "hour": "시간대"}),
                                       "[장표14] 평일 vs 주말 시간대별 운행 횟수·실제 도착 간격 평균·편차·몰림 비율 (서울 도심 정류장 평균)", False)

    # 15쪽: 대표 정류장 요일 지수 / 시간대 비중 (원값 없음)
    fpat = r["flow_pattern"]
    di = fpat["dow_index"].reindex(seoul)
    di.index = [label_map.get(q, q) for q in di.index]
    di.columns = [DOW_KR[k] for k in di.columns]
    out["22FlowDowIndex"] = (di.apply(lambda c: _round_to(c, 1)).rename_axis("정류장").reset_index(),
                             "[장표15] 정류장별 요일 유동인구 지수 (7개 요일 평균 = 100, SKT 반경 150m)", False)
    hsh = fpat["hour_share"].reindex(seoul).T * 100
    hsh.columns = [label_map.get(q, q) for q in hsh.columns]
    out["23FlowHourShare"] = (hsh.apply(lambda c: _round_to(c, 0.1)).rename_axis("시간대").reset_index(),
                              "[장표15] 정류장별 하루 유동인구 중 시간대 비중(%) (SKT 반경 150m)", False)

    # 17쪽: 정류장별 9401 점유율 (18시 / 하루 전체), 정차 노선 수
    sh = r.get("share")
    if sh is not None:
        s18 = sh[sh["hour"] == 18].drop_duplicates("seq").set_index("seq").reindex(seoul)
        out["24Route9401Share"] = (pd.DataFrame({
            "정류장": [label_map.get(q, q) for q in seoul],
            "9401점유율_평일18시": _round_to(s18["share_weekday"], 0.01),
            "9401점유율_18시(모든요일)": _round_to(s18["share_all"], 0.01),
            "9401점유율_하루전체": _round_to(s18["share_all_stop"], 0.01),
            "정차노선수(한달승차30명이상)": s18["n_routes"].to_numpy(),
        }), "[장표17] 정류장별 전체 노선 승차 중 9401 비율(평일 18시 / 모든 요일 18시 / 한 달 전체)과 정차 노선 수", False)

    # 18쪽: 모델 신뢰성 검증
    val = r.get("validation", {})
    vt = pd.DataFrame([(k, round(v, 3) if isinstance(v, float) else v) for k, v in val.items()], columns=["항목", "값"]).astype(str)
    out["25ValidationSummary"] = (vt, "[장표18] 모델 신뢰성 검증: ① 재현 오차 ② 시간 분할(1~3주 -> 4주차) ④ SKT 기여", False)
    if len(r.get("split_by_grade", [])):
        sg = r["split_by_grade"].copy()
        for c in sg.columns[2:]:
            sg[c] = _round_to(sg[c], 0.01)
        out["25bTimeSplitByGrade"] = (sg.rename(columns={"grade": "1~3주_등급"}),
                                      "[장표18] 1~3주로 매긴 등급별 4주차 실제 만석 칸 비율 (등급이 나쁠수록 높아야 함)", False)
    ds = r["demand_sens"].copy()
    for c in ds.columns[1:]:
        ds[c] = _round_to(ds[c], 0.01 if "확률" in c or "겹침" in c else 1)
    out["25cDemandSensitivity"] = (ds, "[장표18] ③ 복원 수요 ±20%: 위험 칸 수·평균 확률·대기·위험 top10 겹침", False)

    # 19쪽: 위험 칸 Top5 (요일·시간대)
    top = _risk_rank(r["scored_by_dow"], seoul).head(5)
    out["26RiskTop5"] = (pd.DataFrame({
        "정류장": top["seq"].map(label_map).to_numpy(),
        "요일·시간": [f"{DOW_KR[k]} {h}시" for k, h in zip(top["dow"], top["hour"])],
        "탑승확률": _round_to(top["p_board_mid"], 0.05),
        "놓치는버스_평균_대": _round_to(top["missed_buses_mean"], 0.1),
        "놓치는버스_상위10%_대": _round_to(top["missed_buses_p90"], 1),
        "예상대기_평균_분": _round_to(top["wait_total_min"], 1),
        "예상대기_중앙값_분": _round_to(top["wait_p50_min"], 1),
        "예상대기_상위10%_분": _round_to(top["wait_p90_min"], 1),
    }), "[장표19] 요일·시간대별 탑승 실패 위험 칸 Top5: 첫 버스 탑승확률, 만석이라 보내는 버스 수, 사람별 대기 시간 분포", False)

    out["27DeckNumbers"] = (_deck_numbers(r, label_map), "장표 빈칸(○)에 들어갈 숫자 모음 (모두 위 표·그림과 같은 계산)", False)
    return out


def _risk_rank(scored_by_dow: dict, seoul: list) -> pd.DataFrame:
    """요일·시간대 칸을 위험한 순서로. 확률이 같으면(예: 여러 칸이 0%) 기대 대기가 긴 칸이 먼저.
    첫차·막차 시간(6시 전, 시간당 버스 2대 미만)은 버스가 드물어서 뺌"""
    allc = pd.concat([d.assign(dow=k) for k, d in scored_by_dow.items()], ignore_index=True)
    allc = allc[allc["has_skt"] & allc["seq"].isin(seoul) & allc["p_board_mid"].notna()
                & allc["hour"].between(6, 23) & (allc["n_buses"] >= 2)]
    return allc.sort_values(["p_board_mid", "wait_total_min"], ascending=[True, False])


def _deck_numbers(r: dict, label_map: dict) -> pd.DataFrame:
    deck, seoul = r["deck"], _seoul_real(r)
    focus = deck["focus"]
    fl = label_map.get(focus, str(focus))
    bp = r["bus_profile"].set_index(["daytype", "seq", "hour"])
    hw, hwe = deck["weekday_pm"], deck["weekend_pm"]
    rows = []

    def add(page, item, value):
        rows.append((page, item, value))

    def pct(x):
        return f"{x:.0%}" if np.isfinite(x) else "-"

    # 13
    for dt, h, kr in (("weekday", hw, "평일"), ("weekend", hwe, "주말")):
        if (dt, focus, h) in bp.index:
            v = bp.loc[(dt, focus, h)]
            add(13, f"{kr} {h}시 {fl} 도착 시 남은 좌석 평균", f"{v['seats_left_mean']:.1f}석")
            add(13, f"{kr} {h}시 {fl} 만석(1석 이하) 도착 비율", pct(v["full_arrival_share"]))
    # 14
    b = r["bus_profile"]
    b = b[b["seq"].isin(seoul) & (b["trips_per_day"] > 0)]
    wd = b[(b["daytype"] == "weekday") & (b["hour"] == hw)]
    add(14, f"평일 {hw}시 서울 도심 정류장 평균 도착 간격", f"{wd['gap_mean'].mean():.1f}분")
    add(14, f"평일 {hw}시 도착 간격 표준편차", f"{wd['gap_sd'].mean():.1f}분")
    add(14, f"평일 {hw}시 몰림 비율(평균 간격 50% 미만)", pct(wd["bunch_share"].mean()))
    we = b[(b["daytype"] == "weekend") & b["hour"].between(12, 20)].groupby("hour")["gap_mean"].mean()
    if len(we):
        add(14, "주말 오후(12~20시) 평균 도착 간격 최대", f"{we.max():.1f}분 ({int(we.idxmax())}시)")
    # 15
    fpat = r["flow_pattern"]
    add(15, "서울 도심 정류장 간 최근접 거리 중앙값", f"{fpat['nearest_stop_median_m']:.0f}m")
    add(15, "반경 150m 안 유동인구 셀 수 중앙값", f"{fpat['cells_in_radius_median']:.0f}개")
    sens = r["sensitivity"]
    cfg = r["cfg"]
    for rad in (100.0, 200.0):
        s = sens[(sens["radius_m"] == rad) & (sens["tau"] == cfg.tau) & (sens["signal_variant"] == cfg.signal)]
        if len(s):
            add(15, f"반경 {rad:.0f}m: 150m 대비 탑승확률 순위 상관 / 위험 top10 겹침",
                f"{s['spearman_vs_ref'].iloc[0]:.2f} / {s['top10_overlap_vs_ref'].iloc[0]:.0%}")
    add(15, "같은 18시라도 정류장·요일에 따른 유동인구 최대/최소", f"{fpat['max_ratio_18h']:.1f}배")
    di = fpat["dow_index"]
    for kind, q in deck["stops"].items():
        if q in di.index:
            wk = di.loc[q, ["mon", "tue", "wed", "thu", "fri"]].mean()
            add(15, f"[{kind}] {label_map.get(q, q)} 주말(토·일) 평균 vs 평일 평균",
                f"{(di.loc[q, ['sat', 'sun']].mean() / wk - 1):+.0%}")
            add(15, f"[{kind}] {label_map.get(q, q)} 토요일 vs 평일 평균", f"{(di.loc[q, 'sat'] / wk - 1):+.0%}")
            hsh = fpat["hour_share"].loc[q]
            add(15, f"[{kind}] {label_map.get(q, q)} 유동인구 피크 시간", f"{int(hsh.idxmax())}시 ({hsh.max():.1%})")
    # 16
    sc, we_sc, sdow = r["model_scored"], r["model_scored_weekend"], r["scored_by_dow"]

    def ratio(d, seqs=None, hours=None, col="latent_daily"):
        """col 합 ÷ 유동인구 합. col = latent_daily(9401 대기 수요) / station_demand_daily(정류장 대기 수요)"""
        d = d[d["has_skt"] & d["flow_day"].gt(0) & d[col].notna()]
        if seqs is not None:
            d = d[d["seq"].isin(seqs)]
        if hours is not None:
            d = d[d["hour"].isin(hours)]
        return d[col].sum() / d["flow_day"].sum() if len(d) else np.nan
    st = "station_demand_daily"
    r_wd18, r_sat18 = ratio(sc, [focus], [18], st), ratio(sdow["sat"], [focus], [18], st)
    add(16, f"{fl} 평일 18시 대기 비율(정류장 대기 수요 ÷ 유동인구)", f"{r_wd18:.1%}")
    add(16, f"{fl} 평일 18시 대기 비율 ÷ 토요일 18시", f"{r_wd18 / r_sat18:.1f}배" if r_sat18 > 0 else "-")
    add(16, "유동인구 중 정류장 대기 수요 비율 (서울 도심, 평일 / 주말)", f"{ratio(sc, col=st):.1%} / {ratio(we_sc, col=st):.1%}")
    skt = sc[sc["has_skt"] & (sc["n_buses"] > 0)]
    con = skt[~skt["unconstrained"]]
    m = r["model"]
    add(16, "만석 도착 칸 / 전체 칸 (정류장 x 요일 x 시간대, 운행·유동인구 있는 칸)",
        f"{m.n_full} / {m.n_cells} ({m.n_full / max(m.n_cells, 1):.0%})")
    add(16, "기준 대기 비율 (정류장 대기 수요 ÷ 유동인구, 배수 평균일 때)", f"{np.exp(m.base):.1%}")
    if con["daily_boarding"].sum() > 0:
        add(16, "[평일] 만석 칸 복원 대기 수요 ÷ 승차 기록", f"{con['latent_daily'].sum() / con['daily_boarding'].sum() - 1:+.0%}")
    # 17
    add(17, "유동인구 중 9401 대기 수요 비율 (서울 도심, 평일 / 주말)", f"{ratio(sc):.2%} / {ratio(we_sc):.2%}")
    q = r.get("card_quality", {})
    add(17, "통합: 다른 정산사업자 정류장ID 수 / 통합 ID로 연결된 승차 비율",
        f"{q.get('통합한_정류장ID_수', 0)}개 / {q.get('통합ID로_연결된_승차_비율', 0):.1%}")
    dg = r["diag"]
    if dg.get("n_share_cells"):
        add(17, "점유율 대체 칸 (정류장 x 요일 x 시간대): 소수 칸 / 만석 도착 칸 / 전체",
            f"{dg['n_share_few']} / {dg['n_share_full']} / {dg['n_share_cells']}")
    sh = r.get("share")
    if sh is not None:
        s0 = sh.drop_duplicates("seq").set_index("seq").reindex(seoul)
        multi = int((s0["n_routes"] >= 2).sum())
        add(17, "서울 도심 정류장 중 복수 노선 정차", f"{multi} / {len(seoul)}개")
        if s0["n_routes"].notna().any():
            q = int(s0["n_routes"].idxmax())
            s18 = sh[(sh["seq"] == q) & (sh["hour"] == 18)]["share_weekday"]
            add(17, f"정차 노선이 가장 많은 {label_map.get(q, q)}: 노선 수 / 평일 18시 9401 점유율",
                f"{int(s0.at[q, 'n_routes'])}개 / {pct(float(s18.iloc[0])) if len(s18) else '-'}")
    pk = sc[sc["has_skt"] & sc["hour"].isin([17, 18, 19])].groupby("seq")["latent_daily"].sum()
    if len(pk):
        add(17, "평일 퇴근(17~19시) 9401 대기 수요가 가장 많은 정류장", label_map.get(int(pk.idxmax()), ""))
    f18 = sc[(sc["seq"] == focus) & (sc["hour"] == 18)]
    if len(f18) and f18["flow_day"].iloc[0] > 0:
        v = f18.iloc[0]
        add(17, f"{fl} 평일 18시 단계별 축소 (유동인구 = 100)",
            f"100 -> 정류장 대기 수요 {v['station_demand_daily'] / v['flow_day'] * 100:.1f} -> 9401 대기 수요 {v['latent_daily'] / v['flow_day'] * 100:.1f}"
            f" (참고: 실제 9401 승차 {v['daily_boarding'] / v['flow_day'] * 100:.1f})")
    # 18
    for k, v in r.get("validation", {}).items():
        add(18, k, f"{v:.0%}" if isinstance(v, float) and ("률" in k or "WAPE" in k) else v)
    ds = r["demand_sens"]
    for _, row in ds[ds["수요배율"] != 1.0].iterrows():
        add(18, f"수요 x{row['수요배율']:.1f}: 위험 top10 유지", f"{row['위험top10_겹침']:.0%}")
    # 19
    out_seqs = [q for q in seoul if r["geom"].at[q, "direction"] == "out"]
    p18 = sc[(sc["hour"] == 18)].set_index("seq")["p_board_mid"]
    if out_seqs:
        add(19, f"평일 18시 탑승확률: 성남 방향 첫 정류장 {label_map.get(out_seqs[0])} -> {fl}",
            f"{pct(p18.get(out_seqs[0], np.nan))} -> {pct(p18.get(focus, np.nan))}")
    add(19, "토요일 탑승확률이 가장 낮은 시간", f"{deck['sat_worst']}시")
    fh = {k: d[(d["seq"] == focus) & d["hour"].between(6, 23)].set_index("hour")["p_board_mid"] for k, d in sdow.items()}
    tw = pd.concat([fh[k] for k in ("tue", "wed", "thu")], axis=1).mean(axis=1).dropna()
    if len(tw):
        add(19, f"{fl} 화~목 최저 탑승확률", f"{pct(tw.min())} ({int(tw.idxmin())}시)")
        add(19, f"{fl} 금요일 같은 시간", pct(fh["fri"].get(int(tw.idxmin()), np.nan)))
    wke = pd.concat([fh["sat"], fh["sun"]], axis=1).min(axis=1).dropna()
    if len(wke):
        add(19, f"{fl} 주말 최저 탑승확률", f"{pct(wke.min())} ({int(wke.idxmin())}시)")
    allc = _risk_rank(sdow, seoul)
    if len(allc):
        w = allc.iloc[0]
        add(19, "가장 위험한 칸", f"{label_map.get(int(w['seq']))} {DOW_KR[w['dow']]} {int(w['hour'])}시, 확률 {pct(w['p_board_mid'])}")
        add(19, "그 칸 최악(상위 10%): 보내는 버스 / 대기", f"{w['missed_buses_p90']:.0f}대 / {w['wait_p90_min']:.0f}분")
    return pd.DataFrame(rows, columns=["장표", "항목", "값"]).astype(str)


# %%
def write_manifest(out_dir: Path, manifest: list) -> None:
    """반출 신청서에 붙일 파일 목록(ExportManifest.md)."""
    lines = [
        "# 반출 결과물 목록 (ExportManifest)",
        "",
        "모든 파일은 분석 최종 결과(집계표·통계량·모델 성능지표·변수 중요도·시각화)이며,",
        "원자료(교통카드 이용내역, SKT 유동인구) 및 행 단위 분석용 데이터셋은 포함하지 않습니다.",
        "",
        "- 유동인구·승하차 원값, 카드·차량·정류장ID, 좌표 컬럼은 포함하지 않음(코드에서 자동 검사)",
        "- 확률 5%p, 대기시간 1분, 인원 10명 단위 반올림",
        f"- 표 1개당 최대 {MAX_ROWS}행",
        "- 그림은 집계값(비율·확률·평균·지수)만 그림. 일별 이용량도 평균=100 지수",
    ]
    sections = [
        ("요약표", lambda p: p.startswith("tables/")),
        ("그림 - 탐색(EDA): 교통카드 원자료 집계", lambda p: p.startswith("figures/E")),
        ("그림 - 요약", lambda p: p.startswith("figures/F0")),
        ("그림 - 탑승가능확률·미승차", lambda p: any(p.startswith(f"figures/F{n}") for n in ("1B", "2", "8", "9", "10", "11", "12"))),
        ("그림 - 검증", lambda p: any(p.startswith(f"figures/F{n}") for n in ("3", "4", "5", "7"))),
        ("그림 - 서비스", lambda p: any(p.startswith(f"figures/F{n}") for n in ("6", "13", "14", "15", "16"))),
        ("그림 - 장표용 (파일 이름 숫자 = 장표 쪽)", lambda p: p.startswith("figures/S")),
    ]
    used = set()
    for title, match in sections:
        rows = [m for m in manifest if match(m[1]) and m[1] not in used]
        if not rows:
            continue
        used |= {m[1] for m in rows}
        lines += ["", f"## {title} ({len(rows)}개)", "", "| 구분 | 파일 | 내용 | 규모 |", "|---|---|---|---|"]
        lines += [f"| {k} | {p} | {d} | {n} |" for k, p, d, n in rows]
    rest = [m for m in manifest if m[1] not in used]
    if rest:
        lines += ["", "## 기타", "", "| 구분 | 파일 | 내용 | 규모 |", "|---|---|---|---|"]
        lines += [f"| {k} | {p} | {d} | {n} |" for k, p, d, n in rest]
    (out_dir / "ExportManifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_all(r: dict, out_dir: Path, m09) -> None:
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    # 이전 실행에서 남은 표/그림 지우기 (이름이 바뀐 옛 파일이 반출에 섞이지 않게)
    for old in list((out_dir / "tables").glob("*.csv")) + list((out_dir / "figures").glob("*.png")):
        old.unlink()
    manifest: list = []
    for name, (df, desc, index) in build_tables(r, m09).items():
        _save_table(out_dir, name, df, manifest, desc, index=index)
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from Loader import load_module
        load_module("modeling/Figures.py").build_all(r, out_dir / "figures", manifest, m09)
    except ImportError:
        print("⚠ matplotlib 없음 -> 그림 생략, 표만 반출")
    write_manifest(out_dir, manifest)
    print(f"반출 패키지 -> {out_dir} ({len(manifest)}개 파일)")
