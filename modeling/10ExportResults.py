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
    "route_max_load_hour", "per_bus_load", "latent_daily", "demand_pred", "unmet_daily",
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
        "추가대기_분": _round_to(worst["extra_wait_min_mid"], 1),
        "추가대기_분_나쁜날(60=60분이상)": _round_to(worst["extra_wait_min_low"], 1),
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
    waits = m09.add_expected_wait(r["model_scored"])
    src = waits[waits["has_skt"] & waits["hour"].isin(COMMUTE_HOURS)].copy()
    src["정류장"] = src["seq"].map(label_map)
    wait_pivot = src.pivot_table(index="정류장", columns="hour", values="wait_total_min", aggfunc="first")
    wait_pivot = wait_pivot.reindex([label_map[s] for s in sorted(src["seq"].unique())]).apply(lambda c: _round_to(c, 1))

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
        "시간대": rec["hour"].to_numpy(),
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
        "시간대": pt["hour"].to_numpy(),
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
        "16ExpectedWaitPivot": (wait_pivot, "[평일] 출퇴근 시간 정류장x시간대 기대 대기 시간(분) = 실제 배차/2 x (1+몰림^2) + 만차 추가 대기", True),
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

    def _plain(d: dict) -> dict:  # 버스별 기록(DataFrame) 같은 큰 값은 빼고
        return {k: v for k, v in d.items() if not isinstance(v, (pd.DataFrame, pd.Series))}

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
    for col in ("몰림정도_CV", "1분이내_연속도착_비율", "평균의2배넘는_공백_비율", "거의만석_도착_비율"):
        mh[col] = _round_to(mh[col], 0.01)

    tables.update({
        "04UnmetDemandByHour": (unmet.reset_index(), "서울 도심 구간 시간대별 추정 미승차 수요(평일/주말, 구간 합계, 10명 단위)와 이를 태우는 데 필요한 추가 운행 대수", False),
        "05ModelCoefficients": (coef, "수요 복원 Poisson GLM 계수(평일 학습, 정류장·시간대·통근 효과, multiplier=exp(coef))", False),
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
    return tables


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
