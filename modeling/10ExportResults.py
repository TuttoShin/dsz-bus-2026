"""
10ExportResults.py

안심구역에서 가지고 나갈 결과물만 export/ 폴더에 모은다.
1차 반출 때 features.parquet, scores.parquet이 "행 단위 데이터"라서 반려됨
-> 이번엔 요약표, 통계량, 모델 성능, 그래프만 만든다.

반출 규칙 (코드에서 자동으로 검사)
1. 유동인구, 승하차 인원 같은 원본 값은 표에 넣지 않는다 (FORBIDDEN_COLUMNS)
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
    "flow_mon", "flow_tue", "flow_wed", "flow_thu", "flow_fri", "flow_sat", "flow_sun", "signal", "n_cells_matched", "boarding", "alighting", "net", "daily_boarding",
    "load_lower", "load_before_board", "cum_net", "route_max_load_hour", "latent_daily", "demand_pred", "unmet_daily",
    "표준버스정류장ID", "X_COORD", "Y_COORD", "utmk_x", "utmk_y",
}
MAX_ROWS = 100
PIVOT_HOURS = list(range(24))  # 표(CSV)는 24시간 전체
COMMUTE_HOURS = list(range(6, 10)) + list(range(16, 24))  # 그래프는 출근 6~9시, 퇴근 16~23시만


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
    path = out_dir / "tables" / f"{name}.csv"
    df.to_csv(path, index=index, encoding="utf-8-sig")
    manifest.append(("표", f"tables/{name}.csv", desc, f"{len(df)}행"))


# %%
def setup_font() -> None:
    """그래프 한글 폰트 설정. 맑은 고딕 -> 나눔고딕 순으로 찾고, 없으면 경고 (한글이 네모로 깨짐)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    fonts_dir = Path(__file__).resolve().parents[1] / "fonts"
    if fonts_dir.exists():
        for f in fonts_dir.glob("*.ttf"):
            font_manager.fontManager.addfont(str(f))
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Noto Sans CJK KR", "Noto Sans KR"]:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    else:
        print("⚠ 한글 폰트 없음 -> 그림의 한글이 깨질 수 있음. fonts/ 폴더에 NanumGothic.ttf를 넣어주세요.")
    plt.rcParams["axes.unicode_minus"] = False


# %%
DAYTYPE_KR = {"weekday": "평일", "weekend": "주말"}
DOW_KR = {"mon": "월", "tue": "화", "wed": "수", "thu": "목", "fri": "금", "sat": "토", "sun": "일·공휴일"}


def _stop_labels(scored: pd.DataFrame) -> tuple[pd.DataFrame, list, dict]:
    order = scored.drop_duplicates("seq").sort_values("seq")[["seq", "stop_label", "segment"]]
    labels = [f"{row.seq:02d} {row.stop_label}" for row in order.itertuples()]
    return order, labels, dict(zip(order["seq"], labels))


def _daytype_tables(scored: pd.DataFrame, kec_stop: pd.DataFrame, daytype: str, suffix: str) -> dict:
    """평일 또는 주말 하나에 대한 확률표 / 등급표 / 위험 top10."""
    kr = DAYTYPE_KR[daytype]
    order, labels, label_map = _stop_labels(scored)
    src = scored[scored["hour"].isin(PIVOT_HOURS)].copy()
    src["정류장"] = src["seq"].map(label_map)

    p_pivot = src.pivot_table(index="정류장", columns="hour", values="p_board_mid", aggfunc="first")
    p_pivot = p_pivot.reindex(labels).apply(lambda c: _round_to(c, 0.05))
    p_pivot.insert(0, "구간", order["segment"].to_numpy())
    p_pivot.insert(1, "주변도로혼잡지수", _round_to(order["seq"].map(kec_stop.set_index("seq")["stop_congestion_index"]), 1))
    g_pivot = src.pivot_table(index="정류장", columns="hour", values="grade", aggfunc="first").reindex(labels)

    worst = scored[scored["has_skt"]].nsmallest(10, "p_board_mid")
    worst_tbl = pd.DataFrame({
        "정류장": worst["seq"].map(label_map).to_numpy(),
        "시간대": worst["hour"].to_numpy(),
        "탑승확률_하한(배차최장)": _round_to(worst["p_board_low"], 0.05),
        "탑승확률_중앙": _round_to(worst["p_board_mid"], 0.05),
        "탑승확률_상한(배차최단)": _round_to(worst["p_board_high"], 0.05),
        "추가대기_분_최소": _round_to(worst["extra_wait_min_high"], 1),
        "추가대기_분_최대(60=60분이상)": _round_to(worst["extra_wait_min_low"], 1),
        "등급": worst["grade"].to_numpy(),
    })
    return {
        f"01{suffix}BoardingProbabilityPivot{daytype.capitalize()}": (
            p_pivot, f"[{kr}] 정류장x시간대 탑승가능확률(배차 중앙 시나리오, 5%p 반올림). 서비스 입력값", True),
        f"02{suffix}GradePivot{daytype.capitalize()}": (g_pivot, f"[{kr}] 정류장x시간대 등급(여유/보통/위험)", True),
        f"03{suffix}WorstCellsTop10{daytype.capitalize()}": (
            worst_tbl, f"[{kr}] 탑승가능확률 최하위 10개 정류장·시간대와 확률 범위·추가 대기시간", False),
    }


def _unmet_by_hour(scored: pd.DataFrame, kr: str) -> pd.DataFrame:
    skt = scored[scored["has_skt"]]
    by_hour = skt.groupby("hour").agg(latent=("latent_daily", "sum"), unmet=("unmet_daily", "sum"))
    return pd.DataFrame({
        f"{kr}_추정_미승차_일평균_명(10명단위)": _round_to(by_hour["unmet"], 10),
        f"{kr}_미승차_비율_%": _round_to(100 * by_hour["unmet"] / by_hour["latent"].where(by_hour["latent"] > 0), 1),
    }, index=by_hour.index)


def _daytype_summary(scored_by_day: dict) -> pd.DataFrame:
    """시간대별 평일 vs 주말 요약 (서울 도심 11개 정류장)."""
    cols = {}
    for daytype, scored in scored_by_day.items():
        kr = DAYTYPE_KR[daytype]
        skt = scored[scored["has_skt"]]
        g = skt.groupby("hour")
        cols[f"{kr}_평균확률"] = _round_to(g["p_board_mid"].mean(), 0.05)
        cols[f"{kr}_최저확률"] = _round_to(g["p_board_mid"].min(), 0.05)
        cols[f"{kr}_위험칸수"] = g["grade"].apply(lambda x: int((x == "위험").sum()))
        cols[f"{kr}_보통칸수"] = g["grade"].apply(lambda x: int((x == "보통").sum()))
    out = pd.DataFrame(cols)
    out.index.name = "시간대"
    return out.reset_index()


def _dow_tables(scored_by_dow: dict) -> dict:
    """요일별 정류장x시간대 확률(한 표에 요일 열로) + 시간대별 요일 요약."""
    long_rows, summary = [], {}
    for k, sc in scored_by_dow.items():
        _, labels, label_map = _stop_labels(sc)
        piv = sc.pivot_table(index="seq", columns="hour", values="p_board_mid", aggfunc="first")
        piv = piv.apply(lambda c: _round_to(c, 0.05))
        piv.insert(0, "정류장", piv.index.map(label_map))
        piv.insert(1, "요일", DOW_KR[k])
        long_rows.append(piv.reset_index(drop=True))
        skt = sc[sc["has_skt"]]
        g = skt.groupby("hour")
        summary[f"{DOW_KR[k]}_최저확률"] = _round_to(g["p_board_mid"].min(), 0.05)
        summary[f"{DOW_KR[k]}_위험칸수"] = g["grade"].apply(lambda x: int((x == "위험").sum()))
    summ = pd.DataFrame(summary)
    summ.index.name = "시간대"
    return {
        "16BoardingProbabilityByDow": (pd.concat(long_rows, ignore_index=True),
                                       "요일별(월~일) 정류장x시간대 탑승가능확률(배차 중앙 시나리오, 5%p 반올림)", False),
        "17DowSummaryByHour": (summ.reset_index(), "시간대별 요일별 최저 탑승확률과 위험 칸 수", False),
    }


def build_tables(r: dict) -> dict[str, tuple[pd.DataFrame, str, bool]]:
    """반출할 표를 만든다. {파일명: (표, 설명, 인덱스 저장 여부)}"""
    scored: pd.DataFrame = r["model_scored"]
    scored_we: pd.DataFrame | None = r.get("model_scored_weekend")
    _, _, label_map = _stop_labels(scored)

    tables = {}
    tables.update(_daytype_tables(scored, r["kec_stop"], "weekday", ""))
    if scored_we is not None:
        tables.update(_daytype_tables(scored_we, r["kec_stop"], "weekend", "b"))

    unmet_tbl = _unmet_by_hour(scored, "평일")
    if scored_we is not None:
        unmet_tbl = unmet_tbl.join(_unmet_by_hour(scored_we, "주말"))
    unmet_tbl.index.name = "시간대"
    unmet_tbl = unmet_tbl.reset_index()

    stop_kec = r["kec_stop"].copy()
    stop_kec.insert(0, "정류장", stop_kec["seq"].map(label_map))
    stop_kec = stop_kec.drop(columns=["seq", "stop_name"]).round(3)

    coef = r["model"].coef_table().copy()
    coef["term"] = coef["term"].str.replace(r"seq(\d+)", lambda m: label_map.get(int(m.group(1)), m.group(0)), regex=True)
    coef[["coef", "multiplier"]] = coef[["coef", "multiplier"]].round(4)

    cfg = r["cfg"]
    items = {**vars(cfg), **r["diag"], **r.get("run_params", {})}
    if r.get("cfg_weekend") is not None:
        items.update({f"weekend_{k}": v for k, v in {**vars(r["cfg_weekend"]), **r["diag_weekend"]}.items()})
    config_tbl = pd.DataFrame(list(items.items()), columns=["항목", "값"])
    config_tbl["값"] = config_tbl["값"].astype(str)

    tables.update({
        "04UnmetDemandByHour": (unmet_tbl, "서울 도심 구간 시간대별 추정 미승차 수요(평일/주말, 구간 합계, 10명 단위)", False),
        "05ModelCoefficients": (coef, "수요 복원 Poisson GLM 계수(평일 학습, 정류장·시간대·통근 효과, multiplier=exp(coef))", False),
        "06CvAblation": (r["cv"].round(4), "[평일] 교차검증(무작위 5-fold / 정류장 제외 LOSO): SKT 유동인구 신호 유무·변형별 오차 비교", False),
        "07Sensitivity": (r["sensitivity"].round(4), "[평일] 반경x tau x 신호 변형 민감도: 기준 대비 순위 상관·top10 겹침", False),
        "08SaturationCurve": (r["saturation"].round(4), "[평일] 버스 도착 시 재차율 구간별 실제 승차 / 예측 수요 (1보다 작으면 자리가 없어 덜 탐)", False),
        "09RuleVsModel": (r["rule_vs_model"].round(4), "[평일] 규칙 기반 지수(07) vs 모델 확률(09) 순위 일치도", False),
        "10RunConfig": (config_tbl, "실행 파라미터·진단값(데이터 없음)", False),
        "11RouteCongestionIndex": (r["kec_table"].round(3), "9401 경유 경부선 구간 혼잡 노출 지수(전체 + 시군구별, 도로 길이 가중평균)", False),
        "12StopCongestionIndex": (stop_kec, "정류장별 주변 도로 혼잡 지수(정류장이 있는 행정동 도로, 도로 길이 가중평균)", False),
    })
    if scored_we is not None:
        tables["13DaytypeSummaryByHour"] = (
            _daytype_summary({"weekday": scored, "weekend": scored_we}),
            "시간대별 평일 vs 주말 탑승가능확률 요약(평균·최저 확률, 위험/보통 칸 수)", False)
    if r.get("bunching") is not None:
        tables["14BunchingEffect"] = (
            r["bunching"].round(4),
            "버스 몰림 정도(CV)별 출퇴근 시간 탑승가능확률·위험/보통 칸 수 (0 = 몰림 없음, 기본 0.5)", False)
    if r.get("scored_by_dow"):
        tables.update(_dow_tables(r["scored_by_dow"]))
    return tables


# %%
def _heatmap(ax, scored: pd.DataFrame, hours: list, title: str, show_ylabels: bool = True):
    order = scored.drop_duplicates("seq").sort_values("seq")
    piv = scored[scored["hour"].isin(hours)].pivot_table(index="seq", columns="hour", values="p_board_mid")
    piv = piv.reindex(index=order["seq"], columns=hours)
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1)
    ax.set_xticks(range(len(hours)), hours)
    # 출근(6~9시)과 퇴근(16~23시) 사이에 구분선
    for i in range(1, len(hours)):
        if hours[i] - hours[i - 1] > 1:
            ax.axvline(i - 0.5, color="white", lw=4)
    if show_ylabels:
        ax.set_yticks(range(len(piv)), [f"{s:02d} {n}" for s, n in zip(order["seq"], order["stop_label"])], fontsize=9)
    else:
        ax.set_yticks(range(len(piv)), [])  # 오른쪽 패널은 왼쪽과 같은 정류장이라 이름 생략
    ax.set_xlabel("시간대 (출근 6~9시 | 퇴근 16~23시)")
    ax.set_title(title)
    return im


def build_figures(r: dict, out_dir: Path, manifest: list) -> None:
    """반출할 그래프(F1~F5)를 PNG로 저장한다. F1, F2는 출퇴근 시간대만."""
    import matplotlib.pyplot as plt

    scored = r["model_scored"]
    scored_we = r.get("model_scored_weekend")
    by_dow = r.get("scored_by_dow") or {}
    fig_dir = out_dir / "figures"

    def _save(fig, name, desc):
        fig.savefig(fig_dir / f"{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        manifest.append(("그림", f"figures/{name}.png", desc, ""))

    # F1 히트맵: 출퇴근 시간대만. 요일별(월~일) 있으면 7칸, 없으면 평일 | 주말
    if by_dow:
        panels = [(DOW_KR[k], sc) for k, sc in by_dow.items()]
        ncols = 4
    else:
        panels = [("평일", scored)] + ([("주말", scored_we)] if scored_we is not None else [])
        ncols = len(panels)
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols + 1.5, 4.6 * nrows), squeeze=False)
    for i, (name, sc) in enumerate(panels):
        ax = axes[i // ncols][i % ncols]
        im = _heatmap(ax, sc, COMMUTE_HOURS, f"[{name}] 탑승가능확률", show_ylabels=(i % ncols == 0))
    for j in range(len(panels), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.subplots_adjust(hspace=0.4, wspace=0.08)  # 아랫줄 제목이 윗줄 x축 설명과 겹치지 않게
    fig.colorbar(im, ax=axes.ravel().tolist(), label="탑승가능확률 (배차 중앙 시나리오)", shrink=0.8)
    fig.suptitle("9401 서울 도심 구간 출퇴근 시간 탑승가능확률", fontsize=14)
    _save(fig, "F1BoardingProbabilityHeatmap", "출퇴근 시간대 정류장x시간대 탑승가능확률 히트맵(요일별)")

    # F2 정류장 순서별 확률: 출퇴근 시간 중 평일 평균 확률이 가장 낮은 시간 하나를 골라 요일별로 비교
    skt = scored[scored["has_skt"] & scored["hour"].isin(COMMUTE_HOURS)]
    worst_hour = int(skt.groupby("hour")["p_board_mid"].mean().idxmin())
    fig, ax = plt.subplots(figsize=(9, 4.5))
    d = skt[skt["hour"] == worst_hour].sort_values("seq")
    ax.fill_between(d["seq"], d["p_board_low"], d["p_board_high"], alpha=0.12, color="#822727",
                    label="평일 배차 범위")
    lines = [(DOW_KR[k], sc) for k, sc in by_dow.items()] or (
        [("평일", scored)] + ([("주말", scored_we)] if scored_we is not None else []))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(lines)))
    for (name, sc), col in zip(lines, colors):
        w = sc[sc["has_skt"] & (sc["hour"] == worst_hour)].sort_values("seq")
        ls = "--" if name in ("토", "일·공휴일", "주말") else "-"
        ax.plot(w["seq"], w["p_board_mid"], marker="o", ls=ls, color=col, label=name)
    ax.set_xticks(sorted(skt["seq"].unique()))
    ax.set_xlabel("정류장 순번(seq, 서울 도심 구간)")
    ax.set_ylabel("탑승가능확률")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=4)
    ax.set_title(f"{worst_hour}시 정류장 순서별 탑승가능확률 (요일별, 점선 = 주말)")
    _save(fig, "F2PeakProfileAlongRoute", f"가장 위험한 출퇴근 시간({worst_hour}시)의 정류장 순서별 탑승가능확률(요일별)")

    # F3 버스가 찰수록 승차가 줄어드는가 (칸 수 5개 미만 구간은 흐리게)
    sat = r["saturation"]
    fig, ax = plt.subplots(figsize=(7, 4))
    few = sat["n_cells"] < 5
    bars = ax.bar(sat["load_bin"], sat["rel_capture_median"], color=np.where(few, "#cbd5e0", "#718096"))
    ax.axhline(1, color="k", lw=0.8, ls="--")
    for bar, n in zip(bars, sat["n_cells"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"n={n}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("버스 도착 시 재차율 (승차 전 재차/좌석)")
    ax.set_ylabel("실제 승차 / 예측 수요 (중앙값, 1 = 다 탐)")
    ax.set_title("버스가 찰수록 예측보다 덜 타는가 (흐린 막대 = 칸 수 5개 미만)")
    _save(fig, "F3SaturationCurve", "재차율 구간별 실제 승차 / 예측 수요")

    # F4 수요신호 종류별 예측 오차: 무작위 5-fold vs 정류장 제외(LOSO)
    # 시간대별 분담률(hourly)은 분담률 계산에 9401 승차(정답)가 들어가 있어 그래프에서 제외 (표에는 참고용으로 남김)
    cv = r["cv"][r["cv"]["signal_variant"] != "flow_x_share_hourly"]
    variants = list(dict.fromkeys(cv["signal_variant"]))
    types = list(dict.fromkeys(cv["cv_type"])) if "cv_type" in cv else [None]
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / len(types)
    x = np.arange(len(variants))
    for i, t in enumerate(types):
        sub = cv if t is None else cv[cv["cv_type"] == t]
        vals = sub.set_index("signal_variant").reindex(variants)["mae_daily_boarding"]
        ax.bar(x + (i - (len(types) - 1) / 2) * width, vals, width, label=t or "CV")
    ax.set_xticks(x, variants, rotation=15)
    ax.set_ylabel("교차검증 MAE (하루 평균 승차, 명)")
    ax.legend(fontsize=8)
    ax.set_title("수요신호별 예측 오차 (정류장 제외 = 처음 보는 정류장 맞히기)")
    _save(fig, "F4CvAblation", "신호 변형별 교차검증 오차(무작위 5-fold / 정류장 제외)")

    # F5 설정을 바꿔도 순위가 유지되는가 (y축 0.8~1.0)
    sens = r["sensitivity"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for v, d in sens.groupby("signal_variant"):
        ax.plot(d["radius_m"].astype(str) + "/" + d["tau"].astype(str), d["spearman_vs_ref"], marker="o", label=v)
    ax.set_xlabel("반경(m)/tau")
    ax.set_ylabel("기준 대비 Spearman")
    ax.set_ylim(min(0.8, float(sens["spearman_vs_ref"].min()) - 0.02), 1.005)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8)
    ax.set_title("설정을 바꿔도 위험 순위가 유지되는가")
    _save(fig, "F5Sensitivity", "파라미터 민감도")


# %%
def write_manifest(out_dir: Path, manifest: list) -> None:
    """반출 신청서에 붙일 파일 목록(ExportManifest.md)."""
    lines = [
        "# 반출 결과물 목록 (ExportManifest)",
        "",
        "모든 파일은 분석 최종 결과(집계표·통계량·모델 성능지표·변수 중요도·시각화)이며,",
        "원자료(SKT 유동인구, 도로공사 통행지표) 및 행 단위 분석용 데이터셋은 포함하지 않습니다.",
        "",
        "- 유동인구·승하차 원값 컬럼은 포함하지 않음(코드에서 자동 검사)",
        "- 확률 5%p, 대기시간 1분, 인원 10명 단위 반올림",
        f"- 표 1개당 최대 {MAX_ROWS}행",
        "",
        "| 구분 | 파일 | 내용 | 규모 |",
        "|---|---|---|---|",
    ]
    lines += [f"| {k} | {p} | {d} | {n} |" for k, p, d, n in manifest]
    (out_dir / "ExportManifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_all(r: dict, out_dir: Path) -> None:
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    # 이전 실행에서 남은 표/그림 지우기 (이름이 바뀐 옛 파일이 반출에 섞이지 않게)
    for old in list((out_dir / "tables").glob("*.csv")) + list((out_dir / "figures").glob("*.png")):
        old.unlink()
    manifest: list = []
    for name, (df, desc, index) in build_tables(r).items():
        _save_table(out_dir, name, df, manifest, desc, index=index)
    try:
        setup_font()
        build_figures(r, out_dir, manifest)
    except ImportError:
        print("⚠ matplotlib 없음 -> 그림 생략, 표만 반출")
    write_manifest(out_dir, manifest)
    print(f"반출 패키지 -> {out_dir} ({len(manifest)}개 파일)")
