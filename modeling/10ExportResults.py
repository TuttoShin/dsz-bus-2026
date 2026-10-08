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
    "flow_pop", "flow_weekday", "flow_weekend", "signal", "n_cells_matched", "boarding", "alighting", "net", "daily_boarding",
    "load_lower", "load_before_board", "cum_net", "route_max_load_hour", "latent_daily", "demand_pred", "unmet_daily",
    "표준버스정류장ID", "X_COORD", "Y_COORD", "utmk_x", "utmk_y",
}
MAX_ROWS = 100
PIVOT_HOURS = list(range(24))  # 24시간 전체


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
def build_tables(r: dict) -> dict[str, tuple[pd.DataFrame, str, bool]]:
    """반출할 표 12개를 만든다. {파일명: (표, 설명, 인덱스 저장 여부)}"""
    scored: pd.DataFrame = r["model_scored"]
    order = scored.drop_duplicates("seq").sort_values("seq")[["seq", "stop_label", "segment", "prob_basis"]]
    labels = [f"{row.seq:02d} {row.stop_label}" for row in order.itertuples()]
    label_map = dict(zip(order["seq"], labels))

    pivot_src = scored[scored["hour"].isin(PIVOT_HOURS)].copy()
    pivot_src["정류장"] = pivot_src["seq"].map(label_map)
    p_pivot = pivot_src.pivot_table(index="정류장", columns="hour", values="p_board_mid", aggfunc="first")
    p_pivot = p_pivot.reindex(labels).apply(lambda c: _round_to(c, 0.05))
    p_pivot.insert(0, "구간", order.set_index("seq")["segment"].to_numpy())
    kec_stop = r["kec_stop"].set_index("seq")
    p_pivot.insert(1, "주변도로혼잡지수", _round_to(order["seq"].map(kec_stop["stop_congestion_index"]), 1))

    stop_kec = r["kec_stop"].copy()
    stop_kec.insert(0, "정류장", stop_kec["seq"].map(label_map))
    stop_kec = stop_kec.drop(columns=["seq", "stop_name"]).round(3)
    g_pivot = pivot_src.pivot_table(index="정류장", columns="hour", values="grade", aggfunc="first").reindex(labels)

    skt = scored[scored["has_skt"]]
    worst = skt.nsmallest(10, "p_board_mid")
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

    by_hour = skt.groupby("hour").agg(latent=("latent_daily", "sum"), unmet=("unmet_daily", "sum"))
    unmet_tbl = pd.DataFrame({
        "시간대": by_hour.index,
        "추정_미승차_일평균_명(10명단위)": _round_to(by_hour["unmet"], 10),
        "미승차_비율_%": _round_to(100 * by_hour["unmet"] / by_hour["latent"].where(by_hour["latent"] > 0), 1),
    })

    coef = r["model"].coef_table().copy()
    coef["term"] = coef["term"].str.replace(r"seq(\d+)", lambda m: label_map.get(int(m.group(1)), m.group(0)), regex=True)
    coef[["coef", "multiplier"]] = coef[["coef", "multiplier"]].round(4)

    cfg = r["cfg"]
    config_tbl = pd.DataFrame(
        [(k, v) for k, v in {**vars(cfg), **r["diag"], **r.get("run_params", {})}.items()],
        columns=["항목", "값"],
    )
    config_tbl["값"] = config_tbl["값"].astype(str)

    tables = {
        "01BoardingProbabilityPivot": (p_pivot, "정류장x시간대 탑승가능확률(배차 중앙 시나리오, 5%p 반올림). 서비스 입력값", True),
        "02GradePivot": (g_pivot, "정류장x시간대 등급(여유/보통/위험)", True),
        "03WorstCellsTop10": (worst_tbl, "탑승가능확률 최하위 10개 정류장·시간대와 확률 범위·추가 대기시간", False),
        "04UnmetDemandByHour": (unmet_tbl, "서울 도심 구간 시간대별 추정 미승차 수요(구간 합계, 10명 단위)", False),
        "05ModelCoefficients": (coef, "수요 복원 Poisson GLM 계수(정류장·시간대 효과, multiplier=exp(coef))", False),
        "06CvAblation": (r["cv"].round(4), "5-fold CV 성능: SKT 유동인구 신호 유무·변형별 비교(데이터 기여도)", False),
        "07Sensitivity": (r["sensitivity"].round(4), "반경x tau x 신호 변형 민감도: 기준 대비 순위 상관·top10 겹침", False),
        "08SaturationCurve": (r["saturation"].round(4), "버스 도착 시 재차율 구간별 실제 승차 / 예측 수요 (1보다 작으면 자리가 없어 덜 탐)", False),
        "09RuleVsModel": (r["rule_vs_model"].round(4), "규칙 기반 지수(07) vs 모델 확률(09) 순위 일치도", False),
        "10RunConfig": (config_tbl, "실행 파라미터·진단값(데이터 없음)", False),
        "11RouteCongestionIndex": (r["kec_table"].round(3), "9401 경유 경부선 구간 혼잡 노출 지수(전체 + 시군구별, 도로 길이 가중평균)", False),
        "12StopCongestionIndex": (stop_kec, "정류장별 주변 도로 혼잡 지수(정류장이 있는 행정동 도로, 도로 길이 가중평균)", False),
    }
    return tables


# %%
def build_figures(r: dict, out_dir: Path, manifest: list) -> None:
    """반출할 그래프 5개(F1~F5)를 PNG로 저장한다."""
    import matplotlib.pyplot as plt

    scored = r["model_scored"]
    fig_dir = out_dir / "figures"

    def _save(fig, name, desc):
        fig.tight_layout()
        fig.savefig(fig_dir / f"{name}.png", dpi=160)
        plt.close(fig)
        manifest.append(("그림", f"figures/{name}.png", desc, ""))

    # F1 히트맵
    order = scored.drop_duplicates("seq").sort_values("seq")
    piv = scored[scored["hour"].isin(PIVOT_HOURS)].pivot_table(index="seq", columns="hour", values="p_board_mid")
    piv = piv.reindex(order["seq"])
    fig, ax = plt.subplots(figsize=(11, max(4, len(piv) * 0.4)))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(piv.columns)), piv.columns)
    ylabels = [f"{s:02d} {n}" for s, n in zip(order["seq"], order["stop_label"])]
    ax.set_yticks(range(len(piv)), ylabels, fontsize=9)
    ax.set_xlabel("시간대")
    ax.set_title("9401 서울 도심 구간 정류장x시간대 탑승가능확률 (배차 중앙 시나리오)")
    fig.colorbar(im, ax=ax, label="탑승가능확률")
    _save(fig, "F1BoardingProbabilityHeatmap", "정류장x시간대 탑승가능확률 히트맵")

    # F2 정류장 순서별 확률: 평균 확률이 가장 낮은 시간대 3개를 자동으로 골라서 그림
    skt = scored[scored["has_skt"]]
    worst_hours = skt.groupby("hour")["p_board_mid"].mean().nsmallest(3).index
    fig, ax = plt.subplots(figsize=(9, 4))
    for h, c in zip(sorted(worst_hours), ["#d69e2e", "#c05621", "#822727"]):
        d = skt[skt["hour"] == h].sort_values("seq")
        ax.fill_between(d["seq"], d["p_board_low"], d["p_board_high"], alpha=0.2, color=c)
        ax.plot(d["seq"], d["p_board_mid"], marker="o", color=c, label=f"{h}시 (음영=배차 3~7분 범위)")
    ax.set_xticks(sorted(skt["seq"].unique()))
    ax.set_xlabel("정류장 순번(seq, 서울 도심 구간)")
    ax.set_ylabel("탑승가능확률")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("탑승가능확률이 가장 낮은 3개 시간대의 정류장 순서별 확률")
    _save(fig, "F2PeakProfileAlongRoute", "평균 확률 최저 3개 시간대의 정류장 순서별 탑승가능확률(범위 포함)")

    # F3 버스가 찰수록 승차가 줄어드는가
    sat = r["saturation"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(sat["load_bin"], sat["rel_capture_median"], color="#718096")
    ax.axhline(1, color="k", lw=0.8, ls="--")
    for i, n in enumerate(sat["n_cells"]):
        ax.text(i, 0.02, f"n={n}", ha="center", fontsize=8, color="white")
    ax.set_xlabel("버스 도착 시 재차율 (승차 전 재차/좌석)")
    ax.set_ylabel("실제 승차 / 예측 수요 (1 = 다 탐)")
    ax.set_title("버스가 찰수록 주변 인구 대비 승차가 줄어드는가 (수요 잘림)")
    _save(fig, "F3SaturationCurve", "재차율 구간별 상대 승차율")

    # F4 수요신호 종류별 예측 오차
    cv = r["cv"]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(cv["signal_variant"], cv["mae_daily_boarding"], color=["#a0aec0" if s == "no_skt" else "#dd6b20" for s in cv["signal_variant"]])
    ax.set_ylabel("CV MAE (일평균 승차, 명)")
    ax.set_title("SKT 유동인구 신호 유무에 따른 수요 예측 오차")
    _save(fig, "F4CvAblation", "신호 변형별 교차검증 오차")

    # F5 설정을 바꿔도 순위가 유지되는가
    sens = r["sensitivity"]
    fig, ax = plt.subplots(figsize=(8, 4))
    for v, d in sens.groupby("signal_variant"):
        ax.plot(d["radius_m"].astype(str) + "/" + d["tau"].astype(str), d["spearman_vs_ref"], marker="o", label=v)
    ax.set_xlabel("반경(m)/tau")
    ax.set_ylabel("기준 대비 Spearman")
    ax.set_ylim(min(0.0, sens["spearman_vs_ref"].min()) , 1.02)
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
