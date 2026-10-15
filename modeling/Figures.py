"""
Figures.py

10번 반출 패키지의 그림을 전부 여기서 만든다. 집계된 값(비율·확률·평균·지수)만 그린다.
버스별·사람별 원자료는 그림에도 들어가지 않는다 (일별 이용량도 평균=100 지수로).

색 규칙 (한 그림 안에서도, 그림끼리도 같은 뜻 = 같은 색)
  범주(요일·평일/주말 등) : 정해진 순서의 8색 (색맹 검증 통과 순서), 순서를 바꾸지 않음
  크기(재차율·대기 시간 등): 한 가지 색의 밝음 -> 진함
  탑승가능확률            : 빨강(위험) <- 회색(경계 0.8) -> 파랑(여유)  (F1만 기존 빨강-노랑-초록)
  격자·축은 옅은 실선, 막대·선은 가늘게, 숫자는 꼭 필요한 곳에만

그림 목록 (번호 = 파일 이름)
  E1~E5  탐색(EDA): 교통카드 원자료를 모형 없이 집계한 것
  F0     한눈에 보는 요약
  F1~F2, F8~F12  탑승가능확률과 미승차
  F3~F5  검증
  F6, F13~F16    서비스
  F7     실측 버스 몰림
  S13~S19 장표용 (하늘색 박스): 파일 이름의 숫자 = 장표 쪽 번호
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 색·글꼴
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
NEUTRAL = "#f0efec"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORANGE_RAMP = ["#fde6db", "#f8c1a8", "#f39a74", "#eb6834", "#c44f20", "#943a16", "#62260e"]
PROB_DIVERGING = ["#a8282a", "#e34948", "#f3a8a4", NEUTRAL, "#9ec5f4", "#3987e5", "#184f95"]
PROB_CENTER = 0.8  # 보통(0.7~0.9)의 가운데. 이보다 낮으면 빨강 쪽

DOW_ORDER = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
DOW_KR = {"mon": "월", "tue": "화", "wed": "수", "thu": "목", "fri": "금", "sat": "토", "sun": "일·공휴일"}
DOW_COLOR = dict(zip(DOW_ORDER, SERIES))
DAY_COLOR = {"평일": SERIES[0], "주말": SERIES[1]}
SEG_ORDER = ["성남_상행", "서울_도심루프", "성남_하행"]
SEG_KR = {"성남_상행": "성남 → 서울 (상행)", "서울_도심루프": "서울 도심 루프", "성남_하행": "서울 → 성남 (하행)"}
COMMUTE_HOURS = list(range(6, 10)) + list(range(16, 24))
MIN_BUSES_FOR_WAIT = 2  # F14: 시간당 버스가 이보다 적은 칸은 대기 시간을 그리지 않음


def setup_style() -> None:
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
    plt.rcParams.update({
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False, "legend.fontsize": 8, "legend.labelcolor": INK2,
        "lines.linewidth": 2, "lines.markersize": 5, "text.color": INK,
    })


def _cmap(colors: list, name: str):
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list(name, colors)
    cm.set_bad(SURFACE)  # 운행 없음 = 빈칸
    return cm


def _prob_norm():
    from matplotlib.colors import TwoSlopeNorm
    return TwoSlopeNorm(vmin=0.0, vcenter=PROB_CENTER, vmax=1.0)


def _note(fig, text: str) -> None:
    fig.text(0.01, -0.02, text, fontsize=7.5, color=MUTED, ha="left", va="top")


def _stop_names(scored: pd.DataFrame) -> dict:
    o = scored.drop_duplicates("seq").sort_values("seq")
    return {r.seq: f"{r.seq:02d} {r.stop_label}" for r in o.itertuples()}


def _service_hours(scored_list: list) -> list:
    hours = set()
    for sc in scored_list:
        hours |= set(sc.loc[sc["n_buses"] > 0, "hour"].astype(int))
    return sorted(hours, key=lambda h: (h < 4, h))  # 자정 넘은 운행은 23시 뒤에


def _hour_ticks(ax, hours: list, axis: str = "x") -> None:
    labels = [str(h) for h in hours]
    if axis == "x":
        ax.set_xticks(range(len(hours)), labels)
    else:
        ax.set_yticks(range(len(hours)), labels)


def _prob_heatmap(ax, piv: pd.DataFrame, cmap, norm):
    im = ax.imshow(np.ma.masked_invalid(piv.to_numpy(float)), aspect="auto", cmap=cmap, norm=norm,
                   interpolation="nearest")
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return im


# ================================================================ E. 탐색(EDA)
def e1_daily_ridership(ev: pd.DataFrame, holidays: list, save) -> None:
    """일별 9401 승차 (기간 평균 = 100 지수). 평일 / 주말·공휴일 색 구분."""
    import matplotlib.pyplot as plt
    day = ev.groupby("date").agg(board=("boarding", "sum"), daytype=("daytype", "first"), dow=("dow", "first"))
    day = day.sort_index()
    idx = day["board"] / day["board"].mean() * 100
    d = pd.to_datetime(day.index, format="%Y%m%d")
    fig, ax = plt.subplots(figsize=(11, 3.6))
    colors = [DAY_COLOR["주말"] if t == "weekend" else DAY_COLOR["평일"] for t in day["daytype"]]
    ax.bar(range(len(day)), idx, color=colors, width=0.8)
    ax.axhline(100, color=AXIS, lw=0.8)
    ax.set_xticks(range(len(day)), [f"{x.month}/{x.day}\n{'월화수목금토일'[x.dayofweek]}" for x in d], fontsize=7)
    ax.set_ylabel("일별 승차 지수 (기간 평균 = 100)")
    ax.set_xlim(-0.6, len(day) - 0.4)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=DAY_COLOR["평일"], label="평일"), Patch(color=DAY_COLOR["주말"], label="주말·공휴일")],
              loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(axis="x", visible=False)
    wd, we = idx[day["daytype"] == "weekday"].mean(), idx[day["daytype"] == "weekend"].mean()
    ax.set_title("9401 일별 이용량 지수 (평일 vs 주말·공휴일)")
    _note(fig, f"교통카드 이용내역 9401 전체 정류장 승차 합계를 기간 평균 = 100으로 나타냄. 공휴일({', '.join(holidays)})은 주말로. "
               f"주말·공휴일 평균 = 평일 평균의 {we / wd:.0%}")
    save(fig, "E1DailyRidership", "[EDA] 일별 9401 승차 지수(기간 평균=100), 평일 vs 주말·공휴일")


def e2_hourly_by_segment(ev: pd.DataFrame, stops: pd.DataFrame, save) -> None:
    """구간별 시간대 승차·하차 비중 (평일 vs 주말). 아침엔 성남에서 타고, 저녁엔 서울에서 탄다."""
    import matplotlib.pyplot as plt
    seg = stops.drop_duplicates("seq").set_index("seq")["segment"]
    e = ev.assign(segment=ev["seq"].map(seg))
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True, sharey="row")
    for c, sg in enumerate(SEG_ORDER):
        for r, (col, kr) in enumerate((("boarding", "승차"), ("alighting", "하차"))):
            ax = axes[r][c]
            for dt, name in (("weekday", "평일"), ("weekend", "주말")):
                s = e[(e["segment"] == sg) & (e["daytype"] == dt)].groupby("hour")[col].sum().reindex(range(24), fill_value=0)
                share = s / max(s.sum(), 1) * 100
                ax.plot(range(24), share, color=DAY_COLOR[name], label=name)
            ax.set_xticks(range(0, 24, 3))
            if r == 0:
                ax.set_title(SEG_KR[sg])
            if c == 0:
                ax.set_ylabel(f"{kr} 비중 (%, 하루 = 100)")
            if r == 1:
                ax.set_xlabel("시간대")
    axes[0][0].legend(loc="upper left")
    fig.suptitle("구간별 시간대 승하차 비중", fontsize=12, color=INK)
    fig.tight_layout()
    save(fig, "E2HourlyBySegment", "[EDA] 구간(성남 상행 / 서울 도심 / 성남 하행)별 시간대 승차·하차 비중, 평일 vs 주말")


def e3_od_matrix(od: pd.DataFrame, stops: pd.DataFrame, save) -> None:
    """승차 정류장 x 하차 정류장 이동 비중 (%). 하차 기록이 있는 승객만."""
    import matplotlib.pyplot as plt
    o = od.copy()
    tot = o.to_numpy().sum()
    share = o / max(tot, 1) * 100
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(np.ma.masked_less_equal(share.to_numpy(float), 0), cmap=_cmap(BLUE_RAMP, "od"), aspect="equal",
                   interpolation="nearest")
    ax.grid(False)
    seqs = list(share.index)
    seg = stops.drop_duplicates("seq").set_index("seq")["segment"]
    bounds = [i for i in range(1, len(seqs)) if seg.get(seqs[i]) != seg.get(seqs[i - 1])]
    for b in bounds:
        ax.axhline(b - 0.5, color=AXIS, lw=0.8)
        ax.axvline(b - 0.5, color=AXIS, lw=0.8)
    step = max(1, len(seqs) // 15)
    ax.set_xticks(range(0, len(seqs), step), [seqs[i] for i in range(0, len(seqs), step)])
    ax.set_yticks(range(0, len(seqs), step), [seqs[i] for i in range(0, len(seqs), step)])
    ax.set_xlabel("하차 정류장 (seq)")
    ax.set_ylabel("승차 정류장 (seq)")
    fig.colorbar(im, ax=ax, shrink=0.75, label="전체 이동 중 비중 (%)")
    ax.set_title("승차 정류장 x 하차 정류장 이동 비중 (구분선 = 구간 경계)")
    _note(fig, "하차 태그가 있는 승객 기준. 빈칸 = 이동 없음. 대각선 위쪽만 값이 있음(노선 순서대로 이동).")
    save(fig, "E3OriginDestination", "[EDA] 승차 정류장 x 하차 정류장 이동 비중(%), 구간 경계 표시")


def e4_load_along_route(ev: pd.DataFrame, stops: pd.DataFrame, seats: float, save) -> None:
    """정류장 순서 x 시간대 평균 재차율 (평일). 버스가 어디서부터 차는지."""
    import matplotlib.pyplot as plt
    e = ev[ev["daytype"] == "weekday"]
    piv = (e.groupby(["hour", "seq"])["load_before"].mean() / seats).unstack()
    hours = [h for h in sorted(piv.index, key=lambda h: (h < 4, h)) if e[e["hour"] == h]["date"].nunique() > 0]
    piv = piv.reindex(hours)
    seqs = list(piv.columns)
    fig, ax = plt.subplots(figsize=(13, 5.2))
    im = ax.imshow(piv.to_numpy(float), aspect="auto", cmap=_cmap(ORANGE_RAMP, "load"), vmin=0, vmax=1,
                   interpolation="nearest")
    ax.grid(False)
    seg = stops.drop_duplicates("seq").set_index("seq")["segment"]
    for i in range(1, len(seqs)):
        if seg.get(seqs[i]) != seg.get(seqs[i - 1]):
            ax.axvline(i - 0.5, color=INK2, lw=1)
    starts = [0] + [i for i in range(1, len(seqs)) if seg.get(seqs[i]) != seg.get(seqs[i - 1])] + [len(seqs)]
    for a, b in zip(starts[:-1], starts[1:]):
        ax.text((a + b - 1) / 2, -0.9, SEG_KR.get(seg.get(seqs[a]), ""), ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_xticks(range(len(seqs)), seqs, fontsize=6)
    _hour_ticks(ax, hours, "y")
    ax.set_xlabel("정류장 순번 (seq)")
    ax.set_ylabel("시간대")
    fig.colorbar(im, ax=ax, shrink=0.85, label="버스 1대당 도착 시 재차율 (재차 / 41석)")
    ax.set_title("[평일] 정류장 순서 x 시간대 평균 재차율", pad=22)
    save(fig, "E4LoadAlongRoute", "[EDA] 평일 정류장 순서 x 시간대 버스 1대당 평균 도착 시 재차율")


def e5_headway_distribution(ev: pd.DataFrame, seoul_seqs: list, save) -> None:
    """실제 배차 공백 분포 (평일 서울 구간, 출근 / 퇴근). 몰림 = 1분 이내 연속 도착 + 긴 공백."""
    import matplotlib.pyplot as plt
    e = ev[(ev["daytype"] == "weekday") & ev["seq"].isin(seoul_seqs)].dropna(subset=["gap_min"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, (name, hrs) in zip(axes, (("출근 6~9시", range(6, 10)), ("퇴근 16~23시", range(16, 24)))):
        g = e[e["hour"].isin(hrs)]["gap_min"]
        g = g[g <= 30]
        if g.empty:
            continue
        bins = np.arange(0, 30.5, 0.5)
        w = np.ones(len(g)) / len(g) * 100
        ax.hist(g, bins=bins, weights=w, color=SERIES[0], rwidth=0.85)
        m = g.mean()
        ax.axvline(m, color=INK, lw=1)
        ax.text(m + 0.3, ax.get_ylim()[1] * 0.92, f"평균 {m:.1f}분", fontsize=8, color=INK)
        ax.set_title(name)
        ax.text(0.98, 0.75, f"1분 이내 연속 도착 {float((g <= 1).mean()):.0%}\n평균의 2배 넘는 공백 {float((g > 2 * m).mean()):.0%}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)
        ax.set_xlabel("앞 버스와의 실제 간격 (분)")
    axes[0].set_ylabel("버스 비중 (%)")
    fig.suptitle("[평일 서울 도심] 실제 배차 간격 분포", fontsize=12, color=INK)
    fig.tight_layout()
    save(fig, "E5HeadwayDistribution", "[EDA] 평일 서울 도심 구간 실제 배차 간격 분포(출근/퇴근), 연속 도착·긴 공백 비율")


# ================================================================ F. 결과
def f0_summary(r: dict, save) -> None:
    """한눈에 보는 요약 (숫자 타일 + 가장 위험한 곳)."""
    import matplotlib.pyplot as plt
    sc, we = r["model_scored"], r["model_scored_weekend"]
    s = sc[sc["has_skt"]]
    names = _stop_names(sc)
    worst = s.nsmallest(1, "p_board_mid").iloc[0]
    burden = r["wait_burden"]
    tiles = [
        ("평일 위험 칸", f"{int((s['grade'] == '위험').sum())}칸", f"서울 도심 {s['seq'].nunique()}개 정류장 x 운행 시간대 중 탑승확률 0.7 미만"),
        ("가장 위험한 곳", f"{worst['p_board_mid']:.2f}", f"{names[worst['seq']]} {int(worst['hour'])}시 평일 탑승가능확률"),
        ("주말 위험 칸", f"{int((we[we['has_skt']]['grade'] == '위험').sum())}칸", "같은 기준, 주말·공휴일"),
        ("평일 하루 초과 대기", f"{burden['excess_wait_person_hours_day'].sum():,.0f}인·시간", "만차로 버스를 보내며 더 기다린 시간 합계"),
        ("연간 환산", f"{burden['excess_wait_person_hours_year'].sum() / 1000:,.0f}천 인·시간", f"평일 {r['service_cfg'].get('workdays_per_year', 250)}일 기준"),
        ("실측 몰림 정도", f"{float(np.nanmean(s['gap_cv'])):.2f}", "배차 공백 표준편차 / 평균 (0 = 일정, 1 = 무작위)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 4.6))
    for ax, (label, value, sub) in zip(axes.ravel(), tiles):
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, color="#f4f3ef", zorder=0))
        ax.text(0.06, 0.78, label, fontsize=10, color=INK2, transform=ax.transAxes)
        ax.text(0.06, 0.40, value, fontsize=24, color=INK, transform=ax.transAxes, weight="bold")
        ax.text(0.06, 0.12, sub, fontsize=7.5, color=MUTED, transform=ax.transAxes, wrap=True)
    fig.suptitle("9401 서울 도심 구간 탑승 실패 요약 (교통카드 + SKT 유동인구)", fontsize=13, color=INK)
    fig.tight_layout()
    save(fig, "F0SummaryBoard", "핵심 숫자 요약(위험 칸 수, 최저 확률, 하루·연간 초과 대기, 실측 몰림)")


def f1_heatmap_by_dow(by_dow: dict, save) -> None:
    import matplotlib.pyplot as plt
    hours = _service_hours(list(by_dow.values()))
    # F1은 기존 빨강-노랑-초록 (발표용으로 보기 편해서). 다른 확률 그림은 색맹 안전한 빨강-파랑
    import matplotlib
    from matplotlib.colors import Normalize
    cmap = matplotlib.colormaps["RdYlGn"].copy()
    cmap.set_bad(SURFACE)
    norm = Normalize(vmin=0.0, vmax=1.0)
    fig, axes = plt.subplots(2, 4, figsize=(max(5.2, 0.38 * len(hours)) * 4 + 1.5, 9.4), squeeze=False)
    for i, k in enumerate(DOW_ORDER):
        ax = axes[i // 4][i % 4]
        sc = by_dow[k]
        names = _stop_names(sc)
        piv = sc.pivot_table(index="seq", columns="hour", values="p_board_mid", dropna=False).reindex(columns=hours)
        im = _prob_heatmap(ax, piv, cmap, norm)
        _hour_ticks(ax, hours)
        ax.set_yticks(range(len(piv)), [names[s] for s in piv.index] if i % 4 == 0 else [], fontsize=8)
        ax.set_title(DOW_KR[k])
        ax.set_xlabel("시간대")
    axes[1][3].axis("off")
    fig.subplots_adjust(hspace=0.35, wspace=0.06)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="탑승가능확률")
    cb.set_ticks([0, 0.5, 0.7, 0.8, 0.9, 1.0])
    fig.suptitle(f"요일별 정류장 x 시간대 탑승가능확률 (운행 {hours[0]}시~{hours[-1]}시, 빨강 = 위험, 초록 = 여유, 빈칸 = 운행 없음)",
                 fontsize=13, color=INK)
    save(fig, "F1BoardingProbabilityHeatmap", "요일별 정류장 x 운행 시간 전체 탑승가능확률 히트맵(빨강 위험 ~ 초록 여유)")


def f2_profile_worst_hour(sc: pd.DataFrame, by_dow: dict, save) -> int:
    import matplotlib.pyplot as plt
    s = sc[sc["has_skt"] & sc["hour"].isin(COMMUTE_HOURS)]
    worst_hour = int(s.groupby("hour")["p_board_mid"].mean().idxmin())
    names = _stop_names(sc)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    d = s[s["hour"] == worst_hour].sort_values("seq")
    x = np.arange(len(d))
    ax.fill_between(x, d["p_board_low"], d["p_board_high"], color=SERIES[0], alpha=0.12, lw=0,
                    label="평일 날마다 차이 (하위~상위 25%)")
    for k in DOW_ORDER:
        w = by_dow[k]
        w = w[w["has_skt"] & (w["hour"] == worst_hour)].set_index("seq").reindex(d["seq"])
        ax.plot(x, w["p_board_mid"], marker="o", color=DOW_COLOR[k], label=DOW_KR[k], lw=1.6)
    ax.axhline(0.7, color=AXIS, lw=0.8)
    ax.text(len(x) - 0.5, 0.71, "위험 기준 0.7", fontsize=7.5, color=MUTED, ha="right")
    ax.set_xticks(x, [names[q] for q in d["seq"]], rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("탑승가능확률")
    ax.legend(ncol=4, fontsize=8, loc="lower left")
    ax.set_title(f"{worst_hour}시 정류장 순서별 탑승가능확률 (요일별)")
    save(fig, "F2PeakProfileAlongRoute", f"가장 위험한 출퇴근 시간({worst_hour}시)의 정류장 순서별 탑승가능확률(요일별 + 평일 날마다 차이)")
    return worst_hour


def f8_hourly_by_dow(by_dow: dict, save) -> None:
    """시간대별 평균·최저 탑승확률을 요일별 선으로 (두 지표는 다른 그림 칸에, 축 하나씩)."""
    import matplotlib.pyplot as plt
    hours = _service_hours(list(by_dow.values()))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for ax, (stat, title) in zip(axes, (("mean", "서울 도심 정류장 평균"), ("min", "가장 위험한 정류장"))):
        for k in DOW_ORDER:
            s = by_dow[k]
            g = s[s["has_skt"]].groupby("hour")["p_board_mid"].agg(stat).reindex(hours)
            ax.plot(range(len(hours)), g, color=DOW_COLOR[k], label=DOW_KR[k], lw=1.6, marker="o", ms=3)
        ax.axhline(0.7, color=AXIS, lw=0.8)
        _hour_ticks(ax, hours)
        ax.set_xlabel("시간대")
        ax.set_title(title)
    axes[0].set_ylabel("탑승가능확률")
    axes[0].set_ylim(0, 1.05)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle("요일별 시간대 탑승가능확률", fontsize=12, color=INK)
    fig.tight_layout()
    save(fig, "F8HourlyProbabilityByDow", "요일별 시간대 탑승가능확률(서울 도심 평균 / 가장 위험한 정류장)")


def f9_stop_dow_hour(by_dow: dict, save) -> None:
    """정류장마다 요일 x 시간대 확률 (작은 히트맵 여러 개)."""
    import matplotlib.pyplot as plt
    hours = _service_hours(list(by_dow.values()))
    base = by_dow["mon"]
    seqs = sorted(base.loc[base["has_skt"], "seq"].unique())
    names = _stop_names(base)
    cmap, norm = _cmap(PROB_DIVERGING, "prob"), _prob_norm()
    ncols = 4
    nrows = -(-len(seqs) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols + 1, 2.3 * nrows + 0.8), squeeze=False)
    for i, q in enumerate(seqs):
        ax = axes[i // ncols][i % ncols]
        piv = pd.DataFrame({k: by_dow[k][by_dow[k]["seq"] == q].set_index("hour")["p_board_mid"] for k in DOW_ORDER}).T
        piv = piv.reindex(columns=hours)
        im = _prob_heatmap(ax, piv, cmap, norm)
        ax.set_yticks(range(7), [DOW_KR[k][0] for k in DOW_ORDER], fontsize=7)
        step = 3
        ax.set_xticks(range(0, len(hours), step), [hours[j] for j in range(0, len(hours), step)], fontsize=7)
        ax.set_title(names[q], fontsize=9)
    for j in range(len(seqs), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.subplots_adjust(hspace=0.55, wspace=0.12)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="탑승가능확률")
    fig.suptitle("정류장별 요일 x 시간대 탑승가능확률 (빨강 = 위험, 빈칸 = 운행 없음)", fontsize=13, color=INK)
    save(fig, "F9StopDowHourGrid", "서울 도심 정류장마다 요일 x 운행 시간대 탑승가능확률(작은 히트맵)")


def f10_day_variation(sc: pd.DataFrame, save, top_n: int = 15) -> None:
    """위험한 칸의 날마다 차이: 나쁜 날(하위 25%) ~ 좋은 날(상위 25%)."""
    import matplotlib.pyplot as plt
    s = sc[sc["has_skt"]].nsmallest(top_n, "p_board_mid").iloc[::-1]
    names = _stop_names(sc)
    fig, ax = plt.subplots(figsize=(9, 0.38 * len(s) + 1.2))
    y = np.arange(len(s))
    ax.hlines(y, s["p_board_low"], s["p_board_high"], color=SERIES[0], lw=3, alpha=0.35)
    ax.plot(s["p_board_low"], y, "o", color=SERIES[1], label="나쁜 날 (하위 25%)")
    ax.plot(s["p_board_mid"], y, "o", color=INK, ms=6, label="한 달 전체")
    ax.plot(s["p_board_high"], y, "o", color=SERIES[0], label="좋은 날 (상위 25%)")
    ax.set_yticks(y, [f"{names[q]}  {int(h)}시" for q, h in zip(s["seq"], s["hour"])])
    ax.set_xlim(0, 1)
    ax.set_xlabel("탑승가능확률")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.grid(axis="y", visible=False)
    ax.set_title(f"[평일] 탑승가능확률 하위 {top_n}칸의 날짜별 변동")
    save(fig, "F10DayToDayVariation", f"평일 탑승가능확률 최하위 {top_n}칸의 날마다 차이(하위 25% ~ 상위 25%)")


def f11_bunching_by_stop(sc: pd.DataFrame, uniform: pd.DataFrame, save) -> None:
    """정류장별 실측 vs 몰림 무시(버스가 고르게 왔다면). 차이 = 몰림 때문에 잃는 탑승 기회."""
    import matplotlib.pyplot as plt
    s = sc[sc["has_skt"]]
    names = _stop_names(sc)
    am = int(s[s["hour"].isin(range(6, 10))].groupby("hour")["p_board_mid"].mean().idxmin())
    pm = int(s[s["hour"].isin(range(16, 24))].groupby("hour")["p_board_mid"].mean().idxmin())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for ax, h in zip(axes, (am, pm)):
        a = s[s["hour"] == h].set_index("seq")["p_board_mid"]
        b = uniform[uniform["has_skt"] & (uniform["hour"] == h)].set_index("seq")["p_board_mid"].reindex(a.index)
        y = np.arange(len(a))
        ax.hlines(y, a, b, color=AXIS, lw=2)
        ax.plot(b, y, "o", color=SERIES[0], label="버스가 고르게 왔다면")
        ax.plot(a, y, "o", color=SERIES[1], label="실측 (몰림 포함)")
        ax.set_yticks(y, [names[q] for q in a.index])
        ax.invert_yaxis()
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("탑승가능확률")
        ax.set_title(f"평일 {h}시")
        ax.grid(axis="y", visible=False)
    axes[0].legend(loc="lower left")
    fig.suptitle("정류장별 탑승가능확률: 실측 vs 버스가 평균 간격·평균 재차로 고르게 왔을 때", fontsize=12, color=INK)
    fig.tight_layout()
    save(fig, "F11BunchingEffectByStop", "정류장별 탑승가능확률: 실측 vs 몰림 무시(평일 출근·퇴근 가장 위험한 시간)")


def f12_unmet(sc: pd.DataFrame, we: pd.DataFrame, seats: float, save) -> None:
    import matplotlib.pyplot as plt
    hours = _service_hours([sc, we])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    width = 0.4
    x = np.arange(len(hours))
    for i, (name, s) in enumerate((("평일", sc), ("주말", we))):
        g = s[s["has_skt"]].groupby("hour").agg(latent=("latent_daily", "sum"), unmet=("unmet_daily", "sum")).reindex(hours)
        rate = 100 * g["unmet"] / g["latent"].where(g["latent"] > 0)
        axes[0].bar(x + (i - 0.5) * width, rate, width * 0.92, color=DAY_COLOR[name], label=name)
        axes[1].bar(x + (i - 0.5) * width, g["unmet"] / seats, width * 0.92, color=DAY_COLOR[name], label=name)
    for ax in axes:
        _hour_ticks(ax, hours)
        ax.set_xlabel("시간대")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("못 탄 사람 비율 (%)")
    axes[0].set_title("대기 수요 중 미승차 비율")
    axes[1].set_ylabel("필요한 추가 운행 (대/일)")
    axes[1].set_title("미승차 수요를 태우는 데 필요한 추가 운행 (41석 기준)")
    axes[0].legend()
    fig.suptitle("서울 도심 구간 시간대별 미승차 수요 (SKT 유동인구로 복원)", fontsize=12, color=INK)
    fig.tight_layout()
    save(fig, "F12UnmetDemandByHour", "시간대별 미승차 비율과 필요한 추가 운행 대수(평일 vs 주말)")


# ================================================================ 검증
def f3_saturation(sat: pd.DataFrame, save) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    few = sat["n_cells"] < 5
    bars = ax.bar(sat["load_bin"], sat["rel_capture_median"], color=np.where(few, BLUE_RAMP[1], SERIES[0]), width=0.6)
    ax.axhline(1, color=INK2, lw=0.8)
    for bar, n in zip(bars, sat["n_cells"]):
        ax.text(bar.get_x() + bar.get_width() / 2, np.nan_to_num(bar.get_height()) + 0.02, f"n={n}",
                ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xlabel("버스 도착 시 재차율 (승차 전 재차 / 41석)")
    ax.set_ylabel("실제 승차 / 예측 수요 (중앙값)")
    ax.grid(axis="x", visible=False)
    ax.set_title("도착 시 재차율 구간별 실제 승차 / 예측 수요 (흐린 막대 = 칸 5개 미만)")
    save(fig, "F3SaturationCurve", "[검증] 재차율 구간별 실제 승차 / 예측 수요")


def f4_cv(cv: pd.DataFrame, save) -> None:
    import matplotlib.pyplot as plt
    cv = cv[cv["signal_variant"] != "flow_x_share_hourly"]
    variants = list(dict.fromkeys(cv["signal_variant"]))
    types = list(dict.fromkeys(cv["cv_type"]))
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / len(types)
    x = np.arange(len(variants))
    for i, t in enumerate(types):
        vals = cv[cv["cv_type"] == t].set_index("signal_variant").reindex(variants)["mae_daily_boarding"]
        ax.bar(x + (i - (len(types) - 1) / 2) * width, vals, width * 0.92, color=SERIES[i], label=t)
    ax.set_xticks(x, variants)
    ax.set_ylabel("교차검증 MAE (하루 평균 승차, 명)")
    ax.grid(axis="x", visible=False)
    ax.legend()
    ax.set_title("[검증] 수요신호별 예측 오차 (무작위 5-fold / 정류장 제외)")
    save(fig, "F4CvAblation", "[검증] 수요신호 변형별 교차검증 오차(무작위 5-fold / 정류장 제외)")


def f5_sensitivity(sens: pd.DataFrame, save) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (v, d) in enumerate(sens.groupby("signal_variant", sort=False)):
        ax.plot(d["radius_m"].astype(int).astype(str) + "m / " + d["tau"].astype(str), d["spearman_vs_ref"],
                marker="o", color=SERIES[i], label=v, lw=1.6)
    ax.set_xlabel("반경 / tau")
    ax.set_ylabel("기준 설정 대비 순위 상관 (Spearman)")
    ax.set_ylim(min(0.8, float(sens["spearman_vs_ref"].min()) - 0.02), 1.005)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(ncol=2)
    ax.set_title("[검증] 설정별 기준 대비 위험 순위 상관·상위 10칸 일치율")
    save(fig, "F5Sensitivity", "[검증] 반경 x tau x 수요신호 민감도(순위 상관)")


def f7_measured_bunching(mh: pd.DataFrame, names: dict, save) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    for ax, dt in zip(axes, ("평일", "주말")):
        m = mh[mh["daytype"] == dt]
        seqs = sorted(m["seq"].unique())
        x = np.arange(len(seqs))
        for i, band in enumerate(["출근", "퇴근"]):
            vals = m[m["band"] == band].set_index("seq").reindex(seqs)["몰림정도_CV"]
            ax.bar(x + (i - 0.5) * 0.4, vals, 0.37, color=SERIES[i], label=band)
        ax.axhline(1, color=INK2, lw=0.8)
        ax.text(len(seqs) - 0.5, 1.02, "무작위로 오는 수준", fontsize=7.5, color=MUTED, ha="right")
        ax.set_xticks(x, [names.get(q, q) for q in seqs], rotation=35, ha="right", fontsize=7.5)
        ax.set_title(dt)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("배차 공백 CV (0 = 일정)")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.suptitle("교통카드로 잰 정류장별 버스 몰림 정도", fontsize=12, color=INK)
    fig.tight_layout()
    save(fig, "F7MeasuredBunching", "정류장별 실측 배차 공백 몰림 정도(CV), 평일/주말 x 출근/퇴근")


# ================================================================ 서비스
def f6_service_panels(r: dict, m09, names: dict, save) -> None:
    """요일·시간대 여러 개: 정류장마다 '그대로 기다리기' vs '다른 정류장으로 옮겨 타기'를 구성 요소별로 쌓아서."""
    import matplotlib.pyplot as plt
    recs = r["service_rec_by_day"]
    cases = []
    for key, kr in (("weekday", "평일"), ("sat", "토"), ("sun", "일·공휴일")):
        rec = recs.get(key)
        if rec is None or rec.empty:
            continue
        mean_wait = rec.groupby("slot")["wait_here_min"].mean()
        slot_hour = rec.drop_duplicates("slot").set_index("slot")["hour"]
        for band, hrs in (("출근", range(6, 10)), ("퇴근", range(16, 24))):
            mw = mean_wait[slot_hour.reindex(mean_wait.index).isin(hrs).to_numpy()]
            if len(mw):
                cases.append((key, kr, int(mw.idxmax()), band))
    if not cases:
        return
    C_MOVE, C_WAIT, C_RIDE = SERIES[2], SERIES[1], SERIES[0]
    ncols = 2
    nrows = -(-len(cases) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.9 * nrows), squeeze=False, sharey=True)
    for i, (key, kr, sl, band) in enumerate(cases):
        ax = axes[i // ncols][i % ncols]
        rh = recs[key][recs[key]["slot"] == sl].sort_values("seq")
        h = rh["time"].iloc[0] if len(rh) else ""
        x = np.arange(len(rh))
        ride_here = rh["total_here_min"] - rh["wait_here_min"]
        ax.bar(x - 0.2, rh["wait_here_min"], 0.37, color=C_WAIT)
        ax.bar(x - 0.2, ride_here, 0.37, bottom=rh["wait_here_min"], color=C_RIDE)
        alt = rh["saving_min"] > 0
        acc = rh["alt_access_min"].where(alt)
        wt = rh["alt_wait_min"].where(alt)
        rd = (rh["alt_total_min"] - rh["alt_access_min"] - rh["alt_wait_min"]).where(alt)
        ax.bar(x + 0.2, acc, 0.37, color=C_MOVE)
        ax.bar(x + 0.2, wt, 0.37, bottom=acc, color=C_WAIT)
        ax.bar(x + 0.2, rd, 0.37, bottom=acc + wt, color=C_RIDE)
        for xi, (_, row) in zip(x, rh.iterrows()):
            if row["saving_min"] > 0:
                ax.text(xi + 0.2, row["alt_total_min"] + 0.8, f"→{int(row['alt_seq'])}\n{row['alt_mode']}\n-{row['saving_min']:.0f}분",
                        ha="center", va="bottom", fontsize=6.5, color=INK2)
        ax.set_xticks(x, [names.get(q, q) for q in rh["seq"]], rotation=35, ha="right", fontsize=7)
        dirs = rh["direction"].to_numpy()
        cut = next((k for k in range(1, len(dirs)) if dirs[k] != dirs[k - 1]), None)
        if cut is not None:
            ax.axvline(cut - 0.5, color=INK2, lw=1)
            tr = ax.get_xaxis_transform()  # x는 데이터, y는 축 높이 비율 (축 범위가 바뀌어도 맨 위에)
            ax.text(cut - 0.7, 0.97, "← 서울역 방향 (끝 = 서울역)", ha="right", va="top", fontsize=7.5, color=INK2, transform=tr)
            ax.text(cut - 0.3, 0.97, "성남 방향 (끝 = 서울 마지막 정류장) →", ha="left", va="top", fontsize=7.5, color=INK2, transform=tr)
        ax.set_title(f"{kr} {h} 도착 ({band} 대기가 가장 긴 10분)")
        ax.grid(axis="x", visible=False)
    for j in range(len(cases), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    for row in axes:
        row[0].set_ylabel("그 방향 끝까지 (분)")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=C_MOVE, label="다른 정류장으로 이동 (버스 또는 도보)"),
                        Patch(color=C_WAIT, label="9401 기다리기 (배차 + 몰림 + 만차로 보내는 시간)"),
                        Patch(color=C_RIDE, label="9401 타고 그 방향 끝까지 (서울역 / 서울 마지막 정류장)")],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("정류장별 그대로 기다리기(왼쪽) vs 같은 방향 추천 정류장으로 옮겨 타기(오른쪽) (→seq, 수단, 단축 시간)",
                 fontsize=12, color=INK, y=1.035)
    fig.tight_layout()
    save(fig, "F6ServiceRecommendation", "요일(평일/토/일)·출퇴근별 대기가 가장 긴 10분에 정류장마다 그대로 기다리기 vs 다른 정류장으로 옮겨 타기(이동·대기·차내 구성)")


def f13_recommended_stops(r: dict, names: dict, save) -> None:
    """추천이 어디서 어디로 나오는가: 현재 정류장 x 추천 정류장 (모든 요일 x 출퇴근 시간 합산)."""
    import matplotlib.pyplot as plt
    rows = []
    for key, rec in r["service_rec_by_day"].items():
        if key in DOW_ORDER and rec is not None and len(rec):
            rows.append(rec[rec["saving_min"] > 0].assign(day=key))
    if not rows or sum(len(x) for x in rows) == 0:
        fig, ax = plt.subplots(figsize=(10, 2.2))
        ax.axis("off")
        ax.text(0.5, 0.5, "모든 요일 x 출퇴근 시간에서 같은 방향 다른 정류장으로 옮겨 타도 3분 이상 빨라지는 경우가 없음\n"
                "(지금 정류장에서 기다리는 것이 가장 빠름)", ha="center", va="center", fontsize=11, color=INK2)
        save(fig, "F13RecommendedStops", "정류장 추천 패턴: 추천이 나온 경우 없음")
        return
    rec = pd.concat(rows, ignore_index=True)
    rec["alt_seq"] = rec["alt_seq"].astype(int)
    cnt = rec.pivot_table(index="seq", columns="alt_seq", values="saving_min", aggfunc="size", fill_value=0)
    avg = rec.pivot_table(index="seq", columns="alt_seq", values="saving_min", aggfunc="mean")
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.5 * len(cnt) + 2.6), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    im = ax.imshow(np.ma.masked_equal(cnt.to_numpy(float), 0), cmap=_cmap(BLUE_RAMP, "cnt"), aspect="auto", vmin=0)
    ax.grid(False)
    for i in range(cnt.shape[0]):
        for j in range(cnt.shape[1]):
            if cnt.iat[i, j] > 0:
                v = cnt.iat[i, j]
                ax.text(j, i, f"{v}회\n평균 {avg.iat[i, j]:.0f}분↓", ha="center", va="center", fontsize=7,
                        color="white" if v > cnt.to_numpy().max() * 0.55 else INK)
    ax.set_yticks(range(len(cnt)), [names.get(q, q) for q in cnt.index])
    ax.set_xticks(range(cnt.shape[1]), [names.get(q, q) for q in cnt.columns], rotation=30, ha="right")
    ax.set_ylabel("지금 정류장")
    ax.set_xlabel("추천 정류장 (버스·도보로 옮겨 가서 9401을 탈 곳)")
    ax.set_title("현재 정류장 x 추천 정류장 경우 수 (요일 x 출퇴근 시간)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="추천된 경우 수")
    ax2 = axes[1]
    by_day = rec.groupby("day").size().reindex(DOW_ORDER, fill_value=0)
    ax2.bar(range(7), by_day, color=[DOW_COLOR[k] for k in DOW_ORDER], width=0.7)
    ax2.set_xticks(range(7), [DOW_KR[k] for k in DOW_ORDER])
    ax2.set_ylabel("추천이 나온 정류장·시간대 수")
    from matplotlib.ticker import MaxNLocator
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.grid(axis="x", visible=False)
    ax2.set_title("요일별 추천 횟수")
    fig.suptitle("정류장 추천 경우 수 (옮겨 가는 시간까지 더해 서울을 더 빨리 빠져나가는 경우)", fontsize=12, color=INK)
    fig.tight_layout()
    save(fig, "F13RecommendedStops", "정류장 추천 패턴: 지금 정류장 x 추천 정류장 경우 수·평균 단축, 요일별 추천 횟수")


def f14_expected_wait(r: dict, save) -> None:
    """평일 정류장 x 도착 시각(10분 단위) 기대 대기 시간 (07번 줄 서기)."""
    import matplotlib.pyplot as plt
    sc = r["model_scored"]
    sl = r["slots"]["weekday"]
    names = _stop_names(sc)
    seqs = sorted(sc.loc[sc["has_skt"], "seq"].unique())
    hours = _service_hours([sc])
    # 버스가 시간당 2대 미만인 시간(첫차·막차)은 빈칸
    nb = sc.set_index(["seq", "hour"])["n_buses"]
    sl = sl[sl["seq"].isin(seqs) & sl["hour"].isin(hours)].copy()
    sl = sl.assign(wait_total_min=sl["wait_total_min"].where(
        nb.reindex(pd.MultiIndex.from_frame(sl[["seq", "hour"]])).to_numpy() >= MIN_BUSES_FOR_WAIT))
    order = sorted(sl["slot"].unique(), key=lambda k: (k * 10 // 60 < 4, k))
    piv = sl.pivot_table(index="seq", columns="slot", values="wait_total_min", dropna=False).reindex(index=seqs, columns=order)
    vals = piv.to_numpy(float)
    vmax = float(np.nanpercentile(vals, 98)) if np.isfinite(vals).any() else 1
    fig, ax = plt.subplots(figsize=(max(10, 0.11 * len(order) + 3), 4.8))
    im = ax.imshow(np.ma.masked_invalid(vals), cmap=_cmap(ORANGE_RAMP, "wait"), aspect="auto", vmin=0, vmax=max(vmax, 1),
                   interpolation="nearest")
    ax.grid(False)
    ticks = [j for j, k in enumerate(order) if k * 10 % 60 == 0]
    ax.set_xticks(ticks, [f"{order[j] * 10 // 60}시" for j in ticks])
    ax.set_yticks(range(len(piv)), [names[q] for q in piv.index])
    ax.set_xlabel("정류장 도착 시각 (10분 단위)")
    fig.colorbar(im, ax=ax, shrink=0.85, label="버스에 타기까지 기대 시간 (분)")
    ax.set_title("[평일] 정류장 x 도착 시각(10분) 기대 대기 시간")
    _note(fig, f"빈칸 = 그 시간 버스가 하루 평균 {MIN_BUSES_FOR_WAIT}대 미만 (첫차·막차). "
               "사람이 오는 속도는 1시간 단위 수요를 시간대 사이 직선으로 이어 1분마다 나눔")
    save(fig, "F14ExpectedWaitHeatmap", "평일 정류장 x 도착 시각(10분 단위) 기대 대기 시간(분)")


def f15_wait_burden(r: dict, save) -> None:
    import matplotlib.pyplot as plt
    b = {"평일": r["wait_burden"], "주말": r.get("wait_burden_weekend")}
    hours = _service_hours([r["model_scored"]])
    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(hours))
    for i, (name, d) in enumerate(b.items()):
        if d is None:
            continue
        v = d.set_index("hour").reindex(hours)
        ax.bar(x + (i - 0.5) * 0.4, v["excess_wait_person_hours_day"], 0.37, color=DAY_COLOR[name], label=f"{name} 초과 대기")
    v = b["평일"].set_index("hour").reindex(hours)
    ax.plot(x - 0.2, v["saving_upper_person_hours_day"], "o", color=INK, ms=4, label="평일 추천 이용 시 절감 상한")
    _hour_ticks(ax, hours)
    ax.set_xlabel("시간대")
    ax.set_ylabel("하루 초과 대기 (인·시간)")
    ax.grid(axis="x", visible=False)
    ax.legend()
    ax.set_title("시간대별 하루 초과 대기 부담 (인·시간)")
    save(fig, "F15WaitBurdenByHour", "시간대별 하루 초과 대기 부담(평일 vs 주말)과 추천 이용 시 절감 상한")


def f16_location_map(r: dict, names: dict, save) -> None:
    """정류장이 아닌 곳에서: 어디로 가야 9401을 가장 빨리 타고 서울을 빠져나가나 (평일 출근·퇴근 가장 붐비는 시간)."""
    import matplotlib.pyplot as plt
    grid, g, pts = r.get("service_grid"), r["geom"], r.get("service_points")
    if grid is None or grid.empty:
        return
    slots = r.get("map_slots") or sorted(grid["slot"].unique())
    fig, axes = plt.subplots(1, len(slots), figsize=(7.2 * len(slots), 7), squeeze=False)
    vmax = max(5.0, float(grid["saving_min"].quantile(0.98)))
    for ax, sl in zip(axes[0], slots):
        d = grid[grid["slot"] == sl]
        h = d["time"].iloc[0] if len(d) else ""
        x0, y0 = g["utmk_x"].min() - 500, g["utmk_y"].min() - 500
        sc = ax.scatter(d["x"] - x0, d["y"] - y0, c=d["saving_min"].where(d["saving_min"] > 0), cmap=_cmap(BLUE_RAMP, "save"),
                        vmin=0, vmax=vmax, s=62, marker="s", linewidths=0)
        ax.scatter(d.loc[d["saving_min"] <= 0, "x"] - x0, d.loc[d["saving_min"] <= 0, "y"] - y0, color="#efeee9", s=62,
                   marker="s", linewidths=0, zorder=0)
        ax.plot(g["utmk_x"] - x0, g["utmk_y"] - y0, color=AXIS, lw=1.5, zorder=2)
        out = g["direction"] == "out"
        ax.scatter(g.loc[out, "utmk_x"] - x0, g.loc[out, "utmk_y"] - y0, color=INK, s=34, zorder=3, label="성남 방향 정류장 (추천 후보)")
        ax.scatter(g.loc[~out, "utmk_x"] - x0, g.loc[~out, "utmk_y"] - y0, facecolor=SURFACE, edgecolor=MUTED, s=28,
                   zorder=3, label="서울역 방향 정류장 (반대 방향)")
        for q, row in g.iterrows():
            # 길 건너 붙어 있는 정류장(28/38, 29/37)은 번호가 겹치지 않게 위아래로
            ax.annotate(f"{q}", (row["utmk_x"] - x0, row["utmk_y"] - y0), textcoords="offset points",
                        xytext=(6, 5) if q % 2 == 0 else (6, -13), fontsize=8, color=INK, weight="bold")
        if pts is not None and len(pts):
            pp = pts[(pts["slot"] == sl) & (pts["day"] == "평일")]
            for _, row in pp.iterrows():
                px, py = row["x"] - x0, row["y"] - y0
                ax.scatter(px, py, marker="D", color=SERIES[1], s=36, zorder=4)
                ax.annotate(row["name"], (px, py), textcoords="offset points", xytext=(5, -10), fontsize=7.5, color=SERIES[1])
                if row["saving_min"] > 0:
                    t = g.loc[row["best_seq"]]
                    ax.annotate("", xy=(t["utmk_x"] - x0, t["utmk_y"] - y0), xytext=(px, py),
                                arrowprops=dict(arrowstyle="->", color=SERIES[1], lw=1.2), zorder=4)
        ax.set_aspect("equal")
        ax.set_xlabel("동서 거리 (m)")
        ax.set_ylabel("남북 거리 (m)")
        ax.grid(False)
        share = float((d["saving_min"] > 0).mean())
        ax.set_title(f"평일 {h} 도착")
        ax.text(0.01, 0.01, f"가장 가까운 정류장보다 다른 정류장이 빠른 곳 {share:.0%}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=8, color=INK2)
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.7, label="가장 가까운 정류장 대신 추천 정류장으로 가면 줄어드는 시간 (분)")
    fig.suptitle("성남 방향 9401 위치별 추천 정류장 (파란 칸 = 가장 가까운 정류장보다 다른 정류장이 빠른 곳, 화살표 = 지하철역별 추천)",
                 fontsize=12, color=INK)
    axes[0][0].legend(loc="lower left", fontsize=7.5)
    _note(fig, "성남으로 가는 사람 기준(성남 방향 정류장만 후보). 숫자 = seq, 회색 칸 = 가장 가까운 성남 방향 정류장이 가장 빠름. "
               "도보 1.2km 이내, 걷는 시간 = 직선거리 x 1.3 / 4.5km/h")
    save(fig, "F16LocationRecommendationMap", "위치 기반 추천 지도: 도심 격자마다 가장 가까운 정류장 대신 추천 정류장으로 가면 줄어드는 시간, 지하철역별 추천 화살표")


# ================================================================ S. 장표용 (하늘색 박스, 쪽 번호 = 파일 이름)
def _deck_note(fig, text: str) -> None:
    """장표 그림 각주: 눕힌 정류장 이름과 겹치지 않게 먼저 여백을 맞추고 그 아래에."""
    fig.tight_layout()
    fig.text(0.01, 0.0, text, fontsize=7.5, color=MUTED, ha="left", va="top")


def _bar_label(ax, x, y, text, **kw) -> None:
    """막대 위 숫자 (축 단위가 아니라 글자 크기만큼 띄움)."""
    if np.isfinite(y):
        ax.annotate(text, (x, y), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, **kw)


def _seoul_axis(ax, r: dict, seqs: list, names: dict, rotate: int = 35) -> None:
    """x축 = 서울 도심 정류장 순서. 회차 정류장 앞에 세로선 + 방향 글자."""
    ax.set_xticks(range(len(seqs)), [names.get(q, q) for q in seqs], rotation=rotate, ha="right", fontsize=7.5)
    g = r["geom"]
    outs = [i for i, q in enumerate(seqs) if g.at[q, "direction"] == "out"]
    if outs and outs[0] > 0:
        x = outs[0] - 0.5
        ax.axvline(x, color=INK2, lw=0.8, ls="--")
        tr = ax.get_xaxis_transform()
        ax.text(x - 0.1, 1.01, "← 서울역 방향", transform=tr, ha="right", va="bottom", fontsize=7.5, color=INK2)
        ax.text(x + 0.1, 1.01, "성남 방향 →", transform=tr, ha="left", va="bottom", fontsize=7.5, color=INK2)


def _deck_names(r: dict) -> dict:
    g = r["geom"]
    return {q: f"{q:02d} {g.at[q, 'stop_label']}" for q in g.index}


def s13_seats(r: dict, save) -> None:
    """13쪽 메인: 정류장 순서 x 도착 시 남은 좌석 (평일 퇴근 피크 / 주말 오후 피크). 보조: 정류장 x 시간대 만석 도착 비율."""
    import matplotlib.pyplot as plt
    deck, seats = r["deck"], r["cfg"].seat_capacity
    seqs, names = sorted(r["geom"].index), _deck_names(r)
    bp = r["bus_profile"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for (dt, h, kr), col in zip((("weekday", deck["weekday_pm"], "평일"), ("weekend", deck["weekend_pm"], "주말")),
                                (DAY_COLOR["평일"], DAY_COLOR["주말"])):
        d = bp[(bp["daytype"] == dt) & (bp["hour"] == h)].set_index("seq").reindex(seqs)
        x = np.arange(len(seqs))
        ax.fill_between(x, d["seats_left_p25"], d["seats_left_p75"], color=col, alpha=0.12, lw=0)
        ax.plot(x, d["seats_left_mean"], marker="o", color=col, label=f"{kr} {h}시 (평균, 띠 = 버스 25~75%)")
        if deck["focus"] in seqs:
            i = seqs.index(deck["focus"])
            v = d["seats_left_mean"].iloc[i]
            if np.isfinite(v):
                ax.annotate(f"{v:.1f}석", (i, v), xytext=(8, -9 if dt == "weekday" else 9), textcoords="offset points",
                            fontsize=8.5, color=col, fontweight="bold", va="center")
    ax.axhline(seats, color=MUTED, lw=0.8, ls=":")
    ax.text(len(seqs) - 0.6, seats, f"좌석 {seats:.0f}석", fontsize=7.5, color=MUTED, va="bottom", ha="right")
    ax.set_ylim(0, seats * 1.08)
    ax.set_ylabel("버스 도착 시 남은 좌석 (석)")
    _seoul_axis(ax, r, seqs, names)
    ax.legend(loc="lower left")
    ax.set_title("정류장 순서별 남은 좌석 (평일 vs 주말)", pad=18)
    _deck_note(fig, "교통카드 개별통행으로 운행 회차를 나눠 정류장 순서대로 (승차 - 하차) 누적 = 도착 시 재차. 남은 좌석 = 41 - 재차. "
               "피크 = 말단 정류장 남은 좌석이 가장 적은 시간 (평일 16~20시, 주말 12~20시)")
    save(fig, "S13aSeatsAlongRoute", "[장표13 메인] 서울 도심 정류장 순서별 버스 도착 시 남은 좌석 (평일 퇴근 피크 vs 주말 오후 피크)")

    # 보조: 만석 도착 비율 히트맵 (평일)
    d = bp[(bp["daytype"] == "weekday") & bp["seq"].isin(seqs) & (bp["trips_per_day"] > 0)]
    hours = sorted(d["hour"].unique(), key=lambda h: (h < 4, h))
    piv = d.pivot_table(index="seq", columns="hour", values="full_arrival_share").reindex(index=seqs, columns=hours)
    fig, ax = plt.subplots(figsize=(max(8, 0.42 * len(hours) + 3), 4.4))
    im = ax.imshow(np.ma.masked_invalid(piv.to_numpy(float)), aspect="auto", cmap=_cmap(ORANGE_RAMP, "full"),
                   vmin=0, vmax=max(0.5, float(np.nanmax(piv.to_numpy(float))) if np.isfinite(piv.to_numpy(float)).any() else 1),
                   interpolation="nearest")
    ax.grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if np.isfinite(v) and v >= 0.1:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=6.5, color="white" if v >= 0.4 else INK)
    _hour_ticks(ax, hours)
    ax.set_yticks(range(len(seqs)), [names[q] for q in seqs], fontsize=7.5)
    ax.set_xlabel("시간대")
    fig.colorbar(im, ax=ax, shrink=0.85, label="만석(남은 좌석 1석 이하)으로 도착한 버스 비율")
    ax.set_title("[평일] 정류장 x 시간대 만석 도착 비율")
    _deck_note(fig, "숫자는 10% 이상인 칸만 표시")
    save(fig, "S13bFullArrivalHeatmap", "[장표13 보조] 평일 정류장 x 시간대 만석(1석 이하) 도착 버스 비율")


def s14_headway(r: dict, save) -> None:
    """14쪽 그래프1: 정류장 순서별 도착 간격 편차·몰림 비율. 그래프2: 평일 vs 주말 시간대별 운행 횟수·평균 간격."""
    import matplotlib.pyplot as plt
    seqs, names = sorted(r["geom"].index), _deck_names(r)
    bp = r["bus_profile"]
    bp = bp[bp["seq"].isin(seqs) & (bp["trips_per_day"] > 0)]
    bands = [("평일 출근 7~9시", "weekday", [7, 8, 9], SERIES[0]), ("평일 퇴근 17~19시", "weekday", [17, 18, 19], SERIES[6]),
             ("주말 낮 12~18시", "weekend", list(range(12, 19)), SERIES[1])]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    x = np.arange(len(seqs))
    for lab, dt, hrs, col in bands:
        d = bp[(bp["daytype"] == dt) & bp["hour"].isin(hrs)]
        w = d.assign(wt=d["trips_per_day"])
        agg = w.groupby("seq").apply(lambda g: pd.Series({
            "sd": np.average(g["gap_sd"].fillna(0), weights=g["wt"]) if g["wt"].sum() > 0 else np.nan,
            "bunch": np.average(g["bunch_share"].fillna(0), weights=g["wt"]) if g["wt"].sum() > 0 else np.nan,
        })).reindex(seqs)
        a1.plot(x, agg["sd"], marker="o", color=col, label=lab)
        a2.plot(x, agg["bunch"], marker="o", color=col, label=lab)
    a1.set_ylabel("도착 간격 표준편차 (분)")
    a1.set_ylim(bottom=0)
    a1.legend(loc="lower right", ncol=3)
    a1.set_title("정류장 순서별 도착 간격 편차", pad=18)
    a2.set_ylabel("몰림 비율")
    a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    a2.set_ylim(bottom=0)
    _seoul_axis(a2, r, seqs, names)
    a1.tick_params(labelbottom=False)
    _deck_note(fig, "도착 시각 = 회차별 그 정류장 첫 승하차 태그. 간격 = 같은 정류장 연속 회차의 도착 시각 차. "
               "몰림 비율 = 간격이 그 정류장·시간대 평균 간격의 50% 미만인 버스 비율. 시간대 안은 운행 횟수로 가중 평균")
    save(fig, "S14aHeadwayAlongRoute", "[장표14 그래프1] 정류장 순서별 도착 간격 표준편차·몰림 비율 (평일 출근/퇴근, 주말 낮)")

    g = bp.groupby(["daytype", "hour"])
    hs = pd.DataFrame({"trips": g["trips_per_day"].mean(), "gap": g["gap_mean"].mean(), "sd": g["gap_sd"].mean()}).reset_index()
    hs = hs[hs["trips"] >= 1]  # 첫차·막차 시간(하루 1회 미만)은 간격이 불안정해서 뺌
    hours = sorted(hs["hour"].unique(), key=lambda h: (h < 4, h))
    pos = {h: i for i, h in enumerate(hours)}
    fig, (b1, b2) = plt.subplots(1, 2, figsize=(12, 4))
    for k, (dt, kr) in enumerate((("weekday", "평일"), ("weekend", "주말"))):
        d = hs[hs["daytype"] == dt].assign(x=lambda t: t["hour"].map(pos)).sort_values("x")
        xs = d["x"].to_numpy(float)
        b1.bar(xs + (k - 0.5) * 0.4, d["trips"], width=0.4, color=DAY_COLOR[kr], label=kr)
        b2.plot(xs, d["gap"], marker="o", color=DAY_COLOR[kr], label=f"{kr} 평균 간격")
        b2.fill_between(xs, (d["gap"] - d["sd"]).clip(lower=0), d["gap"] + d["sd"], color=DAY_COLOR[kr], alpha=0.12, lw=0)
    for b in (b1, b2):
        _hour_ticks(b, hours)
        b.set_xlabel("시간대")
        b.legend(loc="upper left")
    b1.set_ylabel("정류장당 하루 평균 운행 횟수 (회)")
    b1.set_title("시간대별 운행 횟수")
    b2.set_ylabel("실제 도착 간격 (분)")
    b2.set_ylim(bottom=0)
    b2.set_title("시간대별 평균 도착 간격 (띠 = ± 표준편차)")
    _deck_note(fig, "서울 도심 정류장(가상 정류장 제외) 평균")
    save(fig, "S14bHourlyServiceByDaytype", "[장표14 그래프2] 평일 vs 주말 시간대별 운행 횟수와 평균 도착 간격(± 표준편차)")


def s15_flow(r: dict, save) -> None:
    """15쪽: 반경 민감도 미니 막대, 대표 정류장 요일 지수, 시간대 비중."""
    import matplotlib.pyplot as plt
    sens, cfg = r["sensitivity"], r["cfg"]
    s = sens[(sens["tau"] == cfg.tau) & (sens["signal_variant"] == cfg.signal) & ~sens["is_reference"]].sort_values("radius_m")
    if len(s):
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        x = np.arange(len(s))
        ax.bar(x - 0.18, s["spearman_vs_ref"], width=0.36, color=SERIES[0], label="순위 상관")
        ax.bar(x + 0.18, s["top10_overlap_vs_ref"], width=0.36, color=BLUE_RAMP[1], label="위험 상위 10칸 일치율")
        for i, (a, b) in enumerate(zip(s["spearman_vs_ref"], s["top10_overlap_vs_ref"])):
            _bar_label(ax, i - 0.18, a, f"{a:.2f}")
            _bar_label(ax, i + 0.18, b, f"{b:.0%}")
        ax.axhline(0.9, color=SERIES[7], lw=1, ls="--")
        ax.text(len(s) - 0.5, 0.91, "0.9", color=SERIES[7], fontsize=8, ha="right", va="bottom")
        ax.set_xticks(x, [f"{v:.0f}m" for v in s["radius_m"]])
        ax.set_ylim(0, 1.35)
        ax.legend(loc="upper center", ncol=2)
        ax.set_title(f"반경별 민감도 (기준 {r['run_params'].get('ref_radius', 150):.0f}m 대비)", fontsize=10)
        _deck_note(fig, "정류장 x 시간대 탑승가능확률 순위 비교 (평일)")
        save(fig, "S15aRadiusSensitivity", "[장표15 미니] 반경 100/200m로 바꿨을 때 150m 대비 순위 상관·위험 상위 10칸 일치율")

    fp, deck = r["flow_pattern"], r["deck"]
    names = _deck_names(r)
    picks = [(k, q) for k, q in deck["stops"].items() if q in fp["dow_index"].index]
    if not picks:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, (kind, q) in enumerate(picks):
        v = fp["dow_index"].loc[q, DOW_ORDER]
        ax.plot(range(7), v, marker="o", color=SERIES[i], label=f"[{kind}] {names[q][3:]}")
    ax.axhline(100, color=MUTED, lw=0.8, ls=":")
    ax.set_xticks(range(7), ["월", "화", "수", "목", "금", "토", "일"])
    ax.set_ylabel("유동인구 지수 (7개 요일 평균 = 100)")
    ax.legend(loc="lower left", fontsize=7.5)
    ax.set_title("대표 정류장 요일별 유동인구 지수")
    _deck_note(fig, "SKT 요일별 유동인구, 정류장 반경 150m 합산")
    save(fig, "S15bFlowDowIndex", "[장표15 상단] 대표 정류장 4곳 요일별 유동인구 지수(요일 평균 = 100)")

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, (kind, q) in enumerate(picks):
        v = fp["hour_share"].loc[q].reindex(range(24)) * 100
        ax.plot(range(24), v, color=SERIES[i], label=f"[{kind}] {names[q][3:]}")
        ax.plot(int(v.idxmax()), v.max(), "o", color=SERIES[i])
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("시간대")
    ax.set_ylabel("하루 유동인구 중 비중 (%)")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.set_title("대표 정류장 시간대별 유동인구 분포 (점 = 피크)")
    _deck_note(fig, "SKT 시간대별 유동인구(월 일평균), 정류장 반경 150m 합산")
    save(fig, "S15cFlowHourShare", "[장표15 하단] 대표 정류장 4곳 시간대별 유동인구 비중(%)")


def _wait_ratio(d: pd.DataFrame) -> pd.Series:
    """정류장 대기 비율 = 정류장 대기 수요(모든 노선) ÷ 그 요일 유동인구 (장표 16쪽)."""
    d = d[d["has_skt"] & (d["flow_day"] > 0)]
    return (d["station_demand_daily"] / d["flow_day"]).groupby([d["seq"], d["hour"]]).first()


def s16_wait_ratio(r: dict, save) -> None:
    """16쪽: 대기 비율(9401 대기 수요 ÷ 유동인구) - 정류장별 평일 18시 vs 토 18시, 말단 정류장 요일 x 시간.
    그래프2: 만석 칸의 승차 기록 vs 복원 대기 수요."""
    import matplotlib.pyplot as plt
    seqs, names, deck = sorted(r["geom"].index), _deck_names(r), r["deck"]
    sdow, sc = r["scored_by_dow"], r["model_scored"]
    focus = deck["focus"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1, 1.25]})
    wd, sat = _wait_ratio(sc), _wait_ratio(sdow["sat"])
    x = np.arange(len(seqs))
    v1 = np.array([wd.get((q, 18), np.nan) for q in seqs]) * 100
    v2 = np.array([sat.get((q, 18), np.nan) for q in seqs]) * 100
    a1.bar(x - 0.2, v1, width=0.4, color=DAY_COLOR["평일"], label="평일 18시")
    a1.bar(x + 0.2, v2, width=0.4, color=DOW_COLOR["sat"], label="토요일 18시")
    a1.set_ylabel("대기 비율 (%)")
    a1.legend(loc="upper left")
    _seoul_axis(a1, r, seqs, names)
    a1.set_title("정류장별 대기 비율 (평일 vs 토요일 18시)", pad=18)
    hours = [h for h in range(5, 24)]
    piv = pd.DataFrame({DOW_KR[k]: [_wait_ratio(sdow[k]).get((focus, h), np.nan) * 100 for h in hours] for k in DOW_ORDER},
                       index=hours).T
    im = a2.imshow(np.ma.masked_invalid(piv.to_numpy(float)), aspect="auto", cmap=_cmap(BLUE_RAMP, "ratio"), interpolation="nearest")
    a2.grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if np.isfinite(v):
                a2.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > np.nanpercentile(piv.to_numpy(float), 70) else INK)
    _hour_ticks(a2, hours)
    a2.set_yticks(range(7), list(piv.index))
    a2.set_xlabel("시간대")
    fig.colorbar(im, ax=a2, shrink=0.85, label="대기 비율 (%)")
    a2.set_title(f"{names.get(focus, focus)} 요일 x 시간대 대기 비율 (%)")
    _deck_note(fig, "대기 비율 = 정류장 대기 수요(모든 노선, = 9401 대기 수요 ÷ 9401 점유율) ÷ 그 요일 반경 150m 유동인구(시간대별 x 요일 지수). "
               "유동인구가 중복 집계를 허용한 추정치라 실제 인원 비율이 아닌 상대 지표")
    save(fig, "S16aWaitRatio", "[장표16] 정류장 대기 비율(정류장 대기 수요 ÷ 유동인구): 정류장별 평일 vs 토요일 18시, 말단 정류장 요일 x 시간대")

    d = sc[sc["has_skt"] & ~sc["unconstrained"] & (sc["n_buses"] > 0) & (sc["daily_boarding"] > 0)]
    d = d.assign(ratio=d["latent_daily"] / d["daily_boarding"]).nlargest(8, "unmet_daily").iloc[::-1]
    if len(d):
        fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(d) + 1.6))
        y = np.arange(len(d))
        ax.barh(y + 0.2, 100, height=0.4, color=AXIS, label="승차 기록 (= 100)")
        ax.barh(y - 0.2, d["ratio"] * 100, height=0.4, color=SERIES[0], label="복원 대기 수요")
        for i, v in enumerate(d["ratio"]):
            ax.text(v * 100 + 2, i - 0.2, f"{v - 1:+.0%}", va="center", fontsize=8, color=SERIES[0], fontweight="bold")
        ax.set_yticks(y, [f"{names.get(q, q)} {h}시" for q, h in zip(d["seq"], d["hour"])], fontsize=8)
        ax.set_xlabel("승차 기록 = 100 기준 지수")
        ax.legend(loc="lower right")
        ax.set_title("[평일] 만석 칸의 승차 기록 vs 복원 대기 수요")
        _deck_note(fig, "만석 칸 = 버스 도착 시 재차율 60% 초과 또는 거의 만석으로 떠난 버스 5% 이상. "
                   "복원 = 자리 넉넉 칸으로 학습한 수요 모델(SKT x 9401 점유율)과 교통카드 하한 중 큰 값")
        save(fig, "S16bObservedVsRestored", "[장표16 그래프2] 평일 만석 칸 상위 8개: 승차 기록(=100) 대비 복원 대기 수요")


def s17_share(r: dict, save) -> None:
    """17쪽: 정류장별 9401 점유율 (18시), 대표 정류장 단계별 수요 축소."""
    import matplotlib.pyplot as plt
    seqs, names, deck = sorted(r["geom"].index), _deck_names(r), r["deck"]
    sh = r.get("share")
    if sh is not None:
        s18 = sh[sh["hour"] == 18].drop_duplicates("seq").set_index("seq").reindex(seqs)
        v = s18["share_weekday"].fillna(s18["share_all"]).to_numpy(float) * 100
        x = np.arange(len(seqs))
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(x, v, color=SERIES[0], width=0.6, label="9401")
        ax.bar(x, 100 - v, bottom=v, color=NEUTRAL, edgecolor=AXIS, lw=0.5, width=0.6, label="다른 노선")
        for i, (p, n) in enumerate(zip(v, s18["n_routes"])):
            if np.isfinite(p):
                ax.text(i, p / 2, f"{p:.0f}", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
            if np.isfinite(n):
                ax.text(i, 101, f"{n:.0f}개", ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_ylim(0, 110)
        ax.set_ylabel("평일 18시 정류장 전체 승차 중 비율 (%)")
        _seoul_axis(ax, r, seqs, names)
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
        ax.set_title("정류장별 9401 점유율 (평일 18시, 위 숫자 = 정차 노선 수)", pad=18)
        _deck_note(fig, "교통카드 승차 기준, 한 달 평일(공휴일 제외) 18시 합산. 정차 노선 수 = 그 정류장에서 한 달 승차 30명 이상인 노선")
        save(fig, "S17aRouteShare", "[장표17] 정류장별 평일 18시 전체 노선 승차 중 9401 점유율과 정차 노선 수")

    sc, focus = r["model_scored"], deck["focus"]
    f = sc[(sc["seq"] == focus) & (sc["hour"] == 18)]
    if len(f) and f["flow_day"].iloc[0] > 0:
        v = f.iloc[0]
        stages = [("유동인구 (반경 150m)", 100.0),
                  ("정류장 대기 수요 (모든 노선)", v["station_demand_daily"] / v["flow_day"] * 100),
                  ("9401 대기 수요", v["latent_daily"] / v["flow_day"] * 100)]
        board = v["daily_boarding"] / v["flow_day"] * 100
        fig, ax = plt.subplots(figsize=(7.5, 3.0))
        y = np.arange(len(stages))[::-1]
        cols = [BLUE_RAMP[2], BLUE_RAMP[3], BLUE_RAMP[5]]
        for yi, (lab, val), c in zip(y, stages, cols):
            ax.barh(yi, val, color=c, height=0.6)
            ax.text(val * 1.15, yi, f"{val:.2f}" if val < 10 else f"{val:.0f}", va="center", fontsize=9, fontweight="bold")
        ax.set_xscale("log")
        ax.set_xlim(min(s[1] for s in stages if s[1] > 0) / 3, 400)
        ax.set_yticks(y, [s[0] for s in stages])
        ax.set_xlabel("유동인구 = 100 기준 (로그 눈금)")
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{names.get(focus, focus)} 평일 18시 단계별 수요")
        _deck_note(fig, "정류장 대기 수요 = 9401 대기 수요 ÷ 9401 점유율(교통카드 정류장 전체 노선 승차 중 9401). "
                   "9401 대기 수요 = 자리 넉넉 칸으로 학습한 수요 모델 예측(만석 칸은 교통카드 하한과 비교해 큰 값). "
                   f"참고: 실제 9401 승차(교통카드) = {board:.1f}")
        save(fig, "S17bDemandFunnel", "[장표17 그래프3] 말단 정류장 평일 18시 단계별 수요 축소 (유동인구 = 100 지수)")


def s18_validation(r: dict, save) -> None:
    """18쪽: 모델 신뢰성 검증 4가지를 작은 그림 4개로."""
    import matplotlib.pyplot as plt
    val, sg, ds = r.get("validation", {}), r.get("split_by_grade", pd.DataFrame()), r["demand_sens"]
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    # ① 재현 오차
    bars = [("학습 기간\n(자리 넉넉 칸)", val.get("재현오차_WAPE(자리넉넉칸)", np.nan)),
            ("4주차 시험\n(1~3주 모델)", val.get("시험기간_재현오차_WAPE", np.nan))]
    ax[0].bar(range(2), [b[1] for b in bars], color=[SERIES[0], BLUE_RAMP[1]], width=0.55)
    for i, b in enumerate(bars):
        _bar_label(ax[0], i, b[1], f"{b[1]:.0%}")
    ax[0].set_ylim(0, np.nanmax([b[1] for b in bars] + [0.05]) * 1.25)
    ax[0].set_xticks(range(2), [b[0] for b in bars])
    ax[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax[0].set_title("① 재현 오차 (WAPE)", fontsize=10)
    # ② 시간 분할
    if len(sg):
        order = [g for g in ("여유", "보통", "위험") if g in set(sg["grade"])]
        d = sg.set_index("grade").reindex(order)
        cols = {"여유": PROB_DIVERGING[5], "보통": AXIS, "위험": PROB_DIVERGING[1]}
        ax[1].bar(range(len(order)), d["4주차_만석칸_비율"], color=[cols[g] for g in order], width=0.55)
        for i, (v, n) in enumerate(zip(d["4주차_만석칸_비율"], d["칸수"])):
            _bar_label(ax[1], i, v, f"{v:.0%}\n({n}칸)")
        ax[1].set_xticks(range(len(order)), [f"{g}" for g in order])
        ax[1].set_ylim(0, 1.15)
        ax[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax[1].set_xlabel("1~3주 데이터로 매긴 등급")
    ax[1].set_title("② 4주차 실제 만석 칸 비율", fontsize=10)
    # ③ 수요 ±20%
    k = [c for c in ds.columns if "겹침" in c][0]
    ax[2].bar(range(len(ds)), ds["위험칸수"], color=[BLUE_RAMP[1] if s != 1.0 else SERIES[0] for s in ds["수요배율"]], width=0.55)
    for i, (n, o) in enumerate(zip(ds["위험칸수"], ds[k])):
        _bar_label(ax[2], i, n, f"{n}칸\ntop10 {o:.0%}")
    ax[2].set_xticks(range(len(ds)), [f"수요 x{s:.1f}" for s in ds["수요배율"]])
    ax[2].set_ylim(0, max(ds["위험칸수"].max() * 1.35, 1))
    ax[2].set_title("③ 수요 ±20% 위험 칸 수 (top10 유지율)", fontsize=10)
    # ④ SKT 기여
    a, b = val.get("SKT없이_위험칸수", np.nan), val.get("SKT반영_위험칸수", np.nan)
    ax[3].bar([0, 1], [a, b], color=[AXIS, SERIES[0]], width=0.55)
    for i, n in enumerate((a, b)):
        _bar_label(ax[3], i, n, f"{n}칸")
    ax[3].set_xticks([0, 1], ["SKT 없이\n(승차 기록만)", "SKT 반영"])
    ax[3].set_ylim(0, max(np.nan_to_num(max(a, b)) * 1.3, 1))
    ax[3].set_title("④ SKT 반영 여부별 위험 칸 수", fontsize=10)
    fig.suptitle("모델 신뢰성 검증 (평일)", fontsize=12, y=1.04)
    _deck_note(fig, "① WAPE = |예측 수요 - 실제 승차| 합 ÷ 실제 승차 합, 자리 넉넉 칸(실제 승차 = 수요)에서. "
               "② 만석 칸 = 그 칸 버스의 30% 이상이 만석으로 떠남. ③ 같은 모델, 줄 서기에 넣는 수요만 ±20%. "
               "위험 = 첫 버스 탑승확률 0.7 미만")
    save(fig, "S18Validation", "[장표18] 모델 신뢰성 검증: ① 재현 오차 ② 시간 분할(1~3주 -> 4주차) ③ 수요 ±20% ④ SKT 기여")


# 18쪽 S18b: 08번 교차검증 변형을 하나씩 더해가는 순서로 (마지막 = 최종 모델)
CV_STEPS = [("no_skt", "SKT 없이\n(정류장 배수 x\n요일·시간 배수)"), ("flow_only", "유동인구 x\n평균 대기 비율"),
            ("stay", "+ 체류 목적 보정\n(정류장·요일·시간 배수,\n최종 모델)")]
CV_TYPES = [("무작위 5-fold", "아는 정류장의 다른 시간대 맞히기\n(무작위 5-fold)"),
            ("정류장 제외(LOSO)", "처음 보는 정류장 맞히기\n(정류장 하나씩 빼고 학습)")]


def s18_cv(r: dict, save) -> None:
    """18쪽: 수요 모델에 SKT·보정을 하나씩 더할 때 교차검증 오차 (F4를 장표용으로)."""
    import matplotlib.pyplot as plt
    cv = r["cv"]
    steps = [(k, lab) for k, lab in CV_STEPS if k in set(cv["signal_variant"])]
    if not steps:
        return
    fig, axes = plt.subplots(1, len(CV_TYPES), figsize=(12, 4.2), squeeze=False)
    x = np.arange(len(steps))
    for ax, (t, title) in zip(axes[0], CV_TYPES):
        d = cv[cv["cv_type"] == t].set_index("signal_variant").reindex([k for k, _ in steps])
        vals = d["mae_daily_boarding"].to_numpy(float)
        cols = [AXIS] + [BLUE_RAMP[2]] * (len(steps) - 2) + [SERIES[0]] if len(steps) > 1 else [SERIES[0]]
        ax.bar(x, vals, color=cols, width=0.6)
        base = vals[0]
        for i, v in enumerate(vals):
            lab = f"{v:.1f}" if i == 0 or not np.isfinite(base) or base <= 0 else f"{v:.1f}\n({v / base - 1:+.0%})"
            _bar_label(ax, i, v, lab)
        ax.set_xticks(x, [lab for _, lab in steps], fontsize=8)
        ax.set_ylim(0, np.nanmax(vals) * 1.3 if np.isfinite(vals).any() else 1)
        ax.set_ylabel("예측 오차 MAE (칸당 하루 정류장 승차, 명)")
        ax.grid(axis="x", visible=False)
        ax.set_title(title, fontsize=10)
    fig.suptitle("수요 모델 구성별 정류장 대기 수요 교차검증 오차 (요일 전체)", fontsize=12, y=1.03)
    _deck_note(fig, "정류장 x 요일 x 시간대 중 9401 자리 넉넉 칸(정류장 전체 노선 승차 = 대기 수요)만 사용. 괄호 = SKT 없이 대비 오차 변화(음수 = 개선). "
                    "무작위 5-fold = 칸을 5등분해 4개로 학습·1개로 시험, 정류장 제외 = 정류장 하나를 통째로 빼고 학습해 그 정류장을 맞힘")
    save(fig, "S18bCvByModelStep", "[장표18] 수요 모델 구성(SKT 없이 / 유동인구 x 평균 대기 비율 / +체류 목적 보정)별 정류장 대기 수요 교차검증 오차")


def s19_results(r: dict, save) -> None:
    """19쪽: ① 정류장 순서별 탑승확률·대기 (평일 18시 / 토요일 최악 시간) ② 말단 정류장 요일 x 시간대 탑승확률."""
    import matplotlib.pyplot as plt
    seqs, names, deck = sorted(r["geom"].index), _deck_names(r), r["deck"]
    sc, sat = r["model_scored"], r["scored_by_dow"]["sat"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    x = np.arange(len(seqs))
    for d, h, lab, col in ((sc, 18, "평일 18시", DAY_COLOR["평일"]), (sat, deck["sat_worst"], f"토요일 {deck['sat_worst']}시", DOW_COLOR["sat"])):
        v = d[d["hour"] == h].set_index("seq").reindex(seqs)
        a1.plot(x, v["p_board_mid"], marker="o", color=col, label=lab)
        a1.fill_between(x, v["p_board_low"], v["p_board_high"], color=col, alpha=0.12, lw=0)
        a2.plot(x, v["wait_total_min"], marker="o", color=col, label=f"{lab} 평균")
        if "wait_p90_min" in v:
            a2.plot(x, v["wait_p90_min"], ls="--", color=col, lw=1.2, label=f"{lab} 상위 10%")
        for i in (len(seqs) - 1,):
            p = v["p_board_mid"].iloc[i]
            if np.isfinite(p):
                a1.annotate(f"{p:.0%}", (i, p), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8, color=col)
    a1.axhline(0.7, color=PROB_DIVERGING[1], lw=0.8, ls=":")
    a1.text(len(seqs) - 0.5, 0.7, "위험 0.7", color=PROB_DIVERGING[1], fontsize=7.5, ha="right", va="bottom")
    a1.set_ylim(0, 1.08)
    a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    a1.set_ylabel("첫 버스 탑승확률")
    a1.legend(loc="lower left")
    a1.set_title("정류장 순서별 탑승확률·예상 대기 시간", pad=18)
    a2.set_ylabel("예상 대기 시간 (분)")
    a2.set_ylim(bottom=0)
    a2.legend(loc="upper left", ncol=2, fontsize=7.5)
    _seoul_axis(a2, r, seqs, names)
    _deck_note(fig, "띠 = 날마다 다른 확률의 하위~상위 25%. 대기 = 정류장 도착부터 실제 탄 버스까지 (줄 서기 계산, 만석 버스를 보낸 시간 포함)")
    save(fig, "S19aProbabilityAlongRoute", "[장표19 ①] 정류장 순서별 탑승확률·예상 대기 시간 (평일 18시 vs 토요일 최악 시간)")

    focus = deck["focus"]
    hours = _service_hours([r["scored_by_dow"][k] for k in DOW_ORDER])
    piv = pd.DataFrame({DOW_KR[k]: r["scored_by_dow"][k][r["scored_by_dow"][k]["seq"] == focus].set_index("hour")["p_board_mid"].reindex(hours)
                        for k in DOW_ORDER}).T
    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(hours) + 2.5), 3.8))
    im = _prob_heatmap(ax, piv, _cmap(PROB_DIVERGING, "prob"), _prob_norm())
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=7, color="white" if v < 0.5 else INK)
    _hour_ticks(ax, hours)
    ax.set_yticks(range(7), list(piv.index))
    ax.set_xlabel("시간대")
    cb = fig.colorbar(im, ax=ax, shrink=0.85, label="첫 버스 탑승확률")
    cb.set_ticks([0, 0.5, 0.7, 0.8, 0.9, 1.0])
    ax.set_title(f"{names.get(focus, focus)} 요일 x 시간대 탑승확률 (%)")
    _deck_note(fig, "빨강 = 위험(0.7 미만), 파랑 = 여유(0.9 이상), 빈칸 = 운행 없음")
    save(fig, "S19bFocusStopDowHour", "[장표19 ②] 말단 정류장 요일 x 시간대 첫 버스 탑승확률")


def build_deck(r: dict, save) -> None:
    """장표 하늘색 박스 그림 (13~19쪽). 하나가 실패해도 나머지는 그린다."""
    if not r.get("deck"):
        return
    for fn in (s13_seats, s14_headway, s15_flow, s16_wait_ratio, s17_share, s18_validation, s18_cv, s19_results):
        try:
            fn(r, save)
        except Exception as e:  # noqa: BLE001  안심구역에서 그림 하나 때문에 반출 전체가 멈추지 않게
            import traceback
            print(f"⚠ 장표 그림 {fn.__name__} 생략: {e}")
            traceback.print_exc()


# ================================================================ 전체
def build_all(r: dict, fig_dir: Path, manifest: list, m09) -> None:
    import matplotlib.pyplot as plt
    setup_style()

    def save(fig, name, desc):
        fig.savefig(fig_dir / f"{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
        manifest.append(("그림", f"figures/{name}.png", desc, ""))

    sc, we, by_dow = r["model_scored"], r["model_scored_weekend"], r["scored_by_dow"]
    names = _stop_names(sc)
    ev, stops = r["events"], r["stops"]
    seats = r["cfg"].seat_capacity
    seoul = sorted(sc.loc[sc["has_skt"], "seq"].unique())

    # 탐색(EDA)
    e1_daily_ridership(ev, r.get("holidays", []), save)
    e2_hourly_by_segment(ev, stops, save)
    if r.get("od") is not None:
        e3_od_matrix(r["od"], stops, save)
    e4_load_along_route(ev, stops, seats, save)
    e5_headway_distribution(ev, seoul, save)
    # 결과
    f0_summary(r, save)
    f1_heatmap_by_dow(by_dow, save)
    f2_profile_worst_hour(sc, by_dow, save)
    f8_hourly_by_dow(by_dow, save)
    f9_stop_dow_hour(by_dow, save)
    f10_day_variation(sc, save)
    if r.get("scored_uniform") is not None:
        f11_bunching_by_stop(sc, r["scored_uniform"], save)
    f12_unmet(sc, we, seats, save)
    # 검증
    f3_saturation(r["saturation"], save)
    f4_cv(r["cv"], save)
    f5_sensitivity(r["sensitivity"], save)
    f7_measured_bunching(r["measured_headway"], names, save)
    # 서비스
    f6_service_panels(r, m09, names, save)
    f13_recommended_stops(r, names, save)
    f14_expected_wait(r, save)
    f15_wait_burden(r, save)
    f16_location_map(r, names, save)
    # 장표용 (하늘색 박스)
    build_deck(r, save)
