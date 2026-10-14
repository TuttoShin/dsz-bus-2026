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
    ax.set_title(f"9401 일별 이용량: 주말·공휴일은 평일의 {we / wd:.0%} 수준")
    _note(fig, f"교통카드 이용내역 9401 전체 정류장 승차 합계를 기간 평균 = 100으로 나타냄. 공휴일({', '.join(holidays)})은 주말로.")
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
    fig.suptitle("구간별 시간대 승하차 비중: 아침엔 성남에서 타서 서울 도심에서 내리고, 저녁엔 반대", fontsize=12, color=INK)
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
    ax.set_title("어디서 타서 어디서 내리나 (승하차 정류장 쌍, 구분선 = 구간 경계)")
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
    ax.set_title("[평일] 버스가 어디서부터 차는가: 정류장 순서 x 시간대 평균 재차율", pad=22)
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
        ax.set_title(f"{name}: 1분 이내 연속 도착 {float((g <= 1).mean()):.0%}, 평균의 2배 넘는 공백 {float((g > 2 * m).mean()):.0%}")
        ax.set_xlabel("앞 버스와의 실제 간격 (분)")
    axes[0].set_ylabel("버스 비중 (%)")
    fig.suptitle("[평일 서울 도심] 버스는 일정하게 오지 않는다: 실제 배차 간격 분포", fontsize=12, color=INK)
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
    fig.suptitle("9401 서울 도심 구간 탑승 실패 한눈에 보기 (교통카드 + SKT 유동인구)", fontsize=13, color=INK)
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
    ax.set_title(f"평일 가장 위험한 시간({worst_hour}시) 정류장 순서별 탑승가능확률: 요일별 비교")
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
    fig.suptitle("요일별 시간대 탑승가능확률: 평일 퇴근 시간에 가장 낮고, 주말은 하루 종일 여유", fontsize=12, color=INK)
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
    ax.set_title(f"[평일] 가장 위험한 {top_n}칸: 같은 시간이라도 날마다 얼마나 다른가")
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
    fig.suptitle("버스 몰림 때문에 잃는 탑승 기회: 실측 vs 버스가 평균 간격·평균 재차로 고르게 왔다면", fontsize=12, color=INK)
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
    axes[0].set_title("원래 타려던 사람 중 못 탄 비율")
    axes[1].set_ylabel("필요한 추가 운행 (대/일)")
    axes[1].set_title("못 탄 사람을 다 태우려면 더 필요한 버스 (41석 기준)")
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
    ax.set_title("버스가 찰수록 예측보다 덜 탄다 (1 = 다 탐, 흐린 막대 = 칸 5개 미만)")
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
    ax.set_title("[검증] 수요신호별 예측 오차 (정류장 제외 = 처음 보는 정류장 맞히기)")
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
    ax.set_title("[검증] 설정을 바꿔도 위험 순위가 유지되는가")
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
        mean_wait = rec.groupby("hour")["wait_here_min"].mean()
        for band, hrs in (("출근", range(6, 10)), ("퇴근", range(16, 24))):
            mw = mean_wait[mean_wait.index.isin(hrs)]
            if len(mw):
                cases.append((key, kr, int(mw.idxmax()), band))
    if not cases:
        return
    C_MOVE, C_WAIT, C_RIDE = SERIES[2], SERIES[1], SERIES[0]
    ncols = 2
    nrows = -(-len(cases) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.9 * nrows), squeeze=False, sharey=True)
    for i, (key, kr, h, band) in enumerate(cases):
        ax = axes[i // ncols][i % ncols]
        rh = recs[key][recs[key]["hour"] == h].sort_values("seq")
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
        ax.set_title(f"{kr} {h}시 ({band} 대기가 가장 긴 시간)")
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
    fig.suptitle("정류장마다 왼쪽 = 그대로 기다리기, 오른쪽 = 같은 방향 추천 정류장으로 옮겨 타기 (→seq, 수단, 단축 시간)",
                 fontsize=12, color=INK, y=1.035)
    fig.tight_layout()
    save(fig, "F6ServiceRecommendation", "요일(평일/토/일)·출퇴근별 정류장마다 그대로 기다리기 vs 다른 정류장으로 옮겨 타기(이동·대기·차내 구성)")


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
    ax.set_title("어디서 어디로 추천되나 (요일 x 출퇴근 시간 경우 수)")
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
    fig.suptitle("정류장 추천은 보통 어디로? (옮겨 가는 시간까지 더해도 서울을 더 빨리 빠져나가는 경우만)", fontsize=12, color=INK)
    fig.tight_layout()
    save(fig, "F13RecommendedStops", "정류장 추천 패턴: 지금 정류장 x 추천 정류장 경우 수·평균 단축, 요일별 추천 횟수")


def f14_expected_wait(sc: pd.DataFrame, m09, save) -> None:
    import matplotlib.pyplot as plt
    w = m09.add_expected_wait(sc)
    w = w[w["has_skt"]]
    hours = _service_hours([sc])
    names = _stop_names(sc)
    w = w.assign(wait_total_min=w["wait_total_min"].where(w["n_buses"] >= MIN_BUSES_FOR_WAIT))
    piv = w.pivot_table(index="seq", columns="hour", values="wait_total_min", dropna=False).reindex(columns=hours)
    vmax = float(np.nanpercentile(piv.to_numpy(float), 98)) if np.isfinite(piv.to_numpy(float)).any() else 1
    fig, ax = plt.subplots(figsize=(max(9, 0.45 * len(hours)), 4.8))
    im = ax.imshow(np.ma.masked_invalid(piv.to_numpy(float)), cmap=_cmap(ORANGE_RAMP, "wait"), aspect="auto",
                   vmin=0, vmax=max(vmax, 1), interpolation="nearest")
    ax.grid(False)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if np.isfinite(v) and v >= 15:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5, color="white")
    _hour_ticks(ax, hours)
    ax.set_yticks(range(len(piv)), [names[q] for q in piv.index])
    ax.set_xlabel("시간대")
    fig.colorbar(im, ax=ax, shrink=0.85, label="버스에 타기까지 기대 시간 (분)")
    ax.set_title("[평일] 정류장 x 시간대 기대 대기 시간 (실제 배차 + 몰림 + 만차로 보내는 시간, 15분 이상만 숫자)")
    _note(fig, f"빈칸 = 그 시간 버스가 하루 평균 {MIN_BUSES_FOR_WAIT}대 미만 (첫차·막차 시간은 배차가 길어 대기 시간이 의미 없이 커짐)")
    save(fig, "F14ExpectedWaitHeatmap", "평일 정류장 x 운행 시간대 기대 대기 시간(분)")


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
    ax.set_title("시간대별 하루 초과 대기 부담: 만차로 버스를 보내며 더 기다린 시간의 합")
    save(fig, "F15WaitBurdenByHour", "시간대별 하루 초과 대기 부담(평일 vs 주말)과 추천 이용 시 절감 상한")


def f16_location_map(r: dict, names: dict, save) -> None:
    """정류장이 아닌 곳에서: 어디로 가야 9401을 가장 빨리 타고 서울을 빠져나가나 (평일 출근·퇴근 가장 붐비는 시간)."""
    import matplotlib.pyplot as plt
    grid, g, pts = r.get("service_grid"), r["geom"], r.get("service_points")
    if grid is None or grid.empty:
        return
    hours = r.get("map_hours") or sorted(grid["hour"].unique())
    fig, axes = plt.subplots(1, len(hours), figsize=(7.2 * len(hours), 7), squeeze=False)
    vmax = max(5.0, float(grid["saving_min"].quantile(0.98)))
    for ax, h in zip(axes[0], hours):
        d = grid[grid["hour"] == h]
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
            pp = pts[(pts["hour"] == h) & (pts["day"] == "평일")]
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
        ax.set_title(f"평일 {h}시: 가장 가까운 정류장보다 다른 정류장이 빠른 곳 {share:.0%}")
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.7, label="가장 가까운 정류장 대신 추천 정류장으로 가면 줄어드는 시간 (분)")
    fig.suptitle("성남 가는 9401, 내 위치에서 어느 정류장으로 가야 할까: 파란 칸 = 가까운 정류장보다 다른 정류장이 빠른 곳, 화살표 = 지하철역별 추천",
                 fontsize=12, color=INK)
    axes[0][0].legend(loc="lower left", fontsize=7.5)
    _note(fig, "성남으로 가는 사람 기준(성남 방향 정류장만 후보). 숫자 = seq, 회색 칸 = 가장 가까운 성남 방향 정류장이 가장 빠름. "
               "도보 1.2km 이내, 걷는 시간 = 직선거리 x 1.3 / 4.5km/h")
    save(fig, "F16LocationRecommendationMap", "위치 기반 추천 지도: 도심 격자마다 가장 가까운 정류장 대신 추천 정류장으로 가면 줄어드는 시간, 지하철역별 추천 화살표")


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
    f14_expected_wait(sc, m09, save)
    f15_wait_burden(r, save)
    f16_location_map(r, names, save)
