"""
RunAll.py

안심구역 안에서 01 -> 10 을 한 번에 실행한다.

  01 노선/노선정류장/정류장 정보 -> 9401 노선ID, 정류장 순서·좌표·서울 여부
  02 교통카드 이용내역 -> 9401 기록 + 9401 정류장의 전체 노선 승차 집계
  03 버스 한 대씩 정류장 도착 시각·재차·앞차와의 공백
  04 정류장x시간대 하루 평균 (평일/주말/요일별) + 9401 분담률
  05 SKT 유동인구 정류장 반경 합산
  06 합치기 (서울 도심 구간만)
  07 수요 복원 + 탑승가능확률 (평일 학습 -> 주말·요일 적용)
  08 검증
  09 서비스: 기대 대기 시간, 상류 정류장 추천, 대기 시간 부담
  10 반출 패키지 (export/)

1차 반출이 반려돼서(행 단위 데이터), 모델링과 검증까지 안에서 다 끝내고 결과(export/)만 가지고 나온다.
안에서 코드를 고칠 기회가 거의 없으니, 비교할 설정(반경, tau, 수요신호 종류)을 미리 다 정해두고 한 번에 돌린다.

[반입] 코드만 (파일 이름에 언더바(_)가 있으면 반입 불가). 데이터는 전부 안심구역 데이터
[안심구역 밖에서 리허설]  python RunAll.py --dummy   -> 가짜 데이터로 전체 실행 -> exportDummy/
[안심구역 안에서]         data-engineering/teminal.txt 순서대로

실행 로그(data/processed/run_log.txt)와 data/processed/ 는 행 단위 값이 있어 반출 X
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(ROOT / "modeling"))
from Loader import load_module  # noqa: E402

PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"

# =====================================================================
# TODO(안심구역): 실제 파일명으로 교체 (data/raw 기준)
# =====================================================================
CONFIG = {
    "route": "9401",
    # 교통카드 (한국교통안전공단, 안심구역 데이터)
    "card_file": "/kdata/import_data/TB_KTS_DWTCD_METROPOLITAN/202408/*.csv",  # 수도권 승하차 이용내역. *로 여러 파일 (예: 일자별로 나뉜 경우)
    "route_file": "/kdata/import_data/TB_KTS_ROUTE/202408/TB_KTS_ROUTE_20240801.csv",  # 노선 정보
    "route_stop_file": "/kdata/import_data/TB_KTS_ROUTESTTN/202408/TB_KTS_ROUTESTTN_20240801.csv",  # 노선정류장 정보
    "stop_file": "/kdata/import_data/TB_KTS_STTN/202408/TB_KTS_STTN_20240801.csv",  # 정류장 정보
    "route_id": None,  # 9401 노선ID를 직접 지정할 때만 (보통 None: 노선 정보에서 찾음)
    "turnaround_name": "서울역버스환승센터",
    "stop_merge_m": 30.0,  # 장표 17쪽 통합: 9401 정류장과 이름이 같고 이 거리(m) 안이면 다른 정산사업자 ID도 같은 정류장  # 회차 정류장 이름. 이 전 = 서울역 방향(진입), 이후 = 성남 방향(진출)
    "card_ym": "202408",  # 교통카드에서 쓸 달 (SKT 기준년월과 같게). 파일에 다른 달이 있어도 이 달만 씀
    "holidays": ["20240815"],  # 교통카드 기간의 평일 공휴일 (2024년 8월 15일 광복절, 목) -> 일요일로 침
    # SKT
    "flow_file": "seoul_flow_time_202408.csv",  # SKT_003 시간대별
    "wkdy_file": "seoul_flow_wkdy_202408.csv",  # SKT_002 요일별 (체류 목적 보정용)
    "std_ym": "202408",  # SKT 기준년월 = 교통카드와 같은 달
    "radii": [100.0, 150.0, 200.0],  # 정류장 반경(m). 세 개 다 돌려서 비교
    "ref_radius": 150.0,  # 최종 결과에 쓸 반경
    # 모델
    "seat_capacity": 41,
    "tau": 0.6,  # 버스 1대당 도착 시 재차가 좌석의 60% 이하면 "자리가 넉넉했다"로 봄
    "patience_min": None,  # 줄 서기 이탈 규칙. None = 이탈 없음. --patience 120 으로 실행하면 120분 이탈 버전
    "taus": [0.5, 0.6, 0.7],  # 08번에서 비교할 tau
    "signal": "stay",  # 수요 모델: 유동인구 x 기준 대기 비율 x 정류장 배수 x 요일·시간 배수 (체류 목적 보정)
    # 09번 서비스 계산용 기준값 (모델 가정이 아니라 안내 계산용 일반값)
    "service": {"walk_speed_kmh": 4.5, "detour": 1.3, "max_walk_m": 500, "max_walk_point_m": 1200, "bus_speed_kmh": 15.0,
                "min_saving_min": 3.0, "max_ride_back_min": 15.0, "workdays_per_year": 250},
    # 위치 기반 추천 예시 (정류장이 아닌 곳). 지하철역 대략 좌표 (이름, 위도, 경도)
    "service_points": [
        ("을지로입구역", 37.5660, 126.9826), ("시청역", 37.5657, 126.9769), ("서울역", 37.5547, 126.9707),
        ("명동역", 37.5609, 126.9863), ("종각역", 37.5702, 126.9831), ("광화문역", 37.5710, 126.9768),
        ("회현역", 37.5585, 126.9783), ("충무로역", 37.5612, 126.9942),
    ],
    # 장표 그림용 대표 정류장 (이름에 이 글자가 들어간 서울 정류장, 같은 이름이 둘이면 뒤쪽 = 성남 방향)
    "deck_stops": {"업무": "세종문화회관", "상업·관광": "명동성당", "환승": "서울역버스환승센터", "말단": "순천향대학병원"},
    "export_dir": "export",
}

DUMMY_OVERRIDES = {
    "card_file": "dummyCard.csv",
    "route_file": "dummyRoute.csv",
    "route_stop_file": "dummyRouteStop.csv",
    "stop_file": "dummyStop.csv",
    "flow_file": "dummyFlowTime.csv",
    "wkdy_file": "dummyFlowWkdy.csv",
    "export_dir": "exportDummy",
}
COMMUTE_HOURS = list(range(6, 10)) + list(range(16, 24))  # 출근 6~9시, 퇴근 16~23시
DOW_KEYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


class _Tee:
    """화면에 출력하면서 run_log.txt에도 같이 저장."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def step(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    parser = argparse.ArgumentParser(description="안심구역 일괄 실행")
    parser.add_argument("--dummy", action="store_true", help="가짜 데이터로 리허설 (MakeDummyRaw.py가 만듦)")
    parser.add_argument("--recard", action="store_true",
                        help="저장해 둔 02 교통카드 추출 결과를 쓰지 않고 다시 훑음 (보통은 입력이 바뀌면 자동으로 다시 훑음)")
    parser.add_argument("--patience", type=float, default=None,
                        help="줄 서기 이탈: 이 분(예: 120)을 기다려도 못 타면 줄에서 빠짐. 안 쓰면 이탈 없음")
    args = parser.parse_args()
    cfg_d = {**CONFIG, **(DUMMY_OVERRIDES if args.dummy else {})}
    if args.patience is not None:
        cfg_d["patience_min"] = args.patience
    # 이탈 버전은 결과 폴더·로그를 따로 (예: export -> exportQuit120), 두 번 돌려도 덮어쓰지 않게
    suffix = "" if cfg_d["patience_min"] is None else f"Quit{int(cfg_d['patience_min'])}"
    cfg_d["export_dir"] = cfg_d["export_dir"] + suffix

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    log = open(PROCESSED_DIR / f"run_log{suffix}.txt", "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log)
    try:
        run(cfg_d, dummy=args.dummy, recard=args.recard)
    finally:
        sys.stdout = sys.__stdout__
        log.close()


def run(c: dict, dummy: bool = False, recard: bool = False) -> None:
    m01 = load_module("data-engineering/01RouteStops.py")
    m02 = load_module("data-engineering/02CardExtract.py")
    m03 = load_module("data-engineering/03BusEvents.py")
    m04 = load_module("data-engineering/04LoadProfile.py")
    m05 = load_module("data-engineering/05SktCatchment.py")
    m06 = load_module("data-engineering/06BuildFeatures.py")
    m07 = load_module("modeling/07BoardingProbability.py")
    m08 = load_module("modeling/08Validate.py")
    m09 = load_module("modeling/09ServiceRecommend.py")
    m10 = load_module("modeling/10ExportResults.py")
    seats = c["seat_capacity"]

    if dummy and not (RAW_DIR / c["card_file"]).exists():
        step("(리허설) 가짜 데이터 생성")
        load_module("data-engineering/MakeDummyRaw.py").make_all()

    step("01. 9401 노선ID와 정류장 순서 (노선 / 노선정류장 / 정류장 정보)")
    if c["route_id"]:
        route_ids = [str(c["route_id"])]
    else:
        route_ids = m01.find_route_ids(RAW_DIR / c["route_file"], c["route"])["route_id"].tolist()
    stop_all = m01.build_stop_master(RAW_DIR / c["route_stop_file"], RAW_DIR / c["stop_file"], route_ids)
    stop_all = m01.add_direction(stop_all, c["turnaround_name"])
    m01.summary(stop_all)

    step("02. 교통카드: 9401 기록 + 9401 정류장 전체 노선 승차 (처음 한 번만 전체를 훑음, 오래 걸림)")
    seoul_ids = set(stop_all.loc[stop_all["segment"] == "서울_도심루프", "sttn_id"])
    aliases = m01.stop_aliases(RAW_DIR / c["stop_file"], stop_all, c["stop_merge_m"])
    print(f"다른 정산사업자 정류장ID 통합 (이름 같고 {c['stop_merge_m']:.0f}m 안): {len(aliases)}개")
    # 한 번 훑은 결과는 data/processed/cardCache/ 에 저장해 두고, 입력이 같으면 다음 실행부터 읽기만 함
    cache_dir = PROCESSED_DIR / ("cardCacheDummy" if dummy else "cardCache")
    ext = m02.extract_cached(m02.card_files(c["card_file"]), route_ids, set(stop_all["sttn_id"]), seoul_ids, c["card_ym"],
                             aliases, cache_dir, refresh=recard)
    route_id = ext["route_id"]
    stops = stop_all[stop_all["route_id"].str.lstrip("0") == route_id].reset_index(drop=True)
    stops.to_parquet(PROCESSED_DIR / "stopMaster.parquet", index=False)
    print(f"9401 노선ID = {route_id} (정류장 {len(stops)}개)")
    print(stops[["seq", "stop_label", "segment", "is_seoul"]].to_string(index=False))
    quality = {"카드파일_전체기간": ext["days"],
               "9401_노선ID후보별_기록수": str(ext["route_counts"]), "합친_노선ID": ", ".join(ext["route_ids_merged"]),
               "통합한_정류장ID_수": len(aliases), "통합ID로_연결된_승차_비율": ext["alias_board_share"],
               **m02.describe(ext["card"])}

    step("03. 버스 한 대씩 도착 시각·재차·앞차와의 공백")
    events, q3, od = m03.build_bus_events(ext["card"], stops, c["holidays"])
    quality.update(q3)
    for k, v in q3.items():
        print(f"  {k}: {v:.1%}" if isinstance(v, float) and "비율" in k else f"  {k}: {v}")
    print("  (정류장순서_일치도는 1에 가까워야 정류장 연결·운행 나누기가 맞음)")
    print("  (30초이내_연속도착_비율이 높으면 같은 버스가 두 번 잡힌 것, 재차_41명초과_운행비율이 높으면 입석·큰 차량)")
    del ext["card"]
    # 가상 정류장(이름에 '가상', 숫자뿐인 이름, 승하차 태그 없음)은 서울 도심 분석·추천·그림에서 뺌
    stops = m01.mark_virtual(stops, events)
    virtual = stops.loc[stops["is_virtual"], ["seq", "stop_name"]]
    print(f"가상 정류장 {len(virtual)}개 분석 제외: {dict(zip(virtual['seq'], virtual['stop_name']))}")
    quality["가상정류장_제외"] = ", ".join(f"{q} {n}" for q, n in zip(virtual["seq"], virtual["stop_name"]))
    stops.to_parquet(PROCESSED_DIR / "stopMaster.parquet", index=False)

    step("04. 정류장x시간대 하루 평균 (평일 / 주말 / 요일별) + 9401 분담률 + 정류장 대기 수요(전체 노선 승차)")
    profile = {k: m04.build_profile(events, stops, k, seats) for k in ["weekday", "weekend"] + DOW_KEYS}
    stop_board = ext["stop_board"]
    if len(stop_board):  # 날짜 -> 요일 (공휴일 = 일요일)
        stop_board["dow"] = m03.day_keys(stop_board["date"].astype(str), c["holidays"])[1]
    share = m04.route_share(stop_board, stops, ext.get("stop_routes"))
    station = m04.station_boarding(stop_board, stops)
    print(f"정류장 대기 수요(전체 노선 승차) 칸: {len(station):,}개 (정류장 x 평일·주말·요일 x 시간대)")
    wd = profile["weekday"]
    print("서울 구간 평일 18시:")
    print(wd[(wd["segment"] == "서울_도심루프") & (wd["hour"] == 18)][
        ["seq", "stop_label", "n_buses", "headway_obs", "gap_cv", "full_bus_share"]].round(2).to_string(index=False))
    if share is not None:
        print("정류장별 9401 분담률:", share.drop_duplicates("seq").set_index("seq")["share_all_stop"].round(3).to_dict())

    step("05 + 06. SKT 유동인구 반경별 집계 -> 합치기")
    flow = m05.load_skt(RAW_DIR / c["flow_file"], c["std_ym"])
    wkdy = m05.load_skt(RAW_DIR / c["wkdy_file"], c["std_ym"])
    print(f"SKT 시간대별 셀 {len(flow):,}개, 요일별 셀 {len(wkdy):,}개")
    def build(prof: dict, fp, wp, st: pd.DataFrame, key: str) -> pd.DataFrame:
        """06번 features + 그 요일 정류장 대기 수요 관측값(station_board)."""
        f = m06.merge_all(prof[key], fp, share, wp)
        return f.merge(st.loc[st["key"] == key, ["seq", "hour", "station_board", "share_dh", "n_total"]], on=["seq", "hour"], how="left")

    features_by_radius, stay_by_radius = {}, {}  # 체류 목적 보정은 요일 7개를 한꺼번에 학습
    for radius in c["radii"]:
        fp = m05.build_flow_profile(stops, flow, radius)
        wp = m05.build_weekday_profile(stops, wkdy, radius)
        features_by_radius[radius] = build(profile, fp, wp, station, "weekday")
        stay_by_radius[radius] = {k: build(profile, fp, wp, station, k) for k in DOW_KEYS}
        if radius == c["ref_radius"]:
            ref_fp, ref_wp = fp, wp
    features = features_by_radius[c["ref_radius"]]
    features_by_day = {k: build(profile, ref_fp, ref_wp, station, k) for k in profile}
    flow_pattern = flow_pattern_summary(stops, ref_fp, ref_wp)  # 장표 15쪽 (지수·비중만)
    print("유동인구 결합 기준:", {k: v for k, v in flow_pattern.items() if not isinstance(v, pd.DataFrame)})
    features.to_parquet(PROCESSED_DIR / "features.parquet", index=False)  # 안심구역 안에서만 (반출 X)

    step("07. 수요 복원(체류 목적 보정, 요일 7개 한꺼번에 학습) + 탑승가능확률")
    cfg = m07.ModelConfig(seat_capacity=seats, tau=c["tau"], signal=c["signal"], daytype="weekday", bus_events=events,
                          patience_min=c["patience_min"], stay_features=stay_by_radius[c["ref_radius"]])
    scored, model, diag = m07.run_model(features, cfg)
    show = lambda d: {k: v for k, v in d.items() if k != "slots"}  # 10분 단위 표는 화면에 안 찍음
    print("[평일]", show(diag))
    print(m07.worst_cells(scored).round(3).to_string(index=False))
    scored_we, _, diag_we = m07.run_model(features_by_day["weekend"], replace(cfg, daytype="weekend"), model=model)
    print("[주말]", show(diag_we))
    print(m07.worst_cells(scored_we).round(3).to_string(index=False))
    res_dow = {k: m07.run_model(features_by_day[k], replace(cfg, daytype=k), model=model) for k in DOW_KEYS}
    scored_dow = {k: r[0] for k, r in res_dow.items()}
    # 10분 단위 줄 서기 결과 (서비스·그림용): "18:20에 도착하면"
    slots = {"weekday": diag["slots"], "weekend": diag_we["slots"], **{k: r[2]["slots"] for k, r in res_dow.items()}}
    print("[요일별 최저 탑승확률]", {k: round(float(s.loc[s["has_skt"], "p_board_mid"].min()), 3) for k, s in scored_dow.items()})

    step("08. 검증")
    cv = m08.cv_ablation(cfg.stay_features, cfg)
    print(cv.round(3).to_string(index=False))
    saturation = m08.saturation_curve(scored)
    print(saturation.round(3).to_string(index=False))
    sens = m08.sensitivity(features_by_radius, stay_by_radius, cfg, c["ref_radius"], c["taus"], m07.SIGNAL_VARIANTS)
    print(sens.round(3).to_string(index=False))
    bunching = pd.concat([
        m08.bunching_effect(features, cfg, model, COMMUTE_HOURS).assign(구분="평일"),
        m08.bunching_effect(features_by_day["weekend"], replace(cfg, daytype="weekend"), model, COMMUTE_HOURS).assign(구분="주말"),
    ], ignore_index=True)
    print(bunching.round(3).to_string(index=False))
    headway = pd.concat([
        m08.measured_headway(events, features, "weekday", COMMUTE_HOURS).assign(daytype="평일"),
        m08.measured_headway(events, features, "weekend", COMMUTE_HOURS).assign(daytype="주말"),
    ], ignore_index=True)
    print(headway.round(2).to_string(index=False))
    # 그림용: 버스가 고르게 왔다고 칠 때 (정류장별 몰림 효과 F11)
    scored_uniform = m07.run_model(features, replace(cfg, uniform=True), model=model)[0]
    # 장표 13·14쪽: 정류장·시간대별 남은 좌석, 만석 도착, 도착 간격·몰림 비율
    bus_prof = m08.bus_profile(events, seats)
    # 장표 18쪽 모델 신뢰성 검증 4가지
    def make_features(e: pd.DataFrame):
        """그 기간 버스 기록·정류장 승차만으로 (평일 features, 요일별 features). 시간 분할 검증용."""
        st = m04.station_boarding(stop_board[stop_board["date"].isin(set(e["date"].astype(str).str[:8]))], stops)
        prof = {k: m04.build_profile(e, stops, k, seats) for k in ["weekday"] + DOW_KEYS}
        return build(prof, ref_fp, ref_wp, st, "weekday"), {k: build(prof, ref_fp, ref_wp, st, k) for k in DOW_KEYS}
    split_summary, split_by_grade = m08.time_split(events, make_features, cfg)
    demand_sens = m08.demand_sensitivity(features, cfg, model)
    validation = {"재현오차_WAPE(자리넉넉칸)": m08.reproduction_error(scored), **split_summary,
                  **m08.skt_contribution(features, cfg, scored)}
    print("검증:", validation)
    print(split_by_grade.round(3).to_string(index=False) if len(split_by_grade) else "")
    print(demand_sens.round(3).to_string(index=False))

    step("09. 서비스: 기대 대기 시간 + 상류 정류장 추천 + 대기 시간 부담 (평일 / 주말 / 요일별)")
    geom = m09.stop_geometry(stops)
    other_bus = m04.other_bus_times(ext["pair_trips"], stops, route_id, c["holidays"])
    print(f"서울 정류장끼리 다른 버스로 이동 가능한 (정류장 쌍 x 시간대): {len(other_bus)}개")
    rec_by_day = {k: m09.recommend(sl, geom, COMMUTE_HOURS, c["service"], m04.ride_times(events, k), other_bus, k)
                  for k, sl in slots.items()}  # 10분 단위
    rec = rec_by_day["weekday"]
    # 위치 기반: 예시 위치(지하철역) + 도심 격자 (평일 출근·퇴근에서 가장 붐비는 시간)
    pts = m09.points_utmk(c["service_points"])
    rides_wd = m04.ride_times(events, "weekday")
    rec_points = pd.concat([
        m09.recommend_from_points(slots["weekday"], geom, pts, COMMUTE_HOURS, c["service"], rides_wd).assign(day="평일"),
        m09.recommend_from_points(slots["weekend"], geom, pts, COMMUTE_HOURS, c["service"],
                                  m04.ride_times(events, "weekend")).assign(day="주말"),
    ], ignore_index=True)
    # 지도(F16)는 평일 출근·퇴근에서 성남 방향 정류장 평균 대기가 가장 긴 10분 하나씩
    sw = slots["weekday"][slots["weekday"]["seq"].isin(geom.index[geom["direction"] == "out"])]
    mean_wait = sw.groupby("slot")["wait_total_min"].mean()
    map_slots = [int(mean_wait[mean_wait.index.map(lambda k: k * m07.SLOT_MIN // 60 in hrs)].idxmax())
                 for hrs in (range(6, 10), range(16, 24)) if mean_wait.index.map(lambda k: k * m07.SLOT_MIN // 60 in hrs).any()]
    map_slot_tbl = slots["weekday"][slots["weekday"]["slot"].isin(map_slots)]
    rec_grid = m09.recommend_from_points(map_slot_tbl, geom, m09.grid_points(geom), COMMUTE_HOURS, c["service"], rides_wd)
    print(rec_points[rec_points["saving_min"] > 0].round(1).to_string(index=False) if len(rec_points) else "위치 추천 없음")
    burden = m09.wait_burden(scored, rec, c["service"])
    burden_we = m09.wait_burden(scored_we, rec_by_day["weekend"], c["service"])
    print(rec[rec["saving_min"] > 0].round(1).to_string(index=False) if len(rec) else "정류장 추천 없음")
    print("요일별 추천 수:", {k: int((v["saving_min"] > 0).sum()) if len(v) else 0 for k, v in rec_by_day.items()})
    print(burden.round(1).to_string(index=False))
    # 회피 가능한 탑승 실패: 첫 9401을 만석으로 못 탄 사람 중 정류장 변경·출발 시각 조정으로 더 빨리 탈 수 있었던 비율
    avoid, avoid_detail = m09.avoidable_failures(slots["weekday"], rec, geom, COMMUTE_HOURS)
    a0 = avoid["회피가능_비율"]
    print(f"회피 가능한 탑승 실패 (평일 출퇴근, 가정 9가지): {a0.min():.0%} ~ {a0.max():.0%}")
    print(avoid.round(3).to_string(index=False))
    deck = deck_context(geom, c["deck_stops"], bus_prof, scored_dow)
    print("장표 대표 정류장·시간:", deck)

    step("10. 반출 패키지")
    run_params = {k: c[k] for k in ["card_file", "card_ym", "route_file", "route_stop_file", "stop_file", "holidays",
                                    "flow_file", "wkdy_file", "std_ym", "radii", "ref_radius", "taus", "patience_min"]}
    run_params["route_id_used"] = route_id
    run_params["patience_min"] = c["patience_min"] if c["patience_min"] is not None else "이탈 없음"
    m10.export_all(
        {
            "model_scored": scored, "model": model, "diag": diag, "cfg": cfg,
            "model_scored_weekend": scored_we, "diag_weekend": diag_we, "scored_by_dow": scored_dow,
            "cv": cv, "saturation": saturation, "sensitivity": sens, "bunching": bunching,
            "measured_headway": headway, "card_quality": quality,
            "service_rec": rec, "wait_burden": burden, "service_cfg": c["service"],
            "service_rec_by_day": rec_by_day, "wait_burden_weekend": burden_we,
            "service_points": rec_points, "service_grid": rec_grid, "geom": geom, "map_slots": map_slots,
            "slots": slots,
            # 장표용
            "bus_profile": bus_prof, "validation": validation, "split_by_grade": split_by_grade, "demand_sens": demand_sens,
            "flow_pattern": flow_pattern, "share": share, "deck": deck, "avoid": avoid, "avoid_detail": avoid_detail,
            # 그림용 (집계해서 그리기만 하고 표로 내보내지 않음)
            "events": events, "stops": stops, "od": od, "scored_uniform": scored_uniform, "holidays": c["holidays"],
            "run_params": run_params,
        },
        ROOT / c["export_dir"],
        m09,
    )


def deck_context(geom: pd.DataFrame, deck_stops: dict, bus_prof: pd.DataFrame, scored_dow: dict) -> dict:
    """장표 그림·숫자에 공통으로 쓸 대표 정류장과 시간.

    stops      : {"업무": seq, ...} 이름에 글자가 들어간 서울 정류장 (같은 이름이 둘이면 뒤쪽 = 성남 방향)
    focus      : 말단 정류장 (없으면 성남 방향 마지막 정류장). 13·16·17·19쪽 대표
    weekday_pm : 평일 16~20시 중 말단 정류장 도착 시 남은 좌석이 가장 적은 시간
    weekend_pm : 주말 12~20시 중 말단 정류장 남은 좌석이 가장 적은 시간
    sat_worst  : 토요일 서울 정류장 평균 기대 대기 시간이 가장 긴 시간 (19쪽)
    """
    found = {}
    for kind, name in deck_stops.items():
        hit = [q for q in geom.index if name in str(geom.at[q, "stop_name"])]
        if hit:
            found[kind] = int(hit[-1])
    outs = [q for q in geom.index if geom.at[q, "direction"] == "out"]
    focus = found.get("말단", int(outs[-1] if outs else geom.index[-1]))
    bp = bus_prof[bus_prof["seq"] == focus].set_index(["daytype", "hour"])["seats_left_mean"]

    def _min_hour(dt, hours):
        v = bp.reindex([(dt, h) for h in hours]).dropna()
        return int(v.idxmin()[1]) if len(v) else hours[len(hours) // 2]
    sat = scored_dow.get("sat")
    sat_worst = 14
    if sat is not None:
        m = sat[sat["has_skt"] & sat["hour"].between(6, 23)].groupby("hour")["wait_total_min"].mean().dropna()
        sat_worst = int(m.idxmax()) if len(m) else 14
    return {"stops": found, "focus": focus, "weekday_pm": _min_hour("weekday", list(range(16, 21))),
            "weekend_pm": _min_hour("weekend", list(range(12, 21))), "sat_worst": sat_worst}


def flow_pattern_summary(stops: pd.DataFrame, fp: pd.DataFrame, wp: pd.DataFrame) -> dict:
    """장표 15쪽: 정류장-유동인구 결합 기준과 정류장별 유동인구 패턴. 원값 없이 지수·비중만.

    dow_index  : 정류장별 요일 유동인구 / 7개 요일 평균 x 100
    hour_share : 정류장별 하루 유동인구 중 그 시간 비중
    """
    import numpy as np
    s = stops[(stops["segment"] == "서울_도심루프") & ~stops["is_virtual"].fillna(False).astype(bool)].drop_duplicates("seq")
    xy = s[["utmk_x", "utmk_y"]].to_numpy(float)
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2))
    same = s["stop_name"].to_numpy()[:, None] == s["stop_name"].to_numpy()[None, :]
    d[same] = np.inf  # 같은 이름(양방향 같은 정류장)은 빼고 가장 가까운 다른 정류장
    near = d.min(axis=1)
    dows = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    w = wp[wp["seq"].isin(s["seq"])].set_index("seq")[[f"flow_{k}" for k in dows]]
    dow_index = w.div(w.mean(axis=1), axis=0).mul(100)
    dow_index.columns = dows
    f = fp[fp["seq"].isin(s["seq"])].pivot_table(index="seq", columns="hour", values="flow_pop")
    hour_share = f.div(f.sum(axis=1), axis=0)
    # 같은 18시라도 정류장·요일에 따라 유동인구가 몇 배 차이 나는가 (18시 유동인구 x 요일 지수)
    at18 = (f[18].to_frame().values * dow_index.reindex(f.index).to_numpy() / 100).ravel() if 18 in f else np.array([])
    at18 = at18[np.isfinite(at18) & (at18 > 0)]
    cells = fp.drop_duplicates("seq").set_index("seq")["n_cells_matched"].reindex(s["seq"])
    return {
        "nearest_stop_median_m": float(np.median(near[np.isfinite(near)])) if np.isfinite(near).any() else np.nan,
        "cells_in_radius_median": float(cells.median()),
        "max_ratio_18h": float(at18.max() / at18.min()) if len(at18) else np.nan,
        "dow_index": dow_index, "hour_share": hour_share,
    }


if __name__ == "__main__":
    main()
