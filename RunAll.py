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
    "card_file": "교통카드이용내역*.csv",  # 수도권 승하차 이용내역. *로 여러 파일 (예: 일자별로 나뉜 경우)
    "route_file": "노선정보.csv",  # 노선 정보
    "route_stop_file": "노선정류장정보.csv",  # 노선정류장 정보
    "stop_file": "정류장정보.csv",  # 정류장 정보
    "route_id": None,  # 9401 노선ID를 직접 지정할 때만 (보통 None: 노선 정보에서 찾음)
    "turnaround_name": "서울역버스환승센터",  # 회차 정류장 이름. 이 전 = 서울역 방향(진입), 이후 = 성남 방향(진출)
    "card_ym": "202408",  # 교통카드에서 쓸 달 (SKT 기준년월과 같게). 파일에 다른 달이 있어도 이 달만 씀
    "holidays": ["20240815"],  # 교통카드 기간의 평일 공휴일 (2024년 8월 15일 광복절, 목) -> 일요일로 침
    # SKT
    "flow_file": "seoul_flow_time_202408.csv",  # SKT_003 시간대별
    "wkdy_file": "seoul_flow_wkdy_202408.csv",  # SKT_002 요일별 (통근 보정용)
    "std_ym": "202408",  # SKT 기준년월 = 교통카드와 같은 달
    "radii": [100.0, 150.0, 200.0],  # 정류장 반경(m). 세 개 다 돌려서 비교
    "ref_radius": 150.0,  # 최종 결과에 쓸 반경
    # 모델
    "seat_capacity": 41,
    "tau": 0.6,  # 버스 1대당 도착 시 재차가 좌석의 60% 이하면 "자리가 넉넉했다"로 봄
    "taus": [0.5, 0.6, 0.7],  # 08번에서 비교할 tau
    "signal": "flow_x_share",  # 수요신호 = 유동인구 x 9401 분담률(정류장마다 기간 전체로 한 번)
    "use_commute": True,  # 통근 보정: 요일별 유동인구로 직장인 비중 반영 (출퇴근 시간대만)
    # 09번 서비스 계산용 기준값 (모델 가정이 아니라 안내 계산용 일반값)
    "service": {"walk_speed_kmh": 4.5, "detour": 1.3, "max_walk_m": 500, "max_walk_point_m": 1200, "bus_speed_kmh": 15.0,
                "min_saving_min": 3.0, "max_ride_back_min": 15.0, "workdays_per_year": 250},
    # 위치 기반 추천 예시 (정류장이 아닌 곳). 지하철역 대략 좌표 (이름, 위도, 경도)
    "service_points": [
        ("을지로입구역", 37.5660, 126.9826), ("시청역", 37.5657, 126.9769), ("서울역", 37.5547, 126.9707),
        ("명동역", 37.5609, 126.9863), ("종각역", 37.5702, 126.9831), ("광화문역", 37.5710, 126.9768),
        ("회현역", 37.5585, 126.9783), ("충무로역", 37.5612, 126.9942),
    ],
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
    args = parser.parse_args()
    cfg_d = {**CONFIG, **(DUMMY_OVERRIDES if args.dummy else {})}

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    log = open(PROCESSED_DIR / "run_log.txt", "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log)
    try:
        run(cfg_d, dummy=args.dummy)
    finally:
        sys.stdout = sys.__stdout__
        log.close()


def run(c: dict, dummy: bool = False) -> None:
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

    step("02. 교통카드: 9401 기록 + 9401 정류장 전체 노선 승차 (전체를 한 번 훑음, 오래 걸림)")
    seoul_ids = set(stop_all.loc[stop_all["segment"] == "서울_도심루프", "sttn_id"])
    ext = m02.extract(m02.card_files(c["card_file"]), route_ids, set(stop_all["sttn_id"]), seoul_ids, c["card_ym"])
    route_id = ext["route_id"]
    stops = stop_all[stop_all["route_id"].str.lstrip("0") == route_id].reset_index(drop=True)
    stops.to_parquet(PROCESSED_DIR / "stopMaster.parquet", index=False)
    print(f"9401 노선ID = {route_id} (정류장 {len(stops)}개)")
    print(stops[["seq", "stop_label", "segment", "is_seoul"]].to_string(index=False))
    quality = {"카드파일_전체기간": ext["days"], **m02.describe(ext["card"])}

    step("03. 버스 한 대씩 도착 시각·재차·앞차와의 공백")
    events, q3, od = m03.build_bus_events(ext["card"], stops, c["holidays"])
    quality.update(q3)
    for k, v in q3.items():
        print(f"  {k}: {v:.1%}" if isinstance(v, float) and "비율" in k else f"  {k}: {v}")
    print("  (정류장순서_일치도는 1에 가까워야 정류장 연결·운행 나누기가 맞음)")
    del ext["card"]

    step("04. 정류장x시간대 하루 평균 (평일 / 주말 / 요일별) + 9401 분담률")
    profile = {k: m04.build_profile(events, stops, k, seats) for k in ["weekday", "weekend"] + DOW_KEYS}
    share = m04.route_share(ext["stop_board"], stops)
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
    features_by_radius = {}
    for radius in c["radii"]:
        fp = m05.build_flow_profile(stops, flow, radius)
        wp = m05.build_weekday_profile(stops, wkdy, radius)
        features_by_radius[radius] = m06.merge_all(profile["weekday"], fp, share, wp)
        if radius == c["ref_radius"]:
            ref_fp, ref_wp = fp, wp
    features = features_by_radius[c["ref_radius"]]
    features_by_day = {k: m06.merge_all(p, ref_fp, share, ref_wp) for k, p in profile.items()}
    features.to_parquet(PROCESSED_DIR / "features.parquet", index=False)  # 안심구역 안에서만 (반출 X)

    step("07. 수요 복원 + 탑승가능확률 (평일 학습 -> 주말·요일 적용)")
    cfg = m07.ModelConfig(seat_capacity=seats, tau=c["tau"], signal=c["signal"], use_commute=c["use_commute"],
                          daytype="weekday", bus_events=events)
    scored, model, diag = m07.run_model(features, cfg)
    print("[평일]", diag)
    print(m07.worst_cells(scored).round(3).to_string(index=False))
    scored_we, _, diag_we = m07.run_model(features_by_day["weekend"], replace(cfg, daytype="weekend"), model=model)
    print("[주말]", diag_we)
    print(m07.worst_cells(scored_we).round(3).to_string(index=False))
    scored_dow = {k: m07.run_model(features_by_day[k], replace(cfg, daytype=k), model=model)[0] for k in DOW_KEYS}
    print("[요일별 최저 탑승확률]", {k: round(float(s.loc[s["has_skt"], "p_board_mid"].min()), 3) for k, s in scored_dow.items()})

    step("08. 검증")
    cv = m08.cv_ablation(features, cfg)
    print(cv.round(3).to_string(index=False))
    saturation = m08.saturation_curve(scored)
    print(saturation.round(3).to_string(index=False))
    sens = m08.sensitivity(features_by_radius, cfg, c["ref_radius"], c["taus"], m07.SIGNAL_VARIANTS)
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

    step("09. 서비스: 기대 대기 시간 + 상류 정류장 추천 + 대기 시간 부담 (평일 / 주말 / 요일별)")
    geom = m09.stop_geometry(stops)
    other_bus = m04.other_bus_times(ext["pair_trips"], stops, route_id, c["holidays"])
    print(f"서울 정류장끼리 다른 버스로 이동 가능한 (정류장 쌍 x 시간대): {len(other_bus)}개")
    by_day = {"weekday": scored, "weekend": scored_we, **scored_dow}
    rec_by_day = {k: m09.recommend(sc_k, geom, COMMUTE_HOURS, c["service"], m04.ride_times(events, k), other_bus, k)
                  for k, sc_k in by_day.items()}
    rec = rec_by_day["weekday"]
    # 위치 기반: 예시 위치(지하철역) + 도심 격자 (평일 출근·퇴근에서 가장 붐비는 시간)
    pts = m09.points_utmk(c["service_points"])
    rides_wd = m04.ride_times(events, "weekday")
    rec_points = pd.concat([
        m09.recommend_from_points(scored, geom, pts, COMMUTE_HOURS, c["service"], rides_wd).assign(day="평일"),
        m09.recommend_from_points(scored_we, geom, pts, COMMUTE_HOURS, c["service"], m04.ride_times(events, "weekend")).assign(day="주말"),
    ], ignore_index=True)
    sk = scored[scored["has_skt"]]
    map_hours = [int(sk[sk["hour"].isin(range(6, 10))].groupby("hour")["p_board_mid"].mean().idxmin()),
                 int(sk[sk["hour"].isin(range(16, 24))].groupby("hour")["p_board_mid"].mean().idxmin())]
    rec_grid = m09.recommend_from_points(scored, geom, m09.grid_points(geom), map_hours, c["service"], rides_wd)
    print(rec_points[rec_points["saving_min"] > 0].round(1).to_string(index=False) if len(rec_points) else "위치 추천 없음")
    burden = m09.wait_burden(scored, rec, c["service"])
    burden_we = m09.wait_burden(scored_we, rec_by_day["weekend"], c["service"])
    print(rec[rec["saving_min"] > 0].round(1).to_string(index=False) if len(rec) else "정류장 추천 없음")
    print("요일별 추천 수:", {k: int((v["saving_min"] > 0).sum()) if len(v) else 0 for k, v in rec_by_day.items()})
    print(burden.round(1).to_string(index=False))

    step("10. 반출 패키지")
    run_params = {k: c[k] for k in ["card_file", "card_ym", "route_file", "route_stop_file", "stop_file", "holidays",
                                    "flow_file", "wkdy_file", "std_ym", "radii", "ref_radius", "taus"]}
    run_params["route_id_used"] = route_id
    m10.export_all(
        {
            "model_scored": scored, "model": model, "diag": diag, "cfg": cfg,
            "model_scored_weekend": scored_we, "diag_weekend": diag_we, "scored_by_dow": scored_dow,
            "cv": cv, "saturation": saturation, "sensitivity": sens, "bunching": bunching,
            "measured_headway": headway, "card_quality": quality,
            "service_rec": rec, "wait_burden": burden, "service_cfg": c["service"],
            "service_rec_by_day": rec_by_day, "wait_burden_weekend": burden_we,
            "service_points": rec_points, "service_grid": rec_grid, "geom": geom, "map_hours": map_hours,
            # 그림용 (집계해서 그리기만 하고 표로 내보내지 않음)
            "events": events, "stops": stops, "od": od, "scored_uniform": scored_uniform, "holidays": c["holidays"],
            "run_params": run_params,
        },
        ROOT / c["export_dir"],
        m09,
    )


if __name__ == "__main__":
    main()
