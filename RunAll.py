"""
RunAll.py

안심구역 안에서 01 -> 02 -> 02b -> 03 -> 03b -> 05 -> 04 -> 06 -> 07 -> 09 -> 08 -> 10 을 한 번에 실행한다.

1차 반출이 반려돼서(행 단위 데이터), 모델링과 검증까지 안에서 다 끝내고 결과(export/)만 가지고 나온다.
안에서 코드를 고칠 기회가 거의 없으니, 비교할 설정(반경, tau, 수요신호 종류)을 미리 다 정해두고 한 번에 돌린다.

[반입할 것] 파일 이름에 언더바(_)가 있으면 반입 불가
  - 레포 전체 (data/ 폴더 제외)
  - data/raw/ 에 넣을 공개 데이터 3개 (이름에서 언더바를 빼둘 것)
      2026년버스노선별정류장별시간대별승하차인원정보(08월).csv   (서울 열린데이터광장)
      서울시버스정류소위치정보(20260902).xlsx                     (서울 열린데이터광장)
      HangJeongDongSeoulVer20260701.csv                          (github.com/vuski/admdongkor 경계를 CSV로 변환)
  - SKT, 도로공사 데이터는 안심구역에 있는 파일을 data/raw/ 에 복사해서 씀

[안심구역 밖에서 리허설]
  python RunAll.py --dummy      # 가짜 SKT/도로공사 데이터로 01~10 전체 실행 -> exportDummy/

[안심구역 안에서]
  1) 경부고속도로가 지나는 시군구 표기 확인 후 CONFIG의 road_names/sigungu_names 채우기
       %run data-engineering/05KecRouteIndex.py --list-only
  2) 원본 파일명 확인 후 CONFIG 수정 (bus_file, flow_file, wkdy_file, road_file, std_ym)
  3) %run RunAll.py
  4) 정류장별 혼잡 지수에서 '시군구(동 불일치)'가 나오면
       %run data-engineering/05KecRouteIndex.py --list-emd
     로 동 이름 표기를 보고 CONFIG의 stop_emd_override를 채운 뒤 3) 다시 실행
  5) export/ 폴더만 반출 신청 (ExportManifest.md 첨부)

실행 로그(data/processed/run_log.txt)에는 정류장별 원본 값이 찍히므로 반출 X
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
import os

import pandas as pd

ROOT = Path(os.path.abspath(__file__)).parent
sys.path.insert(0, str(ROOT / "modeling"))
from Loader import load_module  # noqa: E402

PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"

# =====================================================================
# TODO(안심구역): 실제 파일명, 도로명으로 교체
# =====================================================================
CONFIG = {
    "route": "9401",
    # 공개 데이터 (반입)
    "bus_file": "2026년버스노선별정류장별시간대별승하차인원정보(08월).csv",
    "coord_file": "서울시버스정류소위치정보(20260902).xlsx",
    "boundary_file": "HangJeongDongSeoulVer20260701.csv",  # 행정동 경계 (geojson은 반입 불가라 CSV로 변환해 감)
    "daily_file": "BusStationBoardingMonth202608.csv",  # 일별 승하차 (평일/주말 나누기용, OA-12912)
    "holidays": ["20260817"],  # 평일에 낀 공휴일 (8/15 광복절이 토요일 -> 8/17 대체공휴일)
    # SKT
    "flow_file": "seoul_flow_time_202408.csv",  # SKT_003 시간대별
    "wkdy_file": "seoul_flow_wkdy_202408.csv",  # SKT_002 요일별 (통근 보정용)
    "std_ym": "202408",  # SKT는 2024년 데이터만 있음. 승하차(2026년 8월)와 같은 8월로 맞춤
    "radii": [100.0, 150.0, 200.0],  # 정류장 반경(m). 세 개 다 돌려서 비교
    "ref_radius": 150.0,  # 최종 결과에 쓸 반경
    # 도로공사: 9401이 지나는 경부고속도로 구간 (05 --list-only 로 시군구 표기 확인 후 채울 것)
    "road_file": "TB_KOTI_ROAD TRAFFIC INDICATOR.csv",
    "road_names": ["경부고속도로"],
    "sigungu_names": ["성남시 분당구", "서초구"],
    # 정류장별 지수: 도로공사 EMD_NAME 표기가 행정동 이름과 다를 때만 채움 {seq: [EMD_NAME, ...]}
    # 05 --list-emd 로 표기 확인
    "stop_emd_override": {},
    # 모델
    "seat_capacity": 41,
    "days": 20,  # 일별 파일이 없을 때만 씀 (월 합계 / 20일). 있으면 평일/주말 하루 평균으로 자동 계산
    "headway_min": 3,  # 평일 배차간격 범위(분)
    "headway_max": 7,
    # 주말 배차간격(분). 공개 자료: 토요일 4~5분, 공휴일 5~7분
    "headway_weekend_min": 4,
    "headway_weekend_max": 7,
    # 요일별 배차간격(분). 월~금은 평일 값(headway_min/max)을 씀
    "headway_sat_min": 4,  # 토요일 4~5분
    "headway_sat_max": 5,
    "headway_sun_min": 5,  # 일요일·공휴일 5~7분
    "headway_sun_max": 7,
    "tau": 0.6,  # 좌석의 60% 이하로 찼으면 "자리가 넉넉했다"로 봄
    "taus": [0.5, 0.6, 0.7],  # 08번에서 비교할 tau
    "signal": "flow_x_share",  # 수요신호 = 유동인구 x 9401 분담률(정류장마다 하루 전체로 한 번 계산)
    "use_commute": True,  # 통근 보정: 요일별 유동인구로 직장인 비중 반영 (출퇴근 시간대만)
    # 버스 몰림 정도 (배차간격 표준편차 / 평균). 0 = 일정, 1 = 완전 무작위. 실측값이 없어 중간값 사용
    # 근거: 평일에 2~4대가 몰려 온 뒤 6~10분 이상 공백이 생기는 일이 잦다 (나무위키 9401, 2026)
    "bunching_cv": 0.5,
    "bunching_cvs": [0.0, 0.5, 1.0],  # 08번에서 비교할 값
    # 도로공사 혼잡 지수로 정류장마다 몰림 정도를 다르게 (평균은 bunching_cv 그대로)
    "bunching_by_road": True,
    # 11번 서비스 계산용 기준값 (모델 가정이 아니라 안내 계산용 일반값)
    "service": {"walk_speed_kmh": 4.5, "detour": 1.3, "max_walk_m": 800, "dwell_min": 0.5, "workdays_per_year": 250},
    # 07번 가중합 점수의 가중치 (09번과 비교용)
    "rule_weights": {"w_load": 0.35, "w_flow": 0.4, "w_congestion": 0.25},
    "export_dir": "export",
}

DUMMY_OVERRIDES = {
    "flow_file": "dummy_flow_time.csv",
    "wkdy_file": "dummy_flow_wkdy.csv",
    "daily_file": "dummy_bus_daily.csv",
    "std_ym": None,
    "road_file": "dummy_road_indicator.csv",
    "export_dir": "exportDummy",
}


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


def headway_of(c: dict, daytype: str) -> tuple[float, float]:
    """요일별 배차간격 (최소, 최대). 월~금 = 평일, 토 / 일·공휴일 = 각각."""
    if daytype in ("sat", "sun"):
        return c[f"headway_{daytype}_min"], c[f"headway_{daytype}_max"]
    if daytype == "weekend":
        return c["headway_weekend_min"], c["headway_weekend_max"]
    return c["headway_min"], c["headway_max"]


def step(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    parser = argparse.ArgumentParser(description="안심구역 일괄 실행")
    parser.add_argument("--dummy", action="store_true", help="가짜 SKT/도로공사 데이터로 리허설")
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
    m01 = load_module("data-engineering/01IngestBus.py")
    m01b = load_module("data-engineering/01bDayTypeRatio.py")
    m02 = load_module("data-engineering/02StopMaster.py")
    m02b = load_module("data-engineering/02bStopDong.py")
    m03 = load_module("data-engineering/03LoadProfile.py")
    m03b = load_module("data-engineering/03bRouteShare.py")
    m04 = load_module("data-engineering/04SktCatchment.py")
    m05 = load_module("data-engineering/05KecRouteIndex.py")
    m06 = load_module("data-engineering/06BuildFeatures.py")
    m07 = load_module("modeling/07RuleBasedScore.py")
    m08 = load_module("modeling/08Validate.py")
    m09 = load_module("modeling/09BoardingProbability.py")
    m10 = load_module("modeling/10ExportResults.py")
    m11 = load_module("modeling/11ServiceRecommend.py")
    route = c["route"]

    step("01. 승하차 원본 CSV -> 정류장x시간대")
    bus_long_all = m01.ingest(RAW_DIR / c["bus_file"])  # 전체 노선 (03b 분담률 계산용)
    bus_long_all.to_parquet(PROCESSED_DIR / "bus_long_all.parquet", index=False)
    bus_long = bus_long_all[bus_long_all["노선번호"].astype(str) == route].reset_index(drop=True)
    bus_long.to_parquet(PROCESSED_DIR / f"bus_long_{route}.parquet", index=False)
    print(f"전체 노선 {bus_long_all['노선번호'].nunique()}개, {route}번 {len(bus_long):,}행")

    step("01b. 평일/주말 승하차 비율 (일별 파일)")
    daily_path = RAW_DIR / c["daily_file"]
    if dummy and not daily_path.exists():
        load_module("data-engineering/MakeDummyRaw.py").make_dummy_daily(route, bus_long)
    if daily_path.exists():
        ratio = m01b.build_daytype_ratio(m01b.load_daily(daily_path), route, c["holidays"])
        ratio.to_parquet(PROCESSED_DIR / f"daytype_ratio_{route}.parquet", index=False)
        print(f"평일 {ratio['n_weekday_days'].iloc[0]}일, 주말·공휴일 {ratio['n_weekend_days'].iloc[0]}일, "
              f"노선 전체 주말/평일 승차 비율 {ratio['board_weekend'].sum() / ratio['board_weekday'].sum():.2f}")
    else:
        ratio = None
        print(f"⚠ {daily_path.name} 없음 -> 평일/주말 구분 없이 진행 (월 합계 / {c['days']}일)")

    step("02. 정류장 좌표")
    stop_master = m02.build_stop_master(
        m02.load_bus_stops(PROCESSED_DIR / f"bus_long_{route}.parquet"),
        m02.load_coord_source(RAW_DIR / c["coord_file"]),
    )
    stop_master.to_parquet(PROCESSED_DIR / f"stop_master_{route}.parquet", index=False)

    step("02b. 정류장 -> 행정동")
    stop_dong = m02b.build_stop_dong(stop_master, RAW_DIR / c["boundary_file"])
    stop_dong.to_parquet(PROCESSED_DIR / f"stop_dong_{route}.parquet", index=False)
    print(stop_dong[["seq", "stop_name", "sgg_nm", "dong_nm"]].to_string(index=False))

    step("03. 정류장x시간대 재차 (평일 / 주말)")
    if ratio is not None:
        # 승하차가 하루 평균 단위가 되므로 이후 계산의 일수는 1
        days = 1
        load_profile = m03.build_load_profile_from(
            m03.split_daytype(bus_long, ratio, "weekday"), stop_master, c["headway_min"], c["headway_max"], days)
        load_profile_we = m03.build_load_profile_from(
            m03.split_daytype(bus_long, ratio, "weekend"), stop_master,
            c["headway_weekend_min"], c["headway_weekend_max"], days)
        load_profile_we.to_parquet(PROCESSED_DIR / f"load_profile_weekend_{route}.parquet", index=False)
        # 요일별 (월~일). 배차는 월~금 평일, 토/일 각각
        load_profile_dow = {
            k: m03.build_load_profile_from(m03.split_daytype(bus_long, ratio, k), stop_master, *headway_of(c, k), days)
            for k in m01b.DOW_KEYS
        }
    else:
        days = c["days"]
        load_profile = m03.build_load_profile(route, c["headway_min"], c["headway_max"], days)
        load_profile_we = None
        load_profile_dow = {}
    load_profile.to_parquet(PROCESSED_DIR / f"load_profile_{route}.parquet", index=False)

    step("03b. 9401 분담률")
    route_share = m03b.build_route_share(bus_long_all, route)
    route_share.to_parquet(PROCESSED_DIR / f"route_share_{route}.parquet", index=False)

    if dummy:
        step("(리허설) 가짜 SKT/도로공사 데이터 생성")
        mdummy = load_module("data-engineering/MakeDummyRaw.py")
        mdummy.make_dummy_flow(route)
        mdummy.make_dummy_wkdy(route)
        mdummy.make_dummy_road_indicator()

    step("05. 도로공사 혼잡 노출 지수 (노선 단위 + 정류장별)")
    road = m05.load_road_indicator(RAW_DIR / c["road_file"])
    segs = m05.filter_route_segments(road, c["road_names"], c["sigungu_names"])
    kec_overall = pd.DataFrame([m05.compute_congestion_exposure_index(segs)])
    kec_by_sigungu = m05.breakdown_by_sigungu(segs)
    kec_table = pd.concat([kec_overall.assign(SIGUNGU_NAME="전체"), kec_by_sigungu], ignore_index=True)
    print(kec_table.round(3).to_string(index=False))
    kec_stop = m05.compute_stop_congestion_index(road, stop_dong, c["stop_emd_override"])
    print(kec_stop.round(3).to_string(index=False))

    step("04 + 06. SKT 유동인구 반경별 집계 -> feature 병합")
    flow = m04.load_flow_population(RAW_DIR / c["flow_file"], std_ym=c["std_ym"])
    wkdy = m04.load_weekday_population(RAW_DIR / c["wkdy_file"], std_ym=c["std_ym"])
    print(f"SKT 요일별 셀 수: {len(wkdy):,}")
    print(f"SKT 셀 수: {len(flow):,}")
    features_by_radius = {}
    for radius in c["radii"]:
        print(f"\n--- 반경 {radius}m ---")
        fp = m04.build_flow_profile(route, c["flow_file"], radius, flow=flow)
        wp = m04.build_weekday_profile(route, wkdy, radius)
        features_by_radius[radius] = m06.merge_all(load_profile, fp, kec_overall, route_share, kec_stop, wp)
        if radius == c["ref_radius"]:
            ref_fp, ref_wp = fp, wp
    features = features_by_radius[c["ref_radius"]]
    features_we = (
        m06.merge_all(load_profile_we, ref_fp, kec_overall, route_share, kec_stop, ref_wp)
        if load_profile_we is not None else None
    )
    features_dow = {
        k: m06.merge_all(lp, ref_fp, kec_overall, route_share, kec_stop, ref_wp) for k, lp in load_profile_dow.items()
    }
    features.to_parquet(PROCESSED_DIR / "features.parquet", index=False)  # 안심구역 안에서만 (반출 X)

    step("07. 규칙 기반 탑승 여유 지수 (비교용)")
    rule_scored = m07.compute_margin_score(features, c["seat_capacity"], **c["rule_weights"])

    step("09. 수요 복원 + 탑승가능확률 (평일 학습 -> 주말 적용)")
    cfg = m09.ModelConfig(
        seat_capacity=c["seat_capacity"], days=days, headway_min=c["headway_min"],
        headway_max=c["headway_max"], tau=c["tau"], signal=c["signal"], use_commute=c["use_commute"],
        daytype="weekday" if ratio is not None else "all", bunching_cv=c["bunching_cv"], bunching_by_road=c["bunching_by_road"],
    )
    model_scored, model, diag = m09.run_model(features, cfg)
    print("[평일]", diag)
    print(m09.worst_cells(model_scored).round(3).to_string(index=False))
    model_scored_we = diag_we = cfg_we = None
    if features_we is not None:
        cfg_we = replace(cfg, headway_min=c["headway_weekend_min"], headway_max=c["headway_weekend_max"], daytype="weekend")
        model_scored_we, _, diag_we = m09.run_model(features_we, cfg_we, model=model)
        print("[주말]", diag_we)
        print(m09.worst_cells(model_scored_we).round(3).to_string(index=False))
    # 요일별: 평일 모델을 그대로 적용 (그 요일 승하차·SKT 요일 유동인구·배차)
    scored_dow = {}
    for k, feats in features_dow.items():
        hmin, hmax = headway_of(c, k)
        scored_dow[k], _, _ = m09.run_model(feats, replace(cfg, headway_min=hmin, headway_max=hmax, daytype=k), model=model)
    if scored_dow:
        print("[요일별 최저 탑승확률]", {k: round(float(s.loc[s["has_skt"], "p_board_mid"].min()), 3) for k, s in scored_dow.items()})

    step("08. 검증")
    cv = m08.cv_ablation(features, cfg)
    print(cv.round(3).to_string(index=False))
    saturation = m08.saturation_curve(model_scored)
    print(saturation.round(3).to_string(index=False))
    sens = m08.sensitivity(features_by_radius, cfg, c["ref_radius"], c["taus"], m09.SIGNAL_VARIANTS)
    print(sens.round(3).to_string(index=False))
    rvm = m08.rule_vs_model(rule_scored, model_scored)
    print(rvm.round(3).to_string(index=False))
    commute_hours = list(range(6, 10)) + list(range(16, 24))
    bunch = m08.bunching_effect(features, cfg, model=model, cvs=c["bunching_cvs"], hours=commute_hours).assign(daytype="평일")
    if features_we is not None:
        bunch = pd.concat([bunch, m08.bunching_effect(
            features_we, cfg_we, model=model, cvs=c["bunching_cvs"], hours=commute_hours).assign(daytype="주말")])
    print(bunch.round(3).to_string(index=False))
    # 도로공사 데이터 효과: 모든 정류장 같은 몰림 vs 지나온 길 혼잡도 반영
    road = m08.assumption_effect(features, cfg, "bunching_by_road", [False, True], model=model,
                                 hours=commute_hours).assign(daytype="평일")
    if features_we is not None:
        road = pd.concat([road, m08.assumption_effect(features_we, cfg_we, "bunching_by_road", [False, True],
                                                      model=model, hours=commute_hours).assign(daytype="주말")])
    print(road.round(3).to_string(index=False))

    step("11. 서비스: 상류 정류장 추천 + 대기 시간 부담 (평일)")
    geom = m11.stop_geometry(stop_master, kec_stop)
    rec = m11.recommend(model_scored, geom, commute_hours, c["service"])
    burden = m11.wait_burden(model_scored, rec, c["service"])
    print(rec[rec["saving_min"] > 0].round(1).to_string(index=False))
    print(burden.round(1).to_string(index=False))

    step("10. 반출 패키지")
    run_params = {k: c[k] for k in ["bus_file", "daily_file", "flow_file", "wkdy_file", "std_ym", "radii", "ref_radius", "road_names", "sigungu_names", "taus"]}
    m10.export_all(
        {
            "model_scored": model_scored, "model": model, "diag": diag, "cfg": cfg,
            "model_scored_weekend": model_scored_we, "diag_weekend": diag_we, "cfg_weekend": cfg_we,
            "scored_by_dow": scored_dow,
            "service_rec": rec, "wait_burden": burden, "service_cfg": c["service"],
            "cv": cv, "saturation": saturation, "sensitivity": sens, "rule_vs_model": rvm,
            "kec_table": kec_table, "kec_stop": kec_stop, "bunching": bunch, "road_bunching": road,
            "run_params": run_params,
        },
        ROOT / c["export_dir"],
    )


if __name__ == "__main__":
    main()
