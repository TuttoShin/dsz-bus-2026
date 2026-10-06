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
      HangJeongDongVer20260701.geojson                           (github.com/vuski/admdongkor)
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

import pandas as pd

ROOT = Path(__file__).resolve().parent
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
    "boundary_file": "HangJeongDongVer20260701.geojson",
    # SKT
    "flow_file": "seoul_flow_time.csv",  # SKT_003 시간대별
    "wkdy_file": "seoul_flow_wkdy.csv",  # SKT_002 요일별 (통근 보정용)
    "std_ym": "202408",  # SKT는 2024년 데이터만 있음. 승하차(2026년 8월)와 같은 8월로 맞춤
    "radii": [100.0, 150.0, 200.0],  # 정류장 반경(m). 세 개 다 돌려서 비교
    "ref_radius": 150.0,  # 최종 결과에 쓸 반경
    # 도로공사: 9401이 지나는 경부고속도로 구간 (05 --list-only 로 시군구 표기 확인 후 채울 것)
    "road_file": "TB_KOTI_ROAD TRAFFIC INDICATOR.csv",
    "road_names": ["경부고속도로"],
    "sigungu_names": ["성남시분당구", "서초구"],
    # 정류장별 지수: 도로공사 EMD_NAME 표기가 행정동 이름과 다를 때만 채움 {seq: [EMD_NAME, ...]}
    # 05 --list-emd 로 표기 확인
    "stop_emd_override": {},
    # 모델
    "seat_capacity": 41,
    "days": 20,  # 한 달 평일 수
    "headway_min": 3,  # 배차간격 범위(분)
    "headway_max": 7,
    "tau": 0.6,  # 좌석의 60% 이하로 찼으면 "자리가 넉넉했다"로 봄
    "taus": [0.5, 0.6, 0.7],  # 08번에서 비교할 tau
    "signal": "flow_x_share",  # 수요신호 = 유동인구 x 9401 분담률
    "use_commute": True,  # 통근 보정: 요일별 유동인구로 직장인 비중 반영 (출퇴근 시간대만)
    # 07번 가중합 점수의 가중치 (09번과 비교용)
    "rule_weights": {"w_load": 0.35, "w_flow": 0.4, "w_congestion": 0.25},
    "export_dir": "export",
}

DUMMY_OVERRIDES = {
    "flow_file": "dummy_flow_time.csv",
    "wkdy_file": "dummy_flow_wkdy.csv",
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
    route = c["route"]

    step("01. 승하차 원본 CSV -> 정류장x시간대")
    bus_long_all = m01.ingest(RAW_DIR / c["bus_file"])  # 전체 노선 (03b 분담률 계산용)
    bus_long_all.to_parquet(PROCESSED_DIR / "bus_long_all.parquet", index=False)
    bus_long = bus_long_all[bus_long_all["노선번호"].astype(str) == route].reset_index(drop=True)
    bus_long.to_parquet(PROCESSED_DIR / f"bus_long_{route}.parquet", index=False)
    print(f"전체 노선 {bus_long_all['노선번호'].nunique()}개, {route}번 {len(bus_long):,}행")

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

    step("03. 정류장x시간대 재차")
    load_profile = m03.build_load_profile(route, c["headway_min"], c["headway_max"], c["days"])
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
    features = features_by_radius[c["ref_radius"]]
    features.to_parquet(PROCESSED_DIR / "features.parquet", index=False)  # 안심구역 안에서만 (반출 X)

    step("07. 규칙 기반 탑승 여유 지수 (비교용)")
    rule_scored = m07.compute_margin_score(features, c["seat_capacity"], **c["rule_weights"])

    step("09. 수요 복원 + 탑승가능확률")
    cfg = m09.ModelConfig(
        seat_capacity=c["seat_capacity"], days=c["days"], headway_min=c["headway_min"],
        headway_max=c["headway_max"], tau=c["tau"], signal=c["signal"], use_commute=c["use_commute"],
    )
    model_scored, model, diag = m09.run_model(features, cfg)
    print(diag)
    print(m09.worst_cells(model_scored).round(3).to_string(index=False))

    step("08. 검증")
    cv = m08.cv_ablation(features, cfg)
    print(cv.round(3).to_string(index=False))
    saturation = m08.saturation_curve(model_scored)
    print(saturation.round(3).to_string(index=False))
    sens = m08.sensitivity(features_by_radius, cfg, c["ref_radius"], c["taus"], m09.SIGNAL_VARIANTS)
    print(sens.round(3).to_string(index=False))
    rvm = m08.rule_vs_model(rule_scored, model_scored)
    print(rvm.round(3).to_string(index=False))

    step("10. 반출 패키지")
    run_params = {k: c[k] for k in ["bus_file", "flow_file", "wkdy_file", "std_ym", "radii", "ref_radius", "road_names", "sigungu_names", "taus"]}
    m10.export_all(
        {
            "model_scored": model_scored, "model": model, "diag": diag, "cfg": cfg,
            "cv": cv, "saturation": saturation, "sensitivity": sens, "rule_vs_model": rvm,
            "kec_table": kec_table, "kec_stop": kec_stop,
            "run_params": run_params,
        },
        ROOT / c["export_dir"],
    )


if __name__ == "__main__":
    main()
