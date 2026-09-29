# 9401 세그먼트 정의, 정류장/시간대별 재차 상한/하한 계산
"""
03_load_profile.py

02_stop_master.py 산출물(정류장 순번 seq + 서울 좌표 매칭 여부)과
01_ingest_bus.py 산출물(정류장x시간대 승하차)을 합쳐서,
'이 버스가 정류장을 지날 때마다 대략 몇 명이 타고 있었을까'를 추정한다.

핵심 아이디어)
승하차 데이터는 월간 합계지, 실제 버스 한 대의 실시간 재차인원이 아님!
그래서 절대적인 인원수가 아니라 두 단계로 나눠 추정한다.

1. 재차 하한(load_lower): seq 순서대로 (승차-하차)를 누적합산
   -> 절댓값이 아니라 정류장 간 상대적인 격차를 보는 지표

2. 대당 평균 재차 추정 범위(per_bus_avg_low/high): 재차 하한을
   '이 시간대에 실제로 몇 대가 다녔는가'로 나눠서 버스 한 대당 평균으로 환산.
   배차간격을 정확히 모르니(3~7분 범위), 해당 불확실성을 상/하한 범위로 남긴다.
   버스가 많이 다녔다고 가정하면(3분 간격) 대당 평균은 낮아지고,
   적게 다녔다고 가정하면(7분 간격) 대당 평균은 높아진다.

구간 라벨(성남 상행 / 서울 도심 / 성남 하행)은 하드코딩하지 않고,
stop_master의 좌표 매칭 여부(서울 데이터셋에 있는 정류장인지)로부터 자동으로 경계를 계산
서울 정류장이 시작되는 seq ~ 끝나는 seq 사이가 도심 구간!!

CLI: python data-engineering/03_load_profile.py --route 9401 --headway-min 3 --headway-max 7 --weekdays 20
"""


from __future__ import annotations
 
import argparse
from pathlib import Path
 
import pandas as pd
 
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
 
 
# %%
def load_inputs(route: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    bus_long = pd.read_parquet(PROCESSED_DIR / f"bus_long_{route}.parquet")
    stop_master = pd.read_parquet(PROCESSED_DIR / f"stop_master_{route}.parquet")
    return bus_long, stop_master
 
 
# %%
def derive_segments(stop_master: pd.DataFrame) -> pd.DataFrame:
    """좌표 매칭 여부(서울 데이터인가)로 상행/도심/하행 구간을 자동 도출"""
    seoul_seqs = stop_master.loc[stop_master["lat"].notna(), "seq"]
    if seoul_seqs.empty:
        raise ValueError(
            "서울 좌표가 매칭된 정류장이 없습니다. 02_stop_master.py를 먼저 확인하세요."
        )
    seoul_min, seoul_max = seoul_seqs.min(), seoul_seqs.max()
 
    seg = stop_master[["표준버스정류장ID", "seq", "stop_name"]].drop_duplicates().copy()
    # 02_stop_master.py는 표준버스정류장ID를 문자열로 저장하는데, bus_long(01번 산출물)은
    # 정수 그대로다. merge 전에 타입을 맞춰준다 (안 그러면 ValueError 발생).
    seg["표준버스정류장ID"] = seg["표준버스정류장ID"].astype(str)
 
    def _label(seq: int) -> str:
        if seq < seoul_min:
            return "성남_상행"  # 경기 -> 서울, 아직 서울 진입 전
        if seq > seoul_max:
            return "성남_하행"  # 서울 -> 경기, 서울을 빠져나온 후
        return "서울_도심루프"
 
    seg["segment"] = seg["seq"].apply(_label)
    print(
        f"서울 도심 구간: seq {seoul_min}~{seoul_max} "
        f"(상행: seq<{seoul_min}, 하행: seq>{seoul_max})"
    )
    return seg
 
 
# %%
def compute_load_lower_bound(bus_long: pd.DataFrame) -> pd.DataFrame:
    """정류장 seq 순서대로 (승차-하차)를 누적합산 -> 시간대별 재차 하한 추정."""
    df = bus_long.sort_values(["hour", "seq"]).copy()
    # merge 시 stop_master(문자열)와 타입을 맞추기 위해 여기서도 문자열로 통일
    df["표준버스정류장ID"] = df["표준버스정류장ID"].astype(str)
    df["net"] = df["boarding"] - df["alighting"]
    df["cum_net"] = df.groupby("hour")["net"].cumsum()
 
    # 월간 집계 오차로 음수가 나올 수 있으므로, 시간대별 최솟값만큼 위로 밀어 0 이상으로 보정
    hour_min = df.groupby("hour")["cum_net"].transform("min")
    df["load_lower"] = df["cum_net"] - hour_min
    return df
 
 
# %%
def add_per_bus_estimate(
    df: pd.DataFrame, headway_min: int, headway_max: int, weekdays: int
) -> pd.DataFrame:
    """월간 누적치를 '대당 평균 재차'로 환산 (배차간격 불확실성 -> 범위로 표현)"""
    departures_min = 60 / headway_max  # 배차간격이 길수록(7분) 대수는 적음
    departures_max = 60 / headway_min  # 배차간격이 짧을수록(3분) 대수는 많음
 
    df = df.copy()
    df["per_bus_avg_low"] = df["load_lower"] / (weekdays * departures_max)
    df["per_bus_avg_high"] = df["load_lower"] / (weekdays * departures_min)
    return df

# %%
def add_stop_label(profile: pd.DataFrame) -> pd.DataFrame:
    """stop_name이 겹치는 정류장을 seq로 구분한 stop_label 컬럼을 추가
    (순천향대학병원 정류장이 두 개임...) 

    stop_name만 보고 groupby/drop_duplicates를 하면 이 둘이 하나로 섞여서,
    어느 시간대엔 진입 지점이 1위, 어느 시간대엔 이탈 지점이 1위인데 겉보기엔
    같은 정류장이 왔다갔다하는 것처럼 헷갈리게 된다. 
    stop_name이 2개 이상의 서로 다른 seq에 걸쳐 있으면 seq28처럼 붙여서 구분한다.
    """
    profile = profile.copy()
    seq_per_name = profile.groupby("stop_name")["seq"].transform("nunique")
    profile["stop_label"] = profile["stop_name"]
    dup = seq_per_name > 1
    profile.loc[dup, "stop_label"] = (
        profile.loc[dup, "stop_name"] + "(seq" + profile.loc[dup, "seq"].astype(str) + ")"
    )
    return profile
 
# %%
def build_load_profile(
    route: str, headway_min: int = 3, headway_max: int = 7, weekdays: int = 20
) -> pd.DataFrame:
    bus_long, stop_master = load_inputs(route)
    segments = derive_segments(stop_master)
 
    profile = compute_load_lower_bound(bus_long)
    profile = profile.merge(segments[["표준버스정류장ID", "segment"]], on="표준버스정류장ID", how="left")
    profile = add_per_bus_estimate(profile, headway_min, headway_max, weekdays)
    profile = add_stop_label(profile) 

    cols = [
        "노선번호", "seq", "stop_name", "stop_label", "segment", "hour", 
        "boarding", "alighting", "net", "load_lower", "per_bus_avg_low", "per_bus_avg_high",
    ]
    return profile[cols].sort_values(["hour", "seq"]).reset_index(drop=True)
 
 
# %%
def top_congested_by_segment(profile: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """구간별로 load_lower가 가장 높은 (시간대, 정류장) 조합 top_n개 뽑기
 
    load_lower는 seq 순서대로 누적합산한 값이라, 서울 구간 안에서는 항상 순천향대학병원이 1위를 차지함.
    버그가 아니라 누적 지표의 구조적 특성. '버스가 가장 꽉 차는 지점이 어디냐'에는 유효한 답이지만, 
    '여러 정류장에 걸쳐 혼잡이 어떻게 분포하는가'를 보고 싶으면
    top_stops_by_metric()을 net 기준으로 쓰는 게 낫다.
    """
    ordered = profile.sort_values(["segment", "load_lower"], ascending=[True, False])
    return ordered.groupby("segment").head(top_n)[
        ["segment", "hour", "stop_label", "seq", "load_lower"]
    ]
 
 
# %%
def top_stops_by_metric(
    profile: pd.DataFrame, metric: str = "net", top_n: int = 5, segment: str | None = None
) -> pd.DataFrame:
    """정류장마다 metric이 가장 높았던 시간대 1개씩만 남기고, 그 값 기준으로 top_n 정류장을 뽑기
 
    load_lower(누적값) 대신 net(그 정류장 자체의 순수 증가분 = 승차-하차)을 기본값으로 쓴다.
    net은 누적되지 않으므로 '어느 정류장에서 혼잡이 만들어지는가'를 정류장별로 비교할 수 있다.
    """
    df = profile if segment is None else profile[profile["segment"] == segment]
    best_per_stop = df.sort_values(metric, ascending=False).drop_duplicates("stop_label") 
    return best_per_stop.nlargest(top_n, metric)[["segment", "stop_label", "seq", "hour", metric]]  
 
 
# %%
def hourly_rank1_dominance(
    profile: pd.DataFrame, metric: str = "load_lower", segment: str | None = None
) -> pd.DataFrame:
    """시간대(0~23시)마다 metric 1위 정류장이 누구인지 계산하는 함수
 
    top_congested_by_segment()의 top_n개만 보면 1등이 항상 같은 정류장이란 것만 알 수 있는데,
    이 함수는 그 정류장이 순천향대학병원임을 명확히 한다. 
    """
    df = profile if segment is None else profile[profile["segment"] == segment]
    top1_per_hour = (
        df.sort_values(["hour", metric], ascending=[True, False]).groupby("hour").head(1)
    )
    n_hours = top1_per_hour["hour"].nunique()
    counts = top1_per_hour["stop_label"].value_counts().rename("hours_as_rank1").reset_index()
    counts.columns = ["stop_label", "hours_as_rank1"]
    counts["out_of"] = n_hours
    return counts
 
 
# %%
def main() -> None:
    parser = argparse.ArgumentParser(description="정류장x시간대 재차 하한/대당 평균 추정")
    parser.add_argument("--route", type=str, default="9401")
    parser.add_argument("--headway-min", type=int, default=3, help="최소 배차간격(분)")
    parser.add_argument("--headway-max", type=int, default=7, help="최대 배차간격(분)")
    parser.add_argument("--weekdays", type=int, default=20, help="해당 월의 평일 수")
    parser.add_argument("--top-n", type=int, default=3, help="구간별로 보여줄 상위 정류장/시간대 개수")
    args = parser.parse_args()
 
    profile = build_load_profile(
        args.route, args.headway_min, args.headway_max, args.weekdays
    )
 
    out_path = PROCESSED_DIR / f"load_profile_{args.route}.parquet"
    profile.to_parquet(out_path, index=False)
 
    print(f"rows: {len(profile):,}")
 
    top_cum = top_congested_by_segment(profile, top_n=args.top_n)
    print(f"\n[1] 구간별 재차 하한(load_lower) 상위 {args.top_n}개: '버스가 가장 꽉 차는 시점'")
    print("    (누적값이라 구간의 마지막 정류장이 독점하는 게 정상)")
    print(top_cum.to_string(index=False))
 
    dominance = hourly_rank1_dominance(profile, metric="load_lower", segment="서울_도심루프")
    top_row = dominance.iloc[0]
    print(
        f"\n    -> 독점도: 서울 도심 구간에서 '{top_row['stop_label']}'이(가) "
        f"{top_row['out_of']}시간 중 {top_row['hours_as_rank1']}시간 동안 1위"
        f"{' (완전 독점)' if top_row['hours_as_rank1'] == top_row['out_of'] else ''}"
    )
 
    top_net = top_stops_by_metric(profile, metric="net", top_n=args.top_n, segment="서울_도심루프")
    print(f"\n[2] 서울 도심 구간, 정류장별 순수 증가분(net) 상위 {args.top_n}곳: 혼잡이 어디서 만들어지는가")
    print("    (정류장마다 최고 시간대 1개씩만. load_lower와 달리 여러 정류장이 골고루 나와야 정상)")
    print(top_net.to_string(index=False))
 
    print(f"\nsaved -> {out_path}")
 
 
if __name__ == "__main__":
    main()
 