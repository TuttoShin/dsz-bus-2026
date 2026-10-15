"""
MakeDummyRaw.py

안심구역 밖에서 RunAll.py를 미리 돌려보기 위한 가짜 데이터. 정의서와 같은 컬럼명으로 만든다.
결과 숫자는 의미 없고, 파이프라인이 안 깨지고 끝까지 도는지 확인하는 용도.
다른 파일 없이 혼자 만든다 (반입 직후 data/ 가 비어 있어도 됨).

  dummyRoute.csv       노선 정보 (9401 + 다른 지역 같은 번호 9401 + 다른 노선)
  dummyRouteStop.csv   노선정류장 정보 (운행일자 2개 -> 최근 것을 고르는지 확인)
  dummyStop.csv        정류장 정보 (서울 법정동코드 11..., 성남 41...)
  dummyCard.csv        교통카드 이용내역 2024년 8월
                       버스가 정류장마다 평균 2.5분씩(들쭉날쭉하게) 걸려 가서 뒤로 갈수록 몰리고,
                       41석이 차면 못 타고 다음 버스를 기다린다 (실제처럼 만차가 생김)
                       순천향대학병원 두 정류장(seq28, seq38)은 같은 정류장ID로 만들어서 구분 로직을 확인
                       다른 노선 승차도 섞어서 9401 분담률을 계산할 수 있게
                       성남 방향 정류장에서 서울역 쪽으로 거꾸로 가는 다른 노선도 넣음 (09번 버스로 옮겨 타기 확인)
  dummyFlowTime.csv    SKT 시간대별 유동인구 (서울 정류장 주변 3x3 셀)
  dummyFlowWkdy.csv    SKT 요일별 유동인구

실행: python data-engineering/MakeDummyRaw.py   (RunAll.py --dummy 가 자동으로 부름)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
rng = np.random.default_rng(42)

ROUTE_ID = "1100094010"
SEATS = 41
# (seq, 이름, ARS, 위도, 경도, 한 달 승차, 한 달 하차) - 서울 열린데이터광장 2026년 8월 9401 집계
STOPS = [
    (1, "구미동차고지앞", "07476", None, None, 350, 32), (2, "성우스타우스앞", "47167", None, None, 869, 7),
    (3, "LG아파트.무지개마을사거리.신한아파트", "47163", None, None, 2758, 18), (4, "주공4단지", "47170", None, None, 5266, 20),
    (5, "하얀마을.그랜드빌.벽산빌라", "47148", None, None, 442, 24), (6, "오리초등학교", "47155", None, None, 1794, 54),
    (7, "미금초등학교", "47150", None, None, 549, 147), (8, "대우.롯데아파트상가", "47157", None, None, 3178, 72),
    (9, "대원사거리.까치마을", "47154", None, None, 1507, 101), (10, "더샵분당파크리버.헬스케어혁신파크", "47036", None, None, 3281, 115),
    (11, "정자3동주민센터", "47041", None, None, 6049, 214), (12, "정든마을.우성아파트", "47043", None, None, 3982, 105),
    (13, "한솔마을.LG아파트", "47029", None, None, 3306, 234), (14, "푸른마을", "47017", None, None, 14427, 817),
    (15, "샛별마을.우방아파트", "47024", None, None, 4902, 177), (16, "중앙공원.샛별마을.라이프아파트", "47022", None, None, 4221, 573),
    (17, "효자촌", "47054", None, None, 20812, 1232), (18, "서현중학교", "47065", None, None, 5473, 831),
    (19, "양현보도육교", "47059", None, None, 5886, 1206), (20, "이매촌한신.서현역.AK프라자", "47105", None, None, 14243, 3513),
    (21, "성남역.백현마을2단지", "47631", None, None, 4052, 1492), (22, "백현마을1단지", "47632", None, None, 2658, 2755),
    (23, "낙생육교", "47638", None, None, 23879, 7421),
    (28, "순천향대학병원", "03163", 37.53561, 127.0063, 2631, 26520), (29, "국가인권위.안중근활동터", "02002", 37.56435, 126.98785, 4937, 33177),
    (30, "우리은행종로지점", "02139", 37.56784, 126.98287, 812, 24931), (31, "조계사", "01267", 37.5728, 126.98316, 3097, 15581),
    (32, "세종문화회관", "01272", 37.57337, 126.97702, 13069, 9196), (33, "신한은행본점", "02233", 37.5612, 126.97559, 4019, 3821),
    (34, "서울역버스환승센터", "02005", 37.55539, 126.97272, 21174, 5914), (35, "숭례문", "02121", 37.55876, 126.97436, 6972, 438),
    (36, "명동.롯데영프라자", "02142", 37.56289, 126.98201, 14798, 1068), (37, "명동성당.남대문세무서(중)", "02001", 37.56498, 126.98753, 25188, 717),
    (38, "순천향대학병원", "03164", 37.53579, 127.0055, 24255, 2208),
    (43, "낙생육교", "47633", None, None, 5573, 32312), (44, "백현마을4단지", "47634", None, None, 507, 3707),
    (45, "성남역.백현마을3단지", "47635", None, None, 494, 2266), (46, "이매촌한신.서현역.AK프라자", "07616", None, None, 4480, 18251),
    (47, "양현보도육교.시범삼성한신아파트", "47055", None, None, 483, 6016), (48, "서현중학교", "47063", None, None, 587, 3674),
    (49, "동아우성아파트.효자촌.시범단지", "47064", None, None, 1595, 13157), (50, "중앙공원.샛별마을라이프아파트", "47023", None, None, 153, 2546),
    (51, "중앙공원.샛별마을우방아파트", "47018", None, None, 98, 4142), (52, "푸른마을", "47021", None, None, 1251, 9372),
    (53, "한솔마을.LG아파트", "47025", None, None, 68, 2426), (54, "정든마을.우성아파트", "47037", None, None, 95, 3598),
    (55, "정자3동주민센터", "47035", None, None, 102, 4381), (56, "더샵분당파크리버.헬스케어혁신파크", "47044", None, None, 89, 2627),
    (57, "청솔마을", "07410", None, None, 23, 1677), (58, "신원아파트", "47160", None, None, 167, 2121),
    (59, "미금초등학교", "47149", None, None, 29, 1005), (60, "오리초등학교", "47159", None, None, 92, 766),
    (61, "하얀마을.그랜드빌.벽산빌라", "47156", None, None, 10, 1598), (62, "무지개마을.건영아파트", "47166", None, None, 35, 3788),
    (63, "신한아파트", "47168", None, None, 24, 1234), (64, "낙생농협.오리역", "47162", None, None, 8, 1381),
    (65, "구미동차고지", "47172", None, None, 4, 226),
]
# 구간별 시간대 승차 비중 (천분율, 0~23시)
BOARD_SHAPE = {
    "out": [0, 0, 0, 0, 6, 37, 80, 105, 88, 75, 73, 67, 65, 56, 48, 45, 47, 50, 47, 33, 27, 23, 20, 5],
    "seoul": [6, 0, 0, 0, 0, 12, 30, 47, 48, 38, 28, 30, 35, 42, 49, 56, 68, 83, 91, 77, 75, 72, 70, 45],
    "in": [33, 0, 0, 0, 0, 0, 19, 41, 57, 38, 32, 28, 37, 48, 41, 55, 64, 75, 85, 85, 76, 62, 68, 55],
}


def _stop_table() -> pd.DataFrame:
    s = pd.DataFrame(STOPS, columns=["seq", "name", "ars", "lat", "lon", "board_m", "alight_m"])
    s["seg"] = np.select([s["seq"] < 28, s["seq"] > 38], ["out", "in"], "seoul")
    out = s["seg"] != "seoul"  # 성남 정류장 좌표는 분당 안에 대충 흩뿌림
    s.loc[out, "lat"] = 37.34 + (s.loc[out, "seq"] % 23) * 0.003
    s.loc[out, "lon"] = 127.11 + rng.uniform(0, 0.02, out.sum())
    s["sttn_id"] = [f"1{seq:03d}00{i % 7}" for i, seq in enumerate(s["seq"])]
    s.loc[s["seq"] == 38, "sttn_id"] = s.loc[s["seq"] == 28, "sttn_id"].iloc[0]  # 같은 정류장ID 두 번
    s["area"] = np.where(s["seg"] == "seoul", "1114010100", "4113510900")
    return s


# %%
def make_dummy_masters(s: pd.DataFrame) -> None:
    pd.DataFrame({
        "OPRAT_DATE": "20240801000000", "TCBO_ID": ["01", "02", "01"], "EXCCLC_AREA_CD": ["MM10100000", "IC10100000", "MM10100000"],
        "ROUTE_ID": [ROUTE_ID, "2200094010", "1100012340"], "ROUTE_NMA": ["9401(구미동~서울역)", "9401(다른 지역)", "1234"],
        "ROUTE_NO": ["9401", "9401", "1234"], "TFCMN_GB_CD": "1", "ROUTE_DST": 0, "ROUTE_STTN_CNT": [len(s), 20, 30],
    }).to_csv(RAW_DIR / "dummyRoute.csv", index=False, encoding="utf-8")
    rows = []
    for date, shift in (("20240701000000", 1), ("20240801000000", 0)):  # 7월 것은 순번이 틀어진 옛 버전
        for r in s.itertuples():
            rows.append({"OPRAT_DATE": date, "TCBO_ID": "01", "EXCCLC_AREA_CD": "MM10100000", "ROUTE_ID": ROUTE_ID,
                         "ROUTE_NMA": "9401", "TFCMN_GB_CD": "1", "STTN_SEQ": r.seq + shift, "STTN_ID": r.sttn_id,
                         "ROUTE_NO": r.name, "STTN_X_XYZ": f"{r.lat:.6f}", "STTN_Y_XYZ": f"{r.lon:.6f}",
                         "STTN_ARSNO": r.ars, "ROUTE_ACM_DST": 0, "STTN_DST": 0})
    pd.DataFrame(rows).to_csv(RAW_DIR / "dummyRouteStop.csv", index=False, encoding="utf-8")
    st = s.drop_duplicates("sttn_id")
    pd.DataFrame({
        "OPRAT_DATE": "20240801000000", "TCBO_ID": "01", "EXCCLC_AREA_CD": "MM10100000", "TFCMN_GB_CD": "1",
        "STTN_ID": st["sttn_id"], "STTN_NMA": st["name"].str.replace(".", " ", regex=False), "USE_AREA_CD": st["area"],
        "STTN_ARSNO": st["ars"], "STTN_GPS_X_XYZ": st["lat"].map(lambda v: f"{v:.6f}"),
        "STTN_GPS_Y_XYZ": st["lon"].map(lambda v: f"{v:.6f}"),
    }).to_csv(RAW_DIR / "dummyStop.csv", index=False, encoding="utf-8")


def make_dummy_card(s: pd.DataFrame, ym: str = "202408") -> None:
    """버스를 한 대씩 돌려서 승하차 기록을 만든다 (좌석 41, 못 탄 사람은 다음 버스)."""
    days = pd.date_range(f"{ym[:4]}-{ym[4:]}-01", periods=pd.Period(f"{ym[:4]}-{ym[4:]}").days_in_month)
    S = len(s)
    shape = np.array([BOARD_SHAPE[g] for g in s["seg"]], float) / 1000  # [정류장, 시간]
    rate_h = s["board_m"].to_numpy(float)[:, None] * shape / 31 * 1.3  # 시간당 원래 타려던 사람 (실제보다 조금 많게)
    alight_w = s["alight_m"].to_numpy(float) + 1
    rec = []
    for d, day in enumerate(days):
        weekend = day.dayofweek >= 5 or day.strftime("%Y%m%d") == "20240815"
        headway, factor = (8, 0.6) if weekend else (6, 1.0)
        deps = np.arange(5 * 60, 23 * 60, headway).astype(float)
        R = len(deps)
        hop = np.clip(rng.normal(2.5, 0.8, (R, S)), 0.5, None)
        hop[:, 0] = rng.normal(0, 1.0, R)
        arr = deps[:, None] + np.cumsum(hop, axis=1)  # [운행, 정류장] 도착 시각(분)
        dest = np.zeros((R, S))
        for j in range(S):
            later = np.arange(j + 1, S)
            p = alight_w[later] / alight_w[later].sum() if len(later) else None
            queue, last = 0.0, arr[:, j].min() - headway
            for r in np.argsort(arr[:, j]):
                t = arr[r, j]
                hour = int(t // 60) % 24
                queue += rng.poisson(rate_h[j, hour] * factor * max(t - last, 0) / 60)
                last = t
                dest[r, j] = 0  # 내림
                n = int(min(queue, SEATS - dest[r].sum())) if p is not None else 0
                if n > 0:
                    k = rng.multinomial(n, p)
                    dest[r, later] += k
                    for jj, cnt in zip(later[k > 0], k[k > 0]):
                        rec.append((d, r, j, jj, cnt, arr[r, j], arr[r, jj]))
                    queue -= n
                queue *= 0.9  # 너무 오래 기다린 사람은 포기
    df = pd.DataFrame(rec, columns=["d", "run", "j", "jj", "cnt", "t_on", "t_off"])
    df = df.loc[df.index.repeat(df["cnt"])].reset_index(drop=True)
    base = days[df["d"]]
    gin = pd.Series(base + pd.to_timedelta(df["t_on"].to_numpy() * 60 + rng.integers(0, 40, len(df)), unit="s"))
    gff = pd.Series(base + pd.to_timedelta(df["t_off"].to_numpy() * 60 + 10, unit="s"))
    missing = rng.random(len(df)) < 0.05  # 하차 태그 누락 5%
    ids = s["sttn_id"].to_numpy()
    card = pd.DataFrame({
        "OPRAT_DATE": base.strftime("%Y%m%d"), "TCBO_ID": "01", "VIRTL_CARDNO": "DUMMY",
        "EXCCLC_AREA_CD": "MM10100000", "CARD_GB_CD": "C",
        "TCBO_CAR_ID": [f"7{(r % 30) + 1:03d}" for r in df["run"]], "TRFFC_WAY_CD": "131", "TCBO_ROUTE_ID": ROUTE_ID,
        "GIN_DT": gin.dt.strftime("%Y%m%d%H%M%S"), "TCBO_GIN_STTN_ID": ids[df["j"]],
        "GFF_DT": np.where(missing, "", gff.dt.strftime("%Y%m%d%H%M%S")),
        "TCBO_GFF_STTN_ID": np.where(missing, "", ids[df["jj"]]),
        "TRNSC_ID": "1", "TRS_CNT": 0, "TCBO_USER_TYPE_CD": "01", "USER_CNT": 1, "USE_DST": 0, "BRDNG_TM": 0,
        "UMD_CD": "11140",
    })
    # 다른 노선 승차 (서울 정류장은 많이, 성남은 조금) -> 9401 분담률 계산용
    seoul_ids = set(s.loc[s["seg"] == "seoul", "sttn_id"])
    other = card[card["TCBO_GIN_STTN_ID"].isin(seoul_ids) | (rng.random(len(card)) < 0.2)].copy()
    other["TCBO_ROUTE_ID"] = rng.choice(["1100001000", "1100002000", "1100003000"], len(other))
    other["TCBO_CAR_ID"] = "8001"
    # 반대 방향 다른 노선: 성남 방향 정류장(순천향대 등)에서 타서 서울역 쪽으로 거꾸로 가는 버스
    #   (09번 "순천향대에서 다른 버스로 서울역까지 가서 9401 타기" 확인용). 정류장 하나에 3분
    turn = int(s.index[s["name"] == "서울역버스환승센터"][0])
    out_pos = [i for i in range(turn, len(s)) if s["seg"].iat[i] == "seoul"]
    rev = card[card["TCBO_GIN_STTN_ID"].isin(set(ids[out_pos[1:]]))].sample(frac=0.6, random_state=2).copy()
    pos_of = {sid: i for i, sid in enumerate(ids) if i in out_pos}
    back = rng.integers(1, 4, len(rev))
    gin_pos = rev["TCBO_GIN_STTN_ID"].map(pos_of).to_numpy()
    off_pos = np.maximum(gin_pos - back, out_pos[0])
    t_on = pd.to_datetime(rev["GIN_DT"], format="%Y%m%d%H%M%S")
    rev["TCBO_ROUTE_ID"] = "1100005000"
    rev["TCBO_CAR_ID"] = [f"9{k:03d}" for k in rng.integers(1, 40, len(rev))]
    rev["TCBO_GFF_STTN_ID"] = ids[off_pos]
    rev["GFF_DT"] = (t_on + pd.to_timedelta((gin_pos - off_pos) * 3 * 60, unit="s")).dt.strftime("%Y%m%d%H%M%S").to_numpy()
    card = pd.concat([card, other, rev], ignore_index=True)
    card.to_csv(RAW_DIR / "dummyCard.csv", index=False, encoding="utf-8")
    print(f"dummy card saved: {len(card):,} rows")


def make_dummy_skt(s: pd.DataFrame) -> None:
    """서울 정류장 주변 3x3 셀(50m). 유동인구는 그 정류장 승하차 규모·시간 모양을 따라가게 (가짜지만 관계가 있게)."""
    tf = Transformer.from_crs(4326, 5179, always_xy=True)
    seoul = s[s["seg"] == "seoul"].drop_duplicates("sttn_id")
    x, y = tf.transform(seoul["lon"].to_numpy(), seoul["lat"].to_numpy())
    shape = np.array(BOARD_SHAPE["seoul"], float) / 1000 + 0.01
    days = ["MON", "TUS", "WED", "THU", "FRI", "SAT", "SUN"]
    flow, wkdy = [], []
    for xi, yi, r in zip(x, y, seoul.itertuples()):
        scale = (r.board_m + r.alight_m) / 31 * 3
        weekend_ratio = float(rng.uniform(0.3, 1.1))
        for dx in (-50, 0, 50):
            for dy in (-50, 0, 50):
                row = {"STD_YM": "202408", "X_COORD": f"{xi + dx:.1f}", "Y_COORD": f"{yi + dy:.1f}"}
                for h in range(24):
                    row[f"TMST_{h:02d}"] = float(rng.poisson(scale * shape[h] / 9 + 5))
                flow.append(row)
                w = {"STD_YM": "202408", "X_COORD": row["X_COORD"], "Y_COORD": row["Y_COORD"]}
                base = float(rng.poisson(scale / 9 + 50))
                for dname in days:
                    w[f"FLOW_POP_CNT_{dname}"] = base * (weekend_ratio if dname in ("SAT", "SUN") else 1.0)
                wkdy.append(w)
    pd.DataFrame(flow).to_csv(RAW_DIR / "dummyFlowTime.csv", sep="|", index=False, encoding="utf-8")
    pd.DataFrame(wkdy).to_csv(RAW_DIR / "dummyFlowWkdy.csv", sep="|", index=False, encoding="utf-8")


def make_all() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    s = _stop_table()
    make_dummy_masters(s)
    make_dummy_card(s)
    make_dummy_skt(s)


if __name__ == "__main__":
    make_all()
