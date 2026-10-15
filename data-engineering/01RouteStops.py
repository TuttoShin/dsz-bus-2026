# 9401 노선ID와 경유 정류장 순서·좌표 (교통카드 노선/노선정류장/정류장 정보)
"""
01RouteStops.py

교통카드 이용내역의 노선ID·정류장ID는 정산사 번호다.
같은 기관(한국교통안전공단)의 노선 정보 / 노선정류장 정보 / 정류장 정보로 9401을 찾고,
정류장 순서(seq), 이름, 좌표, 서울 여부를 정리한다. 이후 모든 단계가 이 표(stop_master)를 쓴다.

1. 노선 정보에서 노선번호(ROUTE_NO) = 9401인 노선ID(ROUTE_ID) 후보
   (다른 지역에 같은 번호가 있을 수 있어서 후보가 여러 개면 02번에서 교통카드 기록이 가장 많은 것을 고름)
2. 노선정류장 정보에서 그 노선ID의 정류장 순번(STTN_SEQ)과 정류장ID(STTN_ID)
   운행일자(OPRAT_DATE)별로 여러 번 들어 있으면 가장 최근 것
3. 정류장 정보에서 이름(STTN_NMA), ARS번호, GPS 좌표, 법정동코드(USE_AREA_CD)
   좌표는 UTM-K(m)로 바꿔서 SKT 유동인구 셀과 거리를 잴 수 있게 한다
4. 서울 정류장 = 법정동코드가 11로 시작 (서울특별시). 없으면 좌표로 판단
   서울 첫 정류장 ~ 마지막 정류장 = 서울_도심루프, 그 앞 = 성남_상행, 뒤 = 성남_하행
   SKT 유동인구는 서울만 있어서 분석은 서울_도심루프만 한다 (성남 구간은 재차 계산에만 씀)
5. 가상 정류장(이름에 "가상", 숫자뿐인 이름, 승하차 태그가 한 번도 없는 곳)은 분석에서 뺀다 (mark_virtual)
6. 방향: 9401은 서울역버스환승센터에서 회차한다. 그 전까지는 서울역 방향(진입), 그 정류장부터는 성남 방향(진출)
   09번 추천 서비스는 같은 방향 정류장끼리만 비교한다

컬럼 이름은 정의서 그대로 (영어 코드, 또는 "운행일자(OPRAT_DATE)" 같은 한글+영어 표기 둘 다 찾음).
주의: 노선정류장 정보의 정류장 이름 컬럼은 정의서상 ROUTE_NO로 되어 있다 (정류장명칭(ROUTE_NO)).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
_TRANSFORMER = Transformer.from_crs(4326, 5179, always_xy=True)  # WGS84 -> UTM-K (SKT 좌표계)

ROUTE_COLS = {"date": "OPRAT_DATE", "tcbo": "TCBO_ID", "route_id": "ROUTE_ID", "route_name": "ROUTE_NMA",
              "route_no": "ROUTE_NO", "n_stops": "ROUTE_STTN_CNT"}
ROUTE_STOP_COLS = {"date": "OPRAT_DATE", "route_id": "ROUTE_ID", "seq": "STTN_SEQ", "sttn_id": "STTN_ID",
                   "name": "ROUTE_NO", "x": "STTN_X_XYZ", "y": "STTN_Y_XYZ", "ars": "STTN_ARSNO"}
STOP_COLS = {"date": "OPRAT_DATE", "sttn_id": "STTN_ID", "name": "STTN_NMA", "area": "USE_AREA_CD",
             "ars": "STTN_ARSNO", "x": "STTN_GPS_X_XYZ", "y": "STTN_GPS_Y_XYZ"}
CHUNK = 1_000_000
KR = {"OPRAT_DATE": "운행일자", "TCBO_ID": "정산사ID", "ROUTE_ID": "노선ID", "ROUTE_NMA": "노선명칭", "ROUTE_NO": "노선번호", "ROUTE_STTN_CNT": "노선정류장수", "STTN_SEQ": "정류장순번", "STTN_ID": "정류장ID", "STTN_NMA": "정류장명칭", "STTN_X_XYZ": "정류장X좌표", "STTN_Y_XYZ": "정류장Y좌표", "STTN_ARSNO": "정류장ARS번호", "USE_AREA_CD": "법정동코드", "STTN_GPS_X_XYZ": "정류장GPSX좌표", "STTN_GPS_Y_XYZ": "정류장GPSY좌표", "TCBO_CAR_ID": "정산사차량ID", "TRFFC_WAY_CD": "교통수단코드", "TCBO_ROUTE_ID": "정산사노선ID", "GIN_DT": "승차일시", "TCBO_GIN_STTN_ID": "정산사승차정류장ID", "GFF_DT": "하차일시", "TCBO_GFF_STTN_ID": "정산사하차정류장ID", "USER_CNT": "이용자수"}
SEOUL_AREA_PREFIX = "11"  # 법정동코드 앞 두 자리 (서울특별시)


def norm_id(s: pd.Series) -> pd.Series:
    """ID 비교용: 공백과 앞자리 0 제거."""
    return s.astype(str).str.strip().str.lstrip("0")


# %%
def _header(path: Path) -> tuple[list[str], str, str]:
    """csv/txt의 컬럼 이름, 구분자, 인코딩."""
    for enc in ("utf-8-sig", "cp949"):
        try:
            with open(path, encoding=enc) as f:
                line = f.readline()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError("utf-8/cp949", b"", 0, 1, f"{path} 인코딩을 알 수 없습니다.")
    sep = max([",", "\t", "|"], key=line.count)
    return [c.strip().strip('"') for c in line.rstrip("\r\n").split(sep)], sep, enc


def resolve_columns(cols: list[str], spec: dict, name: str, required: tuple = ()) -> dict:
    """정의서 영어 코드로 실제 컬럼 찾기. "운행일자(OPRAT_DATE)"처럼 붙은 표기도 됨. 반환 {실제 컬럼: 우리 이름}"""
    out = {}
    for key, code in spec.items():
        hit = next((c for c in cols if c.upper() == code or c.upper().endswith(f"({code})") or c.replace(" ", "") == KR.get(code)), None)
        if hit is None and key in required:
            raise KeyError(f"{name}에 {code} 컬럼이 없습니다. 실제 컬럼: {cols}")
        if hit is not None:
            out[hit] = key
    return out


def read_filtered(path: Path, spec: dict, name: str, key: str, keep, required: tuple = ()) -> pd.DataFrame:
    """큰 파일을 나눠 읽으면서 key 컬럼 값이 keep(함수)에 맞는 행만 남김. 전부 문자로."""
    if path.suffix.lower() == ".parquet":
        raw = pd.read_parquet(path)
        ren = resolve_columns(list(raw.columns), spec, name, required)
        df = raw[list(ren)].rename(columns=ren).astype(str)
        return df[keep(df[key])].reset_index(drop=True)
    cols, sep, enc = _header(path)
    ren = resolve_columns(cols, spec, name, required)
    parts = []
    for chunk in pd.read_csv(path, sep=sep, encoding=enc, usecols=list(ren), dtype=str, chunksize=CHUNK, low_memory=False):
        chunk = chunk.rename(columns=ren)
        parts.append(chunk[keep(chunk[key])])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(ren.values()))


# %%
def find_route_ids(route_file: Path, route_no: str) -> pd.DataFrame:
    """노선 정보에서 노선번호 = 9401 (값에 붙은 '번'은 떼고 비교)."""
    clean = lambda s: s.fillna("").str.strip().str.replace("번", "", regex=False)
    r = read_filtered(route_file, ROUTE_COLS, "노선 정보", "route_no", lambda s: clean(s) == route_no,
                      required=("route_id", "route_no"))
    if r.empty:
        raise ValueError(f"노선 정보에 노선번호 {route_no}가 없습니다 ({route_file.name}).")
    r["route_id"] = r["route_id"].str.strip()
    cand = r.drop_duplicates("route_id")
    print(f"노선 정보에서 {route_no} 노선ID 후보 {len(cand)}개:")
    show = [c for c in ("tcbo", "route_id", "route_name", "route_no", "n_stops") if c in cand]
    print(cand[show].to_string(index=False))
    return cand


def _to_degree(v: pd.Series) -> pd.Series:
    """좌표 문자열 -> 도(degree). '127123456'처럼 소수점 없이 오면 자릿수를 맞춤."""
    x = pd.to_numeric(v, errors="coerce")
    for _ in range(8):
        big = x > 1000
        if not big.any():
            break
        x = x.where(~big, x / 10)
    return x


def _lat_lon(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """두 좌표 컬럼 중 어느 쪽이 위도(33~39)이고 경도(124~132)인지 값으로 판단 (정의서의 X/Y 표기가 섞여 있음)."""
    a, b = _to_degree(a), _to_degree(b)
    if a.between(33, 39).mean() >= b.between(33, 39).mean():
        lat, lon = a, b
    else:
        lat, lon = b, a
    ok = lat.between(33, 39) & lon.between(124, 132)
    return lat.where(ok), lon.where(ok)


def build_stop_master(route_stop_file: Path, stop_file: Path, route_ids: list[str]) -> pd.DataFrame:
    """노선ID별 정류장 순서표 (후보가 여러 개면 전부 만들고 route_id 컬럼으로 구분)."""
    ids = set(route_ids)
    rs = read_filtered(route_stop_file, ROUTE_STOP_COLS, "노선정류장 정보", "route_id",
                       lambda s: s.fillna("").str.strip().isin(ids), required=("route_id", "seq", "sttn_id"))
    if rs.empty:
        raise ValueError(f"노선정류장 정보에 노선ID {sorted(ids)}가 없습니다.")
    rs["route_id"] = rs["route_id"].str.strip()
    rs["seq"] = pd.to_numeric(rs["seq"], errors="coerce").astype("Int64")
    # 운행일자별로 여러 번 들어 있으면 노선마다 가장 최근 것
    if "date" in rs:
        latest = rs.groupby("route_id")["date"].transform("max")
        n_dates = rs.groupby("route_id")["date"].nunique()
        if (n_dates > 1).any():
            print(f"노선정류장 정보 운행일자 {n_dates.to_dict()}개 중 가장 최근 것 사용")
        rs = rs[rs["date"] == latest]
    rs = rs.dropna(subset=["seq"]).drop_duplicates(["route_id", "seq"]).sort_values(["route_id", "seq"])
    rs["sttn_id"] = norm_id(rs["sttn_id"])

    st = read_filtered(stop_file, STOP_COLS, "정류장 정보", "sttn_id",
                       lambda s: norm_id(s.fillna("")).isin(set(rs["sttn_id"])), required=("sttn_id",))
    st["sttn_id"] = norm_id(st["sttn_id"])
    if "date" in st:
        st = st.sort_values("date")
    st = st.drop_duplicates("sttn_id", keep="last").set_index("sttn_id")

    out = pd.DataFrame({"route_id": rs["route_id"].to_numpy(), "seq": rs["seq"].astype(int).to_numpy(),
                        "sttn_id": rs["sttn_id"].to_numpy()})
    get = lambda col: out["sttn_id"].map(st[col]) if col in st else pd.Series(np.nan, index=out.index)
    rs_get = lambda col: pd.Series(rs[col].to_numpy(), index=out.index) if col in rs else pd.Series(np.nan, index=out.index)
    out["stop_name"] = get("name").fillna(rs_get("name")).fillna(out["sttn_id"])
    out["ars"] = get("ars").fillna(rs_get("ars")).fillna("")
    lat, lon = _lat_lon(get("x"), get("y"))
    lat2, lon2 = _lat_lon(rs_get("x"), rs_get("y"))  # 정류장 정보에 좌표가 없으면 노선정류장 정보 좌표
    out["lat"], out["lon"] = lat.fillna(lat2), lon.fillna(lon2)
    out["area_cd"] = get("area").fillna("")
    out = _add_seoul_and_segment(out)
    has = out["lat"].notna()
    x, y = _TRANSFORMER.transform(out.loc[has, "lon"].to_numpy(), out.loc[has, "lat"].to_numpy())
    out["utmk_x"], out["utmk_y"] = np.nan, np.nan
    out.loc[has, "utmk_x"], out.loc[has, "utmk_y"] = x, y
    return add_stop_label(out)


def _add_seoul_and_segment(df: pd.DataFrame) -> pd.DataFrame:
    """서울 여부와 구간 이름. 법정동코드가 없으면 좌표(서울 대략 범위)로."""
    df = df.copy()
    by_area = df["area_cd"].astype(str).str.strip()
    by_coord = df["lat"].between(37.42, 37.72) & df["lon"].between(126.76, 127.19)
    df["is_seoul"] = np.where(by_area != "", by_area.str.startswith(SEOUL_AREA_PREFIX), by_coord)
    df["segment"] = ""
    for rid, g in df.groupby("route_id"):
        seoul = g.loc[g["is_seoul"], "seq"]
        if seoul.empty:
            df.loc[g.index, "segment"] = "서울_없음"
            continue
        lo, hi = seoul.min(), seoul.max()
        df.loc[g.index, "segment"] = np.select(
            [g["seq"] < lo, g["seq"] > hi], ["성남_상행", "성남_하행"], "서울_도심루프")
    return df


def add_stop_label(df: pd.DataFrame) -> pd.DataFrame:
    """이름이 같은 정류장(순천향대학병원처럼 루프의 처음과 끝)을 seq로 구분한 이름."""
    df = df.copy()
    dup = df.groupby(["route_id", "stop_name"])["seq"].transform("nunique") > 1
    df["stop_label"] = np.where(dup, df["stop_name"] + "(seq" + df["seq"].astype(str) + ")", df["stop_name"])
    return df


def _norm_name(s: str) -> str:
    return "".join(ch for ch in str(s) if ch.isalnum())


def add_direction(df: pd.DataFrame, turnaround_name: str) -> pd.DataFrame:
    """회차 정류장(서울역버스환승센터) 기준으로 방향.

    direction = "in"  : 회차 정류장 전까지 (성남 -> 서울역 방향, 서울 진입)
                "out" : 회차 정류장부터 (서울역 -> 성남 방향, 서울 진출)
    회차 정류장은 이름으로 찾는다 (노선정류장 정보의 순번이 공개 데이터 번호와 다를 수 있어서).
    이름이 없으면 서울 첫 정류장에서 가장 먼 서울 정류장을 회차 지점으로 본다 (경고 출력).
    """
    df = df.copy()
    df["direction"], df["turn_seq"] = "", np.nan
    keys = [_norm_name(turnaround_name), "서울역"]  # 정확한 이름이 없으면 "서울역"이 들어간 정류장
    for rid, g in df.groupby("route_id"):
        seoul = g[g["segment"] == "서울_도심루프"].sort_values("seq")
        if seoul.empty:
            continue
        names = seoul["stop_name"].map(_norm_name)
        hit = seoul.iloc[0:0]
        for key in keys:
            hit = seoul[names.str.contains(key, regex=False)]
            if len(hit):
                break
        if len(hit):
            turn = int(hit["seq"].iloc[0])
        else:
            first = seoul.iloc[0]
            d = np.hypot(seoul["utmk_x"] - first["utmk_x"], seoul["utmk_y"] - first["utmk_y"])
            turn = int(seoul.loc[d.idxmax(), "seq"])
            print(f"⚠ 노선ID {rid}: 회차 정류장 '{turnaround_name}'을 이름으로 못 찾음 -> "
                  f"서울 첫 정류장에서 가장 먼 seq{turn} {seoul.loc[d.idxmax(), 'stop_name']}을 회차로 봄 (CONFIG turnaround_name 확인)")
        df.loc[g.index, "direction"] = np.where(g["seq"] < turn, "in", "out")
        df.loc[g.index, "turn_seq"] = turn
    return df


VIRTUAL_SEGMENT = "가상정류장"


def mark_virtual(stops: pd.DataFrame, events: pd.DataFrame | None = None) -> pd.DataFrame:
    """가상 정류장(요금·위치 계산용으로 노선에만 있고 사람이 타고 내리지 않는 곳)을 분석에서 뺀다.

    가상 = 이름에 "가상"이 있거나, 이름이 숫자뿐이거나(예: 8502913), 한 달 동안 승하차 태그가 한 번도 없는 정류장
    버스 재차 계산(03번)에는 그대로 두고, 구간 이름만 "가상정류장"으로 바꿔서 서울 도심 분석·추천·그림에서 빠지게 한다.
    """
    df = stops.copy()
    name = df["stop_name"].astype(str)
    virtual = name.str.contains("가상") | name.str.fullmatch(r"\s*\d+\s*")
    if events is not None and len(events):
        tapped = events.groupby("seq")["arr_observed"].any()
        virtual |= ~df["seq"].map(tapped).fillna(False).astype(bool)
    df["is_virtual"] = virtual
    df.loc[virtual, "segment"] = VIRTUAL_SEGMENT
    return df


def summary(stop_master: pd.DataFrame) -> None:
    for rid, g in stop_master.groupby("route_id"):
        seoul = g[g["segment"] == "서울_도심루프"]
        turn = g["turn_seq"].dropna()
        turn_txt = ""
        if len(turn):
            t = int(turn.iloc[0])
            turn_txt = f", 회차 seq{t} {g.loc[g['seq'] == t, 'stop_name'].iloc[0]} (이전 = 서울역 방향, 이후 = 성남 방향)"
        print(f"노선ID {rid}: 정류장 {len(g)}개, 좌표 있음 {int(g['lat'].notna().sum())}개, "
              f"서울 구간 seq {seoul['seq'].min() if len(seoul) else '-'}~{seoul['seq'].max() if len(seoul) else '-'} "
              f"({len(seoul)}개){turn_txt}")
