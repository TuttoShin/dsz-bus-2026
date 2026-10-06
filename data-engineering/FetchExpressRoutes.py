# 정류장별 광역버스 노선 목록 + 배차간격 수집 (공공 API, 인터넷 필요 -> 안심구역 밖에서 실행)
"""
FetchExpressRoutes.py

9401 서울 정류장마다 "어떤 광역버스가 몇 분 간격으로 서는가"를 공공 API로 모은다.
03bRouteShare.py가 이걸로 9401 대기 비율을 계산한다.
  9401 대기 비율 = 9401 운행 대수 / 그 정류장 광역버스 전체 운행 대수

왜 API가 필요한가)
서울시 승하차 데이터에는 서울시 노선만 있어서, 서울역 같은 곳에 서는 경기도 광역버스가 빠져 있다.
(서울역버스환승센터도 승하차 데이터에서는 광역버스가 9401 하나뿐으로 나옴)

1. 서울특별시_정류소정보조회 getRouteByStation(arsId)
   - 정류장에 서는 모든 노선 (경기·인천 노선 포함)
   - busRouteType: 6 광역, 7 인천, 8 경기 / term: 배차간격(분)
2. 경기도_버스노선 조회 getBusRouteInfoItemv2(routeId)
   - 경기 노선은 서울 API에서 "8 경기"로만 나와서 광역인지 모름 -> 노선 유형(routeTypeCd) 확인
   - peekAlloc / nPeekAlloc: 평일 최소 / 최대 배차(분)

광역버스로 보는 기준
   - 서울 광역(6): 전부
   - 경기(8): routeTypeCd가 EXPRESS_GG_TYPES (11 직행좌석, 14 광역급행 M버스)
   - 인천(7): 서울 도심에 들어오는 인천 노선은 대부분 광역이라 전부 포함 (경기 API로 확인 불가)

인증키: 레포 맨 위 .env 파일의 DATA_GO_KR_KEY (Encoding/Decoding 둘 다 됨)
결과: data/raw/ExpressRoutesByStop.csv  (반입용. 이름에 언더바 없음)
응답 원본: data/raw/apiCache/ (나중에 다시 확인용)

CLI: python data-engineering/FetchExpressRoutes.py --route 9401
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CACHE_DIR = RAW_DIR / "apiCache"

SEOUL_URL = "http://ws.bus.go.kr/api/rest/stationinfo/getRouteByStation"
GG_ROUTE_URL = "https://apis.data.go.kr/6410000/busrouteservice/v2/getBusRouteInfoItemv2"

SEOUL_EXPRESS_TYPES = {"6", "7", "8"}  # 광역, 인천, 경기 (시내버스는 제외)
EXPRESS_GG_TYPES = {11, 14}  # 경기 노선 중 광역으로 볼 유형: 직행좌석형, 광역급행형(M버스)


# %%
def load_key() -> str:
    """.env에서 DATA_GO_KR_KEY를 읽는다. Encoding 키(%포함)면 풀어서 Decoding 키로 맞춘다."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        raise FileNotFoundError("레포 맨 위에 .env 파일이 없습니다. DATA_GO_KR_KEY=인증키 를 적어주세요.")
    env = {}
    for line in env_path.read_text(encoding="utf-8-sig").splitlines():
        name, sep, value = line.partition("=")
        if sep:
            env[name.strip()] = value.strip().strip('"').strip("'")
    key = env.get("DATA_GO_KR_KEY", "")
    if not key:
        raise KeyError(".env에 DATA_GO_KR_KEY가 없습니다.")
    return urllib.parse.unquote(key)


def _get_json(url: str, params: dict, key: str, cache_name: str) -> dict:
    """API 호출 + 응답 원본 저장. 실패하면 에러 메시지를 담은 dict."""
    query = urllib.parse.urlencode({"serviceKey": key, **params})
    try:
        raw = urllib.request.urlopen(f"{url}?{query}", timeout=20).read().decode("utf-8")
    except Exception as e:  # 인증 실패(401/403) 등
        return {"_error": str(e)}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{cache_name}.json").write_text(raw, encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_error": f"JSON 아님: {raw[:200]}"}


# %%
def fetch_routes_at_stop(ars_id: str, key: str) -> list[dict]:
    """1번 API: 정류장에 서는 노선 목록."""
    d = _get_json(SEOUL_URL, {"arsId": ars_id, "resultType": "json"}, key, f"seoulStation{ars_id}")
    if "_error" in d:
        raise RuntimeError(f"서울 API 실패 (arsId={ars_id}): {d['_error']}")
    header = d.get("msgHeader") or {}
    if str(header.get("headerCd", "0")) != "0":
        raise RuntimeError(f"서울 API 오류 (arsId={ars_id}): {header.get('headerMsg')}")
    return (d.get("msgBody") or {}).get("itemList") or []


def fetch_gg_route_info(route_id: str, key: str) -> dict:
    """2번 API: 경기 노선 유형, 평일 최소/최대 배차. 실패하면 빈 dict."""
    d = _get_json(GG_ROUTE_URL, {"routeId": route_id, "format": "json"}, key, f"ggRoute{route_id}")
    if "_error" in d:
        print(f"⚠ 경기 API 실패 (routeId={route_id}): {d['_error']}")
        return {}
    body = (d.get("response") or d).get("msgBody") or {}
    return body.get("busRouteInfoItem") or {}


def _to_int(x) -> int | None:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


# %%
def build_express_routes(stops: pd.DataFrame, key: str) -> pd.DataFrame:
    rows = []
    gg_cache: dict[str, dict] = {}
    for s in stops.itertuples():
        for it in fetch_routes_at_stop(s.ars_id, key):
            seoul_type = str(it.get("busRouteType"))
            if seoul_type not in SEOUL_EXPRESS_TYPES:
                continue  # 시내버스 제외
            route_id = str(it.get("busRouteId"))
            row = {
                "seq": s.seq, "stop_id": s.stop_id, "ars_id": s.ars_id, "stop_name": s.stop_name,
                "route_name": it.get("busRouteNm"), "route_id": route_id,
                "seoul_route_type": seoul_type, "term_min": _to_int(it.get("term")),
                "gg_route_type_cd": None, "gg_route_type_name": None,
                "peek_alloc_min": None, "npeek_alloc_min": None,
            }
            if seoul_type == "8":  # 경기 노선: 광역인지 확인
                if route_id not in gg_cache:
                    gg_cache[route_id] = fetch_gg_route_info(route_id, key)
                    time.sleep(0.1)
                info = gg_cache[route_id]
                row.update({
                    "gg_route_type_cd": _to_int(info.get("routeTypeCd")),
                    "gg_route_type_name": info.get("routeTypeName"),
                    "peek_alloc_min": _to_int(info.get("peekAlloc")),
                    "npeek_alloc_min": _to_int(info.get("nPeekAlloc")),
                })
            rows.append(row)
        time.sleep(0.1)

    df = pd.DataFrame(rows)
    df["is_express"] = df.apply(
        lambda r: True if r["seoul_route_type"] in ("6", "7")
        else (r["gg_route_type_cd"] in EXPRESS_GG_TYPES if pd.notna(r["gg_route_type_cd"]) else None),
        axis=1,
    )
    df["fetched_at"] = date.today().isoformat()
    unknown = df[df["is_express"].isna()]
    if not unknown.empty:
        print(f"⚠ 경기 노선 유형을 못 받은 노선 {unknown['route_name'].nunique()}개 -> is_express 빈 값 (직접 확인 필요)")
    return df


# %%
def main() -> None:
    parser = argparse.ArgumentParser(description="정류장별 광역버스 노선 + 배차 수집")
    parser.add_argument("--route", type=str, default="9401")
    args = parser.parse_args()

    key = load_key()
    sm = pd.read_parquet(PROCESSED_DIR / f"stop_master_{args.route}.parquet")
    stops = sm.dropna(subset=["lat"]).drop_duplicates("seq").rename(
        columns={"표준버스정류장ID": "stop_id", "버스정류장ARS번호": "ars_id"}
    )[["seq", "stop_id", "ars_id", "stop_name"]]
    stops["stop_id"] = stops["stop_id"].astype(str)

    df = build_express_routes(stops, key)
    out_path = RAW_DIR / "ExpressRoutesByStop.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(df.groupby(["seq", "stop_name"])["route_name"].apply(lambda x: ", ".join(map(str, x))).to_string())
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
