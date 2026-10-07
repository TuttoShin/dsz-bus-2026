# 정류장 좌표 -> 행정동 매칭 (공개 데이터. RunAll.py가 안심구역 안에서 실행)
"""
02bStopDong.py

서울시 버스정류소 위치정보에는 정류장이 어느 동에 있는지 정보가 없다(좌표만 있음).
그래서 행정동 경계 지도에 정류장 좌표를 찍어서, 어느 동 안에 들어가는지 판정한다.
05번이 정류장별로 "그 동네 도로가 얼마나 막히는가"를 계산할 때 쓴다.

1. 02StopMaster.py 결과에서 서울 좌표가 있는 정류장만 가져온다
2. 행정동 경계(GeoJSON)를 읽는다
   출처: github.com/vuski/admdongkor (통계청 행정동 경계, 2026-07 기준)
   data/raw/HangJeongDongVer20260701.geojson 에 받아둘 것
3. 정류장 좌표가 어느 동 경계 안에 있는지 판정 (point-in-polygon)
4. data/processed/stop_dong_9401.parquet 저장

CLI: python data-engineering/02bStopDong.py --route 9401
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


# %%
def _point_in_ring(x: float, y: float, ring: np.ndarray) -> bool:
    """점에서 오른쪽으로 선을 그었을 때 경계선과 홀수 번 만나면 안쪽 (ray casting)."""
    xs, ys = ring[:, 0], ring[:, 1]
    xs2, ys2 = np.roll(xs, -1), np.roll(ys, -1)
    crosses = (ys > y) != (ys2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_at_y = xs + (y - ys) * (xs2 - xs) / (ys2 - ys)
    return bool(np.sum(crosses & (x < x_at_y)) % 2 == 1)


def _point_in_polygon(x: float, y: float, polygon: list) -> bool:
    """polygon = [바깥 경계, 구멍1, 구멍2, ...]. 바깥 안이고 구멍 밖이어야 안쪽."""
    if not _point_in_ring(x, y, np.asarray(polygon[0])):
        return False
    return not any(_point_in_ring(x, y, np.asarray(hole)) for hole in polygon[1:])


# %%
def load_dong_boundaries(path: Path, sido: str = "서울특별시") -> list[dict]:
    """행정동 경계 중 서울만 남긴다. 각 동: 이름, 코드, 폴리곤 목록."""
    with open(path, encoding="utf-8") as f:
        gj = json.load(f)
    dongs = []
    for feat in gj["features"]:
        prop = feat["properties"]
        if prop.get("sidonm") != sido:
            continue
        geom = feat["geometry"]
        polygons = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        dongs.append({
            "sido_nm": prop["sidonm"],
            "sgg_nm": prop["sggnm"],
            "dong_nm": prop["adm_nm"].split()[-1],  # "서울특별시 중구 명동" -> "명동"
            "adm_cd": prop.get("adm_cd2") or prop.get("adm_cd"),
            "polygons": polygons,
        })
    return dongs


# %%
def match_stops_to_dong(stops: pd.DataFrame, dongs: list[dict]) -> pd.DataFrame:
    """정류장마다 좌표(경도 lon, 위도 lat)가 들어가는 행정동을 찾는다."""
    rows = []
    for s in stops.itertuples():
        hit = next(
            (d for d in dongs if any(_point_in_polygon(s.lon, s.lat, poly) for poly in d["polygons"])),
            None,
        )
        rows.append({
            "표준버스정류장ID": s.표준버스정류장ID, "seq": s.seq, "stop_name": s.stop_name,
            "sido_nm": hit["sido_nm"] if hit else None,
            "sgg_nm": hit["sgg_nm"] if hit else None,
            "dong_nm": hit["dong_nm"] if hit else None,
            "adm_cd": hit["adm_cd"] if hit else None,
        })
    out = pd.DataFrame(rows)
    missing = out["dong_nm"].isna().sum()
    if missing:
        print(f"⚠ 동을 못 찾은 정류장 {missing}개 (좌표가 경계 밖?)")
    return out


# %%
def build_stop_dong(stop_master: pd.DataFrame, boundary_path: Path) -> pd.DataFrame:
    """02번 결과에서 서울 좌표 있는 정류장만 골라 행정동을 붙인다."""
    stops = stop_master.dropna(subset=["lat", "lon"]).drop_duplicates("seq").copy()
    stops["표준버스정류장ID"] = stops["표준버스정류장ID"].astype(str)
    return match_stops_to_dong(stops, load_dong_boundaries(boundary_path))


# %%
def main() -> None:
    parser = argparse.ArgumentParser(description="정류장 좌표 -> 행정동")
    parser.add_argument("--route", type=str, default="9401")
    parser.add_argument("--boundary-file", type=str, default="HangJeongDongVer20260701.geojson")
    args = parser.parse_args()

    stop_master = pd.read_parquet(PROCESSED_DIR / f"stop_master_{args.route}.parquet")
    result = build_stop_dong(stop_master, RAW_DIR / args.boundary_file)

    out_path = PROCESSED_DIR / f"stop_dong_{args.route}.parquet"
    result.to_parquet(out_path, index=False)
    print(result[["seq", "stop_name", "sgg_nm", "dong_nm"]].to_string(index=False))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
