# 교통카드 이용내역에서 9401 기록 추리기 + 9401 정류장의 전체 노선 승차 집계
"""
02CardExtract.py

교통카드 이용내역(수도권)은 매우 크다. 한 번만 훑으면서 두 가지를 모은다.
1. 9401 노선ID 후보(01번)의 행 -> 03번 버스별 계산에 씀
   필요한 컬럼만 읽는다 (가상카드번호·트랜잭션ID는 읽지도 않음)
2. 9401 정류장에서 승차한 모든 노선의 승차 인원 (정류장 x 시간대 x 9401여부 합계)
   -> 04번 9401 분담률 = 9401 승차 / 그 정류장 전체 노선 승차
3. 서울 구간 9401 정류장끼리 오간 모든 노선 기록 (승차 정류장 x 하차 정류장 x 날짜 x 시간대 x 노선 x 차량)
   -> 04번 "정류장 A에서 다른 버스를 타고 정류장 B로 가는 시간"(차내 시간 + 그 구간 버스 대기)
   -> 09번 추천 서비스: 지금 정류장보다 덜 붐비는 9401 정류장으로 버스 타고 이동

사람 수는 줄 수가 아니라 이용자수(USER_CNT)의 합으로 센다.
후보 노선ID가 여러 개면(다른 지역 9401 등) 우리 정류장에서 기록이 가장 많은 것 하나를 고른다.

컬럼 (정의서): OPRAT_DATE, TCBO_ID, TCBO_CAR_ID, TRFFC_WAY_CD, TCBO_ROUTE_ID, GIN_DT, TCBO_GIN_STTN_ID,
               GFF_DT, TCBO_GFF_STTN_ID, USER_CNT

저장해 두고 다시 쓰기 (extract_cached)
   교통카드 전체를 훑는 데 1시간 넘게 걸려서, 결과를 data/processed/cardCache/ 에 저장한다.
   다음 실행 때 입력(카드 파일 목록·크기·수정시각, 기준 달, 9401 노선ID 후보, 정류장ID, 통합 ID)이 같으면
   저장한 것을 읽기만 한다 (몇 초). 하나라도 바뀌면 자동으로 다시 훑는다.
   강제로 다시 훑으려면 RunAll.py --recard
   cardCache/ 는 행 단위 자료라 반출 X
"""

from __future__ import annotations

import glob
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE.parent / "data" / "raw"

CARD_COLS = {
    "date": "OPRAT_DATE", "tcbo": "TCBO_ID", "car": "TCBO_CAR_ID", "way": "TRFFC_WAY_CD",
    "route": "TCBO_ROUTE_ID", "gin_dt": "GIN_DT", "gin_stop": "TCBO_GIN_STTN_ID",
    "gff_dt": "GFF_DT", "gff_stop": "TCBO_GFF_STTN_ID", "users": "USER_CNT",
}
CHUNK = 2_000_000


def _m01():
    if "m01RouteStops" not in sys.modules:
        spec = importlib.util.spec_from_file_location("m01RouteStops", HERE / "01RouteStops.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules["m01RouteStops"] = m
        spec.loader.exec_module(m)
    return sys.modules["m01RouteStops"]


def card_files(pattern: str) -> list[Path]:
    p = Path(pattern)
    hits = sorted(glob.glob(str(p if p.is_absolute() else RAW_DIR / pattern)))
    if not hits:
        raise FileNotFoundError(f"교통카드 파일이 없습니다: {pattern} (data/raw 기준)")
    return [Path(h) for h in hits]


def read_chunks(paths: list[Path]):
    """필요한 컬럼만 나눠서 읽기. 전부 문자, 이용자수만 숫자."""
    m01 = _m01()
    for path in paths:
        if path.suffix.lower() == ".parquet":
            raw = pd.read_parquet(path)
            ren = m01.resolve_columns(list(raw.columns), CARD_COLS, "교통카드", tuple(CARD_COLS))
            yield _clean(raw[list(ren)].rename(columns=ren).astype(str))
            continue
        cols, sep, enc = m01._header(path)
        ren = m01.resolve_columns(cols, CARD_COLS, "교통카드", tuple(CARD_COLS))
        print(f"[{path.name}] 구분자 {sep!r}, 인코딩 {enc}")
        for chunk in pd.read_csv(path, sep=sep, encoding=enc, usecols=list(ren), dtype=str,
                                 chunksize=CHUNK, low_memory=False):
            yield _clean(chunk.rename(columns=ren))


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ("date", "tcbo", "car", "way", "route", "gin_dt", "gin_stop", "gff_dt", "gff_stop"):
        df[c] = df[c].fillna("").astype(str).str.strip().replace({"nan": "", "None": ""})
    df["users"] = pd.to_numeric(df["users"], errors="coerce").fillna(1)
    return df


# %%
def extract(paths: list[Path], route_ids: list[str], stop_ids: set[str], seoul_ids: set[str] | None = None,
            card_ym: str | None = None, aliases: dict | None = None) -> dict:
    """한 번 훑어서 9401 후보 노선 행 + 9401 정류장 전체 노선 승차 집계 + 서울 정류장끼리 오간 기록.

    반환 {"card": 9401 행, "route_id": 고른 노선ID, "stop_board": 정류장x시간대x9401여부x날짜 승차 합,
          "pair_trips": 서울 정류장 A->B 이동 집계, "n_rows": 전체 행 수, "days": 날짜 범위}
    card_ym (예: "202408")을 주면 그 달 운행일자만 쓴다 (SKT와 같은 달로 맞춤)
    노선ID는 앞자리 0을 떼고 비교한다 (노선 정보와 교통카드의 표기 차이 대비)
    stop_board는 날짜별로 남긴다 (요일별 정류장 대기 수요·평일 점유율·시간 분할 검증에 씀)
    aliases = {다른 정산사업자 정류장ID: 9401 정류장ID} (01번 stop_aliases). 정류장별 승차·정류장 간 이동에 합쳐 센다
    """
    norm = _m01().norm_id
    aliases = aliases or {}
    alias_users, stop_users = 0, 0
    ids = {str(r).strip().lstrip("0") for r in route_ids}
    n_other_month = 0
    seoul_ids = seoul_ids or set()
    keep, share, pairs, days = [], [], [], set()
    n_rows = 0
    for df in read_chunks(paths):
        if card_ym:
            in_month = df["date"].str[:6] == card_ym
            n_other_month += int((~in_month).sum())
            df = df[in_month]
        n_rows += len(df)
        days.update(df["date"].unique())
        df = df.assign(route=df["route"].str.lstrip("0"))
        keep.append(df[df["route"].isin(ids)])
        g = norm(df["gin_stop"])
        if aliases:  # 같은 정류장 다른 ID -> 9401 정류장ID
            is_alias = g.isin(aliases.keys())
            alias_users += int(df.loc[is_alias, "users"].sum())
            g = g.where(~is_alias, g.map(aliases))
        at = df[g.isin(stop_ids)].assign(stop=g[g.isin(stop_ids)])
        stop_users += int(at["users"].sum())
        if len(at):
            at["hour"] = pd.to_numeric(at["gin_dt"].str[8:10], errors="coerce")
            at["is_route"] = at["route"].isin(ids)
            share.append(at.assign(date=at["date"].str[:8]).groupby(["stop", "hour", "is_route", "route", "date"])["users"].sum())
        # 서울 정류장 A에서 타서 서울 정류장 B에서 내린 기록 (모든 노선)
        o = norm(df["gff_stop"])
        if aliases:
            o = o.where(~o.isin(aliases.keys()), o.map(aliases))
        ab = df[g.isin(seoul_ids) & o.isin(seoul_ids) & (g != o)].assign(a=g, b=o)
        if len(ab):
            t0 = pd.to_datetime(ab["gin_dt"].str[:14], format="%Y%m%d%H%M%S", errors="coerce")
            t1 = pd.to_datetime(ab["gff_dt"].str[:14], format="%Y%m%d%H%M%S", errors="coerce")
            ab = ab.assign(ride=(t1 - t0).dt.total_seconds() / 60, hour=t0.dt.hour)
            ab = ab[(ab["ride"] > 0) & (ab["ride"] < 60)]
            pairs.append(ab.groupby(["a", "b", "date", "hour", "route", "car"])["ride"].agg(["sum", "size"]))
    if n_other_month:
        print(f"교통카드 {card_ym} 이외 운행일자 {n_other_month:,}행은 제외")
    card = pd.concat(keep, ignore_index=True)
    if card.empty:
        raise ValueError(f"교통카드에 노선ID {sorted(ids)} 기록이 없습니다. 노선 정보의 ROUTE_ID와 카드의 "
                         "TCBO_ROUTE_ID 체계가 다른지 확인 (CONFIG route_id에 직접 지정 가능)")
    counts = card.groupby("route").size().sort_values(ascending=False)
    print(f"교통카드 전체 {n_rows:,}행, 노선ID별 기록 수: {counts.to_dict()}")
    chosen = counts.index[0]
    # 같은 9401이 정산사 등에 따라 노선ID가 둘 이상으로 나뉘어 있을 수 있음
    # -> 승차 정류장이 기록이 가장 많은 노선ID와 80% 이상 겹치면 같은 노선으로 보고 합친다 (다른 지역 9401은 안 겹침)
    stops_of = card.groupby("route")["gin_stop"].apply(lambda x: set(norm(x)))
    main = stops_of[chosen]
    merged = [r for r in counts.index if len(stops_of[r] & main) >= 0.8 * max(len(stops_of[r]), 1)]
    if len(merged) > 1:
        print(f"노선ID {merged}는 정류장이 같아서 같은 9401로 합침")
    card = card[card["route"].isin(merged)].reset_index(drop=True)
    # 차량ID는 정산사(노선ID)마다 따로 매겨질 수 있어서 노선ID를 붙여 구분
    card["car"] = card["route"] + ":" + card["car"]
    route_counts = {str(k): int(v) for k, v in counts.items()}
    card["gin_time"] = pd.to_datetime(card["gin_dt"].str[:14], format="%Y%m%d%H%M%S", errors="coerce")
    card["gff_time"] = pd.to_datetime(card["gff_dt"].str[:14], format="%Y%m%d%H%M%S", errors="coerce")
    stop_board = pd.concat(share).groupby(level=[0, 1, 2, 3, 4]).sum().reset_index() if share else pd.DataFrame()
    stop_routes = pd.Series(dtype=float)
    if len(stop_board):  # 합친 노선ID만 9401로 (다른 후보는 다른 노선 취급)
        stop_board["is_route"] = stop_board["route"].isin(merged)
        # 정류장별 승차 기록이 있는 노선 수 (9401 합친 ID는 1개로). 한 달 승차 30명 미만 노선은 경유·오기록으로 보고 뺌
        rt = stop_board.assign(route=stop_board["route"].where(~stop_board["is_route"], chosen))
        rt = rt.groupby(["stop", "route"])["users"].sum()
        stop_routes = rt[rt >= 30].groupby(level=0).size()
        stop_board = stop_board.groupby(["stop", "hour", "is_route", "date"])["users"].sum().reset_index()
    pair_trips = pd.DataFrame()
    if pairs:
        pair_trips = pd.concat(pairs).groupby(level=[0, 1, 2, 3, 4, 5]).sum().reset_index()
        pair_trips = pair_trips.rename(columns={"sum": "ride_sum", "size": "n"})
    days = sorted(d for d in days if d)
    if len(pair_trips):  # 9401은 다른 버스 이동에서 뺄 수 있게 표시를 하나로
        pair_trips.loc[pair_trips["route"].isin(merged), "route"] = chosen
    return {"card": card, "route_id": chosen, "route_ids_merged": merged, "route_counts": route_counts,
            "stop_board": stop_board, "stop_routes": stop_routes, "pair_trips": pair_trips, "n_rows": n_rows,
            "alias_board_share": alias_users / stop_users if stop_users else 0.0,  # 9401 정류장 승차 중 통합 ID로 들어온 비율
            "days": f"{days[0]} ~ {days[-1]}" if days else ""}


# %%
CACHE_TABLES = ["card", "stop_board", "pair_trips", "stop_routes"]  # parquet으로 저장하는 표
CACHE_VALUES = ["route_id", "route_ids_merged", "route_counts", "n_rows", "alias_board_share", "days"]  # meta.json


def cache_key(paths: list[Path], route_ids: list[str], stop_ids: set[str], seoul_ids: set[str] | None,
              card_ym: str | None, aliases: dict | None) -> str:
    """입력이 같은지 확인하는 지문. 카드 파일은 내용 대신 이름·크기·수정시각으로 (수십 GB를 다시 읽지 않게)."""
    files = [[str(p), p.stat().st_size, int(p.stat().st_mtime)] for p in paths]
    blob = json.dumps({
        "files": files, "card_ym": card_ym, "route_ids": sorted(map(str, route_ids)),
        "stop_ids": sorted(stop_ids), "seoul_ids": sorted(seoul_ids or []),
        "aliases": sorted((aliases or {}).items()), "version": 1,
    }, ensure_ascii=False)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()


def _save_cache(ext: dict, cache_dir: Path, key: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for name in CACHE_TABLES:
        v = ext.get(name)
        if isinstance(v, pd.Series):
            v = v.rename("n").rename_axis("stop").reset_index()
        if v is None or len(v) == 0:
            continue
        v.to_parquet(cache_dir / f"{name}.parquet", index=False)
        saved.append(name)
    meta = {"key": key, "tables": saved, **{k: ext[k] for k in CACHE_VALUES}}
    # meta.json을 마지막에 써야 중간에 멈췄을 때 반쪽짜리 저장본을 쓰지 않음
    (cache_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def _load_cache(cache_dir: Path, meta: dict) -> dict:
    ext = {k: meta[k] for k in CACHE_VALUES}
    for name in CACHE_TABLES:
        path = cache_dir / f"{name}.parquet"
        df = pd.read_parquet(path) if name in meta["tables"] else pd.DataFrame()
        if name == "stop_routes":
            df = df.set_index("stop")["n"] if len(df) else pd.Series(dtype=float)
        ext[name] = df
    return ext


def extract_cached(paths: list[Path], route_ids: list[str], stop_ids: set[str], seoul_ids: set[str] | None,
                   card_ym: str | None, aliases: dict | None, cache_dir: Path, refresh: bool = False) -> dict:
    """extract()와 같은 결과. 입력이 지난번과 같으면 저장해 둔 것을 읽고, 아니면 다시 훑고 저장한다."""
    key = cache_key(paths, route_ids, stop_ids, seoul_ids, card_ym, aliases)
    meta_path = cache_dir / "meta.json"
    if meta_path.exists() and not refresh:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("key") == key and all((cache_dir / f"{t}.parquet").exists() for t in meta.get("tables", [])):
            print(f"02 저장본 사용 ({cache_dir.name}/, 교통카드를 다시 훑지 않음. 다시 훑으려면 --recard)")
            return _load_cache(cache_dir, meta)
        print("02 저장본과 입력이 달라서(카드 파일·기준 달·노선·정류장 변경) 교통카드를 다시 훑음")
    elif refresh:
        print("--recard: 교통카드를 다시 훑음")
    ext = extract(paths, route_ids, stop_ids, seoul_ids, card_ym, aliases)
    _save_cache(ext, cache_dir, key)
    print(f"02 결과 저장 -> {cache_dir} (다음 실행부터 재사용)")
    return ext


def describe(card: pd.DataFrame) -> dict:
    """9401 행 상태 (화면 출력 + 10번 품질표). 인원 합계는 넣지 않음."""
    q = {
        "기간": f"{card['date'].min()} ~ {card['date'].max()}",
        "일수": int(card["date"].nunique()),
        "승차일시_읽기실패_비율": float(card["gin_time"].isna().mean()),
        "하차기록없음_비율": float((card["gff_stop"] == "").mean()),
        "이용자수2명이상_비율": float((card["users"] > 1).mean()),
        "하루_운행차량": round(float(card.groupby("date")["car"].nunique().mean()), 1),
        "정산사ID_종류수": int(card["tcbo"].nunique()),
    }
    for k, v in q.items():
        print(f"  {k}: {v:.1%}" if isinstance(v, float) and "비율" in k else f"  {k}: {v}")
    return q
