"""
01IngestBus.py

서울 열린데이터광장 "버스노선별_정류장별_시간대별_승하차_인원_정보" 공공데이터
원본 CSV를 읽고, 다음 전처리를 수행한다.

1. cp949 인코딩으로 로드
2. '역명' 컬럼 끝의 순번을 파싱해 노선 내 정류장 순서를 복원
3. 가상 정류장(한남대교 등, 경로 표시용이라 승하차X) 제거
4. 00~23시 승차/하차 wide 컬럼을 정류장, 시간대 단위 long 포맷으로 변환 (pandas 사용하려고...)
5. data/processed/ 에 parquet으로 저장

Jupyter: VSCode에서는 # %% 블록 단위로 인터랙티브 실행 가능
CLI: python data-engineering/01IngestBus.py --route 9401
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

# 원본 컬럼명은 '00시승차총승객수' ~ '23시승차총승객수'
HOURS = list(range(24))


def _board_col(h: int) -> str:
    return f"{h:02d}시승차총승객수" if h == 0 else f"{h}시승차총승객수"


def _alight_col(h: int) -> str:
    return f"{h:02d}시하차총승객수" if h == 0 else f"{h}시하차총승객수"


# %%
def load_raw(path: Path) -> pd.DataFrame:
    """원본 CSV 로드. ARS번호에 '~' 같은 비숫자 값이 섞여 있어 문자열로 고정."""
    df = pd.read_csv(
        path,
        encoding="cp949",
        dtype={"버스정류장ARS번호": str, "교통수단타입코드": str},
        low_memory=False,
    )
    return df


# %%
def parse_stop_order(df: pd.DataFrame) -> pd.DataFrame:
    """역명 끝 괄호 숫자를 노선 내 정류장 순번(seq)으로 분리하고, 가상 정류장을 제거."""
    df = df.copy()
    df["seq"] = df["역명"].str.extract(r"\((\d+)\)$")
    df = df.dropna(subset=["seq"])
    df["seq"] = df["seq"].astype(int)
    df["stop_name"] = df["역명"].str.replace(r"\(\d+\)$", "", regex=True).str.strip()
    df = df[~df["stop_name"].str.contains("가상")]
    return df


# %%
def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """wide(시간대별 컬럼) -> long(정류장 x 시간대 x 승차/하차) 변환."""
    id_cols = [
        "사용년월",
        "노선번호",
        "노선명",
        "표준버스정류장ID",
        "버스정류장ARS번호",
        "stop_name",
        "seq",
        "교통수단타입명",
    ]
    records = []
    for h in HOURS:
        b_col, a_col = _board_col(h), _alight_col(h)
        chunk = df[id_cols + [b_col, a_col]].copy()
        chunk = chunk.rename(columns={b_col: "boarding", a_col: "alighting"})
        chunk["hour"] = h
        records.append(chunk)
    long_df = pd.concat(records, ignore_index=True)
    return long_df.sort_values(["노선번호", "seq", "hour"]).reset_index(drop=True)


# %%
def ingest(raw_path: Path, route: str | None = None) -> pd.DataFrame:
    df = load_raw(raw_path)
    df = parse_stop_order(df)
    if route is not None:
        df = df[df["노선번호"].astype(str) == route]
    long_df = to_long(df)
    return long_df


# %%
def main() -> None:
    parser = argparse.ArgumentParser(description="버스 승하차 원본 CSV 인제스트")
    parser.add_argument(
        "--raw-file",
        type=str,
        default="2026년버스노선별정류장별시간대별승하차인원정보(08월).csv",
        help="data/raw/ 아래 원본 CSV 파일명",
    )
    parser.add_argument(
        "--route",
        type=str,
        default=None,
        help="특정 노선번호만 필터링 (예: 9401). 미지정 시 전체 노선 처리",
    )
    args = parser.parse_args()

    raw_path = RAW_DIR / args.raw_file
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} 를 찾을 수 없습니다. 원본 CSV를 data/raw/ 에 넣어주세요."
        )

    long_df = ingest(raw_path, route=args.route)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.route}" if args.route else "_all"
    out_path = PROCESSED_DIR / f"bus_long{suffix}.parquet"
    long_df.to_parquet(out_path, index=False)

    print(f"rows: {len(long_df):,}")
    print(f"routes: {long_df['노선번호'].nunique()}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()