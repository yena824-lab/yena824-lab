#-*- coding: utf-8 -*-
"""
button.company_finance
======================

재무부문(금융/매출/성장률) 화면 모듈.

Streamlit 기반 BIGx 기업혁신성장 보고서에서
'재무부문' 탭을 렌더링하는 모듈이다.
공통 유틸리티 함수와 메인 렌더링 함수 :func:`render_finance`를 제공한다.
"""

import streamlit as st
import pandas as pd
import altair as alt
import os, math
import numpy as np
import json
import re
from typing import List, Dict, Any, Optional

from main.sql import sql_company_detail as Q

from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------
# 기업 유사도(ChromaDB) 기반 필터링 유틸
# --------------------------------------
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None
    Settings = None

CHROMA_DB_DIR = "/home/kibo/peoples/hanjoo/corp_similarity_final/corp_vector_db"
COLLECTION_NAME = "corp_vectors_1223"

# ---------------------------------------------------
# ✅ 세션 캐시(DF/유사ID) 유틸
# ---------------------------------------------------
def _fin_cache() -> dict:
    if "_fin_cache" not in st.session_state:
        st.session_state["_fin_cache"] = {}
    return st.session_state["_fin_cache"]


def _cache_get_df(key: tuple, loader_fn):
    """
    send_sql 결과 DF를 session_state에 캐시.
    key는 tuple 권장. loader_fn은 DF를 반환하는 함수.
    """
    cache = _fin_cache()
    if key in cache:
        df = cache[key]
        return df.copy() if isinstance(df, pd.DataFrame) else df
    df = loader_fn()
    cache[key] = df
    return df.copy() if isinstance(df, pd.DataFrame) else df


# ---------------------------------------------------
# 공통: 법인번호 컬럼명 통일 유틸
# ---------------------------------------------------
ID_COL_CANDIDATES = ["법인번호_ENC", "법인번호"]


def _get_id_col(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return "법인번호_ENC"
    for c in ID_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None


# ---------------------------------------------------
# ✅ 법인번호(평문) 표시 유틸: ENC 노출 방지
# ---------------------------------------------------
_RE_PLAIN_CORPNO = re.compile(r"^\d{7}-\d{6}$")


def _plain_corpno_or_blank(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    s = str(v).strip()
    return s if _RE_PLAIN_CORPNO.match(s) else ""


def _ensure_plain_corpno_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    df에 '법인번호' 컬럼을 보장.
    - 이미 '법인번호'가 있으면 13자리 숫자만 남김
    - 없으면 빈 문자열 컬럼 생성
    """
    if df is None or df.empty:
        return df
    out = df.copy()

    if "법인번호" in out.columns:
        out["법인번호"] = out["법인번호"].map(_plain_corpno_or_blank)
    else:
        out["법인번호"] = ""

    return out


def create_chroma_client(db_dir: str):
    if chromadb is None:
        raise RuntimeError("chromadb 패키지가 설치되어 있지 않습니다.")
    return chromadb.PersistentClient(
        path=db_dir,
        settings=Settings(anonymized_telemetry=False),
    )


def _parse_codes_from_metadata(meta: Dict[str, Any]) -> List[str]:
    codes_val = (meta or {}).get("codes")
    if codes_val is None:
        return []

    if isinstance(codes_val, list):
        return [str(x) for x in codes_val]

    if isinstance(codes_val, str):
        s = codes_val.strip()

        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass
        if "|" in s:
            return [c.strip() for c in s.split("|") if c.strip()]
        return [s]

    return []


def _has_code_in_metadata(meta: Dict[str, Any], code_filter: str) -> bool:
    return code_filter in _parse_codes_from_metadata(meta)


def _load_corp_order_from_collection(collection) -> List[str]:
    raw = (collection.metadata or {}).get("corp_order")
    if raw is None:
        raise ValueError("collection.metadata에 corp_order가 없습니다.")

    if isinstance(raw, list):
        corp_order = raw
    elif isinstance(raw, str):
        corp_order = json.loads(raw)
    else:
        raise ValueError(f"corp_order 타입 오류: {type(raw)}")

    if not corp_order:
        raise ValueError("corp_order가 비어 있습니다.")

    return corp_order


def aggregate_vector(vec: np.ndarray, code_order: List[str], level: str) -> np.ndarray:
    agg: Dict[str, float] = {}

    for i, code in enumerate(code_order):
        if level == "sub":
            key = code
        elif level == "mid":
            key = code[:4]
        elif level == "large":
            key = code[:2]
        else:
            raise ValueError(level)

        agg[key] = agg.get(key, 0.0) + float(vec[i])

    keys = sorted(agg.keys())
    return np.array([agg[k] for k in keys], dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(cosine_similarity([a], [b])[0][0])


# ---------------------------------------------------
# ✅ Chroma 전체 로딩/스캔 1회 캐시
# ---------------------------------------------------
@st.cache_resource(show_spinner=False)
def _get_chroma_collection_cached():
    client = create_chroma_client(CHROMA_DB_DIR)
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=None)

    code_order = _load_corp_order_from_collection(collection)

    # 가장 무거운 구간: 전체 embeddings/metadatas 로딩
    all_data = collection.get(include=["embeddings", "metadatas"])
    return collection, code_order, all_data


def _get_similar_ids_cached(
    base_id: str,
    code_filter: Optional[str],
    weights: Dict[str, float],
    sim_th: float = 0.0,
    num_th: int = 20,
) -> set:
    """
    (base_id, code_filter, weights, sim_th, num_th) 조합 단위로 유사 ID set 캐시
    """
    cache = _fin_cache()
    key = (
        "sim_ids",
        str(base_id),
        str(code_filter) if code_filter is not None else None,
        tuple(sorted((weights or {}).items())),
        float(sim_th),
        int(num_th),
    )
    if key in cache:
        return set(cache[key])

    if abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError("가중치 합은 반드시 1이어야 합니다.")

    collection, code_order, all_data = _get_chroma_collection_cached()

    base = collection.get(ids=[str(base_id)], include=["embeddings"])
    if not base["ids"]:
        cache[key] = set()
        return set()

    vec_sub_a = np.array(base["embeddings"][0], dtype=np.float32)
    vec_mid_a = aggregate_vector(vec_sub_a, code_order, "mid")
    vec_large_a = aggregate_vector(vec_sub_a, code_order, "large")

    results = []
    for cid, emb, meta in zip(all_data["ids"], all_data["embeddings"], all_data["metadatas"]):
        if cid == str(base_id):
            continue

        meta = meta or {}
        if code_filter and not _has_code_in_metadata(meta, code_filter):
            continue

        vec_sub_b = np.array(emb, dtype=np.float32)
        sim_sub = cosine(vec_sub_a, vec_sub_b)
        if sim_sub < sim_th:
            continue

        vec_mid_b = aggregate_vector(vec_sub_b, code_order, "mid")
        vec_large_b = aggregate_vector(vec_sub_b, code_order, "large")

        sim_mid = cosine(vec_mid_a, vec_mid_b)
        sim_large = cosine(vec_large_a, vec_large_b)

        final_sim = (
            weights["sub"] * sim_sub +
            weights["mid"] * sim_mid +
            weights["large"] * sim_large
        )
        results.append((cid, float(final_sim)))

    results.sort(key=lambda x: x[1], reverse=True)
    sim_ids = {cid for cid, _ in results[:num_th]}
    sim_ids.add(str(base_id))

    cache[key] = list(sim_ids)
    return sim_ids


def _toggle_fin_similarity_filter():
    st.session_state["fin_use_similarity"] = not st.session_state.get("fin_use_similarity", False)


def filter_peers_by_similarity(
    df_raw: pd.DataFrame,
    sel_subcat_code: str,
    applicant_corp_no: str,
) -> pd.DataFrame:
    """
    유사도 ON일 때만 적용.
    - df_raw는 SQL에서 이미 소분류로 제한된 결과라고 가정한다.
    - 소분류 내 기업 수가 50개 이하이면 유사도 ON이어도 그대로 반환.
    - 50개 초과일 때만, 신청기업 기준 유사기업 최대 20개(신청기업 포함)로 줄인다.
    """
    if df_raw is None or df_raw.empty:
        return df_raw

    v = st.session_state.get("fin_use_similarity", False)
    if isinstance(v, str):
        use_similarity = (v.strip().lower() in {"on", "true", "1", "yes", "y"})
    else:
        use_similarity = bool(v)

    if not use_similarity:
        return df_raw

    if not applicant_corp_no:
        return df_raw

    # 유사도 비교는 ENC 기준
    if "법인번호_ENC" in df_raw.columns:
        id_col = "법인번호_ENC"
    else:
        id_col = _get_id_col(df_raw)

    if not id_col or id_col not in df_raw.columns:
        return df_raw

    n_unique = int(df_raw[id_col].dropna().astype(str).nunique())
    if n_unique <= 50:
        return df_raw

    weights = {"sub": 1.0, "mid": 0.0, "large": 0.0}

    try:
        similar_ids = _get_similar_ids_cached(
            base_id=str(applicant_corp_no),
            code_filter=sel_subcat_code,
            weights=weights,
            sim_th=0.0,
            num_th=20,
        )
    except Exception:
        return df_raw

    if not similar_ids:
        return df_raw

    filtered = df_raw[df_raw[id_col].astype(str).isin(similar_ids)].copy()
    return df_raw if filtered.empty else filtered


# ---------------------------------------------------
# 공통 상수
# ---------------------------------------------------
BAD_VALUES = {"", "-", "None", "(상호 없음)", "nan", "NaN"}


# ---------------------------------------------------
# 숫자/텍스트 처리 함수들
# ---------------------------------------------------
def to_eok(v):
    """
    매출액 숫자를 '억원' 단위로 변환
    FINANCE_UNIT: won / thousand(default) / million / eok
    """
    UNIT_ENV = os.getenv("FINANCE_UNIT", "thousand").strip().lower()
    x = pd.to_numeric(v, errors="coerce")
    if pd.isna(x):
        return None
    if UNIT_ENV == "won":
        return x / 100_000_000
    if UNIT_ENV == "thousand":
        return x / 100
    if UNIT_ENV == "million":
        return (x * 1_000_000) / 100_000_000
    if UNIT_ENV == "eok":
        return x
    return x


def norm_id(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    return s[:-2] if s.endswith(".0") else s


def wrap4(s) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    if s.lower() in {"", "nan", "none"}:
        return ""
    return "\n".join([s[i: i + 4] for i in range(0, len(s), 4)])


def _has_applicant_2024_finance(df: pd.DataFrame) -> bool:
    """
    신청기업 2024 재무정보 존재 판정(매출액 기준)
    """
    if df is None or df.empty:
        return False

    for col in ["매출액_2024", "매출액"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if not s.empty:
                return True
    return False


def _is_corrected_flag(v) -> bool:
    """
    보정여부 규칙:
      - 0: 보정치
      - 1: 비보정(원값)
    """
    if v is None:
        return False
    try:
        if pd.isna(v):
            return False
    except Exception:
        pass
    try:
        return int(float(v)) == 0
    except Exception:
        s = str(v).strip()
        return s == "0"


def _fmt_num(v, decimals: int) -> str:
    x = pd.to_numeric(v, errors="coerce")
    if pd.isna(x):
        return ""
    try:
        if decimals <= 0:
            return f"{float(x):,.0f}"
        return f"{float(x):,.{decimals}f}"
    except Exception:
        return ""


def _pick_sales_for_growth(row: pd.Series, year: int):
    """
    성장률 계산용 매출액 선택 규칙:
    - f"매출액_{year}_보정여부"가 있고 값이 0이면(보정),
      보정 매출 후보 컬럼을 찾아 우선 사용
    - 없으면 원 매출액(매출액_{year}) fallback
    """
    base_col = f"매출액_{year}"
    flag_col = f"{base_col}_보정여부"
    base_val = row.get(base_col)

    if flag_col not in row.index:
        return base_val

    if _is_corrected_flag(row.get(flag_col)):
        candidates = [
            f"{base_col}_보정치",
            f"{base_col}_보정값",
            f"{base_col}_보정",
            f"{base_col}_corrected",
            f"{base_col}_corr",
            f"{base_col}_adj",
        ]
        for c in candidates:
            if c in row.index:
                v = row.get(c)
                try:
                    if v is not None and not pd.isna(v):
                        return v
                except Exception:
                    if v is not None:
                        return v
        return base_val

    return base_val


# ---------------------------------------------------
# 성장률 우수기업: 새 레이아웃 지원
# ---------------------------------------------------
def _infer_bucket_100_from_sales2024(sales_2024):
    eok = to_eok(sales_2024)
    if eok is None:
        return None
    try:
        return "100억원 초과" if float(eok) > 100 else "100억원 이하"
    except (TypeError, ValueError):
        return None


def _get_applicant_bucket_100(df_tg: pd.DataFrame):
    if df_tg is None or df_tg.empty:
        return None

    row = df_tg.iloc[0]
    for col in ["매출100억구분", "매출구간"]:
        if col in df_tg.columns:
            v = str(row.get(col) or "").strip()
            if v:
                if "초과" in v:
                    return "100억원 초과"
                if "이하" in v:
                    return "100억원 이하"

    return _infer_bucket_100_from_sales2024(row.get("매출액_2024"))


def _calc_cagr_3y(s22, s24):
    s22 = pd.to_numeric(s22, errors="coerce")
    s24 = pd.to_numeric(s24, errors="coerce")
    if pd.isna(s22) or pd.isna(s24):
        return np.nan
    try:
        s22_f = float(s22)
        s24_f = float(s24)
        if s22_f <= 0 or s24_f <= 0:
            return np.nan
        ratio = s24_f / s22_f
        return (math.pow(ratio, 1.0 / 2.0) - 1.0) * 100.0
    except (TypeError, ValueError, ZeroDivisionError):
        return np.nan


def prepare_growth_rank_panel(
    df_group: pd.DataFrame,
    group_label: str,
    applicant_name: str | None,
    applicant_growth: float | None,
    applicant_bucket_100: str | None,
    applicant_corp_no: str | None = None,
    max_rank: int = 10,
    applicant_is_corrected: bool = False
):
    base = df_group.copy() if df_group is not None else pd.DataFrame()

    base = _ensure_plain_corpno_col(base) if (base is not None and not base.empty) else base

    if (base is not None) and (not base.empty):
        if "성장률(%)" not in base.columns:
            base["성장률(%)"] = base.apply(
                lambda r: _calc_cagr_3y(
                    _pick_sales_for_growth(r, 2022),
                    _pick_sales_for_growth(r, 2024),
                ),
                axis=1,
            )

        base["성장률표기"] = base.apply(
            lambda r: _fmt_num(r.get("성장률(%)"), 2) + (
                " (F)" if _is_corrected_flag(r.get("매출액_2022_보정여부")) or _is_corrected_flag(r.get("매출액_2024_보정여부"))
                else ""
            ),
            axis=1
        )

    if base is not None and not base.empty and "성장률(%)" in base.columns:
        base["성장률(%)"] = pd.to_numeric(base["성장률(%)"], errors="coerce")

    group_avg = None
    if base is not None and not base.empty and "성장률(%)" in base.columns:
        gvals = pd.to_numeric(base["성장률(%)"], errors="coerce").dropna()
        if not gvals.empty:
            group_avg = float(gvals.mean())

    if base is None or base.empty:
        keep_cols = ["법인번호", "기업명", "성장률(%)", "성장률표기"]
        top = pd.DataFrame(columns=keep_cols)
    else:
        keep_cols = ["법인번호", "기업명", "성장률(%)", "성장률표기"]
        top = (
            base.sort_values("성장률(%)", ascending=False)
            .head(max_rank)
            .loc[:, [c for c in keep_cols if c in base.columns]]
            .reset_index(drop=True)
        )
        for c in keep_cols:
            if c not in top.columns:
                top[c] = ""

    if not top.empty:
        top["order"] = np.arange(1, len(top) + 1)

        v = st.session_state.get("fin_use_similarity", False)
        if isinstance(v, str):
            use_similarity = (v.strip().lower() in {"on", "true", "1", "yes", "y"})
        else:
            use_similarity = bool(v)

        if use_similarity:
            top["표기"] = top["기업명"].astype(str)
        else:
            top["표기"] = top.apply(lambda r: f"[{int(r['order'])}위] {str(r.get('기업명',''))}", axis=1)

        top["is_applicant"] = False
        top["is_placeholder"] = False
    else:
        top = pd.DataFrame({
            "법인번호": [],
            "기업명": [],
            "성장률(%)": [],
            "성장률표기": [],
            "order": [],
            "표기": [],
            "is_applicant": [],
            "is_placeholder": [],
        })

    applicant_in_this = (applicant_bucket_100 is not None and applicant_bucket_100 == group_label)

    if applicant_in_this and applicant_growth is not None:
        row_dict = {
            "법인번호": "",
            "기업명": applicant_name or "신청기업",
            "성장률(%)": float(applicant_growth),
            "성장률표기": _fmt_num(applicant_growth, 2) + (" (F)" if applicant_is_corrected else ""),
            "order": 12,
            "표기": f"신청기업 {applicant_name or ''}".strip(),
            "is_applicant": True,
            "is_placeholder": False,
        }
        last = pd.DataFrame([row_dict])
    else:
        row_dict = {
            "법인번호": "",
            "기업명": "",
            "성장률(%)": 0.0,
            "성장률표기": "",
            "order": 12,
            "표기": "",
            "is_applicant": False,
            "is_placeholder": True,
        }
        last = pd.DataFrame([row_dict])

    out = pd.concat([top, last], ignore_index=True)

    if len(out) < 11:
        missing = 11 - len(out)
        existing_orders = set(pd.to_numeric(out["order"], errors="coerce").tolist())
        fill_rows = []
        for o in range(1, 11):
            if o not in existing_orders and missing > 0:
                r = {
                    "법인번호": "",
                    "기업명": "",
                    "성장률(%)": 0.0,
                    "성장률표기": "",
                    "order": o,
                    "표기": "",
                    "is_applicant": False,
                    "is_placeholder": True
                }
                fill_rows.append(r)
                missing -= 1
        if fill_rows:
            out = pd.concat([out, pd.DataFrame(fill_rows)], ignore_index=True)
        out = out.sort_values("order").reset_index(drop=True)

    if group_avg is not None:
        avg_row = {
            "법인번호": "",
            "기업명": f"{group_label} 평균",
            "성장률(%)": group_avg,
            "성장률표기": _fmt_num(group_avg, 2),
            "order": 11,
            "표기": "평균",
            "is_applicant": False,
            "is_placeholder": False,
        }
        out = pd.concat([out, pd.DataFrame([avg_row])], ignore_index=True)

    out = _ensure_plain_corpno_col(out)
    return out


def growth_rank_chart(df_panel: pd.DataFrame, panel_title: str, height: int = 460, width: int = 650):
    d = df_panel.copy()
    if d is None or d.empty:
        return alt.Chart(pd.DataFrame({"표기": [], "growth_value": [], "order": []}))

    d = _ensure_plain_corpno_col(d)

    if "표기" not in d.columns:
        d["표기"] = d.get("기업명", "").astype(str)
    if "성장률(%)" not in d.columns:
        d["성장률(%)"] = 0.0
    if "성장률표기" not in d.columns:
        d["성장률표기"] = ""
    if "order" not in d.columns:
        d["order"] = np.arange(len(d))

    if "is_applicant" not in d.columns:
        d["is_applicant"] = False
    if "is_placeholder" not in d.columns:
        d["is_placeholder"] = False

    d["order"] = pd.to_numeric(d["order"], errors="coerce")
    d = d.sort_values("order", ascending=True).reset_index(drop=True)

    d["growth_value"] = pd.to_numeric(d["성장률(%)"], errors="coerce").fillna(0.0)
    d["is_mean"] = d["표기"].astype(str).str.strip().eq("평균")

    y_order = d["표기"].tolist()

    vals = d["growth_value"]
    v_max = float(vals.max(skipna=True)) if vals.notna().any() else 0.0
    v_min = float(vals.min(skipna=True)) if vals.notna().any() else 0.0
    pad = max(1.0, max(abs(v_max), abs(v_min)) * 0.15)
    x_domain = [v_min - pad, v_max + pad]

    x_enc = alt.X(
        "growth_value:Q",
        scale=alt.Scale(domain=x_domain),
        axis=alt.Axis(title="성장률(%)", grid=True, tickCount=6),
    )

    y_enc = alt.Y(
        "표기:N",
        sort=y_order,
        axis=alt.Axis(
            title=None,
            ticks=False,
            labelLimit=240,
            labelPadding=8,
            labelFontSize=13,
        ),
    )

    base = alt.Chart(d).properties(width=width, height=height)

    tooltip = []
    if "기업명" in d.columns:
        tooltip.append(alt.Tooltip("기업명:N", title="기업"))
    else:
        tooltip.append(alt.Tooltip("표기:N", title="기업"))
    tooltip.append(alt.Tooltip("법인번호:N", title="법인번호"))
    tooltip.append(alt.Tooltip("성장률표기:N", title="성장률(%)"))

    bars = base.mark_bar(size=22).encode(
        x=x_enc,
        y=y_enc,
        opacity=alt.condition(alt.datum.is_placeholder, alt.value(0.0), alt.value(1.0)),
        color=alt.condition(alt.datum.is_applicant, alt.value("#f2c94c"), alt.value("#2f80ed")),
        tooltip=tooltip,
    )

    text_pos = base.transform_filter("datum.growth_value >= 0").mark_text(
        align="left", baseline="middle", dx=6, fontSize=15
    ).encode(
        x=x_enc,
        y=y_enc,
        text=alt.condition(alt.datum.is_placeholder, alt.value(""), alt.Text("성장률표기:N")),
    )

    text_neg = base.transform_filter("datum.growth_value < 0").mark_text(
        align="right", baseline="middle", dx=-6, fontSize=15
    ).encode(
        x=x_enc,
        y=y_enc,
        text=alt.condition(alt.datum.is_placeholder, alt.value(""), alt.Text("성장률표기:N")),
    )

    zero_rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
        stroke="black", strokeWidth=1
    ).encode(x="x:Q")

    chart = alt.layer(bars, text_pos, text_neg, zero_rule).resolve_scale(x="shared")
    return chart.properties(title=panel_title)


# ---------------------------------------------------
# 매출 경쟁사 비교 로직
# ---------------------------------------------------
def _make_label(row):
    v = st.session_state.get("fin_use_similarity", False)
    if isinstance(v, str):
        use_similarity = (v.strip().lower() in {"on", "true", "1", "yes", "y"})
    else:
        use_similarity = bool(v)

    if row.get("구분") == "평균":
        base = "평균"
    else:
        name = str(row.get("표시명") or "").strip()
        disp_rank = row.get("표시순위")

        if use_similarity:
            base = name
        else:
            try:
                if disp_rank is not None and not pd.isna(disp_rank):
                    rank_int = int(float(disp_rank))
                    base = f"[{rank_int}위] {name}"
                else:
                    base = name
            except (TypeError, ValueError):
                base = name

    return wrap4(base)


def _slice_window_by_applicant(
    df_comp_sorted: pd.DataFrame,
    window: int = 11,
    top_n: int = 5,
) -> pd.DataFrame:
    if df_comp_sorted is None or df_comp_sorted.empty:
        return df_comp_sorted

    n = len(df_comp_sorted)
    if n <= window:
        return df_comp_sorted

    mask_app = (df_comp_sorted.get("구분") == "신청기업")
    if mask_app is None or (not mask_app.any()):
        return df_comp_sorted.iloc[:window]

    app_pos = int(df_comp_sorted[mask_app].index[0])

    start = app_pos - top_n
    if start < 0:
        start = 0
    end = start + window
    if end > n:
        end = n
        start = max(0, end - window)

    return df_comp_sorted.iloc[start:end]


def prepare_peer_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    full = df.copy()
    id_col = _get_id_col(full)

    full_is_mean = (full["구분"] == "평균") if "구분" in full.columns else pd.Series(False, index=full.index)
    full_comp = full[~full_is_mean].copy()

    use_similarity = st.session_state.get("fin_use_similarity", False)

    if "rank_num" not in full_comp.columns:
        sales_tmp = pd.to_numeric(full_comp.get("매출액"), errors="coerce")
        full_comp = full_comp.copy()
        full_comp["rank_num"] = sales_tmp.rank(method="dense", ascending=False)

    if use_similarity and (not full_comp.empty):
        tmp = full_comp.sort_values("rank_num", ascending=True).reset_index()
        tmp["sim_rank"] = np.arange(1, len(tmp) + 1)
        sim_map = dict(zip(tmp["index"], tmp["sim_rank"]))
        full["sim_rank"] = full.index.map(sim_map)
    else:
        full["sim_rank"] = np.nan

    full_sales = pd.to_numeric(full_comp.get("매출액"), errors="coerce")
    full_margin = pd.to_numeric(full_comp.get("매출액영업이익률"), errors="coerce")

    avg_sales: float | None = None
    avg_margin: float | None = None

    if use_similarity:
        if not full_sales.empty:
            avg_sales = float(full_sales.mean(skipna=True))
        if not full_margin.empty:
            avg_margin = float(full_margin.mean(skipna=True))
    else:
        if "평균매출액" in full.columns:
            s = pd.to_numeric(full["평균매출액"], errors="coerce").dropna()
            if not s.empty:
                avg_sales = float(s.iloc[0])
        if "평균매출액영업이익률" in full.columns:
            m = pd.to_numeric(full["평균매출액영업이익률"], errors="coerce").dropna()
            if not m.empty:
                avg_margin = float(m.iloc[0])

    order_col = "sim_rank" if use_similarity else "rank_num"
    df_comp_sorted = full_comp.copy()
    df_comp_sorted[order_col] = pd.to_numeric(df_comp_sorted.get(order_col), errors="coerce")
    df_comp_sorted = df_comp_sorted.sort_values(order_col, ascending=True).reset_index(drop=False).rename(columns={"index": "orig_idx"})

    df_sel_comp = _slice_window_by_applicant(df_comp_sorted, window=11, top_n=5)

    picked_idx = df_sel_comp["orig_idx"].tolist() if ("orig_idx" in df_sel_comp.columns) else []
    d_comp = full.loc[picked_idx].copy() if picked_idx else full_comp.iloc[:11].copy()

    if "매출액_보정여부" not in d_comp.columns:
        d_comp["매출액_보정여부"] = 1
    if "매출액영업이익률_보정여부" not in d_comp.columns:
        d_comp["매출액영업이익률_보정여부"] = 1

    d_comp["영업이익률(%)"] = pd.to_numeric(d_comp.get("매출액영업이익률"), errors="coerce")

    avg_row = {
        "상호": "평균",
        "매출액": avg_sales,
        "매출액_보정여부": 1,
        "매출액구간": None,
        "최종매출액구간": None,
        "rank_num": None,
        "sim_rank": None,
        "구분": "평균",
        "평균매출액": avg_sales,
        "매출액영업이익률": avg_margin,
        "매출액영업이익률_보정여부": 1,
        "영업이익률(%)": avg_margin,
    }
    if id_col:
        avg_row[id_col] = None

    df_avg = pd.DataFrame([avg_row])

    d = pd.concat([d_comp, df_avg], ignore_index=True)

    d["유형"] = d["구분"].fillna("경쟁기업")
    d.loc[~d["유형"].isin(["신청기업", "평균"]), "유형"] = "경쟁기업"

    if id_col:
        d["표시명"] = d["상호"].fillna(d[id_col]).fillna("(식별자 없음)")
    else:
        d["표시명"] = d["상호"].fillna("(식별자 없음)")

    d["표시순위"] = np.nan
    is_mean_row = (d["구분"] == "평균") if "구분" in d.columns else pd.Series(False, index=d.index)

    if use_similarity:
        d.loc[~is_mean_row, "표시순위"] = pd.to_numeric(d.loc[~is_mean_row, "sim_rank"], errors="coerce")
    else:
        d.loc[~is_mean_row, "표시순위"] = pd.to_numeric(d.loc[~is_mean_row, "rank_num"], errors="coerce")

    d["라벨"] = d.apply(_make_label, axis=1)
    d["매출액(억원)"] = d["매출액"].map(to_eok)
    d["sort_key"] = pd.to_numeric(d.get("표시순위"), errors="coerce").fillna(9999).astype(int)

    d["매출액표기"] = d.apply(
        lambda r: (
            (_fmt_num(r.get("매출액(억원)"), 2) + (" (F)" if _is_corrected_flag(r.get("매출액_보정여부")) else ""))
            if _fmt_num(r.get("매출액(억원)"), 2) != "" else ""
        ),
        axis=1,
    )
    d["영업이익률표기"] = d.apply(
        lambda r: (
            (_fmt_num(r.get("영업이익률(%)"), 1) + (" (F)" if _is_corrected_flag(r.get("매출액영업이익률_보정여부")) else ""))
            if _fmt_num(r.get("영업이익률(%)"), 1) != "" else ""
        ),
        axis=1,
    )

    d = _ensure_plain_corpno_col(d)
    return d


def create_peer_window_chart(df: pd.DataFrame, title: str | None = None):
    """
    반환: (chart, avg_sales_eok, avg_margin)
    """
    if df.empty:
        return None, None, None

    required_cols = {
        "라벨", "매출액(억원)", "유형", "표시명", "sort_key",
        "영업이익률(%)", "구분", "매출액표기", "영업이익률표기",
        "매출액_보정여부", "매출액영업이익률_보정여부",
        "법인번호",
    }

    d = df.copy() if required_cols.issubset(df.columns) else prepare_peer_df(df)
    if d.empty:
        return None, None, None

    d = _ensure_plain_corpno_col(d)

    avg_sales_eok = None
    avg_margin = None
    is_avg = (d.get("구분") == "평균") if "구분" in d.columns else pd.Series(False, index=d.index)
    if is_avg.any():
        s = pd.to_numeric(d.loc[is_avg, "매출액(억원)"], errors="coerce").dropna()
        if not s.empty:
            avg_sales_eok = float(s.iloc[0])
        m = pd.to_numeric(d.loc[is_avg, "영업이익률(%)"], errors="coerce").dropna()
        if not m.empty:
            avg_margin = float(m.iloc[0])
        d = d.loc[~is_avg].copy()

    if d.empty:
        return None, avg_sales_eok, avg_margin

    x_order = d.sort_values("sort_key")["라벨"].tolist()

    color_domain = ["경쟁기업", "신청기업"]
    color_range = ["#cfcfcf", "#59ff008d"]

    n = len(d)
    bar_width = 50
    width = max(900, 80 * n)

    max_sales = d["매출액(억원)"].max()
    max_sales = 0.0 if pd.isna(max_sales) else float(max_sales)
    sales_y_max = max(max_sales * 1.35, 1.0)

    has_margin = ("영업이익률(%)" in d.columns) and d["영업이익률(%)"].notna().any()
    margin_scale = None
    if has_margin:
        m = pd.to_numeric(d["영업이익률(%)"], errors="coerce")
        margin_max = float(m.max(skipna=True))
        margin_min = float(m.min(skipna=True))
        if (margin_max == 0) and (margin_min == 0):
            margin_y_min, margin_y_max = -10.0, 10.0
        else:
            max_abs = max(abs(margin_max), abs(margin_min))
            pad = max_abs * 0.3
            margin_y_min = margin_min - pad
            margin_y_max = margin_max + pad * 3
        margin_scale = alt.Scale(domain=[margin_y_min, margin_y_max])

    base = alt.Chart(d).properties(width=width, height=620)
    if title is not None:
        base = base.properties(title=title)

    tooltip = [
        alt.Tooltip("표시명:N", title="기업명"),
        alt.Tooltip("법인번호:N", title="법인번호"),
        alt.Tooltip("유형:N", title="구분"),
        alt.Tooltip("표시순위:Q", title="순위"),
        alt.Tooltip("매출액표기:N", title="매출액(억원)"),
        alt.Tooltip("매출액_보정여부:Q", title="매출액 보정여부(0=보정,1=원값)"),
    ]
    if has_margin:
        tooltip.extend([
            alt.Tooltip("영업이익률표기:N", title="영업이익률(%)"),
            alt.Tooltip("매출액영업이익률_보정여부:Q", title="영업이익률 보정여부(0=보정,1=원값)"),
        ])

    bars = base.mark_bar(
        size=bar_width,
        cornerRadiusTopLeft=4,
        cornerRadiusTopRight=4,
    ).encode(
        x=alt.X(
            "라벨:N",
            sort=x_order,
            axis=alt.Axis(
                title=None,
                labelAngle=90,
                labelAlign="left",
                labelBaseline="middle",
                labelLimit=2000,
                labelPadding=10,
                ticks=False,
            ),
        ),
        y=alt.Y(
            "매출액(억원):Q",
            axis=alt.Axis(title="매출액(억원)", orient="left"),
        ),
        color=alt.Color(
            "유형:N",
            legend=alt.Legend(title="구분"),
            scale=alt.Scale(domain=color_domain, range=color_range),
        ),
        tooltip=tooltip,
    )

    text_bar = base.mark_text(dy=-8, fontSize=15).encode(
        x=alt.X("라벨:N", sort=x_order),
        y=alt.Y("매출액(억원):Q", axis=None, scale=alt.Scale(domain=[0, sales_y_max])),
        text=alt.Text("매출액표기:N"),
        tooltip=tooltip,
    )

    if not has_margin:
        return (bars + text_bar).configure_view(stroke="#cbd5e1", strokeWidth=1), avg_sales_eok, avg_margin

    line = base.mark_line(point=True, strokeWidth=3).encode(
        x=alt.X("라벨:N", sort=x_order),
        y=alt.Y(
            "영업이익률(%)",
            type="quantitative",
            axis=alt.Axis(title="영업이익률(%)", orient="right"),
            scale=margin_scale,
        ),
        tooltip=tooltip,
    )

    line_text = base.mark_text(baseline="top", dy=6, fontSize=15).encode(
        x=alt.X("라벨:N", sort=x_order),
        y=alt.Y(
            "영업이익률(%)",
            type="quantitative",
            axis=None,
            scale=margin_scale,
        ),
        text=alt.Text("영업이익률표기:N"),
        tooltip=tooltip,
    )

    layer = alt.layer(bars, text_bar, line, line_text).resolve_scale(y="independent")
    return layer.configure_view(stroke="#cbd5e1", strokeWidth=1), avg_sales_eok, avg_margin


# ---------------------------------------------------
# 메인: 재무부문 화면
# ---------------------------------------------------
def render_finance(
    sel_subcat_name: str,
    sel_subcat_code: str,
    applicant_corp_no: str,
    applicant_name: str,
    send_sql,
):
    st.markdown(
        """
        <div class="sec-banner" style="--accent:#5b9bd5;">
        <div class="sec-label">재무부문</div>
        <div class="sec-rule"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sel_subcat_name = st.session_state.get("selected_subcat_name", sel_subcat_name)
    sel_subcat_code = st.session_state.get("selected_subcat_code", sel_subcat_code)
    applicant_corp_no = st.session_state.get("applicant_corp_no", applicant_corp_no)
    applicant_name = st.session_state.get("applicant_name", applicant_name)

    st.markdown(
        """
        <style>
        .fin-metrics{
            display:flex;
            width:100%;
            align-items:center;
            gap:16px;
            justify-content:center;
            margin-top:12px;
            margin-bottom:38px;
        }
        .fin-metrics.dual{
            justify-content:center;
        }
        .pill{
            display:inline-block;
            border-radius:12px;
            padding:14px 24px;
            font-weight:800;
            text-align:center;
            border:1.5px solid #cbd5e1;
            white-space:nowrap;
        }
        .pill.blue { background:#eaf0ff; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # session state 초기화
    if "company_num" not in st.session_state:
        st.session_state["company_num"] = None

    company_num = st.session_state.get("company_num")
    applicant_sales_missing = False

    # 기본값 보장
    if "fin_use_similarity" not in st.session_state:
        st.session_state["fin_use_similarity"] = False

    use_sim = bool(st.session_state.get("fin_use_similarity", False))
    btn_label = "기업 유사도 필터링\nOFF" if use_sim else "기업 유사도 필터링\nON"

    c1, c2 = st.columns([0.92, 0.08], vertical_alignment="center")

    with c1:
        st.markdown(f"### [{sel_subcat_name}] 매출액 기준 경쟁사 비교 (단위: 억원)(2024년 기준)")

    with c2:
        st.markdown(
            """
            <style>
            div[data-testid="column"] button[kind="secondary"]{
                width: 100%;
                white-space: pre-line;
                padding-top: 6px;
                padding-bottom: 6px;
                font-size: 12px;
                font-weight: 700;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.button(
            btn_label,
            key="btn_fin_similarity_filter_top",
            on_click=_toggle_fin_similarity_filter,
            use_container_width=True,
        )

    # -------------------------------------------------
    # 1) 매출 경쟁사 비교: SQL 결과를 세션 캐시로 재사용
    # -------------------------------------------------
    if not st.session_state["company_num"]:
        peer_key = ("peer_raw", "q_fin_peer_window", sel_subcat_code, str(applicant_corp_no))
        df_raw = _cache_get_df(
            peer_key,
            lambda: send_sql(
                Q.q_fin_peer_window(),
                params={"sel_subcat_code": sel_subcat_code, "applicant_corp_no": applicant_corp_no},
            ),
        )

        # 유사도 ON일 때만 필터 적용(Chroma 쪽도 캐시됨)
        df_raw = filter_peers_by_similarity(
            df_raw=df_raw,
            sel_subcat_code=sel_subcat_code,
            applicant_corp_no=applicant_corp_no,
        )

    else:
        # 기존 로직 유지(쿼리명은 그대로)
        raw_key = ("peer_raw", "qqqqqqqq", sel_subcat_code)
        app_key = ("peer_app", "qqqqqqqq1", str(applicant_corp_no))

        df_raw = _cache_get_df(
            raw_key,
            lambda: send_sql(
                Q.qqqqqqqq(),
                params={"sel_subcat_code": sel_subcat_code},
            ),
        )
        df_applicant = _cache_get_df(
            app_key,
            lambda: send_sql(
                Q.qqqqqqqq1(),
                params={"applicant_corp_no": applicant_corp_no},
            ),
        )

        df_raw = df_raw.copy()
        df_raw["구분"] = "경쟁기업"

        applicant_sales_missing = (not _has_applicant_2024_finance(df_applicant))

        if (df_applicant is not None) and (not df_applicant.empty) and (not applicant_sales_missing):
            df_applicant = df_applicant.copy()
            df_applicant["구분"] = "신청기업"

            for col in ["평균매출액", "평균매출액영업이익률"]:
                if col in df_raw.columns and col not in df_applicant.columns:
                    df_applicant[col] = df_raw[col].iloc[0]

            if "법인번호_ENC" in df_raw.columns and "법인번호_ENC" in df_applicant.columns:
                app_enc = str(df_applicant.iloc[0]["법인번호_ENC"])
                df_raw = df_raw[df_raw["법인번호_ENC"].astype(str) != app_enc]

            df_raw = pd.concat([df_raw, df_applicant], ignore_index=True)

    if df_raw is None or df_raw.empty:
        applicant_sales_missing = True
    else:
        df_raw = df_raw.copy()

        df_raw["매출액_num"] = pd.to_numeric(df_raw["매출액"], errors="coerce")
        df_raw["이익률_num"] = pd.to_numeric(df_raw["매출액영업이익률"], errors="coerce")

        df_raw = df_raw.sort_values(
            by=["매출액_num", "이익률_num"],
            ascending=[False, False],
            kind="mergesort",
        ).reset_index(drop=True)

        df_raw["rank_num"] = (
            df_raw[["매출액_num", "이익률_num"]]
            .apply(tuple, axis=1)
            .rank(method="dense", ascending=False)
            .astype(int)
        )

        df_raw = df_raw.drop(columns=["매출액_num", "이익률_num"], errors="ignore")

    if not st.session_state["company_num"]:
        st.write(f"###### 기업 유사도 필터링 OFF: [{sel_subcat_name}] 내 모든 기업들의 재무비교")
        st.write(
            f"###### 기업 유사도 필터링 ON: [{sel_subcat_name}] 내 모든 기업들 중 "
            f"[{applicant_name}]과 유사한 기업들(신청기업 포함 최대 20개)의 재무비교"
        )

    with st.container(border=True):
        chart, avg_sales_eok, avg_margin = create_peer_window_chart(df_raw, title=None)

        if chart is not None:
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("표시할 매출 데이터가 없습니다.")

        avg_sales_txt = (f"{avg_sales_eok:,.2f}억원" if avg_sales_eok is not None else "정보 없음")
        avg_margin_txt = (f"{avg_margin:,.1f}%" if avg_margin is not None else "정보 없음")

        st.markdown(
            f"""
            <div class="fin-metrics dual">
              <div class="pill blue"><b>{sel_subcat_name}</b> 매출액 평균&nbsp;<b>{avg_sales_txt}</b></div>
              <div class="pill blue"><b>{sel_subcat_name}</b> 영업이익률 평균&nbsp;<b>{avg_margin_txt}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if applicant_sales_missing:
        st.info("신청기업의 2024년 재무정보가 없어 신청기업 데이터가 없습니다.")

    # -------------------------------------------------
    # 2) 성장률 우수기업: SQL 결과도 세션 캐시로 재사용
    # -------------------------------------------------
    st.markdown(
        f"### [{sel_subcat_name}] 성장률 우수기업(최근 3개년 연평균성장률, 단위: %)(2024년 기준)",
        unsafe_allow_html=True,
    )

    if not sel_subcat_code:
        st.info("소분류 코드가 없습니다.")
        st.stop()

    g_key = ("growth_raw", "q_growth_top10_by_sales_bucket", sel_subcat_code)
    df_g = _cache_get_df(
        g_key,
        lambda: send_sql(Q.q_growth_top10_by_sales_bucket(), params={"sel_subcat_code": sel_subcat_code}),
    )

    df_g = filter_peers_by_similarity(df_raw=df_g, sel_subcat_code=sel_subcat_code, applicant_corp_no=applicant_corp_no)

    if df_g is not None and (not df_g.empty):
        df_g = df_g.copy()

        if "매출100억구분" in df_g.columns and "매출구간" not in df_g.columns:
            df_g["매출구간"] = df_g["매출100억구분"]

        df_g["성장률(%)"] = df_g.apply(
            lambda r: _calc_cagr_3y(
                _pick_sales_for_growth(r, 2022),
                _pick_sales_for_growth(r, 2024),
            ),
            axis=1,
        )

        if "상호" in df_g.columns:
            df_g["기업명"] = df_g["상호"].fillna("")
        else:
            df_g["기업명"] = ""

        df_g["기업명"] = df_g["기업명"].astype(str).str.strip()
        empty_name = df_g["기업명"] == ""
        id_col_g = _get_id_col(df_g)
        if id_col_g and (id_col_g in df_g.columns):
            df_g.loc[empty_name, "기업명"] = df_g.loc[empty_name, id_col_g].astype(str)
        df_g["기업명"] = df_g["기업명"].replace(BAD_VALUES, "(식별자 없음)")

        df_g = _ensure_plain_corpno_col(df_g)

    target_growth = None
    applicant_disp_name = "신청기업"
    applicant_is_corrected = False
    df_tg = pd.DataFrame()
    applicant_growth_fin2024_missing = False

    if applicant_corp_no:
        tg_key = ("target_growth", "q_target_growth_for_subcat", str(applicant_corp_no))
        df_tg = _cache_get_df(
            tg_key,
            lambda: send_sql(
                Q.q_target_growth_for_subcat(),
                params={"applicant_corp_no": applicant_corp_no},
            ),
        )

        applicant_growth_fin2024_missing = (not _has_applicant_2024_finance(df_tg))
        if applicant_growth_fin2024_missing:
            st.info("신청기업의 2024년 재무정보가 없어 신청기업 데이터가 없습니다.")

        if (not applicant_growth_fin2024_missing) and (df_tg is not None) and (not df_tg.empty):
            row = df_tg.iloc[0]
            s22 = _pick_sales_for_growth(row, 2022)
            s24 = _pick_sales_for_growth(row, 2024)
            target_growth = _calc_cagr_3y(s22, s24)
            if pd.isna(target_growth):
                target_growth = None

            applicant_is_corrected = _is_corrected_flag(row.get("매출액_2022_보정여부")) or _is_corrected_flag(row.get("매출액_2024_보정여부"))

            nm = str(row.get("상호") or "").strip()
            if nm and nm not in BAD_VALUES:
                applicant_disp_name = nm

    applicant_bucket_100 = _get_applicant_bucket_100(df_tg)

    tech_avg_growth = None
    vals: list[float] = []

    if df_g is not None and not df_g.empty:
        growth_series = pd.to_numeric(df_g.get("성장률(%)"), errors="coerce")
        id_col_g = _get_id_col(df_g)
        if id_col_g and (id_col_g in df_g.columns) and applicant_corp_no:
            mask_app = df_g[id_col_g].astype(str) == str(applicant_corp_no)
            growth_base = growth_series[~mask_app]
        else:
            growth_base = growth_series

        growth_base = growth_base.dropna()
        if not growth_base.empty:
            vals.extend(growth_base.tolist())

    if target_growth is not None:
        vals.append(float(target_growth))

    if vals:
        tech_avg_growth = float(np.mean(vals))

    df_100le = df_g[df_g.get("매출구간") == "100억원 이하"] if (df_g is not None and not df_g.empty) else pd.DataFrame()
    df_100gt = df_g[df_g.get("매출구간") == "100억원 초과"] if (df_g is not None and not df_g.empty) else pd.DataFrame()

    panel_le = prepare_growth_rank_panel(
        df_group=df_100le,
        group_label="100억원 이하",
        applicant_name=applicant_disp_name,
        applicant_growth=target_growth,
        applicant_bucket_100=applicant_bucket_100,
        applicant_corp_no=applicant_corp_no,
        max_rank=10,
        applicant_is_corrected=applicant_is_corrected
    )
    panel_gt = prepare_growth_rank_panel(
        df_group=df_100gt,
        group_label="100억원 초과",
        applicant_name=applicant_disp_name,
        applicant_growth=target_growth,
        applicant_bucket_100=applicant_bucket_100,
        applicant_corp_no=applicant_corp_no,
        max_rank=10,
        applicant_is_corrected=applicant_is_corrected
    )

    with st.container(border=True):
        has_any = ((panel_le is not None and not panel_le.empty) or (panel_gt is not None and not panel_gt.empty))
        if not has_any:
            st.info("성장률 데이터가 없습니다.")
        else:
            ch_le = growth_rank_chart(panel_le, "매출 100억원 이하 그룹")
            ch_gt = growth_rank_chart(panel_gt, "매출 100억원 초과 그룹")

            combo = (
                alt.hconcat(ch_le, ch_gt)
                .resolve_scale(x="independent")
                .configure_axisX(grid=True, tickMinStep=5, gridColor="#e5e7eb", gridWidth=1)
                .configure_view(stroke="#cbd5e1", strokeWidth=1)
            )
            st.altair_chart(combo, use_container_width=True)

        g_right = f"{tech_avg_growth:.2f}%" if tech_avg_growth is not None else "정보 없음"

        st.markdown(
            f"""
            <div class="fin-metrics">
              <div class="pill blue"><b>{sel_subcat_name}</b> 평균&nbsp;<b>{g_right}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
