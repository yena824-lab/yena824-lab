#-*- coding: utf-8 -*-
"""
button.tech_finance
===================

재무부문(Finance) 화면 모듈.

Streamlit 기반 BIGx 보고서에서 **재무부문 섹션**을
렌더링하는 모듈이다.
공통 유틸리티 함수들과 메인 렌더링 함수 :func:`render_finance` 를 제공한다.

기능 개요
--------

* 매출액 구간/그룹 매핑을 위한 상수를 정의한다.
* 기업명, 법인번호, 성장률 등 텍스트/숫자 처리용 유틸리티 함수를 제공한다.
* 매출 구간별 Top3 기업 및 평균 값을 조회하여
  시각화에 활용 가능한 형태로 데이터를 정규화한다.
* Altair 를 사용하여 매출/성장률 막대 차트를 생성한다.
* :func:`render_finance` 를 통해
  재무부문 전체 레이아웃과 데이터를 연동하여 화면을 렌더링한다.
"""

import os
import math
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
import re

from main.sql import sql_tech_detail as Q

# ---------------------------------------------------
# 공통 상수
# ---------------------------------------------------
BUCKET_TO_GROUP = {
    "50억원 이하": "그룹1",
    "50억원 초과~100억원 이하": "그룹2",
    "100억원 초과~300억원 이하": "그룹3",
    "300억원 초과~1000억원 이하": "그룹4",
    "1000억원 초과": "그룹5",
}

BUCKET_LABEL_MAP = {
    "그룹1": "50억원 이하",
    "그룹2": "50억원 초과~100억원 이하",
    "그룹3": "100억원 초과~300억원 이하",
    "그룹4": "300억원 초과~1000억원 이하",
    "그룹5": "1000억원 초과",
}

GROUP_RIGHT_LABEL = {
    "그룹1": "50억원 이하",
    "그룹2": "50억원 초과 100억원 이하",
    "그룹3": "100억원 초과 300억원 이하",
    "그룹4": "300억원 초과 1,000억원 이하",
    "그룹5": "1,000억원 초과",
}

BAD_NAMES = {"", "-", "None", "(상호 없음)", "nan", "NaN"}


# ---------------------------------------------------
# 숫자/텍스트 유틸
# ---------------------------------------------------
def to_eok(v: object) -> float | None:
    """
    원/천원/백만원/억원 단위를 '억원'으로 통일해서 변환한다.
    """
    unit_env = os.getenv("FINANCE_UNIT", "thousand").strip().lower()
    x = pd.to_numeric(v, errors="coerce")
    if pd.isna(x):
        return None

    if unit_env == "won":
        return float(x) / 100_000_000
    if unit_env == "thousand":
        return float(x) / 100
    if unit_env == "million":
        return (float(x) * 1_000_000) / 100_000_000
    if unit_env == "eok":
        return float(x)
    return float(x)


def norm_id(x) -> str:
    """
    법인번호 등의 ID를 문자열 형태로 정리한다.
    """
    if pd.isna(x):
        return ""
    s = str(x).strip()
    return s[:-2] if s.endswith(".0") else s


def wrap4(s) -> str:
    """
    문자열을 4글자씩 잘라 줄바꿈하여 세로 라벨용으로 사용한다.
    """
    if s is None:
        return ""
    try:
        s = str(s).strip()
    except Exception:
        return ""
    if s.lower() in {"", "nan", "none"}:
        return ""
    return "\n".join([s[i: i + 4] for i in range(0, len(s), 4)])


def _is_adjusted_flag(v) -> bool:
    """
    보정여부: 0=보정, 1=미보정(표기 없음)
    """
    x = pd.to_numeric(v, errors="coerce")
    if pd.isna(x):
        return False
    try:
        return int(float(x)) == 0
    except Exception:
        return False


def _adj_suffix(v) -> str:
    return " (F)" if _is_adjusted_flag(v) else ""


def _fmt_or_none(x, fmt: str) -> str:
    v = pd.to_numeric(x, errors="coerce")
    if pd.isna(v):
        return "정보 없음"
    try:
        return format(float(v), fmt)
    except Exception:
        return "정보 없음"


def display_name_top(row: pd.Series) -> str:
    """
    상단 '매출 우수기업' 차트용 표시명 생성.
    """
    if row.get("그룹") == "평균":
        return "평균"
    nm = str(row.get("상호") or "").strip()
    if nm and nm.lower() not in {"nan", "none"}:
        return nm
    cid = norm_id(row.get("법인번호_ENC"))
    return cid if cid else "(식별자 없음)"


def clean_company_name_growth(row: pd.Series, gmap: dict) -> str:
    """
    성장률 그래프용 기업명 정리.
    """
    corp_id_raw = str(row.get("법인번호_ENC", "")).strip()
    corp_id = norm_id(corp_id_raw) or corp_id_raw

    nm = gmap.get(norm_id(row.get("법인번호_ENC")), "")
    nm = (nm or "").strip()
    if not nm or nm in BAD_NAMES:
        return corp_id
    return nm


# ---------------------------------------------------
# DB 관련 함수
# ---------------------------------------------------
def fetch_names_map(enc_ids: list[str], send_sql) -> pd.DataFrame:
    """
    법인번호(암호화) 리스트로부터 상호를 조회한다.
    """
    if not enc_ids:
        return pd.DataFrame(columns=["법인번호_ENC", "상호"])

    placeholders = ", ".join([f":id{i}" for i in range(len(enc_ids))])
    params_ids = {f"id{i}": v for i, v in enumerate(enc_ids)}

    return (
        send_sql(
            Q.q_corp_names(placeholders),
            params=params_ids,
        )
        .drop_duplicates("법인번호_ENC", keep="first")
        .reset_index(drop=True)
    )


def normalize_and_label_group(
    df: pd.DataFrame,
    group_name: str,
    bucket_label: str,
    send_sql,
) -> pd.DataFrame:
    """
    매출액 구간별 상위 3개 기업 + 평균을 하나의 DataFrame으로 정규화한다.
    """
    if df.empty:
        return pd.DataFrame()

    # 1) 상위 3개 기업
    df_top = df[["법인번호_ENC", "매출액"]].rename(columns={"매출액": "기업매출_2024"})
    df_top["그룹"] = group_name
    df_top["기업매출액영업이익률_2024"] = None
    df_top["최신매출액구간"] = bucket_label

    # 2) 평균 행
    avg_cols = [c for c in df.columns if c.endswith("매출액평균")]
    avg_val = pd.to_numeric(df[avg_cols[0]].iloc[0], errors="coerce") if avg_cols else np.nan
    df_avg = pd.DataFrame(
        [
            {
                "법인번호_ENC": None,
                "기업매출_2024": float(avg_val) if not pd.isna(avg_val) else np.nan,
                "기업매출액영업이익률_2024": None,
                "최신매출액구간": "평균",
                "그룹": "평균",
            }
        ]
    )

    out = pd.concat([df_top, df_avg], ignore_index=True)

    # 3) 상호 매핑
    ids = out["법인번호_ENC"].dropna().astype(str).tolist()
    df_nm = fetch_names_map(ids, send_sql) if ids else pd.DataFrame(columns=["법인번호_ENC", "상호"])
    out = out.merge(df_nm, on="법인번호_ENC", how="left")

    def _display(row):
        if row["그룹"] == "평균":
            return "평균"
        nm = str(row.get("상호") or "").strip()
        if nm and nm.lower() not in {"nan", "none"}:
            return nm
        cid = norm_id(row.get("법인번호_ENC"))
        return cid if cid else "(식별자 없음)"

    out["표시명"] = out.apply(_display, axis=1)
    out["라벨"] = out["표시명"].apply(wrap4)

    out["매출(억원)"] = out["기업매출_2024"].map(to_eok)
    out["영업이익률(%)"] = pd.to_numeric(out["기업매출액영업이익률_2024"], errors="coerce")

    order_map = {"그룹1": 1, "그룹2": 2, "그룹3": 3, "그룹4": 4, "그룹5": 5, "평균": 6}
    out["__group_order"] = out["그룹"].map(order_map).fillna(99)
    out["__rank_in_group"] = out.groupby("그룹")["매출(억원)"].rank(method="first", ascending=False)
    out.loc[out["그룹"] == "평균", "__rank_in_group"] = 1

    return out.sort_values(["__group_order", "__rank_in_group"]).reset_index(drop=True)


def run_group_query(
    group_name: str,
    bucket_label: str,
    sel_subcat_code: str,
    send_sql,
) -> pd.DataFrame:
    """
    매출액 구간별 상위 3개 기업 + 평균을 조회한 뒤 정규화한다.

    SQL 쿼리는 :applicant_corp_no 바인딩을 요구하지만,
    이 화면에서는 신청기업 정보를 사용하지 않는다.
    SQL 오류를 피하기 위해 더미 값(None)을 바인딩한다.
    """
    d = send_sql(
        Q.q_fin_group_top3_and_bucket_avg(group_name),
        params={
            "sel_subcat_code": sel_subcat_code,
            "applicant_corp_no": None,
        },
    )
    if d.empty:
        return pd.DataFrame()
    return normalize_and_label_group(d, group_name, bucket_label, send_sql)


# ---------------------------------------------------
# 막대 그래프 함수 (매출 Top3 + 평균)
# ---------------------------------------------------
def create_sales_chart(df_plot: pd.DataFrame, chart_title: str | None = None):
    """
    매출액 막대차트를 생성한다.
    (현재 render_finance에서는 사용하지 않지만, 다른 탭에서 활용 가능)
    """
    n = max(1, len(df_plot))
    width = max(560, 76 * n)

    y_min = 0.0
    y_max_val = df_plot["매출(억원)"].max() or 0
    y_max = float(max(y_max_val, 0)) * 1.10
    if y_max <= 0:
        y_max = 1.0

    chart_props = {"width": width, "height": 420}
    if chart_title:
        chart_props["title"] = alt.TitleParams(text=chart_title, anchor="middle")

    base = alt.Chart(df_plot).properties(**chart_props)

    color_order = ["그룹1", "그룹2", "그룹3", "그룹4", "그룹5", "평균"]
    domain = [g for g in color_order if g in df_plot["그룹"].unique().tolist()]
    palette = {
        "그룹1": "#1f77b4",
        "그룹2": "#ff7f0e",
        "그룹3": "#2ca02c",
        "그룹4": "#d62728",
        "그룹5": "#9467bd",
        "신청기업": "#94632d",
        "평균": "#7f7f7f",
    }
    range_colors = [palette[g] for g in domain]

    x_sort_order = df_plot["라벨"].tolist()

    common_tooltip = [
        alt.Tooltip("표시명:N", title="기업명"),
        alt.Tooltip("법인번호_ENC:N", title="법인번호_ENC"),
        alt.Tooltip("그룹:N", title="그룹"),
        alt.Tooltip("매출(억원):Q", title="매출(억원)", format=",.2f"),
    ]

    bar_sales = base.mark_bar(
        size=22,
        cornerRadiusTopLeft=4,
        cornerRadiusTopRight=4,
    ).encode(
        x=alt.X(
            "라벨:N",
            sort=x_sort_order,
            axis=alt.Axis(
                title=None,
                labelAngle=90,
                labelAlign="left",
                labelBaseline="middle",
                labelLimit=100,
                labelPadding=6,
                ticks=False,
            ),
            scale=alt.Scale(paddingInner=0.15, paddingOuter=0.2),
        ),
        y=alt.Y(
            "매출(억원):Q",
            title="금액(억원)",
            scale=alt.Scale(domain=[y_min, y_max]),
        ),
        color=alt.Color(
            "그룹:N",
            legend=alt.Legend(title="범주"),
            scale=alt.Scale(domain=domain, range=range_colors),
        ),
        tooltip=common_tooltip,
    )

    text_sales = base.mark_text(dy=-8, fontSize=12).encode(
        x=alt.X("라벨:N", sort=x_sort_order),
        y=alt.Y("매출(억원):Q", scale=alt.Scale(domain=[y_min, y_max])),
        text=alt.Text("label:N"),
        tooltip=common_tooltip,
    ).transform_calculate(label="format(datum['매출(억원)'], ',.2f')")

    chart = alt.layer(bar_sales, text_sales).configure_view(stroke="#cbd5e1", strokeWidth=1)
    return chart


def create_finance_chart(df_plot: pd.DataFrame, chart_title: str | None = None):
    """
    재무 차트 생성용 alias 함수.
    """
    return create_sales_chart(df_plot, chart_title)


# ---------------------------------------------------
# 매출 경쟁사 비교용 유틸
# ---------------------------------------------------
def _make_label(row):
    """
    X축에 표시할 라벨 생성.
    - 평균 행: '평균'
    - 그 외: '[n위] 기업명' (표시순위가 있으면)
    """
    if row.get("구분") == "평균":
        base = "평균"
    else:
        name = str(row.get("표시명") or "").strip()
        disp_rank = row.get("표시순위")
        try:
            if disp_rank is not None and not pd.isna(disp_rank):
                rank_int = int(float(disp_rank))
                base = f"[{rank_int}위] {name}"
            else:
                base = name
        except (TypeError, ValueError):
            base = name

    return wrap4(base)


def _slice_around_target(df: pd.DataFrame, top_n: int = 5, bottom_n: int = 5) -> pd.DataFrame:
    """
    company_finance와 동일한 로직.
    (이 모듈은 신청기업이 없지만, 호환성 유지용)
    """
    if df.empty:
        return df
    if "rank_num" not in df.columns or "구분" not in df.columns:
        return df

    is_mean = df["구분"] == "평균"
    df_mean = df[is_mean].copy()
    df_comp = df[~is_mean].copy()

    mask_target = df_comp["구분"] == "신청기업"
    if not mask_target.any():
        df_top = df_comp.sort_values("rank_num").head(top_n + bottom_n + 1)
        return pd.concat([df_top, df_mean], ignore_index=True)

    target_rank = df_comp.loc[mask_target, "rank_num"].iloc[0]
    df_comp["dist"] = df_comp["rank_num"] - target_rank

    above = df_comp[df_comp["dist"] < 0].sort_values("rank_num").tail(top_n)
    target = df_comp[df_comp["dist"] == 0].sort_values("rank_num")
    below = df_comp[df_comp["dist"] > 0].sort_values("rank_num").head(bottom_n)

    df_sel = pd.concat([above, target, below])

    wanted = top_n + bottom_n + 1
    need = wanted - len(df_sel)
    if need > 0:
        rest = df_comp.drop(df_sel.index, errors="ignore").sort_values("rank_num").head(need)
        df_sel = pd.concat([df_sel, rest])

    df_sel = df_sel.drop(columns=["dist"], errors="ignore").sort_values("rank_num")
    out = pd.concat([df_sel, df_mean], ignore_index=True)
    return out

# Jin Change
_RE_PLAIN_CORPNO = re.compile(r"^\d{7}-\d{6}$")
def _ensure_plain_corpno_col(d: pd.DataFrame) -> pd.DataFrame:
    if "법인번호" in d.columns:
        d["법인번호"] = d["법인번호"].astype(str).map(norm_id)
        return d
    candidates = ["corp_no", "법인번호_평문", "법인번호_plain", "법인번호_raw"]
    for c in candidates:
        if c in d.columns:
            d["법인번호"] = d[c].astype(str).map(norm_id)
            return d
    d["법인번호"] = ""
    return d


def prepare_peer_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    매출 경쟁사 비교용 DataFrame 가공.
    - 10개 기업만 표시 (신청기업 없음)
    - 평균/영업이익률은 pill 표기용으로만 유지
    - 보정여부(0=보정) 표기 컬럼 추가
    """
    if df.empty:
        return df

    full = df.copy()

    # 평균값(매출/영업이익률) 및 보정 플래그를 먼저 확보
    avg_sales: float | None = None
    avg_margin: float | None = None

    if "평균매출액" in full.columns:
        s = pd.to_numeric(full["평균매출액"], errors="coerce").dropna()
        if not s.empty:
            avg_sales = float(s.iloc[0])
    if "평균매출액영업이익률" in full.columns:
        m = pd.to_numeric(full["평균매출액영업이익률"], errors="coerce").dropna()
        if not m.empty:
            avg_margin = float(m.iloc[0])

    # 평균 플래그(있으면 사용)
    avg_sales_adj_flag = None
    avg_margin_adj_flag = None

    for col in ["평균매출액_보정여부", "평균매출액보정여부"]:
        if col in full.columns:
            v = pd.to_numeric(full[col], errors="coerce").dropna()
            if not v.empty:
                avg_sales_adj_flag = v.iloc[0]
                break

    for col in ["평균매출액영업이익률_보정여부", "평균매출액영업이익률보정여부"]:
        if col in full.columns:
            v = pd.to_numeric(full[col], errors="coerce").dropna()
            if not v.empty:
                avg_margin_adj_flag = v.iloc[0]
                break

    # fallback: 평균 행이 들어오는 경우
    if avg_sales_adj_flag is None and "구분" in full.columns and "매출액_보정여부" in full.columns:
        mean_row = full[full["구분"] == "평균"]
        if not mean_row.empty:
            avg_sales_adj_flag = mean_row["매출액_보정여부"].iloc[0]
    if avg_margin_adj_flag is None and "구분" in full.columns and "매출액영업이익률_보정여부" in full.columns:
        mean_row = full[full["구분"] == "평균"]
        if not mean_row.empty:
            avg_margin_adj_flag = mean_row["매출액영업이익률_보정여부"].iloc[0]

    # 평균 플래그 fallback: 평균 컬럼이 없으면 풀에서 보정 여부를 종합(0이 하나라도 있으면 보정으로 표기)
    if avg_sales_adj_flag is None and "매출액_보정여부" in full.columns:
        flags = pd.to_numeric(full["매출액_보정여부"], errors="coerce").dropna()
        if not flags.empty:
            avg_sales_adj_flag = 0 if (flags == 0).any() else 1

    if avg_margin_adj_flag is None and "매출액영업이익률_보정여부" in full.columns:
        flags = pd.to_numeric(full["매출액영업이익률_보정여부"], errors="coerce").dropna()
        if not flags.empty:
            avg_margin_adj_flag = 0 if (flags == 0).any() else 1

    # 슬라이스(신청기업 없으면 상위 11개 후보), 그 뒤 10개만 남김
    d = _slice_around_target(full, top_n=5, bottom_n=5)

    # 숫자 변환
    d["영업이익률(%)"] = pd.to_numeric(d.get("매출액영업이익률"), errors="coerce")

    is_mean_row = (d["구분"] == "평균") if "구분" in d.columns else pd.Series(False, index=d.index)
    d_comp = d[~is_mean_row].copy()

    if "rank_num" in d_comp.columns:
        d_comp["rank_num"] = pd.to_numeric(d_comp["rank_num"], errors="coerce")
        d_comp = d_comp.sort_values("rank_num", na_position="last").head(10)
    else:
        d_comp = d_comp.head(10)

    # 평균 행 생성(차트에서는 제거, pill/계산용 유지)
    df_avg = pd.DataFrame(
        [{
            "법인번호_ENC": None,
            "상호": "평균",
            "매출액": avg_sales,
            "매출액_보정여부": avg_sales_adj_flag,
            "매출액구간": None,
            "최종매출액구간": None,
            "rank_num": None,
            "구분": "평균",
            "평균매출액": avg_sales,
            "매출액영업이익률": avg_margin,
            "매출액영업이익률_보정여부": avg_margin_adj_flag,
            "영업이익률(%)": avg_margin,
        }]
    )

    d = pd.concat([d_comp, df_avg], ignore_index=True)

    d["유형"] = d.get("구분")
    d["유형"] = d["유형"].fillna("경쟁기업")
    d.loc[~d["유형"].isin(["신청기업", "평균"]), "유형"] = "경쟁기업"

    d["표시명"] = d["상호"].fillna(d["법인번호_ENC"])
    d["표시명"] = d["표시명"].fillna("(식별자 없음)")

    # 표시순위
    d["표시순위"] = np.nan
    is_mean_row = (d["구분"] == "평균") if "구분" in d.columns else pd.Series(False, index=d.index)
    if "rank_num" in d.columns:
        d.loc[~is_mean_row, "표시순위"] = pd.to_numeric(d.loc[~is_mean_row, "rank_num"], errors="coerce")

    # 라벨
    d["라벨"] = d.apply(_make_label, axis=1)

    # 값/표기
    d["매출액(억원)"] = d["매출액"].map(to_eok)
    d["매출액표기"] = d.apply(
        lambda r: f"{_fmt_or_none(r.get('매출액(억원)'), ',.2f')}{_adj_suffix(r.get('매출액_보정여부'))}",
        axis=1,
    )

    d["영업이익률표기"] = d.apply(
        lambda r: (
            f"{_fmt_or_none(r.get('영업이익률(%)'), ',.1f')}%{_adj_suffix(r.get('매출액영업이익률_보정여부'))}"
            if pd.notna(pd.to_numeric(r.get("영업이익률(%)"), errors="coerce"))
            else "정보 없음"
        ),
        axis=1,
    )

    # jin change
    d = _ensure_plain_corpno_col(d)

    # 정렬 키
    if "rank_num" in d.columns:
        d["sort_key"] = pd.to_numeric(d["rank_num"], errors="coerce").fillna(9999).astype(int)
    else:
        d["sort_key"] = np.arange(len(d))

    return d


def create_peer_window_chart(df: pd.DataFrame, title: str | None = None):
    """
    매출 경쟁사 비교(막대 + 우측 y축 선택).
    - 차트에서는 '평균' 막대 제거
    - 보정여부(0=보정) 표기는 막대/툴팁 문자열에서 처리
    """
    if df is None or df.empty:
        return None

    required_cols = {
        "라벨", "매출액(억원)", "유형", "표시명", "법인번호_ENC",
        "sort_key", "영업이익률(%)", "매출액표기", "영업이익률표기",
    }
    d = df.copy() if required_cols.issubset(df.columns) else prepare_peer_df(df)
    if d is None or d.empty:
        return None

    # 평균 제거(평균/영업이익률은 pill로 표기)
    if "유형" in d.columns:
        d = d[d["유형"] != "평균"].copy()
        if d.empty:
            return None

    x_order = d.sort_values("sort_key")["라벨"].tolist()

    # 팔레트: 경쟁기업/신청기업(신청기업이 없으면 경쟁기업만)
    type_order = ["경쟁기업", "신청기업"]
    present = [t for t in type_order if t in d["유형"].unique().tolist()]
    if not present:
        present = ["경쟁기업"]

    palette = {"경쟁기업": "#cfcfcf", "신청기업": "#b7ff4d"}
    color_domain = present
    color_range = [palette[t] for t in present]

    n = len(d)
    bar_width = 50
    width = max(900, 80 * n)

    max_sales = pd.to_numeric(d["매출액(억원)"], errors="coerce").max()
    max_sales = 0.0 if pd.isna(max_sales) else float(max_sales)
    sales_y_max = max(max_sales * 1.35, 1.0)

    has_margin = "영업이익률(%)" in d.columns and pd.to_numeric(d["영업이익률(%)"], errors="coerce").notna().any()
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
    else:
        margin_scale = None

    base = alt.Chart(d).properties(width=width, height=480)
    if title is not None:
        base = base.properties(title=title)

    tooltip = [
        alt.Tooltip("표시명:N", title="기업명"),
        alt.Tooltip("법인번호_ENC:N", title="법인번호_ENC"),
        alt.Tooltip("유형:N", title="구분"),
        alt.Tooltip("표시순위:Q", title="순위"),
        alt.Tooltip("매출액표기:N", title="매출액(억원)"),
        alt.Tooltip("매출액_보정여부:Q", title="매출액 보정여부(0=보정,1=원값)"),
    ]
    if has_margin:
        tooltip.append(alt.Tooltip("영업이익률표기:N", title="영업이익률(%)"))

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
                labelLimit=200,
                labelPadding=8,
                ticks=False,
            ),
        ),
        y=alt.Y(
            "매출액(억원):Q",
            axis=alt.Axis(title="매출액(억원)", orient="left"),
            scale=alt.Scale(domain=[0, sales_y_max]),
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
        layer = bars + text_bar
        return layer.configure_view(stroke="#cbd5e1", strokeWidth=1)

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

    line_text = base.mark_text(
        baseline="top",
        dy=6,
        fontSize=15,
    ).encode(
        x=alt.X("라벨:N", sort=x_order),
        y=alt.Y("영업이익률(%)", type="quantitative", axis=None, scale=margin_scale),
        text=alt.Text("영업이익률표기:N"),
        tooltip=tooltip,
    )

    layer = alt.layer(bars, text_bar, line, line_text).resolve_scale(y="independent")
    return layer.configure_view(stroke="#cbd5e1", strokeWidth=1)


# ---------------------------------------------------
# 성장률 우수기업(수평 바 + 좌/우 패널)
#   - 신청기업 없는 버전
#   - 11번째 빈칸 제거: 평균을 바로 다음 줄로 붙임
# ---------------------------------------------------
def _calc_cagr_3y(s22, s24):
    """
    3년 연평균 성장률(%):
    ((2024/2022)^(1/3) - 1) * 100
    """
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
    max_rank: int = 10,
) -> pd.DataFrame:
    """
    한 패널(100억 이하/초과)용 슬롯 DataFrame 생성.

    구성:
    - 상위 10개(1~10위)
    - 평균(11번째)
    """
    base = df_group.copy() if df_group is not None else pd.DataFrame()

    if (base is not None) and (not base.empty) and ("성장률(%)" not in base.columns):
        base["성장률(%)"] = base.apply(
            lambda r: _calc_cagr_3y(r.get("매출액_2022"), r.get("매출액_2024")),
            axis=1,
        )

    if base is not None and not base.empty and "성장률(%)" in base.columns:
        base["성장률(%)"] = pd.to_numeric(base["성장률(%)"], errors="coerce")

    group_avg = None
    if base is not None and not base.empty and "성장률(%)" in base.columns:
        gvals = pd.to_numeric(base["성장률(%)"], errors="coerce").dropna()
        if not gvals.empty:
            group_avg = float(gvals.mean())

    if base is None or base.empty:
        top = pd.DataFrame(columns=["기업명", "성장률(%)"])
    else:
        top = (
            base.sort_values("성장률(%)", ascending=False)
            .head(max_rank)
            .loc[:, ["기업명", "성장률(%)"]]
            .reset_index(drop=True)
        )

    if not top.empty:
        top["order"] = np.arange(1, len(top) + 1)
        top["표기"] = top.apply(lambda r: f"[{int(r['order'])}위] {str(r['기업명'])}", axis=1)
        top["is_applicant"] = False
        top["is_placeholder"] = False
    else:
        top = pd.DataFrame(
            {
                "기업명": [],
                "성장률(%)": [],
                "order": [],
                "표기": [],
                "is_applicant": [],
                "is_placeholder": [],
            }
        )

    out = top.copy()

    if len(out) < max_rank:
        existing_orders = set(pd.to_numeric(out["order"], errors="coerce").tolist())
        fill_rows = []
        for o in range(1, max_rank + 1):
            if o not in existing_orders:
                fill_rows.append(
                    {
                        "기업명": "",
                        "성장률(%)": 0.0,
                        "order": o,
                        "표기": "",
                        "is_applicant": False,
                        "is_placeholder": True,
                    }
                )
        if fill_rows:
            out = pd.concat([out, pd.DataFrame(fill_rows)], ignore_index=True)
        out = out.sort_values("order").reset_index(drop=True)

    if group_avg is not None:
        avg_row = pd.DataFrame(
            [{
                "기업명": f"{group_label} 평균",
                "성장률(%)": group_avg,
                "order": max_rank + 1,
                "표기": "평균",
                "is_applicant": False,
                "is_placeholder": False,
            }]
        )
        out = pd.concat([out, avg_row], ignore_index=True)

    return out

def growth_rank_chart(
    df_panel: pd.DataFrame,
    panel_title: str,
    height: int = 420,
    width: int = 600,
):
    """
    패널용 가로 막대 차트
    - '평균' 라벨만 막대색(#2f80ed)으로 표시
    - 나머지 라벨은 기존처럼 회색 계열 유지
    """
    d = df_panel.copy()
    if d is None or d.empty:
        return alt.Chart(pd.DataFrame({"y_key": [], "y_label": [], "growth_value": []}))

    d["order"] = pd.to_numeric(d.get("order"), errors="coerce")
    d = d.sort_values("order", ascending=True).reset_index(drop=True)

    d["growth_value"] = pd.to_numeric(d.get("성장률(%)"), errors="coerce").fillna(0.0)

    # ✅ y축을 'order 기반 고유키'로 만들고, 표시 라벨은 따로 둠(빈칸/placeholder 중복 방지)
    d["y_key"] = d["order"].fillna(0).astype(int).astype(str)
    d["y_label"] = d.get("표기").fillna("")

    y_sort = d["y_key"].tolist()

    vals = d["growth_value"]
    v_max = float(vals.max(skipna=True)) if vals.notna().any() else 0.0
    v_min = float(vals.min(skipna=True)) if vals.notna().any() else 0.0
    pad = max(1.0, max(abs(v_max), abs(v_min)) * 0.15)
    x_domain = [v_min - pad, v_max + pad]

    # 전체 폭을 "라벨 패널 + 막대 패널"로 분할
    label_w = 210
    bar_w = max(200, width - label_w)

    base = alt.Chart(d).properties(height=height, title=panel_title)

    x_enc = alt.X(
        "growth_value:Q",
        scale=alt.Scale(domain=x_domain),
        axis=alt.Axis(
            title="성장률(%)",
            grid=True,
            tickCount=6,
            tickMinStep=5,
            gridColor="#e5e7eb",
            gridWidth=1,
        ),
    )

    y_enc_hidden = alt.Y(
        "y_key:N",
        sort=y_sort,
        axis=alt.Axis(title=None, labels=False, ticks=False),
    )

    # ✅ (1) 왼쪽 라벨 전용 패널: "평균"만 파란색
    labels = base.properties(width=label_w).mark_text(
        align="left",
        baseline="middle",
        dx=2,
        fontSize=15,
    ).encode(
        y=alt.Y("y_key:N", sort=y_sort, axis=None),
        text=alt.Text("y_label:N"),
        color=alt.condition(
            alt.datum.y_label == "평균",
            alt.value("#2f80ed"),
            alt.value("#6b7280"),
        ),
        opacity=alt.condition(
            alt.datum.is_placeholder,
            alt.value(0.0),
            alt.value(1.0),
        ),
    )

    # ✅ (2) 막대 패널
    bars = base.properties(width=bar_w).mark_bar(size=22, cornerRadiusEnd=6).encode(
        x=x_enc,
        y=y_enc_hidden,
        opacity=alt.condition(
            alt.datum.is_placeholder,
            alt.value(0.0),
            alt.value(1.0),
        ),
        color=alt.value("#2f80ed"),
        tooltip=[
            alt.Tooltip("기업명:N", title="기업"),
            alt.Tooltip("growth_value:Q", title="성장률(%)", format=",.2f"),
        ],
    )

    text_pos = base.properties(width=bar_w).transform_filter("datum.growth_value >= 0").mark_text(
        align="left",
        baseline="middle",
        dx=6,
        fontSize=15,
    ).encode(
        x=x_enc,
        y=y_enc_hidden,
        text=alt.condition(
            alt.datum.is_placeholder,
            alt.value(""),
            alt.Text("growth_value:Q", format=",.2f"),
        ),
    )

    text_neg = base.properties(width=bar_w).transform_filter("datum.growth_value < 0").mark_text(
        align="right",
        baseline="middle",
        dx=-6,
        fontSize=15,
    ).encode(
        x=x_enc,
        y=y_enc_hidden,
        text=alt.condition(
            alt.datum.is_placeholder,
            alt.value(""),
            alt.Text("growth_value:Q", format=",.2f"),
        ),
    )

    zero_rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
        stroke="black",
        strokeWidth=1,
    ).encode(x="x:Q")

    bars_layer = alt.layer(bars, text_pos, text_neg, zero_rule).resolve_scale(x="shared")

    # ✅ 라벨+막대 결합(같은 y 스케일 공유)
    return alt.hconcat(labels, bars_layer).resolve_scale(y="shared")


# ---------------------------------------------------
# 메인: 재무부문 화면
# ---------------------------------------------------
def render_finance(sel_subcat_name: str, sel_subcat_code: str, send_sql) -> None:
    """
    재무부문 화면 전체를 그리는 메인 함수.

    - 상단: 매출액 기준 경쟁사 비교 (막대)
    - 하단: 성장률 우수기업(최근 3개년 연평균 성장률, 100억원 기준 좌/우 패널)
    """

    st.markdown(
        """
        <div class="sec-banner" style="--accent:#5b9bd5;">
        <div class="sec-label">재무부문</div>
        <div class="sec-rule"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # pill 공통 스타일
    st.markdown(
        """
        <style>
        .fin-metrics{
            display:flex;
            justify-content:center;
            align-items:center;
            gap:14px;
            margin-top:12px;
            margin-bottom:38px;
            flex-wrap:wrap;
        }
        .pill{
            display:inline-block;
            border-radius:12px;
            padding:14px 24px;
            font-weight:800;
            text-align:center;
            border:1.5px solid #cbd5e1;
            line-height:1.15;
            min-width:260px;
        }
        .pill.blue{ background:#eaf0ff; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 세션 상태 소분류 우선
    sel_subcat_name = st.session_state.get("selected_subcat_name", sel_subcat_name)
    sel_subcat_code = st.session_state.get("selected_subcat_code", sel_subcat_code)

    # ---------------------- 1) 매출액 기준 경쟁사 비교 ----------------------
    df_raw = send_sql(
        Q.q_fin_peer_window(),
        params={
            "sel_subcat_code": sel_subcat_code,
            "applicant_corp_no": None,
        },
    )

    df_plot = prepare_peer_df(df_raw) if (df_raw is not None and not df_raw.empty) else df_raw

    # pill 표기용 평균(매출/영업이익률 + 보정 플래그)
    avg_sales = None
    avg_margin = None
    avg_sales_adj_flag = None
    avg_margin_adj_flag = None

    if df_plot is not None and (not df_plot.empty):
        avg_row = None
        if "유형" in df_plot.columns:
            x = df_plot[df_plot["유형"] == "평균"]
            avg_row = x.iloc[0] if not x.empty else None
        elif "구분" in df_plot.columns:
            x = df_plot[df_plot["구분"] == "평균"]
            avg_row = x.iloc[0] if not x.empty else None

        if avg_row is not None:
            avg_sales = pd.to_numeric(avg_row.get("매출액(억원)"), errors="coerce")
            avg_sales = float(avg_sales) if pd.notna(avg_sales) else None

            avg_margin = pd.to_numeric(avg_row.get("영업이익률(%)"), errors="coerce")
            avg_margin = float(avg_margin) if pd.notna(avg_margin) else None

            avg_sales_adj_flag = avg_row.get("매출액_보정여부")
            avg_margin_adj_flag = avg_row.get("매출액영업이익률_보정여부")

    # 차트에는 평균 막대 제거(prepare_peer_df가 평균 포함 → 여기서 제외)
    df_chart = df_plot
    if df_chart is not None and (not df_chart.empty) and ("유형" in df_chart.columns):
        df_chart = df_chart[df_chart["유형"] != "평균"].copy()

    st.markdown(f"### [{sel_subcat_name}] 매출액 기준 경쟁사 비교 (단위: 억원)(2024년 기준)")
    with st.container(border=True):
        chart = create_peer_window_chart(df_chart, title=None)
        if chart is not None:
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("표시할 매출 데이터가 없습니다.")

        sales_txt = f"{_fmt_or_none(avg_sales, ',.2f')}억원" if avg_sales is not None else "정보 없음"
        margin_txt = f"{_fmt_or_none(avg_margin, ',.1f')}%" if avg_margin is not None else "정보 없음"

        st.markdown(
            f"""
            <div class="fin-metrics">
              <div class="pill blue">{sel_subcat_name} 매출액 평균&nbsp;<b>{sales_txt}</b></div>
              <div class="pill blue">{sel_subcat_name} 영업이익률 평균&nbsp;<b>{margin_txt}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ----------------- 2) 성장률 우수기업 -----------------
    sel_subcat_name_state = st.session_state.get("selected_subcat_name", "") or sel_subcat_name

    st.markdown(
        f"### [{sel_subcat_name_state}] 성장률 우수기업(최근 3개년 연평균성장률, 단위: %)(2024년 기준)",
        unsafe_allow_html=True,
    )

    if not sel_subcat_code:
        st.info("소분류 코드가 없습니다.")
        st.stop()

    df_g = send_sql(
        Q.q_growth_top10_by_sales_bucket(),
        params={"sel_subcat_code": sel_subcat_code},
    )

    # 성장률 데이터 상호 매핑용 gmap
    if df_g is not None and not df_g.empty:
        if "매출100억구분" in df_g.columns and "매출구간" not in df_g.columns:
            df_g = df_g.copy()
            df_g["매출구간"] = df_g["매출100억구분"]

        g_ids = df_g["법인번호_ENC"].dropna().astype(str).unique().tolist()
        df_gn = fetch_names_map(g_ids, send_sql) if g_ids else pd.DataFrame(columns=["법인번호_ENC", "상호"])

        df_gn["corp_id"] = df_gn["법인번호_ENC"].map(norm_id)
        gmap = dict(zip(df_gn["corp_id"], df_gn["상호"].map(lambda s: (s or "").strip())))
    else:
        gmap = {}

    if df_g is not None and not df_g.empty:
        df_g = df_g.copy()
        df_g["기업명"] = df_g.apply(lambda row: clean_company_name_growth(row, gmap), axis=1)

        sales_2022 = pd.to_numeric(df_g.get("매출액_2022"), errors="coerce")
        sales_2024 = pd.to_numeric(df_g.get("매출액_2024"), errors="coerce")
        valid = (sales_2022 > 0) & (sales_2024 > 0)

        growth = pd.Series(np.nan, index=df_g.index, dtype="float64")
        ratio = sales_2024[valid] / sales_2022[valid]
        growth.loc[valid] = (np.power(ratio, 1.0 / 3.0) - 1.0) * 100.0
        df_g["성장률(%)"] = growth

    # 소분류 평균 성장률
    df_ag = send_sql(
        Q.q_avg_growth_in_subcat(),
        params={"sel_subcat_code": sel_subcat_code},
    )

    tech_avg_growth = None
    if df_ag is not None and not df_ag.empty:
        s22 = pd.to_numeric(df_ag.get("매출액_2022"), errors="coerce")
        s24 = pd.to_numeric(df_ag.get("매출액_2024"), errors="coerce")
        valid = (s22 > 0) & (s24 > 0)
        if valid.any():
            ratio = s24[valid] / s22[valid]
            growth_series = (np.power(ratio, 1.0 / 3.0) - 1.0) * 100.0
            tech_avg_growth = float(growth_series.mean())

    # 성장률 차트(좌/우 패널)
    with st.container(border=True):
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        has_growth_col = (
            df_g is not None
            and not df_g.empty
            and ("성장률(%)" in df_g.columns)
            and pd.to_numeric(df_g["성장률(%)"], errors="coerce").notna().any()
        )

        if df_g is None or df_g.empty or not has_growth_col:
            st.info("성장률 데이터가 존재하지 않습니다.")
        else:
            df_100le = df_g[df_g["매출구간"] == "100억원 이하"]
            panel_le = prepare_growth_rank_panel(df_group=df_100le, group_label="100억원 이하", max_rank=10)

            df_100gt = df_g[df_g["매출구간"] == "100억원 초과"]
            panel_gt = prepare_growth_rank_panel(df_group=df_100gt, group_label="100억원 초과", max_rank=10)

            has_any = (
                (panel_le is not None and not panel_le.empty)
                or (panel_gt is not None and not panel_gt.empty)
            )
            if not has_any:
                st.info("성장률 데이터가 존재하지 않습니다.")
            else:
                ch1 = growth_rank_chart(panel_le, "매출 100억원 이하 그룹")
                ch2 = growth_rank_chart(panel_gt, "매출 100억원 초과 그룹")

                combo = (
                    alt.hconcat(ch1, ch2)
                    .resolve_scale(x="independent")
                    .configure_axisX(
                        grid=True,
                        tickMinStep=5,
                        gridColor="#e5e7eb",
                        gridWidth=1,
                    )
                    .configure_view(stroke=None)
                )

                st.altair_chart(combo, use_container_width=True)

        g_right = f"{tech_avg_growth:.2f}%" if tech_avg_growth is not None else "정보 없음"
        st.markdown(
            f"""
            <div class="fin-metrics">
              <div class="pill blue"><b>{sel_subcat_name_state}</b> 평균&nbsp;<b>{g_right}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
