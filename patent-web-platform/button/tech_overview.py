"""
button.tech_overview
====================

기술분야 개요(Overview) 화면 모듈.

Streamlit 기반 BIGx/기술혁신정보서비스에서
기업/특허 개요 화면을 렌더링하는 모듈이다.
공통 UI 함수와 메인 렌더링 함수 :func:`render_overview` 를 제공한다.

기능 개요
--------

* 매출액 구간별 그룹 정의를 위한 상수
  ( :data:`BUCKET_ORDER`, :data:`GROUP_LABELS` )를 제공한다.
* 정보 박스를 렌더링하기 위한 HTML 유틸리티 함수를 제공한다.
* 매출 그룹별 기업/특허 수를 파이 차트로 시각화한다.
* :func:`render_overview` 를 통해
  기술분야 개요 전체 레이아웃과 데이터를 연동하여 화면을 렌더링한다.
"""

import os
import pandas as pd
import altair as alt
import streamlit as st
from typing import Optional

from main.sql import sql_tech_detail as Q
# from main.sql import sql_company_detail as Q1


# ---------------------------------------------------
# 공통 상수
# ---------------------------------------------------
if "sales_intervals" not in st.session_state:
        st.session_state.sales_intervals = [
            [0.0,    10.0],
            [10.0,   30.0],
            [30.0,   100.0],
            [100.0,  300.0],
            [300.0,  None],
        ]
s = st.session_state.sales_intervals
BUCKET_ORDER = [
    f"{int(s[0][1])}억원 이하",
    f"{int(s[1][1])}억원 이하",
    f"{int(s[2][1])}억원 이하",
    f"{int(s[3][1])}억원 이하",
    f"{int(s[3][1])}억원 초과",
]

GROUP_LABELS = {
    "그룹1": f"{int(s[0][1])}억원 이하",
    "그룹2": f"{int(s[1][1])}억원 이하",
    "그룹3": f"{int(s[2][1])}억원 이하",
    "그룹4": f"{int(s[3][1])}억원 이하",
    "그룹5": f"{int(s[3][1])}억원 초과",
}

def _normalize_sales_bucket(v: str) -> str:
    # ✅ 미분류/결측은 10억원 이하로
    if v == "-" or v is None or str(v).strip() == "":
        return GROUP_LABELS["그룹1"]

    s = str(v).strip()

    if s in GROUP_LABELS:  # '그룹1~5'
        return GROUP_LABELS[s]
    if s in GROUP_LABELS.values() or s in BUCKET_TO_GROUP:  # 이미 구간 라벨
        return s

    if s.startswith("~") and s.endswith("억원"):  # 레거시 '~10억원'
        num = s[1:-2]
        return f"{num}억원 이하"
    if s.endswith("억원~"):  # 레거시 '300억원~'
        num = s.replace("억원~", "")
        return f"{num}억원 초과"

    return s


def _to_group(v: object) -> str:
    """DB 값이 '그룹1~5'든, '10억원 이하' 같은 구간이든 -> 그룹명으로 통일"""
    # ✅ 미분류/결측은 그룹1로
    if pd.isna(v) or str(v).strip() == "":
        return "그룹1"

    s = str(v).strip()

    if s == "미분류":
        return "그룹1"

    if s in GROUP_LABELS:     # 이미 '그룹1~5'
        return s
    if s in BUCKET_TO_GROUP:  # '10억원 이하' 같은 구간
        return BUCKET_TO_GROUP[s]
    return s


def _to_bucket_display(row) -> str:
    """화면 표시용: 그룹명은 숨기고, 구간만 표시한다."""
    g = row["그룹"]

    # ✅ 혹시 남아있는 '미분류'도 10억원 이하로 표시
    if g == "미분류":
        return GROUP_LABELS["그룹1"]

    bucket = GROUP_LABELS.get(g)
    if bucket:
        return bucket

    raw_bucket = str(row["구간"]).strip()
    return raw_bucket or g

# ---------------------------------------------------
# 공통 UI 함수
# ---------------------------------------------------
def make_info_box_html(title: str, subtitle: str | None, fields: list[tuple[str, str]]) -> str:
    """
    소분류 정보 박스용 HTML 문자열을 생성한다.

    :param title: 박스 상단 제목이다.
    :type title: str
    :param subtitle: 제목 아래 작은 설명이다. 필요 없으면 ``None`` 을 넘긴다.
    :type subtitle: str | None
    :param fields: (필드명, 값) 튜플 리스트이다.
    :type fields: list[tuple[str, str]]
    :return: 렌더링에 사용할 HTML 문자열이다.
    :rtype: str
    """
    html = ['<div class="y-box">']  # 바깥 박스 wrapper

    # 타이틀 렌더링
    html.append(f'<div class="y-title">{title}</div>')

    # 서브 타이틀이 있으면 추가
    if subtitle:
        html.append(f'<div class="y-sub">{subtitle}</div>')

    # 필드/값이 있으면 2열 그리드 형태로 렌더링
    if fields:
        html.append('<div class="y-grid">')
        for k, v in fields:
            # 왼쪽은 필드명, 오른쪽은 값
            html.append(f'<div class="y-field">{k}</div><div class="y-value">{v}</div>')
        html.append('</div>')

    html.append('</div>')
    return "\n".join(html)

def make_info_box_html_1(title: str, subtitle: str | None, fields: list[tuple[str, str]]) -> str:
    """
    CSS 스타일 정의까지 포함된 정보 박스(info box) HTML 문자열을 생성한다.

    원래 :func:`render_overview` 함수 내부에 있던 버전을
    별도 함수로 분리한 형태이다.

    :param title: 박스의 제목 텍스트
    :type title: str
    :param subtitle: 박스의 부제목 텍스트. 필요 없으면 ``None``
    :type subtitle: str or None
    :param fields: (필드명, 값) 쌍의 리스트.
                   각 튜플은 왼쪽 라벨과 오른쪽 값을 의미한다.
    :type fields: list[tuple[str, str]]
    :return: CSS 및 내용이 포함된 HTML 문자열
    :rtype: str
    """
    html = ["""
    <style>
    .y-box {
        width: 100%;
        background-color: #fafafa;
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    .y-title {
        font-weight: 600;
        font-size: 18px;
        margin-bottom: 8px;
    }
    .y-sub {
        color: #666;
        font-size: 14px;
        margin-bottom: 12px;
    }
    .y-grid {
        display: grid;
        grid-template-columns: 15% 85%;
        column-gap: 8px;
        row-gap: 6px;
        align-items: start;
        word-break: break-word;
    }
    .y-field {
        font-weight: 600;
        color: #333;
        white-space: nowrap;
    }
    .y-value {
        color: #444;
        word-break: break-word;
    }
    .y-muted {
        color: #666;
    }
    </style>
    """]
    html.append(f'<div class="y-box">')
    html.append(f'<div class="y-title">{title}</div>')
    if subtitle:
        html.append(f'<div class="y-sub">{subtitle}</div>')
    if fields:
        html.append('<div class="y-grid">')
        for k, v in fields:
            html.append(
                f'<div class="y-field">{k}</div><div class="y-value">{v}</div>'
            )
        html.append('</div>')
    html.append('</div>')
    return "\n".join(html)

def render_info_box(
    container: "st.delta_generator.DeltaGenerator",
    title: str,
    subtitle: str | None,
    fields: list[tuple[str, str]],
) -> None:
    """
    지정한 컨테이너에 정보 박스(info box)를 렌더링한다

    기본적으로 CSS가 포함된 :func:`make_info_box_html_1` 을 사용한다.

    :param container: 내용을 렌더링할 Streamlit 컨테이너
    :type container: streamlit.delta_generator.DeltaGenerator
    :param title: 박스 제목
    :type title: str
    :param subtitle: 박스 부제목. 없으면 ``None``
    :type subtitle: str or None
    :param fields: (필드명, 값) 쌍 리스트
    :type fields: list[tuple[str, str]]
    :return: 없음
    :rtype: None
    """
    container.markdown(
        make_info_box_html_1(title, subtitle, fields),
        unsafe_allow_html=True
    )


def render_subcat_box(
    container,
    sel_subcat_name: str,
    subcat_desc: str,
    company_cnt: int | None = None,
    patent_cnt: int | None = None,
) -> None:
    """
    소분류 정보 박스를 그리는 공통 함수이다.

    :param container: 박스를 렌더링할 Streamlit 컨테이너이다.
    :type container: Any
    :param sel_subcat_name: 현재 선택된 소분류명이다.
    :type sel_subcat_name: str
    :param subcat_desc: 소분류 설명 텍스트이다.
    :type subcat_desc: str
    :param company_cnt: 소분류에 속한 기업 수이다. 없으면 ``None`` 이다.
    :type company_cnt: int | None
    :param patent_cnt: 소분류에 속한 특허 수이다. 없으면 ``None`` 이다.
    :type patent_cnt: int | None
    :return: 없음.
    :rtype: None
    """
    # 기본 필드: 소분류명 / 소분류 설명
    fields: list[tuple[str, str]] = [
        ("소분류명", sel_subcat_name),
        ("소분류 설명", f'<span class="y-muted">{subcat_desc}</span>'),
    ]

    # 기업수/특허수가 주어졌다면 추가 필드로 표시
    if company_cnt is not None and patent_cnt is not None:
        fields += [
            ("소분류에 속한 기업수", f"{company_cnt:,}개"),
            ("소분류에 속한 특허수", f"{patent_cnt:,}개"),
        ]

    # 앞서 만든 HTML 박스를 container에 렌더링
    container.markdown(
        make_info_box_html("소분류 정보", None, fields),
        unsafe_allow_html=True,
    )


def prepare_pie_data(df_pie_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    # 입력 데이터가 비어있으면 빈 DF들과 0 카운트 반환
    if df_pie_raw.empty:
        return (
            pd.DataFrame(columns=["구간", "그룹", "구간표시", "기업수", "비율(%)", "legend_label"]),
            pd.DataFrame(columns=["구간", "그룹", "구간표시", "특허수", "비율(%)", "legend_label"]),
            0,
            0,
        )

    # 컬럼명 통일: "최신매출액구간" -> "구간"
    df_pie = df_pie_raw.rename(columns={"최신매출액구간": "구간"}).copy()

    # 구간 기준 정렬(원본 구간이 들어온다는 가정)
    df_pie = df_pie.sort_values("구간")
    df_pie["그룹"] = df_pie["구간"].apply(_to_group)
    df_pie["구간표시"] = df_pie.apply(_to_bucket_display, axis=1)

    # 전체 기업수/특허수 합계
    company_cnt = int(df_pie["기업수"].sum())
    patent_cnt  = int(df_pie["특허수"].sum())

    # -----------------------------
    # 1) 기업수 기준 파이차트 데이터 (dfA)
    # -----------------------------
    dfA = df_pie.copy()
    totalA = dfA["기업수"].sum()
    dfA["비율(%)"] = (dfA["기업수"] / totalA * 100).round(1) if totalA else 0.0

    # ✅ 범례도 '그룹 (구간)' 기준으로 보이게
    dfA["legend_label"] = dfA.apply(
        lambda r: f'{r["구간표시"]} ({int(r["기업수"]):,}개, {r["비율(%)"]:.1f}%)',
        axis=1,
    )

    # -----------------------------
    # 2) 특허수 기준 파이차트 데이터 (dfB)
    # -----------------------------
    dfB = df_pie.copy()
    totalB = dfB["특허수"].sum()
    dfB["비율(%)"] = (dfB["특허수"] / totalB * 100).round(1) if totalB else 0.0

    dfB["legend_label"] = dfB.apply(
        lambda r: f'{r["구간표시"]} ({int(r["특허수"]):,}개, {r["비율(%)"]:.1f}%)',
        axis=1,
    )

    return dfA, dfB, company_cnt, patent_cnt


def render_pie_charts(
    container: "st.delta_generator.DeltaGenerator",
    dfA: pd.DataFrame,
    dfB: pd.DataFrame,
) -> None:
    if dfA.empty or dfB.empty:
        return

    bucket_order = BUCKET_ORDER.copy()
    if "미분류" in set(dfA["구간표시"]) or "미분류" in set(dfB["구간표시"]):
        bucket_order.append("미분류")

    present = set(dfA["구간표시"]).union(set(dfB["구간표시"]))
    bucket_order = [x for x in bucket_order if x in present]
    if not bucket_order:
        bucket_order = list(present)

    # --------------------------
    # ✅ 구간별 색상 고정 팔레트/스케일
    # --------------------------
    COLOR_RANGE = [
        "#0168C9", "#85C8FC", "#FF2C2C", "#FCAEAB", "#2AB09D",
    ]
    bar_color_scale = alt.Scale(
        domain=bucket_order,
        range=COLOR_RANGE[: len(bucket_order)]
    )

    # ✅ (기존 코드에 sort_key를 쓰고 있는데 컬럼이 없어서) 여기서 만들어줌
    dfA = dfA.copy()
    dfB = dfB.copy()
    dfA["sort_key"] = dfA["구간표시"].apply(lambda x: bucket_order.index(x) if x in bucket_order else 999)
    dfB["sort_key"] = dfB["구간표시"].apply(lambda x: bucket_order.index(x) if x in bucket_order else 999)

    max_a = float(dfA["기업수"].max()) if len(dfA) else 0.0
    max_b = float(dfB["특허수"].max()) if len(dfB) else 0.0
    max_all = max(max_a, max_b)
    x_min, x_max = 0, max_all * 1.1

    # ==========================
    # 1) 상단: 가로 막대그래프
    # ==========================
    bar_colA, bar_colB = container.columns(2)

    with bar_colA:
        st.markdown("#### 매출 규모별 기업현황")
        baseA = alt.Chart(dfA)
        barA = (
            baseA.mark_bar(cornerRadius=6)
            .encode(
                y=alt.Y("구간표시:N", sort=bucket_order, title=None, axis=None),
                x=alt.X("기업수:Q", title="기업수", scale=alt.Scale(domain=[x_max, x_min], nice=False)),
                # ✅ 막대도 구간별 색상 고정
                color=alt.Color("구간표시:N", scale=bar_color_scale, legend=None),
                tooltip=[
                    alt.Tooltip("구간표시:N", title="구간"),
                    alt.Tooltip("기업수:Q", title="기업수", format=","),
                ],
            )
        )
        textA = (
            baseA.mark_text(align="right", baseline="middle", dx=-4)
            .encode(
                y=alt.Y("구간표시:N", sort=bucket_order),
                x=alt.X("기업수:Q"),
                text=alt.Text("기업수:Q", format=","),
            )
        )
        st.altair_chart((barA + textA).properties(height=40 * len(bucket_order), width=450), use_container_width=True)

    with bar_colB:
        st.markdown("#### 매출 규모별 특허현황")
        baseB = alt.Chart(dfB)
        barB = (
            baseB.mark_bar(cornerRadius=6)
            .encode(
                y=alt.Y("구간표시:N", sort=bucket_order, title=None, axis=alt.Axis(labelPadding=10)),
                x=alt.X("특허수:Q", title="특허수", scale=alt.Scale(domain=[x_min, x_max], nice=False)),
                # ✅ 막대도 구간별 색상 고정
                color=alt.Color("구간표시:N", scale=bar_color_scale, legend=None),
                tooltip=[
                    alt.Tooltip("구간표시:N", title="구간"),
                    alt.Tooltip("특허수:Q", title="특허수", format=","),
                ],
            )
        )
        textB = (
            baseB.mark_text(align="left", baseline="middle", dx=4)
            .encode(
                y=alt.Y("구간표시:N", sort=bucket_order),
                x=alt.X("특허수:Q"),
                text=alt.Text("특허수:Q", format=","),
            )
        )
        st.altair_chart((barB + textB).properties(height=40 * len(bucket_order), width=450), use_container_width=True)

    # ==========================
    # 2) 하단: 파이차트
    # ==========================
    pie_colA, pie_colB = container.columns(2)

    # ✅ 파이는 legend_label(문구)로 색을 주되, domain 순서를 bucket_order에 맞추고 range를 동일 팔레트로 고정
    pie_domain_A = dfA.sort_values("sort_key")["legend_label"].tolist()
    pie_domain_B = dfB.sort_values("sort_key")["legend_label"].tolist()

    pie_scale_A = alt.Scale(domain=pie_domain_A, range=COLOR_RANGE[: len(pie_domain_A)])
    pie_scale_B = alt.Scale(domain=pie_domain_B, range=COLOR_RANGE[: len(pie_domain_B)])

    with pie_colA:
        pieA = (
            alt.Chart(dfA)
            .mark_arc(outerRadius=110)
            .encode(
                theta=alt.Theta("기업수:Q", title="기업수"),
                # ✅ 파이도 구간별 색상(순서) 고정
                color=alt.Color(
                    "legend_label:N",
                    scale=pie_scale_A,
                    sort=alt.SortField(field="sort_key", order="ascending"),
                    legend=alt.Legend(title=None),
                ),
                tooltip=[
                    alt.Tooltip("구간표시:N", title="구간"),
                    alt.Tooltip("기업수:Q", title="기업수", format=","),
                ],
            )
        )
        st.altair_chart(pieA, use_container_width=True)

    with pie_colB:
        pieB = (
            alt.Chart(dfB)
            .mark_arc(outerRadius=110)
            .encode(
                theta=alt.Theta("특허수:Q", title="특허수"),
                # ✅ 파이도 구간별 색상(순서) 고정
                color=alt.Color(
                    "legend_label:N",
                    scale=pie_scale_B,
                    sort=alt.SortField(field="sort_key", order="ascending"),
                    legend=alt.Legend(title=None),
                ),
                tooltip=[
                    alt.Tooltip("구간표시:N", title="구간"),
                    alt.Tooltip("특허수:Q", title="특허수", format=","),
                ],
            )
        )
        st.altair_chart(pieB, use_container_width=True)



# ---------------------------------------------------
# 메인: 기술분야 개요 화면
# ---------------------------------------------------
def render_overview(sel_subcat_name: str, sel_subcat_code: str, applicant_corp_no: Optional[str] = None, send_sql=None) -> None:
    """
    기술분야 개요 화면 전체를 그리는 메인 함수이다.

    동작 흐름
    ---------

    1. 상단에 ``"기술분야 개요"`` 섹션 배너를 렌더링한다.
    2. 소분류 설명을 조회하여 소분류 정보 박스(1차)를 렌더링한다.
    3. 매출 구간별 기업수/특허수를 조회하여 파이차트용 데이터로 전처리한다.
    4. 전처리 결과로부터 전체 기업수/특허수를 다시 반영해
       소분류 정보 박스(2차, 기업수/특허수 포함)를 렌더링한다.
    5. 전체 기업 수를 ``st.session_state["company_cnt"]`` 에 저장한다.
    6. 기업/특허 파이차트를 렌더링하고, 하단에 그룹 범례 박스를 표시한다.

    세션 상태
    ---------

    * ``st.session_state["company_cnt"]``:
      선택된 기술분야(소분류)에 속한 전체 기업 수를 저장한다.

    :param sel_subcat_name: 현재 선택된 소분류명이다.
    :type sel_subcat_name: str
    :param sel_subcat_code: 현재 선택된 소분류 코드이다.
    :type sel_subcat_code: str
    :param send_sql: SQL 문자열과 파라미터를 받아 ``pandas.DataFrame`` 을 반환하는 DB 조회 함수이다.
    :type send_sql: Callable[[str, dict | None], pandas.DataFrame]
    :return: 없음.
    :rtype: None
    """

    # 상단 섹션 배너 (파란색 레이블 + 가로 바)
    st.markdown(
        f"""
        <div class="sec-banner" style="--accent:{"#5b9bd5"};">
          <div class="sec-label">{"기술분야 개요"}</div>
          <div class="sec-rule"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 위쪽에는 소분류 요약 박스, 아래쪽에는 차트와 범례를 위한 컨테이너 생성
    c_subcat = st.empty()      # 소분류 정보 박스를 그릴 자리
    c_fin = st.container()
    c_charts = st.container()  # 파이차트 영역
    c_legend = st.container()  # 범례 박스 영역

    # ----------------------------------------------------------------
    # 1. 소분류 정보 조회
    # ----------------------------------------------------------------
    df_desc = send_sql(
        Q.q_subcat_desc(),          # 소분류 설명 조회용 SQL
        params={"nm": sel_subcat_name},
    )

    # 설명 컬럼이 비어있으면 "정보 없음"으로 표시
    subcat_desc_disp = (
        "" if df_desc.empty else str(df_desc.iloc[0]["소분류설명"]).strip()
    ) or "정보 없음"

    # (1차) 소분류 정보 박스 렌더: 설명만 표시
    render_subcat_box(
        container=c_subcat,
        sel_subcat_name=sel_subcat_name,
        subcat_desc=subcat_desc_disp,
    )

    # 기업/재무 정보
    if "company_num" not in st.session_state:
        st.session_state["company_num"] = None
    company_num = st.session_state.get("company_num")
    if company_num is None or str(company_num).strip() == "":
        1 == 1
    else:
        df_desc2 = send_sql(Q.qqqq(), params={"applicant_corp_no": applicant_corp_no})

        if df_desc2.empty:
            corp_name = "-"
            sales_range = "-"
        else:
            corp_name = str(df_desc2.iloc[0].get("기업명", "")).strip() or "-"
            sales_range = str(df_desc2.iloc[0].get("매출액구간", "")).strip() or "-"

        sales_range_disp = _normalize_sales_bucket(sales_range)

        render_info_box(
            c_fin,
            "기업/재무 정보",
            None,
            [
                ("기업명", corp_name),
                ("매출액 구간", sales_range_disp)
            ]
        )

    # ----------------------------------------------------------------
    # 2. 매출 그룹별 기업/특허 수 (파이차트용)
    # ----------------------------------------------------------------
    df_pie_raw = send_sql(
        Q.q_overview_pie_and_counts(),   # 매출구간별 기업수/특허수 조회용 SQL
        params={"sel_subcat_code": sel_subcat_code},
    )

    # df_pie_raw가 비어있으면 0으로 세팅, 아니면 전처리 함수로 가공
    if df_pie_raw.empty:
        company_cnt = 0
        patent_cnt = 0
        dfA = pd.DataFrame()
        dfB = pd.DataFrame()
    else:
        dfA, dfB, company_cnt, patent_cnt = prepare_pie_data(df_pie_raw)

    # (2차) 소분류 정보 박스를 다시 그려서 기업수/특허수까지 함께 표시
    render_subcat_box(
        container=c_subcat,
        sel_subcat_name=sel_subcat_name,
        subcat_desc=subcat_desc_disp,
        company_cnt=company_cnt,
        patent_cnt=patent_cnt,
    )

    # 다른 화면(예: company_detail 전체)에서 활용할 수 있도록 세션에 기업 수 저장
    st.session_state["company_cnt"] = company_cnt

    # 파이차트용 데이터가 있을 때만 차트 렌더링
    if not dfA.empty and not dfB.empty:
        render_pie_charts(c_charts, dfA, dfB)

