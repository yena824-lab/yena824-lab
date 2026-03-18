"""
button.company_overview
====================

기술분야 개요(Overview) 화면 모듈.

Streamlit 기반 BIGx/기술혁신정보서비스에서
기업/특허 개요 화면을 구성하는 모듈이다.
공통 UI 함수와 메인 렌더링 함수 :func:`render_overview` 를 제공한다.

기능 개요
--------

* 매출액 구간별 그룹 정의를 위한 상수
  ( :data:`BUCKET_ORDER`, :data:`GROUP_LABELS` )를 제공한다.
* 정보 박스 영역을 렌더링하기 위한 HTML 유틸리티 함수를 제공한다.
* 매출 그룹별 기업/특허 수를 시각화하는 파이 차트를 렌더링한다.
* :func:`render_overview` 를 통해
  기술분야 개요 전체 레이아웃과 데이터를 연동하여 화면을 렌더링한다.
"""

import streamlit as st
import pandas as pd
import altair as alt
import os

from main.sql import sql_company_detail as Q

# --------------------------------------------------------------------
# 공통 상수
# --------------------------------------------------------------------
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

# BUCKET_TO_GROUP = {v: k for k, v in GROUP_LABELS.items()}

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


# --------------------------------------------------------------------
# 공통 UI 함수
# --------------------------------------------------------------------
def make_info_box_html(title: str, subtitle: str | None, fields: list[tuple[str, str]]) -> str:
    """
    CSS 스타일을 포함하지 않는 기본 정보 박스(info box) HTML 문자열을 생성한다.

    :param title: 박스의 제목 텍스트
    :type title: str
    :param subtitle: 박스의 부제목 텍스트. 필요 없으면 ``None``
    :type subtitle: str or None
    :param fields: (필드명, 값) 쌍의 리스트.
                   각 튜플은 왼쪽 라벨과 오른쪽 값을 의미한다.
    :type fields: list[tuple[str, str]]
    :return: 생성된 HTML 문자열
    :rtype: str
    """
    html = [f'<div class="y-box">']
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


def draw_subcat_box(
    container: "st.delta_generator.DeltaGenerator",
    subcat_name_disp: str,
    subcat_desc_disp: str,
    company_cnt: int | None = None,
    patent_cnt: int | None = None,
) -> None:
    """
    소분류 관련 정보를 요약한 박스를 렌더링한다.

    기업 수/특허 수는 선택적으로 표시할 수 있다.

    :param container: 내용을 렌더링할 Streamlit 컨테이너
    :type container: streamlit.delta_generator.DeltaGenerator
    :param subcat_name_disp: 소분류명 표시 문자열
    :type subcat_name_disp: str
    :param subcat_desc_disp: 소분류 설명 표시 문자열
    :type subcat_desc_disp: str
    :param subcat_reason_disp: 소분류 매핑 이유 표시 문자열
    :type subcat_reason_disp: str
    :param company_cnt: 해당 소분류에 속한 기업 수. 미표시하려면 ``None``
    :type company_cnt: int or None
    :param patent_cnt: 해당 소분류에 속한 특허 수. 미표시하려면 ``None``
    :type patent_cnt: int or None
    :return: 없음
    :rtype: None
    """
    # ---------------------------------------------------
    # 2) 소분류 정보 박스 (아래쪽 전체 폭)
    # ---------------------------------------------------
    fields: list[tuple[str, str]] = [
        ("소분류명", subcat_name_disp),
        ("소분류 설명", f'<span class="y-muted">{subcat_desc_disp}</span>'),
    ]

    if company_cnt is not None and patent_cnt is not None:
        fields += [
            ("소분류에 속한 기업수", f"{company_cnt:,}개"),
            ("소분류에 속한 특허수", f"{patent_cnt:,}개"),
        ]

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

    # ✅ 그룹 컬럼은 '그룹1~5/미분류' 로 통일
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

# --------------------------------------------------------------------
# 메인: 기술분야 개요 화면
# --------------------------------------------------------------------
def render_overview(
    sel_subcat_name: str,
    sel_subcat_code: str,
    applicant_corp_no: str,
    applicant_name: str,
    selected_patent_title: str,
    send_sql,
) -> None:
    """
    기술분야 개요 화면 전체를 그리는 메인 렌더링 함수이다.

    동작 흐름
    ---------
    1. 상단에 '기술분야 개요' 섹션 배너를 렌더링하고, 화면 구성에 사용할
       컨테이너(c_basic, c_subcat, c_fin, c_charts, c_legend)를 초기화한다.
    2. ``st.session_state["ui_selected_small"]`` 값이 존재하면
       ``"코드 - 명칭"`` 형식의 값을 파싱하여 소분류 코드/명을 갱신하고,
       그렇지 않으면 기존 세션 상태 값을 유지한다.
    3. 세션 상태에서 선택된 특허명/출원번호/기업명을 읽어오고,
       필요 시 파라미터로 전달된 기본값을 사용한다.
    4. :func:`sql_company_detail.q_patent_abs` 를 이용해 선택 특허의
       설명/출원번호를 조회하고, 특허 기본 정보를 정보 박스로 렌더링한다.
    5. 선택된 소분류/출원번호 정보를 이용해
       :func:`sql_company_detail.q_subcat_info` 또는
       :func:`sql_company_detail.q_subcat_desc_only` 쿼리를 수행하여
       소분류 설명/매핑이유/기업수/특허수를 조회하고,
       :func:`draw_subcat_box` 로 요약 박스를 렌더링한다.
    6. :func:`sql_company_detail.q_corp_finance` 쿼리로 기업/재무 정보를 조회하여
       매출액 구간 및 그룹(그룹1~5/미분류)을 계산하고 정보 박스를 렌더링한다.
    7. :func:`sql_company_detail.q_overview_pie_and_counts` 쿼리 결과를
       매출구간 정렬/그룹 라벨 매핑 후 파이차트용 데이터프레임으로 가공하고,
       :func:`render_pie_charts` 로 기업/특허 파이차트를 렌더링한다.
    8. 소분류별 기업·특허수 정보를 포함한 소분류 박스를 다시 한 번 렌더링하고,
       하단에 :func:`render_group_legend` 로 매출 그룹 범례 박스를 표시한다.

    세션 상태
    ---------
    이 함수는 다음과 같은 ``st.session_state`` 키를 읽거나 설정한다.

    * ``"selected_patent_title"``:
      선택된 특허명 문자열. 기본값이 없으면 빈 문자열로 초기화한다.
    * ``"selected_subcat_name"`` / ``"selected_subcat_code"``:
      현재 화면에서 사용 중인 소분류명/소분류 코드.
      ``"ui_selected_small"`` 값이 있으면 이를 기준으로 갱신한다.
    * ``"selected_appno_enc"``:
      선택된 특허의 출원번호_ENC. 문자열로 저장된다.
    * ``"applicant_name"``:
      신청 기업명. 기본적으로 상위 화면에서 설정된 값을 사용한다.
    * ``"ui_selected_small"``:
      UI(소분류 선택 위젯)에서 전달되는 ``"코드 - 명칭"`` 형식의 라벨.
      존재할 경우 소분류 코드/명을 이 값으로 덮어쓴다.
    * ``"company_cnt2"``:
      소분류 정보 조회 시 집계된 기업 수(최초 소분류 정보 박스용).
    * ``"company_cnt"``:
      파이차트에 사용된 전체 기업 수(두 번째 소분류 정보 박스용).

    :param sel_subcat_name: 선택된 소분류명(기본값). 세션 상태에서 다시 갱신될 수 있다.
    :type sel_subcat_name: str
    :param sel_subcat_code: 선택된 소분류 코드(기본값). 세션 상태에서 다시 갱신될 수 있다.
    :type sel_subcat_code: str
    :param applicant_corp_no: 신청 기업 법인번호(암호화된 값 등).
    :type applicant_corp_no: str
    :param applicant_name: 신청 기업명.
    :type applicant_name: str
    :param selected_patent_title: 상위 화면에서 전달된 선택 특허명(초기값).
    :type selected_patent_title: str
    :param send_sql: SQL 실행 콜백 함수.
                     ``(query: str, params: dict) -> pandas.DataFrame`` 형태를 가정한다.
    :type send_sql: Callable[[str, dict], pandas.DataFrame]
    :return: 없음.
    :rtype: None
    """
    # 섹션 배너
    st.markdown(
        f"""
        <div class="sec-banner" style="--accent:{"#5b9bd5"};">
        <div class="sec-label">{"기술분야 개요"}</div>
        <div class="sec-rule"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c_basic  = st.container()
    c_subcat = st.empty()
    c_fin    = st.container()
    c_charts = st.container()
    c_legend = st.container()

    # 세션 상태 기본값
    st.session_state.setdefault("selected_patent_title", "")
    st.session_state.setdefault("selected_subcat_name", "")
    st.session_state.setdefault("selected_subcat_code", "")
    st.session_state.setdefault("selected_appno_enc", "")
    st.session_state.setdefault("applicant_name", "")

    # 소분류 선택 처리 (UI에서 선택된 값 우선 적용)
    _small_label = st.session_state.get("ui_selected_small", "")
    if _small_label and " - " in _small_label:
        _small_code, _small_name = _small_label.split(" - ", 1)
        sel_subcat_name = _small_name.strip()
        sel_subcat_code = _small_code.strip()
        st.session_state["selected_subcat_name"] = sel_subcat_name
        st.session_state["selected_subcat_code"] = sel_subcat_code
    else:
        sel_subcat_name = st.session_state.get("selected_subcat_name", "")
        sel_subcat_code = st.session_state.get("selected_subcat_code", "")

    sel_patent_title = st.session_state.get("selected_patent_title", "")
    sel_appno_enc = str(
        st.session_state.get("selected_appno_enc")
        or st.session_state.get("appno_enc")
        or ""
    )
    applicant_name = st.session_state.get("applicant_name", "")

    # ----------------------------------------------------------------
    # 1. 특허 기본 정보
    # ----------------------------------------------------------------
    patent_title_disp = selected_patent_title or "정보 없음"

    df_abs = send_sql(Q.q_patent_abs(), params={"title": sel_patent_title})
    if df_abs.empty:
        patent_abs_disp = "정보 없음"
        patent_appno = ""
    else:
        patent_abs_disp = (str(df_abs.iloc[0].get("특허설명", "")).strip()
                           or "정보 없음")
        patent_appno = str(df_abs.iloc[0].get("출원번호", "")).strip()
    
    render_info_box(
        c_basic,
        "특허 기본 정보",
        None,
        [
            (f"특허명 ({patent_appno})",
             f"{patent_title_disp} ({patent_appno})"),
            ("특허 설명", f'<span class="y-muted">{patent_abs_disp}</span>')
        ]
    )

    # ----------------------------------------------------------------
    # 2. 소분류 정보 조회
    # ----------------------------------------------------------------
    subcat_name_disp   = sel_subcat_name or "정보 없음"
    subcat_reason_disp = "정보 없음"
    subcat_desc_disp   = "정보 없음"

    company_cnt2 = 0
    patent_cnt2  = 0

    if sel_appno_enc:
        df_subcat = send_sql(
            Q.q_subcat_info(),
            params={
                "selected_appno_enc": sel_appno_enc,
                "sel_subcat_code": sel_subcat_code or ""
            }
        )

        if not df_subcat.empty:
            subcat_name_disp   = (str(df_subcat.iloc[0].get("소분류명", "")).strip()
                                  or subcat_name_disp)
            subcat_reason_disp = (str(df_subcat.iloc[0].get("매핑이유", "")).strip()
                                  or "정보 없음")
            subcat_desc_disp   = (str(df_subcat.iloc[0].get("소분류설명", "")).strip()
                                  or "정보 없음")
            company_cnt2 = int(df_subcat.iloc[0].get("기업수", 0))
            patent_cnt2  = int(df_subcat.iloc[0].get("특허수", 0))

    render_info_box(
        c_basic,
        "AI 판단근거",   # 박스 제목
        None,
        [
            # 라벨은 비우고, 내용만 subcat_reason_disp로 한 줄 넣기
            ("", f'<span class="y-muted">{subcat_reason_disp}</span>'),
        ],
    )

    st.session_state["company_cnt2"] = company_cnt2

    draw_subcat_box(
        c_subcat,
        subcat_name_disp,
        subcat_desc_disp,
    )

    # ----------------------------------------------------------------
    # 3. 기업/재무 정보
    # ----------------------------------------------------------------
    df_desc2 = send_sql(Q.q_corp_finance(), params={"nm": sel_appno_enc})

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

    # 4. 매출 그룹별 기업/특허 수 (파이차트용)
    # ----------------------------------------------------------------
    df_pie_raw = send_sql(
        Q.q_overview_pie_and_counts(),
        params={"sel_subcat_code": sel_subcat_code}
    )

    # 이미 만들어둔 헬퍼 함수 사용
    dfA, dfB, company_cnt, patent_cnt = prepare_pie_data(df_pie_raw)

    # 상단 박스 + 세션 상태
    draw_subcat_box(
        c_subcat,
        subcat_name_disp,
        subcat_desc_disp,
        company_cnt,
        patent_cnt,
    )
    st.session_state["company_cnt"] = company_cnt

    # 기업/특허 파이 + 막대 차트
    if not dfA.empty and not dfB.empty:
        render_pie_charts(c_charts, dfA, dfB)


