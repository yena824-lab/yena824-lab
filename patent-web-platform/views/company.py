"""
views.company
=============

기업 검색 및 특허 목록 화면을 렌더링하는 모듈.

Streamlit 기반 BIGx/기술혁신정보서비스에서 기업 검색 및 특허 목록 화면을 제공한다.

기능 개요
--------

* 기업명 / 법인번호_ENC / 사업자번호를 기준으로 기업을 검색한다.
* 검색된 기업에 대해 보유 특허 수를 조회하여 함께 표시한다.
* 사용자가 기업을 선택하면 해당 기업의 특허 목록을 조회한다.
* 사용자가 특허 한 건을 선택하면 선택 정보를 ``st.session_state`` 에 저장하고
  ``company_detail`` 페이지로 네비게이션한다.
"""

import math
import pandas as pd
import streamlit as st
import altair as alt

from core.db import send_sql
from main.sql import sql_company as Q
import numpy as np

try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode
except Exception:
    st.error("st-aggrid 설치 필요: pip install streamlit-aggrid")
    st.stop()


def go(page: str) -> None:
    """
    현재 페이지 상태를 변경한다.

    다른 화면 컴포넌트에서 페이지 이동이 필요할 때 사용한다.
    보통 버튼의 ``on_click`` 콜백으로 연결된다.

    :param page: 이동할 페이지 이름 (예: ``"home"``, ``"company"``, ``"company_detail"`` 등).
    :type page: str
    :return: 없음.
    :rtype: None
    """
    st.session_state.page = page


def select_row_from_df_page(
    df: pd.DataFrame,
    key_prefix: str,
    default_applicant_corp: str | None = None,
    default_applicant_name: str | None = None,
    page_size: int = 300,
) -> None:
    """
    특허 목록 DataFrame을 페이지 단위로 나누어 AgGrid에 표시하고,
    선택된 행의 정보를 세션에 저장한 뒤 상세 페이지로 이동한다.

    이 함수는 다음과 같은 역할을 수행한다.

    * ``page_size`` 기준으로 DataFrame을 슬라이싱하여 대용량 렌더링 비용을 줄인다.
    * AgGrid에 단일 선택(체크박스) 기능을 활성화한다.
    * 선택된 특허의 주요 정보(특허명, 소분류명/코드, 출원번호_ENC)를
      ``st.session_state`` 에 기록한다.
    * 기업 정보(법인번호_ENC, 상호)가 DataFrame에 없을 경우
      인자로 전달된 기본값을 사용한다.
    * 선택이 발생하면 ``pending_nav`` 를 ``"company_detail"`` 로 설정하고
      ``st.rerun()`` 을 호출하여 상세 화면으로 네비게이션한다.
    * Grid 하단에 페이지 이동(이전/다음/직접 입력) UI와 현재 페이지/총 페이지 정보를 표시한다.

    :param df: 전체 특허 목록 DataFrame.
    :type df: pandas.DataFrame
    :param key_prefix: 위젯 키 구분을 위한 접두어. 한 화면에 Grid가 여러 개 있을 때 충돌을 방지한다.
    :type key_prefix: str
    :param default_applicant_corp: DataFrame에 법인번호 컬럼이 없을 때 사용할 기본 법인번호_ENC.
    :type default_applicant_corp: str | None
    :param default_applicant_name: DataFrame에 상호 컬럼이 없을 때 사용할 기본 기업명.
    :type default_applicant_name: str | None
    :param page_size: 한 페이지에 표시할 행(row)의 개수.
    :type page_size: int
    :return: 없음. 선택 시 내부에서 ``st.rerun()`` 이 호출될 수 있다.
    :rtype: None
    """
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

    try:
        # --- 페이지 계산 ---
        total_rows = len(df)
        total_pages = max(1, math.ceil(total_rows / page_size))
        page_key = f"{key_prefix}_page"

        page = st.session_state.get(page_key, 1)
        if page < 1:
            page = 1
        if page > total_pages:
            page = total_pages

        start = (page - 1) * page_size
        end = min(start + page_size, total_rows)
        df_view = df.iloc[start:end]

        # --- Grid 옵션 구성 ---
        gb = GridOptionsBuilder.from_dataframe(df_view)
        gb.configure_selection("single", use_checkbox=True)
        gb.configure_default_column(resizable=True, min_column_width=120)

        # 내부 식별자로 사용하는 컬럼은 숨김 처리
        if "출원번호_ENC" in df_view.columns:
            gb.configure_column("출원번호_ENC", hide=True)

        gb.configure_grid_options(
            domLayout="autoHeight",
            animateRows=False,
            ensureDomOrder=True,
            suppressRowTransform=True,
        )

        grid = AgGrid(
            df_view,
            gridOptions=gb.build(),
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            fit_columns_on_grid_load=True,
            key=f"{key_prefix}_aggrid_{page}",
        )

        # --- 선택된 행 처리 ---
        sel = grid.get("selected_rows", [])
        if sel:
            # df 전체에서 원본 행을 다시 찾기 위한 기준 컬럼
            ref_col = next(
                (c for c in ["출원번호_ENC", "특허명", "소분류코드", "소분류명"] if c in df.columns),
                None,
            )
            if ref_col:
                ref_val = sel[0].get(ref_col)
                if ref_val is not None:
                    idx = df.index[df[ref_col] == ref_val]
                    if len(idx) > 0:
                        row = df.loc[idx[0]]

                        # 특허/기술 분류 정보
                        st.session_state.selected_patent_title = str(row.get("특허명", ""))
                        st.session_state.selected_subcat_name = str(row.get("소분류명", ""))
                        st.session_state.selected_subcat_code = str(row.get("소분류코드", ""))
                        st.session_state.selected_appno_enc = str(
                            row.get("출원번호_ENC", "")
                        )

                        # 기업 정보 (없으면 기본값 사용)
                        st.session_state.applicant_corp_no = (
                            str(row.get("법인번호_ENC", ""))
                            if "법인번호_ENC" in df.columns
                            else (default_applicant_corp or "")
                        )
                        st.session_state.applicant_name = (
                            str(row.get("상호", ""))
                            if "상호" in df.columns
                            else (default_applicant_name or "")
                        )

                        # ✅ company_detail 진입 시 항상 "기술분야 개요"부터 시작
                        st.session_state.subnav = "overview"

                        # 상세 페이지로 이동 플래그 설정
                        st.session_state.pending_nav = "company_detail"
                        st.rerun()

        # --- 페이지네이션 UI ---
        st.divider()
        c1, c2, c3, c4, c5 = st.columns([1, 2, 2, 2, 1])

        with c2:
            if st.button("⬅ 이전", use_container_width=True, key=f"{key_prefix}_prev"):
                st.session_state[page_key] = max(1, page - 1)
                st.rerun()

        with c3:
            st.markdown(f"**페이지**: {page} / {total_pages}")
            new_page = st.number_input(
                "이동",
                min_value=1,
                max_value=total_pages,
                value=page,
                step=1,
                key=f"{key_prefix}_jump",
            )

        with c4:
            if st.button("다음 ➡", use_container_width=True, key=f"{key_prefix}_next"):
                st.session_state[page_key] = min(total_pages, page + 1)
                st.rerun()

        if new_page != page:
            st.session_state[page_key] = int(new_page)
            st.rerun()

        st.caption(
            f"표시 범위: {start+1:,}–{end:,} / 총 {total_rows:,}건 (페이지 크기 {page_size:,})"
        )
        return None

    except Exception as e:  # pragma: no cover - UI 예외 처리
        st.exception(e)
        return None


@st.cache_data(ttl=600)
def fetch_patents_for_company(corp_id: str) -> pd.DataFrame:
    """
    특정 기업(법인번호_ENC 기준)의 특허 목록을 조회한다.

    데이터베이스 조회 결과는 600초 동안 캐싱된다.
    특허 출원/등록일자는 ``YYYY-MM-DD`` 형식의 문자열로 변환된다.

    :param corp_id: 조회 대상 기업의 법인번호_ENC.
    :type corp_id: str
    :return: 특허명, 출원번호/출원번호_ENC, 출원/등록일자, 소분류명/코드가 포함된 DataFrame.
    :rtype: pandas.DataFrame
    """
    df = send_sql(Q.q_patents_for_company(), params={"corp_id": corp_id}).copy()

    if df.empty:
        return df

    df["특허등록일자"] = pd.to_datetime(df["특허등록일자"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    df["특허출원일자"] = pd.to_datetime(df["특허출원일자"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )

    return df[
        ["특허명", "출원번호_ENC", "출원번호", "특허등록일자", "특허출원일자", "소분류명", "소분류코드"]
    ]


def prepare_midclass_pie_data(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    동종업체 기술 중분류별 특허 분포를 파이차트용 데이터로 전처리한다.
    """
    if df_raw.empty:
        empty_cols = ["중분류코드", "중분류명", "특허수", "비율(%)", "legend_label"]
        return pd.DataFrame(columns=empty_cols), 0

    df = df_raw.copy()

    # 전체 특허 수
    total_patent = int(df["특허수"].sum())

    if total_patent > 0:
        df["비율(%)"] = (df["특허수"] / total_patent * 100).round(1)
    else:
        df["비율(%)"] = 0.0

    df["legend_label"] = df.apply(
        lambda r: f'{r["중분류명"]} ({int(r["특허수"]):,}건, {r["비율(%)"]:.1f}%)',
        axis=1,
    )

    return df, total_patent


def render_peer_tech_pies(
    container: "st.delta_generator.DeltaGenerator",
    df_mid: pd.DataFrame,
    df_sub: pd.DataFrame,
) -> None:
    """
    동종업체 기술 중분류/소분류별 특허 분포 파이차트를
    한 행에 나란히 렌더링한다. 각 차트는 상위 10개 + 기타만 표시한다.
    """
    if df_mid.empty and df_sub.empty:
        container.info("표시할 동종업체 기술 분포 데이터가 없습니다.")
        return

    # 상위 10개 + 기타로 축소
    df_mid_top = _top_n_with_other(df_mid, code_col="중분류코드", name_col="중분류명", n=10)
    df_sub_top = _top_n_with_other(df_sub, code_col="소분류코드", name_col="소분류명", n=10)

    with container:
        col_mid, col_sub = st.columns(2)

        # ----- 왼쪽: 중분류 -----
        with col_mid:
            st.markdown("#### 기술보유현황-중분류")

            if df_mid_top.empty:
                st.info("중분류 데이터가 없습니다.")
            else:
                pie_mid = (
                    alt.Chart(df_mid_top)
                    .mark_arc(outerRadius=110)
                    .encode(
                        theta=alt.Theta("특허수:Q", title="특허수"),
                        color=alt.Color(
                            "legend_label:N",
                            legend=alt.Legend(title=None),
                            sort=alt.SortField(field="_legend_sort", order="descending"),
                            # 중분류용 색 팔레트 (파란/초록 계열)
                            scale=alt.Scale(
                                range=[
                                    "#1f77b4", "#2ca02c", "#17becf", "#aec7e8",
                                    "#98df8a", "#ffbb78", "#c5b0d5", "#9edae5",
                                    "#c7c7c7", "#bcbd22", "#7f7f7f",
                                ]
                            ),
                        ),
                        tooltip=[
                            alt.Tooltip("중분류코드:N", title="중분류코드"),
                            alt.Tooltip("중분류명:N", title="중분류명"),
                            alt.Tooltip("특허수:Q", title="특허수", format=","),
                            alt.Tooltip("비율(%):Q", title="비율(%)", format=".1f"),
                        ],
                    )
                ).properties(width=380, height=300)

                st.altair_chart(pie_mid, use_container_width=True)

        # ----- 오른쪽: 소분류 -----
        with col_sub:
            st.markdown("#### 기술보유현황-소분류")

            if df_sub_top.empty:
                st.info("소분류 데이터가 없습니다.")
            else:
                pie_sub = (
                    alt.Chart(df_sub_top)
                    .mark_arc(outerRadius=110)
                    .encode(
                        theta=alt.Theta("특허수:Q", title="특허수"),
                        color=alt.Color(
                            "legend_label:N",
                            legend=alt.Legend(title=None),
                            sort=alt.SortField(field="_legend_sort", order="descending"),
                            # 소분류용 색 팔레트 (주황/보라 계열)
                            scale=alt.Scale(
                                range=[
                                    "#ff7f0e", "#d62728", "#9467bd", "#8c564b",
                                    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
                                    "#ff9896", "#c49c94", "#f7b6d2",
                                ]
                            ),
                        ),
                        tooltip=[
                            alt.Tooltip("소분류코드:N", title="소분류코드"),
                            alt.Tooltip("소분류명:N", title="소분류명"),
                            alt.Tooltip("특허수:Q", title="특허수", format=","),
                            alt.Tooltip("비율(%):Q", title="비율(%)", format=".1f"),
                        ],
                    )
                ).properties(width=380, height=300)

                st.altair_chart(pie_sub, use_container_width=True)


def prepare_subclass_pie_data(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    동종업체 기술 소분류별 특허 분포를 파이차트용 데이터로 전처리한다.

    :param df_raw: (소분류코드, 소분류명, 특허수) 컬럼을 포함한 원본 DataFrame이다.
    :type df_raw: pandas.DataFrame
    :return: (파이차트용 DataFrame, 전체 특허 수) 튜플이다.
    :rtype: tuple[pandas.DataFrame, int]
    """
    if df_raw.empty:
        empty_cols = ["소분류코드", "소분류명", "특허수", "비율(%)", "legend_label"]
        return pd.DataFrame(columns=empty_cols), 0

    df = df_raw.copy()
    total_patent = int(df["특허수"].sum())

    if total_patent > 0:
        df["비율(%)"] = (df["특허수"] / total_patent * 100).round(1)
    else:
        df["비율(%)"] = 0.0

    df["legend_label"] = df.apply(
        lambda r: f'{r["소분류명"]} ({int(r["특허수"]):,}건, {r["비율(%)"]:.1f}%)',
        axis=1,
    )

    return df, total_patent


def render_company_header(back_page: str = "home") -> None:
    """
    기업 검색/특허 목록 화면 상단의 공통 헤더를 렌더링한다.

    좌측에는 뒤로 가기 버튼, 우측에는 Home 버튼이 배치된다.
    각각 ``go()`` 를 통해 페이지 상태를 변경한다.

    :param back_page: 뒤로 가기 버튼 클릭 시 이동할 페이지 이름. 기본값은 ``"home"`` 이다.
    :type back_page: str
    :return: 없음.
    :rtype: None
    """
    left, _, right = st.columns([1, 6, 1])

    with left:
        st.button(
            "⬅︎",
            on_click=lambda: go(back_page),
            key=f"btn_back_company_{back_page}",
        )

    with right:
        st.button(
            "Home",
            on_click=lambda: go("home"),
            key=f"btn_home_company_{back_page}",
        )


def _top_n_with_other(df: pd.DataFrame, code_col: str, name_col: str,
                      n: int = 10) -> pd.DataFrame:
    """
    특허수 기준 상위 N개만 남기고 나머지는 '기타'로 묶는다.
    code_col/name_col 은 중분류/소분류 컬럼명.
    """
    if df.empty:
        return df

    df = df.sort_values("특허수", ascending=False).copy()
    top = df.head(n)
    rest = df.iloc[n:]

    if not rest.empty:
        other_row = {
            code_col: "OTHER",
            name_col: "기타",
            "특허수": int(rest["특허수"].sum()),
        }
        top = pd.concat([top, pd.DataFrame([other_row])], ignore_index=True)

    # 비율(%) 다시 계산
    total = int(top["특허수"].sum())
    if total > 0:
        top["비율(%)"] = (top["특허수"] / total * 100).round(1)
    else:
        top["비율(%)"] = 0.0

    # legend_label 재생성
    top["legend_label"] = top.apply(
        lambda r: f'{r[name_col]} ({int(r["특허수"]):,}건, {r["비율(%)"]:.1f}%)',
        axis=1,
    )

    top["_legend_sort"] = top["특허수"]
    top.loc[top[name_col] == "기타", "_legend_sort"] = -1

    return top


def render_company() -> None:
    """
    기업 검색 및 특허 목록 화면 전체를 렌더링한다.

    :return: 없음
    :rtype: None
    """
    render_company_header(back_page="home")

    if "selected_company_row" not in st.session_state:
        st.session_state.selected_company_row = None

    if "sales_intervals" not in st.session_state:
        st.session_state.sales_intervals = [
            [0.0,    10.0],
            [10.0,   30.0],
            [30.0,   100.0],
            [100.0,  300.0],
            [300.0,  None],
        ]

    st.title("기업혁신성장보고서")

    st.markdown(
        """
        <div style="
            border: 1px dashed #999;
            padding: 16px 18px;
            /* 위/아래 0, 오른쪽만 크게 잘라서 Home 버튼 쪽에서 끊기게 */
            margin: 0 150px 12px 0;
            font-size: 15px;
            line-height: 1.8;
            background-color: #fafafa;
        ">
            본 보고서는 국내 등록특허에 대한 AI 분석 결과에 기반하고 있습니다.
            기술보증기금은 AI를 통해 귀사가 보유하고 있는 특허를 분석하여
            ‘국가과학기술표준분류체계’에 따른 기술분야로 분류하고,
            분류된 기술분야를 기준으로 다양한 기술혁신정보를 맞춤형으로 제공하고 있습니다.<br>
            본 보고서의 내용은 AI 기반으로 자동 생성된 것으로,
            AI 특성상 분석 결과 등에 일부 오류 또는 불완전한 정보가 포함될 수 있음을
            유의하여 주시기 바랍니다.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- 검색 대상 선택 ---
    search_target = st.radio(
        "검색 대상:",
        options=["기업명", "법인번호", "사업자번호"],
        index=0,
        horizontal=True,
    )

    placeholder_text = {
        "기업명": "기업명을 입력하세요 (예: 삼성)(공공기관, 산학협력단 입력금지)",
        "법인번호": "법인번호를 입력하세요.",
        "사업자번호": "사업자번호를 입력하세요.",
    }[search_target]

    # --- 검색 입력 폼 ---
    with st.form("company_search_form", clear_on_submit=False):
        c1, c2 = st.columns([5, 1])
        with c1:
            keyword = st.text_input(
                "",
                placeholder=placeholder_text,
                label_visibility="collapsed",
            )
        with c2:
            do_search = st.form_submit_button("검색", use_container_width=True)

    # 검색 버튼 클릭 시 상태 값 갱신
    if do_search and keyword.strip():
        st.session_state.search_keyword = keyword.strip()
        st.session_state.search_target = search_target
        st.session_state.selected_company_row = None

    # --- 검색 상태가 있을 때만 실제 쿼리 실행 ---
    if st.session_state.get("search_keyword", ""):
        keyword = st.session_state["search_keyword"]
        search_target = st.session_state.get("search_target", "기업명")

        has_patent_in_result = False

        # 검색 대상별 SQL 선택
        if search_target == "기업명":
            sql_company_fast = Q.q_company_fast_by_name()
            params = {"kw": f"%{keyword}%"}
        elif search_target == "법인번호":
            sql_company_fast = Q.q_company_fast_by_corp_enc()
            params = {"corp_id": f"%{keyword.strip()}%"}
        else:  # 사업자번호
            sql_company_fast = Q.q_company_fast_by_bizno()
            params = {"bizno": f"%{keyword.strip()}%"}
            
        companies_fast = send_sql(sql_company_fast, params=params)
        if companies_fast.empty:
            st.warning("검색 결과가 없습니다.")
        else:
            # --- 기업명 컬럼 정리 ---
            name_col = None
            for c in ["기업명", "상호"]:
                if c in companies_fast.columns:
                    name_col = c
                    break

            if name_col is None:
                st.error(
                    f"기업명 컬럼(기업명/상호)을 찾을 수 없습니다. "
                    f"현재 컬럼: {companies_fast.columns.tolist()}"
                )
                return

            if name_col != "기업명":
                companies_fast = companies_fast.rename(columns={name_col: "기업명"})

            df_merged = companies_fast.copy()

            # 법인번호_ENC 타입 통일 (문자열로)
            if "법인번호" in df_merged.columns:
                df_merged["법인번호"] = df_merged["법인번호"].astype(str)
            else:
                st.error(
                    f"'법인번호' 컬럼이 없습니다. 현재 컬럼: {df_merged.columns.tolist()}"
                )
                return

            # 특허수 컬럼이 없으면 기본값 0으로 생성
            if "대표특허수" not in df_merged.columns:
                df_merged["대표특허수"] = 0

            # 1) 전체특허보유수는 숫자로 강제 (문자열이어도 콤마 제거 후 변환)
            df_merged["대표특허수"] = (
                df_merged["대표특허수"]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("개", "", regex=False)
            )
            df_merged["대표특허수"] = pd.to_numeric(df_merged["대표특허수"], errors="coerce").fillna(0).astype(int)

            # 2) 정렬용 숫자 컬럼
            df_merged["특허수_정렬"] = df_merged["대표특허수"]

            # 3) 표시용 문자열 컬럼
            df_merged["특허수"] = df_merged["특허수_정렬"].map(lambda x: f"{x:,}개")

            # 4) 정렬은 숫자로
            df_merged = df_merged.sort_values(["특허수_정렬", "기업명"], ascending=[False, True])

            st.markdown("### 검색 결과")
            st.markdown(
                f"검색 키워드: **{st.session_state.get('search_keyword','')}** "
            )

            # --- 기업 목록 Grid ---
            disp_df = df_merged[["기업명", "법인번호_ENC", "법인번호", "대표특허수", "특허수_정렬"]].rename(columns={"기업명":"기업", "대표특허수":"특허수"})

            gb = GridOptionsBuilder.from_dataframe(disp_df)
            gb.configure_default_column(
                resizable=True,
                sortable=True,
                filter=True,
                flex=1,
                minWidth=160,
            )
            gb.configure_column("기업", flex=2, minWidth=220)
            gb.configure_column("법인번호", flex=2, minWidth=260)
            gb.configure_column("법인번호_ENC", hide=True)
            gb.configure_column("특허수_정렬", hide=True, sort="desc")
            gb.configure_column("특허수", header_name="특허수", flex=1, minWidth=140)
            gb.configure_selection(selection_mode="single", use_checkbox=True)

            # ✅ 핵심: autoHeight 끄고, height로 최대 높이 제한(그리드 내부 스크롤)
            gb.configure_grid_options(domLayout="normal", rowSelection="single")
            gb.configure_grid_options(
                getRowId=JsCode("function(p){ return p.data.법인번호; }")
            )

            ROW_H = 36
            HEADER_H = 36
            MAX_VISIBLE = 12

            if len(disp_df) <= MAX_VISIBLE:
                gb.configure_grid_options(domLayout="autoHeight", rowSelection="single")
                grid_height = None  # height를 강제하지 않음(또는 충분히 큰 값으로)
            else:
                gb.configure_grid_options(domLayout="normal", rowSelection="single")
                grid_height = HEADER_H + ROW_H * MAX_VISIBLE + 8  # 내부 스크롤

            gb.configure_grid_options(
                getRowId=JsCode("function(p){ return p.data.법인번호; }")
            )

            grid = AgGrid(
                disp_df,
                gridOptions=gb.build(),
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                fit_columns_on_grid_load=True,
                height=grid_height,              # autoHeight일 때 None이면 무시/안전
                allow_unsafe_jscode=True,
                theme="alpine",
            )

            sel = grid.get("selected_rows", [])
            if sel:
                st.session_state.selected_company_row = sel[0]

            row = st.session_state.selected_company_row

            if not row:
                st.info("위 표에서 기업을 선택하면 동종업체 기술 분포와 특허 목록이 표시됩니다.")
                return

            corp_id_enc = str(row.get("법인번호_ENC", "")).strip()
            corp_id_plain = str(row.get("법인번호", "")).strip()
            corp_name = str(row.get("기업", "")).strip()

            # 상단에 선택 기업 정보 + 특허 수 표시
            pat_cnt_disp = df_merged.loc[
                df_merged["법인번호"] == corp_id_plain, "특허수"
            ]
            pat_cnt_disp = pat_cnt_disp.iloc[0] if not pat_cnt_disp.empty else "0개"

            try:
                pat_cnt_num = int(str(row.get("특허수", 0)).replace(",", "").strip() or 0)
            except Exception:
                pat_cnt_num = 0

            if pat_cnt_num <= 0 and st.session_state.get("pending_nav") != "company_0_detail":
                # company_0_detail 페이지에서 사용하는 신청기업 정보 세팅
                st.session_state.applicant_corp_no = corp_id_enc
                st.session_state.applicant_name = corp_name

                # 0건 상세에서 사용하는 선택/가드 값 초기화(있으면)
                st.session_state.selected_subcat_row_0detail = None
                st.session_state.company_0_detail_init_corp = None
                st.session_state.company_0_last_nav_subcode = None

                st.session_state.pending_nav = "company_0_detail"
                st.rerun()

            st.markdown(
                """
                <style>
                .company-head {
                    display: flex;
                    gap: 8px;
                    align-items: center;
                    margin: 6px 0 2px;
                    font-size: 15px;
                }
                .company-head .sep { opacity: .6; }
                </style>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="company-head">
                <span><b>기업:</b> {corp_name}</span>
                <span class="sep">|</span>
                <span><b>법인번호:</b> {corp_id_plain}</span>
                <span class="sep">|</span>
                <span><b>특허수:</b> {pat_cnt_disp}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ---------------------------------------------------
            # 1) 동종업체 기술 중분류/소분류별 특허 분포 (파이차트 2개 나란히)
            # ---------------------------------------------------
            pie_container = st.container()

            # 중분류
            df_mid_raw = send_sql(
                Q.q_peer_midclass_patents(),
                params={"corp_id": corp_id_enc},
            )
            df_mid_pie, total_mid_patent = prepare_midclass_pie_data(df_mid_raw)

            # 소분류
            df_sub_raw = send_sql(
                Q.q_peer_subclass_patents(),
                params={"corp_id": corp_id_enc},
            )
            df_sub_pie, total_sub_patent = prepare_subclass_pie_data(df_sub_raw)

            # 두 개 파이차트를 한 행에 렌더링
            render_peer_tech_pies(pie_container, df_mid_pie, df_sub_pie)

            # ---------------------------------------------------
            # 2) 선택 기업의 특허 목록 - 제목은 파이차트 아래로 이동
            # ---------------------------------------------------
            st.markdown("### 선택 기업의 특허 목록")

            with st.spinner("특허 불러오는 중..."):
                patents_df = fetch_patents_for_company(corp_id_enc)

            if patents_df.empty:
                st.info("등록된 특허가 없습니다.")
            else:
                select_row_from_df_page(
                    patents_df[
                        [
                            "특허명",
                            "출원번호",
                            "출원번호_ENC",
                            "특허출원일자",
                            "특허등록일자",
                            "소분류명",
                            "소분류코드",
                        ]
                    ],
                    key_prefix=f"patents_table_{corp_id_enc}",
                    default_applicant_corp=corp_id_enc,
                    default_applicant_name=corp_name,
                )
