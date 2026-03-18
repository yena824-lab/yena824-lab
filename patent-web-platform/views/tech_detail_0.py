"""
views.tech_detail
=================

기술혁신정보 상세 화면(소분류별 탭) 모듈.

Streamlit 기반 BIGx/기술혁신정보서비스에서
기술혁신정보 메인 화면에서 소분류를 선택한 뒤,
해당 소분류의 상세 분석 화면(개요/재무/기술/R&D 탭)을 렌더링하는 모듈이다.

기능 개요
--------

* 상단 공통 내비게이션(뒤로가기 / Home) 버튼을 렌더링한다.
* 소분류별 상세 분석 탭(기술분야 개요 / 재무부문 / 기술부문 / R&D부문)을 제공한다.
* 각 탭에 대해 개별 섹션 렌더링 함수를 호출한다.
* 기술혁신정보 메인 화면에서 선택된 소분류 정보를 기준으로
  상세 분석 결과를 표시한다.

"""

import math
import pandas as pd
import streamlit as st

from main.button.tech_overview import render_overview
# from main.button.company_overview import render_overview
from main.button.company_finance import render_finance
from main.button.tech_tech import render_tech
from main.button.tech_issue import render_issue

from core.db import send_sql


def go(page: str) -> None:
    st.session_state.page = page


def back_from_tech_detail() -> None:
    # 기본은 기술 메인(tech)으로, 외부(예: company_0_detail)에서 들어오면 그 값으로 복귀
    back_page = st.session_state.get("tech_detail_back_page", "tech")

    st.session_state.page = back_page
    st.session_state.pending_nav = None

    # tech_detail에만 걸어두고 싶으면 뒤로가기 1회 사용 후 제거
    if "tech_detail_back_page" in st.session_state:
        del st.session_state["tech_detail_back_page"] 


def set_nav_buttons() -> None:
    left, mid, _, fin_filter, right = st.columns([1, 1, 6, 1, 1])

    # ⬅︎ 뒤로가기 (진입 경로로 복귀)
    with left:
        st.button("⬅︎", on_click=back_from_tech_detail, key="btn_back4")

    # Home 버튼
    with right:
        def _go_home():
            st.session_state.pending_nav = None
            if "tech_detail_back_page" in st.session_state:
                del st.session_state["tech_detail_back_page"]
            go("home")

        st.button("Home", on_click=_go_home, key="btn_home4")

# 🔹 재무 탭 유사도 필터 ON/OFF 토글
def toggle_similarity_filter():
    """
    재무 탭에서 쓸 '유사도 필터링' ON/OFF 토글 함수.
    기본값은 False (미적용).
    """
    current = st.session_state.get("fin_use_similarity", False)
    st.session_state["fin_use_similarity"] = not current


# ---------- (선택) 유사도 기준값 모달 ----------
def filter_modal():
    st.session_state.show_filter_modal = not st.session_state.get(
        "show_filter_modal", False
    )


@st.dialog("기업 유사도 필터 설정", width="small", on_dismiss=filter_modal)
def show_filter_modal():
    if "filter_threshold" not in st.session_state:
        st.session_state.filter_threshold = 1.0

    threshold = st.slider(
        "유사도 기준값",
        0.0,
        1.0,
        float(st.session_state.filter_threshold),
        key="filter_threshold_slider",
    )
    if st.button("적용", key="btn_filter_apply"):
        st.session_state.filter_threshold = threshold
        filter_modal()
        st.rerun()


# ---------- 매출액 구간 설정 모달 ----------
def filter_modal2():
    st.session_state.show_interval_modal = not st.session_state.get(
        "show_interval_modal", False
    )


@st.dialog("매출액 구간 설정", width="small", on_dismiss=filter_modal2)
def show_interval_modal():
    st.markdown(
        "매출액 구간 5개를 설정합니다. (예: 단위 억 원)  \n"
        "- 1구간 최대 = 2구간 최소, 2구간 최대 = 3구간 최소 … 로 자동 연결됩니다.  \n"
        "- 각 구간에서 **최소 < 최대** 이어야 저장되며, **5구간은 최소만 있고 최대는 ∞(이상)** 으로 간주합니다."
    )

    # 세션 값 초기화 (처음 열었을 때)
    if "sales_intervals" not in st.session_state:
        st.session_state.sales_intervals = [
            [0.0, 10.0],
            [10.0, 30.0],
            [30.0, 100.0],
            [100.0, 300.0],
            [300.0, None],
        ]

    for i in range(5):
        min_key = f"interval_{i}_min"
        max_key = f"interval_{i}_max"

        if min_key not in st.session_state:
            st.session_state[min_key] = st.session_state.sales_intervals[i][0]
        if i < 4 and max_key not in st.session_state:
            st.session_state[max_key] = st.session_state.sales_intervals[i][1]

    # 앞 구간 최대 = 뒷 구간 최소 고정
    for i in range(1, 5):
        prev_max_key = f"interval_{i-1}_max"
        curr_min_key = f"interval_{i}_min"
        st.session_state[curr_min_key] = st.session_state[prev_max_key]

    # 입력 UI
    for i in range(5):
        c1, c2 = st.columns(2)
        with c1:
            if i == 0:
                st.number_input(
                    "1구간 최소", key="interval_0_min", step=1.0, format="%.0f"
                )
            else:
                st.number_input(
                    f"{i+1}구간 최소",
                    key=f"interval_{i}_min",
                    step=1.0,
                    format="%.0f",
                    disabled=True,
                )

        with c2:
            if i < 4:
                st.number_input(
                    f"{i+1}구간 최대",
                    key=f"interval_{i}_max",
                    step=1.0,
                    format="%.0f",
                )
            else:
                min_v = st.session_state[f"interval_{i}_min"]
                st.markdown(f"**{int(min_v):,} 이상 (∞)**")

    st.markdown("---")

    col_save, col_cancel = st.columns(2)

    # 저장 버튼
    with col_save:
        if st.button("저장", type="primary", use_container_width=True):
            new_intervals: list[list[float | None]] = []

            for i in range(5):
                min_v = float(st.session_state[f"interval_{i}_min"])

                if i < 4:
                    max_v = float(st.session_state[f"interval_{i}_max"])
                    if max_v <= min_v:
                        st.error(
                            f"{i+1}구간: 최대값({max_v:.0f})은 최소값({min_v:.0f})보다 커야 합니다."
                        )
                        return

                    new_intervals.append([min_v, max_v])
                else:
                    new_intervals.append([min_v, None])

            # 세션에 저장
            st.session_state.sales_intervals = new_intervals
            st.session_state.show_interval_modal = False

            # DB에도 저장 (company_detail과 동일 로직)
            send_sql(
                Q.q_stored_col(),
                params={
                    "c1": new_intervals[0][1],
                    "c2": new_intervals[1][1],
                    "c3": new_intervals[2][1],
                    "c4": new_intervals[3][1],
                },
            )
            st.rerun()

    with col_cancel:
        if st.button("취소", use_container_width=True):
            st.session_state.show_interval_modal = False
            st.rerun()


def render_tech_detail_0() -> None:
    """
    기술혁신정보 상세 화면 전체를 렌더링하는 메인 함수.

    동작 흐름
    ---------

    1. :func:`set_nav_buttons` 를 호출해 상단 [뒤로가기] / [Home] 버튼과
       페이지 타이틀("기술혁신정보")을 렌더링한다.
    2. ``st.session_state["tech_selected"]`` 에서 기술 메인 화면에서 선택된 소분류 정보를 읽어온다.
       * 값이 없으면 경고 메시지를 표시하고 :func:`go("tech")` 호출 후 ``st.stop()`` 한다.
    3. ``st.session_state["subnav"]`` 가 없으면 기본값으로 ``"overview"`` 를 설정한다.
       * 가능한 값: ``"overview"``, ``"finance"``, ``"tech"``, ``"issue"``.
    4. 화면을 좌측 서브 내비게이션 컬럼 / 우측 메인 컨텐츠 컬럼으로 나눈다.
       * 좌측: 네 개의 서브 탭 버튼
         - 1. 기술분야 개요 (``"overview"``)
         - 2. 재무부문 (``"finance"``)
         - 3. 기술부문 (``"tech"``)
         - 4. R&D부문 (``"issue"``)
       * 각 버튼 클릭 시 ``st.session_state["subnav"]`` 값을 변경하고
         :func:`st.rerun` 을 호출하여 동일 페이지를 다시 렌더링한다.
    5. 우측 메인 영역에서는 현재 ``subnav`` 값에 따라 아래 렌더링 함수 중 하나를 호출한다.
       * ``"overview"`` → :func:`button.tech_overview.render_overview`
       * ``"finance"`` → :func:`button.tech_finance.render_finance`
       * ``"tech"`` → :func:`button.tech_tech.render_tech`
       * ``"issue"`` → :func:`button.tech_issue.render_issue`
       * 각 함수에는 선택된 소분류명/코드와 :func:`send_sql` 핸들러를 인자로 전달한다.

    세션 상태
    ---------

    * ``st.session_state["page"]``  
      상위 라우팅에서 사용하는 현재 페이지 키. :func:`go` 에 의해 변경된다.
    * ``st.session_state["tech_selected"]``  
      기술 메인 화면(:func:`views.tech.render_tech`)에서 설정한
      선택 소분류 정보 딕셔너리. 예:  
      ``{"code": "A01B", "name": "소분류명", "label": "A01B - 소분류명"}``
    * ``st.session_state["subnav"]``  
      현재 활성화된 기술 상세 서브 탭. ``"overview"`` / ``"finance"`` / ``"tech"`` /
      ``"issue"`` 중 하나이며, 서브 탭 버튼 클릭 시 변경된다.

    :return: 없음
    :rtype: None
    """
    # 상단 [뒤로가기] / [Home] 버튼
    set_nav_buttons()
    st.title("기업혁신성장 보고서 (BIGx)")

    # ───────── 선택값: tech 페이지에서 넘어온 소분류(필수) ─────────
    sel = st.session_state.get("tech_selected")
    if not sel:
        st.warning("먼저 기술분야를 선택하세요.")
        go("tech")
        st.stop()

    sel_subcat_code = sel.get("code")
    sel_subcat_name = sel.get("name")

    print('-----------------------')
    print(sel_subcat_name)
    print('-----------------------')
    print(st.session_state.get("selected_subcat_name", sel_subcat_name))
    print('-----------------------')


    if not sel_subcat_name:
        st.warning("선택된 소분류명이 없습니다.")
        st.stop()

    # --- 서브 네비 / 필터 기본값 설정 ---
    if "subnav" not in st.session_state:
        # overview | finance | tech | issue
        st.session_state.subnav = "overview"
    if "fin_use_similarity" not in st.session_state:
        st.session_state["fin_use_similarity"] = False
    if "show_interval_modal" not in st.session_state:
        st.session_state.show_interval_modal = False

    # --- 좌/우 레이아웃 (좌측 폭을 줄여 콘텐츠 영역 확대) ---
    nav_col, main_col = st.columns([0.1, 0.9])

    # ---------- 좌측 네비: 큼지막한 버튼 ----------
    with nav_col:
        st.markdown(
            """
            <style>
            .subnav-btn { padding: 14px 16px !important; font-size: 18px !important; font-weight: 800 !important; }
            .subnav-wrap > div { margin-bottom: 10px; }
            </style>
            """,
            unsafe_allow_html=True,
        )

        cur = st.session_state.get("subnav", "overview")

        with st.container():
            st.markdown('<div class="subnav-wrap"></div>', unsafe_allow_html=True)
            b_over = st.button(
                "1. 기술분야 개요",
                type=("primary" if cur == "overview" else "secondary"),
                use_container_width=True,
                key="subnav_overview",
            )
            b_fin = st.button(
                "2. 재무부문",
                type=("primary" if cur == "finance" else "secondary"),
                use_container_width=True,
                key="subnav_finance",
            )
            b_tech = st.button(
                "3. 기술부문",
                type=("primary" if cur == "tech" else "secondary"),
                use_container_width=True,
                key="subnav_tech",
            )
            b_issue = st.button(
                "4. R&D부문",
                type=("primary" if cur == "issue" else "secondary"),
                use_container_width=True,
                key="subnav_issue",
            )

            # Streamlit 버튼에 CSS 클래스 적용을 위한 JS 스니펫
            st.markdown(
                """
                <script>
                const roots = window.parent.document.querySelectorAll('button[kind]');
                roots.forEach(btn => { btn.classList.add('subnav-btn'); });
                </script>
                """,
                unsafe_allow_html=True,
            )

        # 버튼 클릭 시 subnav 변경 및 rerun
        if b_over and cur != "overview":
            st.session_state.subnav = "overview"
            st.rerun()
        if b_fin and cur != "finance":
            st.session_state.subnav = "finance"
            st.rerun()
        if b_tech and cur != "tech":
            st.session_state.subnav = "tech"
            st.rerun()
        if b_issue and cur != "issue":
            st.session_state.subnav = "issue"
            st.rerun()

    # ---------- 우측 콘텐츠 ----------
    with main_col:
        # 공통 CSS
        st.markdown(
            """
            <style>
            .sec-banner{display:flex;align-items:center;margin:8px 0 16px 0;}
            .sec-label{background:#5b9bd5;color:#fff;padding:10px 16px;font-weight:700;font-size:22px;letter-spacing:1px;border-radius:2px;line-height:1;}
            .sec-rule{flex:1;height:6px;background:#5b9bd5;margin-left:12px;border-radius:2px;}
            @media (max-width:480px){.sec-label{font-size:18px;padding:8px 12px}.sec-rule{height:5px}}
            .y-wrap{display:flex;flex-direction:column;gap:12px;}
            .y-box{margin-top:8px;padding:14px 16px;border:2px solid #666;border-radius:10px;background:#f2f6f9;}
            .y-title{display:flex;align-items:center;gap:10px;font-size:18px;font-weight:800;}
            .y-sub{margin-top:4px;font-size:14px;font-weight:500;color:#333;}
            .y-grid{display:grid;grid-template-columns:180px 1fr;gap:8px 14px;margin-top:10px;}
            .y-field{font-weight:700;color:#222;}
            .y-value{color:#111;}
            .y-muted{color:#444;font-size:13px;}
            div[data-testid="stDataFrame"] td, div[data-testid="stDataFrame"] th { padding: 10px 12px; }
            div[data-testid="stDataFrame"] tbody tr:nth-child(odd) { background: #eaf2fb22; }
            div[data-testid="stDataFrame"] tbody tr:nth-child(even){ background: #eaf2fb55; }
            div[data-testid="stDataFrame"] thead th { font-weight: 700; }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # 🔹 매출액 구간 기본값 (company_detail과 동일)
        if "sales_intervals" not in st.session_state:
            st.session_state.sales_intervals = [
                [0.0, 10.0],
                [10.0, 30.0],
                [30.0, 100.0],
                [100.0, 300.0],
                [300.0, None],
            ]

        subnav = st.session_state.subnav

        applicant_corp_no = st.session_state.get("applicant_corp_no", "")
        applicant_name = st.session_state.get("applicant_name", "")

        # 기술분야 개요 탭
        if subnav == "overview":
            render_overview(
                sel_subcat_name=sel_subcat_name,
                sel_subcat_code=sel_subcat_code,
                applicant_corp_no=applicant_corp_no,
                send_sql=send_sql,
            )

        # 재무부문 탭
        elif subnav == "finance":
            render_finance(
                sel_subcat_name=sel_subcat_name,
                sel_subcat_code=sel_subcat_code,
                applicant_corp_no=applicant_corp_no,
                applicant_name=applicant_name,
                send_sql=send_sql,
            )

        # 기술부문 탭
        elif subnav == "tech":
            render_tech(
                sel_subcat_name=sel_subcat_name,
                sel_subcat_code=sel_subcat_code,
                send_sql=send_sql,
            )

        # R&D부문 탭
        elif subnav == "issue":
            render_issue(
                sel_subcat_name=sel_subcat_name,
                sel_subcat_code=sel_subcat_code,
                send_sql=send_sql,
            )

        if st.session_state.get("show_filter_modal"):
            show_filter_modal()

        if st.session_state.get("show_interval_modal"):
            show_interval_modal()