"""
views.company_detail
====================

기업혁신성장 보고서(BIGx) 상세 화면 렌더링 모듈.

Streamlit 기반 BIGx/기술혁신정보서비스에서
기업 단위 BIGx 보고서 상세 페이지를 렌더링한다.

기능 개요
--------

* :func:`go`
  - ``st.session_state.page`` 값을 변경하여 페이지 전환을 수행한다.
* :func:`render_settings2`
  - 상단 좌우에 [뒤로가기] / [Home] 버튼을 렌더링한다.
* :func:`render_company_detail`
  - 서브 내비게이션(기술분야 개요 / 재무부문 / 기술부문 / R&D부문)과
    메인 컨텐츠 영역을 구성하고,
    각 서브 탭에 대해 개별 버튼 모듈의 렌더링 함수를 호출한다.

* ``overview`` – 기술분야 개요
* ``finance`` – 재무부문
* ``tech`` – 기술부문
* ``issue`` – R&D 부문

* ``selected_subcat_name`` – 선택된 소분류명
* ``selected_subcat_code`` – 선택된 소분류 코드
* ``applicant_corp_no`` – 신청 기업 법인번호
* ``applicant_name`` – 신청 기업명
* ``selected_patent_title`` – 선택 특허 제목 (기술분야 개요에서 활용)
"""

import streamlit as st
import streamlit.components.v1 as components
from main.button.company_overview import render_overview
from main.button.company_finance import render_finance
from main.button.company_tech import render_tech
from main.button.company_issue import render_issue
from main.sql import sql_company_detail as Q

import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime, date

from core.db import send_sql  
def _scroll_top_once():
    if st.session_state.pop("_scroll_to_top", False):
        components.html(
            """
            <script>
              // Home 키처럼 맨 위로
              window.parent.scrollTo(0, 0);
              document.body.scrollTop = 0;
              document.documentElement.scrollTop = 0;
            </script>
            """,
            height=0,
        )

def go(page: str) -> None:
    _scroll_top_once()
    """
    현재 페이지 상태를 변경한다.

    다른 화면 컴포넌트에서 페이지 이동이 필요할 때 사용한다.
    보통 버튼의 ``on_click`` 콜백으로 연결된다.

    :param page: 이동할 페이지 이름 (예: ``"home"``, ``"company"``, ``"company_detail"`` 등).
    :type page: str
    :return: 없음
    :rtype: None
    """
    st.session_state.page = page

def toggle_similarity_filter():
    """
    재무 탭에서 쓸 '유사도 필터링' ON/OFF 토글 함수.
    기본값은 False (미적용).
    """
    current = st.session_state.get("fin_use_similarity", False)
    st.session_state["fin_use_similarity"] = not current


def render_settings2() -> None:
    #left, mid, _, fin_filter, right = st.columns([1, 1, 6, 1, 1])
    left, mid, _, right = st.columns([1, 1, 7, 1])
    with left:
        st.button("⬅︎", on_click=lambda: go("company"), key="btn_back2")
    # with mid:
    #     st.button('⚙️ 매출액 구분 설정', on_click=filter_modal2, key='btn_interval1')
    with right:
        st.button("Home", on_click=lambda: go("home"), key="btn_home2")
    # with fin_filter:
    #     if st.session_state.subnav == 'finance':
    #         use_sim = st.session_state.get("fin_use_similarity", False)
    #         # 🔹 현재 상태에 따라 버튼 라벨을 "ON" / "OFF" 로 표기
    #         #   - 기본 False(미적용) → 라벨 "ON"
    #         #   - True(적용 중)     → 라벨 "OFF"
    #         label = "기업 유사도 필터링 ON" if not use_sim else "기업 유사도 필터링 OFF"

    #         st.button(
    #             label,
    #             on_click=toggle_similarity_filter,
    #             key="btn_filter1",
    #         )


def filter_modal():
    st.session_state.show_filter_modal = not st.session_state.show_filter_modal

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

def filter_modal2():
    st.session_state.show_interval_modal = not st.session_state.show_interval_modal


@st.dialog("매출액 구간 설정", width="small", on_dismiss=filter_modal2)
def show_interval_modal():
    st.markdown(
        "매출액 구간 5개를 설정합니다. (예: 단위 억 원)  s\n"
        "- 1구간 최대 = 2구간 최소, 2구간 최대 = 3구간 최소 … 로 자동 연결됩니다.  \n"
        "- 각 구간에서 **최소 < 최대** 이어야 저장되며, **5구간은 최소만 있고 최대는 ∞(이상)** 으로 간주합니다."
    )

    for i in range(5):
        min_key = f"interval_{i}_min"
        max_key = f"interval_{i}_max"

        if min_key not in st.session_state:
            st.session_state[min_key] = st.session_state.sales_intervals[i][0]
        if i < 4 and max_key not in st.session_state:
            st.session_state[max_key] = st.session_state.sales_intervals[i][1]

    for i in range(1, 5):
        prev_max_key = f"interval_{i-1}_max"
        curr_min_key = f"interval_{i}_min"
        st.session_state[curr_min_key] = st.session_state[prev_max_key]

    for i in range(5):
        c1, c2 = st.columns(2)
        with c1:
            if i == 0:
                st.number_input("1구간 최소", key="interval_0_min", step=1.0, format="%.0f")
            else:
                st.number_input(f"{i+1}구간 최소", key=f"interval_{i}_min", step=1.0, format="%.0f", disabled=True)

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

    with col_save:
        if st.button("저장", type="primary", use_container_width=True):
            new_intervals = []

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

            st.session_state.sales_intervals = new_intervals
            st.session_state.show_interval_modal = False
            print('send sql 준비')

            # send_sql (Q. )
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



def render_company_detail() -> None:
    """
    기업혁신성장 보고서(BIGx) 상세 페이지 메인 렌더링 함수.

    동작 흐름
    ---------

    1. :func:`render_settings2` 를 호출해 상단 [뒤로가기] / [Home] 버튼과
       페이지 타이틀("기업혁신성장 보고서 (BIGx)")을 렌더링한다.
    2. ``st.session_state["subnav"]`` 가 없으면 기본값으로 ``"overview"`` 를 설정한다.
       * 가능한 값: ``"overview"``, ``"finance"``, ``"tech"``, ``"issue"``.
    3. 상세 페이지 표시를 위해 아래 세션 값을 읽어온다.
       * ``selected_subcat_name`` – 선택된 소분류명 (필수)
       * ``selected_subcat_code`` – 선택된 소분류 코드
       * ``applicant_corp_no`` – 신청 기업 법인번호
       * ``applicant_name`` – 신청 기업명
       * ``selected_patent_title`` – 선택 특허 제목
    4. ``selected_subcat_name`` 이 비어 있으면 상세 분석을 수행할 수 없으므로
       경고 메시지를 출력하고 렌더링을 중단한다.
    5. 화면을 좌측 서브 내비게이션 컬럼 / 우측 메인 콘텐츠 컬럼으로 분할한다.
       * 좌측: 네 개의 서브 탭 버튼
         - 1. 기술분야 개요 (``"overview"``)
         - 2. 재무부문 (``"finance"``)
         - 3. 기술부문 (``"tech"``)
         - 4. R&D부문 (``"issue"``)
       * 각 버튼 클릭 시 ``st.session_state["subnav"]`` 를 변경하고
         :func:`st.rerun` 을 호출하여 동일 페이지를 다시 렌더링한다.
    6. 우측 메인 영역에서는 현재 ``subnav`` 값에 따라 아래 렌더링 함수 중 하나를 호출한다.
       * ``"overview"`` → :func:`button.company_overview.render_overview`
       * ``"finance"`` → :func:`button.company_finance.render_finance`
       * ``"tech"`` → :func:`button.company_tech.render_tech`
       * ``"issue"`` → :func:`button.company_issue.render_issue`
       * 각 함수에는 선택된 소분류명/코드, 기업 정보, :func:`send_sql` 핸들러 등을 인자로 전달한다.

    세션 상태
    ---------

    * ``st.session_state["page"]``  
      상위 라우팅에서 사용하는 현재 페이지 키. :func:`go` 에 의해 변경된다.
    * ``st.session_state["subnav"]``  
      현재 활성화된 상세 서브 탭. ``"overview"``, ``"finance"``, ``"tech"``, ``"issue"`` 중 하나이며,
      좌측 서브 탭 버튼 클릭 시 변경된다.
    * ``st.session_state["selected_subcat_name"]``  
      선택된 소분류명. 기업/특허 선택 과정에서 설정되며, 값이 없으면 상세 화면을 렌더링하지 않는다.
    * ``st.session_state["selected_subcat_code"]``  
      선택된 소분류 코드.
    * ``st.session_state["applicant_corp_no"]``  
      신청 기업 법인번호(ENC). 특허/기업 목록 선택 시 함께 설정된다.
    * ``st.session_state["applicant_name"]``  
      신청 기업명.
    * ``st.session_state["selected_patent_title"]``  
      선택된 특허 제목. 주로 기술분야 개요 탭에서 하이라이트 정보로 활용된다.
    * 이 외에도 각 버튼 모듈(:mod:`button.company_*`) 내부에서
      개별 분석 결과를 표시하기 위해 추가 세션 키를 사용할 수 있다.

    :return: 없음
    :rtype: None
    """
    # 서브 내비게이션 기본값 설정

    if "subnav" not in st.session_state:
        st.session_state.subnav = "overview" 

    if "fin_use_similarity" not in st.session_state:
        st.session_state["fin_use_similarity"] = False

    if "show_interval_modal" not in st.session_state:
        st.session_state.show_interval_modal = False
        
    # 🔹 유사도 필터 모달 초기값
    if "show_filter_modal" not in st.session_state:
        st.session_state.show_filter_modal = False

    render_settings2() 
    st.title("기업혁신성장 보고서 (BIGx)")

    # 세션 상태에서 필수 값 가져오기
    sel_subcat_name = st.session_state.get("selected_subcat_name", "")
    sel_subcat_code = st.session_state.get("selected_subcat_code", "")
    applicant_corp_no = st.session_state.get("applicant_corp_no", "")
    applicant_name = st.session_state.get("applicant_name", "")
    selected_patent_title = st.session_state.get("selected_patent_title", "")

    # 소분류명이 없을 경우, 추가 정보가 없으므로 렌더링 중단
    if not sel_subcat_name:
        st.warning("선택된 소분류명이 없습니다.")
        st.stop()

    # 좌측 내비게이션 / 우측 메인 영역 레이아웃
    nav_col, main_col = st.columns([0.1, 0.9])

    # ------------------------------------------------------------------
    # 좌측 서브 내비게이션 영역
    # ------------------------------------------------------------------
    with nav_col:
        # 버튼 스타일 커스터마이징 (Streamlit 버튼 공통 스타일)
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

            # 렌더링된 Streamlit 버튼에 공통 CSS 클래스 적용
            st.markdown(
                """
                <script>
                const roots = window.parent.document.querySelectorAll('button[kind]');
                roots.forEach(btn => { btn.classList.add('subnav-btn'); });
                </script>
                """,
                unsafe_allow_html=True,
            )

        # 서브 내비게이션 상태 업데이트 및 페이지 재실행
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

    # ------------------------------------------------------------------
    # 우측 메인 컨텐츠 영역
    # ------------------------------------------------------------------
    with main_col:
        # 공통 스타일 정의 (섹션 배너, 박스, 그리드, 데이터프레임 등)
        st.markdown(
            """
            <style>
            .sec-banner{display:flex;align-items:center;margin:8px 0 16px 0;}
            .sec-label{background:#5b9bd5;color:#fff;padding:10px 16px;font-weight:700;font-size:22px;letter-spacing:1px;border-radius:2px;line-height:1;}
            .sec-rule{flex:1;height:6px;background:#5b9bd5;margin-left:12px;border-radius:2px;}
            @media (max-width:480px){.sec-label{font-size:18px;padding:8px 12px}.sec-rule{height:5px}}

            .y-wrap{display:flex;flex-direction:column;gap:12px;}
            .y-box{margin-top:8px;padding:14px 16px;border:2px solid #666;border-radius:10px;background:#f2f6f9;}
            .y-title{display:flex;align-items:center;gap:10px;font-size:18px;font-weight:800;color:#000;}
            .y-sub{margin-top:4px;font-size:14px;font-weight:500;color:#000;}

            .y-grid{display:grid;grid-template-columns:180px 1fr;gap:8px 14px;margin-top:10px;}
            .y-grid .y-field,
            .y-grid .y-value{font-size:16px;line-height:1.5;color:#000;}
            .y-field{font-weight:700;}
            .y-value{font-weight:600;}

            .y-muted{color:#000 !important;font-size:inherit !important;font-weight:inherit !important;}

            div[data-testid="stDataFrame"] td, div[data-testid="stDataFrame"] th { padding: 10px 12px; }
            div[data-testid="stDataFrame"] tbody tr:nth-child(odd) { background: #eaf2fb22; }
            div[data-testid="stDataFrame"] tbody tr:nth-child(even){ background: #eaf2fb55; }
            div[data-testid="stDataFrame"] thead th { font-weight: 700; }
            </style>
            """,
            unsafe_allow_html=True,
        )
        if "sales_intervals" not in st.session_state:
            st.session_state.sales_intervals = [
                [0.0,    10.0],
                [10.0,   30.0],
                [30.0,   100.0],
                [100.0,  300.0],
                [300.0,  None],
            ]
        subnav = st.session_state.subnav

        # --------------------------------------------------------------
        # 1. 기술분야 개요
        # --------------------------------------------------------------
        if subnav == "overview":
            render_overview(
                sel_subcat_name=sel_subcat_name,
                sel_subcat_code=sel_subcat_code,
                applicant_corp_no=applicant_corp_no,
                applicant_name=applicant_name,
                selected_patent_title=selected_patent_title,
                send_sql=send_sql,
            )

        # --------------------------------------------------------------
        # 2. 재무부문
        # --------------------------------------------------------------
        elif subnav == "finance":
            render_finance(
                sel_subcat_name=sel_subcat_name,
                sel_subcat_code=sel_subcat_code,
                applicant_corp_no=applicant_corp_no,
                applicant_name=applicant_name,
                send_sql=send_sql,
            )

        # --------------------------------------------------------------
        # 3. 기술부문 
        # --------------------------------------------------------------
        elif subnav == "tech":
            render_tech(
                sel_subcat_name=sel_subcat_name,
                sel_subcat_code=sel_subcat_code,
                applicant_corp_no=applicant_corp_no,
                applicant_name=applicant_name,
                send_sql=send_sql,
            )

        # --------------------------------------------------------------
        # 4. R&D 부문
        # --------------------------------------------------------------
        elif subnav == "issue":
            render_issue(
                sel_subcat_name=sel_subcat_name,
                sel_subcat_code=sel_subcat_code,
                applicant_corp_no=applicant_corp_no,
                applicant_name=applicant_name,
                send_sql=send_sql,
            )

        
        if st.session_state.show_filter_modal == True:
            show_filter_modal()

        if st.session_state.show_interval_modal == True:
            show_interval_modal()