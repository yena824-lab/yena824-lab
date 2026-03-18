"""
main.app.py
=================

기술혁신정보서비스 Streamlit 메인 파일.

Streamlit 기반 BIGx/기술혁신정보서비스에서
'기술혁신정보' 상단 화면(대/중/소분류 선택 화면)을 렌더링하는 모듈.

기능 개요
--------

* URL query string의 nav 값을 보고 현재 페이지를 결정하고
* session_state를 초기화한 뒤
* 각 페이지별 render_* 함수를 호출하여 화면을 그린다.
"""

import streamlit as st
from main.views.home import render_home
from main.views.company import render_company
from main.views.company_detail import render_company_detail
from main.views.tech import render_tech
from main.views.tech_detail import render_tech_detail
from main.views.company_0_detail import render_company_0_detail
from main.views.tech_detail_0 import render_tech_detail_0


def main() -> None:
    """
    기술혁신정보서비스 Streamlit 앱 전체 실행을 담당하는 엔트리 함수.

    동작 흐름
    --------
    1. :func:`st.set_page_config` 를 호출하여 기본 페이지 설정
       (페이지 제목, 레이아웃, 사이드바 초기 상태 등)을 지정한다.
    2. HTML/CSS를 삽입하여 기본 Streamlit 사이드바 네비게이션과
       접기/펼치기 버튼을 숨긴다.
    3. ``st.session_state`` 에 아래 키들이 없으면 기본값으로 초기화한다.

       * ``page``: 현재 표시 중인 페이지 식별자
         (``"home"``, ``"company"``, ``"company_detail"``,
         ``"company_0_detail"``, ``"tech"``, ``"tech_detail"`` 중 하나).
       * ``company_patents_loaded``: 기업별 특허 로딩 여부 캐시(dict).
       * ``company_patents_df``: 기업별 특허 DataFrame 캐시(dict).

    4. URL query string 에서 ``nav`` 값을 읽어 페이지 전환 요청이 있는지 확인하고,
       값이 존재하면 해당 값에 따라 ``st.session_state["page"]`` 를 갱신한 뒤
       query string 에서 ``nav`` 를 제거하고 :func:`st.rerun` 을 호출한다.
    5. ``st.session_state["pending_nav"]`` 값이 존재하는 경우,
       이를 ``page`` 로 반영하고 키를 삭제한 뒤 :func:`st.rerun` 한다.
    6. 최종적으로 결정된 ``page`` 값에 따라 각 페이지별
       렌더링 함수(:func:`render_home`, :func:`render_company`,
       :func:`render_company_detail`, :func:`render_company_0_detail`,
       :func:`render_tech`, :func:`render_tech_detail`) 중 하나를 호출한다.
    7. ``page`` 값이 예상하지 못한 값인 경우 오류 메시지를 표시한다.

    세션 상태
    --------
    * ``st.session_state["page"]``: 현재 활성화된 페이지 문자열.
    * ``st.session_state["company_patents_loaded"]``:
      기업별 특허 로딩 여부를 저장하는 dict.
    * ``st.session_state["company_patents_df"]``:
      기업별 특허 DataFrame을 캐시하는 dict.
    * ``st.session_state["pending_nav"]``:
      다음 rerun 시 적용할 예정인 페이지 네비게이션 값(옵션).

    :return: 없음.
    :rtype: None
    """
    # 페이지 설정
    st.set_page_config(
        page_title="기술혁신정보서비스",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # 기본 사이드바/네비 숨기기
    st.markdown(
        """
        <style>
        /* 왼쪽 기본 페이지 네비(페이지 목록) 제거 */
        [data-testid="stSidebarNav"] { display: none; }

        /* 사이드바 접기/펼치기 화살표(<<, >>) 제거 */
        [data-testid="collapsedControl"] { display: none; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 세션 기본값
    if "page" not in st.session_state:
        st.session_state.page = "home"

    if "company_patents_loaded" not in st.session_state:
        st.session_state.company_patents_loaded = {}
    if "company_patents_df" not in st.session_state:
        st.session_state.company_patents_df = {}

    # --------------------------------------------------
    # 1) query string nav 처리
    # --------------------------------------------------
    qs = st.query_params
    nav = qs.get("nav", "")
    if isinstance(nav, list):
        nav = nav[0]

    if nav:
        if nav == "home":
            st.session_state.page = "home"
        elif nav == "company":
            st.session_state.page = "company"
        elif nav == "company_detail":
            st.session_state.page = "company_detail"
        elif nav == "company_0_detail":
            st.session_state.page = "company_0_detail"
        elif nav == "tech":
            st.session_state.page = "tech"
        elif nav == "tech_detail":
            st.session_state.page = "tech_detail"

        if "nav" in st.query_params:
            del st.query_params["nav"]

        st.rerun()

    # --------------------------------------------------
    # 2) 내부 pending_nav 처리
    # --------------------------------------------------
    if st.session_state.get("pending_nav"):
        st.session_state.page = st.session_state.pop("pending_nav")
        st.rerun()

    # --------------------------------------------------
    # 3) 최종 page 값에 따라 렌더링
    # --------------------------------------------------
    page = st.session_state.page

    if page == "home":
        render_home()
    elif page == "company":
        render_company()
    elif page == "company_detail":
        render_company_detail()
    elif page == "company_0_detail":
        render_company_0_detail()
    elif page == "tech":
        render_tech()
    elif page == "tech_detail":
        render_tech_detail()
    elif page == "tech_detail_0":
        render_tech_detail_0()
    else:
        st.error(f"Unknown page: {page}")


if __name__ == "__main__":
    main()
