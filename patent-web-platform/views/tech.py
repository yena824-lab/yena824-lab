"""
views.tech_search
=================

기술혁신정보(Tech Search) 메인 화면 모듈.

Streamlit 기반 BIGx/기술혁신정보서비스에서
'기술혁신정보' 상단 화면(대/중/소분류 선택 화면)을 렌더링하는 모듈.

기능 개요
--------

* 공통 내비게이션 영역을 렌더링한다.
* 대분류/중분류/소분류 계층 구조 선택 UI를 제공한다.
* 소분류 선택 시 세션 상태를 갱신하고 상세 페이지로 이동하기 위한
  플래그를 설정한다.
"""

import math
import pandas as pd
import streamlit as st

from core.db import send_sql
from main.sql import sql_tech as Q  


def go(page: str) -> None:
    """
    페이지 이동을 위한 세션 상태 변경 함수이다.

    ``st.session_state["page"]`` 값만 변경하고, 실제 화면 전환 로직은
    상위 레벨에서 이 값을 참조해 처리한다.

    :param page: 이동하려는 페이지 키 (예: ``"home"``, ``"tech_detail"`` 등).
    :type page: str
    :return: 없음
    :rtype: None
    """
    st.session_state.page = page


def set_nav_buttons() -> None:
    """
    상단 좌우에 [뒤로가기] / [Home] 버튼을 렌더링하는 함수이다.

    왼쪽에는 이전 페이지로 이동하는 화살표 버튼을,
    오른쪽에는 홈으로 이동하는 버튼을 배치한다.
    두 버튼 모두 :func:`go` 함수를 통해 페이지 전환을 수행한다.

    버튼 배치는 다음과 같이 구성된다.

    * 왼쪽 컬럼: ``⬅︎`` (뒤로가기)
    * 오른쪽 컬럼: ``Home`` (홈으로 이동)

    :return: 없음
    :rtype: None
    """
    left, _, right = st.columns([1, 6, 1])
    with left:
        st.button("⬅︎", on_click=lambda: go("home"), key="btn_back4")
    with right:
        st.button("Home", on_click=lambda: go("home"), key="btn_home4")


def _on_small_change() -> None:
    """
    소분류 선택 시 선택 정보를 세션 상태에 저장하고
    기술 상세 페이지로 이동하기 위한 플래그를 설정하는 콜백 함수.

    :return: 없음
    :rtype: None
    """
    val = st.session_state.get("ui_selected_small", "")
    if not val:
        return
    code_part, name_part = val.split(" - ", 1)
    code = code_part.strip()
    sub_name = name_part.strip()
    st.session_state.tech_selected = {"code": code, "name": sub_name, "label": val}
    st.session_state.pending_nav = "tech_detail"
    st.rerun()


def render_tech() -> None:
    """
    기술혁신정보 메인 화면 전체를 렌더링하는 함수.


    동작 흐름
    ---------

    1. :func:`set_nav_buttons` 를 호출하여 상단 내비게이션 버튼과
       페이지 타이틀("기술혁신정보")을 렌더링한다.
    2. :func:`Q.get_tech_categories` SQL을 통해 기술 분류(대/중/소) 전체 목록을 조회한다.
    3. ``big_code``, ``mid_code``, ``small_code`` 가 모두 존재하는 행만 남기도록
       :meth:`pandas.DataFrame.dropna` 로 전처리한다.
    4. 세 개의 컬럼 영역을 생성하여, 각각 대분류/중분류/소분류 선택 UI를 배치한다.
       - 대분류 선택 시 ``"코드 - 명칭"`` 형식의 문자열이
         ``st.session_state["ui_selected_big"]`` 에 저장된다.
       - 중분류 선택 시 ``"코드 - 명칭"`` 형식의 문자열이
         ``st.session_state["ui_selected_mid"]`` 에 저장된다.
       - 소분류 선택 시 ``"코드 - 명칭"`` 형식의 문자열이
         ``st.session_state["ui_selected_small"]`` 에 저장되며,
         값이 변경될 때 :func:`_on_small_change` 콜백이 호출된다.
    5. :func:`_on_small_change` 콜백에서는 선택된 소분류 정보를
       ``st.session_state["tech_selected"]`` 에 저장하고,
       ``st.session_state["pending_nav"] = "tech_detail"`` 로 설정하여
       상위 레벨에서 기술 상세 페이지로 전환할 수 있도록 플래그를 세팅한다.

    세션 상태
    ---------

    * ``st.session_state["page"]``  
      :func:`go` 함수에서 변경하는 현재 페이지 키.
    * ``st.session_state["ui_selected_big"]``  
      대분류 선택 값 (예: ``"A01 - 농업"``, 선택하지 않은 경우 빈 문자열).
    * ``st.session_state["ui_selected_mid"]``  
      중분류 선택 값 (예: ``"A01B - 어떤 중분류"``).
    * ``st.session_state["ui_selected_small"]``  
      소분류 선택 값. 변경 시 :func:`_on_small_change` 가 호출된다.
    * ``st.session_state["tech_selected"]``  
      선택된 소분류의 정보 딕셔너리. 예:  
      ``{"code": "A01B", "name": "소분류명", "label": "A01B - 소분류명"}``
    * ``st.session_state["pending_nav"]``  
      소분류 선택 후 기술 상세 페이지로 이동해야 할 때 ``"tech_detail"`` 로 설정된다.
    """
    set_nav_buttons()
    st.title("기술혁신정보") 

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

    # 기술 분류(대/중/소) 전체 목록 조회
    df = send_sql(
        Q.get_tech_categories(),  
    )

    # 코드가 비어 있는 행 제거
    df = df.dropna(subset=["big_code", "mid_code", "small_code"])
    c1, c2, c3 = st.columns([1, 1, 1])

    # ① 대분류 선택
    with c1:
        big_categories = df[["big_code", "big_name"]].drop_duplicates().sort_values("big_code")
        selected_big = st.selectbox(
            "대분류 선택",
            options=[""] + [f"{r.big_code} - {r.big_name}" for r in big_categories.itertuples()],
            index=0,
            key="ui_selected_big"
        )

    # ② 중분류 선택
    with c2:
        big_label = st.session_state.get("ui_selected_big", "")
        if big_label:
            selected_big_code = big_label.split(" - ", 1)[0].strip()
            mid_df = df[df["big_code"] == selected_big_code]
        else:
            mid_df = df

        mid_categories = mid_df[["mid_code", "mid_name"]].drop_duplicates().sort_values("mid_code")
        selected_mid = st.selectbox(
            "중분류 선택",
            options=[""] + [f"{r.mid_code} - {r.mid_name}" for r in mid_categories.itertuples()],
            index=0,
            key="ui_selected_mid"
        )

    # ③ 소분류 선택
    with c3:
        mid_label = st.session_state.get("ui_selected_mid", "")
        if mid_label:
            selected_mid_code = mid_label.split(" - ", 1)[0].strip()
            small_df = df[df["mid_code"] == selected_mid_code]
        else:
            small_df = df

        small_categories = (
            small_df[["small_code", "small_name"]]
            .drop_duplicates()
            .sort_values("small_code")
        )

        st.selectbox(
            "소분류 선택",
            options=[""] + [f"{r.small_code} - {r.small_name}" for r in small_categories.itertuples()],
            index=0,
            key="ui_selected_small",
            on_change=_on_small_change,
        )
