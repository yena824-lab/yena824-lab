"""
views.home
==========

홈 화면 모듈.

Streamlit 기반 BIGx/기술혁신정보서비스의 초기 진입 화면(Home)을 렌더링하는 모듈.

기능 개요
--------

* 로컬 PNG 이미지를 Base64 문자열로 변환하여 CSS ``background-image`` 등에 사용할 수 있도록 한다.
* 상단 헤더(로고/서비스명)를 렌더링한다.
* 좌측: 기업혁신성장 보고서(BIGx) 진입 카드를 구성한다.
* 우측: 기술혁신정보 진입 카드를 구성한다.
"""

import base64
import streamlit as st

#: 기업 검색 화면 미리보기 이미지 경로.
IMG_COMPANY_PATH = "/home/kibo/peoples/wonjun/Front_end_1204_yena_new/search_company.png"

#: 기술 검색 화면 미리보기 이미지 경로.
IMG_TECH_PATH    = "/home/kibo/peoples/wonjun/Front_end_1204_yena_new/search_tech.png"


@st.cache_data(show_spinner=False)
def load_img_base64(path: str) -> str:
    """
    지정한 파일 경로의 이미지를 읽어 base64 문자열로 변환한다.

    CSS 의 ``background-image`` 나 HTML ``img`` 태그의
    ``src="data:image/png;base64,..."`` 형태로 사용하기 위한
    공통 유틸 함수이다.

    :param path: 로컬 이미지 파일 경로
    :type path: str
    :return: base64로 인코딩된 이미지 문자열
    :rtype: str
    """
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def render_home() -> None:
    """
    동작 흐름
    ---------

    1. :func:`load_img_base64` 를 사용해
       기업/기술 검색 화면 미리보기 이미지를 base64 문자열로 로드한다
    2. :func:`st.markdown` 에 HTML/CSS 를 주입하여 상단 헤더 영역을 렌더링한다.
       - 좌측: Kibo 로고
       - 중앙: 서비스 타이틀 (기술혁신정보서비스 V1.0)
       - 우측: 향후 확장을 위한 여유 슬롯
    3. 동일한 :func:`st.markdown` 블록에서 메인 카드 영역을 렌더링한다.
       - 좌측 카드: 기업혁신성장 보고서(BIGx) 진입용 카드
         * 배경: 기업 검색 화면 스크린샷
         * 라벨: "기업혁신성장 보고서 (BIGx)"
         * 클릭 시 ``?nav=company`` 로 이동하는 링크를 포함한다.
       - 우측 카드: 기술혁신정보 진입용 카드
         * 배경: 기술 검색 화면 스크린샷
         * 라벨: "기술혁신정보"
         * 클릭 시 ``?nav=tech`` 로 이동하는 링크를 포함한다.

    세션 상태
    ---------

    * 이 함수는 ``st.session_state`` 를 직접 읽거나 쓰지 않는다.
    * 홈 화면에서 다른 화면으로의 이동은
      카드 라벨에 설정된 링크(``?nav=company``, ``?nav=tech``)를 통해
      URL 쿼리스트링 ``nav`` 값이 변경되고,
      상위 레벨의 라우팅 로직에서 이 값을 읽어
      :func:`render_company`, :func:`render_tech` 등을 호출하는 방식으로 처리된다.

    :return: 없음
    :rtype: None
    """
    logo_url = "https://i.namu.wiki/i/s33tjbbi4Ysn4AJwyqjGjT2zg9KeCuw_E0OryvcxMmM9BXrrUAEU3R3uJEz1Wzd6H-ki_JrTmvcXkegb3fcZdw.svg"

    img_company = load_img_base64(IMG_COMPANY_PATH)
    img_tech    = load_img_base64(IMG_TECH_PATH)

    st.markdown(
        f"""
    <style>
    /* ===== 헤더: 좌 로고 / 가운데 제목 / 우 더미 ===== */
    .topbar {{
      display: grid;
      grid-template-columns: 180px 1fr 180px;
      align-items: center;
      padding: 20px 30px;
      margin: 0 0 8px 0;
      border: 1.5px solid #000;           /* 검정 테두리 */
      border-radius: 10px;
      background: #fff;
    }}
    .brand-left {{ display:flex; align-items:center; gap:12px; }}
    .brand-left img {{ height:50px; width:auto; display:block; }}
    .brand-title {{ text-align:center; font-size:36px; font-weight:800; letter-spacing:.5px; }}
    .right-slot {{}}
    @media (max-width:600px){{
      .topbar {{ grid-template-columns:140px 1fr 140px; }}
      .brand-title {{ font-size:18px; }}
    }}

    /* ===== 카드 ===== */
    .cards {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
      margin-top: 12px;
    }}
    @media (max-width:900px){{ .cards {{ grid-template-columns:1fr; }} }}

    .card {{
      position: relative;
      height: 680px;
      border-radius: 18px;
      border: 1.5px solid #000;           /* 검정 테두리 */
      overflow: hidden;
    }}
    .card:hover {{
      transform: none !important;
      box-shadow: none !important;
      border-color: #000 !important;
    }}
    .card::before {{
      content: "";
      position: absolute;
      inset: 0;
      background-size: cover;
      background-position: center;
      background-repeat: no-repeat;
      z-index: 0;
    }}
    .card.company::before {{ background-image: url("data:image/png;base64,{img_company}"); }}
    .card.tech::before    {{ background-image: url("data:image/png;base64,{img_tech}"); }}

    /* ===== 버튼 오버레이(라벨) ===== */
    .label {{
      position: absolute;
      bottom: 18px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(255,255,255,.85);
      padding: 25px 35px;
      border-radius: 10px;
      font-weight: 900;
      font-size: 24px;
      text-align: center;
      z-index: 1;
      display: block;
      cursor: pointer;
      border: 2px solid #000;             /* 검정 테두리 */
      box-shadow: 0 2px 6px rgba(0,0,0,.08);
      transition: background .18s, box-shadow .18s, transform .18s, border-color .18s;
    }}
    .label:hover {{
      background: rgba(255,255,255,.98);
      box-shadow: 0 10px 24px rgba(0,0,0,.14);
      transform: translateX(-50%) translateY(-2px);
    }}
    .label:focus-visible {{ outline: 3px solid #1a4fff; outline-offset: 2px; }}
    .label:link, .label:visited, .label:hover, .label:active {{
      color: #000 !important;
      text-decoration: none !important;
    }}
    </style>

    <!-- ===== 헤더 ===== -->
    <div class="topbar">
      <div class="brand-left">
        <img src="{logo_url}" alt="Kibo 로고" />
      </div>
      <div class="brand-title">기술혁신정보서비스 <b>V1.0</b></div>
      <div class="right-slot"></div>
    </div>

    <!-- ===== 카드 ===== -->
    <div class="cards">
      <div class="card company">
        <a class="label" href="?nav=company" target="_self">
          기업 검색 및 분석
        </a>
      </div>
      <div class="card tech">
        <a class="label" href="?nav=tech" target="_self">
          기술 검색 및 분석
        </a>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
