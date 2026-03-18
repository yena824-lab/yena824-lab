"""
button.tech_issue
=================

R&D부문(이슈/전문기관/워드클라우드) 화면 모듈.

Streamlit 기반 BIGx 보고서에서 **R&D부문 섹션**을
렌더링하는 모듈이다.
텍스트 전처리, TF-IDF 기반 워드클라우드 생성, R&D 전문기관 표 생성 유틸과
메인 렌더링 함수 :func:`render_issue` 를 제공한다.

기능 개요
--------

* 한글/영문 불용어 및 단위 단어 목록을 정의한다.
* konlpy(Okt)를 이용하여 한글 명사를 추출하고 TF-IDF 점수를 계산한다.
* ``wordcloud`` 패키지를 이용해 워드클라우드 이미지를 생성한다.
* 소분류별 R&D 키워드(워드클라우드)와
  R&D 전문기관 현황 테이블을 함께 렌더링한다.
* :func:`render_issue` 를 통해
  R&D부문 전체 레이아웃과 데이터를 연동하여 화면을 구성한다.
"""


import os
import sys
import re
import subprocess
from collections import Counter
from io import BytesIO  

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from main.sql import sql_tech_detail as Q
from typing import List, Dict
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from functools import lru_cache
from typing import Literal, Set

import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime, date


FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"  # 환경에 맞게 수정
TOP_N = 30  # 워드클라우드 상위 단어 수


YEAR_WEIGHT_POINTS = {  
    2022: -20,
    2023: 0,
    2024: 20,
}

def year_to_factor(y: int) -> float:
    # -20% => 0.8, 0% => 1.0, +20% => 1.2
    pts = YEAR_WEIGHT_POINTS.get(int(y), 0)
    return 1.0 + (pts / 100.0)


def strip_josa(tok: str) -> str:
    tok = tok.strip()
    _JOSA_SUFFIX_RE = re.compile(
        r"(으로서|으로써|으로부터|으로|에서|에게서|에게|까지|부터|보다|처럼|같이|만|도|까지|조차|마저|뿐|이나|나|과|와|을|를|이|가|은|는|의|에)$"
    )
    # 한 번만 제거(과도 제거 방지). 필요하면 while로 여러 번도 가능.
    return _JOSA_SUFFIX_RE.sub("", tok)


# 3) 토큰 필터 함수 (불용어 / 숫자형 제거)
def is_good(tok: str) -> bool:
    if len(tok) <= 1:
        return False
    if tok in STOPWORDS:
        return False
    if _looks_numeric_like(tok):
        return False
    return True


def _korean_stopword_tokens() -> list[str]:
    """
    한글 특허 명세서용 불용어 토큰 리스트를 생성한다.
    """
    return """
    그리고 그러나 그러나도 그러나서 그리고서 그리고는 그러나마는 또한 또는 및 또 또는의 등의 등은 등과 등으로 등에서 등과같은 등등
    이 그 저 본 해당 당해 이러한 이러한점 이런 이런점 그런 그런점 저런 모든 각 각종 각종의 여러 여러가지 일부 일부의 다른
    것 것들 것이 것을 것은 에서 으로 으로서 으로써 으로의 으로부터 에게 에게서 에 대하여 대해 대한 대로 에 따른 따른 바 바와같이
    수 수준 정도 경우 경우에 경우로 예 예를 예를들어 예를들면 등등처럼 처럼 같이 보다 보다도 또한또는 또는또한 또한그리고
    및 또는및 뿐 아니라 뿐만 아니라 에 있어서 있어서 의 한 하여 하여금 하여도 하여야 하여야만 하여서 하기에 하기 위한 위한
    및또는 과 와 및과 또는과 그리고과
    발명 발명이 본발명 본발명의 발명은 발명에 발명을 해결 과제 문제 문제점 목적 제공 구성 구현 실시 실시예 실시형태 실시방법
    방법 장치 시스템 모듈 유닛 기구 기기 요소 공정 공정의 공정에서 장치의 장치에서 장치로 장치가 장치를 장치에 구성요소
    사용 이용 통해 가능 가능하게 가능하도록 가능함 적어도 적어도하나 최소 최대 제1 제2 제3 일 실시 일실시 예시 예컨대
    도면 도 표시 도시 설명 설명서 명세 명세서 청구 청구항 권리 범위 참고 참조 단계 단계로 단계에서 단계의 단계들 단계간
    형태 구조 부분 일측 타측 내부 외부 전원 연결 결합 분리 제어 제어부 처리 처리부 데이터 신호 정보 입력 출력 메모리 저장
    이상 이하 이전 이후 전 후 전후 각각 전체 일부 반복 선택 선택적으로 예컨대 예시적으로 대략 약 대략적으로 약간
    사용됨 사용되는 사용한다 포함 포함함 포함하는 포함되어 포함될 수 있다 포함될수있다 로부터 언급 때문 위해 관련 향상
    서로 미리 따라서 여기 달라 아래 로부터
    """.split()

def _english_stopword_tokens() -> list[str]:
    """
    영문 특허 텍스트용 불용어 토큰 리스트를 생성하는 내부 함수이다.

    :return: 영문 불용어 단어 리스트이다.
    :rtype: list[str]
    """
    return """
    the a an and or of to in for from with by on at as is are was were be been being this that these those such
    method methods apparatus device system module unit means step steps process processes provide provides provided providing
    include includes including comprise comprises comprising consist consists consisting have has having can may will should could
    according wherein thereby example examples
    """.split()


def _unit_stopword_tokens() -> list[str]:
    """
    단위 관련 숫자/단위 불용어 토큰 리스트를 생성하는 내부 함수이다.

    :return: 단위 관련 불용어 단어 리스트이다.
    :rtype: list[str]
    """
    return [
        "mm", "nm", "um", "cm", "m",
        "kg", "g", "mg", "l", "ml",
        "s", "sec", "hr", "hrs", "min", "mins",
        "℃", "°c", "kwh", "kw", "w", "v", "ma",
        "시간", "분", "초", "온도", "압력", "농도", "용량",
    ]


@lru_cache(maxsize=1)
def get_stopwords(
    kind: Literal["all", "ko", "en", "unit"] = "all",
) -> Set[str]:
    """
    R&D 워드클라우드/텍스트 분석에 사용하는 불용어 집합을 반환한다.

    :param kind: 반환할 불용어 종류이다.
                 ``"ko"``(한글), ``"en"``(영문), ``"unit"``(단위),
                 ``"all"``(전체 합집합) 중 하나이다.
    :type kind: Literal["all", "ko", "en", "unit"]
    :return: 요청된 종류의 불용어 집합이다.
    :rtype: set[str]
    """
    korean = set(_korean_stopword_tokens())
    english = set(_english_stopword_tokens())
    unit = set(_unit_stopword_tokens())

    if kind == "ko":
        return korean
    if kind == "en":
        return english
    if kind == "unit":
        return unit
    # kind == "all"
    return korean | english | unit


# 기존 코드와 호환을 위해 전역 STOPWORDS 하나 유지
STOPWORDS: Set[str] = get_stopwords("all")


def _looks_numeric_like(tok: str) -> bool:
    """
    토큰이 숫자 또는 숫자+알파벳 형태인지 여부를 판단한다.

    예시: ``"123"``, ``"10a"``, ``"a10"`` 등.

    :param tok: 검사할 토큰 문자열이다.
    :type tok: str
    :return: 숫자/숫자혼합 토큰이면 ``True`` 이고, 아니면 ``False`` 이다.
    :rtype: bool
    """
    t = tok.lower()
    if t.isdigit():
        return True
    if re.fullmatch(r"\d+[a-z]+", t):
        return True
    if re.fullmatch(r"[a-z]+\d+", t):
        return True
    return False


def clean_text(text: str) -> str:
    """
    텍스트에서 한글과 공백만 남기고 나머지 문자를 제거한 뒤,
    연속 공백을 하나로 줄이는 간단 전처리를 수행한다.

    :param text: 원본 텍스트이다.
    :type text: str
    :return: 전처리된 텍스트이다.
    :rtype: str
    """
    text = re.sub(r"[^ㄱ-ㅎ가-힣\s]", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()


# ===== Okt 명사 토큰화 =====
# konlpy(Okt)를 이용한 한국어 명사 추출. 모듈이 없을 경우 안내 메시지 출력.
try:
    from konlpy.tag import Okt
    _OKT_AVAILABLE = True
    okt = Okt()
except ImportError:
    _OKT_AVAILABLE = False
    okt = None


def korean_wc_analyzer(text: str) -> List[str]:
    """
    워드클라우드용 한글 분석기.

    절차는 다음과 같다.

    1. 원문에서 명사 전체 시퀀스를 추출한다. (``Okt.nouns``)
    2. 같은 단어가 연속되는 경우 제거한다. (예: ``"각도 각도 포신"`` → ``"각도 포신"``)
    3. 정제된 명사 시퀀스를 기준으로 1~4그램을 생성한다.
    4. 각 n-gram 안에 불용어/숫자형 토큰이 하나라도 포함되면 해당 n-gram은 버린다.

    이 방식은 불용어 제거 때문에 ``각도 포신`` 과 같이
    인위적인 n-gram이 생성되는 문제를 줄이기 위한 것이다.

    :param text: 원본 텍스트.
    :type text: str
    :return: 워드클라우드용 토큰(1~4그램) 리스트.
    :rtype: list[str]
    """
    if not _OKT_AVAILABLE:
        raise SystemExit("konlpy가 필요합니다. `pip install konlpy JPype1` 후 다시 실행하세요.")
    if not text:
        return []

    txt = clean_text(text)
    if not txt:
        return []

    # 1) 불용어 제거 전, 원본 명사 시퀀스
    raw_nouns = okt.nouns(txt)
    raw_nouns = [t.strip() for t in raw_nouns if t and t.strip()]

    raw_nouns = [strip_josa(t) for t in raw_nouns]
    raw_nouns = [t for t in raw_nouns if t]

    if not raw_nouns:
        return []

    # 2) 연속 중복 제거
    nouns: List[str] = []
    prev = None
    for t in raw_nouns:
        if t == prev:
            continue
        nouns.append(t)
        prev = t

    terms: List[str] = []

    # 4) unigram: 좋은 토큰만
    for t in nouns:
        if is_good(t):
            terms.append(t)

    # 5) 2~4-gram: 원본 명사 시퀀스에서 연속된 구간만 사용
    max_n = 4
    L = len(nouns)
    for n in range(2, max_n + 1):
        if L < n:
            break
        for i in range(L - n + 1):
            window = nouns[i : i + n]

            # window 안에 STOPWORDS / 숫자형이 하나라도 있으면 그 n-gram은 버림
            if not all(is_good(w) for w in window):
                continue

            phrase = " ".join(window)
            terms.append(phrase)

    return terms


def get_ngram_weight(term: str) -> float:
    """
    n-gram 길이에 따라 가중치를 부여한다.

    예를 들어 ``"잔디"``, ``"잔디 상태"``, ``"잔디 상태 이상"`` 과 같이
    공백 개수 + 1 = n-gram 길이가 된다.

    가중치는 다음과 같이 설정한다.

    * 1-gram: 2.0
    * 2-gram: 3.0
    * 3-gram: 3.0
    * 4-gram: 2.0
    * 그 외: 1.0

    :param term: n-gram 문자열.
    :type term: str
    :return: 가중치 값.
    :rtype: float
    """
    n = term.count(" ") + 1

    if n == 1:
        return 2.0   # unigram 가중치
    elif n == 2:
        return 3.0   # bigram 가중치
    elif n == 3:
        return 3.0   # trigram 가중치
    elif n == 4:
        return 2.0   # quadgram 가중치
    else:
        return 1.0   # 혹시 모를 예외


def make_vectorizer(min_df: int = 1, max_df: float = 0.95) -> TfidfVectorizer:
    """
    TF-IDF 벡터라이저를 생성한다.

    - tokenizer: :func:`tokenize_korean_nouns`
    - ngram_range: (1, 2) → 1~2그램
    - min_df / max_df: 등장 문서 수 기준 필터링

    :param min_df: 용어 최소 등장 문서 수이다.
    :type min_df: int
    :param max_df: 용어 최대 등장 문서 비율 또는 개수이다.
    :type max_df: float
    :return: 설정된 :class:`TfidfVectorizer` 인스턴스이다.
    :rtype: sklearn.feature_extraction.text.TfidfVectorizer
    """
    return TfidfVectorizer(
        tokenizer=korean_wc_analyzer,
        lowercase=False,
        ngram_range=(1, 1),
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=True,
        norm=None,
        dtype=float
    )


def doc_has_valid_tokens(text: str) -> bool:
    """
    주어진 텍스트에서 유효한 명사 토큰이 존재하는지 여부를 판단한다.

    :param text: 검사할 텍스트이다.
    :type text: str
    :return: 유효한 토큰이 하나 이상 존재하면 ``True`` 이다.
    :rtype: bool
    """
    return len(korean_wc_analyzer(text)) > 0



def compute_wc_freqs_from_texts(texts: List[str], years: List[int]) -> dict[str, float]:
    """
    명세서(해결과제) 텍스트 리스트를 TF-IDF 기반 워드클라우드용
    빈도 딕셔너리로 변환한다.

    - quadgram(4-gram)이 하나라도 나올 수 있는지 먼저 확인
    - 문서 수에 따라 min_df / max_df 다르게 설정
    - 각 term 의 TF-IDF 합계 × n-gram 가중치(get_ngram_weight)를 스코어로 사용
    """

    # 1) 빈 문자열 제거 + years도 같이 정렬
    sub_corpus: list[str] = []
    sub_years: list[int] = []
    for t, y in zip(texts, years):
        t = (t or "").strip()
        if not t:
            continue
        sub_corpus.append(t)
        sub_years.append(int(y) if y is not None else 0)

    if not sub_corpus:
        return {}

    # 2) 유효 토큰 문서가 하나라도 있는지
    if not any(doc_has_valid_tokens(t) for t in sub_corpus):
        return {}

    # 3) 벡터라이저 설정
    n_docs = len(sub_corpus)
    if n_docs <= 3:
        vectorizer = make_vectorizer(min_df=1, max_df=1.0)
    else:
        vectorizer = make_vectorizer(min_df=1, max_df=0.95)

    # 4) TF-IDF 벡터화
    X_sub = vectorizer.fit_transform(sub_corpus)  # (n_docs, n_terms)
    terms = vectorizer.get_feature_names_out()
    if len(terms) == 0:
        return {}

    # 5) 연도별 문서 가중치 벡터 만들기
    doc_w = np.array([year_to_factor(y) for y in sub_years], dtype=float)  # (n_docs,)

    # 6) term별 가중합: sum_i (doc_w[i] * X[i, term])
    weighted_scores = np.asarray(
        X_sub.multiply(doc_w.reshape(-1, 1)).sum(axis=0)
    ).ravel()

    # 7) n-gram 가중치 적용
    freqs = {
        term: float(score * get_ngram_weight(term))
        for term, score in zip(terms, weighted_scores)
        if score > 0
    }

    return freqs


def fetch_wc_freqs_for_subcat(sel_subcat_name: str, sel_subcat_code: str, send_sql) -> dict:
    """
    소분류 코드 하나에 대해 워드클라우드용 빈도 딕셔너리를 계산한다.

    절차는 다음과 같다.

    1. :func:`Q.q_wordcloud` 쿼리를 실행한다.
    2. 명세서(해결과제) 텍스트 컬럼을 추출한다.
    3. :func:`compute_wc_freqs_from_texts` 로 TF-IDF 기반 빈도 딕셔너리를 계산한다.

    :param sel_subcat_name: 소분류명(로그/디버깅용)이다.
    :type sel_subcat_name: str
    :param sel_subcat_code: 소분류 코드이다.
    :type sel_subcat_code: str
    :param send_sql: DB 조회 함수이다.
    :type send_sql: Callable[[str, dict], pandas.DataFrame]
    :return: 워드클라우드용 ``{term: score}`` 딕셔너리이다. 데이터가 없으면 빈 dict이다.
    :rtype: dict
    """
    params_rnd = {"sel_subcat_code": sel_subcat_code}
    df_freq = send_sql(Q.q_wordcloud(), params=params_rnd)

    if df_freq.empty:
        return {}

    texts = df_freq["명세서"].astype(str).tolist()

    # 등록일자에서 연도 추출 (datetime/string 모두 대응)
    years = pd.to_datetime(df_freq["등록일자"], errors="coerce").dt.year.fillna(0).astype(int).tolist()

    freqs = compute_wc_freqs_from_texts(texts, years) 
    return freqs


# ---------------------------------------------------
# wordcloud 관련 유틸
# ---------------------------------------------------
def import_WordCloud():
    """
    ``wordcloud`` 패키지를 임포트하고, 파일명 충돌을 검사한 뒤
    필요 시 설치까지 수행한다.

    - 프로젝트 폴더에 ``wordcloud.py`` / ``wordcloud`` 디렉터리가 있으면
      공식 패키지와 충돌하므로 경고 후 중단한다.
    - 패키지가 없으면 ``pip`` 로 설치를 시도한다.

    :return: :class:`wordcloud.WordCloud` 클래스이다.
    :rtype: type
    """
    import importlib.util

    spec = importlib.util.find_spec("wordcloud")
    if spec is not None and spec.origin:
        base = os.path.basename(spec.origin)
        # 로컬 파일/디렉터리가 공식 패키지를 가리는 경우
        if base == "wordcloud.py" or os.path.isdir(spec.origin):
            st.error(
                "프로젝트 폴더에 'wordcloud.py' 또는 'wordcloud' 디렉터리가 있어 "
                "공식 패키지를 가리고 있습니다. 파일/폴더명을 변경하고 다시 실행하세요.\n\n"
                f"(충돌 경로: {spec.origin})"
            )
            st.stop()
        try:
            from wordcloud import WordCloud
            return WordCloud
        except Exception as e:
            st.warning(f"wordcloud 패키지 임포트 실패: {e}. 재설치를 시도합니다…")

    # 패키지 설치 시도
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "wordcloud==1.9.3", "pillow", "--quiet"]
        )
        from wordcloud import WordCloud
        return WordCloud
    except Exception as e:
        st.error(
            "wordcloud 패키지를 불러오지 못했습니다.\n"
            "- 패키지 설치 여부(pip/requirements.txt) 확인\n"
            "- 파일명 충돌(wordcloud.py/폴더) 제거 후 재실행\n\n"
            f"원인: {e}"
        )
        st.stop()


# WordCloud 클래스 로드 (모듈 로딩 시 한 번만 실행)
WordCloud = import_WordCloud()


def generate_wordcloud(
    freqs: dict,
    font_path: str,
    top_n: int = 30,
    w: int = 1600,
    h: int = 1000,
) -> Image.Image:
    """
    주어진 빈도 딕셔너리로 WordCloud 이미지를 생성한다.

    :param freqs: 단어 → 가중치(TF-IDF 스코어 등) 딕셔너리이다.
    :type freqs: dict
    :param font_path: 사용할 폰트 파일 경로이다.
    :type font_path: str
    :param top_n: 상위 몇 개 단어를 사용할지 지정한다.
    :type top_n: int
    :param w: 워드클라우드 캔버스 가로 크기(px)이다.
    :type w: int
    :param h: 워드클라우드 캔버스 세로 크기(px)이다.
    :type h: int
    :return: 생성된 워드클라우드 PIL 이미지이다.
    :rtype: PIL.Image.Image
    """
    # 상위 top_n 단어만 사용
    top_freqs = dict(Counter(freqs).most_common(top_n))
    wc = WordCloud(
        width=800,
        height=600,
        background_color='white',
        font_path=FONT_PATH if os.path.isfile(FONT_PATH) else None,
        collocations=False,
        max_words=30,
        prefer_horizontal=0.9,   # 가로 90%
        random_state=1,
        scale=3,                 # 텍스트 해상도 향상
        relative_scaling=0.6,
    ).generate_from_frequencies(top_freqs)
    return wc.to_image()


# ---------------------------------------------------
# 비중(%) 계산 유틸 
# ---------------------------------------------------
def compute_percentage_shares(counts: pd.Series, decimals: int = 1) -> pd.Series:
    """
    비중(%)를 소수점 ``decimals`` 자리까지 반올림하면서
    총합이 정확히 100.0이 되도록 조정한다.

    (최다 특허 보유 기관 등에서 사용한다.)

    :param counts: 개수(또는 합계) 시리즈이다.
    :type counts: pandas.Series
    :param decimals: 소수점 자리수이다.
    :type decimals: int
    :return: 각 행에 대한 비중(%) 시리즈이다.
    :rtype: pandas.Series
    """
    counts = counts.astype("Float64").fillna(0.0)
    total = float(counts.sum())

    if total <= 0:
        return pd.Series([0.0] * len(counts), index=counts.index, dtype="Float64")

    factor = 10 ** decimals
    shares = counts / total * 100.0

    # 스케일링 후 반올림 보정
    scaled = (shares * factor).to_numpy()
    base = np.floor(scaled)
    diff = int(round(100 * factor - base.sum()))
    frac = scaled - base

    if diff > 0:
        order = np.argsort(-frac)
        base[order[:diff]] += 1
    elif diff < 0:
        order = np.argsort(frac)
        base[order[:(-diff)]] -= 1

    return pd.Series(base / factor, index=counts.index, dtype="Float64")

def build_wc_issue_table(
    freqs: dict[str, float],
    top_n: int = 10,
    decimals: int = 0,
    others_label: str = "기타",
) -> pd.DataFrame:
    """
    워드클라우드 상위 단어들의 비중(%) 테이블을 생성한다.

    - 상위 top_n은 개별로 표시
    - 나머지는 합산해서 '기타' 1개로 표시
    - 마지막 행에 '합계' 100 표시
    """
    if not freqs:
        return pd.DataFrame(columns=["No", "R&D 이슈", "비중(%)"]).astype(
            {"No": "Int64", "R&D 이슈": "string", "비중(%)": "Float64"}
        )

    # 전체 정렬 (freqs는 보통 top30이 들어옴)
    items_all = sorted(freqs.items(), key=lambda x: x[1], reverse=True)

    head = items_all[:top_n]
    tail = items_all[top_n:]

    terms = [k for k, _ in head]
    vals = [float(v) for _, v in head]

    # 나머지(예: 20개)는 '기타'로 합산
    others_score = float(sum(v for _, v in tail)) if tail else 0.0
    if tail and others_score > 0:
        terms.append(others_label)
        vals.append(others_score)

    scores = pd.Series(vals, index=terms, dtype="Float64")

    # 비중(%) 계산 (합계 100으로 보정)
    shares = compute_percentage_shares(scores, decimals=decimals)

    df = pd.DataFrame(
        {
            "No": pd.Series(range(1, len(terms) + 1), dtype="Int64"),
            "R&D 이슈": pd.Series(terms, dtype="string"),
            "비중(%)": shares.to_numpy(),
        }
    )

    # 합계 행 추가
    total_row = pd.DataFrame(
        {
            "No": pd.Series([pd.NA], dtype="Int64"),
            "R&D 이슈": pd.Series(["합계"], dtype="string"),
            "비중(%)": pd.Series([float(shares.sum())], dtype="Float64"),
        }
    )

    df = pd.concat([df, total_row], ignore_index=True)
    return df.astype({"No": "Int64", "R&D 이슈": "string", "비중(%)": "Float64"})



# ---------------------------------------------------
# R&D 전문기관 테이블 함수
# ---------------------------------------------------
def build_rnd_table(df_raw: pd.DataFrame, title: str) -> pd.DataFrame:
    """
    R&D 전문기관 표를 생성하는 공통 함수이다.

    (정부출연연구소, 산학협력단 모두 이 함수를 사용한다.)

    :param df_raw: 원본 전문기관 데이터 DataFrame이다.
    :type df_raw: pandas.DataFrame
    :param title: 사용처 설명용 타이틀이다 (예: ``"정부출연연구소"``).
    :type title: str
    :return: ``[순번, 기관명, 전체특허보유수, 특허보유수, 비중(%)]`` 컬럼을 갖는 표 DataFrame이다.
    :rtype: pandas.DataFrame
    """
    if df_raw.empty:
        st.info(f"{title} 데이터가 없습니다.")
        # 빈 DF 템플릿 반환 (타입까지 지정)
        return pd.DataFrame(
            columns=["순번", "기관명", "전체특허보유수", "특허보유수", "비중(%)"]
        ).astype(
            {
                "순번": "Int64",
                "기관명": "string",
                "전체특허보유수": "Int64",
                "특허보유수": "Int64",
                "비중(%)": "Float64",
            }
        )

    df = df_raw.rename(columns={"기업명": "기관명"}).copy()
    df["특허보유수"] = pd.to_numeric(df["특허보유수"], errors="coerce").astype("Int64")
    df["전체특허보유수"] = pd.to_numeric(df["전체특허보유수"], errors="coerce").astype(
        "Int64"
    )
    df["기관명"] = df["기관명"].astype("string")

    # 기관명/전체특허보유수 단위로 특허보유수 합계 계산
    g = (
        df.groupby(["기관명", "전체특허보유수"], as_index=False)["특허보유수"]
        .sum()
        .sort_values("특허보유수", ascending=False)
        .reset_index(drop=True)
    )

    # 순번 부여
    g.insert(0, "순번", pd.Series(range(1, len(g) + 1), dtype="Int64"))

    # 비중(%) 계산
    counts = g["특허보유수"].astype("Float64").fillna(0.0)
    g["비중(%)"] = compute_percentage_shares(counts, decimals=1)

    # 타입 정리
    g = g.astype(
        {
            "순번": "Int64",
            "기관명": "string",
            "전체특허보유수": "Int64",
            "특허보유수": "Int64",
            "비중(%)": "Float64",
        }
    )
    return g


def build_rnd_combined_table(
    df_gov: pd.DataFrame,
    df_uni: pd.DataFrame,
    add_total_row: bool = True,
) -> pd.DataFrame:
    """
    정부출연연구소 / 산학협력단 표를 하나의 좌우 결합 테이블로 만든다.

    컬럼 구조
    ---------
    NO,
    gov_name, gov_cnt, gov_share,
    uni_name, uni_cnt, uni_share
    """
    # 둘 다 비어 있으면 빈 템플릿 반환
    if df_gov.empty and df_uni.empty:
        return pd.DataFrame(
            columns=[
                "NO",
                "gov_name",
                "gov_cnt",
                "gov_share",
                "uni_name",
                "uni_cnt",
                "uni_share",
            ]
        ).astype(
            {
                "NO": "Int64",
                "gov_name": "string",
                "gov_cnt": "Int64",
                "gov_share": "Float64",
                "uni_name": "string",
                "uni_cnt": "Int64",
                "uni_share": "Float64",
            }
        )

    # 길이를 맞추기 위해 최대 길이 기준으로 재인덱스
    max_len = max(len(df_gov), len(df_uni))

    gov = (
        df_gov[["기관명", "특허보유수", "비중(%)"]].copy()
        if not df_gov.empty
        else pd.DataFrame(columns=["기관명", "특허보유수", "비중(%)"])
    )
    uni = (
        df_uni[["기관명", "특허보유수", "비중(%)"]].copy()
        if not df_uni.empty
        else pd.DataFrame(columns=["기관명", "특허보유수", "비중(%)"])
    )

    gov = gov.reindex(range(max_len)).reset_index(drop=True)
    uni = uni.reindex(range(max_len)).reset_index(drop=True)

    df = pd.DataFrame(
        {
            "NO": pd.Series(range(1, max_len + 1), dtype="Int64"),
            "gov_name": gov["기관명"].astype("string"),
            "gov_cnt": pd.to_numeric(gov["특허보유수"], errors="coerce").astype("Int64"),
            "gov_share": pd.to_numeric(gov["비중(%)"], errors="coerce").astype(
                "Float64"
            ),
            "uni_name": uni["기관명"].astype("string"),
            "uni_cnt": pd.to_numeric(uni["특허보유수"], errors="coerce").astype("Int64"),
            "uni_share": pd.to_numeric(uni["비중(%)"], errors="coerce").astype(
                "Float64"
            ),
        }
    )

    # 맨 아래에 "합계" 행 추가
    if add_total_row:
        gov_total_cnt = df["gov_cnt"].astype("Float64").sum(skipna=True)
        gov_total_share = df["gov_share"].astype("Float64").sum(skipna=True)
        uni_total_cnt = df["uni_cnt"].astype("Float64").sum(skipna=True)
        uni_total_share = df["uni_share"].astype("Float64").sum(skipna=True)

        total_row = pd.DataFrame(
            {
                "NO": pd.Series([pd.NA], dtype="Int64"),
                "gov_name": pd.Series(["합계"], dtype="string"),
                "gov_cnt": pd.Series([gov_total_cnt], dtype="Int64"),
                "gov_share": pd.Series([gov_total_share], dtype="Float64"),
                "uni_name": pd.Series(["합계"], dtype="string"),
                "uni_cnt": pd.Series([uni_total_cnt], dtype="Int64"),
                "uni_share": pd.Series([uni_total_share], dtype="Float64"),
            }
        )

        df = pd.concat([df, total_row], ignore_index=True)

    return df


def render_issue_header() -> None:
    """
    R&D부문 상단 배너를 렌더링한다.

    :return: 없음.
    :rtype: None
    """
    st.markdown(
        f"""
        <div class="sec-banner" style="--accent:{"#5b9bd5"};">
        <div class="sec-label">{"R&D부문"}</div>
        <div class="sec-rule"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------
# 섹션 1: R&D부문 - 워드클라우드
# ---------------------------------------------------
def render_rnd_issue_wordcloud_section(
    sel_subcat_name: str,
    sel_subcat_code: str,
    send_sql,
) -> None:
    """
    R&D 이슈 워드클라우드 섹션을 렌더링한다.

    - TF-IDF 기반 워드클라우드
    - 상위 키워드 비중(%) 테이블을 함께 표시 (data_editor, 스크롤 없음)
    """
    # 상단 큰 제목
    st.markdown(
        f"### 소분류 [{sel_subcat_name}] R&D이슈",
        unsafe_allow_html=True,
    )

    freqs = fetch_wc_freqs_for_subcat(sel_subcat_name, sel_subcat_code, send_sql)

    if not freqs:
        st.info(
            f"워드클라우드에 사용할 명세서 텍스트가 없습니다. "
            f"(코드: {sel_subcat_code}, 명칭: {sel_subcat_name})"
        )
        return

    # 👉 워드클라우드는 30개, 비중 테이블은 10개만 사용
    TOP_N_WORDCLOUD = 30
    TOP_N_TABLE = 10

    top_freqs = dict(Counter(freqs).most_common(TOP_N_WORDCLOUD))

    # -----------------------------------
    # 1) 비중(%) 테이블 먼저 생성 (상위 10개만)
    # -----------------------------------
    df_wc = build_wc_issue_table(top_freqs, top_n=TOP_N_TABLE, decimals=0)

    # ✅ 합계 행 No 공백 처리 (순서 중요!)
    # 1) No 컬럼에서 합계 행을 NA로 만들어주고
    df_wc.loc[df_wc["R&D 이슈"] == "합계", "No"] = pd.NA

    # 2) 전체를 문자열 타입으로 바꾼 다음
    df_wc["No"] = df_wc["No"].astype("string")

    # 3) 다시 합계 행만 "" 로 바꾸기
    df_wc.loc[df_wc["R&D 이슈"] == "합계", "No"] = ""


    # 행 개수 기반으로 data_editor 전체가 보이도록 높이 계산
    n_rows = len(df_wc)
    row_h = 34        # 데이터 행 높이
    header_h = 32     # 헤더 높이
    toolbar_h = 38    # data_editor 상단 툴바 높이
    padding = 16      # 위/아래 여백

    table_height = n_rows * row_h + header_h + toolbar_h + padding

    # -----------------------------------
    # 2) 워드클라우드 이미지 생성
    #    → 표 높이에 맞춰서 같은 높이로 리사이즈
    # -----------------------------------
    img = generate_wordcloud(
        freqs=top_freqs,
        font_path="/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        top_n=TOP_N_WORDCLOUD,
        w=1600,
        h=1000,
    )

    TARGET_WIDTH = 680
    img_show = img.resize((TARGET_WIDTH, table_height), Image.LANCZOS)

    # -----------------------------------
    # 3) 좌/우 레이아웃 구성
    # -----------------------------------
    TITLE_BOX_HEIGHT = 36
    col_l, col_r = st.columns([3, 2], gap="small")  # gap 줄여서 더 붙게

    with col_l:
        # 왼쪽 설명 문구
        st.markdown(
            f"""
            <div style="height:{TITLE_BOX_HEIGHT}px;
                        display:flex;
                        align-items:flex-end;">
                <p style="margin:0; font-weight:600; font-size:20px;">
                    소분류 내 최근 3년 이내의 핵심 키워드
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.image(img_show, caption=sel_subcat_name, width=TARGET_WIDTH)

    with col_r:
        # 오른쪽 설명 문구
        st.markdown(
            f"""
            <div style="height:{TITLE_BOX_HEIGHT}px;
                        display:flex;
                        align-items:flex-end;">
                <p style="margin:0; font-weight:600; font-size:20px;">
                    R&amp;D 이슈 상위 키워드 비중(%)
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.data_editor(
            df_wc,
            height=35 * len(df_wc) + 38,
            use_container_width=False,
            hide_index=True, 
            disabled=True,   
            column_config={
                "No": st.column_config.NumberColumn("No", width=60),
                "R&D 이슈": st.column_config.TextColumn("R&D 이슈", width=200),
                "비중(%)": st.column_config.NumberColumn(
                    "비중(%)", format="%.0f", width=100
                ),
            },
        )


# ---------------------------------------------------
# 섹션 2: R&D 전문기관 현황
# ---------------------------------------------------
def render_rnd_institute_section(sel_subcat_name: str, sel_subcat_code: str, send_sql) -> None:
    """
    R&D 전문기관 현황 섹션을 렌더링한다.

    정부출연연구소 / 산학협력단 상위 기관을
    한 프레임 안에 좌·우로 배치하고,
    마지막 행에 '합계' 행을 추가한다.
    """
    st.markdown(
        f"""
        ### 소분류 [{sel_subcat_name}] R&D 전문기관 현황
        ##### 소분류 내 정부출연연구소와 산학협력단이 보유한 특허수
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    params_rnd = {"sel_subcat_code": sel_subcat_code}

    # 1) 원 데이터 조회
    df_gov_raw = send_sql(Q.q_rnd_gov_top(), params=params_rnd)
    df_uni_raw = send_sql(Q.q_rnd_uni_top(), params=params_rnd)

    # 2) 공통 가공 (순번/비중 계산 등)
    df_gov = build_rnd_table(df_gov_raw, "정부출연연구소")
    df_uni = build_rnd_table(df_uni_raw, "산학협력단")

    # 둘 다 없으면 바로 종료
    if df_gov.empty and df_uni.empty:
        st.info("R&D 전문기관 데이터가 없습니다.")
        return

    # 3) 비중 계산 (여기까지는 그대로)
    total_gov = float(df_gov["특허보유수"].sum()) if not df_gov.empty else 0.0
    total_uni = float(df_uni["특허보유수"].sum()) if not df_uni.empty else 0.0

    total_series = pd.Series(
        {"정부출연연구소": total_gov, "산학협력단": total_uni},
        dtype="Float64",
    )
    share_series = compute_percentage_shares(total_series, decimals=1)
    gov_share = float(share_series.get("정부출연연구소", 0.0))
    uni_share = float(share_series.get("산학협력단", 0.0))

    # ⛔ 이 부분은 삭제 (표 밖에 텍스트로 표시하던 줄)
    # st.markdown(
    #     f"**정부출연연구소 ({gov_share:.1f}%)  /  산학협력단 ({uni_share:.1f}%)**"
    # )

    # 헤더 라벨에 비중(%)를 같이 넣기
    gov_col_label = f"기관명(정부출연연구소 {gov_share:.1f}%)"
    uni_col_label = f"기관명(산학협력단 {uni_share:.1f}%)"

    # 4) 결합 테이블 생성 (합계 행 포함)
    df_combined = build_rnd_combined_table(df_gov, df_uni, add_total_row=True)
    df_combined = df_combined.rename(columns={"NO": "No"})

    # 5) 데이터프레임 렌더링
    st.data_editor(
        df_combined,
        height=35 * len(df_combined) + 38,
        use_container_width=False,
        hide_index=True, 
        disabled=True, 
        column_config={
            "No": st.column_config.NumberColumn("No", width=60),
            "gov_name": st.column_config.TextColumn(gov_col_label, width=400),
            "gov_cnt": st.column_config.NumberColumn("특허보유수", width=120),
            "gov_share": st.column_config.NumberColumn("비중(%)", format="%.1f", width=100),
            "uni_name": st.column_config.TextColumn(uni_col_label, width=400),
            "uni_cnt": st.column_config.NumberColumn("특허보유수", width=120),
            "uni_share": st.column_config.NumberColumn("비중(%)", format="%.1f", width=100),
        },
        key="rnd-institute-table-combined2",
    )


def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # 내부 컬럼명 → 화면 컬럼명 매핑
    #   부처명 = 소관부처
    #   담당자 = 사업담당자 연락처
    df = df.rename(
        columns={
            "소관부처": "부처명",
            "사업담당자 연락처": "담당자",
            "전문기관명": "전문기관",  # IRIS에서 이렇게 올 경우 대비
        }
    )

    # 사진에 맞는 컬럼 순서만 사용
    ordered_cols = ["부처명", "공고명", "접수기간", "전문기관", "담당자"]
    exist_cols = [c for c in ordered_cols if c in df.columns]
    df = df[exist_cols].copy()

    # No 컬럼 맨 앞에 추가 (1부터)
    df.insert(0, "No", range(1, len(df) + 1))

    return df

# ---------------------------------------------------
# 섹션 3: 정부 R&D사업 공고 현황 (IRIS)
# ---------------------------------------------------
# ---------------------------------------------------
# 섹션 3: 정부 R&D사업 공고 현황 (IRIS)
# ---------------------------------------------------
def render_gov_rnd_status_section(
    sel_subcat_name: str,
    sel_subcat_code: str,
    send_sql
) -> None:
    """
    IRIS(국가R&D 사업통합지원시스템)를 이용해
    해당 소분류와 연관된 정부 R&D사업 공고 현황을 보여준다.

    - 소분류코드 앞 2글자를 IRIS 기술분야 코드로 사용 (예: ED1234 -> 'ED')
    - 화면에는
        * 접수중 과제
        * 접수기간 도래 과제
      두 개 표를 위아래로 한 번에 보여준다.
    """
    st.markdown(
        f"""
        ### 소분류 [{sel_subcat_name}] 정부 R&amp;D사업 공고 현황
        ##### IRIS(국가R&amp;D 사업통합지원시스템) 2025-12-15 14:30 기준 
        """,
        unsafe_allow_html=True,
    )

    # 소분류코드 앞 2글자를 IRIS 기술분야 코드로 가정
    tech_prefix = (sel_subcat_code or "")[:2]
    tech_codes = [tech_prefix] if tech_prefix else None

    if tech_codes:
        st.caption(f"IRIS 기술분야 코드: {', '.join(tech_codes)} (소분류코드 앞 2자리)")
    else:
        st.caption("IRIS 기술분야 코드를 자동으로 판단할 수 없어 전체 분야 기준으로 조회합니다.")

    try:
        with st.spinner("DB에서 정부 R&D사업 공고를 조회하는 중입니다…"):
            params_rnd = {"tech_codes": tech_codes}
            df_open = send_sql(Q.q_iris_1(), params_rnd)
            df_closed = send_sql(Q.q_iris_2(), params_rnd)
    except Exception as e:
        st.error(f"정부 R&D사업 공고(DB) 조회 중 오류가 발생했습니다: {e}")
        return

    df_open_view = _prep_df(df_open)
    df_closed_view = _prep_df(df_closed)

    # 4) 접수중 과제 표 (위)
    st.markdown("#### 접수중 과제")
    if df_open_view.empty:
        st.info("해당 기술분야의 현재 접수중 공고가 없습니다.")
    else:
        st.data_editor(
            df_open_view,
            hide_index=True,
            disabled=True,
            use_container_width=False,
            height=35 * len(df_open_view) + 38,
            column_config={
                "No": st.column_config.NumberColumn("No", width=60),
                "부처명": st.column_config.TextColumn("부처명", width=150),
                "공고명": st.column_config.TextColumn("공고명", width=480),
                "접수기간": st.column_config.TextColumn("접수기간", width=180),
                "전문기관": st.column_config.TextColumn("전문기관", width=160),
                "담당자": st.column_config.TextColumn("담당자", width=400),
            },
            key=f"iris-open-{sel_subcat_code}",
        )

    # 5) 접수기간 도래 과제 표 (아래)
    st.markdown("#### 접수기간 도래 과제")
    if df_closed_view.empty:
        st.info("작년 동기 기준 마감 공고가 없습니다.")
    else:
        st.data_editor(
            df_closed_view,
            hide_index=True,
            disabled=True,
            use_container_width=False,
            height=35 * len(df_closed_view) + 38,
            column_config={
                "No": st.column_config.NumberColumn("No", width=60),
                "부처명": st.column_config.TextColumn("부처명", width=150),
                "공고명": st.column_config.TextColumn("공고명", width=480),
                "접수기간": st.column_config.TextColumn("접수기간", width=180),
                "전문기관": st.column_config.TextColumn("전문기관", width=160),
                "담당자": st.column_config.TextColumn("담당자", width=400),
            },
            key=f"iris-closed-{sel_subcat_code}",
        )

# ---------------------------------------------------
# 메인: R&D부문 화면
# ---------------------------------------------------
def render_issue(
    sel_subcat_name: str,
    sel_subcat_code: str,
    send_sql,
) -> None:
    """
    R&D부문 화면 전체를 그리는 메인 함수이다.

    동작 흐름
    ---------

    1. :func:`render_issue_header` 를 호출하여 상단 ``"R&D부문"`` 배너를 렌더링한다.
    2. :func:`render_rnd_issue_wordcloud_section` 을 호출하여
       선택된 기술분야에 대한 R&D 이슈 워드클라우드 섹션을 렌더링한다.
    3. :func:`render_rnd_institute_section` 을 호출하여
       정부출연연구소/산학협력단 R&D 전문기관 현황 섹션을 렌더링한다.

    세션 상태
    ---------

    * 이 함수는 세션 상태를 읽거나 변경하지 않는다.

    :param sel_subcat_name: 현재 선택된 소분류명이다.
    :type sel_subcat_name: str
    :param sel_subcat_code: 현재 선택된 소분류 코드이다.
    :type sel_subcat_code: str
    :param send_sql: DB 조회 함수이다.
    :type send_sql: Callable[[str, dict], pandas.DataFrame]
    :return: 없음.
    :rtype: None
    """

    # 0) 상단 헤더(배너)
    render_issue_header()

    # 1) R&D 이슈 워드클라우드 섹션
    render_rnd_issue_wordcloud_section(sel_subcat_name, sel_subcat_code, send_sql)
    st.markdown("<br><br>", unsafe_allow_html=True)

    # 2) R&D 전문기관 현황 섹션
    render_rnd_institute_section(sel_subcat_name, sel_subcat_code, send_sql)
    st.markdown("<br><br>", unsafe_allow_html=True)

    # 3) 정부 R&D사업 공고 현황 (IRIS)
    render_gov_rnd_status_section(sel_subcat_name, sel_subcat_code, send_sql)
