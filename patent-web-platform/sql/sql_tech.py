"""
sql_tech
========

기술분야 검색 SQL 모듈.

기술분류(대·중·소분류) 정보를 조회하여
화면의 기술분야 트리(대분류 → 중분류 → 소분류)를 구성하는 데 사용하는 SQL을 제공한다.

기능 개요
--------

* 대·중·소분류 코드/명을 DISTINCT로 조회한다.
* 기술분야 트리(대분류 → 중분류 → 소분류) 드롭다운/트리 컴포넌트 데이터 소스를 생성한다.
* 분류코드 기준으로 일관된 정렬 순서를 보장한다.
"""


def get_tech_categories() -> str:
    """
    기술분류 체계(대·중·소분류)를 한 번에 조회하는 SQL을 반환한다.

    설명
    ----
    * 국가과학기술표준분류체계 테이블에서 대분류/중분류/소분류 코드·명을 DISTINCT로 조회한다.
    * 코드 기준으로 정렬하여 UI 트리/드롭다운에 그대로 사용할 수 있는 형태로 반환한다.

    바인딩 파라미터
    --------------
    없음.

    반환 컬럼
    --------
    * big_code   : 대분류코드
    * big_name   : 대분류명
    * mid_code   : 중분류코드
    * mid_name   : 중분류명
    * small_code : 소분류코드
    * small_name : 소분류명

    반환값
    ------
    :return: 기술분야(대·중·소분류) 목록 조회용 SQL 문자열.
    :rtype: str
    """
    return """
    -- 기술분야 검색
    SELECT DISTINCT
        대분류코드 AS big_code
        , 대분류명_kr AS big_name
        , 중분류코드 AS mid_code
        , 중분류명_kr AS mid_name
        , 소분류코드 AS small_code
        , 소분류명_kr AS small_name
    FROM 국가과학기술표준분류체계 AS kst
    WHERE (대분류코드 NOT IN ('NA', 'HF', 'HG', 'HH', 'OC')
        OR 소분류코드 = 'HH1201')
    ;
    """
