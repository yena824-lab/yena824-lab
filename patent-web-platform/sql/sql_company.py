"""
sql_company
===========

기업 검색 SQL 모듈.

BIGx/기술혁신정보서비스에서 기업을 검색하고,
선택한 기업의 특허 목록·보유 특허 수를 조회하기 위한 SQL을 제공한다.

기능 개요
--------

* 특정 기업(법인번호_ENC)의 보유 특허 목록을 조회한다.
  - 출원번호(ENC/원본), 특허명, 출원·등록일자, 기술 소분류 정보를 함께 반환한다.
* 기업명을 기준으로 LIKE 검색을 수행하여 기업 목록을 빠르게 조회한다.
* 법인번호_ENC, 사업자등록번호를 기준으로 기업을 검색할 수 있는 빠른 조회 SQL을 제공한다.
* 여러 기업(법인번호 목록)에 대해 보유 특허 수(DISTINCT 출원번호_ENC)를 일괄 집계한다.
* 공공기관·산학협력단 등은 검색 대상에서 제외하여 민간·일반 기업 중심으로 결과를 반환한다.
"""


def q_patents_for_company() -> str:
    """
    특정 기업(법인번호 기준)이 보유한 특허 목록을 조회하는 SQL을 반환한다.

    설명
    ----
    * 기업_특허 테이블에서 해당 기업이 보유한 특허의 출원번호_ENC를 추출하고,
      특허 및 분류체계 테이블과 조인하여 특허 기본 정보와 소분류 정보를 함께 조회한다.
    * 특허출원일자를 기준으로 내림차순 정렬하여 최근 출원 특허가 먼저 보이도록 한다.

    바인딩 파라미터
    --------------
    * ``:corp_id``: 조회 대상 기업의 법인번호(ENC).

    반환 컬럼
    --------
    * 출원번호_ENC
    * 출원번호
    * 특허명(특허제목)
    * 특허등록일자
    * 특허출원일자
    * 소분류코드
    * 소분류명

    반환값
    ------
    :return: 기업별 특허 목록 조회용 SQL 문자열.
    :rtype: str
    """
    return """
    -- 기업명을 눌렀을 때 특허리스트
    WITH
    patent_company AS (
        SELECT cp.출원번호_ENC
            , MIN(cp.법인번호_ENC) AS 법인번호_ENC
        FROM 기업_특허 AS cp
        GROUP BY cp.출원번호_ENC
    ),
    valid_patent AS (
        SELECT DISTINCT
            cp.출원번호_ENC
        FROM 기업_특허 AS cp
        WHERE cp.법인번호_ENC = :corp_id
          AND cp.법인번호_ENC = (
                SELECT MIN(cp2.법인번호_ENC)
                FROM 기업_특허 AS cp2
                WHERE cp2.출원번호_ENC = cp.출원번호_ENC
          )
    )
    SELECT
        valid_patent.출원번호_ENC,
        p.출원번호,
        p.제목 AS 특허명,
        p.등록일자 AS 특허등록일자,
        p.출원일자 AS 특허출원일자,
        ps.소분류코드,
        kst.소분류명_kr AS 소분류명
    FROM valid_patent
    JOIN 특허 AS p
      ON valid_patent.출원번호_ENC = p.출원번호_ENC
    JOIN 특허_국가과학기술표준분류체계 AS ps
      ON valid_patent.출원번호_ENC = ps.출원번호_ENC
    JOIN 국가과학기술표준분류체계 AS kst
      ON ps.소분류코드 = kst.소분류코드
    WHERE (kst.대분류코드 NOT IN ('NA', 'HF', 'HG', 'HH', 'OC')
        OR kst.소분류코드 = 'HH1201')
    ORDER BY p.출원일자 DESC
    ;
    """

def q_company_fast_by_name() -> str:
    """
    기업명을 기준으로 기업 목록을 빠르게 검색하는 SQL을 반환한다.

    설명
    ----
    * 상호(기업명)에 ``LIKE`` 조건을 적용해 부분 일치 검색을 수행한다.
    * 공공기관 및 산학협력단은 검색 대상에서 제외한다.

    바인딩 파라미터
    --------------
    * ``:kw``: 기업명 검색 키워드 (예: ``"%삼성%"``).

    반환 컬럼
    --------
    * 법인번호_ENC
    * 기업명 (NULL인 경우 ``'(상호명 없음)'`` 으로 대체).

    반환값
    ------
    :return: 기업명을 이용한 기업 검색 SQL 문자열.
    :rtype: str
    """
    return """
    -- 대상기업에 해당되는 특허 수 count - 기업명
    WITH 
    target_companies AS (
        SELECT s.법인번호_ENC
            , c.법인번호
            , GROUP_CONCAT(DISTINCT s.상호 ORDER BY s.상호 ASC SEPARATOR ', ') AS 기업명
          FROM 상호 AS s
          JOIN 기업 AS c 
            ON s.법인번호_ENC = c.법인번호_ENC
        WHERE s.상호 LIKE :kw
          AND s.산학협력단여부 = 0
        GROUP BY s.법인번호_ENC, c.법인번호
    ),
    candidate_patents AS (
        SELECT cp.출원번호_ENC
            , cp.법인번호_ENC 
          FROM 기업_특허 cp
          JOIN target_companies tc 
            ON cp.법인번호_ENC = tc.법인번호_ENC
    ),
    verified_counts AS (
        SELECT cp.법인번호_ENC
            , COUNT(cp.출원번호_ENC) AS 대표특허수
          FROM candidate_patents cp
          JOIN 특허_국가과학기술표준분류체계 ps
            ON cp.출원번호_ENC = ps.출원번호_ENC
        WHERE 1=1
          AND (
              (ps.소분류코드 NOT LIKE 'NA%' AND 
                ps.소분류코드 NOT LIKE 'HF%' AND 
                ps.소분류코드 NOT LIKE 'HG%' AND 
                ps.소분류코드 NOT LIKE 'HH%' AND 
                ps.소분류코드 NOT LIKE 'OC%')
              OR ps.소분류코드 = 'HH1201'
          )
          AND cp.법인번호_ENC = (
              SELECT MIN(sub_cp.법인번호_ENC)
                FROM 기업_특허 sub_cp
                WHERE sub_cp.출원번호_ENC = cp.출원번호_ENC
          )
        GROUP BY cp.법인번호_ENC
    )
    SELECT tc.법인번호_ENC
        , tc.법인번호
        , tc.기업명
        , COALESCE(vc.대표특허수, 0) AS 대표특허수
      FROM target_companies tc
      LEFT JOIN verified_counts vc
        ON tc.법인번호_ENC = vc.법인번호_ENC
    ORDER BY 대표특허수 DESC
    ;
    """


def q_company_fast_by_corp_enc() -> str:
    """
    법인번호(ENC)를 기준으로 기업을 검색하는 SQL을 반환한다.

    설명
    ----
    * ``법인번호_ENC LIKE :kw`` 조건으로 검색한다.
    * 공공기관 및 산학협력단은 검색 대상에서 제외한다.

    바인딩 파라미터
    --------------
    * ``:corp_id``: 법인번호_ENC 검색 키워드.

    반환 컬럼
    --------
    * 법인번호_ENC
    * 기업명 (NULL인 경우 ``'(상호명 없음)'`` 으로 대체).

    반환값
    ------
    :return: 법인번호_ENC로 기업을 검색하는 SQL 문자열.
    :rtype: str
    """
    return """
    -- 대상기업에 해당되는 특허 수 count - 법인번호_ENC
    WITH 
    target_companies AS (
        SELECT s.법인번호_ENC
            , c.법인번호
            , GROUP_CONCAT(DISTINCT s.상호 ORDER BY s.상호 ASC SEPARATOR ', ') AS 기업명
          FROM 상호 AS s
          JOIN 기업 AS c 
            ON s.법인번호_ENC = c.법인번호_ENC
        WHERE c.법인번호 LIKE :corp_id
          AND s.산학협력단여부 = 0
        GROUP BY s.법인번호_ENC
    ),
    candidate_patents AS (
        SELECT cp.출원번호_ENC
            , cp.법인번호_ENC -- (검색된 기업의 ID)
          FROM 기업_특허 cp
          JOIN target_companies tc 
            ON cp.법인번호_ENC = tc.법인번호_ENC
    ),
    verified_counts AS (
        SELECT cp.법인번호_ENC
            , COUNT(cp.출원번호_ENC) AS 대표특허수
          FROM candidate_patents cp
          JOIN 특허_국가과학기술표준분류체계 ps
            ON cp.출원번호_ENC = ps.출원번호_ENC
        WHERE 1=1
          AND (
              (ps.소분류코드 NOT LIKE 'NA%' AND 
                ps.소분류코드 NOT LIKE 'HF%' AND 
                ps.소분류코드 NOT LIKE 'HG%' AND 
                ps.소분류코드 NOT LIKE 'HH%' AND 
                ps.소분류코드 NOT LIKE 'OC%')
              OR ps.소분류코드 = 'HH1201'
          )
          AND cp.법인번호_ENC = (
              SELECT MIN(sub_cp.법인번호_ENC)
                FROM 기업_특허 sub_cp
                WHERE sub_cp.출원번호_ENC = cp.출원번호_ENC
          )
        GROUP BY cp.법인번호_ENC
    )
    SELECT tc.법인번호_ENC
        , tc.법인번호
        , tc.기업명
        , COALESCE(vc.대표특허수, 0) AS 대표특허수
      FROM target_companies tc
      LEFT JOIN verified_counts vc
        ON tc.법인번호_ENC = vc.법인번호_ENC
    ORDER BY 대표특허수 DESC
    ;
    """


def q_company_fast_by_bizno() -> str:
    """
    사업자등록번호를 기준으로 기업을 검색하는 SQL을 반환한다.

    설명
    ----
    * ``사업자번호 LIKE :kw`` 조건으로 검색한다.
    * 공공기관 및 산학협력단은 검색 대상에서 제외한다.

    바인딩 파라미터
    --------------
    * ``:bizno``: 사업자번호 검색 키워드.

    반환 컬럼
    --------
    * 법인번호_ENC
    * 기업명 (NULL인 경우 ``'(상호명 없음)'``).

    반환값
    ------
    :return: 사업자번호로 기업을 검색하는 SQL 문자열.
    :rtype: str
    """
    return """
    -- 대상기업에 해당되는 특허 수 count - 사업자번호
    WITH 
    target_companies AS (
        SELECT s.법인번호_ENC
            , c.법인번호
            , GROUP_CONCAT(DISTINCT s.상호 ORDER BY s.상호 ASC SEPARATOR ', ') AS 기업명
          FROM 상호 AS s
          JOIN 기업 AS c 
            ON s.법인번호_ENC = c.법인번호_ENC
        WHERE c.사업자번호 LIKE :bizno
          AND s.산학협력단여부 = 0
        GROUP BY s.법인번호_ENC, c.법인번호
    ),
    candidate_patents AS (
        SELECT cp.출원번호_ENC
            , cp.법인번호_ENC 
          FROM 기업_특허 cp
          JOIN target_companies tc 
            ON cp.법인번호_ENC = tc.법인번호_ENC
    ),
    verified_counts AS (
        SELECT cp.법인번호_ENC
            , COUNT(cp.출원번호_ENC) AS 대표특허수
          FROM candidate_patents cp
          JOIN 특허_국가과학기술표준분류체계 ps
            ON cp.출원번호_ENC = ps.출원번호_ENC
        WHERE 1=1
          AND (
              (ps.소분류코드 NOT LIKE 'NA%' AND 
                ps.소분류코드 NOT LIKE 'HF%' AND 
                ps.소분류코드 NOT LIKE 'HG%' AND 
                ps.소분류코드 NOT LIKE 'HH%' AND 
                ps.소분류코드 NOT LIKE 'OC%')
              OR ps.소분류코드 = 'HH1201'
          )
          AND cp.법인번호_ENC = (
              SELECT MIN(sub_cp.법인번호_ENC)
                FROM 기업_특허 sub_cp
                WHERE sub_cp.출원번호_ENC = cp.출원번호_ENC
          )
        GROUP BY cp.법인번호_ENC
    )
    SELECT tc.법인번호_ENC
        , tc.법인번호
        , tc.기업명
        , COALESCE(vc.대표특허수, 0) AS 대표특허수
      FROM target_companies tc
      LEFT JOIN verified_counts vc
        ON tc.법인번호_ENC = vc.법인번호_ENC
    ORDER BY 대표특허수 DESC
    ;
    """


def q_patent_count_by_corp_ids(placeholders: str) -> str:
    """
    여러 기업(법인번호 목록)에 대해 보유 특허 수를 집계하는 SQL을 반환한다.

    설명
    ----
    * IN 절에 들어갈 named parameter 목록 문자열을 외부에서 만들어 전달한다.
      예) ``":id0, :id1, :id2"``

    바인딩 파라미터
    --------------
    * ``IN (...)`` 에 전달될 개별 파라미터들
      (예: ``:id0``, ``:id1`` 등; 딕셔너리로 바인딩).

    반환 컬럼
    --------
    * 법인번호_ENC
    * 기업명
    * 전체특허보유수

    반환값
    ------
    :param placeholders: IN 절에 사용할 named parameter 목록 문자열.
    :type placeholders: str
    :return: 다수 기업의 특허 보유 수 집계용 SQL 문자열.
    :rtype: str
    """
    return f"""
      WITH 
      target_companies AS (
          SELECT s.법인번호_ENC
              , GROUP_CONCAT(DISTINCT s.상호 ORDER BY s.상호 SEPARATOR ', ') AS 기업명
            FROM 상호 AS s
          WHERE s.상호 LIKE '%엘지%' 
            AND s.산학협력단여부 = 0
          GROUP BY s.법인번호_ENC
      ),
      related_patents AS (
          SELECT DISTINCT cp.출원번호_ENC
            FROM 기업_특허 cp
            JOIN target_companies tc
              ON cp.법인번호_ENC = tc.법인번호_ENC
            JOIN 특허_국가과학기술표준분류체계 AS ps
              ON cp.출원번호_ENC = ps.출원번호_ENC
      ),
      patent_representatives AS (
          SELECT 출원번호_ENC
              , MIN(법인번호_ENC) AS 대표법인번호_ENC
            FROM 기업_특허
          WHERE 출원번호_ENC IN (SELECT 출원번호_ENC FROM related_patents)
          GROUP BY 출원번호_ENC
      )
      SELECT tc.기업명
          , tc.법인번호_ENC
          , COUNT(pr.출원번호_ENC) AS 특허수
        FROM target_companies tc
        LEFT JOIN patent_representatives pr
          ON tc.법인번호_ENC = pr.대표법인번호_ENC
      GROUP BY tc.법인번호_ENC, tc.기업명
      ORDER BY 특허수 DESC, 기업명 ASC
      ;
    """

# 예: queries_company.py 같은 곳에 추가
def q_peer_midclass_patents() -> str:
    """
    동종 업종 내 기업들의 특허를 기준으로
    기술 중분류별 특허 수를 집계하는 쿼리이다.

    :return: SQL 문자열
    :rtype: str
    """
    return """
    WITH 
    target_patents AS (
        SELECT cp.출원번호_ENC
        FROM 기업_특허 AS cp
        WHERE cp.법인번호_ENC = :corp_id
    ),
    base_data AS (
        SELECT target_patents.출원번호_ENC
            , ps.소분류코드
            , kst.중분류코드
            , kst.중분류명_kr AS 중분류명
        FROM target_patents
        JOIN 특허_국가과학기술표준분류체계 ps
          ON target_patents.출원번호_ENC = ps.출원번호_ENC
        JOIN 국가과학기술표준분류체계 AS kst
          ON ps.소분류코드 = kst.소분류코드
      WHERE (kst.대분류코드 NOT IN ('NA', 'HF', 'HG', 'HH', 'OC')
        OR kst.소분류코드 = 'HH1201')
    )
    SELECT 중분류코드
        , 중분류명
        , COUNT(DISTINCT 출원번호_ENC) AS 특허수
      FROM base_data
    GROUP BY 중분류코드, 중분류명
    ;
    """

def q_peer_subclass_patents() -> str:
    """
    동종 업종 내 기업들의 특허를 기준으로
    기술 소분류별 특허 수를 집계하는 쿼리이다.

    :return: SQL 문자열
    :rtype: str
    """
    return """
    WITH 
    target_patents AS (
        SELECT cp.출원번호_ENC
        FROM 기업_특허 AS cp
        WHERE cp.법인번호_ENC = :corp_id
    ),
    base_data AS (
        SELECT target_patents.출원번호_ENC
            , ps.소분류코드
            , kst.소분류명_kr AS 소분류명
            , kst.중분류코드
        FROM target_patents
        JOIN 특허_국가과학기술표준분류체계 ps
          ON target_patents.출원번호_ENC = ps.출원번호_ENC
        JOIN 국가과학기술표준분류체계 AS kst
          ON ps.소분류코드 = kst.소분류코드
      WHERE (kst.대분류코드 NOT IN ('NA', 'HF', 'HG', 'HH', 'OC')
        OR kst.소분류코드 = 'HH1201')
    )
    SELECT 소분류코드
        , 소분류명
        , COUNT(DISTINCT 출원번호_ENC) AS 특허수
      FROM base_data
    GROUP BY 소분류코드
    ;
    """
