"""
sql_company_0_detail
====================

특허가 0건인 기업을 위한
'동업종 내 기술분포' 화면용 SQL 모듈.

역할
----

* 신청기업의 업종코드(10차) / 업종명을 조회한다.
* 해당 업종에 속한 기업 집합을 만든 뒤
  그 기업들의 특허를 기준으로 중분류/소분류 분포를 집계한다.
* 대표 기술분야(소분류)에 대해서는 매출구간별 기업수/특허수 분포를 추가로 제공한다.
"""

def q_industry_basic_info() -> str:
    """
    신청기업의 업종 기본 정보를 조회한다.

    파라미터
    --------
    :applicant_corp_no  신청기업 법인번호_ENC

    반환 컬럼
    --------
    * 업종코드_10차
    * 업종명_10차
    * 동업종_기업수
    * 동업종_특허수
    """
    return """
    WITH target_industry AS (
        SELECT cu.업종코드_10차
        FROM 기업_업종 cu
        WHERE cu.법인번호_ENC = :applicant_corp_no
        LIMIT 1
    ),
    industry_companies AS (
        SELECT cu.법인번호_ENC
        FROM 기업_업종 cu
        JOIN target_industry ti
          ON cu.업종코드_10차 = ti.업종코드_10차
    ),
    -- 공공기관 / 산학협력단 제외
    filtered_companies AS (
        SELECT ic.법인번호_ENC
        FROM industry_companies ic
        JOIN 기업 c
          ON ic.법인번호_ENC = c.법인번호_ENC
        LEFT JOIN 상호 s
          ON ic.법인번호_ENC = s.법인번호_ENC
        WHERE c.정부및공공기관구분 != '공공기관'
          AND (s.산학협력단여부 IS NULL OR s.산학협력단여부 = 0)
    ),
    patent_company AS (
        SELECT DISTINCT cp.출원번호_ENC, cp.법인번호_ENC
        FROM 기업_특허 cp
        JOIN filtered_companies fc
          ON cp.법인번호_ENC = fc.법인번호_ENC
    )
    SELECT
        ti.업종코드_10차,
        ij.업종명_10차,
        COUNT(DISTINCT fc.법인번호_ENC) AS 동업종_기업수,
        COUNT(DISTINCT pc.출원번호_ENC) AS 동업종_특허수
    FROM target_industry ti
    JOIN 업종 ij
      ON ti.업종코드_10차 = ij.업종코드_10차
    LEFT JOIN filtered_companies fc
      ON 1=1
    LEFT JOIN patent_company pc
      ON fc.법인번호_ENC = pc.법인번호_ENC
    GROUP BY ti.업종코드_10차, ij.업종명_10차
    ;
    """


def q_industry_mid_dist() -> str:
    """
    동업종 기업이 보유한 특허를 기준으로
    중분류별 특허 수를 집계한다.

    반환 컬럼
    --------
    * 중분류코드
    * 중분류명
    * 특허수
    """
    return """
    WITH target_industry AS (
        SELECT cu.업종코드_10차
        FROM 기업_업종 cu
        WHERE cu.법인번호_ENC = :applicant_corp_no
        LIMIT 1
    ),
    industry_companies AS (
        SELECT cu.법인번호_ENC
        FROM 기업_업종 cu
        JOIN target_industry ti
          ON cu.업종코드_10차 = ti.업종코드_10차
    ),
    filtered_companies AS (
        SELECT ic.법인번호_ENC
        FROM industry_companies ic
        JOIN 기업 c
          ON ic.법인번호_ENC = c.법인번호_ENC
        LEFT JOIN 상호 s
          ON ic.법인번호_ENC = s.법인번호_ENC
        WHERE c.정부및공공기관구분 != '공공기관'
          AND (s.산학협력단여부 IS NULL OR s.산학협력단여부 = 0)
    ),
    patent_company AS (
        SELECT DISTINCT cp.출원번호_ENC, cp.법인번호_ENC
        FROM 기업_특허 cp
        JOIN filtered_companies fc
          ON cp.법인번호_ENC = fc.법인번호_ENC
    ),
    tech_class AS (
        SELECT
            pc.출원번호_ENC,
            kst.중분류코드,
            kst.중분류명_kr AS 중분류명
        FROM patent_company pc
        JOIN 특허_국가과학기술표준분류체계 ps
          ON pc.출원번호_ENC = ps.출원번호_ENC
        JOIN 국가과학기술표준분류체계 kst
          ON ps.소분류코드 = kst.소분류코드
    )
    SELECT
        tech_class.중분류코드,
        tech_class.중분류명,
        COUNT(DISTINCT tech_class.출원번호_ENC) AS 특허수
    FROM tech_class
    GROUP BY tech_class.중분류코드, tech_class.중분류명
    ORDER BY 특허수 DESC
    ;
    """


def q_industry_sub_dist() -> str:
    """
    동업종 기업이 보유한 특허를 기준으로
    소분류별 특허 수를 집계한다.

    반환 컬럼
    --------
    * 소분류코드
    * 소분류명
    * 특허수
    """
    return """
    WITH target_industry AS (
        SELECT cu.업종코드_10차
        FROM 기업_업종 cu
        WHERE cu.법인번호_ENC = :applicant_corp_no
        LIMIT 1
    ),
    industry_companies AS (
        SELECT cu.법인번호_ENC
        FROM 기업_업종 cu
        JOIN target_industry ti
          ON cu.업종코드_10차 = ti.업종코드_10차
    ),
    filtered_companies AS (
        SELECT ic.법인번호_ENC
        FROM industry_companies ic
        JOIN 기업 c
          ON ic.법인번호_ENC = c.법인번호_ENC
        LEFT JOIN 상호 s
          ON ic.법인번호_ENC = s.법인번호_ENC
        WHERE c.정부및공공기관구분 != '공공기관'
          AND (s.산학협력단여부 IS NULL OR s.산학협력단여부 = 0)
    ),
    patent_company AS (
        SELECT DISTINCT cp.출원번호_ENC, cp.법인번호_ENC
        FROM 기업_특허 cp
        JOIN filtered_companies fc
          ON cp.법인번호_ENC = fc.법인번호_ENC
    ),
    tech_class AS (
        SELECT
            pc.출원번호_ENC,
            kst.소분류코드,
            kst.소분류명_kr AS 소분류명
        FROM patent_company pc
        JOIN 특허_국가과학기술표준분류체계 ps
          ON pc.출원번호_ENC = ps.출원번호_ENC
        JOIN 국가과학기술표준분류체계 kst
          ON ps.소분류코드 = kst.소분류코드
    )
    SELECT
        tech_class.소분류코드,
        tech_class.소분류명,
        COUNT(DISTINCT tech_class.출원번호_ENC) AS 특허수
    FROM tech_class
    GROUP BY tech_class.소분류코드, tech_class.소분류명
    ORDER BY 특허수 DESC
    ;
    """


def q_top_subcat_sales_dist() -> str:
    """
    특정 소분류코드에 대해,
    매출액 구간(그룹1~5)별 기업 수와 특허 수를 조회한다.

    동업종 대표 기술분야(소분류) 매출구간 분포 차트용으로 사용한다.

    파라미터
    --------
    :sel_subcat_code  소분류코드

    반환 컬럼
    --------
    * 최신매출액구간  (그룹1~그룹5)
    * 기업수
    * 특허수
    """
    return """
    -- 소분류 내 매출액 구간별 기업 수 / 특허 수
    WITH
    RANGE_LIST AS (
        SELECT '그룹1' AS 최신매출액구간
        UNION ALL SELECT '그룹2'
        UNION ALL SELECT '그룹3'
        UNION ALL SELECT '그룹4'
        UNION ALL SELECT '그룹5'
    ),
    sc_patent AS (
        SELECT ps.출원번호_ENC
        FROM 특허_국가과학기술표준분류체계 AS ps
        WHERE ps.소분류코드 = :sel_subcat_code
    ),
    cp_ranked AS (
        SELECT sc_patent.출원번호_ENC
            , MIN(cp.법인번호_ENC) AS 법인번호_ENC
        FROM sc_patent
        LEFT JOIN 기업_특허 AS cp
            ON cp.출원번호_ENC = sc_patent.출원번호_ENC
        GROUP BY sc_patent.출원번호_ENC
    ),
    univ_exclude AS (
        SELECT DISTINCT s.법인번호_ENC
        FROM 상호 AS s
        WHERE s.산학협력단여부 = 1
    ),
    latest_financial  AS (
        SELECT fi.법인번호_ENC
            , MAX(fi.결산기준일자) AS 최근결산기준일자
        FROM cp_ranked
        LEFT JOIN 재무정보 fi
            ON cp_ranked.법인번호_ENC = fi.법인번호_ENC
        WHERE fi.결산기준일자 <= '20241231'
        GROUP BY fi.법인번호_ENC
    ),
    corp_sales AS ( 
        SELECT fi.법인번호_ENC
            , fi.매출액구간 AS 최신매출액구간
        FROM 재무정보 fi
        JOIN latest_financial
            ON fi.법인번호_ENC = latest_financial.법인번호_ENC
        AND fi.결산기준일자  = latest_financial.최근결산기준일자
    ),
    PatentFact AS (
        SELECT sc_patent.출원번호_ENC
            , cp_ranked.법인번호_ENC
            , COALESCE(corp_sales.최신매출액구간, '그룹1') AS 최신매출액구간
            , c.정부및공공기관구분
        FROM sc_patent
        LEFT JOIN cp_ranked
            ON sc_patent.출원번호_ENC = cp_ranked.출원번호_ENC
        LEFT JOIN univ_exclude
            ON cp_ranked.법인번호_ENC = univ_exclude.법인번호_ENC
        LEFT JOIN 기업 AS c
            ON cp_ranked.법인번호_ENC = c.법인번호_ENC
        LEFT JOIN corp_sales
            ON cp_ranked.법인번호_ENC = corp_sales.법인번호_ENC
        WHERE univ_exclude.법인번호_ENC IS NULL
    )
    SELECT RANGE_LIST.최신매출액구간
        , COUNT(DISTINCT PatentFact.법인번호_ENC) AS 기업수
        , COUNT(PatentFact.출원번호_ENC) AS 특허수
    FROM RANGE_LIST
    LEFT JOIN PatentFact
        ON RANGE_LIST.최신매출액구간 = PatentFact.최신매출액구간
       AND PatentFact.정부및공공기관구분 != '공공기관'
    GROUP BY RANGE_LIST.최신매출액구간
    ORDER BY CASE RANGE_LIST.최신매출액구간
            WHEN '그룹1'  THEN 1
            WHEN '그룹2'  THEN 2
            WHEN '그룹3'  THEN 3
            WHEN '그룹4'  THEN 4
            WHEN '그룹5'  THEN 5
            END
    ;
    """

def q_top_subcat_info() -> str:
    """
    대표 기술분야(소분류코드) 기준으로
    소분류명과 소분류설명을 조회하는 SQL.

    바인딩 파라미터
    --------------
    * ``:sel_subcat_code``: 선택 소분류 코드.

    반환 컬럼
    --------
    * 소분류코드
    * 소분류명
    * 소분류설명
    """
    return """
    SELECT
        kst.소분류코드,
        kst.소분류명_kr AS 소분류명,
        kst.소분류설명
    FROM 국가과학기술표준분류체계 AS kst
    WHERE kst.소분류코드 = :sel_subcat_code
    ;
    """

