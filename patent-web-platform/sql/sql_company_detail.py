"""
sql_company_detail
==================

기업 상세 분석 SQL 모듈.

신청기업(대상기업)이 속한 기술 소분류를 기준으로
매출구간별 Top3 기업, 소분류 평균, 성장률, 우수기술 등을 조회하여
기업혁신성장 보고서에서 경쟁사 대비 위치를 분석하는 데 사용하는 SQL을 제공한다.

기능 개요
--------

* 선택 특허/소분류 정보를 기반으로 소분류명·소분류설명·매핑이유와
  소분류 내 기업 수/특허 수를 조회한다.
* 특허(출원번호_ENC) 기준으로 대상기업의 기업명·매출액 구간을 조회한다.
* 소분류 내 매출구간별 기업/특허 수를 집계하여 파이차트용 데이터를 생성한다.
* 소분류 내 매출구간별 Top3 기업, 소분류 평균 매출, 대상기업 매출/매출구간을 함께 조회한다.
* 매출액 구간(그룹1~그룹5)별로 상위기업·신청기업·해당 구간 평균 매출을 비교 조회한다.
* 소분류 내 성장률 우수기업(매출규모별 3년 성장률 Top10)과
  대상기업/소분류 평균 3년 성장률을 조회한다.
* 기술부문에서 피인용 기준 우수기술 Top10 등 대상기업이 속한 기술분야의
  기술 경쟁력을 파악할 수 있는 SQL을 제공한다.
"""
def q_stored_col() -> str:
    return"""
    CALL RECREATE_SALESRANGE_COLUMN(:c1, :c2, :c3, :c4);
    """

# ---------------------------------------------------
# 개요부문
# ---------------------------------------------------

def q_patent_abs() -> str:
    """
    선택한 특허 제목에 대한 초록(설명)과 출원번호를 조회하는 SQL을 반환한다.

    설명
    ----
    * 특허 테이블에서 제목 기준으로 단일 특허를 조회하고,
      초록과 출원번호를 함께 반환한다.

    바인딩 파라미터
    --------------
    * ``:title``: 특허제목.

    반환 컬럼
    --------
    * 특허설명 (특허초록)
    * 출원번호

    반환값
    ------
    :return: 특허 초록 및 출원번호 조회용 SQL 문자열.
    :rtype: str
    """
    return """
    -- 특허명
    SELECT p.초록 AS 특허설명
        , p.출원번호
      FROM 특허 AS p
    WHERE p.제목 = :title
    ;
    """


def q_subcat_info() -> str:
    """
    선택된 특허(출원번호, 소분류코드 기준)의 소분류 정보와
    소분류 내 기업 수 및 특허 수를 조회하는 SQL을 반환한다.

    설명
    ----
    * BaseInfo: 선택 특허와 소분류코드로 소분류명, 설명, 매핑이유를 조회.
    * CompanyCount: 해당 소분류에 속한 기업 수를 집계.
    * PatentCount: 해당 소분류에 속한 특허 수를 집계.

    바인딩 파라미터
    --------------
    * ``:selected_appno_enc``: 선택 특허의 출원번호_ENC.
    * ``:sel_subcat_code``: 선택 소분류 코드.

    반환 컬럼
    --------
    * 소분류명
    * 소분류설명
    * 매핑이유
    * 기업수 (해당 소분류에 속한 기업 수)
    * 특허수 (해당 소분류에 속한 특허 수)

    반환값
    ------
    :return: 소분류 기본 정보 및 기업/특허 수 조회용 SQL 문자열.
    :rtype: str
    """
    return """
    -- 소분류정보
    WITH
    BaseInfo AS (
        SELECT
            ps.소분류코드,
            kst.소분류명_kr AS 소분류명,
            kst.소분류설명,
            ps.이유 AS 매핑이유
        FROM 특허_국가과학기술표준분류체계 AS ps
        JOIN 국가과학기술표준분류체계 AS kst
          ON ps.소분류코드 = kst.소분류코드
        WHERE ps.출원번호_ENC = :selected_appno_enc
          AND ps.소분류코드 = :sel_subcat_code
    ),
    cp_ranked AS (
        SELECT
            ps.출원번호_ENC,
            MIN(cp.법인번호_ENC) AS 법인번호_ENC
        FROM 특허_국가과학기술표준분류체계 AS ps
        LEFT JOIN 기업_특허 AS cp
          ON ps.출원번호_ENC = cp.출원번호_ENC
        GROUP BY ps.출원번호_ENC
    ),
    CompanyCount AS (
        SELECT
            b.소분류코드,
            COUNT(DISTINCT g.법인번호_ENC) AS 기업수
        FROM BaseInfo AS b
        JOIN 특허_국가과학기술표준분류체계 AS ps
          ON b.소분류코드 = ps.소분류코드
        LEFT JOIN cp_ranked
          ON ps.출원번호_ENC = cp_ranked.출원번호_ENC
        LEFT JOIN 기업 AS g
          ON cp_ranked.법인번호_ENC = g.법인번호_ENC
        GROUP BY b.소분류코드
    ),
    PatentCount AS (
        SELECT
            b.소분류코드,
            COUNT(ps.출원번호_ENC) AS 특허수
        FROM BaseInfo AS b
        JOIN 특허_국가과학기술표준분류체계 AS ps
          ON b.소분류코드 = ps.소분류코드
        GROUP BY b.소분류코드
    )
    SELECT
        BaseInfo.소분류명,
        BaseInfo.소분류설명,
        BaseInfo.매핑이유,
        CompanyCount.기업수,
        PatentCount.특허수
    FROM BaseInfo
    LEFT JOIN CompanyCount
      ON BaseInfo.소분류코드 = CompanyCount.소분류코드
    LEFT JOIN PatentCount
      ON BaseInfo.소분류코드 = PatentCount.소분류코드;
    """

def q_corp_finance() -> str:
    """
    특허(출원번호_ENC)를 기준으로 기업명과 매출액 구간 정보를 조회하는 SQL을 반환한다.

    설명
    ----
    * BasePatent: 대상 특허(출원번호_ENC)를 기준으로 기본 특허 집합.
    * AppCorp: 해당 특허의 최소 법인번호_ENC(대표 기업)를 추출.
    * CompanyName: 대표 기업에 대한 기업명 집계.
    * latest_financial / corp_sales: 최근 결산기준일자 기준 매출액/매출액구간 추출.

    바인딩 파라미터
    --------------
    * ``:nm``: 출원번호_ENC.

    반환 컬럼
    --------
    * 기업명
    * 매출액구간 (최신매출액구간, 공공기관은 강제로 '50억원 이하')

    반환값
    ------
    :return: 특허 기준 기업/재무 정보 조회용 SQL 문자열.
    :rtype: str
    """
    return """
    -- 기업/재무정보
    WITH
    BasePatent AS (
        SELECT p.출원번호_ENC
        FROM 특허 AS p
        WHERE p.출원번호_ENC = :nm
    ),
    AppCorp AS (
        SELECT BasePatent.출원번호_ENC
            , MIN(cp.법인번호_ENC) AS 법인번호_ENC
        FROM BasePatent
        LEFT JOIN 기업_특허 AS cp
            ON cp.출원번호_ENC = BasePatent.출원번호_ENC
        GROUP BY BasePatent.출원번호_ENC
    ),
    CompanyName AS (
        SELECT s.법인번호_ENC
            , GROUP_CONCAT(DISTINCT s.상호 ORDER BY s.상호 ASC SEPARATOR ', ') AS 기업명
        FROM AppCorp
        JOIN 상호 AS s
            ON AppCorp.법인번호_ENC = s.법인번호_ENC AND s.산학협력단여부 = 0
        GROUP BY s.법인번호_ENC
    ),
    latest_financial AS (
            SELECT fi.법인번호_ENC
            , MAX(fi.결산기준일자) AS 최근결산기준일자
        FROM AppCorp
        LEFT JOIN 재무정보 fi
        ON AppCorp.법인번호_ENC = fi.법인번호_ENC
        WHERE LEFT(fi.결산기준일자, 4) = '2024'
        AND fi.매출액 IS NOT NULL
        GROUP BY fi.법인번호_ENC
    ),
    corp_sales AS ( 
        SELECT fi.법인번호_ENC
            , fi.매출액
            , fi.매출액구간 AS 최신매출액구간
        FROM 재무정보 fi
        JOIN latest_financial
        ON fi.법인번호_ENC = latest_financial.법인번호_ENC
        AND fi.결산기준일자 = latest_financial.최근결산기준일자
    )
    SELECT IFNULL(CompanyName.기업명, '-') AS 기업명
        , CASE WHEN c.정부및공공기관구분 = '공공기관' THEN '50억원 이하' ELSE corp_sales.최신매출액구간 END AS 매출액구간
    FROM AppCorp
    LEFT JOIN CompanyName
        ON AppCorp.법인번호_ENC = CompanyName.법인번호_ENC
    LEFT JOIN corp_sales
        ON AppCorp.법인번호_ENC = corp_sales.법인번호_ENC
    LEFT JOIN 기업 AS c
        ON AppCorp.법인번호_ENC = c.법인번호_ENC
    ;
    """


def q_overview_pie_and_counts() -> str:
    """
    소분류 내 매출액 구간별 기업 수와 특허 수를 조회하는 SQL을 반환한다.

    설명
    ----
    * RANGE_LIST: 5개 매출 구간을 고정 정의.
    * sc_patent: 소분류 코드에 해당하는 특허 목록.
    * cp_ranked: 특허별 대표 기업(최소 법인번호_ENC).
    * latest_financial / corp_sales: 기업별 최신 매출액구간 추출.
    * PatentFact: 공공기관/산학협력단 제외 후 구간별 기업/특허 정보 구성.

    바인딩 파라미터
    --------------
    * ``:sel_subcat_code``: 선택 소분류 코드.

    반환 컬럼
    --------
    * 최신매출액구간
    * 기업수 (해당 구간 내 DISTINCT 법인번호_ENC 수)
    * 특허수 (해당 구간 내 특허 수)

    반환값
    ------
    :return: 매출액 구간별 기업/특허 수 집계용 SQL 문자열.
    :rtype: str
    """
    return """
    -- 파이차트
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
        WHERE LEFT(fi.결산기준일자, 4) = '2024'
            AND fi.매출액 IS NOT NULL
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
            , COALESCE(corp_sales.최신매출액구간, '50억원 이하') AS 최신매출액구간
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

# ---------------------------------------------------
# 재무부문
# ---------------------------------------------------

def q_fin_peer_window() -> str:
    """
    기업 검색: 유사도 고려를 위해 해당 소분류에 매핑된 기업들(전체)의 2024년 매출액을 내림차순 정렬
    변수: :sel_subcat_code, :applicant_corp_no
    """
    return """
-- 기업 검색: 유사도 고려를 위해 해당 소분류에 매핑된 기업들(전체)의 2024년 매출액을 내림차순 정렬
-- 변수: 소분류코드, 신청기업의 법인번호_ENC
-- 251215 추가: 매출액_보정여부, 매출액영업이익률_보정여부
-- 251217 추가: 법인번호
WITH patent_company AS (
    SELECT 
        ps.출원번호_ENC,
        MIN(cp.법인번호_ENC) AS 법인번호_ENC
    FROM 특허_국가과학기술표준분류체계 ps
    JOIN 기업_특허 cp
        ON cp.출원번호_ENC = ps.출원번호_ENC
    WHERE ps.소분류코드 = :sel_subcat_code -- 여기 소분류코드
    GROUP BY ps.출원번호_ENC
),
unique_company AS (
    SELECT DISTINCT 법인번호_ENC
    FROM patent_company
),
finance_2024 AS (
    SELECT *
    FROM 재무정보
    WHERE LEFT(결산기준일자, 4) = '2024'
),
latest_financial AS (
    SELECT 
        법인번호_ENC,
        MAX(결산기준일자) AS 최근결산기준일자
    FROM finance_2024
    GROUP BY 법인번호_ENC
),
profit_ratio_ranked AS (
    SELECT
        f.법인번호_ENC,
        f.매출액영업이익률,
        f.보정여부_매출액영업이익률 AS 매출액영업이익률_보정여부,
        f.결산기준일자,
        ROW_NUMBER() OVER (
            PARTITION BY f.법인번호_ENC
            ORDER BY 
                (f.결산기준일자 = lf.최근결산기준일자 AND f.매출액영업이익률 IS NOT NULL) DESC,
                (f.매출액영업이익률 IS NOT NULL) DESC,
                f.결산기준일자 DESC
        ) AS rn
    FROM finance_2024 f
    JOIN latest_financial lf
        ON f.법인번호_ENC = lf.법인번호_ENC
),
selected_profit_ratio AS (
    SELECT 
        법인번호_ENC,
        매출액영업이익률,
        결산기준일자 AS 이익률_결산기준일자,
        매출액영업이익률_보정여부
    FROM profit_ratio_ranked
    WHERE rn = 1
),
latest_sales AS (
    SELECT 
        f.법인번호_ENC,
        f.매출액,
        f.보정여부 AS 매출액_보정여부,
        CASE
            WHEN f.매출액 IS NULL OR f.매출액 = 0
                THEN NULL
            ELSE pr.매출액영업이익률
        END AS 매출액영업이익률,
        f.결산기준일자 AS 매출액_결산기준일자,
        pr.이익률_결산기준일자,
        pr.매출액영업이익률_보정여부,
        f.매출액구간,
        CASE
            WHEN c.정부및공공기관구분 = '공공기관'
                THEN '그룹1'
            ELSE f.매출액구간
        END AS 최종매출액구간,
        c.정부및공공기관구분
    FROM finance_2024 f
    JOIN latest_financial lf
        ON f.법인번호_ENC = lf.법인번호_ENC
       AND f.결산기준일자 = lf.최근결산기준일자
    LEFT JOIN selected_profit_ratio pr
        ON f.법인번호_ENC = pr.법인번호_ENC
    JOIN 기업 c 
        ON f.법인번호_ENC = c.법인번호_ENC
),
target_companies AS (
    SELECT DISTINCT
        ls.법인번호_ENC,
        ls.매출액,
        ls.매출액_보정여부,
        ls.매출액영업이익률,
        ls.매출액영업이익률_보정여부,
        ls.매출액_결산기준일자,
        ls.이익률_결산기준일자,
        ls.매출액구간,
        ls.최종매출액구간
    FROM unique_company uc
    JOIN latest_sales ls
        ON uc.법인번호_ENC = ls.법인번호_ENC
    LEFT JOIN 상호 s
        ON ls.법인번호_ENC = s.법인번호_ENC
    WHERE ls.정부및공공기관구분 != '공공기관'
      AND (s.산학협력단여부 IS NULL OR s.산학협력단여부 = 0)
      AND ls.매출액 IS NOT NULL
),
RankedCompanies AS (
    SELECT 
        법인번호_ENC,
        매출액,
        매출액_보정여부,
        매출액영업이익률,
        매출액영업이익률_보정여부,
        매출액_결산기준일자,
        이익률_결산기준일자,
        매출액구간,
        최종매출액구간,
        DENSE_RANK() OVER (
            ORDER BY 매출액 DESC, 매출액영업이익률 DESC
        ) AS rank_num
    FROM target_companies
),
AvgPool AS (
    SELECT
        ls.매출액
    FROM unique_company uc
    JOIN latest_sales ls
        ON uc.법인번호_ENC = ls.법인번호_ENC
    LEFT JOIN 상호 s
        ON ls.법인번호_ENC = s.법인번호_ENC
    JOIN 기업 c
        ON ls.법인번호_ENC = c.법인번호_ENC
    WHERE c.정부및공공기관구분 != '공공기관'
      AND (s.산학협력단여부 IS NULL OR s.산학협력단여부 = 0)
      AND ls.매출액 IS NOT NULL
),
AvgSales AS (
    SELECT AVG(매출액) AS 평균매출액
    FROM AvgPool
),
AvgProfitRatioPool AS (
    SELECT
        CASE
            WHEN ls.매출액 IS NULL THEN NULL
            WHEN ls.매출액 = 0 THEN 0
            ELSE ls.매출액영업이익률
        END AS profit_ratio
    FROM unique_company uc
    JOIN latest_sales ls
        ON uc.법인번호_ENC = ls.법인번호_ENC
    LEFT JOIN 상호 s
        ON ls.법인번호_ENC = s.법인번호_ENC
    JOIN 기업 c
        ON ls.법인번호_ENC = c.법인번호_ENC
    WHERE c.정부및공공기관구분 != '공공기관'
      AND (s.산학협력단여부 IS NULL OR s.산학협력단여부 = 0)
      AND ls.매출액 IS NOT NULL
),
AvgProfitRatio AS (
    SELECT AVG(profit_ratio) AS 평균매출액영업이익률
    FROM AvgProfitRatioPool
)
SELECT 
    rc.법인번호_ENC,
    c.법인번호 AS 법인번호,
    GROUP_CONCAT(DISTINCT s.상호 ORDER BY s.상호 SEPARATOR ', ') AS 상호,
    rc.매출액,
    rc.매출액_보정여부,
    rc.매출액영업이익률,
    rc.매출액영업이익률_보정여부,
    rc.매출액구간,
    rc.rank_num,
    CASE 
        WHEN rc.법인번호_ENC = :applicant_corp_no
            THEN '신청기업'
        ELSE '경쟁기업'
    END AS 구분,
    avg.평균매출액,
    avgpr.평균매출액영업이익률
FROM RankedCompanies rc
LEFT JOIN 상호 s
    ON rc.법인번호_ENC = s.법인번호_ENC
LEFT JOIN 기업 c
    ON rc.법인번호_ENC = c.법인번호_ENC
CROSS JOIN AvgSales avg
CROSS JOIN AvgProfitRatio avgpr
GROUP BY 
    rc.법인번호_ENC,
    c.법인번호,
    rc.매출액,
    rc.매출액_보정여부,
    rc.매출액영업이익률,
    rc.매출액영업이익률_보정여부,
    rc.매출액구간,
    rc.rank_num,
    avg.평균매출액,
    avgpr.평균매출액영업이익률
ORDER BY rc.rank_num ASC;
"""


def q_fin_top11_by_subcat() -> str:
    """
    기술검색: 해당 소분류에 매핑된 기업들 중 2024년 매출액 TOP 11
    변수: :sel_subcat_code
    """
    return """
-- 기술검색: 해당 소분류에 매핑된 기업들 중 2024년 매출액 TOP 11
-- 변수: 소분류코드
-- 251215 추가: 매출액_보정여부, 매출액영업이익률_보정여부
-- 251217 추가: 법인번호
WITH patent_company AS (
    SELECT 
        ps.출원번호_ENC,
        MIN(cp.법인번호_ENC) AS 법인번호_ENC
    FROM 특허_국가과학기술표준분류체계 ps
    JOIN 기업_특허 cp
        ON cp.출원번호_ENC = ps.출원번호_ENC
    WHERE ps.소분류코드 = :sel_subcat_code -- 여기 소분류코드
    GROUP BY ps.출원번호_ENC
),
unique_company AS (
    SELECT DISTINCT 법인번호_ENC
    FROM patent_company
),
finance_2024 AS (
    SELECT *
    FROM 재무정보
    WHERE LEFT(결산기준일자, 4) = '2024'
),
latest_financial AS (
    SELECT 
        법인번호_ENC,
        MAX(결산기준일자) AS 최근결산기준일자
    FROM finance_2024
    GROUP BY 법인번호_ENC
),
profit_ratio_ranked AS (
    SELECT
        f.법인번호_ENC,
        f.매출액영업이익률,
        f.보정여부_매출액영업이익률 AS 매출액영업이익률_보정여부,
        ROW_NUMBER() OVER (
            PARTITION BY f.법인번호_ENC
            ORDER BY 
                (f.결산기준일자 = lf.최근결산기준일자 AND f.매출액영업이익률 IS NOT NULL) DESC,
                (f.매출액영업이익률 IS NOT NULL) DESC,
                f.결산기준일자 DESC
        ) AS rn
    FROM finance_2024 f
    JOIN latest_financial lf
        ON f.법인번호_ENC = lf.법인번호_ENC
),
selected_profit_ratio AS (
    SELECT 법인번호_ENC, 매출액영업이익률, 매출액영업이익률_보정여부
    FROM profit_ratio_ranked
    WHERE rn = 1
),
latest_sales AS (
    SELECT 
        f.법인번호_ENC,
        f.매출액,
        f.보정여부 AS 매출액_보정여부,
        CASE
            WHEN f.매출액 IS NULL OR f.매출액 = 0
                THEN NULL
            ELSE pr.매출액영업이익률
        END AS 매출액영업이익률,
        pr.매출액영업이익률_보정여부,
        f.매출액구간,
        CASE
            WHEN c.정부및공공기관구분 = '공공기관'
                THEN '50억원 이하'
            ELSE f.매출액구간
        END AS 최종매출액구간
    FROM finance_2024 f
    JOIN latest_financial lf
        ON f.법인번호_ENC = lf.법인번호_ENC 
       AND f.결산기준일자 = lf.최근결산기준일자
    LEFT JOIN selected_profit_ratio pr
        ON f.법인번호_ENC = pr.법인번호_ENC
    JOIN 기업 c 
        ON f.법인번호_ENC = c.법인번호_ENC
),
filtered_companies AS (
    SELECT DISTINCT
        ls.법인번호_ENC,
        ls.매출액,
        ls.매출액_보정여부,
        ls.매출액영업이익률,
        ls.매출액영업이익률_보정여부,
        ls.매출액구간,
        ls.최종매출액구간
    FROM unique_company uc
    JOIN latest_sales ls
        ON uc.법인번호_ENC = ls.법인번호_ENC
    LEFT JOIN 상호 s
        ON ls.법인번호_ENC = s.법인번호_ENC
    JOIN 기업 c
        ON ls.법인번호_ENC = c.법인번호_ENC
    WHERE c.정부및공공기관구분 != '공공기관'
      AND (s.산학협력단여부 IS NULL OR s.산학협력단여부 = 0)
      AND ls.매출액 IS NOT NULL
),
AvgPool AS (
    SELECT
        ls.매출액
    FROM unique_company uc
    JOIN latest_sales ls
        ON uc.법인번호_ENC = ls.법인번호_ENC
    LEFT JOIN 상호 s
        ON ls.법인번호_ENC = s.법인번호_ENC
    JOIN 기업 c
        ON ls.법인번호_ENC = c.법인번호_ENC
    WHERE c.정부및공공기관구분 != '공공기관'
      AND (s.산학협력단여부 IS NULL OR s.산학협력단여부 = 0)
      AND ls.매출액 IS NOT NULL
),
AvgSales AS (
    SELECT AVG(매출액) AS 평균매출액
    FROM AvgPool
),
AvgProfitRatioPool AS (
    SELECT
        CASE
            WHEN ls.매출액 IS NULL THEN NULL
            WHEN ls.매출액 = 0 THEN 0
            ELSE ls.매출액영업이익률
        END AS profit_ratio
    FROM unique_company uc
    JOIN latest_sales ls
        ON uc.법인번호_ENC = ls.법인번호_ENC
    LEFT JOIN 상호 s
        ON ls.법인번호_ENC = s.법인번호_ENC
    JOIN 기업 c
        ON ls.법인번호_ENC = c.법인번호_ENC
    WHERE c.정부및공공기관구분 != '공공기관'
      AND (s.산학협력단여부 IS NULL OR s.산학협력단여부 = 0)
      AND ls.매출액 IS NOT NULL
),
AvgProfitRatio AS (
    SELECT AVG(profit_ratio) AS 평균매출액영업이익률
    FROM AvgProfitRatioPool
),
RankedCompanies AS (
    SELECT
        fc.*,
        DENSE_RANK() OVER (
            ORDER BY 매출액 DESC, 매출액영업이익률 DESC
        ) AS rank_num
    FROM filtered_companies fc
)
SELECT
    rc.법인번호_ENC,
    c.법인번호 AS 법인번호,
    GROUP_CONCAT(DISTINCT s.상호 ORDER BY s.상호 SEPARATOR ', ') AS 상호,
    rc.매출액,
    rc.매출액_보정여부,
    rc.매출액영업이익률,
    rc.매출액영업이익률_보정여부,
    rc.매출액구간,
    rc.rank_num,
    avg.평균매출액,
    avgpr.평균매출액영업이익률
FROM RankedCompanies rc
LEFT JOIN 상호 s
    ON rc.법인번호_ENC = s.법인번호_ENC
LEFT JOIN 기업 c
    ON rc.법인번호_ENC = c.법인번호_ENC
CROSS JOIN AvgSales avg
CROSS JOIN AvgProfitRatio avgpr
WHERE rc.rank_num <= 11
GROUP BY 
    rc.법인번호_ENC,
    c.법인번호,
    rc.매출액,
    rc.매출액_보정여부,
    rc.매출액영업이익률,
    rc.매출액영업이익률_보정여부,
    rc.매출액구간,
    rc.rank_num,
    avg.평균매출액,
    avgpr.평균매출액영업이익률
ORDER BY rc.rank_num ASC;
"""


def q_growth_top10_by_sales_bucket() -> str:
    """
    성장률 우수기업: 특정 소분류 매핑 기업의 2022/2024 매출액 (성장률은 파이썬에서 계산)
    변수: :sel_subcat_code
    """
    return """
-- 성장률 우수기업을 위한 특정 소분류 매핑 기업의 2022년, 2024년 매출액
-- 변수: 소분류코드
-- 251215 추가: 2022년 매출액 보정여부, 2024년 매출액 보정여부
-- 251217 추가: 법인번호
WITH patent_company AS (
    SELECT 
        ps.출원번호_ENC,
        MIN(cp.법인번호_ENC) AS 법인번호_ENC
    FROM 특허_국가과학기술표준분류체계 ps
    LEFT JOIN 기업_특허 cp
        ON cp.출원번호_ENC = ps.출원번호_ENC
    WHERE ps.소분류코드 = :sel_subcat_code -- 여기 소분류코드
    GROUP BY ps.출원번호_ENC
),
latest_financial_date AS (
    SELECT
        fi.법인번호_ENC,
        LEFT(fi.결산기준일자, 4) AS 연도,
        MAX(fi.결산기준일자) AS 최신결산일자
    FROM 재무정보 fi
    WHERE LEFT(fi.결산기준일자, 4) IN ('2022', '2024')
    GROUP BY fi.법인번호_ENC, LEFT(fi.결산기준일자, 4)
),
latest_sales AS (
    SELECT 
        lf.법인번호_ENC,
        lf.연도,
        fi.매출액,
        fi.보정여부 AS 매출액_보정여부
    FROM latest_financial_date lf
    JOIN 재무정보 fi
      ON fi.법인번호_ENC = lf.법인번호_ENC
     AND fi.결산기준일자 = lf.최신결산일자
),
pivot_sales AS (
    SELECT
        법인번호_ENC,
        MAX(CASE WHEN 연도 = '2022' THEN 매출액 END) AS 매출액_2022,
        MAX(CASE WHEN 연도 = '2022' THEN 매출액_보정여부 END) AS 매출액_2022_보정여부,
        MAX(CASE WHEN 연도 = '2024' THEN 매출액 END) AS 매출액_2024,
        MAX(CASE WHEN 연도 = '2024' THEN 매출액_보정여부 END) AS 매출액_2024_보정여부
    FROM latest_sales
    GROUP BY 법인번호_ENC
    HAVING 
    MAX(CASE WHEN 연도 = '2022' THEN 매출액 END) > 1000
    AND MAX(CASE WHEN 연도 = '2024' THEN 매출액 END) IS NOT NULL
),
final_result AS (
    SELECT DISTINCT   -- 기업 기준 중복 제거
        p.법인번호_ENC,
        p.매출액_2022,
        p.매출액_2022_보정여부,
        p.매출액_2024,
        p.매출액_2024_보정여부,
        CASE
            WHEN p.매출액_2024 <= 10000 THEN '100억원 이하'
            ELSE '100억원 초과'
        END AS 매출100억구분
    FROM pivot_sales p
    JOIN patent_company tk
        ON p.법인번호_ENC = tk.법인번호_ENC
    JOIN 기업 c
        ON p.법인번호_ENC = c.법인번호_ENC
    LEFT JOIN 상호 s
        ON p.법인번호_ENC = s.법인번호_ENC
    WHERE c.정부및공공기관구분 != '공공기관'
      AND (s.산학협력단여부 IS NULL OR s.산학협력단여부 = 0)
)
SELECT
    fr.법인번호_ENC,
    c.법인번호 AS 법인번호,
    GROUP_CONCAT(DISTINCT s.상호 ORDER BY s.상호 SEPARATOR ', ') AS 상호,
    fr.매출액_2022,
    fr.매출액_2022_보정여부,
    fr.매출액_2024,
    fr.매출액_2024_보정여부,
    fr.매출100억구분
FROM final_result fr
LEFT JOIN 상호 s
    ON fr.법인번호_ENC = s.법인번호_ENC
LEFT JOIN 기업 c
    ON fr.법인번호_ENC = c.법인번호_ENC
GROUP BY
    fr.법인번호_ENC,
    c.법인번호,
    fr.매출액_2022,
    fr.매출액_2022_보정여부,
    fr.매출액_2024,
    fr.매출액_2024_보정여부,
    fr.매출100억구분
ORDER BY fr.매출100억구분, fr.매출액_2024 DESC;
"""


def q_target_growth_for_subcat() -> str:
    """
    신청기업 3년성장률 계산용: 신청기업의 2022/2024 매출액
    변수: :applicant_corp_no
    """
    return """
-- 신청기업의 3년성장률을 위한 신청기업의 2022년, 2024년 매출액
-- 변수: 신청기업의 법인번호_ENC
-- 251215 추가: 2022년 매출액 보정여부, 2024년 매출액 보정여부
-- 251217 추가: 법인번호
WITH latest_financial_date AS (
    SELECT
        fi.법인번호_ENC,
        LEFT(fi.결산기준일자, 4) AS 연도,
        MAX(fi.결산기준일자) AS 최신결산일자
    FROM 재무정보 fi
    WHERE fi.법인번호_ENC = :applicant_corp_no
      AND LEFT(fi.결산기준일자, 4) IN ('2022', '2024')
    GROUP BY fi.법인번호_ENC, LEFT(fi.결산기준일자, 4)
),
latest_sales AS (
    SELECT 
        lf.법인번호_ENC,
        lf.연도,
        fi.매출액,
        fi.보정여부
    FROM latest_financial_date lf
    JOIN 재무정보 fi
      ON fi.법인번호_ENC = lf.법인번호_ENC
     AND fi.결산기준일자 = lf.최신결산일자
)
SELECT
    ls.법인번호_ENC,
    c.법인번호 AS 법인번호,
    GROUP_CONCAT(DISTINCT s.상호 ORDER BY s.상호 SEPARATOR ', ') AS 상호,
    MAX(CASE WHEN 연도 = '2022' THEN 매출액 END) AS 매출액_2022,
    MAX(CASE WHEN 연도 = '2022' THEN 보정여부 END) AS 매출액_2022_보정여부,
    MAX(CASE WHEN 연도 = '2024' THEN 매출액 END) AS 매출액_2024,
    MAX(CASE WHEN 연도 = '2024' THEN 보정여부 END) AS 매출액_2024_보정여부
FROM latest_sales ls
LEFT JOIN 상호 s
    ON ls.법인번호_ENC = s.법인번호_ENC
LEFT JOIN 기업 c
        ON ls.법인번호_ENC = c.법인번호_ENC
GROUP BY ls.법인번호_ENC, c.법인번호;
"""


# (호환용) 다른 코드에서 호출되면 q_growth_top10_by_sales_bucket() 재사용
def q_avg_growth_for_subcat() -> str:
    return q_growth_top10_by_sales_bucket()


# ---------------------------------------------------
# 기술부문
# ---------------------------------------------------

def q_view_top_cited() -> str:
    """
    소분류 내에서 피인용지수 기준 상위 10개 특허(우수기술)를 조회하는 SQL을 반환한다.

    설명
    ----
    * ``patent_company``:
      선택된 소분류코드 내의 특허를 기준으로 대표 기업(법인번호_ENC)을 매핑한다.
    * ``company_name``:
      기업별 상호를 집계한다. 산학협력단여부가 0인 상호만 사용한다.
    * ``citation``:
      피인용정보 테이블에서 선행특허(출원번호선행_ENC) 기준으로 피인용횟수를 집계한다.
    * 본문 SELECT 에서는 공공기관(``기업.정부및공공기관구분 = '공공기관'``)을 제외하고
      피인용지수를 계산한 뒤, 피인용지수 기준 상위 10건을 조회한다.
    * 피인용지수는 다음과 같이 계산한다.

    바인딩 파라미터
    --------------
    * ``:sel_subcat_code``: 선택 소분류 코드.

    반환 컬럼
    --------
    * ``특허제목``: 특허 제목 (NULL 인 경우 '-' 로 대체)
    * ``출원번호``: 특허 출원번호
    * ``출원일``: 특허 출원일자
    * ``기업명``: 특허 보유 기업명(상호, NULL 인 경우 '-' 로 대체)
    * ``피인용횟수``: 해당 특허의 총 피인용 횟수
    * ``피인용지수``: 연도 보정된 피인용 지수

    반환값
    ------
    :return: 피인용지수, 피인용횟수 기준 우수기술 Top10 조회용 SQL 문자열.
    :rtype: str
    """
    return """
        WITH
        patent_company AS (
            SELECT p.출원번호_ENC
                , p.출원번호
                , p.출원일자
                , p.제목
                , MIN(cp.법인번호_ENC) AS 법인번호_ENC
            FROM 특허 AS p
            JOIN 특허_국가과학기술표준분류체계 AS ps
                ON p.출원번호_ENC = ps.출원번호_ENC
            LEFT JOIN 기업_특허 cp
                ON cp.출원번호_ENC = ps.출원번호_ENC
            WHERE ps.소분류코드 = :sel_subcat_code
            GROUP BY ps.출원번호_ENC
        ),
        company_name AS (
            SELECT s.법인번호_ENC
                , GROUP_CONCAT(
                    DISTINCT s.상호
                    ORDER BY s.상호 ASC
                    SEPARATOR ', '
                  ) AS 상호
            FROM 상호 AS s
            WHERE s.산학협력단여부 = 0 -- 상호에 산학협력단이 포함되어 있지 않은 것
            GROUP BY s.법인번호_ENC
        ),
        citation AS (
            SELECT ct.출원번호선행_ENC AS 출원번호_ENC
                , COUNT(*) AS 피인용횟수
            FROM 피인용정보 AS ct
            GROUP BY ct.출원번호선행_ENC
        )
        SELECT 
            IFNULL(patent_company.제목, '-') AS 특허제목,
            patent_company.출원번호 AS 출원번호,
            patent_company.출원일자 AS 출원일,
            IFNULL(company_name.상호, '-') AS 기업명,
            IFNULL(citation.피인용횟수, 0) AS 피인용횟수,
            IFNULL(
                ROUND(
                    CASE
                        WHEN (2024 - YEAR(patent_company.출원일자) + 1) <= 0
                            THEN 0
                        ELSE COALESCE(citation.피인용횟수, 0)
                             / (2024 - YEAR(patent_company.출원일자) + 1)
                    END
                    , 2
                ),
                0
            ) AS 피인용지수
        FROM patent_company
        JOIN 기업
            ON patent_company.법인번호_ENC = 기업.법인번호_ENC
        JOIN company_name
            ON 기업.법인번호_ENC = company_name.법인번호_ENC
        LEFT JOIN citation
            ON patent_company.출원번호_ENC = citation.출원번호_ENC
        WHERE 기업.정부및공공기관구분 != '공공기관'
        ORDER BY 피인용지수 DESC
        LIMIT 10
        ;
    """


def q_top_holder() -> str:
    """
    소분류 내 최다 특허 보유 기업 상위 10개와
    나머지 기업들을 '그 외' 그룹으로 묶어 집계하는 SQL을 반환한다.

    설명
    ----
    * patent_company: 소분류 내 특허와 대표 기업 매핑.
    * company_name: 기업명 집계.
    * latest_financial / corp_sales: 최신 매출액 산출.
    * company_patent_counts: 기업별 보유 특허건수 집계.
    * company_info: 매출액·설립년도 포함 기업 정보.
    * RankedCompanies: 특허건수 기준 랭킹 산출 후 상위 10개 + 그 외 그룹화.

    바인딩 파라미터
    --------------
    * ``:sel_subcat_code``: 선택 소분류 코드.

    반환 컬럼
    --------
    * 법인번호_ENC (상위 10개 또는 '그 외')
    * 기업명 (상위 10개 또는 '그 외')
    * 매출액 (그룹별 합산)
    * 설립년도 (상위 10개는 개별, 그 외는 '-')
    * 보유특허건수 (그룹별 합산)
    * sort_key (표시 순서를 위한 내부 컬럼)

    반환값
    ------
    :return: 최다 특허 보유 기업 및 기타 그룹 조회용 SQL 문자열.
    :rtype: str
    """
    return """
    -- 최다특허보유기업
    WITH
    patent_company AS (
        SELECT p.출원번호_ENC
            , p.출원번호
            , p.출원일자
            , MIN(cp.법인번호_ENC) AS 법인번호_ENC
        FROM 특허 AS p
        JOIN 특허_국가과학기술표준분류체계 AS ps
        ON p.출원번호_ENC = ps.출원번호_ENC
        LEFT JOIN 기업_특허 cp
            ON cp.출원번호_ENC = ps.출원번호_ENC
        WHERE ps.소분류코드 = :sel_subcat_code
        GROUP BY ps.출원번호_ENC
    ),
    company_name AS (
        SELECT s.법인번호_ENC
            , GROUP_CONCAT(DISTINCT s.상호 ORDER BY s.상호 ASC SEPARATOR ', ') AS 기업명
        FROM 상호 AS s
        WHERE s.산학협력단여부 = 0 -- 상호에 산학협력단이 아닌것
        GROUP BY s.법인번호_ENC
    ),
    latest_financial AS (
        SELECT fi.법인번호_ENC
            , MAX(fi.결산기준일자) AS 최근결산기준일자
        FROM 재무정보 fi
        WHERE LEFT(fi.결산기준일자, 4) = '2024'
        AND fi.매출액 IS NOT NULL
        GROUP BY fi.법인번호_ENC
    ),
    corp_sales AS ( 
        SELECT 
            fi.법인번호_ENC,
            fi.매출액
        FROM 재무정보 fi
        JOIN latest_financial rc
            ON fi.법인번호_ENC = rc.법인번호_ENC
        AND fi.결산기준일자 = rc.최근결산기준일자
    ),
    company_patent_counts AS (
        SELECT patent_company.법인번호_ENC
            , COUNT(*) AS 보유특허건수
        FROM patent_company
        JOIN 기업 AS c
            ON patent_company.법인번호_ENC = c.법인번호_ENC
        WHERE patent_company.법인번호_ENC IS NOT NULL
        AND c.정부및공공기관구분 != '공공기관'
        GROUP BY patent_company.법인번호_ENC
    ),
    company_info AS (
        SELECT c.법인번호_ENC
            , IFNULL(company_name.기업명, '-') AS 기업명
            , corp_sales.매출액 AS 매출액
            , c.설립년도
        FROM 기업 AS c
        JOIN company_name
            ON c.법인번호_ENC = company_name.법인번호_ENC
        LEFT JOIN latest_financial
            ON c.법인번호_ENC = latest_financial.법인번호_ENC
        LEFT JOIN corp_sales
        ON c.법인번호_ENC = corp_sales.법인번호_ENC
    ),
    RankedCompanies AS (
        SELECT cpc.법인번호_ENC
            , ci.기업명
            , ci.매출액
            , ci.설립년도
            , cpc.보유특허건수
            , ROW_NUMBER() OVER (ORDER BY cpc.보유특허건수 DESC) AS rnk
        FROM company_patent_counts AS cpc
        JOIN company_info AS ci
            ON cpc.법인번호_ENC = ci.법인번호_ENC
    )
    SELECT CASE WHEN RankedCompanies.rnk <= 10 THEN RankedCompanies.법인번호_ENC ELSE '그 외' END AS 법인번호_ENC
        , CASE WHEN RankedCompanies.rnk <= 10 THEN RankedCompanies.기업명        ELSE '그 외' END AS 기업명
        , SUM(RankedCompanies.매출액) AS 매출액
        , CASE WHEN RankedCompanies.rnk <= 10 THEN RankedCompanies.설립년도     ELSE '-'   END AS 설립년도
        , SUM(RankedCompanies.보유특허건수) AS 보유특허건수
        , MIN(CASE WHEN RankedCompanies.rnk <= 10 THEN RankedCompanies.rnk ELSE 11 END) AS sort_key
    FROM RankedCompanies
    GROUP BY CASE WHEN RankedCompanies.rnk <= 10 THEN RankedCompanies.법인번호_ENC ELSE '그 외' END
            , CASE WHEN RankedCompanies.rnk <= 10 THEN RankedCompanies.기업명        ELSE '그 외' END
            , CASE WHEN RankedCompanies.rnk <= 10 THEN RankedCompanies.설립년도     ELSE '-'   END
    ORDER BY sort_key ASC
    ;
    """


def q_competitor_group() -> str:
    """
    소분류 내에서 신청기업과 경쟁관계에 있는 기업군을 도출하고,
    구간별 특허 등록건수를 함께 조회하는 SQL을 반환한다.

    설명
    ----
    * patent_company: 소분류별 특허와 대표 기업 매핑.
    * target_companies: 대상 소분류에 속한 기업 목록.
    * corp_sales: 기업별 최신 매출액.
    * company_name: 기업명 집계.
    * RankedCompanies:
        - sales_rank: 매출 기준 순위.
        - unique_rank: 매출+기업명 기준 고유 순위.
    * TargetInfo: 신청기업의 unique_rank, sales_rank.
    * FinalTargetList:
        - 신청기업이 Top10이면 상위 10개 기업 + 신청기업 중심.
        - Top10 밖이면 신청기업 + 더 상위인 기업들 중 10개.
    * PatentCounts: 기업별 2년 단위 특허 등록건수 및 합계.

    바인딩 파라미터
    --------------
    * ``:sel_subcat_code``: 선택 소분류 코드.
    * ``:applicant_corp_no``: 신청 기업 법인번호_ENC.

    반환 컬럼
    --------
    * 법인번호_ENC
    * 설립년도
    * 순위 (sales_rank)
    * 기업명
    * 매출액 (최신매출액)
    * 구분 (신청기업 / 경쟁기업)
    * 2015~2016년 특허 등록건수
    * 2017~2018년 특허 등록건수
    * 2019~2020년 특허 등록건수
    * 2021~2022년 특허 등록건수
    * 2023~2024년 특허 등록건수
    * 특허 등록건수 합계

    반환값
    ------
    :return: 신청기업 기준 경쟁기업군 및 특허 실적 조회용 SQL 문자열.
    :rtype: str
    """
    return """
    -- 경쟁기업군
    WITH 
    patent_company AS (
        SELECT ps.출원번호_ENC
            , MIN(cp.법인번호_ENC) AS 법인번호_ENC
        FROM 특허_국가과학기술표준분류체계 AS ps
        JOIN 기업_특허 cp 
        ON ps.출원번호_ENC = cp.출원번호_ENC
        WHERE ps.소분류코드 = :sel_subcat_code 
        GROUP BY ps.출원번호_ENC
    ),
    target_companies AS (
        SELECT DISTINCT 법인번호_ENC 
        FROM patent_company
    ),
    corp_sales AS (
        SELECT f.법인번호_ENC
            , f.매출액
        FROM target_companies
        JOIN 재무정보 f ON target_companies.법인번호_ENC = f.법인번호_ENC
        WHERE f.결산기준일자 = (
            SELECT MAX(sub_f.결산기준일자)
            FROM 재무정보 sub_f
            WHERE sub_f.법인번호_ENC = f.법인번호_ENC
            AND LEFT(sub_f.결산기준일자, 4) = '2024'
            AND sub_f.매출액 IS NOT NULL
        )
    ),
    company_name AS (
        SELECT s.법인번호_ENC
            , GROUP_CONCAT(DISTINCT s.상호 ORDER BY s.상호 ASC SEPARATOR ', ') AS 기업명
        FROM target_companies tc
        JOIN 상호 s 
            ON tc.법인번호_ENC = s.법인번호_ENC
        WHERE s.산학협력단여부 = 0 -- 상호에 산학협력단이 포함되어 있지 않은 것
        GROUP BY s.법인번호_ENC
    ),
    RankedCompanies AS (
        SELECT tc.법인번호_ENC
            , cn.기업명
            , c.설립년도
            , IFNULL(cs.매출액, 0) AS 최신매출액
            , c.정부및공공기관구분
            , DENSE_RANK() OVER (ORDER BY IFNULL(cs.매출액, 0) DESC) AS sales_rank
            , ROW_NUMBER() OVER (ORDER BY IFNULL(cs.매출액, 0) DESC, cn.기업명 ASC) AS unique_rank
        FROM target_companies tc
        LEFT JOIN 기업 c ON tc.법인번호_ENC = c.법인번호_ENC
        JOIN company_name cn ON tc.법인번호_ENC = cn.법인번호_ENC
        LEFT JOIN corp_sales cs ON tc.법인번호_ENC = cs.법인번호_ENC
        WHERE c.정부및공공기관구분 != '공공기관'
    ),
    TargetInfo AS (
        SELECT unique_rank, sales_rank
        FROM RankedCompanies
        WHERE 법인번호_ENC = :applicant_corp_no
        LIMIT 1
    ),
    FinalTargetList AS (
        SELECT r.*
        FROM RankedCompanies r
        JOIN TargetInfo t ON 1=1
        WHERE t.unique_rank <= 10 AND r.unique_rank <= 11
        UNION ALL
        SELECT r.*
        FROM RankedCompanies r
        JOIN TargetInfo t ON 1=1
        WHERE t.unique_rank > 10 AND r.법인번호_ENC = :applicant_corp_no
        UNION ALL
        SELECT * FROM (
            SELECT r.*
            FROM RankedCompanies r
            JOIN TargetInfo t ON 1=1
            WHERE t.unique_rank > 10 
            AND r.sales_rank < t.sales_rank
            ORDER BY r.unique_rank DESC
            LIMIT 10
        ) AS sub
    ),
    PatentCounts AS (
        SELECT tp.법인번호_ENC
            , COUNT(CASE WHEN YEAR(p.등록일자) BETWEEN 2015 AND 2016 THEN 1 END) AS cnt_15_16
            , COUNT(CASE WHEN YEAR(p.등록일자) BETWEEN 2017 AND 2018 THEN 1 END) AS cnt_17_18
            , COUNT(CASE WHEN YEAR(p.등록일자) BETWEEN 2019 AND 2020 THEN 1 END) AS cnt_19_20
            , COUNT(CASE WHEN YEAR(p.등록일자) BETWEEN 2021 AND 2022 THEN 1 END) AS cnt_21_22
            , COUNT(CASE WHEN YEAR(p.등록일자) BETWEEN 2023 AND 2024 THEN 1 END) AS cnt_23_24
            , COUNT(CASE WHEN YEAR(p.등록일자) BETWEEN 2015 AND 2024 THEN 1 END) AS cnt_total
        FROM FinalTargetList ft
        JOIN patent_company tp ON ft.법인번호_ENC = tp.법인번호_ENC
        JOIN 특허 p ON tp.출원번호_ENC = p.출원번호_ENC
        GROUP BY tp.법인번호_ENC
    )
    SELECT ft.법인번호_ENC
        , ft.설립년도
        , ft.sales_rank AS '순위'
        , ft.기업명
        , ft.최신매출액 AS 매출액
        , CASE WHEN ft.법인번호_ENC = :applicant_corp_no THEN '신청기업' ELSE '경쟁기업' END AS '구분'
        , IFNULL(pc.cnt_15_16, 0) AS '2015~2016년 특허 등록건수'
        , IFNULL(pc.cnt_17_18, 0) AS '2017~2018년 특허 등록건수'
        , IFNULL(pc.cnt_19_20, 0) AS '2019~2020년 특허 등록건수'
        , IFNULL(pc.cnt_21_22, 0) AS '2021~2022년 특허 등록건수'
        , IFNULL(pc.cnt_23_24, 0) AS '2023~2024년 특허 등록건수'
        , IFNULL(pc.cnt_total, 0) AS '특허 등록건수 합계'
    FROM FinalTargetList ft
    LEFT JOIN PatentCounts pc ON ft.법인번호_ENC = pc.법인번호_ENC
    ORDER BY ft.sales_rank ASC, ft.unique_rank ASC
    ;
    """


def q_emerging_by_mid() -> str:
    """
    선택 소분류가 속한 중분류 코드 기준으로,
    하위 소분류별 특허 등록건수 추이를 조회하는 SQL을 반환한다.

    설명
    ----
    * sc_patent: 선택 소분류의 중분류코드에 속한 특허/소분류 정보.
    * cp_ranked: 특허별 대표 기업 매핑.
    * CorpSanhakFlag: 기업별 산학협력단 여부 플래그.
    * PatentMstDerived: 공공기관/산학 제외한 특허 정보.
    * 최종적으로 소분류별·구간별 특허 등록건수를 집계한다.

    바인딩 파라미터
    --------------
    * ``:sel_subcat_code``: 선택 소분류 코드 (앞 4자리 = 중분류코드).

    반환 컬럼
    --------
    * 중분류코드
    * 중분류명
    * 소분류코드
    * 소분류명
    * 2015~2016년 특허 등록건수
    * 2017~2018년 특허 등록건수
    * 2019~2020년 특허 등록건수
    * 2021~2022년 특허 등록건수
    * 2023~2024년 특허 등록건수
    * 특허 등록건수 합계 (2015~2024)

    반환값
    ------
    :return: 중분류 내 이머징 기술(소분류별 특허 추이) 조회용 SQL 문자열.
    :rtype: str
    """
    return """
      -- 이머징기술
      WITH
      sc_patent AS (
          SELECT p.출원번호_ENC
              , YEAR(p.등록일자) AS 특허등록연도
              , kst.중분류코드
              , kst.중분류명_kr AS 중분류명
              , ps.소분류코드
              , kst.소분류명_kr AS 소분류명
            FROM 특허 AS p
            JOIN 특허_국가과학기술표준분류체계 AS ps
              ON p.출원번호_ENC = ps.출원번호_ENC
            JOIN 국가과학기술표준분류체계 AS kst
              ON ps.소분류코드 = kst.소분류코드
          WHERE kst.중분류코드 = SUBSTRING(:sel_subcat_code, 1, 4)
      ),
      cp_ranked AS (
          SELECT sp.출원번호_ENC
              , MIN(cp2.법인번호_ENC) AS 법인번호_ENC
            FROM sc_patent AS sp
            LEFT JOIN 기업_특허 AS cp2
              ON cp2.출원번호_ENC = sp.출원번호_ENC
          GROUP BY sp.출원번호_ENC
      ),
      CorpSanhakFlag AS (
          SELECT 
              s.법인번호_ENC,
              MAX(s.산학협력단여부) AS 산학협력단여부   -- 1이 하나라도 있으면 1
          FROM 상호 s
          GROUP BY s.법인번호_ENC
      ),
      PatentMstDerived AS (
          SELECT sp.출원번호_ENC
              , sp.특허등록연도
              , sp.중분류코드
              , sp.중분류명
              , sp.소분류코드
              , sp.소분류명
              , c.정부및공공기관구분
              , CorpSanhakFlag.산학협력단여부
            FROM sc_patent AS sp
            LEFT JOIN cp_ranked AS cpr
              ON sp.출원번호_ENC = cpr.출원번호_ENC
            LEFT JOIN 기업 AS c
              ON cpr.법인번호_ENC = c.법인번호_ENC
            JOIN CorpSanhakFlag 
              ON c.법인번호_ENC = CorpSanhakFlag.법인번호_ENC
      )
      SELECT PatentMstDerived.중분류코드
          , PatentMstDerived.중분류명
          , PatentMstDerived.소분류코드
          , PatentMstDerived.소분류명
          , COUNT(CASE WHEN PatentMstDerived.특허등록연도 BETWEEN 2015 AND 2016 THEN 1 END) AS '2015~2016년 특허 등록건수'
          , COUNT(CASE WHEN PatentMstDerived.특허등록연도 BETWEEN 2017 AND 2018 THEN 1 END) AS '2017~2018년 특허 등록건수'
          , COUNT(CASE WHEN PatentMstDerived.특허등록연도 BETWEEN 2019 AND 2020 THEN 1 END) AS '2019~2020년 특허 등록건수'
          , COUNT(CASE WHEN PatentMstDerived.특허등록연도 BETWEEN 2021 AND 2022 THEN 1 END) AS '2021~2022년 특허 등록건수'
          , COUNT(CASE WHEN PatentMstDerived.특허등록연도 BETWEEN 2023 AND 2024 THEN 1 END) AS '2023~2024년 특허 등록건수'
          , COUNT(CASE WHEN PatentMstDerived.특허등록연도 BETWEEN 2015 AND 2024 THEN 1 END) AS '특허 등록건수 합계'
      FROM PatentMstDerived
      WHERE (PatentMstDerived.정부및공공기관구분 IS NULL OR PatentMstDerived.정부및공공기관구분 != '공공기관')
        AND (PatentMstDerived.산학협력단여부 IS NULL OR PatentMstDerived.산학협력단여부 = 0) 
      GROUP BY PatentMstDerived.중분류코드
            , PatentMstDerived.중분류명
            , PatentMstDerived.소분류코드
            , PatentMstDerived.소분류명
      ;
    """


# ---------------------------------------------------
# R&D부문
# ---------------------------------------------------
def q_wordcloud2() -> str:
    """
    소분류 내 최근 3년(2022~2024) 명세서(해결과제) 텍스트를
    워드클라우드 생성을 위해 조회하는 SQL을 반환한다.

    설명
    ----
    * 명세서(a)·특허(b)·특허_국가과학기술표준분류체계(c)를 조인하여
      해결과제 명세서 텍스트와 등록일자, 소분류코드를 함께 조회한다.

    바인딩 파라미터
    --------------
    * ``:sel_subcat_code``: 선택 소분류 코드.

    반환 컬럼
    --------
    * 출원번호_ENC
    * 명세서구분
    * 명세서 (텍스트)
    * 등록일자
    * 소분류코드

    반환값
    ------
    :return: 워드클라우드용 명세서/특허 텍스트 조회용 SQL 문자열.
    :rtype: str
    """
    return"""
    SELECT a.출원번호_ENC, a.명세서구분, a.명세서, b.등록일자, c.소분류코드
    FROM 명세서 a
    JOIN 특허 b
        ON a.출원번호_ENC = b.출원번호_ENC
    JOIN 특허_국가과학기술표준분류체계 c
        ON b.출원번호_ENC = c.출원번호_ENC
    WHERE a.명세서구분 = '해결과제'
    AND c.소분류코드 = :sel_subcat_code
    AND YEAR(b.등록일자) IN (2022, 2023, 2024)
    ;
    """


def q_wordcloud() -> str:
    """
    소분류 내 최근 3년(2022~2024) 명세서(해결과제) 텍스트를
    워드클라우드 생성을 위해 조회하는 SQL을 반환한다.

    설명
    ----
    * 명세서(a)·특허(b)·특허_국가과학기술표준분류체계(c)를 조인하여
      해결과제 명세서 텍스트와 등록일자, 소분류코드를 함께 조회한다.

    바인딩 파라미터
    --------------
    * ``:sel_subcat_code``: 선택 소분류 코드.

    반환 컬럼
    --------
    * 출원번호_ENC
    * 명세서구분
    * 명세서 (텍스트)
    * 등록일자
    * 소분류코드

    반환값
    ------
    :return: 워드클라우드용 명세서/특허 텍스트 조회용 SQL 문자열.
    :rtype: str
    """
    return"""
    SELECT a.출원번호_ENC, a.명세서구분, a.명세서, b.등록일자, c.소분류코드
    FROM 명세서 a
    JOIN 특허 b
        ON a.출원번호_ENC = b.출원번호_ENC
    JOIN 특허_국가과학기술표준분류체계 c
        ON b.출원번호_ENC = c.출원번호_ENC
    WHERE a.명세서구분 IN ('배경기술', '해결과제') 
    AND c.소분류코드 = :sel_subcat_code
    AND YEAR(b.등록일자) IN (2022, 2023, 2024);
    """


def q_rnd_gov_top() -> str:
    """
    R&D 전문기관 현황 - 공공기관(정부출연연구소) Top10 조회용 SQL을 반환한다.

    설명
    ----
    * patent_company: 선택 소분류 내 공공기관 특허/기업 매핑.
    * company_name: 기업명 집계.
    * PatentMstDerived: 공공기관만 필터링.
    * TotalPatentCount: 해당 공공기관의 전체 특허보유수 집계.

    바인딩 파라미터
    --------------
    * ``:sel_subcat_code``: 선택 소분류 코드.

    반환 컬럼
    --------
    * 소분류코드
    * 소분류명
    * 법인번호_ENC
    * 정부및공공기관구분
    * 기업명
    * 전체특허보유수 (해당 기관 전체 특허수)
    * 특허보유수 (선택 소분류 내 특허수)

    반환값
    ------
    :return: 공공기관 R&D 전문기관 Top10 조회용 SQL 문자열.
    :rtype: str
    """
    return """
    WITH
    FilteredData AS (
        SELECT p.출원번호_ENC
            , p.출원번호
            , p.제목
            , ps.소분류코드
            , kst.소분류명_kr AS 소분류명
            , MIN(cp.법인번호_ENC) AS 법인번호_ENC -- 대표 법인번호
        FROM 특허_국가과학기술표준분류체계 AS ps
        JOIN 특허 AS p 
        ON ps.출원번호_ENC = p.출원번호_ENC
        JOIN 국가과학기술표준분류체계 AS kst 
        ON ps.소분류코드 = kst.소분류코드
        LEFT JOIN 기업_특허 AS cp 
        ON cp.출원번호_ENC = ps.출원번호_ENC
        JOIN 기업 AS c 
        ON cp.법인번호_ENC = c.법인번호_ENC
    WHERE ps.소분류코드 = :sel_subcat_code
        AND c.정부및공공기관구분 = '공공기관' 
    GROUP BY p.출원번호_ENC
    ),
    company_name AS (
        SELECT s.법인번호_ENC
            , GROUP_CONCAT(DISTINCT s.상호 ORDER BY s.상호 ASC SEPARATOR ', ') AS 기업명
        FROM 상호 AS s
        WHERE s.법인번호_ENC IN (SELECT DISTINCT 법인번호_ENC FROM FilteredData)
        GROUP BY s.법인번호_ENC
    ),
    TargetCorpList AS (
        SELECT DISTINCT 법인번호_ENC FROM FilteredData
    ),
    TotalPatentCount AS (
        SELECT target.법인번호_ENC
            , COUNT(cp.출원번호_ENC) AS 전체특허보유수
        FROM TargetCorpList AS target
        JOIN 기업_특허 cp
            ON target.법인번호_ENC = cp.법인번호_ENC
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
        GROUP BY target.법인번호_ENC
    )
    SELECT
        fd.소분류코드,
        fd.소분류명,
        fd.법인번호_ENC,
        '공공기관' AS 정부및공공기관구분,
        IFNULL(cn.기업명, '-') AS 기업명,
        IFNULL(tpc.전체특허보유수, 0) AS 전체특허보유수,
        COUNT(fd.출원번호_ENC) AS 특허보유수
    FROM FilteredData fd
    LEFT JOIN company_name cn
    ON fd.법인번호_ENC = cn.법인번호_ENC
    LEFT JOIN TotalPatentCount tpc
    ON fd.법인번호_ENC = tpc.법인번호_ENC
    GROUP BY
        fd.소분류코드,
        fd.소분류명,
        fd.법인번호_ENC,
        cn.기업명,
        tpc.전체특허보유수
    ORDER BY 특허보유수 DESC
    LIMIT 10
    ;
    """


def q_rnd_uni_top() -> str:
    """
    R&D 전문기관 현황 - 산학협력단 Top10 조회용 SQL을 반환한다.

    설명
    ----
    * patent_company: 선택 소분류 내 산학협력단 특허/기업 매핑.
    * company_name: 산학협력단 기업명 집계.
    * PatentMstDerived: 소분류/기관별 특허 레코드.
    * TotalPatentCount: 산학협력단의 전체 특허보유수 집계.

    바인딩 파라미터
    --------------
    * ``:sel_subcat_code``: 선택 소분류 코드.

    반환 컬럼
    --------
    * 소분류코드
    * 소분류명
    * 법인번호_ENC
    * 기업명
    * 전체특허보유수 (해당 기관 전체 특허수)
    * 특허보유수 (선택 소분류 내 특허수)

    반환값
    ------
    :return: 산학협력단 R&D 전문기관 Top10 조회용 SQL 문자열.
    :rtype: str
    """
    return """
    WITH
    FilteredData AS (
        SELECT p.출원번호_ENC
            , p.출원번호
            , p.제목
            , ps.소분류코드
            , kst.소분류명_kr AS 소분류명
            , MIN(cp.법인번호_ENC) AS 법인번호_ENC 
        FROM 특허_국가과학기술표준분류체계 AS ps
        JOIN 특허 AS p 
        ON ps.출원번호_ENC = p.출원번호_ENC
        JOIN 국가과학기술표준분류체계 AS kst 
        ON ps.소분류코드 = kst.소분류코드
        LEFT JOIN 기업_특허 AS cp 
        ON cp.출원번호_ENC = ps.출원번호_ENC
        JOIN 상호 AS s
        ON cp.법인번호_ENC = s.법인번호_ENC
    WHERE ps.소분류코드 = :sel_subcat_code
        AND s.산학협력단여부 = 1      
    GROUP BY p.출원번호_ENC
    ),
    TargetCorpList AS (
        SELECT DISTINCT 법인번호_ENC FROM FilteredData
    ),
    company_name AS (
        SELECT s.법인번호_ENC
            , GROUP_CONCAT(DISTINCT s.상호 ORDER BY s.상호 ASC SEPARATOR ', ') AS 기업명
        FROM 상호 AS s
        WHERE s.법인번호_ENC IN (SELECT 법인번호_ENC FROM TargetCorpList)
        GROUP BY s.법인번호_ENC
    ),
    TotalPatentCount AS (
        SELECT target.법인번호_ENC
            , COUNT(DISTINCT cp.출원번호_ENC) AS 전체특허보유수
        FROM TargetCorpList AS target
        JOIN 기업_특허 cp
            ON target.법인번호_ENC = cp.법인번호_ENC
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
        GROUP BY target.법인번호_ENC
    )
    SELECT
        fd.소분류코드,
        fd.소분류명,
        fd.법인번호_ENC,
        IFNULL(cn.기업명, '-') AS 기업명,
        IFNULL(tpc.전체특허보유수, 0) AS 전체특허보유수,
        COUNT(fd.출원번호_ENC) AS 특허보유수
    FROM FilteredData fd
    LEFT JOIN company_name cn
    ON fd.법인번호_ENC = cn.법인번호_ENC
    LEFT JOIN TotalPatentCount tpc
    ON fd.법인번호_ENC = tpc.법인번호_ENC
    GROUP BY
        fd.소분류코드,
        fd.소분류명,
        fd.법인번호_ENC,
        cn.기업명,
        tpc.전체특허보유수
    ORDER BY 특허보유수 DESC
    LIMIT 10
    ;
    """

def q_iris_1() -> str:
    """
    """
    return"""
    SELECT b.소관부처 AS 부처명
        , b.공고명
        , b.접수기간
        , b.전문기관
        , b.사업담당자연락처 AS 담당자
    FROM 아이리스_공고_대분류 AS a
    JOIN 아이리스_크롤링 AS b
        ON a.ancm_id = b.ancm_id
    WHERE a.대분류코드 = :tech_codes
    AND b.구분자 = '접수중'
    ORDER BY b.공고일자
    LIMIT 10
    ;
    """


def q_iris_2() -> str:
    """
    """
    return"""
    SELECT b.소관부처 AS 부처명
        , b.공고명
        , b.접수기간
        , b.전문기관
        , b.사업담당자연락처 AS 담당자
    FROM 아이리스_공고_대분류 AS a
    JOIN 아이리스_크롤링 AS b
        ON a.ancm_id = b.ancm_id
    WHERE a.대분류코드 = :tech_codes
    AND b.구분자 = '마감'
    ORDER BY b.공고일자
    LIMIT 10
    ;
    """

def qqqqqqqq() -> str:
    return """
    WITH patent_company AS (
    SELECT 
        ps.출원번호_ENC,
        MIN(cp.법인번호_ENC) AS 법인번호_ENC
    FROM 특허_국가과학기술표준분류체계 ps
    JOIN 기업_특허 cp
        ON cp.출원번호_ENC = ps.출원번호_ENC
    WHERE ps.소분류코드 = :sel_subcat_code
    GROUP BY ps.출원번호_ENC
),
unique_company AS (
    SELECT DISTINCT 법인번호_ENC
    FROM patent_company
),
finance_2024 AS (
    SELECT *
    FROM 재무정보
    WHERE LEFT(결산기준일자, 4) = '2024'
),
latest_financial AS (
    SELECT 
        법인번호_ENC,
        MAX(결산기준일자) AS 최근결산기준일자
    FROM finance_2024
    GROUP BY 법인번호_ENC
),
profit_ratio_ranked AS (
    SELECT
        f.법인번호_ENC,
        f.매출액영업이익률,
        f.보정여부_매출액영업이익률 AS 매출액영업이익률_보정여부,
        f.결산기준일자,
        ROW_NUMBER() OVER (
            PARTITION BY f.법인번호_ENC
            ORDER BY 
                (f.결산기준일자 = lf.최근결산기준일자 AND f.매출액영업이익률 IS NOT NULL) DESC,
                (f.매출액영업이익률 IS NOT NULL) DESC,
                f.결산기준일자 DESC
        ) AS rn
    FROM finance_2024 f
    JOIN latest_financial lf
        ON f.법인번호_ENC = lf.법인번호_ENC
),
selected_profit_ratio AS (
    SELECT 
        법인번호_ENC,
        매출액영업이익률,
        결산기준일자 AS 이익률_결산기준일자,
        매출액영업이익률_보정여부
    FROM profit_ratio_ranked
    WHERE rn = 1
),
latest_sales AS (
    SELECT 
        f.법인번호_ENC,
        f.매출액,
        f.보정여부 AS 매출액_보정여부,
        CASE
            WHEN f.매출액 IS NULL OR f.매출액 = 0
                THEN NULL
            ELSE pr.매출액영업이익률
        END AS 매출액영업이익률,
        f.결산기준일자 AS 매출액_결산기준일자,
        pr.이익률_결산기준일자,
        pr.매출액영업이익률_보정여부,
        f.매출액구간,
        CASE
            WHEN c.정부및공공기관구분 = '공공기관'
                THEN '그룹1'
            ELSE f.매출액구간
        END AS 최종매출액구간,
        c.정부및공공기관구분
    FROM finance_2024 f
    JOIN latest_financial lf
        ON f.법인번호_ENC = lf.법인번호_ENC
       AND f.결산기준일자 = lf.최근결산기준일자
    LEFT JOIN selected_profit_ratio pr
        ON f.법인번호_ENC = pr.법인번호_ENC
    JOIN 기업 c 
        ON f.법인번호_ENC = c.법인번호_ENC
),
target_companies AS (
    SELECT DISTINCT
        ls.법인번호_ENC,
        ls.매출액,
        ls.매출액_보정여부,
        ls.매출액영업이익률,
        ls.매출액영업이익률_보정여부,
        ls.매출액_결산기준일자,
        ls.이익률_결산기준일자,
        ls.매출액구간,
        ls.최종매출액구간
    FROM unique_company uc
    JOIN latest_sales ls
        ON uc.법인번호_ENC = ls.법인번호_ENC
    LEFT JOIN 상호 s
        ON ls.법인번호_ENC = s.법인번호_ENC
    WHERE ls.정부및공공기관구분 != '공공기관'
      AND (s.산학협력단여부 IS NULL OR s.산학협력단여부 = 0)
      AND ls.매출액 IS NOT NULL
),
RankedCompanies AS (
    SELECT 
        법인번호_ENC,
        매출액,
        매출액_보정여부,
        매출액영업이익률,
        매출액영업이익률_보정여부,
        매출액_결산기준일자,
        이익률_결산기준일자,
        매출액구간,
        최종매출액구간,
        DENSE_RANK() OVER (
            ORDER BY 매출액 DESC, 매출액영업이익률 DESC
        ) AS rank_num
    FROM target_companies
),
AvgPool AS (
    SELECT
        ls.매출액
    FROM unique_company uc
    JOIN latest_sales ls
        ON uc.법인번호_ENC = ls.법인번호_ENC
    LEFT JOIN 상호 s
        ON ls.법인번호_ENC = s.법인번호_ENC
    JOIN 기업 c
        ON ls.법인번호_ENC = c.법인번호_ENC
    WHERE c.정부및공공기관구분 != '공공기관'
      AND (s.산학협력단여부 IS NULL OR s.산학협력단여부 = 0)
      AND ls.매출액 IS NOT NULL
),
AvgSales AS (
    SELECT AVG(매출액) AS 평균매출액
    FROM AvgPool
),
AvgProfitRatioPool AS (
    SELECT
        CASE
            WHEN ls.매출액 IS NULL THEN NULL
            WHEN ls.매출액 = 0 THEN 0
            ELSE ls.매출액영업이익률
        END AS profit_ratio
    FROM unique_company uc
    JOIN latest_sales ls
        ON uc.법인번호_ENC = ls.법인번호_ENC
    LEFT JOIN 상호 s
        ON ls.법인번호_ENC = s.법인번호_ENC
    JOIN 기업 c
        ON ls.법인번호_ENC = c.법인번호_ENC
    WHERE c.정부및공공기관구분 != '공공기관'
      AND (s.산학협력단여부 IS NULL OR s.산학협력단여부 = 0)
      AND ls.매출액 IS NOT NULL
),
AvgProfitRatio AS (
    SELECT AVG(profit_ratio) AS 평균매출액영업이익률
    FROM AvgProfitRatioPool
)
SELECT 
    rc.법인번호_ENC,
    c.법인번호 AS 법인번호,
    GROUP_CONCAT(DISTINCT s.상호 ORDER BY s.상호 SEPARATOR ', ') AS 상호,
    rc.매출액,
    rc.매출액_보정여부,
    rc.매출액영업이익률,
    rc.매출액영업이익률_보정여부,
    rc.매출액구간,
    rc.rank_num,
    avg.평균매출액,
    avgpr.평균매출액영업이익률
FROM RankedCompanies rc
LEFT JOIN 상호 s
    ON rc.법인번호_ENC = s.법인번호_ENC
LEFT JOIN 기업 c
    ON rc.법인번호_ENC = c.법인번호_ENC
CROSS JOIN AvgSales avg
CROSS JOIN AvgProfitRatio avgpr
GROUP BY 
    rc.법인번호_ENC,
    c.법인번호,
    rc.매출액,
    rc.매출액_보정여부,
    rc.매출액영업이익률,
    rc.매출액영업이익률_보정여부,
    rc.매출액구간,
    rc.rank_num,
    avg.평균매출액,
    avgpr.평균매출액영업이익률
ORDER BY rc.rank_num ASC;
    """

def qqqqqqqq1() -> str:
    return """
    WITH finance_2024 AS (
    SELECT *
    FROM 재무정보
    WHERE LEFT(결산기준일자, 4) = '2024'
),
latest_financial AS (
    SELECT 
        법인번호_ENC,
        MAX(결산기준일자) AS 최근결산기준일자
    FROM finance_2024
    GROUP BY 법인번호_ENC
),
profit_ratio_ranked AS (
    SELECT
        f.법인번호_ENC,
        f.매출액영업이익률,
        f.보정여부_매출액영업이익률 AS 매출액영업이익률_보정여부,
        f.결산기준일자,
        ROW_NUMBER() OVER (
            PARTITION BY f.법인번호_ENC
            ORDER BY 
                (f.결산기준일자 = lf.최근결산기준일자 AND f.매출액영업이익률 IS NOT NULL) DESC,
                (f.매출액영업이익률 IS NOT NULL) DESC,
                f.결산기준일자 DESC
        ) AS rn
    FROM finance_2024 f
    JOIN latest_financial lf
        ON f.법인번호_ENC = lf.법인번호_ENC
),
selected_profit_ratio AS (
    SELECT 
        법인번호_ENC,
        매출액영업이익률,
        매출액영업이익률_보정여부
    FROM profit_ratio_ranked
    WHERE rn = 1
),
latest_sales AS (
    SELECT 
        f.법인번호_ENC,
        f.매출액,
        f.보정여부 AS 매출액_보정여부,
        CASE
            WHEN f.매출액 IS NULL OR f.매출액 = 0
                THEN NULL
            ELSE pr.매출액영업이익률
        END AS 매출액영업이익률,
        pr.매출액영업이익률_보정여부,
        f.매출액구간
    FROM finance_2024 f
    JOIN latest_financial lf
        ON f.법인번호_ENC = lf.법인번호_ENC
       AND f.결산기준일자 = lf.최근결산기준일자
    LEFT JOIN selected_profit_ratio pr
        ON f.법인번호_ENC = pr.법인번호_ENC
)
SELECT
    ls.법인번호_ENC,
    c.법인번호,
    GROUP_CONCAT(DISTINCT s.상호 ORDER BY s.상호 SEPARATOR ', ') AS 상호,
    ls.매출액,
    ls.매출액_보정여부,
    ls.매출액영업이익률,
    ls.매출액영업이익률_보정여부,
    ls.매출액구간
FROM latest_sales ls
JOIN 기업 c
    ON ls.법인번호_ENC = c.법인번호_ENC
LEFT JOIN 상호 s
    ON ls.법인번호_ENC = s.법인번호_ENC
WHERE ls.법인번호_ENC = :applicant_corp_no
GROUP BY
    ls.법인번호_ENC,
    c.법인번호,
    ls.매출액,
    ls.매출액_보정여부,
    ls.매출액영업이익률,
    ls.매출액영업이익률_보정여부,
    ls.매출액구간;
    """