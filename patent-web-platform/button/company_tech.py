"""
button.company_tech
===================

기술부문(Tech) 화면 모듈.

Streamlit 기반 BIGx 기업혁신성장 보고서에서
'기술부문' 탭을 렌더링하는 모듈이다.
공통 유틸리티 함수와 메인 렌더링 함수 :func:`render_tech` 를 제공한다.

기능 개요
--------

* 피인용 횟수를 기준으로 한 우수기술 분석 및 시각화.
* 피인용 지수를 기준으로 한 우수기술 분석 및 시각화.
* 특정 기술분야 내 최다 특허 보유 기업을 조회/표시한다.
* 경쟁 기업군을 도출하고, 특허·기술 관점에서 비교/분석한다.
* 이머징 기술 분석을 위해
  기간별 특허건수, 평균 증가율, 비중 등을 시각화한다.
"""


import streamlit as st
import pandas as pd
import numpy as np

from main.sql import sql_company_detail as Q


# ---------------------------------------------------
# 공통 상수 / 유틸 함수들
# ---------------------------------------------------

def highlight_target(row: pd.Series, target_idx: set[int]) -> list[str]:
    """
    경쟁기업군 테이블에서 신청기업 행을 강조하기 위한 스타일 리스트를 반환한다.

    :param row: 스타일 적용 대상 행.
    :type row: pandas.Series
    :param target_idx: 신청기업 행의 인덱스 모음.
    :type target_idx: set[int]
    :return: 각 컬럼에 대응하는 스타일 문자열 리스트.
    :rtype: list[str]
    """
    if row.name in target_idx:
        return ["background-color: #FFF4CC; font-weight: 600;"] * len(row)
    return [""] * len(row)


def compute_percentage_shares(counts: pd.Series, decimals: int = 1) -> pd.Series:
    """
    비중(%)을 소수점 자리까지 반올림하되, 총합이 100.0이 되도록 조정한다.

    :param counts: 건수 시리즈.
    :type counts: pandas.Series
    :param decimals: 소수점 자릿수(기본 1).
    :type decimals: int
    :return: 보정된 비중(%) 시리즈.
    :rtype: pandas.Series
    """
    counts = counts.astype("Float64").fillna(0.0)
    total_sum = counts.sum()

    if total_sum <= 0:
        return pd.Series([0.0] * len(counts), index=counts.index, dtype="Float64")

    factor = 10 ** decimals

    # 비중 계산
    shares = counts / total_sum * 100.0
    scaled = (shares * factor).to_numpy()
    base = np.floor(scaled)
    diff = int(round(100 * factor - base.sum()))

    frac = scaled - base
    if diff > 0:  # 부족분을 소수 부분이 큰 항목부터 더해줌
        order = np.argsort(-frac)
        base[order[:diff]] += 1
    elif diff < 0:  # 초과분을 소수 부분이 작은 항목부터 빼줌
        order = np.argsort(frac)
        base[order[:(-diff)]] -= 1

    return pd.Series(base / factor, index=counts.index, dtype="Float64")


def add_avg_growth(df: pd.DataFrame, period_cols: list[str]) -> pd.DataFrame:
    """
    기간별 특허건수를 이용해 평균 증가율(%) 컬럼을 추가한다.

    이전 기간 대비 증가율의 산술 평균을 ``평균증가율(%)`` 로 정의한다.

    :param df: 기간별 특허건수가 포함된 데이터프레임.
    :type df: pandas.DataFrame
    :param period_cols: 기간 컬럼명 리스트.
    :type period_cols: list[str]
    :return: ``평균증가율(%)`` 컬럼이 추가된 데이터프레임.
    :rtype: pandas.DataFrame
    """
    if not period_cols or len(period_cols) < 2:
        df["평균증가율(%)"] = pd.Series([0.0] * len(df), dtype="Float64")
        return df

    first_col = period_cols[0]      # 2015~2016년
    last_col = period_cols[-1]      # 2023~2024년
    steps = len(period_cols) - 1    # 5구간이면 4

    first = pd.to_numeric(df[first_col], errors="coerce").astype("Float64")
    last = pd.to_numeric(df[last_col], errors="coerce").astype("Float64")

    # 0 또는 NaN이면 계산 불가 → 0% 처리(원하시면 NA로 남기도록도 변경 가능)
    valid = (first.notna()) & (last.notna()) & (first > 0)

    avg_growth = pd.Series([0.0] * len(df), dtype="Float64")
    avg_growth.loc[valid] = (((last.loc[valid] / first.loc[valid]) ** (1 / steps)) - 1) * 100

    df["평균증가율(%)"] = avg_growth.round(1).astype("Float64")
    return df


def render_tech_header() -> None:
    """
    기술부문 상단 헤더(배너)를 렌더링한다.

    ``sec-banner`` / ``sec-label`` / ``sec-rule`` 스타일을 사용한
    섹션 타이틀 블록을 화면 상단에 출력한다.

    :return: 없음
    :rtype: None
    """
    st.markdown(
        f"""
        <div class="sec-banner" style="--accent:{"#5b9bd5"};">
          <div class="sec-label">{"기술부문"}</div>
          <div class="sec-rule"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

# ---------------------------------------------------
# 섹션 1: 우수특허
# ---------------------------------------------------
def render_top_cited_section(sel_subcat_name: str, sel_subcat_code: str, send_sql) -> None:
    st.markdown(
        f"""
        ### [{sel_subcat_name}] 우수특허
        ##### 소분류 내 보유 특허 중 피인용 지수가 높은 특허
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    df = send_sql(
        Q.q_view_top_cited(),
        params={"sel_subcat_code": sel_subcat_code},
    )

    if df.empty:
        st.info("해당 조건에서 데이터가 없습니다.")
        return

    df_show = df.copy()

    df_show.rename(
        columns={
            "특허제목": "특허명",
            "출원일": "출원일자",
        },
        inplace=True,
    )

    # ===== 원본 자료형 정리 =====
    df_show["출원번호"] = pd.to_numeric(df_show["출원번호"], errors="coerce").astype("Int64")
    df_show["출원일자"] = (
        df_show["출원일자"]
        .astype(str)
        .str.replace("-", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
        .astype("Int64")
    )
    df_show["출원일자"] = pd.to_datetime(df_show["출원일자"], format="%Y%m%d", errors="coerce")
    df_show["피인용횟수"] = pd.to_numeric(df_show["피인용횟수"], errors="coerce").astype("Int64")
    df_show["피인용지수"] = pd.to_numeric(df_show["피인용지수"], errors="coerce")

    df_show = df_show[df_show["피인용지수"] > 0].copy()

    if df_show.empty:
        st.info("피인용지수 > 0인 데이터가 없습니다.")
        return

    # ===== 화면 표시용 데이터프레임 =====
    df_view = df_show[["특허명", "출원번호", "출원일자", "기업명", "피인용횟수", "피인용지수"]].copy()
    df_view.insert(0, "순번", pd.Series(range(1, len(df_view) + 1), dtype="Int64"))

    # 출원번호/출원일자 문자열로 변환 (시간 없어짐 + 가운데 정렬 잘 먹게)
    df_view["출원번호"] = df_view["출원번호"].astype(str)
    df_view["출원일자"] = pd.to_datetime(df_view["출원일자"], errors="coerce").dt.strftime("%Y-%m-%d")

    st.markdown(
    """
    <style>
    /* 출원번호(3번째 컬럼), 출원일자(4번째 컬럼) 셀 가운데 정렬 */
    [data-testid="stDataEditor"] tbody tr td:nth-child(3),
    [data-testid="stDataEditor"] tbody tr td:nth-child(4),
    [data-testid="stDataFrame"] tbody tr td:nth-child(3),
    [data-testid="stDataFrame"] tbody tr td:nth-child(4) {
        text-align: center;
    }

    /* 헤더 텍스트도 가운데 정렬 */
    [data-testid="stDataEditor"] thead tr th:nth-child(3) div:nth-child(1),
    [data-testid="stDataEditor"] thead tr th:nth-child(4) div:nth-child(1),
    [data-testid="stDataFrame"] thead tr th:nth-child(3) div:nth-child(1),
    [data-testid="stDataFrame"] thead tr th:nth-child(4) div:nth-child(1) {
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )

    st.data_editor(
        df_view,  # 👈 df_show 말고 df_view 사용
        height=35 * len(df_view) + 38,
        use_container_width=False,
        hide_index=True,
        column_config={
            "순번": st.column_config.NumberColumn("순번", width=60),
            "특허명": st.column_config.TextColumn("특허명", width=630),

            "출원번호": {
                "label": "출원번호",
                "width": 150,
                "alignment": "center",
            },
            "출원일자": {
                "label": "출원일자",
                "width": 100,
                "alignment": "center",
            },

            "기업명": st.column_config.TextColumn("기업명", width=380),
            "피인용횟수": st.column_config.NumberColumn("피인용횟수", width="small"),
            "피인용지수": st.column_config.NumberColumn(
                "피인용지수",
                width="small",
                format="%.2f",
            ),
        },
        disabled=True,
        key="top-by-citation-count-table",
    )


    df = pd.DataFrame({'Column1': [1, 2, 3], 
                    'Column2': [4, 5, 6],
                    'Column3': [7, 8, 9]})

    df.style.set_properties(**{'text-align': 'center'}).set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}])
    st.markdown('<style>.col_heading{text-align: center;}</style>', unsafe_allow_html=True)
    df.columns = ['<div class="col_heading">'+col+'</div>' for col in df.columns] 

# ---------------------------------------------------
# 섹션 2: 최다 특허 보유 기업
# ---------------------------------------------------
def render_top_holder_section(sel_subcat_name: str, sel_subcat_code: str, send_sql) -> None:
    """
    최다 특허 보유 기업 테이블/지표를 렌더링한다.

    :param sel_subcat_name: 선택된 소분류명이다.
    :type sel_subcat_name: str
    :param sel_subcat_code: 선택된 소분류 코드이다.
    :type sel_subcat_code: str
    :param send_sql: SQL 실행 함수이다.
    :type send_sql: Callable
    :return: 없음
    :rtype: None
    """
    st.markdown(
        f"""
        ### [{sel_subcat_name}] 최다 특허 보유기업
        ##### 소분류 내 보유 특허 건수가 높은 기업
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    df_holder = send_sql(
        Q.q_top_holder(),
        params={"sel_subcat_code": sel_subcat_code},
    )

    if df_holder.empty:
        st.info("해당 조건에서 최다 특허 보유 기업 데이터가 없습니다.")
        return

    # '그 외' 그룹 집계
    df_etc = df_holder[df_holder["기업명"] == "그 외"].copy()
    if not df_etc.empty:
        df_etc["매출액"] = pd.to_numeric(
            df_etc["매출액"], errors="coerce"
        ).fillna(0)
        df_etc["보유특허건수"] = pd.to_numeric(
            df_etc["보유특허건수"], errors="coerce"
        ).fillna(0)
        etc_sales = df_etc["매출액"].sum()
        etc_patents = df_etc["보유특허건수"].sum()
    else:
        etc_sales = 0
        etc_patents = 0

    df3 = df_holder[df_holder["기업명"] != "그 외"].copy()
    df3["매출액"] = pd.to_numeric(df3["매출액"], errors="coerce").fillna(0)
    df3["매출액"] = (df3["매출액"] / 100).round(2)
    df3 = df3.rename(columns={"매출액": "매출액(억원)"})

    df3["보유특허수"] = pd.to_numeric(
        df3["보유특허건수"], errors="coerce"
    ).astype("Int64")
    df3["매출액(억원)"] = pd.to_numeric(
        df3["매출액(억원)"], errors="coerce"
    ).astype("float64")

    # 비중(%) 계산 (소수 1자리, 마지막 행에서 100.0 맞추기)
    total = df3["보유특허수"].fillna(0).sum()
    if total > 0:
        pct = df3["보유특허수"].fillna(0).astype(float) / float(total) * 100
        rounded = (pct * 10).round() / 10
        rounded.iloc[-1] = (100 - rounded.iloc[:-1].sum()).round(1)
        df3["비중(%)"] = rounded.astype("Float64")
    else:
        df3["비중(%)"] = pd.Series([pd.NA] * len(df3), dtype="Float64")

    df3.insert(0, "순번", pd.Series(range(1, len(df3) + 1), dtype="Int64"))
    df_display = df3[
        ["순번", "기업명", "매출액(억원)", "보유특허수", "비중(%)"]
    ].copy()

    st.data_editor(
        df_display,
        height=35 * len(df_display) + 38,
        use_container_width=False,
        hide_index=True,
        disabled=True,
        num_rows="fixed",
        column_config={
            "순번": st.column_config.NumberColumn("순번", width=60),
            "기업명": st.column_config.TextColumn("기업명", width=600),
            "매출액(억원)": st.column_config.NumberColumn(
                "매출액(억원)", format="accounting", width=300
            ),
            "보유특허수": st.column_config.NumberColumn("보유특허수", width=200),
            "비중(%)": st.column_config.NumberColumn(
                "비중(%)", format="%.1f", width=230
            ),
        },
        key="top-holder-editor",
    )

    total_patents_all = pd.to_numeric(
        df_holder["보유특허건수"], errors="coerce"
    ).fillna(0).sum()
    total_sales_all = pd.to_numeric(
        df_holder["매출액"], errors="coerce"
    ).fillna(0).sum()

    st.markdown(
        """
        <style>
        .fin-metrics {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-top: 8px;
            margin-bottom: 24px;
        }
        .pill {
            border-radius: 8px;
            padding: 10px 12px;
            font-weight: 700;
            font-size: 0.9rem;
            text-align: center;
            border: 1px solid #cbd5e1;
            line-height: 1.4;
        }
        .pill.green { background: #e9f8ee; }
        .pill.blue  { background: #eaf0ff; }
        .pill b { font-weight: 800; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="fin-metrics">
            <div class="pill green">
                <b>그 외 그룹</b><br>
                매출액(억원) <b>{etc_sales/100:,.1f}</b> · 특허수 <b>{int(etc_patents):,}</b>
            </div>
            <div class="pill blue">
                <b>전체 합계</b><br>
                매출액(억원) <b>{total_sales_all/100:,.1f}</b> · 특허수 <b>{int(total_patents_all):,}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------
# 섹션 4: 경쟁 기업군
# ---------------------------------------------------
def render_competitor_group_section(
    sel_subcat_name: str,
    sel_subcat_code: str,
    applicant_corp_no: str,
    applicant_name: str,
    company_cnt: int,
    send_sql,
) -> None:
    """
    경쟁 기업군 테이블을 렌더링한다.

    :param sel_subcat_name: 선택된 소분류명이다.
    :type sel_subcat_name: str
    :param sel_subcat_code: 선택된 소분류 코드이다.
    :type sel_subcat_code: str
    :param applicant_corp_no: 신청기업 법인번호_ENC이다.
    :type applicant_corp_no: str
    :param applicant_name: 신청기업명이다.
    :type applicant_name: str
    :param company_cnt: 해당 소분류 내 총 기업 수이다.
    :type company_cnt: int
    :param send_sql: SQL 실행 함수이다.
    :type send_sql: Callable
    :return: 없음
    :rtype: None
    """
    st.markdown(
        f"""
        ### [{applicant_name}] 경쟁 기업군
        ##### 대상기업 대비 매출액이 높은 기업들의 특허 건수
        ###### [{sel_subcat_name}] 내 총 기업수 : {company_cnt}개
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    df_comp = send_sql(
        Q.q_competitor_group(),
        params={
            "sel_subcat_code": sel_subcat_code,
            "applicant_corp_no": applicant_corp_no,
        },
    )

    if df_comp.empty:
        st.info("경쟁기업군 데이터가 없습니다.")
        return

    df4 = df_comp.copy()
    df4["매출액"] = pd.to_numeric(df4["매출액"], errors="coerce").fillna(0)
    df4["매출액"] = (df4["매출액"] / 100).round(2)

    df4 = df4.rename(
        columns={
            "설립년도": "설립일",
            "매출액": "매출액(억원)",
            "2015~2016년 특허 등록건수": "2015~2016년",
            "2017~2018년 특허 등록건수": "2017~2018년",
            "2019~2020년 특허 등록건수": "2019~2020년",
            "2021~2022년 특허 등록건수": "2021~2022년",
            "2023~2024년 특허 등록건수": "2023~2024년",
            "특허 등록건수 합계": "합계",
        }
    )

    period_cols = [
        "2015~2016년",
        "2017~2018년",
        "2019~2020년",
        "2021~2022년",
        "2023~2024년",
    ]

    df4["매출액(억원)"] = pd.to_numeric(
        df4["매출액(억원)"], errors="coerce"
    ).astype("float64")
    df4["설립일"] = pd.to_numeric(df4["설립일"], errors="coerce").astype("Int64")
    df4["순위"] = pd.to_numeric(df4["순위"], errors="coerce").astype("Int64")
    for c in period_cols:
        df4[c] = pd.to_numeric(df4[c], errors="coerce").astype("Int64")
    df4["합계"] = pd.to_numeric(df4["합계"], errors="coerce").astype("Int64")

    df4 = df4.sort_values("순위").reset_index(drop=True)
    display_cols = ["순위", "기업명", "설립일", "매출액(억원)", *period_cols, "합계"]
    df_display2 = df4[display_cols].copy()

    target_corp_set = {str(applicant_corp_no).strip()} if applicant_corp_no else set()
    corp_series = df4["법인번호_ENC"].astype(str).str.strip()
    target_idx = set(df4.index[corp_series.isin(target_corp_set)])

    if target_idx:
        styled = df_display2.style.apply(
            lambda row: highlight_target(row, target_idx), axis=1
        )
    else:
        styled = df_display2

    period_col_config = {
        col: st.column_config.NumberColumn(
            f"{col} (특허등록건수)", width=140, format="%d"
        )
        for col in period_cols
    }

    st.data_editor(
        styled,
        height=35 * len(df_display2) + 38,
        use_container_width=False,
        hide_index=True,
        disabled=True,
        num_rows="fixed",
        column_config={
            "순위": st.column_config.NumberColumn("순위", width=50),
            "기업명": st.column_config.TextColumn("기업명", width=200),
            "매출액(억원)": st.column_config.NumberColumn(
                "매출액(억원)", format="accounting", help="회계식 표기"
            ),
            "설립일": st.column_config.NumberColumn("설립일", width=60),
            **period_col_config,
            "합계": st.column_config.NumberColumn("합계", width=60),
        },
        key="competitor-group-editor",
    )


# ---------------------------------------------------
# 섹션 5: 이머징 기술
# ---------------------------------------------------
def render_emerging_tech_section(sel_subcat_name: str, sel_subcat_code: str, send_sql) -> None:
    """
    이머징 기술 테이블을 렌더링한다.

    :param sel_subcat_name: 선택된 소분류명이다.
    :type sel_subcat_name: str
    :param sel_subcat_code: 선택된 소분류 코드이다.
    :type sel_subcat_code: str
    :param send_sql: SQL 실행 함수이다.
    :type send_sql: Callable
    :return: 없음
    :rtype: None
    """
    df_mid = send_sql(
        Q.q_emerging_by_mid(),
        params={"sel_subcat_code": sel_subcat_code},
    )

    if not df_mid.empty:
        mid_name = df_mid["중분류명"].iloc[0]
    else:
        mid_name = None

    st.markdown(
        f"""
        ### 중분류 [{mid_name}] 내 [{sel_subcat_name}] 이머징기술
        ##### 소분류와 중분류가 동일한 소분류들의 특허건수
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    df_raw = send_sql(
        Q.q_emerging_by_mid(),
        params={"sel_subcat_code": sel_subcat_code},
    )

    if df_raw.empty:
        st.info("이머징 기술 데이터가 없습니다.")
        return

    df = df_raw.rename(
        columns={
            "소분류명": "분류명",
            "2015~2016년 특허 등록건수": "2015~2016년",
            "2017~2018년 특허 등록건수": "2017~2018년",
            "2019~2020년 특허 등록건수": "2019~2020년",
            "2021~2022년 특허 등록건수": "2021~2022년",
            "2023~2024년 특허 등록건수": "2023~2024년",
            "특허 등록건수 합계": "합계",
        }
    ).copy()

    period_cols = [
        "2015~2016년",
        "2017~2018년",
        "2019~2020년",
        "2021~2022년",
        "2023~2024년",
    ]

    for c in period_cols + ["합계"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # 평균증가율(%) 계산
    df = add_avg_growth(df, period_cols)

    # 비중(%) 계산 
    counts = df["합계"].astype("Float64").fillna(0.0)
    total_sum = counts.sum()
    if total_sum > 0:
        df["비중(%)"] = compute_percentage_shares(counts, decimals=1)
    else:
        df["비중(%)"] = pd.Series([0.0] * len(df), dtype="Float64")

    df.insert(0, "순번", pd.Series(range(1, len(df) + 1), dtype="Int64"))

    sum_row = {
        "순번": pd.NA,
        "분류명": "합계",
        **{c: 0 for c in period_cols},
        "합계": int(total_sum) if (pd.notna(total_sum) and int(total_sum) > 0) else 0,
        "평균증가율(%)": 0.0,
        "비중(%)": 100.0 if (pd.notna(total_sum) and int(total_sum) > 0) else 0.0,
    }
    df_total = pd.concat([df, pd.DataFrame([sum_row])], ignore_index=True)

    # 합계 제외 후 정렬 / 순번 재부여
    df_no_total = (
        df_total.loc[df_total["분류명"] != "합계"]
        .drop(columns=["순번"], errors="ignore")
        .copy()
    )
    df_no_total = (
        df_no_total.sort_values("평균증가율(%)", ascending=False)
        .reset_index(drop=True)
    )
    df_no_total["순번"] = pd.Series(
        range(1, len(df_no_total) + 1), dtype="Int64"
    )

    df_no_total["평균증가율(%)"] = df_no_total["평균증가율(%)"].astype("Float64")
    df_no_total["비중(%)"] = df_no_total["비중(%)"].astype("Float64")

    sel_periods = [c for c in period_cols if c in df_no_total.columns]

    df_display = df_no_total[
        ["순번", "분류명", *sel_periods, "합계", "평균증가율(%)", "비중(%)"]
    ].copy()


    st.data_editor(
        df_display,
        hide_index=True,
        disabled=True,
        use_container_width=False,
        height=35 * len(df_display) + 38,
        key="emerging-tech-table",
        column_config={
            "순번": st.column_config.NumberColumn("순번", disabled=True, width=60),
            "분류명": st.column_config.TextColumn("분류명", disabled=True, width=220),
            **{
                col: st.column_config.NumberColumn(
                    f"{col} (특허등록건수)",
                    format="%d",
                    disabled=True,
                    width=180,   # 👈 여기 추가 (원하는 값으로 조정하면 됨)
                )
                for col in sel_periods
            },
            "합계": st.column_config.NumberColumn(
                "합계", format="%d", disabled=True, width=110
            ),
            "평균증가율(%)": st.column_config.NumberColumn(
                "평균증가율(%)", format="%.1f", disabled=True
            ),
            "비중(%)": st.column_config.NumberColumn(
                "비중(%)", format="%.1f", disabled=True, width=80
            ),
        },
    )


# ---------------------------------------------------
# 메인: 기술부문 화면
# ---------------------------------------------------

def render_tech(
    sel_subcat_name: str,
    sel_subcat_code: str,
    applicant_corp_no: str,
    applicant_name: str,
    send_sql,
) -> None:
    """
    기술부문 화면 전체를 그리는 메인 함수이다.

    동작 흐름
    ---------
    1. :func:`render_tech_header` 를 호출해 기술부문 상단 헤더를 렌더링한다.
    2. :func:`render_top_cited_section` 을 호출해
       우수기술(피인용 횟수 기준) 테이블을 렌더링한다.
    3. :func:`render_top_citation_index_section` 을 호출해
       우수기술(피인용 지수 기준) 테이블을 렌더링한다.
    4. :func:`render_top_holder_section` 을 호출해
       최다 특허 보유 기업 현황을 렌더링한다.
    5. :func:`render_competitor_group_section` 을 호출해
       신청기업 기준 경쟁 기업군 테이블을 렌더링한다.
    6. :func:`render_emerging_tech_section` 을 호출해
       중분류 단위 이머징 기술 테이블을 렌더링한다.

    세션 상태
    ---------
    * ``"company_cnt"``:
      경쟁 기업군 섹션 제목에 표시할
      해당 소분류 내 총 기업 수를 저장한 값이다.
      이 함수에서는 값을 수정하지 않고 읽기만 한다.

    :param sel_subcat_name: 선택된 소분류명이다.
    :type sel_subcat_name: str
    :param sel_subcat_code: 선택된 소분류 코드이다.
    :type sel_subcat_code: str
    :param applicant_corp_no: 신청기업 법인번호_ENC이다.
    :type applicant_corp_no: str
    :param applicant_name: 신청기업명이다.
    :type applicant_name: str
    :param send_sql: SQL 실행 함수이다.
    :type send_sql: Callable
    :return: 없음
    :rtype: None
    """
    company_cnt = st.session_state.get("company_cnt")

    # 제목
    render_tech_header()

    # 1) 우수기술(피인용 횟수)
    render_top_cited_section(sel_subcat_name, sel_subcat_code, send_sql)
    st.markdown("<br><br>", unsafe_allow_html=True)

    # # 3) 최다 특허 보유 기업
    # render_top_holder_section(sel_subcat_name, sel_subcat_code, send_sql)
    # st.markdown("<br><br>", unsafe_allow_html=True)

    # # 4) 경쟁 기업군
    # render_competitor_group_section(
    #     sel_subcat_name,
    #     sel_subcat_code,
    #     applicant_corp_no,
    #     applicant_name,
    #     company_cnt,
    #     send_sql,
    # )
    # st.markdown("<br><br>", unsafe_allow_html=True)

    # 5) 이머징 기술
    render_emerging_tech_section(sel_subcat_name, sel_subcat_code, send_sql)
