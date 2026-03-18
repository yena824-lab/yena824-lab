"""
button.tech_tech
================

기술부문(Tech) 화면 모듈.

Streamlit 기반 BIGx 보고서에서 **기술부문 섹션**을
렌더링하는 모듈이다.
유틸리티 함수들과 메인 렌더링 함수 :func:`render_tech` 를 제공한다.

기능 개요
--------

* 기업명/법인번호 후처리 및 비율(%) 계산 유틸리티 함수를 제공한다.
* 피인용 횟수 및 피인용 지수를 기반으로
  우수 특허 목록을 조회·표시한다.
* 최다 특허 보유 기업, 경쟁 기업군, 이머징 기술 목록을 조회·표시한다.
* :func:`render_tech` 를 통해
  기술부문 전체 레이아웃과 데이터를 연동하여 화면을 렌더링한다.
"""


import pandas as pd
import numpy as np
import streamlit as st

from main.sql import sql_tech_detail as Q


# ---------------------------------------------------
# 공통 상수 / 유틸 함수들
# ---------------------------------------------------
def compute_percentage_shares(counts: pd.Series, decimals: int = 1) -> pd.Series:
    """
    건수 시리즈로부터 비중(%)을 계산하고 라운딩 오차를 보정한다.

    - 소수점 ``decimals`` 자리까지 반올림
    - 전체 합계가 정확히 100.0이 되도록 보정

    :param counts: 기준이 되는 건수(또는 합계) 시리즈이다.
    :type counts: pandas.Series
    :param decimals: 소수점 자리수이다.
    :type decimals: int
    :return: 각 항목의 비중(%) 시리즈이다.
    :rtype: pandas.Series
    """
    counts = counts.astype("Float64").fillna(0.0)
    total = float(counts.sum())

    if total <= 0:
        # 전체가 0이면 전부 NA
        return pd.Series([pd.NA] * len(counts), index=counts.index, dtype="Float64")

    factor = 10 ** decimals  # 소수 자리수 처리용 스케일
    shares = counts / total * 100.0  # 기본 비율(%)

    # 스케일 적용 후 내림/보정
    scaled = (shares * factor).to_numpy()
    base = np.floor(scaled)
    diff = int(round(100 * factor - base.sum()))  # 합계 보정값
    frac = scaled - base  # 소수 부분

    # diff > 0 이면 소수 부분이 큰 순서대로 플러스
    if diff > 0:
        order = np.argsort(-frac)
        base[order[:diff]] += 1
    # diff < 0 이면 소수 부분이 작은 순서대로 마이너스
    elif diff < 0:
        order = np.argsort(frac)
        base[order[: (-diff)]] -= 1

    return pd.Series(base / factor, index=counts.index, dtype="Float64")

# ---------------------------------------------------
# 섹션 1: 우수특허
# ---------------------------------------------------
def render_top_cited_section(sel_subcat_name: str, sel_subcat_code: str, send_sql) -> None:
    """
    우수특허 테이블을 렌더링한다.

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


# ---------------------------------------------------
# 섹션 2: 최다 특허 보유 기업
# ---------------------------------------------------
def render_top_holder_section(sel_subcat_name: str, sel_subcat_code: str, send_sql) -> None:
    """
    3. 최다 특허 보유 기업 섹션을 렌더링한다.

    - 소분류 내 보유 특허 건수가 높은 기업 상위 목록 및 비중(%)를 표시한다.
    - ``"그 외"`` 그룹 및 전체 합계도 하단 메트릭으로 제공한다.

    :param sel_subcat_name: 현재 선택된 소분류명이다.
    :type sel_subcat_name: str
    :param sel_subcat_code: 현재 선택된 소분류 코드이다.
    :type sel_subcat_code: str
    :param send_sql: DB 조회 함수이다.
    :type send_sql: Callable[[str, dict], pandas.DataFrame]
    :return: 없음.
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

    # 최다 보유 기업 조회 (상위 + '그 외')
    df_holder = send_sql(
        Q.q_top_holder(),
        params={"sel_subcat_code": sel_subcat_code},
    )

    if df_holder.empty:
        st.info("해당 조건에서 최다 특허 보유 기업 데이터가 없습니다.")
        return

    # "그 외" 집계용
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

    # '그 외' 행 제외한 실제 상위 기업들만 사용
    df = df_holder[df_holder["기업명"] != "그 외"].copy()
    # 매출액 숫자변환 → 억 단위( / 100 )
    df["매출액"] = pd.to_numeric(df["매출액"], errors="coerce").fillna(0)

    df["매출액"] = (df["매출액"] / 100).round(2)
    df = df.rename(columns={"매출액": "매출액(억원)"})

    # 보유 특허 수, 매출액(억원) 타입 정리
    df["보유특허수"] = pd.to_numeric(
        df["보유특허건수"], errors="coerce"
    ).astype("Int64")
    df["매출액(억원)"] = pd.to_numeric(
        df["매출액(억원)"], errors="coerce"
    ).astype("float64")

    # 특허수 비중 계산
    total = df["보유특허수"].fillna(0).sum()
    if total > 0:
        df["비중(%)"] = compute_percentage_shares(df["보유특허수"], decimals=1)
    else:
        df["비중(%)"] = pd.Series([pd.NA] * len(df), dtype="Float64")

    # 순번 컬럼 (1부터)
    df.insert(0, "순번", pd.Series(range(1, len(df) + 1), dtype="Int64"))
    df_display = df[
        ["순번", "기업명", "매출액(억원)", "보유특허수", "비중(%)"]
    ].copy()

    # 상위 기업 표 렌더링
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
            "보유특허수": st.column_config.NumberColumn("보유특허수", width=290),
            "비중(%)": st.column_config.NumberColumn(
                "비중(%)", format="%.1f", width=285
            ),
        },
        key="top-holder-editor1",
    )

    # 전체 합계(상위 + 그 외) 계산
    total_patents_all = pd.to_numeric(
        df_holder["보유특허건수"], errors="coerce"
    ).fillna(0).sum()
    total_sales_all = pd.to_numeric(
        df_holder["매출액"], errors="coerce"
    ).fillna(0).sum()

    # 하단 '그 외 그룹 / 전체 합계' 메트릭 스타일 정의
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

    # 메트릭 박스 렌더링
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
def render_competitor_group_section(sel_subcat_name: str, sel_subcat_code: str, company_cnt, send_sql) -> None:
    """
    4. 경쟁 기업군 섹션을 렌더링한다.

    - 매출액이 높은 상위 기업들을 경쟁 기업군으로 표시한다.
    - 각 기업의 설립일/매출/기간별 특허등록건수/합계를 표로 보여준다.

    :param sel_subcat_name: 현재 선택된 소분류명이다.
    :type sel_subcat_name: str
    :param sel_subcat_code: 현재 선택된 소분류 코드이다.
    :type sel_subcat_code: str
    :param company_cnt: 선택된 소분류에 속한 전체 기업 수이다.
    :type company_cnt: Any
    :param send_sql: DB 조회 함수이다.
    :type send_sql: Callable[[str, dict], pandas.DataFrame]
    :return: 없음.
    :rtype: None
    """
    st.markdown(
        f"""
        ### 경쟁 기업군
        ##### 매출액이 높은 상위 기업들
        ###### [{sel_subcat_name}] 내 총 기업수 : {company_cnt}개
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    # 경쟁 기업군 조회
    df_comp = send_sql(
        Q.q_competitor_group(),
        params={"sel_subcat_code": sel_subcat_code},
    )

    if df_comp.empty:
        st.info("경쟁기업군 데이터가 없습니다.")
        return

    df2 = df_comp.copy()

    # 기업명/법인번호 컬럼이 없으면 생성
    if "기업명" not in df2.columns:
        df2["기업명"] = ""
    if "법인번호_ENC" not in df2.columns:
        df2["법인번호_ENC"] = ""

    # 공백 제거
    df2["기업명"] = df2["기업명"].astype(str).str.strip()
    df2["법인번호_ENC"] = df2["법인번호_ENC"].astype(str).str.strip()

    # 기업명이 비어있거나 None, nan 등인 경우 법인번호로 대체
    missing_mask = (
        df2["기업명"].isna()
        | df2["기업명"].eq("")
        | df2["기업명"].isin(["-", "None", "nan"])
    )
    df2.loc[missing_mask, "기업명"] = df2.loc[missing_mask, "법인번호_ENC"]
    df2["기업명"].replace({"": "-"}, inplace=True)

    # 매출액: 숫자 변환 후 억 단위로 변환
    df2["매출액"] = pd.to_numeric(df2["매출액"], errors="coerce").fillna(0)
    df2["매출액"] = (df2["매출액"] / 100).round(2)

    # 컬럼명 정리 (표시용)
    df2 = df2.rename(
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

    # 기간별 특허 컬럼 목록
    period_cols = [
        "2015~2016년",
        "2017~2018년",
        "2019~2020년",
        "2021~2022년",
        "2023~2024년",
    ]

    # 타입 정리
    df2["매출액(억원)"] = pd.to_numeric(
        df2["매출액(억원)"], errors="coerce"
    ).astype("float64")
    df2["설립일"] = pd.to_numeric(df2["설립일"], errors="coerce").astype("Int64")
    for c in period_cols:
        df2[c] = pd.to_numeric(df2[c], errors="coerce").astype("Int64")
    df2["합계"] = pd.to_numeric(df2["합계"], errors="coerce").astype("Int64")

    # 순위 기준 정렬
    df2 = df2.sort_values("순위").reset_index(drop=True)

    # 표시할 컬럼들
    display_cols = ["순위", "기업명", "설립일", "매출액(억원)", *period_cols, "합계"]
    df_display2 = df2[display_cols].copy()

    # 기간별 컬럼에 대한 column_config 정의
    period_col_config = {
        col: st.column_config.NumberColumn(
            f"{col} (특허출원건수)", width=140, format="%d"
        )
        for col in period_cols
    }

    # 경쟁 기업군 표 렌더링
    st.data_editor(
        df_display2,
        height=35 * len(df_display2) + 38,
        use_container_width=True,
        hide_index=True,
        disabled=True,
        num_rows="fixed",
        column_config={
            "순위": st.column_config.NumberColumn("순위", width=60),
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
    5. 이머징 기술 섹션을 렌더링한다.

    - 동일 중분류에 속한 소분류들의 기간별 특허등록건수/합계/평균증가율/비중을 표로 보여준다.
    - 이머징 후보 기술 분류를 한눈에 볼 수 있도록 정렬 및 순번을 부여한다.

    :param sel_subcat_name: 현재 선택된 소분류명이다.
    :type sel_subcat_name: str
    :param sel_subcat_code: 현재 선택된 소분류 코드이다.
    :type sel_subcat_code: str
    :param send_sql: DB 조회 함수이다.
    :type send_sql: Callable[[str, dict], pandas.DataFrame]
    :return: 없음.
    :rtype: None
    """
    # 해당 소분류가 속한 중분류명 조회
    df_mid = send_sql(
        Q.q_emerging_by_mid(),
        params={"sel_subcat_code": sel_subcat_code},
    )

    if not df_mid.empty:
        mid_name = df_mid["중분류명"].iloc[0]
    else:
        mid_name = None

    # 섹션 제목
    st.markdown(
        f"""
        ### 중분류 [{mid_name}] 내 [{sel_subcat_name}] 이머징기술
        ##### 소분류와 중분류가 동일한 소분류들의 특허건수
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    # 이머징 기술 데이터 조회
    df_raw = send_sql(
        Q.q_emerging_by_mid(),
        params={"sel_subcat_code": sel_subcat_code},
    )

    if df_raw.empty:
        st.info("이머징 기술 데이터가 없습니다.")
        return

    # 컬럼명 정리 (소분류명 → 분류명, 기간별/합계 컬럼명 변경)
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

    # 기간별 컬럼 목록
    period_cols = [
        "2015~2016년",
        "2017~2018년",
        "2019~2020년",
        "2021~2022년",
        "2023~2024년",
    ]

    # 기간별/합계 컬럼 타입 정수형으로 정리
    for c in period_cols + ["합계"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # -------------------------
    # 평균 증가율 계산 (1구간 vs 5구간만 사용)
    # 평균증가율(%) = ((5구간/1구간)^(1/4) - 1) * 100
    # -------------------------
    first_col = period_cols[0]      # 2015~2016년
    last_col = period_cols[-1]      # 2023~2024년
    steps = len(period_cols) - 1    # 5구간이면 4

    first = pd.to_numeric(df[first_col], errors="coerce").astype("Float64")
    last = pd.to_numeric(df[last_col], errors="coerce").astype("Float64")

    valid = (first.notna()) & (last.notna()) & (first > 0)

    avg_growth = pd.Series([0.0] * len(df), dtype="Float64")
    avg_growth.loc[valid] = (((last.loc[valid] / first.loc[valid]) ** (1 / steps)) - 1) * 100

    df["평균증가율(%)"] = avg_growth.round(1).astype("Float64")

    # -------------------------
    # 비중(%) 계산
    # -------------------------
    counts = df["합계"].astype("Float64").fillna(0.0)
    total_sum = counts.sum()

    if total_sum > 0:
        df["비중(%)"] = compute_percentage_shares(counts, decimals=1)
    else:
        df["비중(%)"] = pd.Series([0.0] * len(df), dtype="Float64")

    # 순번 컬럼 추가
    df.insert(0, "순번", pd.Series(range(1, len(df) + 1), dtype="Int64"))

    # '합계' 행 제거 후 순번 재부여 (실제 분류만 사용)
    df_no_total = (
        df.loc[df["분류명"] != "합계"]
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

    # 타입 재정의 (표시용)
    df_no_total["평균증가율(%)"] = df_no_total["평균증가율(%)"].astype("Float64")
    df_no_total["비중(%)"] = df_no_total["비중(%)"].astype("Float64")

    # 실제 존재하는 기간 컬럼만 선택
    sel_periods = [
        c
        for c in period_cols
        if c in df_no_total.columns
    ]

    # 표시용 DataFrame 구성
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
def render_tech(sel_subcat_name: str, sel_subcat_code: str, send_sql) -> None:
    """
    기술부문 화면 전체를 그리는 메인 함수이다.

    동작 흐름
    ---------

    1. ``st.session_state["company_cnt"]`` 에 저장된 기업 수를 조회한다.
    2. 상단에 ``"기술부문"`` 섹션 배너를 렌더링한다.
    3. :func:`render_top_cited_section` 을 호출하여
       피인용 횟수 기준 우수기술 목록을 표시한다.
    4. :func:`render_top_citation_index_section` 을 호출하여
       피인용 지수 기준 우수기술 목록을 표시한다.
    5. :func:`render_top_holder_section` 을 호출하여
       최다 특허 보유 기업 정보를 표로 표시한다.
    6. :func:`render_competitor_group_section` 을 호출하여
       경쟁 기업군(상위 매출 기업) 정보를 표시한다.
    7. :func:`render_emerging_tech_section` 을 호출하여
       이머징 기술 후보 목록을 표시한다.

    세션 상태
    ---------

    * ``st.session_state["company_cnt"]``:
      :mod:`tech_overview` 에서 저장한 소분류 내 전체 기업 수로,
      경쟁 기업군 설명 문구에 사용된다.

    :param sel_subcat_name: 현재 선택된 소분류명이다.
    :type sel_subcat_name: str
    :param sel_subcat_code: 현재 선택된 소분류 코드이다.
    :type sel_subcat_code: str
    :param send_sql: DB 조회 함수이다.
    :type send_sql: Callable[[str, dict], pandas.DataFrame]
    :return: 없음.
    :rtype: None
    """

    # company_detail 단계에서 세션에 저장한 기업 수 (경쟁기업군 설명에 사용)
    company_cnt = st.session_state.get("company_cnt")

    # 상단 섹션 배너 렌더링
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

    # 1) 우수기술(피인용 횟수 기준)
    render_top_cited_section(sel_subcat_name, sel_subcat_code, send_sql)
    st.markdown("<br><br>", unsafe_allow_html=True)

    # # 3) 최다 특허 보유 기업
    # render_top_holder_section(sel_subcat_name, sel_subcat_code, send_sql)
    # st.markdown("<br><br>", unsafe_allow_html=True)

    # # 4) 경쟁 기업군
    # render_competitor_group_section(sel_subcat_name, sel_subcat_code, company_cnt, send_sql)
    # st.markdown("<br><br>", unsafe_allow_html=True)

    # 5) 이머징 기술
    render_emerging_tech_section(sel_subcat_name, sel_subcat_code, send_sql)
