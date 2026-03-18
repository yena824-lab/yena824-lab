# # -*- coding: utf-8 -*-
# """
# views.company_0_detail
# ======================

# 특허가 0건인 기업을 위한
# '동업종 내 기술분포' 전용 상세 화면 모듈.

# 역할
# ----

# * 신청기업의 업종(10차) 정보를 기준으로
#   같은 업종에 속한 기업들의 특허를 분석한다.
# * 동업종 내 기술분포(중분류/소분류 파이차트)를 한 줄에 표시한다.
# * 소분류 선택용 AgGrid를 제공한다.
# * 사용자가 선택한 소분류에 대해
#   소분류 설명 + 매출 규모별 기업/특허 현황(막대/파이)을 제공한다.

# 세션 상태
# ---------

# * ``st.session_state["applicant_corp_no"]`` : 신청 기업 법인번호_ENC
# * ``st.session_state["applicant_name"]``     : 신청 기업명
# * ``st.session_state["page"]``              : 상위 라우팅용 페이지 키
# * ``st.session_state["selected_company_row"]`` : company 화면의 선택 기업 (뒤로가기 시 초기화)
# * ``st.session_state["pending_nav"]``       : 라우팅 플래그
# * ``st.session_state["selected_subcat_row_0detail"]`` : 0건 상세에서 선택한 소분류 행
# * ``st.session_state["company_0_detail_init_corp"]``  : 기업 변경 감지용
# """

# from __future__ import annotations

# import pandas as pd
# import streamlit as st
# import altair as alt

# from core.db import send_sql
# from main.sql import sql_company_0_detail as Q

# # company_overview 에서 사용하는 공통 상수/함수 재사용
# from button.company_overview import (
#     BUCKET_ORDER,
#     GROUP_LABELS,
#     BUCKET_TO_GROUP,
#     prepare_pie_data,
#     render_pie_charts,
#     render_group_legend,
# )

# # AgGrid
# try:
#     from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
# except Exception:
#     AgGrid = None
#     GridOptionsBuilder = None
#     GridUpdateMode = None


# # ----------------------------------------------------------------------
# # 공통 네비게이션 헬퍼
# # ----------------------------------------------------------------------
# def go(page: str) -> None:
#     """현재 페이지 상태를 변경한다."""
#     st.session_state.page = page


# def back_from_zero_detail() -> None:
#     """
#     특허 0건 상세 화면에서 '뒤로가기' 눌렀을 때 호출되는 함수.

#     - 페이지를 company로 돌리고
#     - selected_company_row / pending_nav / selected_subcat_row_0detail 등을 초기화한다.
#     """
#     st.session_state.page = "company"

#     if "selected_company_row" in st.session_state:
#         st.session_state.selected_company_row = None

#     if "pending_nav" in st.session_state:
#         st.session_state.pending_nav = None

#     if "selected_subcat_row_0detail" in st.session_state:
#         st.session_state.selected_subcat_row_0detail = None


# def render_settings0() -> None:
#     """상단 좌우에 [뒤로가기] / [Home] 버튼을 렌더링한다."""
#     left, _, right = st.columns([1, 6, 1])
#     with left:
#         st.button("⬅︎", on_click=back_from_zero_detail, key="btn_back0")
#     with right:
#         st.button("Home", on_click=lambda: go("home"), key="btn_home0")


# # ----------------------------------------------------------------------
# # 파이차트 전처리 유틸
# # ----------------------------------------------------------------------
# def _top_n_with_other(
#     df: pd.DataFrame,
#     code_col: str,
#     name_col: str,
#     n: int = 10,
# ) -> pd.DataFrame:
#     """
#     특허수 기준 상위 N개만 남기고 나머지는 '기타'로 묶는다.
#     """
#     if df.empty:
#         return df

#     df = df.sort_values("특허수", ascending=False).copy()
#     top = df.head(n)
#     rest = df.iloc[n:]

#     if not rest.empty:
#         other_row = {
#             code_col: "OTHER",
#             name_col: "기타",
#             "특허수": int(rest["특허수"].sum()),
#         }
#         top = pd.concat([top, pd.DataFrame([other_row])], ignore_index=True)

#     total = int(top["특허수"].sum())
#     if total > 0:
#         top["비율(%)"] = (top["특허수"] / total * 100).round(1)
#     else:
#         top["비율(%)"] = 0.0

#     top["legend_label"] = top.apply(
#         lambda r: f'{r[name_col]} ({int(r["특허수"]):,}건, {r["비율(%)"]:.1f}%)',
#         axis=1,
#     )

#     return top


# def prepare_midclass_pie_data(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
#     """동업종 기술 중분류별 특허 분포를 파이차트용 데이터로 전처리한다."""
#     if df_raw.empty:
#         empty_cols = ["중분류코드", "중분류명", "특허수", "비율(%)", "legend_label"]
#         return pd.DataFrame(columns=empty_cols), 0

#     df = df_raw.copy()
#     total_patent = int(df["특허수"].sum())

#     if total_patent > 0:
#         df["비율(%)"] = (df["특허수"] / total_patent * 100).round(1)
#     else:
#         df["비율(%)"] = 0.0

#     df["legend_label"] = df.apply(
#         lambda r: f'{r["중분류명"]} ({int(r["특허수"]):,}건, {r["비율(%)"]:.1f}%)',
#         axis=1,
#     )
#     return df, total_patent


# def prepare_subclass_pie_data(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
#     """동업종 기술 소분류별 특허 분포를 파이차트용 데이터로 전처리한다."""
#     if df_raw.empty:
#         empty_cols = ["소분류코드", "소분류명", "특허수", "비율(%)", "legend_label"]
#         return pd.DataFrame(columns=empty_cols), 0

#     df = df_raw.copy()
#     total_patent = int(df["특허수"].sum())

#     if total_patent > 0:
#         df["비율(%)"] = (df["특허수"] / total_patent * 100).round(1)
#     else:
#         df["비율(%)"] = 0.0

#     df["legend_label"] = df.apply(
#         lambda r: f'{r["소분류명"]} ({int(r["특허수"]):,}건, {r["비율(%)"]:.1f}%)',
#         axis=1,
#     )

#     return df, total_patent


# def render_peer_tech_pies(
#     container: "st.delta_generator.DeltaGenerator",
#     df_mid: pd.DataFrame,
#     df_sub: pd.DataFrame,
# ) -> None:
#     """
#     동업종 기술 중분류/소분류별 특허 분포 파이차트를
#     한 행에 나란히 렌더링한다. 각 차트는 상위 10개 + 기타만 표시한다.
#     """
#     if df_mid.empty and df_sub.empty:
#         container.info("표시할 동업종 기술 분포 데이터가 없습니다.")
#         return

#     df_mid_top = _top_n_with_other(df_mid, code_col="중분류코드", name_col="중분류명", n=10)
#     df_sub_top = _top_n_with_other(df_sub, code_col="소분류코드", name_col="소분류명", n=10)

#     with container:
#         col_mid, col_sub = st.columns(2)

#         with col_mid:
#             st.markdown("#### 동업종 기술 중분류별 특허 분포")
#             if df_mid_top.empty:
#                 st.info("중분류 데이터가 없습니다.")
#             else:
#                 pie_mid = (
#                     alt.Chart(df_mid_top)
#                     .mark_arc(outerRadius=110)
#                     .encode(
#                         theta=alt.Theta("특허수:Q", title="특허수"),
#                         color=alt.Color(
#                             "legend_label:N",
#                             sort=None,  # 데이터 순서 그대로: 특허수 내림차순 + 기타 마지막
#                             legend=alt.Legend(title=None, labelLimit=0),
#                             scale=alt.Scale(
#                                 range=[
#                                     "#1f77b4",
#                                     "#2ca02c",
#                                     "#17becf",
#                                     "#aec7e8",
#                                     "#98df8a",
#                                     "#ffbb78",
#                                     "#c5b0d5",
#                                     "#9edae5",
#                                     "#c7c7c7",
#                                     "#bcbd22",
#                                     "#7f7f7f",
#                                 ]
#                             ),
#                         ),
#                         tooltip=[
#                             alt.Tooltip("중분류코드:N", title="중분류코드"),
#                             alt.Tooltip("중분류명:N", title="중분류명"),
#                             alt.Tooltip("특허수:Q", title="특허수", format=","),
#                             alt.Tooltip("비율(%):Q", title="비율(%)", format=".1f"),
#                         ],
#                     )
#                 ).properties(width=380, height=300)
#                 st.altair_chart(pie_mid, use_container_width=True)

#         with col_sub:
#             st.markdown("#### 동업종 기술 소분류별 특허 분포")
#             if df_sub_top.empty:
#                 st.info("소분류 데이터가 없습니다.")
#             else:
#                 pie_sub = (
#                     alt.Chart(df_sub_top)
#                     .mark_arc(outerRadius=110)
#                     .encode(
#                         theta=alt.Theta("특허수:Q", title="특허수"),
#                         color=alt.Color(
#                             "legend_label:N",
#                             sort=None,  # 데이터 순서 그대로: 특허수 내림차순 + 기타 마지막
#                             legend=alt.Legend(title=None, labelLimit=0),
#                             scale=alt.Scale(
#                                 range=[
#                                     "#ff7f0e",
#                                     "#d62728",
#                                     "#9467bd",
#                                     "#8c564b",
#                                     "#e377c2",
#                                     "#7f7f7f",
#                                     "#bcbd22",
#                                     "#17becf",
#                                     "#ff9896",
#                                     "#c49c94",
#                                     "#f7b6d2",
#                                 ]
#                             ),
#                         ),
#                         tooltip=[
#                             alt.Tooltip("소분류코드:N", title="소분류코드"),
#                             alt.Tooltip("소분류명:N", title="소분류명"),
#                             alt.Tooltip("특허수:Q", title="특허수", format=","),
#                             alt.Tooltip("비율(%):Q", title="비율(%)", format=".1f"),
#                         ],
#                     )
#                 ).properties(width=380, height=300)
#                 st.altair_chart(pie_sub, use_container_width=True)


# # ----------------------------------------------------------------------
# # 메인 렌더링 함수
# # ----------------------------------------------------------------------
# def render_company_0_detail() -> None:
#     """특허 0건 기업을 위한 상세 페이지 렌더링 함수."""
#     applicant_corp_no = st.session_state.get("applicant_corp_no", "")
#     applicant_name = st.session_state.get("applicant_name", "")

#     # 기업이 바뀌면 선택 소분류 초기화
#     prev_corp = st.session_state.get("company_0_detail_init_corp")
#     if prev_corp != applicant_corp_no:
#         st.session_state.company_0_detail_init_corp = applicant_corp_no
#         st.session_state.selected_subcat_row_0detail = None

#     render_settings0()
#     st.title("기업혁신성장 보고서 (BIGx)")

#     if not applicant_corp_no:
#         st.warning("신청기업 법인번호 정보가 없습니다. 기업 화면에서 다시 선택해 주세요.")
#         return

#     # 스타일
#     st.markdown(
#         """
#         <style>
#         .sec-banner{display:flex;align-items:center;margin:8px 0 16px 0;}
#         .sec-label{background:#5b9bd5;color:#fff;padding:10px 16px;font-weight:700;font-size:22px;letter-spacing:1px;border-radius:2px;line-height:1;}
#         .sec-rule{flex:1;height:6px;background:#5b9bd5;margin-left:12px;border-radius:2px;}
#         @media (max-width:480px){.sec-label{font-size:18px;padding:8px 12px}.sec-rule{height:5px}}

#         .y-wrap{display:flex;flex-direction:column;gap:12px;}
#         .y-box{margin-top:8px;padding:14px 16px;border:2px solid #666;border-radius:10px;background:#f2f6f9;}
#         .y-title{display:flex;align-items:center;gap:10px;font-size:18px;font-weight:800;color:#000;}
#         .y-sub{margin-top:4px;font-size:14px;font-weight:500;color:#000;}

#         .y-grid{display:grid;grid-template-columns:180px 1fr;gap:8px 14px;margin-top:10px;}
#         .y-grid .y-field,
#         .y-grid .y-value{font-size:16px;line-height:1.5;color:#000;}
#         .y-field{font-weight:700;}
#         .y-value{font-weight:600;}

#         .y-muted{color:#000 !important;font-size:inherit !important;font-weight:inherit !important;}
#         </style>
#         """,
#         unsafe_allow_html=True,
#     )

#     # ------------------------------------------------------------------
#     # 1) 업종 기본 정보
#     # ------------------------------------------------------------------
#     info_df = send_sql(
#         Q.q_industry_basic_info(),
#         params={"applicant_corp_no": applicant_corp_no},
#     )

#     if info_df.empty:
#         st.warning("신청기업의 업종 정보를 찾을 수 없습니다.")
#         return

#     row = info_df.iloc[0]
#     ind_code = str(row.get("업종코드_10차", ""))
#     ind_name = str(row.get("업종명_10차", ""))
#     peer_company_cnt = int(row.get("동업종_기업수", 0) or 0)
#     peer_patent_cnt = int(row.get("동업종_특허수", 0) or 0)

#     st.markdown(
#         """
#         <div class="sec-banner">
#           <div class="sec-label">동업종 내 기술분포</div>
#           <div class="sec-rule"></div>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )

#     info_html = f"""
# <div class="y-wrap">
#   <div class="y-box">
#     <div class="y-title">
#       <span>신청기업 및 업종 개요</span>
#     </div>
#     <div class="y-grid">
#       <div class="y-field">신청기업</div>
#       <div class="y-value">{applicant_name or '-'}</div>
#       <div class="y-field">업종</div>
#       <div class="y-value">{ind_name} ({ind_code})</div>
#       <div class="y-field">동업종 기업 수</div>
#       <div class="y-value">{peer_company_cnt:,}개 기업</div>
#       <div class="y-field">동업종 특허 수</div>
#       <div class="y-value">{peer_patent_cnt:,}건</div>
#     </div>
#   </div>
# </div>
# """
#     st.markdown(info_html, unsafe_allow_html=True)

#     # ------------------------------------------------------------------
#     # 2) 동업종 기술 분포 파이차트
#     # ------------------------------------------------------------------
#     pie_container = st.container()

#     df_mid_raw = send_sql(
#         Q.q_industry_mid_dist(),
#         params={"applicant_corp_no": applicant_corp_no},
#     )
#     df_mid_pie, _ = prepare_midclass_pie_data(df_mid_raw)

#     df_sub_raw = send_sql(
#         Q.q_industry_sub_dist(),
#         params={"applicant_corp_no": applicant_corp_no},
#     )
#     df_sub_pie, _ = prepare_subclass_pie_data(df_sub_raw)

#     render_peer_tech_pies(pie_container, df_mid_pie, df_sub_pie)

#     # ------------------------------------------------------------------
#     # 2-1) AI 판단근거 (표 선택 전, 상위 3개 소분류 기준)
#     # ------------------------------------------------------------------
#     st.markdown(
#         """
#         <div class="sec-banner" style="margin-top:26px;">
#           <div class="sec-label">AI 판단근거</div>
#           <div class="sec-rule"></div>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )

#     if df_sub_raw.empty:
#         # 소분류 데이터가 없으면 AI 판단근거도 산출 불가
#         ai_empty_html = """
# <div class="y-box">
#   <div class="y-title">AI 분석 결과 요약</div>
#   <div class="y-sub">
#     동일 업종 내 소분류 특허 데이터가 부족하여,
#     AI 판단근거를 산출할 수 없습니다.
#   </div>
# </div>
# """
#         st.markdown(ai_empty_html, unsafe_allow_html=True)
#         st.info("소분류 데이터가 없습니다.")
#         return

#     # 상위 3개 소분류 기준으로 문장 생성
#     df_sub_sorted = df_sub_raw.copy()
#     df_sub_sorted["특허수"] = df_sub_sorted["특허수"].fillna(0).astype(int)
#     df_sub_sorted = df_sub_sorted.sort_values("특허수", ascending=False)
#     top3 = df_sub_sorted.head(3)

#     names_counts: list[tuple[str, int]] = []
#     for _, r in top3.iterrows():
#         name = str(r.get("소분류명", "") or "")
#         cnt = int(r.get("특허수", 0) or 0)
#         if not name:
#             continue
#         names_counts.append((name, cnt))

#     if names_counts:
#         parts = [f"{name}에 {cnt:,}건" for name, cnt in names_counts]
#         middle = ", ".join(parts)
#         summary_sentence = (
#             f"AI 분석결과 귀사와 동일한 업종을 영위하고 있는 기업들은 "
#             f"{middle}의 특허를 보유하고 있는 것으로 확인되었습니다."
#         )
#     else:
#         summary_sentence = (
#             "AI 분석결과 귀사와 동일한 업종을 영위하고 있는 기업들의 "
#             "상위 소분류별 특허 정보를 확인할 수 없습니다."
#         )

#     ai_overall_html = f"""
# <div class="y-box">
#   <div class="y-title">AI 분석 결과 요약</div>
#   <div class="y-sub">
#     국내 등록특허에 대한 AI 분석 결과를 바탕으로 분류된 기술분야를 업종별로 재분석하여
#     귀사와 동일한 업종에 포함되어 있는 기술분야를 나타낸 것입니다.
#   </div>
#   <div class="y-sub" style="margin-top:6px;">
#     <span class="y-muted">{summary_sentence}</span>
#   </div>
# </div>
# """
#     st.markdown(ai_overall_html, unsafe_allow_html=True)

#     # ------------------------------------------------------------------
#     # 2-2) 소분류 선택 AgGrid
#     # ------------------------------------------------------------------
#     st.markdown(
#         """
#         <div class="sec-banner" style="margin-top:26px;">
#           <div class="sec-label">소분류 선택</div>
#           <div class="sec-rule"></div>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )

#     selected_subcat_row = None

#     if AgGrid is None or GridOptionsBuilder is None:
#         st.error("st-aggrid 설치 필요: pip install streamlit-aggrid")
#         return

#     df_list = df_sub_raw[["소분류명", "소분류코드", "특허수"]].copy()
#     df_list["특허수"] = df_list["특허수"].fillna(0).astype(int)
#     df_list["특허수(표시)"] = df_list["특허수"].map(lambda x: f"{x:,}건")

#     disp_df = df_list[["소분류명", "소분류코드", "특허수(표시)"]].rename(
#         columns={"특허수(표시)": "특허수"}
#     )

#     gb = GridOptionsBuilder.from_dataframe(disp_df)
#     gb.configure_default_column(
#         resizable=True,
#         sortable=True,
#         filter=True,
#         flex=1,
#         minWidth=150,
#     )
#     gb.configure_column("소분류명", flex=2, minWidth=260)
#     gb.configure_column("소분류코드", flex=1, minWidth=180)
#     gb.configure_column("특허수", flex=1, minWidth=140)
#     gb.configure_selection(selection_mode="single", use_checkbox=True)
#     gb.configure_grid_options(domLayout="normal", rowSelection="single")

#     grid = AgGrid(
#         disp_df,
#         gridOptions=gb.build(),
#         update_mode=GridUpdateMode.SELECTION_CHANGED,
#         fit_columns_on_grid_load=True,
#         height=500,
#         theme="alpine",
#         key="subcat_grid_0detail",
#     )

#     sel = grid.get("selected_rows", [])
#     if sel:
#         st.session_state.selected_subcat_row_0detail = sel[0]

#     selected_subcat_row = st.session_state.get("selected_subcat_row_0detail")

#     # 선택 전이면 여기서 종료
#     if not selected_subcat_row:
#         st.info("선택한 소분류 기준의 상세 내용을 보려면 위 소분류 목록에서 한 개를 선택해 주세요.")
#         return

#     selected_sub_code = str(selected_subcat_row.get("소분류코드", ""))
#     selected_sub_name = str(selected_subcat_row.get("소분류명", ""))

#     # 선택한 소분류 데이터 존재 여부 확인
#     sel_row_df = df_sub_raw[df_sub_raw["소분류코드"] == selected_sub_code]
#     if sel_row_df.empty:
#         st.info("선택한 소분류에 대한 데이터가 없습니다. 위 표에서 다시 선택해 주세요.")
#         return

#     # ------------------------------------------------------------------
#     # 3) 선택 소분류 설명 + 매출 규모별 기업/특허 현황
#     # ------------------------------------------------------------------
#     df_top_info = send_sql(
#         Q.q_top_subcat_info(),
#         params={"sel_subcat_code": selected_sub_code},
#     )
#     if not df_top_info.empty:
#         top_sub_desc = str(df_top_info.iloc[0].get("소분류설명", "") or "")
#     else:
#         top_sub_desc = ""

#     if not top_sub_desc.strip():
#         top_sub_desc = (
#             f"{selected_sub_name} ({selected_sub_code}) 기술분야에 대한 상세 설명이 등록되어 있지 않습니다."
#         )

#     st.markdown(
#         f"""
#         <div class="sec-banner" style="margin-top:26px;">
#           <div class="sec-label">
#             {selected_sub_name} ({selected_sub_code})
#           </div>
#           <div class="sec-rule"></div>
#         </div>
#         """,
#         unsafe_allow_html=True,
#     )

#     intro_html = f"""
# <div class="y-box">
#   <div class="y-title">설명</div>
#   <div class="y-sub">
#     <span class="y-muted">{top_sub_desc}</span>
#   </div>
# </div>
# """
#     st.markdown(intro_html, unsafe_allow_html=True)

#     df_sales = send_sql(
#         Q.q_top_subcat_sales_dist(),
#         params={"sel_subcat_code": selected_sub_code},
#     )

#     if df_sales.empty:
#         st.info("선택한 기술분야에 대한 매출 규모별 데이터가 없습니다.")
#         return

#     dfA, dfB, company_cnt, patent_cnt = prepare_pie_data(df_sales)

#     charts_container = st.container()
#     legend_container = st.container()

#     if not dfA.empty and not dfB.empty:
#         render_pie_charts(charts_container, dfA, dfB)

#     render_group_legend(legend_container)



# -*- coding: utf-8 -*-
"""
views.company_0_detail
======================

특허가 0건인 기업을 위한
'동업종 내 기술분포' 전용 상세 화면 모듈.

역할
----

* 신청기업의 업종(10차) 정보를 기준으로 같은 업종에 속한 기업들의 특허를 분석한다.
* 동업종 내 기술분포(중분류/소분류 파이차트)를 한 줄에 표시한다.
* 소분류 선택용 AgGrid를 제공한다.
* 사용자가 소분류를 선택하면 기술 상세(tech_detail)로 이동한다.

세션 상태
---------

* ``st.session_state["applicant_corp_no"]`` : 신청 기업 법인번호_ENC
* ``st.session_state["applicant_name"]``     : 신청 기업명
* ``st.session_state["page"]``              : 상위 라우팅용 페이지 키
* ``st.session_state["selected_company_row"]`` : company 화면의 선택 기업 (뒤로가기 시 초기화)
* ``st.session_state["pending_nav"]``       : 라우팅 플래그
* ``st.session_state["selected_subcat_row_0detail"]`` : 0건 상세에서 선택한 소분류 행
* ``st.session_state["company_0_detail_init_corp"]``  : 기업 변경 감지용
* ``st.session_state["company_0_last_nav_subcode"]``  : 동일 소분류로 rerun 루프 방지용
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
import altair as alt

from core.db import send_sql
from main.sql import sql_company_0_detail as Q

# AgGrid
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
except Exception:
    AgGrid = None
    GridOptionsBuilder = None
    GridUpdateMode = None


# ----------------------------------------------------------------------
# 공통 네비게이션 헬퍼
# ----------------------------------------------------------------------
def go(page: str) -> None:
    """현재 페이지 상태를 변경한다."""
    st.session_state.page = page


def back_from_zero_detail() -> None:
    """
    특허 0건 상세 화면에서 '뒤로가기' 눌렀을 때 호출되는 함수.

    - 페이지를 company로 돌리고
    - selected_company_row / pending_nav / selected_subcat_row_0detail 등을 초기화한다.
    """
    st.session_state.page = "company"

    if "selected_company_row" in st.session_state:
        st.session_state.selected_company_row = None

    if "pending_nav" in st.session_state:
        st.session_state.pending_nav = None

    if "selected_subcat_row_0detail" in st.session_state:
        st.session_state.selected_subcat_row_0detail = None

    # rerun 루프 방지 가드 초기화
    if "company_0_last_nav_subcode" in st.session_state:
        st.session_state.company_0_last_nav_subcode = None


def render_settings0() -> None:
    """상단 좌우에 [뒤로가기] / [Home] 버튼을 렌더링한다."""
    left, _, right = st.columns([1, 6, 1])
    with left:
        st.button("⬅︎", on_click=back_from_zero_detail, key="btn_back0")
    with right:
        st.button("Home", on_click=lambda: go("home"), key="btn_home0")


# ----------------------------------------------------------------------
# 파이차트 전처리 유틸
# ----------------------------------------------------------------------
def _top_n_with_other(
    df: pd.DataFrame,
    code_col: str,
    name_col: str,
    n: int = 10,
) -> pd.DataFrame:
    """
    특허수 기준 상위 N개만 남기고 나머지는 '기타'로 묶는다.
    """
    if df.empty:
        return df

    df = df.sort_values("특허수", ascending=False).copy()
    top = df.head(n)
    rest = df.iloc[n:]

    if not rest.empty:
        other_row = {
            code_col: "OTHER",
            name_col: "기타",
            "특허수": int(rest["특허수"].sum()),
        }
        top = pd.concat([top, pd.DataFrame([other_row])], ignore_index=True)

    total = int(top["특허수"].sum())
    if total > 0:
        top["비율(%)"] = (top["특허수"] / total * 100).round(1)
    else:
        top["비율(%)"] = 0.0

    top["legend_label"] = top.apply(
        lambda r: f'{r[name_col]} ({int(r["특허수"]):,}건, {r["비율(%)"]:.1f}%)',
        axis=1,
    )

    return top


def prepare_midclass_pie_data(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """동업종 기술 중분류별 특허 분포를 파이차트용 데이터로 전처리한다."""
    if df_raw.empty:
        empty_cols = ["중분류코드", "중분류명", "특허수", "비율(%)", "legend_label"]
        return pd.DataFrame(columns=empty_cols), 0

    df = df_raw.copy()
    total_patent = int(df["특허수"].sum())

    if total_patent > 0:
        df["비율(%)"] = (df["특허수"] / total_patent * 100).round(1)
    else:
        df["비율(%)"] = 0.0

    df["legend_label"] = df.apply(
        lambda r: f'{r["중분류명"]} ({int(r["특허수"]):,}건, {r["비율(%)"]:.1f}%)',
        axis=1,
    )
    return df, total_patent


def prepare_subclass_pie_data(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """동업종 기술 소분류별 특허 분포를 파이차트용 데이터로 전처리한다."""
    if df_raw.empty:
        empty_cols = ["소분류코드", "소분류명", "특허수", "비율(%)", "legend_label"]
        return pd.DataFrame(columns=empty_cols), 0

    df = df_raw.copy()
    total_patent = int(df["특허수"].sum())

    if total_patent > 0:
        df["비율(%)"] = (df["특허수"] / total_patent * 100).round(1)
    else:
        df["비율(%)"] = 0.0

    df["legend_label"] = df.apply(
        lambda r: f'{r["소분류명"]} ({int(r["특허수"]):,}건, {r["비율(%)"]:.1f}%)',
        axis=1,
    )

    return df, total_patent


def render_peer_tech_pies(
    container: "st.delta_generator.DeltaGenerator",
    df_mid: pd.DataFrame,
    df_sub: pd.DataFrame,
) -> None:
    """
    동업종 기술 중분류/소분류별 특허 분포 파이차트를
    한 행에 나란히 렌더링한다. 각 차트는 상위 10개 + 기타만 표시한다.
    """
    if df_mid.empty and df_sub.empty:
        container.info("표시할 동업종 기술 분포 데이터가 없습니다.")
        return

    df_mid_top = _top_n_with_other(df_mid, code_col="중분류코드", name_col="중분류명", n=10)
    df_sub_top = _top_n_with_other(df_sub, code_col="소분류코드", name_col="소분류명", n=10)

    with container:
        col_mid, col_sub = st.columns(2)

        with col_mid:
            st.markdown("#### 기술보유현황-중분류")
            if df_mid_top.empty:
                st.info("중분류 데이터가 없습니다.")
            else:
                pie_mid = (
                    alt.Chart(df_mid_top)
                    .mark_arc(outerRadius=110)
                    .encode(
                        theta=alt.Theta("특허수:Q", title="특허수"),
                        color=alt.Color(
                            "legend_label:N",
                            sort=None,
                            legend=alt.Legend(title=None, labelLimit=0),
                            scale=alt.Scale(
                                range=[
                                    "#1f77b4",
                                    "#2ca02c",
                                    "#17becf",
                                    "#aec7e8",
                                    "#98df8a",
                                    "#ffbb78",
                                    "#c5b0d5",
                                    "#9edae5",
                                    "#c7c7c7",
                                    "#bcbd22",
                                    "#7f7f7f",
                                ]
                            ),
                        ),
                        tooltip=[
                            alt.Tooltip("중분류코드:N", title="중분류코드"),
                            alt.Tooltip("중분류명:N", title="중분류명"),
                            alt.Tooltip("특허수:Q", title="특허수", format=","),
                            alt.Tooltip("비율(%):Q", title="비율(%)", format=".1f"),
                        ],
                    )
                ).properties(width=380, height=300)
                st.altair_chart(pie_mid, use_container_width=True)

        with col_sub:
            st.markdown("#### 기술보유현황-소분류")
            if df_sub_top.empty:
                st.info("소분류 데이터가 없습니다.")
            else:
                pie_sub = (
                    alt.Chart(df_sub_top)
                    .mark_arc(outerRadius=110)
                    .encode(
                        theta=alt.Theta("특허수:Q", title="특허수"),
                        color=alt.Color(
                            "legend_label:N",
                            sort=None,
                            legend=alt.Legend(title=None, labelLimit=0),
                            scale=alt.Scale(
                                range=[
                                    "#ff7f0e",
                                    "#d62728",
                                    "#9467bd",
                                    "#8c564b",
                                    "#e377c2",
                                    "#7f7f7f",
                                    "#bcbd22",
                                    "#17becf",
                                    "#ff9896",
                                    "#c49c94",
                                    "#f7b6d2",
                                ]
                            ),
                        ),
                        tooltip=[
                            alt.Tooltip("소분류코드:N", title="소분류코드"),
                            alt.Tooltip("소분류명:N", title="소분류명"),
                            alt.Tooltip("특허수:Q", title="특허수", format=","),
                            alt.Tooltip("비율(%):Q", title="비율(%)", format=".1f"),
                        ],
                    )
                ).properties(width=380, height=300)
                st.altair_chart(pie_sub, use_container_width=True)


# ----------------------------------------------------------------------
# 메인 렌더링 함수
# ----------------------------------------------------------------------
def render_company_0_detail() -> None:
    """특허 0건 기업을 위한 상세 페이지 렌더링 함수."""
    applicant_corp_no = st.session_state.get("applicant_corp_no", "")
    applicant_name = st.session_state.get("applicant_name", "")

    # 기업이 바뀌면 선택 소분류/가드 초기화
    prev_corp = st.session_state.get("company_0_detail_init_corp")
    if prev_corp != applicant_corp_no:
        st.session_state.company_0_detail_init_corp = applicant_corp_no
        st.session_state.selected_subcat_row_0detail = None
        st.session_state.company_0_last_nav_subcode = None

    render_settings0()
    st.title("기업혁신성장 보고서 (BIGx)")

    if not applicant_corp_no:
        st.warning("신청기업 법인번호 정보가 없습니다. 기업 화면에서 다시 선택해 주세요.")
        return

    # 스타일
    st.markdown(
        """
        <style>
        .sec-banner{display:flex;align-items:center;margin:8px 0 16px 0;}
        .sec-label{background:#5b9bd5;color:#fff;padding:10px 16px;font-weight:700;font-size:22px;letter-spacing:1px;border-radius:2px;line-height:1;}
        .sec-rule{flex:1;height:6px;background:#5b9bd5;margin-left:12px;border-radius:2px;}
        @media (max-width:480px){.sec-label{font-size:18px;padding:8px 12px}.sec-rule{height:5px}}

        .y-wrap{display:flex;flex-direction:column;gap:12px;}
        .y-box{margin-top:8px;padding:14px 16px;border:2px solid #666;border-radius:10px;background:#f2f6f9;}
        .y-title{display:flex;align-items:center;gap:10px;font-size:18px;font-weight:800;color:#000;}
        .y-sub{margin-top:4px;font-size:14px;font-weight:500;color:#000;}

        .y-grid{display:grid;grid-template-columns:180px 1fr;gap:8px 14px;margin-top:10px;}
        .y-grid .y-field,
        .y-grid .y-value{font-size:16px;line-height:1.5;color:#000;}
        .y-field{font-weight:700;}
        .y-value{font-weight:600;}

        .y-muted{color:#000 !important;font-size:inherit !important;font-weight:inherit !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # 1) 업종 기본 정보
    # ------------------------------------------------------------------
    info_df = send_sql(
        Q.q_industry_basic_info(),
        params={"applicant_corp_no": applicant_corp_no},
    )

    if info_df.empty:
        st.warning("신청기업의 업종 정보를 찾을 수 없습니다.")
        return

    row = info_df.iloc[0]
    ind_code = str(row.get("업종코드_10차", ""))
    ind_name = str(row.get("업종명_10차", ""))
    peer_company_cnt = int(row.get("동업종_기업수", 0) or 0)
    peer_patent_cnt = int(row.get("동업종_특허수", 0) or 0)

    st.markdown(
        """
        <div class="sec-banner">
          <div class="sec-label">동업종 내 기술분포</div>
          <div class="sec-rule"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    info_html = f"""
<div class="y-wrap">
  <div class="y-box">
    <div class="y-title">
      <span>신청기업 및 업종 개요</span>
    </div>
    <div class="y-grid">
      <div class="y-field">신청기업</div>
      <div class="y-value">{applicant_name or '-'}</div>
      <div class="y-field">업종</div>
      <div class="y-value">{ind_name} ({ind_code})</div>
      <div class="y-field">동업종 기업 수</div>
      <div class="y-value">{peer_company_cnt:,}개 기업</div>
      <div class="y-field">동업종 특허 수</div>
      <div class="y-value">{peer_patent_cnt:,}건</div>
    </div>
  </div>
</div>
"""
    st.markdown(info_html, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 2) 동업종 기술 분포 파이차트
    # ------------------------------------------------------------------
    pie_container = st.container()

    df_mid_raw = send_sql(
        Q.q_industry_mid_dist(),
        params={"applicant_corp_no": applicant_corp_no},
    )
    df_mid_pie, _ = prepare_midclass_pie_data(df_mid_raw)

    df_sub_raw = send_sql(
        Q.q_industry_sub_dist(),
        params={"applicant_corp_no": applicant_corp_no},
    )
    df_sub_pie, _ = prepare_subclass_pie_data(df_sub_raw)

    render_peer_tech_pies(pie_container, df_mid_pie, df_sub_pie)

    # ------------------------------------------------------------------
    # 2-1) AI 판단근거 (표 선택 전, 상위 3개 소분류 기준)
    # ------------------------------------------------------------------
    st.markdown(
        """
        <div class="sec-banner" style="margin-top:26px;">
          <div class="sec-label">AI 판단근거</div>
          <div class="sec-rule"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df_sub_raw.empty:
        ai_empty_html = """
<div class="y-box">
  <div class="y-title">AI 분석 결과 요약</div>
  <div class="y-sub">
    동일 업종 내 소분류 특허 데이터가 부족하여,
    AI 판단근거를 산출할 수 없습니다.
  </div>
</div>
"""
        st.markdown(ai_empty_html, unsafe_allow_html=True)
        st.info("소분류 데이터가 없습니다.")
        return

    df_sub_sorted = df_sub_raw.copy()
    df_sub_sorted["특허수"] = df_sub_sorted["특허수"].fillna(0).astype(int)
    df_sub_sorted = df_sub_sorted.sort_values("특허수", ascending=False)
    top3 = df_sub_sorted.head(3)

    names_counts: list[tuple[str, int]] = []
    for _, r in top3.iterrows():
        name = str(r.get("소분류명", "") or "")
        cnt = int(r.get("특허수", 0) or 0)
        if not name:
            continue
        names_counts.append((name, cnt))

    if names_counts:
        parts = [f"{name}에 {cnt:,}건" for name, cnt in names_counts]
        middle = ", ".join(parts)
        summary_sentence = (
            f"AI 분석결과 귀사와 동일한 업종을 영위하고 있는 기업들은 "
            f"{middle}의 특허를 보유하고 있는 것으로 확인되었습니다."
        )
    else:
        summary_sentence = (
            "AI 분석결과 귀사와 동일한 업종을 영위하고 있는 기업들의 "
            "상위 소분류별 특허 정보를 확인할 수 없습니다."
        )

    ai_overall_html = f"""
<div class="y-box">
  <div class="y-title">AI 분석 결과 요약</div>
  <div class="y-sub">
    국내 등록특허에 대한 AI 분석 결과를 바탕으로 분류된 기술분야를 업종별로 재분석하여
    귀사와 동일한 업종에 포함되어 있는 기술분야를 나타낸 것입니다.
  </div>
  <div class="y-sub" style="margin-top:6px;">
    <span class="y-muted">{summary_sentence}</span>
  </div>
</div>
"""
    st.markdown(ai_overall_html, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # 2-2) 소분류 선택 AgGrid  -> 선택 즉시 tech_detail로 이동
    # ------------------------------------------------------------------
    st.markdown(
        """
        <div class="sec-banner" style="margin-top:26px;">
          <div class="sec-label">소분류 선택</div>
          <div class="sec-rule"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if AgGrid is None or GridOptionsBuilder is None:
        st.error("st-aggrid 설치 필요: pip install streamlit-aggrid")
        return

    df_list = df_sub_raw[["소분류명", "소분류코드", "특허수"]].copy()
    df_list["특허수"] = df_list["특허수"].fillna(0).astype(int)
    df_list["특허수(표시)"] = df_list["특허수"].map(lambda x: f"{x:,}건")

    disp_df = df_list[["소분류명", "소분류코드", "특허수(표시)"]].rename(
        columns={"특허수(표시)": "특허수"}
    )

    gb = GridOptionsBuilder.from_dataframe(disp_df)
    gb.configure_default_column(
        resizable=True,
        sortable=True,
        filter=True,
        flex=1,
        minWidth=150,
    )
    gb.configure_column("소분류명", flex=2, minWidth=260)
    gb.configure_column("소분류코드", flex=1, minWidth=180)
    gb.configure_column("특허수", flex=1, minWidth=140)
    gb.configure_selection(selection_mode="single", use_checkbox=True)
    gb.configure_grid_options(domLayout="normal", rowSelection="single")

    grid = AgGrid(
        disp_df,
        gridOptions=gb.build(),
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        fit_columns_on_grid_load=True,
        height=500,
        theme="alpine",
        key="subcat_grid_0detail",
    )

    sel = grid.get("selected_rows", [])
    if sel:
        st.session_state.selected_subcat_row_0detail = sel[0]

    selected_subcat_row = st.session_state.get("selected_subcat_row_0detail")

    if not selected_subcat_row:
        st.info("기술분석 전체를 보려면 위 소분류 목록에서 한 개를 선택해 주세요.")
        return

    selected_sub_code = str(selected_subcat_row.get("소분류코드", "") or "").strip()
    selected_sub_name = str(selected_subcat_row.get("소분류명", "") or "").strip()

    # 선택한 소분류 데이터 존재 여부 확인
    sel_row_df = df_sub_raw[df_sub_raw["소분류코드"] == selected_sub_code]
    if sel_row_df.empty:
        st.info("선택한 소분류에 대한 데이터가 없습니다. 위 표에서 다시 선택해 주세요.")
        return

    # ✅ 소분류 선택 즉시 tech_detail로 이동 (rerun 루프 방지)
    last = st.session_state.get("company_0_last_nav_subcode")
    if selected_sub_code and last != selected_sub_code:
        st.session_state.company_0_last_nav_subcode = selected_sub_code
        st.session_state.tech_detail_back_page = "company_0_detail"
        st.session_state.tech_selected = {
            "code": selected_sub_code,
            "name": selected_sub_name,
            "label": f"{selected_sub_code} - {selected_sub_name}",
        }
        # st.session_state.pending_nav = "tech_detail"
        
        # 바꿔야 함
        # st.session_state.selected_subcat_code = selected_sub_code
        # st.session_state.selected_subcat_name = '인공지능'
        # st.session_state.patent_num = False


        st.session_state.pending_nav = "tech_detail_0"
        st.session_state.company_num = True
        st.rerun()

    # (가드에 걸렸다면 여기까지. 이 페이지에서는 더 그리지 않음)
    return
