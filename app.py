# app.py
import streamlit as st
import pandas as pd
import logic
import os

# -------------------------------------------------------------------------
# 1. 페이지 기본 설정 & 세션 상태 초기화 & 다이얼로그 함수 정의
# -------------------------------------------------------------------------
st.set_page_config(page_title="AI 한식 재료 추천", layout="wide")
st.title("🍳 AI 식재료 대체 추천 대시보드")

if 'voted_logs' not in st.session_state:
    st.session_state['voted_logs'] = set()

def format_saving(score, is_multi=False):
    prefix = "총 " if is_multi else ""
    if score > 0: return f"🟢 {prefix}+{score}단계 (절감)"
    elif score < 0: return f"🔴 {prefix}{score}단계 (비쌈)"
    else: return "⚪ 동일 수준"

# 로직 설명 팝업창 (다이얼로그) 함수
@st.dialog("🧠 AI 추천 알고리즘 작동 원리")
def show_logic_dialog():
    st.markdown("""
    본 서비스는 자연어 처리(NLP) 기술을 활용하여 단순히 이름이 비슷한 재료가 아닌, **'의미'와 '맥락'이 통하는 최적의 대체 재료**를 찾아냅니다.
    """)
    
    # 이미지 파일 존재 여부 확인 후 표시
    if os.path.exists("logic_diagram.png"):
        st.image("logic_diagram.png", use_container_width=True)
    else:
        st.error("로직 다이어그램 이미지 파일(logic_diagram.png)을 찾을 수 없습니다. 프로젝트 폴더에 이미지를 추가해주세요.")

    st.markdown("""
    ---
    #### **주요 기술 설명**
    
    **1. 의미 파악 (Word2Vec)**
    * 수십만 개의 레시피를 학습하여 재료 간의 의미적 유사성을 파악합니다.
    * *예: '돼지고기'는 '소고기', '스팸'과 성질이 비슷하다.*

    **2. 문맥 이해 (Doc2Vec)**
    * 현재 요리의 전체적인 재료 구성(문맥)을 벡터화하여, 그 문맥에 자연스럽게 어울리는지 판단합니다.
    * *예: '미역국' 문맥에는 '스팸'보다 '조개'가 더 어울린다.*

    **3. 통계적 적합성 (Ver.1 전용)**
    * 실제 데이터베이스에서 해당 조리법(예: 끓이기)이나 카테고리(예: 국/탕)에 해당 재료가 얼마나 자주 사용되는지 통계적으로 분석합니다.
    """)

# -------------------------------------------------------------------------
# 2. 사이드바 UI (모드 선택 및 가중치 설정)
# -------------------------------------------------------------------------
with st.sidebar:
    st.header("🎛️ 컨트롤 패널")
    selected_mode = st.radio("모드 선택", ["📚 Ver.1 기존 레시피 DB 검색", "✨ Ver.2 나만의 재료 입력 (커스텀)"], index=0)
    st.divider()
    st.subheader("⚖️ 가중치 설정")
    is_v1 = selected_mode == "📚 Ver.1 기존 레시피 DB 검색"
    w_w2v = st.slider("맛·성질 (Word2Vec)", 0.0, 5.0, 5.0, 0.5, help="재료 자체의 의미적 유사도 비중입니다.")
    w_d2v = st.slider("문맥 (Doc2Vec)", 0.0, 5.0, 1.0, 0.5, help="전체 재료 조합과의 어울림 비중입니다.")
    w_method = st.slider("조리법 통계 (Ver.1 전용)", 0.0, 5.0, 1.0, 0.5, disabled=not is_v1, help="Ver.1 모드에서만 작동합니다.")
    w_cat = st.slider("카테고리 통계 (Ver.1 전용)", 0.0, 5.0, 1.0, 0.5, disabled=not is_v1, help="Ver.1 모드에서만 작동합니다.")
    if not is_v1: st.caption("💡 커스텀 모드에서는 통계 가중치가 적용되지 않습니다.")
    st.divider()
    if st.button("❓ 추천 로직 자세히 보기 (팝업)", use_container_width=True):
        show_logic_dialog()

# -------------------------------------------------------------------------
# 3. 메인 UI (선택된 모드에 따라 내용 표시)
# -------------------------------------------------------------------------
col_main, _ = st.columns([0.9, 0.1])
with col_main:
    # =========================================
    # [MODE 1] Ver.1 기존 레시피 DB 검색 모드
    # =========================================
    if selected_mode == "📚 Ver.1 기존 레시피 DB 검색":
        st.markdown("""<div style="background-color: #f0f8ff; padding: 15px; border-radius: 10px; margin-bottom: 20px;"><h4 style="margin:0; color:#0066cc;">[Ver.1] 레시피 데이터베이스에서 검색</h4><p style="margin:5px 0 0 0; font-size:14px;">학습된 12만여 개의 레시피 중 하나를 선택하여 분석합니다. 모든 통계 점수가 활용됩니다.</p></div>""", unsafe_allow_html=True)
        search_keyword = st.text_input("🍽️ 요리명 검색 (키워드 입력 후 엔터)", placeholder="예: 된장찌개")
        final_dish_name = None

        if search_keyword:
            # 1. 정확히 일치하는 요리명 찾기
            exact_match = logic.df[logic.df['요리명'] == search_keyword]
            exact_name = exact_match['요리명'].iloc[0] if not exact_match.empty else None

            # 2. 포함하는 요리명 찾기 (정확히 일치하는 것 제외)
            candidates = logic.df[logic.df['요리명'].str.contains(search_keyword, na=False, case=False)]
            if exact_name:
                candidates = candidates[candidates['요리명'] != exact_name]
            
            candidate_names = candidates['요리명'].unique().tolist()
            candidate_names = sorted(candidate_names)[:30] # 상위 30개만

            # 3. 옵션 구성: 정확한 일치가 있으면 가장 위에 배치
            options = []
            if exact_name:
                options.append(exact_name)
            options.extend(candidate_names)
            
            if not options:
                st.warning(f"🔍 '{search_keyword}'가 포함된 요리명을 찾을 수 없습니다.")
            else:
                # 정확한 일치가 있으면 그것을 기본 선택값으로 함
                index_to_select = 0 if exact_name else None
                
                label_msg = f"🔎 '{search_keyword}' 검색 결과 ({len(options)}개)"
                if exact_name:
                    label_msg += " - 정확한 요리명이 발견되었습니다!"

                selected_option = st.selectbox(label_msg, options, index=index_to_select)
                final_dish_name = selected_option

        if final_dish_name:
            st.success(f"✅ 선택된 요리: **{final_dish_name}**")
            cands = logic.df[logic.df['요리명'] == final_dish_name]
            cands = cands.head(10).reset_index(drop=True)
            if cands.empty: st.error("❌ 해당 요리의 레시피 정보를 불러올 수 없습니다.")
            else:
                st.divider()
                options = {}
                for _, r in cands.iterrows():
                    ing_sum = ', '.join(r['재료토큰'])
                    preview_text = ing_sum[:100] + "..." if len(ing_sum) > 100 else ing_sum
                    label = f"[{r['요리방법별명']}] {r['요리명']} (ID:{r['레시피일련번호']}) - {preview_text}"
                    options[label] = r['레시피일련번호']
                selected_label = st.selectbox("📜 분석할 레시피를 선택하세요", list(options.keys()))
                recipe_id = options[selected_label]
                
                c1, c2 = st.columns(2)
                with c1: target_str = st.text_input("🎯 바꿀 재료", placeholder="돼지고기, 양파")
                with c2: stop_str = st.text_input("🚫 제거할 문구", placeholder="약간, 시판용")
                
                if target_str:
                    targets = [t.strip() for t in target_str.split(',') if t.strip()]
                    stops = [s.strip() for s in stop_str.split(',') if s.strip()]
                    if not targets: st.warning("타겟 재료를 입력해주세요.")
                    else:
                        st.divider()
                        final_recommendations = []
                        has_result = False
                        if len(targets) == 1:
                            st.subheader("🔹 단일 재료 대체 추천 (DB 기반)")
                            t = targets[0]
                            res = logic.substitute_single(recipe_id, t, stops, w_w2v, w_d2v, w_method, w_cat, topn=5)
                            st.markdown(f"**{t}** 대체 결과")
                            if not res.empty:
                                has_result = True
                                final_recommendations = res['대체재료'].head(3).tolist()
                                display_df = res[['대체재료', '최종점수', 'saving_score']].copy()
                                display_df['예상 원가변동'] = display_df['saving_score'].apply(lambda x: format_saving(x))
                                display_df = display_df[['대체재료', '최종점수', '예상 원가변동']]
                                display_df.columns = ['추천재료', '적합도', '예상 원가변동']
                                st.dataframe(display_df.style.format("{:.1%}", subset=['적합도']).background_gradient(cmap='Greens', subset=['적합도']), use_container_width=True, hide_index=True)
                            else: st.warning("결과 없음")
                        elif len(targets) > 1:
                            st.subheader("🧩 최적의 재료 조합 (DB 기반 다중 대체)")
                            multi_res = logic.substitute_multi(recipe_id, targets, stops, w_w2v, w_d2v, w_method, w_cat)
                            if multi_res:
                                has_result = True
                                final_recommendations = [", ".join(subs) for subs, score, saving in multi_res]
                                m_df = pd.DataFrame([(f"{', '.join(subs)}", score, format_saving(saving, True)) for subs, score, saving in multi_res], columns=['추천 조합', '종합 점수', '예상 원가변동 합계'])
                                st.dataframe(m_df.style.format("{:.1%}", subset=['종합 점수']).background_gradient(cmap='Blues', subset=['종합 점수']), use_container_width=True, hide_index=True)
                            else: st.info("조합을 찾을 수 없습니다.")
                        if has_result:
                            current_state = f"DB_{dish_name}_{target_str}_{stop_str}_{w_w2v}_{w_d2v}_{w_method}_{w_cat}_{final_recommendations}"
                            if 'last_log_state' not in st.session_state: st.session_state['last_log_state'] = ""
                            if st.session_state['last_log_state'] != current_state:
                                log_id = logic.save_log_to_db(dish_name, target_str, stops, w_w2v, w_d2v, w_method, w_cat, rec_list=final_recommendations, is_custom=False)
                                st.session_state['current_log_id'] = log_id
                                st.session_state['last_log_state'] = current_state
                            if 'current_log_id' in st.session_state and st.session_state['current_log_id']:
                                cl_id = st.session_state['current_log_id']
                                is_voted = cl_id in st.session_state['voted_logs']
                                st.write(""); b1, b2, _ = st.columns([0.2, 0.2, 0.6])
                                if is_voted: b1.success("✅ 평가 완료!"); b2.write("")
                                else:
                                    b1.button("👍 만족해요", key="btn_sat_db", use_container_width=True, on_click=lambda: (logic.update_feedback_in_db(cl_id, "satisfy"), st.session_state['voted_logs'].add(cl_id), st.toast("감사합니다!")))
                                    b2.button("👎 아쉬워요", key="btn_dis_db", use_container_width=True, on_click=lambda: (logic.update_feedback_in_db(cl_id, "dissatisfy"), st.session_state['voted_logs'].add(cl_id), st.toast("의견 감사합니다.")))

    # =========================================
    # [MODE 2] Ver.2 커스텀 재료 입력 모드
    # =========================================
    elif selected_mode == "✨ Ver.2 나만의 재료 입력 (커스텀)":
        st.markdown("""<div style="background-color: #fff5f0; padding: 15px; border-radius: 10px; margin-bottom: 20px;"><h4 style="margin:0; color:#cc5500;">[Ver.2] 나만의 재료 리스트 입력</h4><p style="margin:5px 0 0 0; font-size:14px;">냉장고 속 재료들을 직접 입력하세요. 문맥을 실시간으로 분석하여 추천합니다. (통계 점수 제외)</p></div>""", unsafe_allow_html=True)
        
        st.markdown("##### 🏷️ 요리명 입력 (참고용)")
        search_keyword_v2 = st.text_input("키워드 입력 후 엔터 (예: 볶음밥) - 선택사항", key="v2_search")
        custom_dish_name = search_keyword_v2

        if search_keyword_v2:
            # 1. 정확히 일치하는 요리명 찾기
            exact_match_v2 = logic.df[logic.df['요리명'] == search_keyword_v2]
            exact_name_v2 = exact_match_v2['요리명'].iloc[0] if not exact_match_v2.empty else None

            # 2. 포함하는 요리명 찾기
            candidates_v2 = logic.df[logic.df['요리명'].str.contains(search_keyword_v2, na=False, case=False)]
            if exact_name_v2:
                candidates_v2 = candidates_v2[candidates_v2['요리명'] != exact_name_v2]
            
            candidate_names_v2 = candidates_v2['요리명'].unique().tolist()
            candidate_names_v2 = sorted(candidate_names_v2)[:30]

            # 3. 옵션 구성: 정확한 일치 -> 직접 입력 -> 나머지 후보 순
            options_v2 = []
            if exact_name_v2:
                options_v2.append(exact_name_v2)
            options_v2.append("(직접 입력한 이름 사용)")
            options_v2.extend(candidate_names_v2)

            if options_v2:
                index_to_select_v2 = 0 if exact_name_v2 else 0 # 정확한 일치가 있으면 그것을, 없으면 '직접 입력'을 기본값으로

                label_msg_v2 = f"💡 관련 요리명 발견 ({len(options_v2)-1}개)"
                if exact_name_v2:
                    label_msg_v2 += " - 정확한 요리명이 발견되었습니다!"

                selected_option_v2 = st.selectbox(label_msg_v2, options_v2, index=index_to_select_v2, key="v2_select")
                
                if selected_option_v2 == "(직접 입력한 이름 사용)":
                    custom_dish_name = search_keyword_v2
                else:
                    custom_dish_name = selected_option_v2

        st.write("")
        context_str = st.text_area("📝 전체 재료 리스트 (쉼표로 구분)", placeholder="예: 밥, 계란, 대파, 간장, 참기름", height=100, key="v2_context")

        if context_str:
            context_ings_list = [ing.strip() for ing in context_str.split(',') if ing.strip()]
            if not context_ings_list: st.warning("재료를 한 개 이상 입력해주세요.")
            else:
                st.caption(f"인식된 재료 ({len(context_ings_list)}개): {', '.join(context_ings_list)}")
                c1_c, c2_c = st.columns(2)
                with c1_c: target_str_c = st.text_input("🎯 바꿀 재료 (위 리스트 중)", placeholder="예: 계란", key="v2_target")
                with c2_c: stop_str_c = st.text_input("🚫 제거할 문구", placeholder="예: 약간", key="v2_stop")
                if target_str_c:
                    targets_c = [t.strip() for t in target_str_c.split(',') if t.strip()]
                    stops_c = [s.strip() for s in stop_str_c.split(',') if s.strip()]
                    invalid_targets = [t for t in targets_c if t not in context_ings_list]
                    if invalid_targets: st.error(f"다음 재료는 전체 리스트에 없습니다: {', '.join(invalid_targets)}")
                    elif not targets_c: st.warning("바꿀 재료를 입력해주세요.")
                    else:
                        st.divider()
                        final_recommendations_c = []
                        has_result_c = False
                        if len(targets_c) == 1:
                            st.subheader("🔹 단일 재료 대체 추천 (커스텀)")
                            t_c = targets_c[0]
                            res_c = logic.substitute_single_custom(t_c, context_ings_list, stops_c, w_w2v, w_d2v, topn=5)
                            st.markdown(f"**{t_c}** 대체 결과")
                            if not res_c.empty:
                                has_result_c = True
                                final_recommendations_c = res_c['대체재료'].head(3).tolist()
                                display_df_c = res_c[['대체재료', '최종점수', 'saving_score']].copy()
                                display_df_c['예상 원가변동'] = display_df_c['saving_score'].apply(lambda x: format_saving(x))
                                display_df_c = display_df_c[['대체재료', '최종점수', '예상 원가변동']]
                                display_df_c.columns = ['추천재료', '적합도', '예상 원가변동']
                                st.dataframe(display_df_c.style.format("{:.1%}", subset=['적합도']).background_gradient(cmap='Greens', subset=['적합도']), use_container_width=True, hide_index=True)
                            else: st.warning("결과 없음")
                        elif len(targets_c) > 1:
                            st.subheader("🧩 최적의 재료 조합 (커스텀 다중 대체)")
                            multi_res_c = logic.substitute_multi_custom(targets_c, context_ings_list, stops_c, w_w2v, w_d2v)
                            if multi_res_c:
                                has_result_c = True
                                final_recommendations_c = [", ".join(subs) for subs, score, saving in multi_res_c]
                                m_df_c = pd.DataFrame([(f"{', '.join(subs)}", score, format_saving(saving, True)) for subs, score, saving in multi_res_c], columns=['추천 조합', '종합 점수', '예상 원가변동 합계'])
                                st.dataframe(m_df_c.style.format("{:.1%}", subset=['종합 점수']).background_gradient(cmap='Blues', subset=['종합 점수']), use_container_width=True, hide_index=True)
                            else: st.info("조합을 찾을 수 없습니다.")
                        if has_result_c:
                            current_state_c = f"Custom_{custom_dish_name}_{target_str_c}_{stop_str_c}_{w_w2v}_{w_d2v}_{final_recommendations_c}"
                            if 'last_log_state_c' not in st.session_state: st.session_state['last_log_state_c'] = ""
                            if st.session_state['last_log_state_c'] != current_state_c:
                                log_id_c = logic.save_log_to_db(custom_dish_name, target_str_c, stops_c, w_w2v, w_d2v, 0, 0, rec_list=final_recommendations_c, is_custom=True)
                                st.session_state['current_log_id_c'] = log_id_c
                                st.session_state['last_log_state_c'] = current_state_c
                            if 'current_log_id_c' in st.session_state and st.session_state['current_log_id_c']:
                                cl_id_c = st.session_state['current_log_id_c']
                                is_voted_c = cl_id_c in st.session_state['voted_logs']
                                st.write(""); b1_c, b2_c, _ = st.columns([0.2, 0.2, 0.6])
                                if is_voted_c: b1_c.success("✅ 평가 완료!"); b2_c.write("")
                                else:
                                    b1_c.button("👍 만족해요", key="btn_sat_custom", use_container_width=True, on_click=lambda: (logic.update_feedback_in_db(cl_id_c, "satisfy"), st.session_state['voted_logs'].add(cl_id_c), st.toast("감사합니다!")))
                                    b2_c.button("👎 아쉬워요", key="btn_dis_custom", use_container_width=True, on_click=lambda: (logic.update_feedback_in_db(cl_id_c, "dissatisfy"), st.session_state['voted_logs'].add(cl_id_c), st.toast("의견 감사합니다.")))
        else: st.info("👆 전체 재료 리스트를 먼저 입력해주세요.")

# -------------------------------------------------------------------------
# 4. 하단 피드백 영역 (기존 동일)
# -------------------------------------------------------------------------
st.divider()
st.subheader("📢 서비스 의견 보내기")
with st.form("feedback_form"):
    text = st.text_area("개선할 점이나 버그가 있다면 알려주세요!")
    submitted = st.form_submit_button("의견 보내기")
    if submitted:
        if text:
            if logic.save_feedback_to_db(text): st.success("의견 감사합니다!"); st.balloons()
        else: st.warning("내용을 입력해주세요.")
