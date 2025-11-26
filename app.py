# app.py
import streamlit as st
import pandas as pd
# 우리가 만든 logic.py 파일을 임포트합니다.
import logic

# -------------------------------------------------------------------------
# 1. 페이지 기본 설정 & 세션 상태 초기화
# -------------------------------------------------------------------------
st.set_page_config(page_title="AI 한식 재료 추천", layout="wide")
st.title("🍳 AI 식재료 대체 추천 대시보드")

if 'voted_logs' not in st.session_state:
    st.session_state['voted_logs'] = set()

# 절감 점수 포맷팅 함수 (공통 사용)
def format_saving(score, is_multi=False):
    prefix = "총 " if is_multi else ""
    if score > 0: return f"🟢 {prefix}+{score}단계 (절감)"
    elif score < 0: return f"🔴 {prefix}{score}단계 (비쌈)"
    else: return "⚪ 동일 수준"

# -------------------------------------------------------------------------
# 2. 사이드바 UI (가중치 설정 및 설명)
# -------------------------------------------------------------------------
with st.sidebar:
    st.header("⚖️ 가중치 설정")
    st.info("💡 커스텀 모드(Ver.2)에서는 '맛'과 '문맥' 점수만 반영됩니다.")
    w_w2v = st.slider("맛·성질 (Word2Vec)", 0.0, 5.0, 5.0, 0.5)
    w_d2v = st.slider("문맥 (Doc2Vec)", 0.0, 5.0, 1.0, 0.5)
    w_method = st.slider("조리법 통계 (Ver.1 전용)", 0.0, 5.0, 1.0, 0.5)
    w_cat = st.slider("카테고리 통계 (Ver.1 전용)", 0.0, 5.0, 1.0, 0.5)
    
    st.divider()
    st.caption("**[Ver.1 DB 모드 수식]**\n(맛+문맥+조리+분류) / 합계")
    st.caption("**[Ver.2 커스텀 모드 수식]**\n(맛+문맥) / 합계 (통계 점수 제외)")

    # [NEW] 사이드바 하단 로직 설명란 추가
    st.divider()
    with st.expander("❓ 어떤 과정을 거쳐 추천되나요?", expanded=False):
        st.markdown("""
        ### 🧠 AI 추천 로직 3단계
        
        **1. 재료의 '의미' 파악 (Word2Vec)**
        * AI가 수많은 레시피를 학습하여 재료 간의 관계를 이해합니다.
        * 예: '돼지고기'는 '소고기', '스팸'과 맛이나 성질이 비슷하다고 판단합니다.
        
        **2. 요리의 '맥락' 이해 (Doc2Vec)**
        * 단순히 비슷한 재료가 아니라, 현재 요리(또는 입력한 재료 리스트)에 어울리는지 판단합니다.
        * 예: '미역국' 맥락에서는 '소고기' 대신 '스팸'보다 '조개'가 더 어울린다고 판단합니다.
        
        **3. 통계적 적합성 (Ver.1 전용)**
        * 실제 레시피 데이터를 분석하여 해당 조리법(예: 끓이기)이나 카테고리(예: 국/탕)에 자주 쓰이는 재료인지 확인합니다.

        ---
        **💰 예상 원가 변동**
        * 재료별 상대적 가격 등급(1~5단계)을 기반으로 계산됩니다.
        * 예: 돼지고기(4등급) → 두부(2등급) = **+2단계 절감**
        """)

# -------------------------------------------------------------------------
# 3. 메인 UI (탭 구성)
# -------------------------------------------------------------------------
col_main, _ = st.columns([0.9, 0.1])
with col_main:
    # 탭 생성 (Ver.1 / Ver.2)
    tab_db, tab_custom = st.tabs(["📚 Ver.1 기존 레시피 DB 검색", "✨ Ver.2 나만의 재료 입력 (커스텀)"])

    # =========================================
    # [Tab 1] Ver.1 기존 레시피 DB 검색 모드
    # =========================================
    with tab_db:
        st.markdown("""
        <div style="background-color: #f0f8ff; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <h4 style="margin:0; color:#0066cc;">[Ver.1] 레시피 데이터베이스에서 검색</h4>
            <p style="margin:5px 0 0 0; font-size:14px;">학습된 12만여 개의 레시피 중 하나를 선택하여 분석합니다. 모든 통계 점수가 활용됩니다.</p>
        </div>
        """, unsafe_allow_html=True)

        dish_name = st.text_input("🍽️ 요리명 검색", placeholder="예: 김치찌개", key="tab1_dish")

        if dish_name:
            cands = logic.df[logic.df['요리명'] == dish_name]
            if cands.empty:
                cands = logic.df[logic.df['요리명'].str.contains(dish_name, na=False)]
            cands = cands.head(10).reset_index(drop=True)

            if cands.empty:
                st.error("❌ 검색 결과가 없습니다.")
            else:
                options = {}
                for _, r in cands.iterrows():
                    ing_sum = ', '.join(r['재료토큰'])
                    preview_text = ing_sum[:100] + "..." if len(ing_sum) > 100 else ing_sum
                    label = f"[{r['요리방법별명']}] {r['요리명']} (ID:{r['레시피일련번호']}) - {preview_text}"
                    options[label] = r['레시피일련번호']
                
                selected_label = st.selectbox("📜 레시피를 선택하세요", list(options.keys()), key="tab1_recipe")
                recipe_id = options[selected_label]
                
                c1, c2 = st.columns(2)
                with c1: target_str = st.text_input("🎯 바꿀 재료", placeholder="돼지고기, 양파", key="tab1_target")
                with c2: stop_str = st.text_input("🚫 제거할 문구", placeholder="약간, 시판용", key="tab1_stop")
                
                if target_str:
                    targets = [t.strip() for t in target_str.split(',') if t.strip()]
                    stops = [s.strip() for s in stop_str.split(',') if s.strip()]
                    
                    if not targets: st.warning("타겟 재료를 입력해주세요.")
                    else:
                        st.divider()
                        final_recommendations = []
                        has_result = False

                        # DB 모드 계산 로직
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
                                st.dataframe(display_df[['추천재료', '적합도', '예상 원가변동']].style.format("{:.1%}", subset=['적합도']).background_gradient(cmap='Greens', subset=['적합도']), use_container_width=True, hide_index=True)
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

                        # 공통 결과 처리 (로그 저장 및 버튼)
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
    # [Tab 2] Ver.2 커스텀 재료 입력 모드
    # =========================================
    with tab_custom:
        st.markdown("""
        <div style="background-color: #fff5f0; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <h4 style="margin:0; color:#cc5500;">[Ver.2] 나만의 재료 리스트 입력</h4>
            <p style="margin:5px 0 0 0; font-size:14px;">냉장고 속 재료들을 직접 입력하세요. 문맥을 실시간으로 분석하여 추천합니다. (통계 점수 제외)</p>
        </div>
        """, unsafe_allow_html=True)
        
        custom_dish_name = st.text_input("🏷️ 요리명 (참고용)", placeholder="예: 내맘대로 볶음밥", key="tab2_dish")
        context_str = st.text_area("📝 전체 재료 리스트 (쉼표로 구분)", placeholder="예: 밥, 계란, 대파, 간장, 참기름", key="tab2_context", height=100)

        if context_str:
            context_ings_list = [ing.strip() for ing in context_str.split(',') if ing.strip()]
            
            if not context_ings_list:
                 st.warning("재료를 한 개 이상 입력해주세요.")
            else:
                st.caption(f"인식된 재료 ({len(context_ings_list)}개): {', '.join(context_ings_list)}")
                
                c1_c, c2_c = st.columns(2)
                with c1_c: target_str_c = st.text_input("🎯 바꿀 재료 (위 리스트 중)", placeholder="예: 계란", key="tab2_target")
                with c2_c: stop_str_c = st.text_input("🚫 제거할 문구", placeholder="예: 약간", key="tab2_stop")

                if target_str_c:
                    targets_c = [t.strip() for t in target_str_c.split(',') if t.strip()]
                    stops_c = [s.strip() for s in stop_str_c.split(',') if s.strip()]
                    
                    invalid_targets = [t for t in targets_c if t not in context_ings_list]
                    if invalid_targets:
                        st.error(f"다음 재료는 전체 리스트에 없습니다: {', '.join(invalid_targets)}")
                    elif not targets_c:
                        st.warning("바꿀 재료를 입력해주세요.")
                    else:
                        st.divider()
                        final_recommendations_c = []
                        has_result_c = False

                        # 커스텀 모드 계산 로직 호출
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
                                st.dataframe(display_df_c[['추천재료', '적합도', '예상 원가변동']].style.format("{:.1%}", subset=['적합도']).background_gradient(cmap='Greens', subset=['적합도']), use_container_width=True, hide_index=True)
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

                        # 공통 결과 처리 (로그 저장 및 버튼) - 커스텀 모드용
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

        else:
            st.info("👆 전체 재료 리스트를 먼저 입력해주세요.")

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
