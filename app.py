# app.py
import streamlit as st
import pandas as pd
# 우리가 만든 logic.py 파일을 임포트합니다.
import logic 

# -------------------------------------------------------------------------
# 1. 페이지 기본 설정
# -------------------------------------------------------------------------
st.set_page_config(page_title="AI 한식 재료 추천", layout="wide")
st.title("🍳 AI 식재료 대체 추천 대시보드")

# -------------------------------------------------------------------------
# 2. 사이드바 UI (가중치 설정)
# -------------------------------------------------------------------------
with st.sidebar:
    st.header("⚖️ 가중치 설정")
    w_w2v = st.slider("맛·성질 (Word2Vec)", 0.0, 5.0, 1.0, 0.5)
    w_d2v = st.slider("문맥 (Doc2Vec)", 0.0, 5.0, 1.0, 0.5)
    w_method = st.slider("조리법 통계", 0.0, 5.0, 1.0, 0.5)
    w_cat = st.slider("카테고리 통계", 0.0, 5.0, 1.0, 0.5)
    st.divider()
    st.info(f"**현재 수식:**\n({w_w2v}×맛 + {w_d2v}×문맥 + {w_method}×조리 + {w_cat}×분류) / 합계")

# -------------------------------------------------------------------------
# 3. 메인 UI (검색 및 결과 표시)
# -------------------------------------------------------------------------
col_main, _ = st.columns([0.9, 0.1])
with col_main:
    st.markdown("""
    <div style="background-color: #f0f8ff; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="margin:0; color:#0066cc;">🍲 AI 한식 재료 대체 추천</h3>
        <p style="margin:5px 0 0 0;">요리의 '맥락'을 이해하는 똑똑한 추천 시스템</p>
    </div>
    """, unsafe_allow_html=True)

    # 3.1 요리 검색 및 선택
    dish_name = st.text_input("🍽️ 요리명 검색", placeholder="예: 김치찌개")

    if dish_name:
        # logic.py에 있는 데이터프레임 사용
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
            
            selected_label = st.selectbox("📜 레시피를 선택하세요", list(options.keys()))
            recipe_id = options[selected_label]
            
            # 3.2 타겟 및 불용어 입력
            c1, c2 = st.columns(2)
            with c1:
                target_str = st.text_input("🎯 바꿀 재료 (쉼표 구분)", placeholder="돼지고기, 양파 (입력 후 엔터)")
            with c2:
                stop_str = st.text_input("🚫 제거할 문구 (쉼표 구분)", placeholder="약간, (, 시판용")
            
            if target_str:
                targets = [t.strip() for t in target_str.split(',') if t.strip()]
                stops = [s.strip() for s in stop_str.split(',') if s.strip()]
                
                if not targets:
                    st.warning("타겟 재료를 올바르게 입력해주세요.")
                else:
                    # 3.3 결과 계산 및 표시
                    st.divider()
                    
                    # -------------------------------------------------
                    # [핵심 로직] 추천 결과 계산 -> 로그 저장 -> 만족도 버튼 표시
                    # -------------------------------------------------
                    
                    # A. 결과 계산 및 추천 리스트 수집
                    final_recommendations = [] # DB에 저장할 최종 추천 결과 리스트
                    has_result = False

                    # A-1. 단일 재료 대체 계산
                    if len(targets) == 1:
                        st.subheader("🔹 단일 재료 대체 추천")
                        t = targets[0]
                        # logic.py의 함수 호출
                        res = logic.substitute_single(recipe_id, t, stops, w_w2v, w_d2v, w_method, w_cat, topn=5)
                        st.markdown(f"**{t}** 대체 결과")
                        if not res.empty:
                            has_result = True
                            # 상위 3개 결과를 추천 리스트에 추가
                            final_recommendations = res['대체재료'].head(3).tolist()
                            
                            # 결과 데이터프레임 표시
                            display_df = res[['대체재료', '최종점수']].copy()
                            display_df.columns = ['추천재료', '적합도']
                            st.dataframe(display_df.style.format("{:.1%}", subset=['적합도']).background_gradient(cmap='Greens', subset=['적합도']), use_container_width=True, hide_index=True)
                        else:
                            st.warning("결과 없음")
                            
                    # A-2. 다중 재료 대체 계산 (타겟이 2개 이상일 때만)
                    elif len(targets) > 1:
                        st.subheader("🧩 최적의 재료 조합 (다중 대체)")
                        # logic.py의 함수 호출
                        multi_res = logic.substitute_multi(recipe_id, targets, stops, w_w2v, w_d2v, w_method, w_cat, beam_width=3)
                        
                        if multi_res:
                            has_result = True
                            # 결과물 형태: [(['재료1', '재료2'], 점수), ...]
                            # 이를 문자열 조합 리스트로 변환하여 추천 리스트에 추가
                            final_recommendations = [", ".join(subs) for subs, score in multi_res]

                            # 결과 데이터프레임 표시
                            m_df = pd.DataFrame([(f"{', '.join(subs)}", score) for subs, score in multi_res], columns=['추천 조합', '종합 점수'])
                            st.dataframe(m_df.style.format("{:.1%}", subset=['종합 점수']).background_gradient(cmap='Blues', subset=['종합 점수']), use_container_width=True, hide_index=True)
                        else:
                            st.info("가능한 재료 조합을 찾을 수 없습니다.")

                    # B. 로그 저장 및 ID 기억 (결과가 있을 때만)
                    if has_result:
                        # 현재 상태 정의 (중복 저장 방지용)
                        # 가중치나 추천 결과가 바뀌면 새로운 상태로 인식
                        current_state = f"{dish_name}_{target_str}_{stop_str}_{w_w2v}_{w_d2v}_{w_method}_{w_cat}_{final_recommendations}"
                        
                        if 'last_log_state' not in st.session_state: st.session_state['last_log_state'] = ""
                            
                        # 상태가 변했을 때만 DB에 저장
                        if st.session_state['last_log_state'] != current_state:
                            # logic.py의 저장 함수 호출하고 로그 ID 받아오기
                            log_id = logic.save_log_to_db(dish_name, target_str, stops, w_w2v, w_d2v, w_method, w_cat, rec_list=final_recommendations)
                            
                            # 세션에 현재 로그 ID 저장 (만족도 버튼용)
                            st.session_state['current_log_id'] = log_id
                            st.session_state['last_log_state'] = current_state
                        
                        # C. 만족도 평가 버튼 UI (전략 1: 전체 결과에 대한 단일 평가)
                        if 'current_log_id' in st.session_state and st.session_state['current_log_id']:
                            st.write("") # 여백
                            st.markdown("##### 🤔 추천 결과가 만족스러우신가요?")
                            st.caption("이 피드백은 더 똑똑한 AI를 만드는 데 사용됩니다.")
                            
                            b1, b2, _ = st.columns([0.2, 0.2, 0.6])
                            # 버튼 클릭 시 logic.py의 업데이트 함수 호출
                            with b1:
                                if st.button("👍 만족해요", key="btn_satisfy", use_container_width=True):
                                    if logic.update_feedback_in_db(st.session_state['current_log_id'], "satisfy"):
                                        st.toast("감사합니다! 만족(👍)으로 기록되었습니다.")
                            with b2:
                                if st.button("👎 아쉬워요", key="btn_dissatisfy", use_container_width=True):
                                    if logic.update_feedback_in_db(st.session_state['current_log_id'], "dissatisfy"):
                                        st.toast("의견 감사합니다. 불만족(👎)으로 기록되었습니다.")
                    
                    if stops:
                        st.divider()
                        st.caption(f"✂️ **적용된 제거 문구:** {', '.join(stops)}")

            else:
                st.info("👆 위 칸에 바꿀 재료를 입력하고 엔터를 누르면 분석 결과가 나타납니다.")

# -------------------------------------------------------------------------
# 4. 하단 피드백 영역 (일반 의견)
# -------------------------------------------------------------------------
st.divider()
st.subheader("📢 서비스 의견 보내기")
with st.form("feedback_form"):
    text = st.text_area("개선할 점이나 버그가 있다면 알려주세요!")
    submitted = st.form_submit_button("의견 보내기")
    
    if submitted:
        if text:
            # logic.py의 함수 호출
            if logic.save_feedback_to_db(text):
                st.success("소중한 의견 감사합니다! 개발자가 확인 후 반영하겠습니다.")
                st.balloons()
        else:
            st.warning("내용을 입력해주세요.")
