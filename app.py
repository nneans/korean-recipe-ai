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

@st.dialog("🧠 AI 추천 알고리즘 작동 원리 상세", width="large")
def show_logic_dialog():
    if os.path.exists("flowchart.png"):
        st.image("flowchart.png", use_container_width=True)
    else:
        st.warning("플로우차트 이미지(flowchart.png)를 찾을 수 없습니다.")

    st.markdown("---")

    st.markdown("""
    ### AI 추천 로직 상세 해부

    이 시스템은 12만여 개의 한식 레시피 데이터를 학습한 AI가 재료의 의미와 문맥을 분석합니다. 단순히 이름이 비슷한 재료가 아닌, '지금 이 요리에 가장 잘 어울리는' 최적의 대안을 찾아내는 과정입니다.

    ---

    #### 💡 AI는 재료를 어떻게 이해할까요? (3차원 벡터 공간 예시)
    AI는 모든 식재료를 거대한 3차원 공간(실제로는 수백 차원) 속의 '좌표(벡터)'로 인식합니다.
    * 유사도가 높다는 뜻: 이 공간에서 두 재료의 좌표가 서로 가까운 위치에 모여 있거나, 원점에서 뻗어나가는 화살표의 방향이 비슷하다는 의미입니다.

    

    위 그림처럼 '돼지고기'와 '소고기'는 '육류'라는 비슷한 성질을 가져 공간상에서 가까운 위치에 모여 있습니다. 반면, '사과'는 성질이 다르기 때문에 멀리 떨어져 있습니다. AI는 이 '거리와 방향의 멂과 가까움'을 계산하여 추천에 활용합니다.

    ---

    #### 1단계. 의미 파악 (Word2Vec): "친구를 보면 너를 알 수 있어"
    * 핵심 원리: AI는 재료의 맛이나 식감을 직접 알지 못합니다. 대신 '함께 자주 쓰이는 주변 재료(문맥)'가 비슷할수록 유사한 역할을 하는 재료로 학습합니다.
    * 예시 (타겟 재료: 돼지고기)
        * 돼지고기의 친구들: [간장, 마늘, 양파, 고추장, 김치, 볶기]
        * 🥩 소고기 (유사도 0.85): [간장, 마늘, 양파, 참기름, 불고기] → 겹치는 친구가 매우 많음 (비슷한 재료!)
        * 🐟 고등어 (유사도 0.45): [간장, 마늘, 무, 생강, 비린내] → 일부 겹치지만, 다른 친구들도 많음 (조금 다른 재료)
        * 🍎 사과 (유사도 0.10): [설탕, 마요네즈, 샐러드, 아침] → 겹치는 친구가 거의 없음 (완전히 다른 재료)

    #### 2단계. 문맥 이해 (Doc2Vec): "같은 조리법 상황에서의 궁합 파악"
    * 핵심 (코드 구현 내용): 단순히 '이 재료가 요리에 어울리나?'를 보는 것이 아니라, '현재의 조리방법(예: 끓이기, 볶기)'과 동일한 상황에서 잘 어울리는지를 판단합니다.
    * 작동 원리 (Ver.1 DB 모드 기준):
        1.  현재 타겟 요리의 '조리방법'(예: 끓이기)을 확인합니다.
        2.  후보 재료가 사용된 수많은 레시피 중, 같은 조리방법('끓이기')이 사용된 레시피들만 골라냅니다.
        3.  골라낸 레시피들의 좌표가 현재 타겟 요리의 좌표와 얼마나 가까운지 비교합니다.
    * 왜 이렇게 하나요? 같은 재료라도 '볶을 때'와 '끓일 때'의 역할이 다르기 때문입니다. 조리법 조건을 걸어 더 정확한 문맥 파악을 합니다.

    #### 3단계. 통계적 적합성 (Ver.1 DB 모드 전용): "데이터 검증 (Lift)"
    * 역할: 실제 데이터에서 해당 재료가 특정 조리법이나 요리 카테고리에 '유독' 많이 쓰이는지 검증합니다. (여기서 카테고리 정보도 함께 활용됩니다.)
    * 핵심 개념 (Lift, 향상도): 평균적인 사용 확률 대비, 특정 조건에서 사용 확률이 얼마나 높아지는지를 봅니다. 기준값은 1입니다.
    * 판단 기준:
        * Lift > 1 (추천): 평균보다 이 조건에서 더 자주 쓰임 (궁합이 좋음)
        * Lift ≈ 1 (보통): 평균적인 수준으로 쓰임
        * Lift < 1 (비추천): 평균보다 이 조건에서 덜 쓰임 (궁합이 안 좋음)
    * 예시 (조리법: 끓이기): 두부(Lift > 1, 끓일 때 필수), 상추(Lift < 1, 끓일 때 안 씀)

    ---

    #### 🚀 추천 알고리즘 심화: 어떻게 최적의 재료를 찾아낼까?

    1. 단일 재료 대체 (Best N 찾기)
    * 위 1~3단계 점수에 가중치를 적용한 최종 종합 점수를 계산하고, 점수가 가장 높은 순서대로 상위 N개의 재료를 추천합니다.

    2. 다중 재료 대체 (최적 조합 찾기 - 빔 서치)
    * 여러 재료를 동시에 바꿀 때는 경우의 수가 폭발적으로 늘어납니다. 이때 효율적인 탐색을 위해 '빔 서치(Beam Search)'를 사용합니다.
    * 작동 원리 (매 단계마다):
        1.  현재까지 구성된 조합에 새로운 재료 후보를 하나씩 추가해봅니다.
        2.  새로운 조합의 점수를 계산합니다. (점수 = 현재까지의 점수 + 새 재료의 AI 점수)
        3.  모든 후보 조합 중 점수가 가장 높은 상위 K개(Beam Width)의 조합만 남기고 나머지는 버립니다.
        4.  이 과정을 목표한 재료 수만큼 반복하여 최종적으로 가장 좋은 조합을 찾아냅니다.
    > 💡 비유: 어두운 숲속에서 보물을 찾을 때, 여러 갈래 길 중 가장 밝은 빛이 비추는 길 3곳(K=3)만 골라서 계속 따라가는 것과 같습니다.

    ---

    #### 🏆 최종 종합 점수 계산 예시 (가중치 적용)
    (시나리오: 김치찌개(끓이기, 국/탕)에서 '돼지고기' 대신 '참치캔' 추천 시)

    * 1. 의미 점수: 0.70 × 가중치 5.0 = 3.50
    * 2. 문맥 점수: 0.95 × 가중치 1.0 = 0.95 (같은 '끓이기' 요리들과 비교)
    * 3. 조리 통계: 0.90 × 가중치 1.0 = 0.90 ('끓이기' 데이터 검증)
    * 4. 분류 통계: 0.85 × 가중치 1.0 = 0.85 ('국/탕' 데이터 검증)

    👉 총점: 3.50 + 0.95 + 0.90 + 0.85 = 6.20 / (총 가중치 8.0) = 최종 적합도 77.5%

    ---

    #### 💰 예상 원가 변동 (별도 계산)
    AI 점수와 별개로 제공되는 참고 정보입니다. 실시간 정확한 시세가 아닌, 사전에 정의된 재료별 상대적 가격 등급(1~5등급)을 기준으로 계산됩니다.
    * 예: 돼지고기(4등급) ➡️ 두부(2등급) 대체 시 4 - 2 = +2 (🟢 총 +2단계 절감 예상)
    """)

# -------------------------------------------------------------------------
# 2. 사이드바 UI (모드 선택 및 가중치 설정 + 통계 대시보드)
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
    if st.button("🤔 어떤 과정을 거쳐 재료가 추천되나요?", use_container_width=True):
        show_logic_dialog()
    
    # [NEW]📊 오늘의 인사이트 (Beta) 섹션 추가
    st.divider()
    st.subheader("📊 오늘의 인사이트 (Beta)")
    
    # logic.py에서 통계 데이터와 불용어 목록 로드
    today_count, top_dishes, top_targets = logic.get_daily_stats()
    stopwords_list = logic.load_global_stopwords()
    stopwords_count = len(stopwords_list)

    # 1. 메트릭 표시 (오늘 사용량, 불용어 수)
    col_m1, col_m2 = st.columns(2)
    col_m1.metric("오늘 사용량", f"{today_count}건", help="오늘 하루 동안 발생한 재료 추천 요청 횟수입니다. (KST 0시 기준 초기화)")
    col_m2.metric("신고된 불용어", f"{stopwords_count}개", help="사용자들이 신고하여 현재 추천에서 제외 중인 단어의 총 개수입니다.")

    # 2. 인기 차트 표시 (데이터가 있을 때만)
    if today_count > 0:
        st.caption("🔥 오늘 가장 많이 찾은 검색어 Top 5")
        tab_dish, tab_target = st.tabs(["요리명", "타겟 재료"])
        with tab_dish:
            if not top_dishes.empty:
                st.bar_chart(top_dishes, color="#FF9F43", height=200)
            else:
                st.caption("데이터가 충분하지 않습니다.")
        with tab_target:
            if not top_targets.empty:
                st.bar_chart(top_targets, color="#2ECC71", height=200)
            else:
                st.caption("데이터가 충분하지 않습니다.")
    else:
        st.info("아직 오늘의 데이터가 없습니다. 첫 번째 사용자가 되어보세요! 😉")

    # 3. 불용어 목록 보기 (익스팬더)
    with st.expander("📋 신고된 불용어 목록 보기"):
        if stopwords_list:
            df_stopwords = pd.DataFrame(stopwords_list, columns=["불용어 단어"])
            st.dataframe(df_stopwords, use_container_width=True, hide_index=True, height=200)
        else:
            st.info("아직 신고된 불용어가 없습니다.")

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
            exact_match = logic.df[logic.df['요리명'] == search_keyword]
            exact_name = exact_match['요리명'].iloc[0] if not exact_match.empty else None
            candidates = logic.df[logic.df['요리명'].str.contains(search_keyword, na=False, case=False)]
            if exact_name:
                candidates = candidates[candidates['요리명'] != exact_name]
            
            candidate_names = candidates['요리명'].unique().tolist()
            candidate_names = sorted(candidate_names)[:30]

            options = []
            if exact_name:
                options.append(exact_name)
            options.extend(candidate_names)
            
            if not options:
                st.warning(f"🔍 '{search_keyword}'가 포함된 요리명을 찾을 수 없습니다.")
            else:
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
                with c2: stop_str = st.text_input("🚫 제거할 문구 (임시)", placeholder="약간, 시판용")
                
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
                            # 임시 불용어 전달
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
                            # 임시 불용어 전달
                            multi_res = logic.substitute_multi(recipe_id, targets, stops, w_w2v, w_d2v, w_method, w_cat)
                            if multi_res:
                                has_result = True
                                final_recommendations = [", ".join(subs) for subs, score, saving in multi_res]
                                m_df = pd.DataFrame([(f"{', '.join(subs)}", score, format_saving(saving, True)) for subs, score, saving in multi_res], columns=['추천 조합', '종합 점수', '예상 원가변동 합계'])
                                st.dataframe(m_df.style.format("{:.1%}", subset=['종합 점수']).background_gradient(cmap='Blues', subset=['종합 점수']), use_container_width=True, hide_index=True)
                            else: st.info("조합을 찾을 수 없습니다.")
                        if has_result:
                            current_state = f"DB_{final_dish_name}_{target_str}_{stop_str}_{w_w2v}_{w_d2v}_{w_method}_{w_cat}_{final_recommendations}"
                            if 'last_log_state' not in st.session_state: st.session_state['last_log_state'] = ""
                            if st.session_state['last_log_state'] != current_state:
                                log_id = logic.save_log_to_db(final_dish_name, target_str, stops, w_w2v, w_d2v, w_method, w_cat, rec_list=final_recommendations, is_custom=False)
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
            exact_match_v2 = logic.df[logic.df['요리명'] == search_keyword_v2]
            exact_name_v2 = exact_match_v2['요리명'].iloc[0] if not exact_match_v2.empty else None
            candidates_v2 = logic.df[logic.df['요리명'].str.contains(search_keyword_v2, na=False, case=False)]
            if exact_name_v2:
                candidates_v2 = candidates_v2[candidates_v2['요리명'] != exact_name_v2]
            
            candidate_names_v2 = candidates_v2['요리명'].unique().tolist()
            candidate_names_v2 = sorted(candidate_names_v2)[:30]

            options_v2 = []
            if exact_name_v2:
                options_v2.append(exact_name_v2)
            options_v2.append("(직접 입력한 이름 사용)")
            options_v2.extend(candidate_names_v2)

            if options_v2:
                index_to_select_v2 = 0 if exact_name_v2 else 0
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
                with c2_c: stop_str_c = st.text_input("🚫 제거할 문구 (임시)", placeholder="예: 약간", key="v2_stop")
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
                            # 임시 불용어 전달
                            res_c = logic.substitute_single_custom(t_c, context_ings_list, stops_c, w_w2v, w_d2v, topn=5)
                            st.markdown(f"**{t_c}** 대체 결과")
                            if not res_c.empty:
                                has_result_c = True
                                final_recommendations_c = res['대체재료'].head(3).tolist()
                                display_df_c = res_c[['대체재료', '최종점수', 'saving_score']].copy()
                                display_df_c['예상 원가변동'] = display_df_c['saving_score'].apply(lambda x: format_saving(x))
                                display_df_c = display_df_c[['대체재료', '최종점수', '예상 원가변동']]
                                display_df_c.columns = ['추천재료', '적합도', '예상 원가변동']
                                st.dataframe(display_df_c.style.format("{:.1%}", subset=['적합도']).background_gradient(cmap='Greens', subset=['적합도']), use_container_width=True, hide_index=True)
                            else: st.warning("결과 없음")
                        elif len(targets_c) > 1:
                            st.subheader("🧩 최적의 재료 조합 (커스텀 다중 대체)")
                            # 임시 불용어 전달
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
# 4. 하단 피드백 및 불용어 신고 영역
# -------------------------------------------------------------------------
st.divider()
col_feedback, col_stopword = st.columns(2)

with col_feedback:
    st.subheader("📢 서비스 의견 보내기")
    with st.form("feedback_form"):
        text = st.text_area("개선할 점이나 버그가 있다면 알려주세요!", height=100)
        submitted = st.form_submit_button("의견 보내기", use_container_width=True)
        if submitted:
            if text:
                if logic.save_feedback_to_db(text): st.success("의견 감사합니다!"); st.balloons()
            else: st.warning("내용을 입력해주세요.")

with col_stopword:
    st.subheader("🚫 불용어(이상한 단어) 신고하기")
    # help 인자를 사용하여 도움말 아이콘과 설명 추가
    st.caption(
        "추천 결과에 이상한 단어가 있나요? 신고해주시면 다음부터 제외됩니다.",
        help="현재 학습 데이터에 포함된 불용어가 너무 많아 일일이 수작업으로 처리하기 어렵습니다. 😥 여러분의 신고가 모이면 데이터의 품질이 높아지고 추천 결과도 더 정확해집니다. 소중한 기여 부탁드립니다! 🙏"
    )
    with st.form("stopword_form"):
        stopword_input = st.text_input("신고할 단어 입력", placeholder="예: 약간, 머그컵으로")
        submitted_stop = st.form_submit_button("신고하기", use_container_width=True)
        if submitted_stop:
            if stopword_input:
                success, msg = logic.save_stopword_to_db(stopword_input)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
            else:
                st.warning("단어를 입력해주세요.")
