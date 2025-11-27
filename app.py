# app.py
import streamlit as st
import pandas as pd
import logic
import os
from datetime import datetime, timedelta, timezone
# [NEW] 워드클라우드 및 시각화 라이브러리 임포트
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# 1. 페이지 기본 설정 & 세션 상태 초기화 & 다이얼로그 함수 정의
# -------------------------------------------------------------------------
st.set_page_config(page_title="AI 한식 재료 추천", layout="wide")
st.title("🍳 AI 식재료 대체 추천 대시보드")

if 'voted_logs' not in st.session_state:
    st.session_state['voted_logs'] = set()

if "stopword_input_field" not in st.session_state:
    st.session_state["stopword_input_field"] = ""

# [NEW] 게시판 닉네임/내용 초기화용
if "board_nick" not in st.session_state: st.session_state["board_nick"] = ""
if "board_msg" not in st.session_state: st.session_state["board_msg"] = ""

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
    (내용 생략 - 기존과 동일)
    """)

# [NEW] 워드클라우드 팝업창 함수
@st.dialog("☁️ 검색 트렌드 워드클라우드", width="large")
def show_wordcloud_dialog(timeframe_text, text_data):
    st.subheader(f"{timeframe_text} 많이 검색된 타겟 재료")
    
    if not text_data:
        st.info("데이터가 충분하지 않습니다.")
        return

    # 폰트 설정 (프로젝트 폴더에 'font.ttf'가 있어야 한글이 안 깨짐)
    font_path = "font.ttf" if os.path.exists("font.ttf") else None
    
    try:
        wordcloud = WordCloud(
            font_path=font_path,
            width=800, height=400,
            background_color='white',
            colormap='viridis',
            random_state=42
        ).generate(text_data)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        
        if not font_path:
            st.caption("⚠️ 한글 폰트 파일('font.ttf')이 없어 글자가 깨질 수 있습니다.")
            
    except Exception as e:
        st.error(f"워드클라우드 생성 중 오류 발생: {e}")

# -------------------------------------------------------------------------
# 2. 사이드바 UI
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
    
    # 제외 재료 설정 (Ver.2)
    excluded_ingredients = []
    if not is_v1:
        st.divider()
        st.subheader("🚫 제외할 재료 설정")
        all_ing_options = sorted(list(logic.all_ingredients_set))
        excluded_ingredients = st.multiselect("제외할 재료 선택", all_ing_options, placeholder="예: 땅콩, 오이")
    
    st.divider()
    if st.button("🤔 어떤 과정을 거쳐 재료가 추천되나요?", use_container_width=True):
        show_logic_dialog()
    
    # 인사이트 대시보드
    st.divider()
    st.subheader("📊 인사이트 대시보드 (Beta)")
    
    kst = timezone(timedelta(hours=9))
    today_date_string = datetime.now(kst).strftime("%Y년 %m월 %d일")

    stopwords_list = logic.load_global_stopwords()
    stopwords_count = len(stopwords_list)

    tab_today, tab_all = st.tabs(["📅 오늘", "📈 누적"])

    # 데이터 미리 로드
    wc_text_today = logic.get_wordcloud_text('today')
    wc_text_all = logic.get_wordcloud_text('all')
    top_pairs_today = logic.get_top_replacement_pairs('today')
    top_pairs_all = logic.get_top_replacement_pairs('all')

    with tab_today:
        st.caption(f"기준일: {today_date_string} (KST)")
        today_count, _, _ = logic.get_usage_stats(timeframe='today')
        col_m1_t, col_m2_t = st.columns(2)
        col_m1_t.metric("오늘 사용량", f"{today_count}건")
        col_m2_t.metric("누적 불용어", f"{stopwords_count}개")

        if today_count > 0:
            # [NEW] 워드클라우드 팝업 버튼
            if st.button("☁️ 오늘의 워드클라우드 보기", key="btn_wc_today", use_container_width=True):
                show_wordcloud_dialog("오늘", wc_text_today)
                
            st.caption("🔄 오늘 많이 대체된 조합 Top 5")
            if not top_pairs_today.empty: st.bar_chart(top_pairs_today, color="#FF6B6B", height=200)
            else: st.caption("데이터 부족")
        else:
            st.info("아직 오늘의 데이터가 없습니다.")

    with tab_all:
        st.caption("서비스 시작 이후 전체 데이터")
        all_count, _, _ = logic.get_usage_stats(timeframe='all')
        col_m1_a, col_m2_a = st.columns(2)
        col_m1_a.metric("총 사용량", f"{all_count}건")
        col_m2_a.metric("누적 불용어", f"{stopwords_count}개")

        if all_count > 0:
            # [NEW] 워드클라우드 팝업 버튼
            if st.button("☁️ 누적 워드클라우드 보기", key="btn_wc_all", use_container_width=True):
                show_wordcloud_dialog("누적", wc_text_all)

            st.caption("🔄 역대 많이 대체된 조합 Top 5")
            if not top_pairs_all.empty: st.bar_chart(top_pairs_all, color="#FF6B6B", height=200)
            else: st.caption("데이터 부족")
        else:
            st.info("누적 데이터가 없습니다.")

    # 불용어 목록 보기 (단순 리스트)
    with st.expander("📋 신고된 불용어 목록 확인"):
        if stopwords_list:
            st.dataframe(pd.DataFrame(stopwords_list, columns=["불용어"]), use_container_width=True, hide_index=True)
        else:
            st.info("아직 신고된 불용어가 없습니다.")
            
    # [NEW] 익명 게시판 (사이드바 하단)
    st.divider()
    with st.expander("💬 익명 게시판 (Beta)", expanded=True):
        # 글쓰기 폼
        with st.form("board_form"):
            nick = st.text_input("닉네임", placeholder="익명", key="board_nick_input")
            msg = st.text_area("내용", placeholder="자유롭게 의견을 남겨주세요", height=80, key="board_msg_input")
            if st.form_submit_button("등록"):
                if nick and msg:
                    if logic.save_board_message(nick, msg):
                        st.toast("게시글이 등록되었습니다!", icon="✅")
                        st.rerun()
                else:
                    st.warning("닉네임과 내용을 모두 입력해주세요.")
        
        # 글 목록 표시
        st.markdown("---")
        messages = logic.get_board_messages()
        if messages:
            for m in messages:
                st.markdown(f"**{m['nickname']}** <span style='color:grey; font-size:0.8em;'>({m['display_time']})</span>", unsafe_allow_html=True)
                st.text(m['content'])
                st.divider()
        else:
            st.caption("아직 게시글이 없습니다.")


# -------------------------------------------------------------------------
# 3. 메인 UI (선택된 모드에 따라 내용 표시)
# -------------------------------------------------------------------------
col_main, _ = st.columns([0.9, 0.1])
with col_main:
    # (메인 UI 코드는 기존과 동일합니다. 위에서 사용했던 코드를 그대로 유지하세요.)
    # ... (Ver.1 DB 모드 및 Ver.2 커스텀 모드 코드) ...
    # (지면 관계상 생략하지만, 이전 답변의 메인 UI 코드를 그대로 붙여넣으시면 됩니다.)
    
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
                            res_c = logic.substitute_single_custom(t_c, context_ings_list, stops_c, w_w2v, w_d2v, excluded_ings=excluded_ingredients, topn=5)
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
                            multi_res_c = logic.substitute_multi_custom(targets_c, context_ings_list, stops_c, w_w2v, w_d2v, excluded_ings=excluded_ingredients)
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
# 4. 하단 피드백 및 불용어 신고 영역 (기존 동일)
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

def handle_stopword_submission():
    current_input = st.session_state.get("stopword_input_field", "")
    if current_input:
        is_success, msg = logic.save_stopwords_to_db(current_input)
        if is_success:
            st.toast(msg, icon="✅")
            st.session_state["stopword_input_field"] = ""
        else:
            st.toast(msg, icon="❌")
    else:
        st.toast("단어를 입력해주세요.", icon="⚠️")

with col_stopword:
    st.subheader("🚫 불용어(이상한 단어) 신고하기")
    st.caption(
        "추천 결과에 이상한 단어가 있나요? 신고해주시면 다음부터 제외됩니다.",
        help="현재 학습 데이터에 포함된 불용어가 너무 많아 일일이 수작업으로 처리하기 어렵습니다. 😥 여러분의 신고가 모이면 데이터의 품질이 높아지고 추천 결과도 더 정확해집니다. 소중한 기여 부탁드립니다! 🙏"
    )
    st.info("💡 Tip: '간장or진간장' 같은 경우 'or'를 신고하면 '간장진간장'으로 합쳐져 추천에서 제외됩니다.")
    
    with st.form("stopword_form"):
        st.text_input("신고할 단어 입력 (쉼표로 구분)", placeholder="예: 면포, 황석어젓, 텃밭", key="stopword_input_field")
        st.form_submit_button("신고하기", use_container_width=True, on_click=handle_stopword_submission)
