import streamlit as st
import pandas as pd
import numpy as np
from gensim.models import Word2Vec, Doc2Vec
from ast import literal_eval
import pickle
import os
from datetime import datetime
from supabase import create_client, Client

# -------------------------------------------------------------------------
# 0. Supabase DB 연동
# -------------------------------------------------------------------------
@st.cache_resource
def init_supabase():
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except Exception as e:
        # 연결 실패 시 사용자에게 에러를 보여주지 않고 조용히 처리 (로그만 남김)
        print(f"DB 연결 실패: {e}")
        return None

def save_feedback_to_db(feedback_text):
    supabase = init_supabase()
    if not supabase: return False
    
    try:
        data = {"content": feedback_text}
        supabase.table("feedback").insert(data).execute()
        return True
    except Exception as e:
        st.error(f"피드백 저장 실패: {e}")
        return False

# [수정됨] 디버깅 알림(Toast) 제거하고 조용히 저장
def log_callback():
    if not st.session_state.get("val_target"):
        return

    supabase = init_supabase()
    if not supabase: return
    
    try:
        dish = st.session_state.get("val_dish", "")
        target = st.session_state.get("val_target", "")
        stop_text = st.session_state.get("val_stops", "")
        w1 = st.session_state.get("val_w1", 1.0)
        w2 = st.session_state.get("val_w2", 1.0)
        w3 = st.session_state.get("val_w3", 1.0)
        w4 = st.session_state.get("val_w4", 1.0)

        data = {
            "dish": dish,
            "target": target if target else "미입력",
            "stops": stop_text if stop_text else "없음",
            "w_w2v": w1,
            "w_d2v": w2,
            "w_method": w3,
            "w_cat": w4
        }
        supabase.table("usage_log").insert(data).execute()
        
    except Exception as e:
        print(f"로그 저장 실패: {e}")

# -------------------------------------------------------------------------
# 1. 페이지 설정
# -------------------------------------------------------------------------
st.set_page_config(page_title="AI 한식 재료 추천", layout="wide")
st.title("🍳 AI 식재료 대체 추천 대시보드")

# -------------------------------------------------------------------------
# 2. 데이터 및 모델 로드
# -------------------------------------------------------------------------
@st.cache_resource
def load_resources():
    w2v = Word2Vec.load("w2v.model")
    d2v = Doc2Vec.load("d2v.model")
    df = pd.read_csv("recipe_data.csv")
    df['재료토큰'] = df['재료토큰'].apply(literal_eval)
    
    with open("stats.pkl", "rb") as f:
        stats = pickle.load(f)
    return w2v, d2v, df, stats

with st.spinner("AI 모델과 데이터를 불러오는 중..."):
    w2v_model, d2v_model, df, stats = load_resources()

method_map = stats["method_map"]
recipes_by_ingredient = stats["recipes_by_ingredient"]
ing_method_counts = stats["ing_method_counts"]
ing_cat_counts = stats["ing_cat_counts"]
total_method_counts = stats["total_method_counts"]
total_cat_counts = stats["total_cat_counts"]
TOTAL_RECIPES = stats["TOTAL_RECIPES"]

# -------------------------------------------------------------------------
# 3. 핵심 로직 함수
# -------------------------------------------------------------------------
def cos_sim(vec_a, vec_b):
    norm = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-9)
    return max(0.0, float(np.dot(vec_a, vec_b) / norm))

def get_stat_score(ingredient, target_key, ing_count_dict, total_count_dict, total_n, min_count=5):
    cnts = ing_count_dict.get(ingredient)
    if not cnts: return 0.0
    ing_target_count = cnts[target_key]
    ing_total_count = sum(cnts.values())
    if ing_total_count < min_count: return 0.0
    
    prob_ing_context = ing_target_count / ing_total_count
    baseline_prob = total_count_dict[target_key] / total_n
    if baseline_prob == 0: return 0.0
    return prob_ing_context / baseline_prob

def substitute_single(recipe_id, target_ing, stopwords, w_w2v, w_d2v, w_method, w_cat, topn=10):
    row = df[df['레시피일련번호'] == recipe_id].iloc[0]
    current_method = row['요리방법별명']
    current_cat = row['요리종류별명_세분화']
    context_ings = row['재료토큰']
    tag = f"recipe_{recipe_id}"
    
    if target_ing not in w2v_model.wv: return pd.DataFrame()
    total_weight = w_w2v + w_d2v + w_method + w_cat
    if total_weight == 0: total_weight = 1.0
    
    vec_recipe = None
    if w_d2v > 0 and tag in d2v_model.dv:
        vec_recipe = d2v_model.dv[tag]
        
    candidates_raw = w2v_model.wv.most_similar(target_ing, topn=50)
    temp_results = []
    seen_candidates = set()
    
    for cand, score_w2v in candidates_raw:
        clean_cand = cand
        if stopwords:
            for stop in stopwords:
                clean_cand = clean_cand.replace(stop, "")
        clean_cand = clean_cand.strip()
        
        if not clean_cand: continue
        if clean_cand in context_ings: continue
        if clean_cand == target_ing: continue
        if clean_cand not in w2v_model.wv: continue
        if clean_cand in seen_candidates: continue
        seen_candidates.add(clean_cand)
        
        real_score_w2v = w2v_model.wv.similarity(target_ing, clean_cand)
        s_w2v = max(0.0, real_score_w2v)
        if s_w2v < 0.35: continue
        
        s_d2v = 0.0
        if w_d2v > 0 and vec_recipe is not None:
            rid_list = recipes_by_ingredient.get(clean_cand, [])
            same_method_ids = [r for r in rid_list if method_map.get(r) == current_method]
            if len(same_method_ids) > 20:
                np.random.seed(42)
                same_method_ids = np.random.choice(same_method_ids, 20, replace=False)
            if same_method_ids is not None and len(same_method_ids) > 0:
                sims = []
                for r in same_method_ids:
                    rt = f"recipe_{r}"
                    if rt in d2v_model.dv:
                        sims.append(cos_sim(vec_recipe, d2v_model.dv[rt]))
                if sims: s_d2v = np.mean(sims)
        
        s_method = 0.0 if w_method <= 0 else get_stat_score(clean_cand, current_method, ing_method_counts, total_method_counts, TOTAL_RECIPES)
        s_cat = 0.0 if w_cat <= 0 else get_stat_score(clean_cand, current_cat, ing_cat_counts, total_cat_counts, TOTAL_RECIPES)
        
        temp_results.append({"대체재료": clean_cand, "raw_W2V": s_w2v, "raw_D2V": s_d2v, "raw_Method": s_method, "raw_Category": s_cat})
        
    if not temp_results: return pd.DataFrame()
    
    df_res = pd.DataFrame(temp_results)
    cols = ["raw_W2V", "raw_D2V", "raw_Method", "raw_Category"]
    norm_cols = ["W2V", "D2V", "Method", "Category"]
    for raw_col, norm_col in zip(cols, norm_cols):
        min_val = df_res[raw_col].min()
        max_val = df_res[raw_col].max()
        if max_val - min_val == 0: df_res[norm_col] = 0.5
        else: df_res[norm_col] = (df_res[raw_col] - min_val) / (max_val - min_val)
        
    df_res["최종점수"] = ((df_res["W2V"]*w_w2v) + (df_res["D2V"]*w_d2v) + (df_res["Method"]*w_method) + (df_res["Category"]*w_cat)) / total_weight
    return df_res.sort_values("최종점수", ascending=False).head(topn).reset_index(drop=True)

def substitute_multi(recipe_id, targets, stopwords, w_w2v, w_d2v, w_method, w_cat, beam_width=3, result_topn=3):
    row = df[df['레시피일련번호'] == recipe_id].iloc[0]
    current_method = row['요리방법별명']
    current_cat = row['요리종류별명_세분화']
    initial_context = row['재료토큰']
    tag = f"recipe_{recipe_id}"
    
    vec_recipe = None
    if w_d2v > 0 and tag in d2v_model.dv:
        vec_recipe = d2v_model.dv[tag]
    
    total_weight = w_w2v + w_d2v + w_method + w_cat
    if total_weight == 0: total_weight = 1.0

    beam = [(0.0, [], initial_context)]
    
    for target_ing in targets:
        next_beam = []
        if target_ing not in w2v_model.wv:
            for score, subs, ctx in beam:
                next_beam.append((score, subs + [target_ing], ctx))
            beam = next_beam
            continue

        for path_score, path_subs, path_ctx in beam:
            current_ctx_ing = [x for x in path_ctx if x != target_ing]
            candidates = w2v_model.wv.most_similar(target_ing, topn=50)
            
            temp_candidates = []
            seen_candidates = set()
            
            for cand, _ in candidates:
                clean_cand = cand
                if stopwords:
                    for stop in stopwords:
                        clean_cand = clean_cand.replace(stop, "")
                clean_cand = clean_cand.strip()
                
                if not clean_cand: continue
                if clean_cand in current_ctx_ing or clean_cand in path_subs: continue
                if clean_cand == target_ing: continue
                if clean_cand not in w2v_model.wv: continue
                if clean_cand in seen_candidates: continue
                seen_candidates.add(clean_cand)
                
                sim_orig = w2v_model.wv.similarity(target_ing, clean_cand)
                sim_orig = max(0.0, sim_orig)
                if sim_orig < 0.3: continue
                
                harmony_scores = [w2v_model.wv.similarity(clean_cand, c) for c in current_ctx_ing if c in w2v_model.wv]
                sim_harmony = np.mean(harmony_scores) if harmony_scores else 0.0
                s_w2v = 0.5 * sim_orig + 0.5 * max(0.0, sim_harmony)
                
                s_d2v = 0.0
                if vec_recipe is not None:
                    rid_list = recipes_by_ingredient.get(clean_cand, [])
                    same_method_ids = [r for r in rid_list if method_map.get(r) == current_method]
                    if len(same_method_ids) > 20:
                        np.random.seed(42)
                        same_method_ids = np.random.choice(same_method_ids, 20, replace=False)
                    if same_method_ids is not None and len(same_method_ids) > 0:
                        sims = []
                        for r in same_method_ids:
                            rt = f"recipe_{r}"
                            if rt in d2v_model.dv:
                                sims.append(cos_sim(vec_recipe, d2v_model.dv[rt]))
                        if sims: s_d2v = np.mean(sims)
                
                s_method = 0.0 if w_method <= 0 else get_stat_score(clean_cand, current_method, ing_method_counts, total_method_counts, TOTAL_RECIPES)
                s_cat = 0.0 if w_cat <= 0 else get_stat_score(clean_cand, current_cat, ing_cat_counts, total_cat_counts, TOTAL_RECIPES)

                temp_candidates.append({"cand": clean_cand, "raw_w2v": s_w2v, "raw_d2v": s_d2v, "raw_method": s_method, "raw_cat": s_cat})
            
            if not temp_candidates: continue

            df_temp = pd.DataFrame(temp_candidates)
            cols = ["raw_w2v", "raw_d2v", "raw_method", "raw_cat"]
            for col in cols:
                min_val = df_temp[col].min()
                max_val = df_temp[col].max()
                if max_val - min_val == 0: df_temp[col + "_norm"] = 0.5
                else: df_temp[col + "_norm"] = (df_temp[col] - min_val) / (max_val - min_val)
            
            for _, r in df_temp.iterrows():
                weighted_sum = ((r["raw_w2v_norm"]*w_w2v) + (r["raw_d2v_norm"]*w_d2v) + (r["raw_method_norm"]*w_method) + (r["raw_cat_norm"]*w_cat)) / total_weight
                new_total_score = path_score + weighted_sum
                new_subs = path_subs + [r["cand"]]
                new_ctx = current_ctx_ing + [r["cand"]]
                next_beam.append((new_total_score, new_subs, new_ctx))
        
        next_beam.sort(key=lambda x: x[0], reverse=True)
        beam = next_beam[:beam_width]
    
    results = []
    for score, subs, _ in beam:
        avg_score = score / len(targets) if targets else 0.0
        results.append((subs, avg_score))
        
    return results[:result_topn]

# -------------------------------------------------------------------------
# 4. UI 구성
# -------------------------------------------------------------------------
with st.sidebar:
    st.header("⚖️ 가중치 설정")
    
    w_w2v = st.slider("맛·성질 (Word2Vec)", 0.0, 5.0, 1.0, 0.5, key="val_w1", on_change=log_callback)
    st.caption("⬆ 높이면: 설탕↔올리고당 처럼 **맛이나 화학적 성질**이 비슷한 재료를 우선합니다.")
    
    w_d2v = st.slider("문맥 (Doc2Vec)", 0.0, 5.0, 1.0, 0.5, key="val_w2", on_change=log_callback)
    st.caption("⬆ 높이면: 현재 요리의 **전체적인 분위기나 재료 조합**에 어울리는 재료를 찾습니다.")
    
    w_method = st.slider("조리법 통계", 0.0, 5.0, 1.0, 0.5, key="val_w3", on_change=log_callback)
    st.caption("⬆ 높이면: '볶음', '찜' 등 **현재 조리 방식**에 자주 쓰이는 재료를 추천합니다.")
    
    w_cat = st.slider("카테고리 통계", 0.0, 5.0, 1.0, 0.5, key="val_w4", on_change=log_callback)
    st.caption("⬆ 높이면: '국/탕', '반찬' 등 **요리 종류**에 적합한 재료를 추천합니다.")
    
    st.markdown("---")
    st.info(f"**현재 수식:**\n({w_w2v}×맛 + {w_d2v}×문맥 + {w_method}×조리 + {w_cat}×분류) / 합계")

col_main, _ = st.columns([0.8, 0.2])
with col_main:
    st.markdown("""
    <div style="background-color: #f0f8ff; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
        <h4 style="margin:0; color:#0066cc;">🍲 요리의 '맥락'을 이해하는 AI 재료 추천</h4>
        <p style="margin:5px 0 0 0;">요리명을 검색하고, 바꾸고 싶은 재료를 입력해보세요.</p>
    </div>
    """, unsafe_allow_html=True)

    # 1. 요리 검색
    dish_name = st.text_input("🍽️ 요리명 검색", placeholder="예: 김치찌개", key="val_dish", on_change=log_callback)

    if dish_name:
        cands = df[df['요리명'] == dish_name]
        if cands.empty:
            cands = df[df['요리명'].str.contains(dish_name, na=False)]
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
            
            # 2. 타겟 & 불용어 입력
            c1, c2 = st.columns(2)
            with c1:
                target_str = st.text_input("🎯 바꿀 재료 (쉼표 구분)", placeholder="돼지고기, 양파 (입력 후 엔터)", key="val_target", on_change=log_callback)
            with c2:
                stop_str = st.text_input("🚫 제거할 문구 (쉼표 구분)", placeholder="약간, (, 시판용", key="val_stops", on_change=log_callback)
                st.caption("💡 데이터에 섞여 있는 불필요한 단어(예: '약간', '시판용')나 특수문자를 입력하면, 해당 문구만 지우고 분석합니다.")
            
            if target_str:
                targets = [t.strip() for t in target_str.split(',') if t.strip()]
                stops = [s.strip() for s in stop_str.split(',') if s.strip()]
                
                if not targets:
                    st.warning("타겟 재료를 올바르게 입력해주세요.")
                else:
                    st.divider()
                    if stops:
                        st.caption(f"✂️ **적용된 제거 문구:** {', '.join(stops)}")
                    
                    st.subheader("🔹 단일 재료 대체 추천")
                    cols = st.columns(len(targets))
                    has_result = False
                    
                    for idx, t in enumerate(targets):
                        with cols[idx]:
                            res = substitute_single(recipe_id, t, stops, w_w2v, w_d2v, w_method, w_cat, topn=5)
                            st.markdown(f"**{t}** 대체 결과")
                            if not res.empty:
                                has_result = True
                                display_df = res[['대체재료', '최종점수']].copy()
                                display_df.columns = ['추천재료', '적합도']
                                st.dataframe(
                                    display_df.style.format("{:.1%}", subset=['적합도'])
                                               .background_gradient(cmap='Greens', subset=['적합도']),
                                    use_container_width=True,
                                    hide_index=True
                                )
                            else:
                                st.warning("결과 없음")
                                
                    if len(targets) > 1 and has_result:
                        st.divider()
                        st.subheader("🧩 최적의 재료 조합 (다중 대체)")
                        multi_res = substitute_multi(recipe_id, targets, stops, w_w2v, w_d2v, w_method, w_cat, beam_width=3)
                        
                        if multi_res:
                            m_df = pd.DataFrame([
                                (f"{', '.join(subs)}", score) for subs, score in multi_res
                            ], columns=['추천 조합', '종합 점수'])
                            st.dataframe(
                                m_df.style.format("{:.1%}", subset=['종합 점수'])
                                    .background_gradient(cmap='Blues', subset=['종합 점수']),
                                use_container_width=True,
                                hide_index=True
                            )
                        else:
                            st.info("가능한 재료 조합을 찾을 수 없습니다.")
            else:
                st.info("👆 위 칸에 바꿀 재료를 입력하고 엔터를 누르면 분석 결과가 나타납니다.")

st.divider()
st.subheader("📢 피드백 보내기")
with st.form("feedback_form"):
    text = st.text_area("개선할 점이나 이상한 추천 결과가 있다면 알려주세요!")
    submitted = st.form_submit_button("의견 보내기")
    
    if submitted:
        if text:
            if save_feedback_to_db(text):
                st.success("소중한 의견 감사합니다! 개발자가 확인 후 반영하겠습니다.")
                st.balloons()
        else:
            st.warning("내용을 입력해주세요.")