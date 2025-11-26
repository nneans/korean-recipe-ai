# logic.py
import streamlit as st
import pandas as pd
import numpy as np
from gensim.models import Word2Vec, Doc2Vec
from ast import literal_eval
import pickle
from datetime import datetime
from supabase import create_client

# ==========================================
# 1. Supabase DB 연동 및 데이터 저장
# ==========================================
@st.cache_resource
def init_supabase():
    """Supabase 클라이언트 연결 초기화"""
    try:
        # secrets.toml에 [supabase] 섹션이 있어야 합니다.
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except Exception as e:
        # 연결 실패 시 UI에서 처리할 수 있도록 에러를 던집니다.
        raise ConnectionError(f"Supabase 연결 실패: {e}")

def save_feedback_to_db(feedback_text):
    """일반 텍스트 피드백 저장"""
    try:
        supabase = init_supabase()
        data = {"content": feedback_text}
        supabase.table("feedback").insert(data).execute()
        return True
    except Exception as e:
        print(f"피드백 저장 에러: {e}")
        return False

def save_log_to_db(dish, target, stops, w1, w2, w3, w4, rec_list=None):
    """
    사용자 로그 및 추천 결과 저장 (전략 1 구현)
    - rec_list: 추천된 결과 문자열 리스트 (최대 3개)
    - 반환값: 저장된 로그의 ID (만족도 평가 연결용)
    """
    try:
        supabase = init_supabase()
        
        # 추천 결과 상위 3개 추출 (없으면 None)
        r1 = rec_list[0] if rec_list and len(rec_list) > 0 else None
        r2 = rec_list[1] if rec_list and len(rec_list) > 1 else None
        r3 = rec_list[2] if rec_list and len(rec_list) > 2 else None

        data = {
            "dish": dish,
            "target": target,
            "stops": ", ".join(stops) if stops else "없음",
            "w_w2v": w1,
            "w_d2v": w2,
            "w_method": w3,
            "w_cat": w4,
            "rec_1": r1, # 1순위 결과 (단일 재료 또는 조합 문자열)
            "rec_2": r2,
            "rec_3": r3
        }
        # insert 후 생성된 데이터의 ID를 반환받습니다.
        response = supabase.table("usage_log").insert(data).execute()
        if response.data and len(response.data) > 0:
             return response.data[0]['id'] # 로그 ID 반환
        return None
    except Exception as e:
        print(f"로그 저장 에러: {e}")
        return None

def update_feedback_in_db(log_id, status):
    """특정 로그 ID에 대한 만족도(satisfaction) 업데이트"""
    try:
        supabase = init_supabase()
        if log_id:
            supabase.table("usage_log").update({"satisfaction": status}).eq("id", log_id).execute()
            return True
        return False
    except Exception as e:
        print(f"만족도 업데이트 에러: {e}")
        return False

# ==========================================
# 2. 데이터 및 모델 로드
# ==========================================
@st.cache_resource
def load_resources():
    """모델과 데이터를 메모리에 로드 (캐싱 사용)"""
    w2v = Word2Vec.load("w2v.model")
    d2v = Doc2Vec.load("d2v.model")
    df = pd.read_csv("recipe_data.csv")
    df['재료토큰'] = df['재료토큰'].apply(literal_eval)
    
    with open("stats.pkl", "rb") as f:
        stats = pickle.load(f)
        
    return w2v, d2v, df, stats

# 전역 변수로 사용하기 위해 리소스 로드
# (실제 환경에서는 try-except로 감싸는게 좋지만 간결함을 위해 생략)
w2v_model, d2v_model, df, stats = load_resources()

# 통계 데이터 풀기
method_map = stats["method_map"]
recipes_by_ingredient = stats["recipes_by_ingredient"]
ing_method_counts = stats["ing_method_counts"]
ing_cat_counts = stats["ing_cat_counts"]
total_method_counts = stats["total_method_counts"]
total_cat_counts = stats["total_cat_counts"]
TOTAL_RECIPES = stats["TOTAL_RECIPES"]

# ==========================================
# 3. 핵심 계산 로직 (유사도, 통계 점수 등)
# ==========================================
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

# ==========================================
# 4. 대체 추천 알고리즘 (단일/다중)
# ==========================================
def substitute_single(recipe_id, target_ing, stopwords, w_w2v, w_d2v, w_method, w_cat, topn=10):
    # ... (기존 단일 대체 로직 코드와 동일) ...
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
        
        temp_results.append({
            "대체재료": clean_cand, "raw_W2V": s_w2v, "raw_D2V": s_d2v, "raw_Method": s_method, "raw_Category": s_cat
        })
        
    if not temp_results: return pd.DataFrame()
    
    df_res = pd.DataFrame(temp_results)
    cols = ["raw_W2V", "raw_D2V", "raw_Method", "raw_Category"]
    norm_cols = ["W2V", "D2V", "Method", "Category"]
    for raw_col, norm_col in zip(cols, norm_cols):
        min_val = df_res[raw_col].min()
        max_val = df_res[raw_col].max()
        if max_val - min_val == 0: df_res[norm_col] = 0.5
        else: df_res[norm_col] = (df_res[raw_col] - min_val) / (max_val - min_val)
        
    df_res["최종점수"] = (
        (df_res["W2V"]*w_w2v) + (df_res["D2V"]*w_d2v) + (df_res["Method"]*w_method) + (df_res["Category"]*w_cat)
    ) / total_weight
    
    return df_res.sort_values("최종점수", ascending=False).head(topn).reset_index(drop=True)

def substitute_multi(recipe_id, targets, stopwords, w_w2v, w_d2v, w_method, w_cat, beam_width=3, result_topn=3):
    # ... (기존 다중 대체 로직 코드와 동일) ...
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

                temp_candidates.append({
                    "cand": clean_cand, "raw_w2v": s_w2v, "raw_d2v": s_d2v, "raw_method": s_method, "raw_cat": s_cat
                })
            
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