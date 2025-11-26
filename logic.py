# logic.py
import streamlit as st
import pandas as pd
import numpy as np
from gensim.models import Word2Vec, Doc2Vec
from ast import literal_eval
import pickle
from datetime import datetime, timedelta, timezone
from supabase import create_client

# ==========================================
# 0. 환경 설정 및 규칙 정의 (기존 동일)
# ==========================================
PRICE_KEYWORD_RULES = [
    (5, ['소고기', '한우', '채끝', '등심', '안심', '갈비살', '전복', '장어']),
    (4, ['돼지', '삼겹', '목살', '앞다리', '뒷다리', '갈비', '오리', '낙지', '오징어', '새우', '명란']),
    (3, ['닭', '치킨', '햄', '소시지', '베이컨', '스팸', '참치', '동원', '어묵', '맛살', '버섯', '치즈']),
    (2, ['두부', '순두부', '콩나물', '숙주', '김치', '무', '감자', '고구마', '당근', '호박']),
    (1, ['양파', '대파', '쪽파', '실파', '마늘', '고추', '물', '소금', '설탕', '간장', '소스', '양념', '육수'])
]
PRICE_RULE_EXCEPTIONS = ['돼지감자', '닭의장풀', '새우젓', '멸치액젓', '다시다']

# ==========================================
# 1. Supabase DB 연동 및 데이터 저장 (기존 동일)
# ==========================================
@st.cache_resource
def init_supabase():
    try:
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        return create_client(url, key)
    except Exception as e:
        raise ConnectionError(f"Supabase 연결 실패: {e}")

def get_kst_now_iso():
    kst_timezone = timezone(timedelta(hours=9))
    now_kst = datetime.now(kst_timezone)
    return now_kst.isoformat()

def save_feedback_to_db(feedback_text):
    try:
        supabase = init_supabase()
        data = {"content": feedback_text, "created_at": get_kst_now_iso()}
        supabase.table("feedback").insert(data).execute()
        return True
    except Exception as e:
        print(f"피드백 저장 에러: {e}")
        return False

def save_log_to_db(dish, target, stops, w1, w2, w3, w4, rec_list=None, is_custom=False):
    """사용자 로그 저장 (한국 시간 적용, is_custom 플래그 추가)"""
    try:
        supabase = init_supabase()
        r1 = rec_list[0] if rec_list and len(rec_list) > 0 else None
        r2 = rec_list[1] if rec_list and len(rec_list) > 1 else None
        r3 = rec_list[2] if rec_list and len(rec_list) > 2 else None

        # 커스텀 모드임을 요리명에 표시
        dish_name_to_save = f"[Custom] {dish}" if is_custom else dish

        data = {
            "dish": dish_name_to_save,
            "target": target,
            "stops": ", ".join(stops) if stops else "없음",
            "w_w2v": w1, "w_d2v": w2, "w_method": w3, "w_cat": w4,
            "rec_1": r1, "rec_2": r2, "rec_3": r3,
            "created_at": get_kst_now_iso()
        }
        response = supabase.table("usage_log").insert(data).execute()
        if response.data and len(response.data) > 0:
             return response.data[0]['id']
        return None
    except Exception as e:
        print(f"로그 저장 에러: {e}")
        return None

def update_feedback_in_db(log_id, status):
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
# 2. 데이터 및 모델 로드 (기존 동일)
# ==========================================
# logic.py 내부의 load_resources 함수

@st.cache_resource
def load_resources():
    """모델과 데이터를 메모리에 로드 (캐싱 사용)"""
    w2v = Word2Vec.load("w2v.model")
    d2v = Doc2Vec.load("d2v.model")
    df = pd.read_csv("final_recipe_data.csv")
    df['재료토큰'] = df['재료토큰'].apply(literal_eval)
    
    with open("stats.pkl", "rb") as f:
        stats = pickle.load(f)

    try:
        # [수정됨] encoding='utf-8-sig' 추가 및 컬럼명 공백 제거
        price_df = pd.read_csv("price_rank.csv", encoding='utf-8-sig')
        price_df.columns = price_df.columns.str.strip() # 컬럼명 앞뒤 공백 제거 안전장치
        price_map = dict(zip(price_df['ingredient'], price_df['rank']))
    except FileNotFoundError:
        print("Warning: price_rank.csv 파일을 찾을 수 없습니다. 가격 정보가 표시되지 않습니다.")
        price_map = {}
    except KeyError as e:
        # 컬럼명이 여전히 안 맞을 경우를 대비한 에러 처리
        print(f"Error: price_rank.csv 파일의 컬럼명을 확인해주세요. (Missing key: {e})")
        price_map = {}

    return w2v, d2v, df, stats, price_map

w2v_model, d2v_model, df, stats, price_map = load_resources()

method_map = stats["method_map"]
recipes_by_ingredient = stats["recipes_by_ingredient"]
ing_method_counts = stats["ing_method_counts"]
ing_cat_counts = stats["ing_cat_counts"]
total_method_counts = stats["total_method_counts"]
total_cat_counts = stats["total_cat_counts"]
TOTAL_RECIPES = stats["TOTAL_RECIPES"]

# ==========================================
# 3. 핵심 계산 로직 (기존 동일)
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

def get_estimated_price_rank(ing_name, price_map):
    if ing_name in price_map: return price_map[ing_name]
    if any(exp in ing_name for exp in PRICE_RULE_EXCEPTIONS): return 3
    for rank, keywords in PRICE_KEYWORD_RULES:
        if any(kw in ing_name for kw in keywords): return rank
    return 3

# ==========================================
# 4. 대체 추천 알고리즘 (기존 DB 기반)
# ==========================================
def substitute_single(recipe_id, target_ing, stopwords, w_w2v, w_d2v, w_method, w_cat, topn=10):
    # (기존 substitute_single 코드 내용과 동일)
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

    target_rank = get_estimated_price_rank(target_ing, price_map)
        
    candidates_raw = w2v_model.wv.most_similar(target_ing, topn=50)
    temp_results = []
    seen_candidates = set()
    
    for cand, score_w2v in candidates_raw:
        clean_cand = cand
        if stopwords:
            for stop in stopwords: clean_cand = clean_cand.replace(stop, "")
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
                    if rt in d2v_model.dv: sims.append(cos_sim(vec_recipe, d2v_model.dv[rt]))
                if sims: s_d2v = np.mean(sims)
        
        s_method = 0.0 if w_method <= 0 else get_stat_score(clean_cand, current_method, ing_method_counts, total_method_counts, TOTAL_RECIPES)
        s_cat = 0.0 if w_cat <= 0 else get_stat_score(clean_cand, current_cat, ing_cat_counts, total_cat_counts, TOTAL_RECIPES)
        cand_rank = get_estimated_price_rank(clean_cand, price_map)
        saving_score = target_rank - cand_rank
        temp_results.append({"대체재료": clean_cand, "raw_W2V": s_w2v, "raw_D2V": s_d2v, "raw_Method": s_method, "raw_Category": s_cat, "saving_score": saving_score})
        
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
    # (기존 substitute_multi 코드 내용과 동일)
    row = df[df['레시피일련번호'] == recipe_id].iloc[0]
    current_method = row['요리방법별명']
    current_cat = row['요리종류별명_세분화']
    initial_context = row['재료토큰']
    tag = f"recipe_{recipe_id}"
    vec_recipe = None
    if w_d2v > 0 and tag in d2v_model.dv: vec_recipe = d2v_model.dv[tag]
    total_weight = w_w2v + w_d2v + w_method + w_cat
    if total_weight == 0: total_weight = 1.0
    target_ranks_sum = 0
    for t in targets: target_ranks_sum += get_estimated_price_rank(t, price_map)
    beam = [(0.0, [], initial_context)]
    for target_ing in targets:
        next_beam = []
        if target_ing not in w2v_model.wv:
            for score, subs, ctx in beam: next_beam.append((score, subs + [target_ing], ctx))
            beam = next_beam
            continue
        for path_score, path_subs, path_ctx in beam:
            current_ctx_ing = [x for x in path_ctx if x != target_ing]
            candidates = w2v_model.wv.most_similar(target_ing, topn=30)
            temp_candidates = []
            seen_candidates = set()
            for cand, _ in candidates:
                clean_cand = cand
                if stopwords:
                    for stop in stopwords: clean_cand = clean_cand.replace(stop, "")
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
                    if len(same_method_ids) > 10:
                        np.random.seed(42)
                        same_method_ids = np.random.choice(same_method_ids, 10, replace=False)
                    if same_method_ids is not None and len(same_method_ids) > 0:
                        sims = []
                        for r in same_method_ids:
                            rt = f"recipe_{r}"
                            if rt in d2v_model.dv: sims.append(cos_sim(vec_recipe, d2v_model.dv[rt]))
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
    final_results = []
    for score, subs, _ in beam:
        avg_score = score / len(targets) if targets else 0.0
        cand_ranks_sum = 0
        for sub_ing in subs: cand_ranks_sum += get_estimated_price_rank(sub_ing, price_map)
        total_saving_score = target_ranks_sum - cand_ranks_sum
        final_results.append((subs, avg_score, total_saving_score))
    return final_results[:result_topn]

# ==========================================
# 5. [NEW] 커스텀 입력 기반 대체 알고리즘
# ==========================================
def substitute_single_custom(target_ing, context_ings_list, stopwords, w_w2v, w_d2v, topn=10):
    """커스텀 입력 기반 단일 대체 (통계 점수 제외)"""
    if target_ing not in w2v_model.wv: return pd.DataFrame()
    total_weight = w_w2v + w_d2v
    if total_weight == 0: total_weight = 1.0

    # 1. 입력된 재료 리스트로 Doc2Vec 벡터 추론 (Inference)
    vec_custom_context = None
    if w_d2v > 0:
        # 모델에 있는 단어만 필터링해서 추론에 사용
        valid_context = [word for word in context_ings_list if word in d2v_model.wv]
        if valid_context:
            vec_custom_context = d2v_model.infer_vector(valid_context)

    target_rank = get_estimated_price_rank(target_ing, price_map)
    candidates_raw = w2v_model.wv.most_similar(target_ing, topn=50)
    temp_results = []
    seen_candidates = set()

    for cand, score_w2v in candidates_raw:
        # ... (필터링 로직 동일) ...
        clean_cand = cand
        if stopwords:
            for stop in stopwords: clean_cand = clean_cand.replace(stop, "")
        clean_cand = clean_cand.strip()
        if not clean_cand: continue
        if clean_cand in context_ings_list: continue
        if clean_cand == target_ing: continue
        if clean_cand not in w2v_model.wv: continue
        if clean_cand in seen_candidates: continue
        seen_candidates.add(clean_cand)

        real_score_w2v = w2v_model.wv.similarity(target_ing, clean_cand)
        s_w2v = max(0.0, real_score_w2v)
        if s_w2v < 0.35: continue

        # 2. D2V 점수: 추론된 벡터와 후보 재료가 사용된 레시피 벡터들 간의 유사도
        s_d2v = 0.0
        if w_d2v > 0 and vec_custom_context is not None:
            rid_list = recipes_by_ingredient.get(clean_cand, [])
            # 방법/카테고리 필터링 없이 랜덤 샘플링
            if len(rid_list) > 20:
                np.random.seed(42)
                rid_list = np.random.choice(rid_list, 20, replace=False)
            if rid_list is not None and len(rid_list) > 0:
                sims = []
                for r in rid_list:
                    rt = f"recipe_{r}"
                    if rt in d2v_model.dv:
                        sims.append(cos_sim(vec_custom_context, d2v_model.dv[rt]))
                if sims: s_d2v = np.mean(sims)
        
        # 커스텀 모드에서는 통계 점수(Method, Category)를 계산할 수 없음 (0점 처리)
        s_method, s_cat = 0.0, 0.0

        cand_rank = get_estimated_price_rank(clean_cand, price_map)
        saving_score = target_rank - cand_rank
        temp_results.append({"대체재료": clean_cand, "raw_W2V": s_w2v, "raw_D2V": s_d2v, "raw_Method": s_method, "raw_Category": s_cat, "saving_score": saving_score})

    if not temp_results: return pd.DataFrame()
    
    # 정규화 및 최종 점수 계산 (통계 점수 제외하고 W2V, D2V만 사용)
    df_res = pd.DataFrame(temp_results)
    # 통계 점수 컬럼은 모두 0이므로 정규화해도 0.5가 됨. W2V, D2V만 의미 있음.
    cols = ["raw_W2V", "raw_D2V"]
    norm_cols = ["W2V", "D2V"]
    for raw_col, norm_col in zip(cols, norm_cols):
        min_val = df_res[raw_col].min()
        max_val = df_res[raw_col].max()
        if max_val - min_val == 0: df_res[norm_col] = 0.5
        else: df_res[norm_col] = (df_res[raw_col] - min_val) / (max_val - min_val)

    df_res["최종점수"] = ((df_res["W2V"]*w_w2v) + (df_res["D2V"]*w_d2v)) / total_weight
    return df_res.sort_values("최종점수", ascending=False).head(topn).reset_index(drop=True)


def substitute_multi_custom(targets, context_ings_list, stopwords, w_w2v, w_d2v, beam_width=3, result_topn=3):
    """커스텀 입력 기반 다중 대체 (Beam Search, 통계 점수 제외)"""
    total_weight = w_w2v + w_d2v
    if total_weight == 0: total_weight = 1.0

    # 1. 초기 문맥 벡터 추론
    vec_custom_context = None
    if w_d2v > 0:
        valid_context = [word for word in context_ings_list if word in d2v_model.wv]
        if valid_context:
            vec_custom_context = d2v_model.infer_vector(valid_context)

    target_ranks_sum = 0
    for t in targets: target_ranks_sum += get_estimated_price_rank(t, price_map)

    # Beam 초기화: [(점수, [대체 리스트], [현재 문맥 리스트])]
    beam = [(0.0, [], context_ings_list)]

    for target_ing in targets:
        next_beam = []
        if target_ing not in w2v_model.wv:
            for score, subs, ctx in beam: next_beam.append((score, subs + [target_ing], ctx))
            beam = next_beam
            continue

        for path_score, path_subs, path_ctx in beam:
            current_ctx_ing = [x for x in path_ctx if x != target_ing]
            candidates = w2v_model.wv.most_similar(target_ing, topn=30)
            temp_candidates = []
            seen_candidates = set()

            for cand, _ in candidates:
                # ... (필터링 및 조화도 계산 로직은 동일) ...
                clean_cand = cand
                if stopwords:
                    for stop in stopwords: clean_cand = clean_cand.replace(stop, "")
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

                # D2V 점수: 현재 경로의 문맥 벡터를 추론하여 비교
                s_d2v = 0.0
                if w_d2v > 0:
                    # 현재 경로의 문맥으로 벡터 추론
                    valid_path_ctx = [word for word in current_ctx_ing if word in d2v_model.wv]
                    if valid_path_ctx:
                        vec_path_context = d2v_model.infer_vector(valid_path_ctx)
                        
                        rid_list = recipes_by_ingredient.get(clean_cand, [])
                        if len(rid_list) > 10:
                            np.random.seed(42)
                            rid_list = np.random.choice(rid_list, 10, replace=False)
                        if rid_list is not None and len(rid_list) > 0:
                            sims = []
                            for r in rid_list:
                                rt = f"recipe_{r}"
                                if rt in d2v_model.dv: sims.append(cos_sim(vec_path_context, d2v_model.dv[rt]))
                            if sims: s_d2v = np.mean(sims)

                # 통계 점수는 0
                s_method, s_cat = 0.0, 0.0
                temp_candidates.append({"cand": clean_cand, "raw_w2v": s_w2v, "raw_d2v": s_d2v})

            if not temp_candidates: continue
            
            # 정규화 및 가중합 (W2V, D2V만 사용)
            df_temp = pd.DataFrame(temp_candidates)
            cols = ["raw_w2v", "raw_d2v"]
            for col in cols:
                min_val = df_temp[col].min()
                max_val = df_temp[col].max()
                if max_val - min_val == 0: df_temp[col + "_norm"] = 0.5
                else: df_temp[col + "_norm"] = (df_temp[col] - min_val) / (max_val - min_val)

            for _, r in df_temp.iterrows():
                weighted_sum = ((r["raw_w2v_norm"]*w_w2v) + (r["raw_d2v_norm"]*w_d2v)) / total_weight
                new_total_score = path_score + weighted_sum
                new_subs = path_subs + [r["cand"]]
                new_ctx = current_ctx_ing + [r["cand"]]
                next_beam.append((new_total_score, new_subs, new_ctx))

        next_beam.sort(key=lambda x: x[0], reverse=True)
        beam = next_beam[:beam_width]

    final_results = []
    for score, subs, _ in beam:
        avg_score = score / len(targets) if targets else 0.0
        cand_ranks_sum = 0
        for sub_ing in subs: cand_ranks_sum += get_estimated_price_rank(sub_ing, price_map)
        total_saving_score = target_ranks_sum - cand_ranks_sum
        final_results.append((subs, avg_score, total_saving_score))
    return final_results[:result_topn]