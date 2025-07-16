import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import sys

# 루트 디렉터리를 import 경로에 추가해 AutoIntMLP 모듈을 읽어올 수 있게 합니다.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
from autoint import AutoIntModel, predict_model
from autointMLP import AutoIntMLPModel  # AutoInt+ (AutoIntMLP)


# streamlit run show_st.py

# 모델·데이터를 캐시해 반복 로딩 시간을 단축합니다.
@st.cache_resource
def load_data(model_type: str = "AutoInt"):
    '''
    앱에서 보여줄 필요 데이터를 가져오는 함수입니다.
    - 사용자, 영화, 평점 데이터를 가져옵니다.
    - 앞서 저장된 모델도 불러오고 구현해 둡니다.
    '''
    # 스크립트가 위치한 디렉터리를 기준으로 경로를 계산해 실행 위치에 상관없이 동작하도록 합니다.
    project_path = os.path.dirname(os.path.abspath(__file__))
    data_dir_nm = 'data'
    movielens_dir_nm = 'ml-1m'
    model_dir_nm = 'model'
    data_path = f"{project_path}/{data_dir_nm}"
    model_path = f"{project_path}/{model_dir_nm}"
    field_dims = np.load(f'{data_path}/field_dims.npy')
    dropout= 0.4
    embed_dim= 16
    
    ratings_df = pd.read_csv(f'{data_path}/{movielens_dir_nm}/ratings_prepro.csv')
    movies_df = pd.read_csv(f'{data_path}/{movielens_dir_nm}/movies_prepro.csv')
    user_df = pd.read_csv(f'{data_path}/{movielens_dir_nm}/users_prepro.csv')

    # --- 모델 선택 --- #
    if model_type == "AutoInt+":
        model = AutoIntMLPModel(field_dims=field_dims, embedding_size=embed_dim,
                                att_layer_num=3, att_head_num=2, dnn_hidden_units=(128, 64),
                                dnn_dropout=dropout)
        weights_fname = "autoIntMLP_model.weights.h5"
    else:  # 기본 AutoInt
        model = AutoIntModel(field_dims, embed_dim, att_layer_num=3, att_head_num=2, att_res=True,
                             l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, dnn_dropout=dropout, init_std=0.0001)
        weights_fname = "autoInt_model.weights.h5"

    # 더미 입력으로 모델 빌드
    dummy_input = tf.zeros((1, len(field_dims)), dtype=tf.int64)
    model(dummy_input)

    # 가중치 로드 (파일이 없으면 경고만 출력)
    weight_path = f"{model_path}/{weights_fname}"
    if os.path.exists(weight_path):
        model.load_weights(weight_path)
    else:
        st.warning(f"[경고] 가중치 파일을 찾을 수 없습니다: {weight_path}. \n새로 학습 후 저장하세요.")
    label_encoders = joblib.load(f'{data_path}/label_encoders.pkl')
    
    return user_df, movies_df, ratings_df, model, label_encoders


def get_user_seen_movies(ratings_df):
    '''
    사용자가 과거에 보았던 영화 리스트를 가져옵니다.
    '''
    user_seen_movies = ratings_df.groupby('user_id')['movie_id'].apply(list).reset_index()
    return user_seen_movies

def get_user_non_seed_dict(movies_df, user_df, user_seen_movies):
    '''
    사용자가 보지 않았던 영화 리스트를 가져옵니다.
    '''
    unique_movies = movies_df['movie_id'].unique()
    unique_users = user_df['user_id'].unique()
    user_non_seen_dict = dict()

    for user in unique_users:
        user_seen_movie_list = user_seen_movies[user_seen_movies['user_id'] == user]['movie_id'].values[0]
        user_non_seen_movie_list = list(set(unique_movies) - set(user_seen_movie_list))
        user_non_seen_dict[user] = user_non_seen_movie_list
        
    return user_non_seen_dict


def get_user_info(user_id):
    '''
    사용자 정보를 가져옵니다.
    '''
    return users_df[users_df['user_id'] == user_id]

def get_user_past_interactions(user_id):
    '''
    사용자 평점 데이터 중 4점 이상(선호했다는 정보)만 가져옵니다. 
    '''
    return ratings_df[ (ratings_df['user_id'] == user_id) & (ratings_df['rating'] >= 4)].merge(movies_df, on='movie_id')


def get_recom(user, user_non_seen_dict, user_df, movies_df, r_year, r_month, model, label_encoders):
    '''
    아래와 같은 순서로 추천 결과를 가져옵니다.
    1. 스트림릿에서 입력 받은 타겟 월, 연도, 사용자 정보를 받아옴
    2. 사용자가 보지 않았던 정보 추출
    3. model input으로 넣을 수 있는 형태로 데이터프레임 구성
    4. label encoder 적용해 모델에 넣을 준비
    5. 모델 예측 수행
    '''
    user_non_seen_movie = user_non_seen_dict.get(user)
    user_id_list = [user for _ in range(len(user_non_seen_movie))]
    r_decade = str(r_year - (r_year % 10)) + 's'
    
    user_non_seen_movie = pd.merge(pd.DataFrame({'movie_id':user_non_seen_movie}), movies_df, on='movie_id')
    user_info = pd.merge(pd.DataFrame({'user_id':user_id_list}), user_df, on='user_id')
    user_info['rating_year'] = r_year
    user_info['rating_month'] = r_month
    user_info['rating_decade'] = r_decade
    
    merge_data = pd.concat([user_non_seen_movie, user_info], axis=1)
    merge_data.fillna('no', inplace=True)
    merge_data = merge_data[['user_id', 'movie_id','movie_decade', 'movie_year', 'rating_year', 'rating_month', 'rating_decade', 'genre1','genre2', 'genre3', 'gender', 'age', 'occupation', 'zip']]
    
    for col, le in label_encoders.items():
        merge_data[col] = le.fit_transform(merge_data[col])
    
    recom_top = predict_model(model, merge_data)
    # 추천 중 영화 id에 해당되는 부분만 가져옴
    recom_top = [r[0] for r in recom_top]
    # 원본 영화 id로 변환
    origin_m_id = label_encoders['movie_id'].inverse_transform(recom_top)
    
    # 영화 리스트를 매칭시켜 가져옴
    return movies_df[movies_df['movie_id'].isin(origin_m_id)]

# 사이드바 – 모델 선택
with st.sidebar:
    st.header("설정")
    model_type = st.selectbox("사용할 추천 모델", ["AutoInt", "AutoInt+"], index=0)

# 데이터 불러오기 (모델 타입에 따라 캐싱 구분)
users_df, movies_df, ratings_df, model, label_encoders = load_data(model_type)
user_seen_movies = get_user_seen_movies(ratings_df)
user_non_seen_dict = get_user_non_seed_dict(movies_df, users_df, user_seen_movies)

# 타이틀
st.title("🎬 영화 추천 데모")
st.subheader(f"현재 모델: {model_type}")

st.header("사용자 정보를 넣어주세요.")
user_id = st.number_input("사용자 ID 입력", min_value=users_df['user_id'].min(), max_value=users_df['user_id'].max(), value=users_df['user_id'].min())
r_year = st.number_input("추천 타겟 연도 입력", min_value=ratings_df['rating_year'].min(), max_value=ratings_df['rating_year'].max(), value=ratings_df['rating_year'].min())
r_month = st.number_input("추천 타겟 월 입력", min_value=ratings_df['rating_month'].min(), max_value=ratings_df['rating_month'].max(), value=ratings_df['rating_month'].min())
 

# streamlit run show_st.py --client.showErrorDetails=false
if st.button("추천 결과 보기"):
    st.write("사용자 기본 정보")
    user_info = get_user_info(user_id)
    st.dataframe(user_info)

    st.write("사용자가 과거에 봤던 이력(평점 4점 이상)")
    user_interactions = get_user_past_interactions(user_id)
    st.dataframe(user_interactions)

    st.write("추천 결과")
    recommendations = get_recom(user_id, user_non_seen_dict, users_df, movies_df, r_year, r_month, model, label_encoders)
    
    st.dataframe(recommendations)