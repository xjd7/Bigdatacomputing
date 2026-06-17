import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# [조건 2] joblib으로 저장한 3개 모델과 메타데이터 불러오기
linear_model = joblib.load('linear_model.pkl')
poly_model = joblib.load('poly_model.pkl')
ridge_model = joblib.load('ridge_model.pkl')
metadata = joblib.load('metadata.pkl')

models = {
    'Linear': linear_model,
    'Poly': poly_model,
    'Ridge': ridge_model
}

features = metadata['features']
slider_info = metadata['slider_info']
summary_df = pd.DataFrame(metadata['summary'])

# Streamlit 기본 화면 구성
st.title('WHO 기대수명 예측 웹 서비스')
st.write('Schooling 특성을 제외한 여러 특성으로 기대수명을 예측합니다.')

# [조건 3] 성능 평가지표 테이블 출력
st.subheader('각 모델의 성능 비교')
st.dataframe(summary_df)

# [조건 3] Test R2 점수 막대그래프 출력
fig, ax = plt.subplots()
ax.bar(summary_df['Model'], summary_df['Test R2'])
ax.set_xlabel('Model')
ax.set_ylabel('Test R2')
ax.set_title('Test R2 Score')
st.pyplot(fig)

# [조건 4] 사이드바 슬라이더 입력
st.sidebar.header('입력값 조절')
user_inputs = []

for feature in features:
    min_value = float(slider_info[feature]['min'])
    max_value = float(slider_info[feature]['max'])
    mean_value = float(slider_info[feature]['mean'])

    value = st.sidebar.slider(
        feature,
        min_value=min_value,
        max_value=max_value,
        value=mean_value
    )
    user_inputs.append(value)

# [조건 4] 모델 선택 인터페이스
# 과제 조건에서 st.selectbox를 이용해 Linear, Poly, Ridge 중 하나를 선택하도록 요구한다.
selected_model_name = st.selectbox(
    '사용할 모델 선택',
    ['Linear', 'Poly', 'Ridge']
)

selected_model = models[selected_model_name]

# [조건 4] 선택 모델로 실시간 기대수명 예측
input_data = np.array([user_inputs])
prediction = selected_model.predict(input_data)

st.subheader('실시간 기대수명 예측 결과')
st.write('선택 모델:', selected_model_name)
st.write('예측 기대수명:', round(prediction[0], 2), '세')

# 입력값 확인용 테이블
input_df = pd.DataFrame([user_inputs], columns=features)
st.table(input_df)
