import streamlit as st
import pandas as pd
import pickle as pkl
import numpy as np

categorical_columns = ['engine_type', 'series', 'model'
    , 'brand', 'assemble_place', 'transmission']
st.markdown("<h1 style='text-align: center; color: red;'>Car Price Predictor</h1>",
            unsafe_allow_html=True)
st.sidebar.write("# Car Information")
st.sidebar.write("## Select the car features")
data = pd.read_csv("data/data_clean.csv")
rf_model = pkl.load(open('models/RandomeForestModel.pkl', 'rb'))
encode_model = pkl.load(open('models/encoders.pkl', 'rb'))


def predict(a, b, c, d, e, f, g, h, i, j):
    if j == 'Lắp ráp trong nước':
        j = 0
    elif j == 'Nhập khẩu':
        j = 1
    if i == 'Số tay':
        i = 0
    elif i == 'Số tự động':
        i = 1
    a = encode_model['engine_type'].transform([a])[0]
    b = encode_model['series'].transform([b])[0]
    c = encode_model['model'].transform([c])[0]
    d = encode_model['brand'].transform([d])[0]
    i = encode_model['transmission'].transform([i])[0]
    j = encode_model['assemble_place'].transform([j])[0]

    # Dự đoán giá xe
    price = rf_model.predict(pd.DataFrame(columns=['engine_type', 'series', 'model', 'brand', 'year', 'driven kms',
                                                   'num_of_seat', 'num_of_door', 'transmission', 'assemble_place'],
                                          data=np.array([a, b, c, d, e, f, g, h, i, j]).reshape(1, 10)))
    return st.write("Giá xe dự đoán cho xe của bạn là: ", conver_price(price[0]), " triệu VND")



def conver_price(price):
    # làm tròn đến hàng trăm nghìn
    price = round(price / 100000) * 100000
    return price / 1000000


# pipe.predict(car[['engine_type','series','model','brand','year','driven kms','num_of_seat','transmission','assemble_place']])
brand = st.sidebar.selectbox("Chọn hãng xe", data['brand'].unique())
model = st.sidebar.selectbox("Chọn Model xe", data['model'].unique())
series = st.sidebar.selectbox("Chọn dòng xe", data['series'].unique())
year = st.sidebar.number_input("Nhập năm sản xuất", min_value=1990, max_value=2023)
year = 2023 - year
engine_type = st.sidebar.selectbox("Chọn loại nhiên liệu sử dụng", data['engine_type'].unique())
driven_kms = st.sidebar.number_input("Nhập số km đã đi", min_value=0, max_value=1000000)
num_of_seat = st.sidebar.number_input("Nhập số ghế", min_value=2, max_value=100)
num_of_door = st.sidebar.number_input("Nhập số cửa", min_value=2, max_value=100)
transmission = st.sidebar.selectbox("Chọn loại xe", data['transmission'].unique())
assemble_place = st.sidebar.selectbox("Chọn nơi lắp ráp", data['assemble_place'].unique())

if st.sidebar.button("Predict"):
    predict(engine_type, series, model, brand, year, driven_kms, num_of_seat, num_of_door, transmission, assemble_place)
