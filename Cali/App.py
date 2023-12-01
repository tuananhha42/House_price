import streamlit as st # pip install streamlit
import pickle
import numpy as np
import matplotlib.pyplot as plt  # Thêm thư viện matplotlib

# Load mô hình hồi quy tuyến tính đã được lưu
model = pickle.load(open('Linear.pkl', 'rb'))
# load mô hình chuẩn hóa Min-Max từ tệp "minmax_scaler_x.pkl"
with open("minmax_scaler_x.pkl", "rb") as scaler_file:
    loaded_minmax_scale = pickle.load(scaler_file)
# Define a function to make predictions
def predict_price(longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income,
                  ocean_proximity):

    # Create a feature vector based on user inputs
    feature_vector = np.array([longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income,
                               0, 0, 0, 0, 0])  # Initialize ocean_proximity variables to 0

    if ocean_proximity == '<1H OCEAN':
        feature_vector[8] = 1
    elif ocean_proximity == 'INLAND':
        feature_vector[9] = 1
    elif ocean_proximity == 'ISLAND':
        feature_vector[10] = 1
    elif ocean_proximity == 'NEAR BAY':
        feature_vector[11] = 1
    elif ocean_proximity == 'NEAR OCEAN':
        feature_vector[12] = 1
# giải thích: Do dữ liệu trong model ta đã chuyển từ chữ về số bằng dummy
# nên bên này phải xử lý dưới dạng số. Nếu click chọn trong selectbox
# thì giá trị sẽ được truyền vào đây

    # Chuẩn hóa dữ liệu input_data bằng mô hình chuẩn hóa Min-Max
    input_data_normalized = loaded_minmax_scale.transform(feature_vector.reshape(1, -1))
    # Dự đoán giá trị bằng mô hình Linear Regressor đã nạp
    predicted_price = model.predict(input_data_normalized)

    return predicted_price[0]

# Create a Streamlit web app
st.title('California Housing Price Prediction')
st.sidebar.header('Input Features')

# Input fields for user to enter feature values
longitude = st.sidebar.number_input('Longitude', value=0.0)
latitude = st.sidebar.number_input('Latitude', value=0.0)
housing_median_age = st.sidebar.number_input('Housing Median Age', value=0)
total_rooms = st.sidebar.number_input('Total Rooms', value=0)
total_bedrooms = st.sidebar.number_input('Total Bedrooms', value=0)
population = st.sidebar.number_input('Population', value=0)
households = st.sidebar.number_input('Households', value=0)
median_income = st.sidebar.number_input('Median Income', value=0.0)
ocean_proximity = st.sidebar.selectbox('Ocean Proximity', ('<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'))

# Định nghĩa CSS trực tiếp bằng cách sử dụng st.markdown
st.markdown(
    """
    <style>
    .red-text {
        color: red;
        font-size: 30px;  /* Thay đổi cỡ chữ thành 30px */
    }
    </style>

    <style>
    .edit-text_yellow {
        color: #D4F005;
        font-size: 16px;  /* Thay đổi cỡ chữ thành 18px */
    }
    </style>
    """,
    unsafe_allow_html=True
)

#Tạo 1 list để lưu dữ liệu dự đoán sau mỗi lần ấn nút predict bằng session state
if "predicted_prices" not in st.session_state:
    st.session_state.predicted_prices = []

# Calculate the predicted house price
if st.sidebar.button('Predict'):
    predicted_price = predict_price(longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity)
    st.session_state.predicted_prices.append(predicted_price)
    st.markdown(f'<p class="red-text">Predicted House Price: ${(predicted_price):,.2f}</p>', unsafe_allow_html=True) # sử dụng markdown đã đc tạo từ trc

    # Vẽ biểu đồ dựa trên danh sách predicted_prices với nền đen
    plt.figure(figsize=(8, 6), facecolor='black')
    plt.plot(range(1, len(st.session_state.predicted_prices) + 1), st.session_state.predicted_prices, color='lime')
    plt.xlabel('Prediction Number', color='white')
    plt.ylabel('Predicted Price', color='white')
    plt.title('Biểu đồ giá dự đoán', color='white')
    plt.gca().set_facecolor('black')
    plt.tick_params(axis='x', colors='white')
    plt.tick_params(axis='y', colors='white')
    st.pyplot(plt)

    # Hiển thị danh sách giá dự đoán từ các lần nhấn trước đó
    if st.session_state.predicted_prices:
        st.write("Danh sách giá dự đoán từ các lần nhấn trước đó:")
        for i, price in enumerate(st.session_state.predicted_prices):
            st.markdown(f'<p class="edit-text_yellow">Dự đoán {i + 1}: ${price:,.2f}</p>', unsafe_allow_html=True)

    # Hiển thị hình ảnh dựa trên giá trị predicted_price
    if predicted_price < 300000:
        st.image("DATA/img1.jpg", use_column_width=True)
    elif 300000 <= predicted_price <= 500000:
        st.image("DATA/img2.jpg", use_column_width=True)
    else:
        st.image("DATA/img3.jpg", use_column_width=True)

# This will clear the user inputs
if st.sidebar.button('Reset'):
    # Đặt lại tất cả các giá trị về giá trị mặc định
    longitude = 0.0
    latitude = 0.0
    housing_median_age = 0
    total_rooms = 0
    total_bedrooms = 0
    population = 0
    households = 0
    median_income = 0.0
    ocean_proximity = '<1H OCEAN'

    # Xóa danh sách predicted_prices
    st.session_state.predicted_prices = []
# Provide some information about the app
st.write('This app predicts California housing prices using a Linear Regression model.')

print(st.session_state.predicted_prices)
