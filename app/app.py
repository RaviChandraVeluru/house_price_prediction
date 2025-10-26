import streamlit as st
import pandas as pd
from prediction import make_prediction

df = pd.read_csv('C:/Users/ravic/OneDrive/Desktop/PJS/ML/RL/Beginner/Project-1/data.csv')

locations = df['location'].unique().tolist()
area = df['area_type'].unique().tolist()
availability = df['availability'].unique().tolist()

st.set_page_config(page_title="House Price Prediction", page_icon="📊")
st.title('Bengaluru House Price Prediction:')

row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)

with row1[0]:
    loc = st.selectbox('Location', locations)
with row1[1]:
    area = st.selectbox('Area Type', area)
with row1[2]:
    avail = st.selectbox('Availability', availability)

with row2[0]:
    bedroom = st.slider('bedrooms', min_value=1, max_value=5)
with row2[1]:
    bathroom = st.slider('bathrooms', min_value=2, max_value=5)
with row2[2]:
    balcony = st.slider('balcony', min_value=0, max_value=5)
with row3[0]:
    sqft = st.number_input('Area square foot')

if st.button('Predict'):
    result = make_prediction(loc, area, avail, bedroom, bathroom, balcony, sqft)
    st.metric('Predicted House Value',f"₹ {result:.3f} lac.")
    st.warning('This is just a prediction Original may vary!')