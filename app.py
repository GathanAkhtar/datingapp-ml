import streamlit as st
import pickle
import numpy as np

# Load model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('le_target.pkl', 'rb') as f:
    le_target = pickle.load(f)

# Load PCA
with open('pca.pkl', 'rb') as f:
    pca = pickle.load(f)

st.title("💘 Dating App Relationship Intent Predictor")
st.write("Predict what someone is looking for on a dating app!")

# Input fields
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ['Male', 'Female', 'Non-binary', 'Genderfluid', 'Prefer Not to Say'])
    sexual_orientation = st.selectbox("Sexual Orientation", ['Straight', 'Gay', 'Bisexual', 'Pansexual', 'Lesbian', 'Asexual'])
    location_type = st.selectbox("Location Type", ['Urban', 'Suburban', 'Metro', 'Small Town', 'Remote Area'])
    income_bracket = st.selectbox("Income Bracket", ['Low', 'Very Low', 'Middle', 'Upper-Middle', 'High'])
    education_level = st.selectbox("Education Level", ["Bachelor's", 'No Formal Education', "Master's", 'Postdoc', "Associate's"])
    age = st.slider("Age", 18, 60, 25)
    zodiac_sign = st.selectbox("Zodiac Sign", ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo', 'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces'])

with col2:
    body_type = st.selectbox("Body Type", ['Slim', 'Athletic', 'Average', 'Curvy', 'Plus Size'])
    app_usage_time_min = st.slider("App Usage Time (min)", 0, 500, 100)
    swipe_right_ratio = st.slider("Swipe Right Ratio", 0.0, 1.0, 0.5)
    likes_received = st.slider("Likes Received", 0, 500, 50)
    mutual_matches = st.slider("Mutual Matches", 0, 100, 10)
    message_sent_count = st.slider("Messages Sent", 0, 500, 50)
    emoji_usage_rate = st.slider("Emoji Usage Rate", 0.0, 1.0, 0.5)

st.markdown("""
    <style>
    div.stButton > button {
        background-color: #0066cc;
        color: white;
        border-radius: 8px;
        padding: 8px 20px;
        font-size: 16px;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #0052a3;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

if st.button("Predict!"):
    # Encode inputs
    gender_map = {'Male': 3, 'Female': 1, 'Non-binary': 4, 'Genderfluid': 2, 'Prefer Not to Say': 5}
    orientation_map = {'Straight': 5, 'Gay': 2, 'Bisexual': 1, 'Pansexual': 4, 'Lesbian': 3, 'Asexual': 0}
    location_map = {'Urban': 4, 'Suburban': 3, 'Metro': 2, 'Small Town': 1, 'Remote Area': 0}
    income_map = {'Low': 2, 'Very Low': 4, 'Middle': 3, 'Upper-Middle': 1, 'High': 0}
    education_map = {"Bachelor's": 0, 'No Formal Education': 1, "Master's": 2, 'Postdoc': 3, "Associate's": 4}
    zodiac_map = {'Aries': 0, 'Taurus': 11, 'Gemini': 3, 'Cancer': 1, 'Leo': 4, 'Virgo': 10, 'Libra': 5, 'Scorpio': 8, 'Sagittarius': 7, 'Capricorn': 2, 'Aquarius': 9, 'Pisces': 6}
    body_map = {'Slim': 4, 'Athletic': 0, 'Average': 1, 'Curvy': 2, 'Plus Size': 3}

    input_data = np.zeros(24)
    input_data[0] = gender_map.get(gender, 0)
    input_data[1] = orientation_map.get(sexual_orientation, 0)
    input_data[2] = location_map.get(location_type, 0)
    input_data[3] = income_map.get(income_bracket, 0)
    input_data[4] = education_map.get(education_level, 0)
    input_data[6] = app_usage_time_min
    input_data[8] = swipe_right_ratio
    input_data[10] = likes_received
    input_data[11] = mutual_matches
    input_data[14] = message_sent_count
    input_data[15] = emoji_usage_rate
    input_data[19] = age
    input_data[21] = zodiac_map.get(zodiac_sign, 0)
    input_data[23] = body_map.get(body_type, 0)

    # Scale & PCA
    input_scaled = scaler.transform([input_data])
    input_pca = pca.transform(input_scaled)

    # Predict
    prediction = model.predict(input_pca)[0]
    result = le_target.inverse_transform([prediction])[0]

    st.success(f"🎯 Predicted Relationship Intent: **{result}**")
    st.balloons()
