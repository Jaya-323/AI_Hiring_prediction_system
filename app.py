import streamlit as st
import pandas as pd
import joblib

# =========================================
# PAGE CONFIG
# =========================================

st.set_page_config(
    page_title="AI Hiring Prediction System",
    page_icon="💼",
    layout="wide"
)

# =========================================
# LOAD MODEL
# =========================================

model = joblib.load("hiring_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# =========================================
# CSS
# =========================================

st.markdown("""
<style>

/* Background */

.stApp{
    background:#081C3A;
}

/* Hide Streamlit menu */

#MainMenu{
visibility:hidden;
}

header{
visibility:hidden;
}

footer{
visibility:hidden;
}

/* Title */

.main-title{

font-size:46px;

font-weight:700;

color:white;

text-align:center;

}

.subtitle{

font-size:18px;

color:#CBD5E1;

text-align:center;

margin-bottom:25px;

}

/* Metric Cards */

.metric-card{

background:#13294B;

padding:20px;

border-radius:15px;

text-align:center;

border:1px solid #355C8C;

}

/* Footer */

.footer{

text-align:center;

color:#CBD5E1;

margin-top:40px;

font-size:15px;

}

/* Predict Button */

.stButton > button{

width:100%;

height:55px;

background:#2563EB;

color:white;

border:none;

border-radius:12px;

font-size:20px;

font-weight:bold;

}

.stButton > button:hover{

background:#1D4ED8;

color:white;

}

</style>
""", unsafe_allow_html=True)

# =========================================
# SIDEBAR
# =========================================

st.sidebar.title("💼 AI Hiring Prediction")

st.sidebar.markdown("---")

st.sidebar.info("""
This application predicts whether a candidate is likely to be hired based on:

• Experience

• Salary Expectation

• Projects Count

• Education

• Certifications

• Job Role
""")

st.sidebar.success("Model : XGBoost")

st.sidebar.success("Accuracy : 97%")

# =========================================
# HEADER
# =========================================

st.markdown("<h1 class='main-title'>💼 AI Hiring Prediction System</h1>", unsafe_allow_html=True)

st.markdown("<p class='subtitle'>Intelligent Candidate Screening Dashboard</p>", unsafe_allow_html=True)

# =========================================
# TOP CARDS
# =========================================

c1,c2,c3=st.columns(3)

with c1:
    st.markdown("""
    <div class="metric-card">
    <h3 style="color:white;">🤖 Model</h3>
    <h2 style="color:#60A5FA;">XGBoost</h2>
    </div>
    """,unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="metric-card">
    <h3 style="color:white;">🎯 Accuracy</h3>
    <h2 style="color:#4ADE80;">97%</h2>
    </div>
    """,unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="metric-card">
    <h3 style="color:white;">📊 Features</h3>
    <h2 style="color:#FACC15;">6</h2>
    </div>
    """,unsafe_allow_html=True)

st.write("")

st.subheader("📋 Candidate Information")

left,right=st.columns(2)

with left:

    experience=st.number_input(
        "Experience (Years)",
        0,
        40,
        2
    )

    projects=st.number_input(
        "Projects Count",
        0,
        30,
        3
    )

    education=st.selectbox(
        "Education",
        [
            "B.Tech",
            "M.Tech",
            "MBA",
            "PhD"
        ]
    )

with right:

    salary=st.number_input(
        "Salary Expectation ($)",
        10000,
        300000,
        60000
    )

    certification=st.selectbox(
        "Certification",
        [
            "Deep Learning Specialization",
            "Google ML",
            "No certifications"
        ]
    )

    job_role=st.selectbox(
        "Job Role",
        [
            "Software Engineer",
            "Data Scientist",
            "Cybersecurity Analyst"
        ]
    )

st.write("")

predict=st.button("🚀 Analyze Candidate")
# =========================================
# PREDICTION
# =========================================

if predict:

    # Create Input DataFrame
    input_data = pd.DataFrame({

        "Experience (Years)": [experience],
        "Salary Expectation ($)": [salary],
        "Projects Count": [projects],

        "Education_B.Tech": [1 if education == "B.Tech" else 0],
        "Education_M.Tech": [1 if education == "M.Tech" else 0],
        "Education_MBA": [1 if education == "MBA" else 0],
        "Education_PhD": [1 if education == "PhD" else 0],

        "Certifications_Deep Learning Specialization":
            [1 if certification == "Deep Learning Specialization" else 0],

        "Certifications_Google ML":
            [1 if certification == "Google ML" else 0],

        "Certifications_No certifications":
            [1 if certification == "No certifications" else 0],

        "Job Role_Cybersecurity Analyst":
            [1 if job_role == "Cybersecurity Analyst" else 0],

        "Job Role_Data Scientist":
            [1 if job_role == "Data Scientist" else 0],

        "Job Role_Software Engineer":
            [1 if job_role == "Software Engineer" else 0]
    })

    # Scale Numerical Features
    numerical_features = [
        "Experience (Years)",
        "Salary Expectation ($)",
        "Projects Count"
    ]

    input_data[numerical_features] = scaler.transform(
        input_data[numerical_features]
    )

    # Prediction
    prediction = model.predict(input_data)
    result = label_encoder.inverse_transform(prediction)[0]

    # Confidence
    confidence = model.predict_proba(input_data).max() * 100

    st.write("")
    st.markdown("---")

    # =========================================
    # RESULT SECTION
    # =========================================

    st.subheader("📊 Prediction Result")

    if result == "Hire":

        st.success(
            f"""
🎉 **Candidate is likely to be HIRED**

**Confidence Score:** {confidence:.2f}%
"""
        )

    else:

        st.error(
            f"""
❌ **Candidate is likely to be REJECTED**

**Confidence Score:** {confidence:.2f}%
"""
        )

    # =========================================
    # CANDIDATE SUMMARY
    # =========================================

    st.subheader("📋 Candidate Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Experience", f"{experience} Years")

    with col2:
        st.metric("Projects", projects)

    with col3:
        st.metric("Salary", f"${salary:,}")

    st.write("")

    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric("Education", education)

    with col5:
        st.metric("Certification", certification)

    with col6:
        st.metric("Job Role", job_role)

    st.markdown("---")

    # =========================================
    # RECOMMENDATION
    # =========================================

    st.subheader("💡 Career Recommendations")

    recommendations = []

    if experience < 2:
        recommendations.append("Gain more hands-on industry experience.")

    if projects < 3:
        recommendations.append("Build more practical projects to strengthen your portfolio.")

    if certification == "No certifications":
        recommendations.append("Earn a professional certification such as Google ML or Deep Learning Specialization.")

    if education == "B.Tech":
        recommendations.append("Consider pursuing higher education or advanced technical training.")

    if result == "Hire":
        st.success(
            "The candidate profile closely matches the hiring criteria used by the trained machine learning model."
        )
    else:
        if recommendations:
            for rec in recommendations:
                st.info(f"✔ {rec}")
        else:
            st.info("Continue improving your technical, communication, and problem-solving skills.")

# =========================================
# FOOTER
# =========================================

st.markdown("---")

st.markdown(
    """
<div style="text-align:center; color:#CBD5E1; padding:20px;">
    <h4>💼 AI Hiring Prediction System</h4>
    <p>Developed by <b>Jaya Mishra</b></p>
    <p>Built using Python, Streamlit, Scikit-learn & XGBoost</p>
</div>
""",
    unsafe_allow_html=True
)