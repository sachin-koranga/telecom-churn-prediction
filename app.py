import streamlit as st
import pickle
import pandas as pd
from src.feature_engineering import FeatureEngineer
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import ConfusionMatrixDisplay


df = pd.read_csv("C:\\Users\\Vikash\\Desktop\\customer_churn_model\\data\WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Load model
with open("C:\\Users\\Vikash\\Desktop\\customer_churn_model\\model\\model_Churn (2).pkl", "rb") as f:
    artifact = pickle.load(f)

model = artifact["model"]
threshold = artifact["threshold"]

#create two tabs one for insights one for prduction
tab1, tab2, tab3 = st.tabs(["Customer Churn prediction ", "Business Insights","About"])


# Example inputs (CHANGE based on your dataset)
with tab1:
    st.title("🎯 Customer Churn Risk Predictor")
    st.caption("Predict whether a customer is likely to churn based on behavior and subscription details.")

    gender = st.selectbox("Gender", ["Female", "Male"])

    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])

    Partner = st.selectbox("Partner", ["Yes", "No"])

    Dependents = st.selectbox("Dependents", ["Yes", "No"])

    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])

    MultipleLines = st.selectbox(
        "Multiple Lines",
        ["No phone service", "No", "Yes"]
    )

    InternetService = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    OnlineSecurity = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

    OnlineBackup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    DeviceProtection = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )

    TechSupport = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )

    StreamingTV = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )

    StreamingMovies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )

    Contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    PaperlessBilling = st.selectbox("Paperless Billing",
        ["Yes", "No"]
    )

    PaymentMethod = st.selectbox("Payment Method",
        ["Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"]
    )

    # --- Numerical Inputs ---
    tenure = st.number_input("Tenure", min_value=0, max_value=72, value=1)

    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)

    TotalCharges = st.number_input("Total Charges", min_value=0.0)
    # Create DataFrame (IMPORTANT)
    input_data = pd.DataFrame({
        "SeniorCitizen": [SeniorCitizen],
        "Partner": [Partner],
        "Dependents": [Dependents],
        "MultipleLines": [MultipleLines],
        "InternetService": [InternetService],
        "OnlineSecurity": [OnlineSecurity],
        "OnlineBackup": [OnlineBackup],
        "DeviceProtection": [DeviceProtection],
        "TechSupport": [TechSupport],
        "StreamingTV": [StreamingTV],
        "StreamingMovies": [StreamingMovies],
        "Contract": [Contract],
        "PaperlessBilling": [PaperlessBilling],
        "PaymentMethod": [PaymentMethod],
        "MonthlyCharges": [MonthlyCharges],
        "TotalCharges": [TotalCharges],
        "tenure": [tenure]
    })
    
if st.button("Predict"):
    input_df = pd.DataFrame(input_data)
    
    prob = model.predict_proba(input_df)[:, 1][0]
    pred = int(prob > threshold)

    st.metric("Churn Probability", f"{prob:.2%}")

    if pred == 1:
        st.error("Churn ❌")
    else:
        st.success("No Churn ✅")

    # Risk Meter
    st.progress(int(prob * 100))

    if prob > 0.7:
        st.error(f"🔴 High Risk ({prob:.2f})")
    elif prob > 0.4:
        st.warning(f"🟠 Medium Risk ({prob:.2f})")
    else:
        st.success(f"🟢 Low Risk ({prob:.2f})")

    # Action Recommendation
    if pred == 1:
        st.subheader("💡 Recommended Action")
        st.write("""
        - Offer discount or retention plan
        - Provide personalized support
        - Check service issues (especially for high charges)
        """)
#With tab2  Business Insights 
with tab2:
    
    st.header("🔥 Key Business Insights")
    slide1 ,slide2 = st.columns(2)
    with slide1:
        insights = [
            "📉 Month-to-month contracts show highest churn → focus retention here",
            "💰 Higher monthly charges lead to higher churn risk",
            "⏳ Customers with low tenure churn the most → improve onboarding",
            "💳 Electronic payment users churn more → possible trust issues",
            "📦 Customers with add-on services churn less → upsell opportunity",
            "👥 Senior citizens have slightly higher churn → targeted engagement needed"
        ]

        for i in insights:
            st.write(i)
    with slide2:
        insights_2 = [
            "New customers are much more likely to churn",
            "Expensive plans increase churn risk",
            "Customers who stayed longer (paid more overall) are less likely to churn",
            "summary"
            "Churn is primarily driven by early-stage dissatisfaction and pricing sensitivity rather than long-term service issues"
        ]
        for i in insights_2:
            st.write(i)

    st.divider()

    # -----------------------------
    # Charts Section
    # -----------------------------
    st.header("📊 Key Visualizations")

    col1, col2 = st.columns(2)

    # -----------------------------
    # Chart 1: Contract vs Churn
    # -----------------------------
    with col1:
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="Contract", hue="Churn", ax=ax)
        st.pyplot(fig)
        st.caption("👉 Month-to-month customers churn the most")

    # -----------------------------
    # Chart 2: Tenure vs Churn
    # -----------------------------
    with col2:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x="Churn", y="tenure", ax=ax)
        st.pyplot(fig)
        st.caption("👉 Low tenure customers are high churn risk")

    # -----------------------------
    # Next Row
    # -----------------------------
    col3, col4 = st.columns(2)

    # -----------------------------
    # Chart 3: Monthly Charges vs Churn
    # -----------------------------
    with col3:


        fig, ax = plt.subplots()
        sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=ax)
        st.pyplot(fig)
        st.caption("👉 Higher charges increase churn probability")

    # -----------------------------
    # Chart 4: Payment Method vs Churn
    # -----------------------------
    with col4:
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="PaymentMethod", hue="Churn", ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.caption("👉 Certain payment methods show higher churn")

    # -----------------------------
    # Optional Chart
    # -----------------------------
    st.subheader("🌐 Internet Service vs Churn")

    fig, ax = plt.subplots()
    sns.countplot(data=df, x="InternetService", hue="Churn", ax=ax)
    st.pyplot(fig)
    st.caption("👉 Fiber users often show higher churn")

    st.markdown("---")
    st.header("Business Insights from Categorical Analysis")
    
    insights_brief = "",
    "1. Contract Type: Customers on Month-to-month contracts have a significantly higher churn rate compared to those on one or two-year contracts. Encouraging long-term contracts is a primary retention lever.",
    "2. Internet Service: Fiber optic users churn at a much higher rate than DSL users. This may indicate pricing dissatisfaction or service reliability issues specifically within the fiber segment.",
    "3. Payment Method: Customers using **Electronic Checks** are the most likely to churn. This group should be targeted with incentives to switch to automated/recurring payment methods like Credit Card or Bank Transfer.",
    "4. Support & Security Services: Customers without **TechSupport, OnlineSecurity, or OnlineBackup** show much higher churn rates. Upselling these 'stickiness' services can reduce the likelihood of customers leaving.",
    "5. Demographics: Senior Citizens** and customers **without partners or dependents** show a higher propensity to churn, suggesting these segments might be more price-sensitive or less 'anchored' to the service.",
    "6. **Paperless Billing**: Interestingly, customers with **Paperless Billing** enabled have a higher churn count, possibly due to a lack of physical engagement with the brand or easier switching behavior."
    
    st.markdown("---")
    #Section 1: Key Drivers
    st.subheader("🚨 Key Churn Drivers")

    st.error("High Risk Segments Identified:")

    st.write("""
    - Month-to-month contract customers
    - Customers with high monthly charges
    - New customers (low tenure)
    - Electronic check users
    """)
    #Section 2: Action Strategy (THIS IS WHAT COMPANIES WANT)
    st.subheader("🎯 Recommended Business Actions")

    col1, col2 = st.columns(2)

    with col1:
        st.success("Retention Strategies")
        st.write("""
        - Offer long-term contracts (1–2 year plans)
        - Provide onboarding support for new customers
        - Bundle services (TV, Security, Backup)
        """)

    with col2:
        st.warning("Risk Mitigation")
        st.write("""
        - Monitor high monthly charge customers
        - Improve trust in electronic payments
        - Target senior citizens with personalized plans
        """)
    #Section 3: Customer Segmentation (VERY POWERFUL)
    st.subheader("👥 Customer Segments")

    st.write("""
    🔴 High Risk:
    - Low tenure + High charges + Month-to-month

    🟡 Medium Risk:
    - Medium tenure + moderate usage

    🟢 Low Risk:
    - Long-term contract + high tenure
    """)
    #Section 4: Add Business KPI Impac
    st.subheader("💰 Business Impact")

    st.info("""
    Improving recall helps identify more churn customers, 
    which can significantly reduce revenue loss through retention campaigns.
    """)
with tab3:
    st.info("""
    The model is optimized for detecting churn customers (high recall), 
    even if it occasionally flags non-churn customers.
    """)

    st.write("""
    - 1278 → Correctly predicted non-churn
    - 475 → Correctly predicted churn
    - 431 → False alarms (predicted churn but not)
    - 133 → Missed churn customers
    """)

    st.subheader("📋 Classification Report")
    report = pd.DataFrame({
        "Metric": ["Precision", "Recall", "F1-score"],
        "Non-Churn (0)": [0.91, 0.75, 0.82],
        "Churn (1)": [0.52, 0.78, 0.63]
    })
    st.table(report)

    metrics = ["Precision", "Recall", "F1"]
    churn_values = [0.52, 0.78, 0.63]
    fig, ax = plt.subplots()
    ax.bar(metrics, churn_values)
    ax.set_title("Churn Class Performance")
    st.pyplot(fig)

    st.subheader("Business Interpretation")

    st.write("""
    - The model successfully identifies most customers who are likely to churn.
    - Some loyal customers may be incorrectly targeted.
    - This is acceptable because retaining a customer is usually cheaper than losing one.
    """)

    st.subheader("🔥 IMPORTANT: Why This is CORRECT")
    imp = [
            "✔ No Data Leakage",
            "SMOTE inside pipeline",
            "Applied only on training folds",
            "✔ Proper CV",
            "StratifiedKFold used",
            "✔ Correct Param Nami"]    
    for i in imp:
        st.write(i)

    
    st.subheader("✅ Model Reliability")
    st.write("""
    - No data leakage
    - SMOTE applied correctly (training only)
    - Stratified cross-validation used
    - Balanced evaluation metrics considered
    """)

    st.success("Model is optimized to identify churn customers effectively.")

