import plotly.express as px
import plotly.graph_objects as go
import joblib
import streamlit as st
import pandas as pd
import joblib
from database import (
    create_db,
    save_prediction,
    get_predictions
)
create_db()

rf_model = joblib.load("models/randomforest.pkl")

st.set_page_config(
    page_title="Insurance Underwriting System",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>

.main{
    background-color:#0B1120;
}

.block-container{
    padding-top:1rem;
}

.card{
    background: linear-gradient(145deg,#111827,#1E293B);
    padding:20px;
    border-radius:15px;
    border:1px solid #334155;
    box-shadow:0px 4px 15px rgba(0,0,0,0.4);
}

.metric-card{
    background: linear-gradient(135deg,#7C3AED,#2563EB);
    padding:20px;
    border-radius:15px;
    color:white;
    text-align:center;
}

.small-card{
    background:#111827;
    border-radius:15px;
    padding:15px;
    border:1px solid #334155;
}

</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
# 🛡️ Insurance AI

### Dashboard

🏠 Premium Prediction

📊 Analytics Dashboard

📜 Prediction History

---

### About System

AI-powered underwriting using:

✔ DBSCAN

✔ PCA

✔ Random Forest

✔ Linear Regression
""")

st.markdown("""
<div style="
padding:25px;
border-radius:18px;
background:linear-gradient(90deg,#8B5CF6,#2563EB);
text-align:center;
color:white;
">

<h1>🛡️ Insurance Premium Underwriting System</h1>

<h3>AI-Powered Risk Assessment & Premium Estimation</h3>

DBSCAN • PCA • Stacked Random Forest • Linear Regression

</div>
""", unsafe_allow_html=True)

# Load model
model = joblib.load("models/premium_model.pkl")

st.title("Insurance Premium Prediction System")

st.write("Enter customer details")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Premium Prediction",
        "Analytics Dashboard",
        "Prediction History",
        "Model Performance"
    ]
)

#prediction tab
with tab1:

    # Inputs
    left, right = st.columns([3, 2])

    with left:

        st.markdown("### 👤 Customer Details")
        age = st.number_input("Age", 18, 100, 30)
        bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
        children = st.number_input("Children", 0, 10, 0)

    with right:

        
        sex = st.selectbox(
            "Gender",
            ["male", "female"]
        )

        smoker = st.selectbox(
            "Smoker",
            ["yes", "no"]
        )

        region = st.selectbox(
            "Region",
            [
                "northeast",
                "northwest",
                "southeast",
                "southwest"
            ]
        )

        predict_btn = st.button(
            "🚀 Calculate Premium",
            use_container_width=True
        )

    if predict_btn:

        risk_score = age * bmi
        family_risk = children * bmi

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={'text': "Risk Score"},
            gauge={
                'axis': {'range':[0,2500]},
                'steps':[
                    {'range':[0,800],'color':'green'},
                    {'range':[800,1500],'color':'orange'},
                    {'range':[1500,2500],'color':'red'}
                ]
            }
        ))

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        data = {
            "age": [age],
            "bmi": [bmi],
            "children": [children],

            "sex_male": [1 if sex == "male" else 0],
            "smoker_yes": [1 if smoker == "yes" else 0],

            "region_northwest": [1 if region == "northwest" else 0],
            "region_southeast": [1 if region == "southeast" else 0],
            "region_southwest": [1 if region == "southwest" else 0],

            "risk_score": [risk_score],
            "family_risk": [family_risk]
        }

        df = pd.DataFrame(data)

        expected_columns = [
            'age',
            'bmi',
            'children',
            'sex_male',
            'smoker_yes',
            'region_northwest',
            'region_southeast',
            'region_southwest',
            'risk_score',
            'family_risk'
        ]

        df = df[expected_columns]

        prediction = model.predict(df)

        premium = prediction[0]

        import shap

        explainer = shap.TreeExplainer(rf_model)

        shap_values = explainer.shap_values(df)

        # Risk Category
        if premium > 30000:
            risk_category = "🔴 High Risk"
        elif premium > 15000:
            risk_category = "🟡 Medium Risk"
        else:
            risk_category = "🟢 Low Risk"

        # Save Prediction
        save_prediction(
            age,
            bmi,
            premium,
            risk_category
        )

        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class='metric-card'>
            <h4>Premium</h4>
            <h1>₹{premium:,.0f}</h1>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class='metric-card'>
            <h4>Risk Score</h4>
            <h1>{risk_score:.0f}</h1>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class='metric-card'>
            <h4>Risk Category</h4>
            <h1>{risk_category}</h1>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class='metric-card'>
            <h4>Family Risk</h4>
            <h1>{family_risk:.0f}</h1>
            </div>
            """, unsafe_allow_html=True)


        st.markdown("<br>", unsafe_allow_html=True)


        from report_generator import generate_report

        if st.button("📄 Generate Report"):

            generate_report(
                age,
                bmi,
                premium,
                risk_category
            )

            with open(
                "customer_report.pdf",
                "rb"
            ) as file:

                st.download_button(
                    "⬇ Download Report",
                    file,
                    file_name="report.pdf"
                )

        c1,c2,c3,c4 = st.columns(4)

        with c1:
            st.markdown("""
            <div class='small-card'>
            <h4>🧠 AI-Powered</h4>
            <p>Advanced ML models for accurate predictions</p>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <div class='small-card'>
            <h4>⚡ Real-Time</h4>
            <p>Instant premium calculations</p>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown("""
            <div class='small-card'>
            <h4>🔒 Secure</h4>
            <p>Your data is encrypted and secure</p>
            </div>
            """, unsafe_allow_html=True)

        with c4:
            st.markdown("""
            <div class='small-card'>
            <h4>✅ Reliable</h4>
            <p>Trained on real insurance datasets</p>
            </div>
            """, unsafe_allow_html=True)


        st.subheader("Why was this premium predicted?")

        contributions = pd.DataFrame({
            "Feature": df.columns,
            "Contribution": shap_values[0].flatten()
        })

        st.dataframe(contributions)

        fig = px.bar(
            contributions,
            x="Contribution",
            y="Feature",
            orientation="h",
            title="Premium Drivers"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        top_feature = contributions.loc[
            contributions["Contribution"].abs().idxmax()
        ]

        st.success(
            f"""
            Biggest premium driver:
            {top_feature['Feature']}
            contributed approximately
            {abs(top_feature['Contribution']):.2f}
            units to the prediction.
            """
        )


#analytic tab

with tab2:

    st.subheader("Analytics Dashboard")

    dataset = pd.read_csv("data/insurance.csv")

    fig1 = px.histogram(
        dataset,
        x="charges",
        title="Premium Distribution"
    )

    region_avg = dataset.groupby(
        "region"
    )["charges"].mean().reset_index()

    fig2 = px.bar(
        region_avg,
        x="region",
        y="charges",
        title="Average Premium by Region"
    )

    fig3 = px.box(
        dataset,
        x="smoker",
        y="charges",
        title="Premium vs Smoker"
    )

    col1,col2,col3 = st.columns(3)

    with col1:
        st.plotly_chart(fig1)

    with col2:
        st.plotly_chart(fig2)

    with col3:
        st.plotly_chart(fig3)

    fig4 = px.scatter(
        dataset,
        x="age",
        y="charges",
        title="Age vs Premium",
        color="smoker"
    )

    st.plotly_chart(fig4, use_container_width=True)

    fig5 = px.scatter(
        dataset,
        x="bmi",
        y="charges",
        title="BMI vs Premium",
        color="smoker"
    )

    st.plotly_chart(fig5, use_container_width=True)

    children_avg = dataset.groupby(
        "children"
    )["charges"].mean().reset_index()

    fig6 = px.line(
        children_avg,
        x="children",
        y="charges",
        title="Average Premium by Children"
    )

    st.plotly_chart(fig6, use_container_width=True)

    gender_count = dataset["sex"].value_counts().reset_index()

    fig7 = px.pie(
        gender_count,
        names="sex",
        values="count",
        title="Gender Distribution"
    )

    st.plotly_chart(fig7, use_container_width=True)

    smoker_count = dataset["smoker"].value_counts().reset_index()

    fig8 = px.pie(
        smoker_count,
        names="smoker",
        values="count",
        title="Smoker Distribution"
    )

    st.plotly_chart(fig8, use_container_width=True)


    dataset["risk"] = pd.cut(
        dataset["charges"],
        bins=[0,15000,30000,100000],
        labels=["Low","Medium","High"]
    )

    risk_count = dataset["risk"].value_counts().reset_index()

    fig10 = px.bar(
        risk_count,
        x="risk",
        y="count",
        title="Risk Category Distribution"
    )

    st.plotly_chart(fig10, use_container_width=True)

    importance = pd.read_csv(
        "reports/feature_importance.csv"
    )

    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    corr = dataset.select_dtypes(
        include="number"
    ).corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with tab3:

    history = get_predictions()

    col1,col2,col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Predictions",
            len(history)
        )

    with col2:
        st.metric(
            "Average Premium",
            f"₹{history['premium'].mean():,.0f}"
        )

    with col3:
        st.metric(
            "Highest Premium",
            f"₹{history['premium'].max():,.0f}"
        )

    st.dataframe(
        history,
        use_container_width=True
    )

    import plotly.express as px

    fig = px.line(
        history,
        y="premium",
        title="Premium Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with tab4:

    st.subheader("Model Performance")

    results = pd.read_csv(
        "reports/model_results.csv"
    )

    st.dataframe(results)

    import plotly.express as px

    fig = px.bar(
        results,
        x="Model",
        y="R2",
        color="Model",
        title="Model Comparison"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    import plotly.express as px

    import pandas as pd

    


st.markdown("---")

st.markdown("""
<center>

Developed by Akshay Singh

B.Tech CSE

General Insurance Premium Underwriting System

</center>
""", unsafe_allow_html=True)