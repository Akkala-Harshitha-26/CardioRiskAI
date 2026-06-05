# CardioRisk AI - Heart Disease Risk Prediction System
import streamlit as st
import numpy as np
import pandas as pd
from data import load_heart_data, clean_data, preprocess_data, get_simple_questions, verify_dataset, get_feature_descriptions
from models import (
    train_models,
    evaluate_models,
    cross_validate_models,
    plot_roc_curves_cv,
    plot_confusion_matrix_cv,
    explain_prediction
)
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioRisk AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Session state ──────────────────────────────────────────────────────────────
if "patient_history" not in st.session_state:
    st.session_state.patient_history = []

# ── Data & model pipeline ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_data():
    df = load_heart_data()
    df = clean_data(df)
    verify_dataset(df)
    return df

@st.cache_resource(show_spinner=False)
def get_trained_pipeline():
    df = get_data()
    X_train, X_test, y_train, y_test, scaler, pca = preprocess_data(df)
    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)
    best_name = max(results, key=lambda n: results[n]["f1"])
    best_model = models[best_name]
    return best_model, best_name, scaler, pca, models, results, X_test, y_test

# ── Render prediction explanation section ─────────────────────────────────────
def render_explanation(model, model_name, X_sample_scaled, original_feature_names, user_input):
    if model_name == 'SVM':
        st.info("ℹ️ Feature-level explanation is not available for SVM models.")
        return
    
    feature_descriptions = get_feature_descriptions()
    explanation = explain_prediction(
        model=model,
        model_name=model_name,
        X_sample=X_sample_scaled,
        feature_names=original_feature_names,
        feature_descriptions=feature_descriptions,
        user_values=user_input
    )
    
    if not explanation or not explanation.get('top_features'):
        st.info("Explanation not available for this model.")
        return
    
    st.markdown("### 🔍 Key Factors Influencing This Prediction")
    
    cols = st.columns(2)
    for i, feature in enumerate(explanation['top_features'][:4]):
        with cols[i % 2]:
            if model_name == 'Logistic Regression' and 'coefficient' in feature:
                if feature['coefficient'] > 0:
                    arrow, impact_text = "↑", "increases risk"
                else:
                    arrow, impact_text = "↓", "decreases risk"
            else:
                if feature.get('direction') == 'increases':
                    arrow, impact_text = "↑", "increases risk"
                else:
                    arrow, impact_text = "↓", "decreases risk"
            
            value = feature.get('value', 'N/A')
            value_display = f"{value:.1f}" if isinstance(value, float) else str(value)
            
            with st.container():
                st.markdown(f"**{feature['feature']}** - {feature['description']}")
                st.markdown(f"Value: **{value_display}** {arrow}")
                st.markdown(f"*{impact_text}*")
                st.divider()
    
    if explanation.get('recommendations'):
        st.markdown("### 📋 Personalized Recommendations")
        for rec in explanation['recommendations'][:3]:
            st.info(rec)

# ── Sidebar navigation ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# ❤️ CardioRisk AI")
    st.markdown("---")
    selected = option_menu(
        menu_title="Navigation",
        options=["Patient Assessment", "Clinical Dashboard", "Model Performance", "About"],
        icons=["heart-pulse", "graph-up", "bar-chart-line", "info-circle"],
        default_index=0,
        styles={
            "container": {"padding": "5px"},
            "icon": {"font-size": "14px"},
            "nav-link": {"font-size": "14px", "text-align": "left", "margin": "2px"},
            "nav-link-selected": {"background-color": "#2dd4bf", "color": "#0d1117"},
        }
    )
    st.markdown("---")
    st.caption(f"📅 {datetime.now().strftime('%B %d, %Y')}")

# =============================================================================
# PAGE: PATIENT ASSESSMENT
# =============================================================================
if selected == "Patient Assessment":
    st.title("📋 Patient Assessment")
    st.markdown("Enter clinical data to receive an AI-powered cardiovascular risk evaluation")
    st.markdown("---")
    
    # Patient info
    col1, col2, col3 = st.columns(3)
    with col1:
        patient_id = st.text_input("Patient ID", value=f"PT-{datetime.now().strftime('%Y%m%d')}-{len(st.session_state.patient_history)+1:03d}")
    with col2:
        patient_name = st.text_input("Patient Name", placeholder="Enter full name")
    with col3:
        visit_date = st.date_input("Visit Date", datetime.now())
    
    questions = get_simple_questions()
    categories = ["Demographics", "Symptoms", "Vitals", "Lab Results", "ECG", "Exercise", "Imaging"]
    user_input = {}
    tabs = st.tabs(categories)
    
    for tab, category in zip(tabs, categories):
        with tab:
            cat_qs = [q for q in questions if q["category"] == category]
            if cat_qs:
                cols = st.columns(2)
                for idx, q in enumerate(cat_qs):
                    with cols[idx % 2]:
                        st.markdown(f"**{q['question']}**")
                        if q["type"] == "number":
                            val = st.number_input(
                                q['feature'],
                                label_visibility="collapsed",
                                min_value=float(q["min"]),
                                max_value=float(q["max"]),
                                value=float(q["default"]),
                                step=float(q.get("step", 1.0)),
                                key=f"{category}_{q['feature']}"
                            )
                        else:
                            opts = q["options"]
                            sel = st.selectbox(
                                q['feature'],
                                label_visibility="collapsed",
                                options=list(opts.keys()),
                                key=f"{category}_{q['feature']}"
                            )
                            val = opts[sel]
                        user_input[q["feature"]] = val
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_btn = st.button("🚀 Run Risk Assessment", use_container_width=True, type="primary")
    
    if run_btn:
        with st.spinner("Analyzing patient data..."):
            best_model, best_name, scaler, pca, all_models, _, _, _ = get_trained_pipeline()
            df = get_data()
            feature_order = df.columns[:-1].tolist()
            X_list = [user_input.get(f, df[f].median()) for f in feature_order]
            X = np.array(X_list).reshape(1, -1)
            X_scaled = scaler.transform(X)
            X_pca = pca.transform(X_scaled)
            prob = best_model.predict_proba(X_pca)[0, 1]
            pred = best_model.predict(X_pca)[0]
            
            if prob < 0.3:
                risk_level, risk_color = "Low", "🟢"
            elif prob < 0.6:
                risk_level, risk_color = "Moderate", "🟡"
            else:
                risk_level, risk_color = "High", "🔴"
            
            st.markdown("---")
            st.subheader("📊 Assessment Results")
            
            # Display results in columns
            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.metric("Risk Level", f"{risk_color} {risk_level}")
            with res_col2:
                st.metric("Risk Score", f"{prob*100:.1f}%")
            with res_col3:
                pred_label = "⚠️ Disease Detected" if pred == 1 else "✅ No Disease"
                st.metric("Prediction", pred_label)
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Risk Score"},
                number={"suffix": "%", "font": {"size": 40}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkblue", "thickness": 0.3},
                    "steps": [
                        {"range": [0, 30], "color": "lightgreen"},
                        {"range": [30, 60], "color": "gold"},
                        {"range": [60, 100], "color": "lightcoral"},
                    ],
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            # Explanation
            render_explanation(
                model=best_model,
                model_name=best_name,
                X_sample_scaled=X_scaled,
                original_feature_names=feature_order,
                user_input=user_input
            )
            
            # Clinical recommendations
            st.markdown("### 💊 Clinical Recommendations")
            if risk_level == "Low":
                st.success("✅ **Follow-up:** Routine annual check-up\n\n✅ **Lifestyle:** Maintain healthy diet and regular exercise\n\n✅ **Monitoring:** Regular blood pressure checks")
            elif risk_level == "Moderate":
                st.warning("⚠️ **Follow-up:** Schedule cardiology consult within 3 months\n\n⚠️ **Testing:** Consider stress test and echocardiogram\n\n⚠️ **Lifestyle:** Implement dietary modifications")
            else:
                st.error("🚨 **Immediate:** Cardiology consultation required\n\n🚨 **Testing:** Urgent diagnostic workup recommended\n\n🚨 **Intervention:** Consider preventive medication")
            
            # Save to history
            if patient_name:
                st.session_state.patient_history.append({
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "patient_id": patient_id,
                    "patient_name": patient_name,
                    "risk_score": round(prob, 3),
                    "risk_level": risk_level,
                    "prediction": "Disease" if pred == 1 else "No Disease",
                    "model_used": best_name,
                })
                st.success("✅ Assessment saved to history!")

# =============================================================================
# PAGE: CLINICAL DASHBOARD
# =============================================================================
elif selected == "Clinical Dashboard":
    st.title("📊 Clinical Dashboard")
    st.markdown("Session analytics and patient assessment history")
    st.markdown("---")
    
    history = st.session_state.patient_history
    total = len(history)
    high_risk = sum(1 for p in history if p["risk_level"] == "High")
    moderate_risk = sum(1 for p in history if p["risk_level"] == "Moderate")
    low_risk = sum(1 for p in history if p["risk_level"] == "Low")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📋 Total Assessed", total)
    with col2:
        st.metric("🔴 High Risk", high_risk, delta="⚠️" if high_risk > 0 else None)
    with col3:
        st.metric("🟡 Moderate Risk", moderate_risk)
    with col4:
        st.metric("🟢 Low Risk", low_risk)
    
    st.markdown("---")
    
    # Pie chart
    if total > 0:
        fig = px.pie(
            names=["Low Risk", "Moderate Risk", "High Risk"],
            values=[low_risk, moderate_risk, high_risk],
            color_discrete_sequence=["green", "orange", "red"],
            title="Risk Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No assessments yet. Run a Patient Assessment first.")
    
    # History table
    st.markdown("### 📜 Assessment History")
    if history:
        df_hist = pd.DataFrame(history)
        st.dataframe(df_hist, use_container_width=True, hide_index=True)
    else:
        st.info("No records yet. Run a Patient Assessment first.")

# =============================================================================
# PAGE: MODEL PERFORMANCE
# =============================================================================
elif selected == "Model Performance":
    st.title("📈 Model Performance")
    st.markdown("Stratified 5-Fold Cross-Validation results")
    st.markdown("---")
    
    with st.spinner("Running 5-fold cross-validation..."):
        df = get_data()
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        cv_results_tuple, _ = cross_validate_models(X, y, n_splits=5)
        cv_results = cv_results_tuple
        
        best_by_f1 = max(cv_results.keys(), key=lambda m: cv_results[m]['f1_mean'])
        best_cv = cv_results[best_by_f1]
        
        st.success(f"🏆 Best model by F1 Score: **{best_by_f1}** ({best_cv['f1_mean']*100:.1f}% ± {best_cv['f1_std']*100:.2f}%)")
        
        st.markdown("### Cross-Validation Results (mean ± std)")
        
        cv_df = pd.DataFrame({
            model: {
                "Accuracy": f"{cv_results[model]['accuracy_mean']*100:.1f}% ± {cv_results[model]['accuracy_std']*100:.2f}%",
                "Precision": f"{cv_results[model]['precision_mean']*100:.1f}% ± {cv_results[model]['precision_std']*100:.2f}%",
                "Recall": f"{cv_results[model]['recall_mean']*100:.1f}% ± {cv_results[model]['recall_std']*100:.2f}%",
                "F1-Score": f"{cv_results[model]['f1_mean']*100:.1f}% ± {cv_results[model]['f1_std']*100:.2f}%",
                "AUC-ROC": f"{cv_results[model]['roc_auc_mean']*100:.1f}% ± {cv_results[model]['roc_auc_std']*100:.2f}%",
            } for model in cv_results
        }).T.rename_axis("Model")
        st.dataframe(cv_df, use_container_width=True)
        
        st.markdown("### ROC Curves - 5-Fold Cross-Validation")
        fig = plot_roc_curves_cv(cv_results)
        st.pyplot(fig)
        plt.close()

# =============================================================================
# PAGE: ABOUT
# =============================================================================
else:
    st.title("ℹ️ About CardioRisk AI")
    st.markdown("Cardiovascular risk assessment using the UCI Cleveland Heart Disease Dataset")
    st.markdown("---")
    
    st.warning(
        "⚠️ **Medical Disclaimer:** This tool is for educational and screening purposes only. "
        "Always consult qualified healthcare professionals. Predictions are based on statistical "
        "patterns in 303 historical records and do not replace clinical judgment."
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Mission")
        st.markdown("To provide accurate, accessible heart disease risk assessment using advanced machine learning on authentic medical data.")
        
        st.markdown("### ✨ Features")
        st.markdown("""
        - 🤖 AI-powered risk assessment with real-time predictions
        - 📝 Prediction explanation showing key factors
        - 💊 Personalized recommendations
        - 📊 Clinical dashboard with analytics
        - 🔬 Multi-model comparison with cross-validation
        - 📜 Patient history tracking
        """)
    
    with col2:
        st.markdown("### 🛠️ Technology Stack")
        st.markdown("""
        - **Dataset:** Cleveland Heart Disease (UCI Repository ID: 45)
        - **Samples:** 303 patient records
        - **Features:** 13 clinical attributes
        - **Models:** Random Forest, SVM, Logistic Regression, Decision Tree
        - **Validation:** Stratified 5-Fold Cross-Validation
        - **Framework:** Streamlit, Python, Scikit-learn
        """)
    
    st.markdown("---")
    st.caption("Developed by Ramya | 2024 CardioRisk AI — All rights reserved")