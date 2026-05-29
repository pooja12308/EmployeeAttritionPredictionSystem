import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .metric-card h2 { font-size: 2rem; margin: 0; }
    .metric-card p  { font-size: 0.85rem; margin: 0; opacity: 0.85; }
    .metric-card-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .metric-card-orange {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
    }
    .metric-card-red {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    }
    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #374151;
        border-left: 4px solid #667eea;
        padding-left: 0.75rem;
        margin: 1.5rem 0 1rem 0;
    }
    .risk-high   { background:#fee2e2; color:#991b1b; padding:4px 12px; border-radius:20px; font-weight:600; }
    .risk-medium { background:#fef3c7; color:#92400e; padding:4px 12px; border-radius:20px; font-weight:600; }
    .risk-low    { background:#d1fae5; color:#065f46; padding:4px 12px; border-radius:20px; font-weight:600; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Data loading & preprocessing
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/WA_Fn-UseC_-HR-Employee-Attrition.csv"
    try:
        df = pd.read_csv(url)
    except Exception:
        st.error("Could not fetch dataset. Please upload a CSV below.")
        return None
    return df

@st.cache_data
def preprocess(df):
    data = df.copy()
    # Encode categoricals
    le_dict = {}
    for col in data.select_dtypes(include='object').columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        le_dict[col] = le
    # Drop useless columns
    drop_cols = [c for c in ['EmployeeCount','StandardHours','Over18','EmployeeNumber'] if c in data.columns]
    data.drop(columns=drop_cols, inplace=True)
    return data, le_dict

@st.cache_resource
def train_models(data):
    X = data.drop(columns=['Attrition'])
    y = data['Attrition']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train_sc, y_train)

    results = {}
    trained  = {}
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', C=0.1, max_iter=2000),
        "Decision Tree":       DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42, class_weight='balanced'),
        "Random Forest":       RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced'),
        "SVM":                 SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced', probability=True, random_state=42),
        "KNN":                 KNeighborsClassifier(n_neighbors=7, weights='distance', metric='manhattan'),
        "Naive Bayes":         GaussianNB(),
    }

    for name, m in models.items():
        m.fit(X_train_sm, y_train_sm)
        y_pred = m.predict(X_test_sc)
        results[name] = {
            "accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
            "precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
            "report":    classification_report(y_test, y_pred, output_dict=True),
            "cm":        confusion_matrix(y_test, y_pred),
        }
        trained[name] = m

    return trained, results, scaler, X_train, X_test, y_train, y_test, X.columns.tolist()

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/conference-call.png", width=70)
    st.markdown("## 👥 HR Attrition AI")
    st.markdown("IBM HR Analytics Dataset")
    st.divider()
    st.markdown("**Navigation**")
    page = st.radio("", [
        "📊 Overview & EDA",
        "🤖 Model Comparison",
        "🔮 Predict Employee",
        "🗺️ Employee Segmentation",
    ], label_visibility="collapsed")
    st.divider()
    uploaded = st.file_uploader("Upload your own CSV", type=["csv"])

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
if uploaded:
    raw = pd.read_csv(uploaded)
else:
    raw = load_data()

if raw is None:
    st.warning("Please upload a CSV file via the sidebar.")
    st.stop()

data, le_dict = preprocess(raw)

# ─────────────────────────────────────────────
# PAGE 1: Overview & EDA
# ─────────────────────────────────────────────
if page == "📊 Overview & EDA":
    st.markdown('<div class="main-header">📊 Employee Attrition Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">IBM HR Analytics · Exploratory Data Analysis</div>', unsafe_allow_html=True)

    total       = len(raw)
    attrition_n = (raw['Attrition'] == 'Yes').sum()
    rate        = round(attrition_n / total * 100, 1)
    staying     = total - attrition_n

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><h2>{total}</h2><p>Total Employees</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card metric-card-green"><h2>{staying}</h2><p>Retained</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card metric-card-red"><h2>{attrition_n}</h2><p>Left</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card metric-card-orange"><h2>{rate}%</h2><p>Attrition Rate</p></div>', unsafe_allow_html=True)

    st.markdown("")

    # Row 1: Attrition pie + Department bar
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Attrition Distribution</div>', unsafe_allow_html=True)
        fig = px.pie(
            values=[staying, attrition_n],
            names=['Stayed', 'Left'],
            color_discrete_sequence=['#11998e','#eb3349'],
            hole=0.45
        )
        fig.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Department-wise Attrition Rate</div>', unsafe_allow_html=True)
        dept_attr = {}
        for dept in raw['Department'].unique():
            sub = raw[raw['Department'] == dept]
            dept_attr[dept] = round(len(sub[sub['Attrition']=='Yes']) / len(sub) * 100, 1)
        fig2 = px.bar(
            x=list(dept_attr.keys()), y=list(dept_attr.values()),
            labels={'x':'Department','y':'Attrition Rate (%)'},
            color=list(dept_attr.values()),
            color_continuous_scale='RdYlGn_r',
            text_auto=True
        )
        fig2.update_layout(margin=dict(t=10,b=10,l=10,r=10), height=300, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2: Gender + Salary
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-title">Gender-wise Attrition Rate</div>', unsafe_allow_html=True)
        gender_attr = {}
        for g in raw['Gender'].unique():
            sub = raw[raw['Gender'] == g]
            gender_attr[g] = round(len(sub[sub['Attrition']=='Yes']) / len(sub) * 100, 1)
        fig3 = px.bar(
            x=list(gender_attr.keys()), y=list(gender_attr.values()),
            labels={'x':'Gender','y':'Attrition Rate (%)'},
            color=list(gender_attr.keys()),
            color_discrete_sequence=['#667eea','#764ba2'],
            text_auto=True
        )
        fig3.update_layout(margin=dict(t=10,b=10), height=300, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-title">Salary Group vs Attrition</div>', unsafe_allow_html=True)
        tmp = raw.copy()
        tmp['salary_grp'] = pd.qcut(tmp['MonthlyIncome'], q=4, labels=['Low','Medium','High','Very High'])
        sal_attr = {}
        for s in ['Low','Medium','High','Very High']:
            sub = tmp[tmp['salary_grp'] == s]
            sal_attr[s] = round(len(sub[sub['Attrition']=='Yes']) / len(sub) * 100, 1)
        fig4 = px.bar(
            x=list(sal_attr.keys()), y=list(sal_attr.values()),
            labels={'x':'Salary Group','y':'Attrition Rate (%)'},
            color=list(sal_attr.values()),
            color_continuous_scale='RdYlGn_r',
            text_auto=True
        )
        fig4.update_layout(margin=dict(t=10,b=10), height=300, coloraxis_showscale=False)
        st.plotly_chart(fig4, use_container_width=True)

    # Row 3: Experience + Correlation
    col5, col6 = st.columns(2)
    with col5:
        st.markdown('<div class="section-title">Experience Group vs Attrition</div>', unsafe_allow_html=True)
        tmp2 = raw.copy()
        tmp2['Exp_Group'] = pd.cut(tmp2['TotalWorkingYears'], bins=[0,5,10,20,40], labels=['0-5','6-10','11-20','20+'])
        exp_attr = {}
        for e in ['0-5','6-10','11-20','20+']:
            sub = tmp2[tmp2['Exp_Group'] == e]
            if len(sub) > 0:
                exp_attr[e] = round(len(sub[sub['Attrition']=='Yes']) / len(sub) * 100, 1)
        fig5 = px.line(
            x=list(exp_attr.keys()), y=list(exp_attr.values()),
            markers=True,
            labels={'x':'Experience Group','y':'Attrition Rate (%)'}
        )
        fig5.update_traces(line_color='#667eea', line_width=3, marker_size=10)
        fig5.update_layout(margin=dict(t=10,b=10), height=300)
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        st.markdown('<div class="section-title">Top Correlated Features with Attrition</div>', unsafe_allow_html=True)
        corr = data.corr(numeric_only=True)['Attrition'].drop('Attrition').sort_values()
        top = pd.concat([corr.head(5), corr.tail(5)])
        colors = ['#eb3349' if v > 0 else '#11998e' for v in top.values]
        fig6 = go.Figure(go.Bar(
            x=top.values, y=top.index, orientation='h',
            marker_color=colors
        ))
        fig6.update_layout(margin=dict(t=10,b=10), height=300, xaxis_title="Correlation")
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown('<div class="section-title">Raw Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(raw.head(20), use_container_width=True, height=280)

# ─────────────────────────────────────────────
# PAGE 2: Model Comparison
# ─────────────────────────────────────────────
elif page == "🤖 Model Comparison":
    st.markdown('<div class="main-header">🤖 ML Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Training 6 models with SMOTE balancing · StandardScaler preprocessing</div>', unsafe_allow_html=True)

    with st.spinner("Training all models… this may take a moment ⚙️"):
        trained, results, scaler, X_train, X_test, y_train, y_test, feat_cols = train_models(data)

    # Summary table
    df_res = pd.DataFrame({
        "Model":     list(results.keys()),
        "Accuracy (%)":  [v['accuracy']  for v in results.values()],
        "Precision (%)": [v['precision'] for v in results.values()],
    }).sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="section-title">Model Leaderboard</div>', unsafe_allow_html=True)
        st.dataframe(df_res.style.highlight_max(subset=["Accuracy (%)","Precision (%)"], color="#d1fae5"), use_container_width=True, height=260)

    with col2:
        st.markdown('<div class="section-title">Accuracy Comparison</div>', unsafe_allow_html=True)
        fig = px.bar(
            df_res, x="Model", y="Accuracy (%)",
            color="Accuracy (%)", color_continuous_scale="Blues",
            text_auto=True
        )
        fig.update_layout(margin=dict(t=10,b=10), height=260, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # Per-model detail
    st.markdown('<div class="section-title">Model Deep Dive</div>', unsafe_allow_html=True)
    sel_model = st.selectbox("Select a model to inspect", list(results.keys()))
    r = results[sel_model]

    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy",  f"{r['accuracy']}%")
    m2.metric("Precision", f"{r['precision']}%")
    m3.metric("F1 (Leave)", f"{round(r['report']['1']['f1-score']*100,1)}%")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Confusion Matrix**")
        cm = r['cm']
        fig_cm = px.imshow(
            cm, text_auto=True,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Stay','Leave'], y=['Stay','Leave'],
            color_continuous_scale='Blues'
        )
        fig_cm.update_layout(margin=dict(t=10,b=10), height=300)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col4:
        st.markdown("**Classification Report**")
        rep = r['report']
        rep_df = pd.DataFrame({
            "Class": ['Stay (0)', 'Leave (1)', 'Macro Avg'],
            "Precision": [
                round(rep['0']['precision']*100,1),
                round(rep['1']['precision']*100,1),
                round(rep['macro avg']['precision']*100,1),
            ],
            "Recall": [
                round(rep['0']['recall']*100,1),
                round(rep['1']['recall']*100,1),
                round(rep['macro avg']['recall']*100,1),
            ],
            "F1-Score": [
                round(rep['0']['f1-score']*100,1),
                round(rep['1']['f1-score']*100,1),
                round(rep['macro avg']['f1-score']*100,1),
            ],
        })
        st.dataframe(rep_df, use_container_width=True, height=160)

    # Feature importances (RF / DT)
    if sel_model in ("Random Forest", "Decision Tree"):
        st.markdown('<div class="section-title">Feature Importances</div>', unsafe_allow_html=True)
        importances = trained[sel_model].feature_importances_
        fi_df = pd.DataFrame({"Feature": feat_cols, "Importance": importances}).sort_values("Importance", ascending=False).head(15)
        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation='h',
                        color="Importance", color_continuous_scale="Purples")
        fig_fi.update_layout(margin=dict(t=10,b=10), height=400, coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 3: Predict Employee
# ─────────────────────────────────────────────
elif page == "🔮 Predict Employee":
    st.markdown('<div class="main-header">🔮 Predict Employee Attrition</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Fill in employee details to get an attrition risk assessment</div>', unsafe_allow_html=True)

    with st.spinner("Training model..."):
        trained, results, scaler, X_train, X_test, y_train, y_test, feat_cols = train_models(data)

    best_model_name = max(results, key=lambda k: results[k]['accuracy'])
    model = trained[best_model_name]

    st.info(f"Using **{best_model_name}** (best accuracy: {results[best_model_name]['accuracy']}%)")

    st.markdown('<div class="section-title">Employee Information</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        age             = st.slider("Age", 18, 60, 35)
        monthly_income  = st.number_input("Monthly Income ($)", 1000, 20000, 5000, 500)
        job_satisfaction= st.select_slider("Job Satisfaction", [1,2,3,4], value=3)
        overtime        = st.radio("OverTime", ["Yes","No"], horizontal=True)
        distance        = st.slider("Distance From Home (km)", 1, 30, 5)

    with col2:
        dept            = st.selectbox("Department", raw['Department'].unique())
        job_role        = st.selectbox("Job Role", raw['JobRole'].unique())
        marital         = st.selectbox("Marital Status", raw['MaritalStatus'].unique())
        gender          = st.radio("Gender", ["Male","Female"], horizontal=True)
        education       = st.select_slider("Education Level", [1,2,3,4,5], value=3)

    with col3:
        total_yrs       = st.slider("Total Working Years", 0, 40, 10)
        yrs_company     = st.slider("Years at Company", 0, 40, 5)
        yrs_role        = st.slider("Years in Current Role", 0, 20, 3)
        yrs_manager     = st.slider("Years With Current Manager", 0, 20, 3)
        stock_opt       = st.select_slider("Stock Option Level", [0,1,2,3], value=1)

    with col1:
        env_sat         = st.select_slider("Environment Satisfaction", [1,2,3,4], value=3)
        job_involve     = st.select_slider("Job Involvement", [1,2,3,4], value=3)
        job_level       = st.select_slider("Job Level", [1,2,3,4,5], value=2)
        wlb             = st.select_slider("Work Life Balance", [1,2,3,4], value=3)

    if st.button("🔮 Predict Attrition Risk", type="primary", use_container_width=True):
        # Build a row matching training columns
        sample_raw = raw.iloc[0:1].copy()
        # override known fields
        sample_raw['Age']                    = age
        sample_raw['MonthlyIncome']          = monthly_income
        sample_raw['JobSatisfaction']        = job_satisfaction
        sample_raw['OverTime']               = overtime
        sample_raw['DistanceFromHome']       = distance
        sample_raw['Department']             = dept
        sample_raw['JobRole']                = job_role
        sample_raw['MaritalStatus']          = marital
        sample_raw['Gender']                 = gender
        sample_raw['Education']              = education
        sample_raw['TotalWorkingYears']      = total_yrs
        sample_raw['YearsAtCompany']         = yrs_company
        sample_raw['YearsInCurrentRole']     = yrs_role
        sample_raw['YearsWithCurrManager']   = yrs_manager
        sample_raw['StockOptionLevel']       = stock_opt
        sample_raw['EnvironmentSatisfaction']= env_sat
        sample_raw['JobInvolvement']         = job_involve
        sample_raw['JobLevel']               = job_level
        sample_raw['WorkLifeBalance']        = wlb

        sample_enc = sample_raw.copy()
        for col in sample_enc.select_dtypes(include='object').columns:
            if col in le_dict:
                try:
                    sample_enc[col] = le_dict[col].transform(sample_enc[col])
                except Exception:
                    sample_enc[col] = 0
            else:
                sample_enc[col] = 0

        drop_cols = [c for c in ['EmployeeCount','StandardHours','Over18','EmployeeNumber','Attrition'] if c in sample_enc.columns]
        sample_enc.drop(columns=drop_cols, inplace=True)

        # align columns
        for c in feat_cols:
            if c not in sample_enc.columns:
                sample_enc[c] = 0
        sample_enc = sample_enc[feat_cols]

        scaled = scaler.transform(sample_enc)
        pred   = model.predict(scaled)[0]
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(scaled)[0][1]
        else:
            prob = 0.8 if pred == 1 else 0.2

        st.divider()
        r1, r2, r3 = st.columns([1,2,1])
        with r2:
            if prob >= 0.6:
                risk_label = "🔴 HIGH RISK"
                risk_class = "risk-high"
                msg = "This employee has a **high likelihood of leaving**. Immediate HR intervention recommended."
            elif prob >= 0.35:
                risk_label = "🟡 MEDIUM RISK"
                risk_class = "risk-medium"
                msg = "This employee shows **moderate attrition signals**. Consider a retention conversation."
            else:
                risk_label = "🟢 LOW RISK"
                risk_class = "risk-low"
                msg = "This employee appears **likely to stay**. Keep up engagement efforts."

            st.markdown(f"### Prediction Result")
            st.markdown(f'<span class="{risk_class}">{risk_label}</span>', unsafe_allow_html=True)
            st.markdown("")
            st.progress(float(prob), text=f"Attrition Probability: **{round(prob*100,1)}%**")
            st.markdown(msg)

# ─────────────────────────────────────────────
# PAGE 4: Employee Segmentation
# ─────────────────────────────────────────────
elif page == "🗺️ Employee Segmentation":
    st.markdown('<div class="main-header">🗺️ Employee Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">K-Means clustering with PCA visualization</div>', unsafe_allow_html=True)

    features = ['Age','MonthlyIncome','TotalWorkingYears','YearsAtCompany',
                'JobSatisfaction','JobInvolvement','WorkLifeBalance',
                'EnvironmentSatisfaction','DistanceFromHome']

    available = [f for f in features if f in data.columns]
    X_seg = data[available]

    scaler_seg = StandardScaler()
    X_sc = scaler_seg.fit_transform(X_seg)

    pca2 = PCA(n_components=2)
    X_pca = pca2.fit_transform(X_sc)

    col_k, col_blank = st.columns([1,3])
    with col_k:
        k = st.slider("Number of Clusters (K)", 2, 6, 3)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pca)

    cluster_labels = {i: f"Cluster {i+1}" for i in range(k)}

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title">Employee Clusters (PCA 2D)</div>', unsafe_allow_html=True)
        fig = px.scatter(
            x=X_pca[:,0], y=X_pca[:,1],
            color=clusters.astype(str),
            labels={'x':'Principal Component 1','y':'Principal Component 2','color':'Cluster'},
            title=""
        )
        centers = kmeans.cluster_centers_
        fig.add_scatter(
            x=centers[:,0], y=centers[:,1],
            mode='markers',
            marker=dict(symbol='x', size=18, color='black', line_width=2),
            name='Centroids'
        )
        fig.update_layout(margin=dict(t=10,b=10), height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Cluster Size Distribution</div>', unsafe_allow_html=True)
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        fig2 = px.pie(
            values=cluster_counts.values,
            names=[f"Cluster {i+1}" for i in cluster_counts.index],
            hole=0.4
        )
        fig2.update_layout(margin=dict(t=10,b=10), height=400)
        st.plotly_chart(fig2, use_container_width=True)

    # Cluster profile heatmap
    st.markdown('<div class="section-title">Cluster Profile Heatmap</div>', unsafe_allow_html=True)
    cluster_df = data[available].copy()
    cluster_df['Cluster'] = clusters
    profile = cluster_df.groupby('Cluster')[available].mean()
    profile.index = [f"Cluster {i+1}" for i in profile.index]

    fig3 = px.imshow(
        profile.T,
        text_auto=".1f",
        color_continuous_scale='RdBu_r',
        labels=dict(x="Cluster", y="Feature", color="Mean Value")
    )
    fig3.update_layout(margin=dict(t=10,b=10), height=420)
    st.plotly_chart(fig3, use_container_width=True)

    # Attrition per cluster
    st.markdown('<div class="section-title">Attrition Rate per Cluster</div>', unsafe_allow_html=True)
    cluster_df['Attrition'] = data['Attrition'].values
    attr_rate = cluster_df.groupby('Cluster')['Attrition'].mean().reset_index()
    attr_rate.columns = ['Cluster','Attrition Rate']
    attr_rate['Cluster'] = attr_rate['Cluster'].apply(lambda x: f"Cluster {x+1}")
    attr_rate['Attrition Rate (%)'] = (attr_rate['Attrition Rate'] * 100).round(1)

    fig4 = px.bar(
        attr_rate, x='Cluster', y='Attrition Rate (%)',
        color='Attrition Rate (%)', color_continuous_scale='RdYlGn_r',
        text_auto=True
    )
    fig4.update_layout(margin=dict(t=10,b=10), height=300, coloraxis_showscale=False)
    st.plotly_chart(fig4, use_container_width=True)

    # Elbow curve
    with st.expander("📈 Elbow Method (optimal K)"):
        wcss = []
        for ki in range(1, 11):
            km = KMeans(n_clusters=ki, random_state=42, n_init=10)
            km.fit(X_pca)
            wcss.append(km.inertia_)
        fig_e = px.line(x=list(range(1,11)), y=wcss, markers=True,
                        labels={'x':'Number of Clusters','y':'WCSS'}, title='Elbow Method')
        fig_e.update_traces(line_color='#667eea', line_width=2, marker_size=8)
        st.plotly_chart(fig_e, use_container_width=True)
