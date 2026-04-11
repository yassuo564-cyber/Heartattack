       gap: 8px;
   }
   .stTabs [data-baseweb="tab"] {
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: 500;
    }
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        padding: 10px 20px;
        border-radius: 5px;
   }
    .header-title {
    .stTabs [aria-selected="true"] {
        background-color: #2196F3;
       color: white;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }
    .header-sub {
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        margin-top: 0.3rem;
    }
    .card {
        background-color: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
   }
    .card-title {
        color: #667eea;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .card-text {
        color: #cccccc;
        font-size: 0.95rem;
        line-height: 1.6;
    .stButton > button {
        background-color: #2196F3;
        color: white;
        border: none;
        padding: 12px;
        font-size: 16px;
        border-radius: 8px;
   }
    .result-danger {
        background: linear-gradient(135deg, #2d1117 0%, #3d1a1a 100%);
        border: 2px solid #ff6b6b;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    .stButton > button:hover {
        background-color: #1976D2;
   }
    .result-safe {
        background: linear-gradient(135deg, #0d1f0d 0%, #1a3d1a 100%);
        border: 2px solid #51cf66;
        border-radius: 12px;
        padding: 1.5rem;
    div[data-testid="stMetric"] {
        background-color: #1E1E2E;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 15px;
       text-align: center;
   }
    .member-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #2a2a4a 100%);
        border-left: 4px solid #667eea;
        padding: 1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
    .block-container {
        padding-top: 2rem;
   }
    .member-header-knn {
        background: linear-gradient(135deg, #1a1a2e 0%, #2a2a4a 100%);
        border-left: 4px solid #FF9800;
        padding: 1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
    h1 {
        color: #2196F3;
   }
</style>
""", unsafe_allow_html=True)
@@ -89,17 +53,12 @@
encoder = load('encoder.joblib')
scaler = load('scaler.joblib')

# Header
st.markdown("""
<div class="header-container">
    <p class="header-title">Heart Disease Prediction System</p>
    <p class="header-sub">Supervised Machine Learning | ANN & KNN | Cleveland Heart Disease Dataset (UCI)</p>
</div>
""", unsafe_allow_html=True)
st.title("Heart Disease Prediction System")
st.caption("Supervised Machine Learning | ANN & KNN | Cleveland Heart Disease Dataset (UCI)")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Home", "Prediction", "Model Comparison", "Member Work", "Dataset Info"])

# ============ HOME ============
with tab1:
c1, c2, c3, c4 = st.columns(4)
c1.metric("Patients", "303")
@@ -111,54 +70,17 @@

col1, col2 = st.columns(2)
with col1:
        st.markdown("""
        <div class="card">
            <div class="card-title">ANN (Artificial Neural Network)</div>
            <div class="card-text">MLPClassifier from scikit-learn with 100 hidden neurons. The model learns patterns in the data through backpropagation. It adjusts its weights during training to minimise prediction error.</div>
        </div>
        """, unsafe_allow_html=True)
        st.subheader("ANN")
        st.write("Artificial Neural Network using MLPClassifier. Has 100 hidden neurons. Learns through backpropagation.")
with col2:
        st.markdown("""
        <div class="card">
            <div class="card-title">KNN (K-Nearest Neighbours)</div>
            <div class="card-text">KNeighborsClassifier from scikit-learn. It classifies new data by looking at the k closest training data points and assigning the majority class among those neighbours using Euclidean distance.</div>
        </div>
        """, unsafe_allow_html=True)
        st.subheader("KNN")
        st.write("K-Nearest Neighbours using KNeighborsClassifier. Classifies based on majority vote of nearest neighbours.")

st.markdown("---")

    st.subheader("About This Project")
    st.write("This system predicts whether a patient has heart disease based on 13 clinical features. Two supervised machine learning algorithms (ANN and KNN) are trained, evaluated and compared. The models are deployed on this Streamlit web application for easy access.")

    st.markdown("---")

st.subheader("How to Use")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-title">Step 1</div>
            <div class="card-text">Go to the Prediction tab. Select ANN or KNN model. Enter patient clinical data.</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-title">Step 2</div>
            <div class="card-text">Click Run Prediction. The system will predict if the patient has heart disease or not.</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="card">
            <div class="card-title">Step 3</div>
            <div class="card-text">Go to Model Comparison to see how ANN and KNN performed. View charts and confusion matrices.</div>
        </div>
        """, unsafe_allow_html=True)

# ============ PREDICTION ============
    st.write("Click on the **Prediction** tab to test the model. Click **Model Comparison** to view results. Click **Member Work** to see each member's contribution.")

with tab2:
    st.subheader("Select Model and Enter Patient Data")
model_choice = st.selectbox("Model", ["ANN", "KNN"])
st.markdown("---")

@@ -207,22 +129,20 @@
st.markdown("---")
if result[0] == 1:
st.markdown("""
            <div class="result-danger">
                <h2 style="color: #ff6b6b; margin: 0;">Heart Disease Detected</h2>
                <p style="color: #cccccc; margin-top: 0.5rem;">The model predicts that this patient has heart disease. Please consult a doctor for further diagnosis.</p>
                <p style="color: #888888; font-size: 0.85rem;">Model: {}</p>
            <div style="background-color:#2D1117; border:2px solid #FF4B4B; border-radius:10px; padding:20px; text-align:center;">
                <h2 style="color:#FF4B4B;">Heart Disease Detected</h2>
                <p style="color:#AAAAAA;">Please consult a doctor for further diagnosis.</p>
           </div>
            """.format(model_choice), unsafe_allow_html=True)
            """, unsafe_allow_html=True)
else:
st.markdown("""
            <div class="result-safe">
                <h2 style="color: #51cf66; margin: 0;">No Heart Disease</h2>
                <p style="color: #cccccc; margin-top: 0.5rem;">The model predicts that this patient does not have heart disease. Regular checkups are still recommended.</p>
                <p style="color: #888888; font-size: 0.85rem;">Model: {}</p>
            <div style="background-color:#0D1F0D; border:2px solid #2ECC71; border-radius:10px; padding:20px; text-align:center;">
                <h2 style="color:#2ECC71;">No Heart Disease</h2>
                <p style="color:#AAAAAA;">Regular checkups are still recommended.</p>
           </div>
            """.format(model_choice), unsafe_allow_html=True)
            """, unsafe_allow_html=True)
        st.caption(f"Model used: {model_choice}")

# ============ COMPARISON ============
with tab3:
c1, c2, c3, c4 = st.columns(4)
c1.metric("ANN Accuracy", "90.2%")
@@ -246,16 +166,16 @@
fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(4)
width = 0.35
        bars1 = ax.bar(x - width/2, [0.902, 0.933, 0.875, 0.903], width, label='ANN', color='#667eea')
        bars1 = ax.bar(x - width/2, [0.902, 0.933, 0.875, 0.903], width, label='ANN', color='#2196F3')
bars2 = ax.bar(x + width/2, [0.9016, 0.9333, 0.8750, 0.9032], width, label='KNN', color='#FF9800')
ax.set_xticks(x)
        ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1'])
        ax.set_xticklabels(['Acc', 'Prec', 'Recall', 'F1'])
ax.legend()
ax.set_ylim(0.8, 1.0)
for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003, f'{bar.get_height():.3f}', ha='center', fontsize=8)
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003, f'{bar.get_height():.3f}', ha='center', fontsize=9)
for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003, f'{bar.get_height():.3f}', ha='center', fontsize=8)
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003, f'{bar.get_height():.3f}', ha='center', fontsize=9)
st.pyplot(fig)

st.markdown("---")
@@ -282,34 +202,22 @@
st.pyplot(fig2)

st.markdown("---")
    st.subheader("Summary")
    st.write("Both ANN and KNN achieved similar performance with accuracy above 90%. Both models correctly identified 28 out of 32 heart disease cases. 4 cases were missed by both models. Neither model showed a clear advantage over the other on this dataset.")
    st.write("Both models achieved accuracy above 90%. Both correctly identified 28 out of 32 heart disease cases. 4 cases were missed. There is no significant difference between ANN and KNN on this dataset.")

# ============ MEMBER WORK ============
with tab4:
    st.markdown("""
    <div class="member-header">
        <h3 style="color: #667eea; margin: 0;">Ng Soon Siang — ANN (MLPClassifier)</h3>
    </div>
    """, unsafe_allow_html=True)
    st.subheader("Ng Soon Siang - ANN (MLPClassifier)")
    st.markdown("---")

    col1, col2 = st.columns(2)
    col1, col2 = st.columns([1, 2])
with col1:
        st.write("**Algorithm:** Artificial Neural Network")
        st.write("**Library:** MLPClassifier (scikit-learn)")
        st.write("**Preprocessing:** Missing values filled with median")
        st.write("**Model Settings:** 100 hidden neurons, max_iter=500, random_state=42")
        st.write("**Train/Test Split:** 80/20")
        st.write("**Algorithm:** MLPClassifier")
        st.write("**Preprocessing:** Median")
        st.write("**Hidden Layers:** (100,)")
        st.write("**Max Iterations:** 500")
        st.write("**Random State:** 42")
with col2:
        st.write("**Results:**")
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", "90.2%")
        c2.metric("F1 Score", "0.903")
        c1.metric("Precision", "93.3%")
        c2.metric("Recall", "87.5%")

    st.write("**Code Overview:**")
    st.code("""
        st.write("**Code Overview:**")
        st.code("""
# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
df = pd.read_csv(url, names=column_names, na_values='?')
@@ -327,7 +235,14 @@
ann = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
ann.fit(X_train, y_train)
y_pred_ann = ann.predict(X_test)
    """, language="python")
        """, language="python")

    st.write("**Results:**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", "90.2%")
    col2.metric("Precision", "93.3%")
    col3.metric("Recall", "87.5%")
    col4.metric("F1 Score", "0.903")

fig1, ax1 = plt.subplots(figsize=(5, 4))
sns.heatmap(np.array([[27, 2], [4, 28]]), annot=True, fmt='d', cmap='Blues',
@@ -338,31 +253,22 @@
ax1.set_title('ANN Confusion Matrix')
st.pyplot(fig1)

    st.markdown("---")
st.markdown("---")

    st.markdown("""
    <div class="member-header-knn">
        <h3 style="color: #FF9800; margin: 0;">Chia Sheng Yang — KNN (KNeighborsClassifier)</h3>
    </div>
    """, unsafe_allow_html=True)
    st.subheader("Chia Sheng Yang - KNN (KNeighborsClassifier)")
    st.markdown("---")

    col1, col2 = st.columns(2)
    col1, col2 = st.columns([1, 2])
with col1:
        st.write("**Algorithm:** K-Nearest Neighbours")
        st.write("**Library:** KNeighborsClassifier (scikit-learn)")
        st.write("**Algorithm:** KNeighborsClassifier")
st.write("**Preprocessing:** Mode, OneHotEncoder, StandardScaler")
        st.write("**Hyperparameter Tuning:** Tested k=1 to 39")
        st.write("**Train/Test Split:** 80/20")
        st.write("**Tuning:** k=1 to 39")
        st.write("**Cross Validation:** 10-fold")
        st.write("**Random State:** 42")
with col2:
        st.write("**Results:**")
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", "90.16%")
        c2.metric("F1 Score", "0.903")
        c1.metric("Precision", "93.33%")
        c2.metric("Recall", "87.50%")

    st.write("**Code Overview:**")
    st.code("""
        st.write("**Code Overview:**")
        st.code("""
# Load dataset
df = pd.read_csv(file_path, names=columns, na_values='?')

@@ -374,16 +280,21 @@
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
scaler = StandardScaler()

# Find best K
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
# Split data
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Train KNN with best K
# Find best K and train
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
    """, language="python")
        """, language="python")

    st.write("**Results:**")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", "90.16%")
    col2.metric("Precision", "93.33%")
    col3.metric("Recall", "87.50%")
    col4.metric("F1 Score", "0.903")

fig2, ax2 = plt.subplots(figsize=(5, 4))
sns.heatmap(np.array([[27, 2], [4, 28]]), annot=True, fmt='d', cmap='Oranges',
@@ -394,29 +305,22 @@
ax2.set_title('KNN Confusion Matrix')
st.pyplot(fig2)

# ============ DATASET ============
with tab5:
col1, col2 = st.columns(2)
with col1:
st.subheader("Overview")
        st.markdown("""
        <div class="card">
            <div class="card-text">
                <b>Source:</b> UCI Machine Learning Repository<br>
                <b>Author:</b> Detrano et al. (1989)<br>
                <b>Records:</b> 303 patients<br>
                <b>Features:</b> 13 clinical attributes<br>
                <b>Target:</b> Heart disease (0 = No, 1 = Yes)
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.write("**Source:** UCI Machine Learning Repository")
        st.write("**Author:** Detrano et al. (1989)")
        st.write("**Records:** 303 patients")
        st.write("**Features:** 13 clinical attributes")
        st.write("**Target:** Heart disease (0 = No, 1 = Yes)")
with col2:
st.subheader("Target Distribution")
fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(['No Disease', 'Disease'], [138, 165], color=['#667eea', '#FF9800'])
        ax.bar(['No Disease', 'Disease'], [138, 165], color=['#2196F3', '#FF9800'])
ax.set_ylabel('Count')
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2, str(int(bar.get_height())), ha='center', fontsize=14)
        for i, v in enumerate([138, 165]):
            ax.text(i, v + 2, str(v), ha='center', fontsize=14)
st.pyplot(fig)

st.markdown("---")
