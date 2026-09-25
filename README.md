Here is the complete Python code for your Streamlit web application. It parses the analysis from your `NaviBayes.ipynb` notebook and converts it into a fully dynamic, interactive data science dashboard using Streamlit, Plotly, and Pandas.

### objective -  Analysing the student score and give feedback by using ml model
### `app.py`
<img width="1600" height="900" alt="Screenshot 2026-09-25 111649" src="https://github.com/user-attachments/assets/4b7592ed-7ae9-4e31-a739-c2b025eb430e" />
<img width="1600" height="900" alt="Screenshot 2026-09-25 111753" src="https://github.com/user-attachments/assets/b27de6f0-c878-46b7-9d6d-dee9a141b773" />



Here is a complete, production-ready Streamlit application structure and codebase based on the exploratory analysis performed in `NaviBayes.ipynb`.
Below, you'll find a complete, production-ready project setup for turning the **Student Performance Analysis** notebook into an interactive, dynamic **Streamlit** dashboard.

---

### 1. Project Objectives & Structure

#### Objectives

1. **Interactive Data Exploration**: Allow users to inspect raw data, filter by scores/join years, and view summary statistics.
2. **Correlation & Trend Analysis**: Dynamically visualize relationship heatmaps and feature distributions using Seaborn and Matplotlib.
3. **Student Performance Classification**: Implement a Naive Bayes classifier (Gaussian / Categorical) to predict placement readiness or performance tiers based on Math, Reading, and Writing scores.
4. **Interactive What-If Scenarios**: Enable users to input custom test scores via sidebar controls and receive real-time predictions with class probabilities.

#### Project Directory Structure

```text
NaviBayes-Streamlit/
│
├── .streamlit/
│   └── config.toml          # Custom theme and configuration
├── data/
│   └── Students_Performance.csv # Dataset file
├── app.py                   # Main Streamlit Application
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation

```

---

### 2. `requirements.txt`

```text
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.2.0

```

---

### 3. Streamlit Application (`app.py`)

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="NaviBayes | Student Performance Analyzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Data Loading & Caching
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    # Attempt to load dataset; fallback to generated mock data if missing
    try:
        df = pd.read_csv("data/Students_Performance.csv")
    except FileNotFoundError:
        try:
            df = pd.read_csv("Students Performance.csv")
        except FileNotFoundError:
            # Synthetic generation matching notebook structure for demo resilience
            np.random.seed(42)
            n_samples = 399
            df = pd.DataFrame({
                'Math_Score': np.random.randint(60, 101, n_samples),
                'Reading_Score': np.random.randint(60, 101, n_samples),
                'Writing_Score': np.random.randint(60, 101, n_samples),
                'Placement_Score': np.random.randint(60, 101, n_samples),
                'Club_Join_Date': np.random.choice([2018, 2019, 2020, 2021], n_samples)
            })
    return df

df_raw = load_data()

# Create a target label for Naive Bayes Classification based on Placement Score
# Threshold: >= 80 -> High Placement Readiness (1), else Standard (0)
df = df_raw.copy()
df['Placement_Class'] = (df['Placement_Score'] >= 80).astype(int)

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
st.sidebar.image("https://img.icons8.com/illustrations/200/education.png", width=120)
st.sidebar.title("NaviBayes Control")

navigation = st.sidebar.radio(
    "Select Module:",
    ["Overview & Dataset", "Exploratory Analytics", "Naive Bayes Classifier", "Interactive Prediction"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filter Data")
selected_years = st.sidebar.multiselect(
    "Club Join Years",
    options=sorted(df['Club_Join_Date'].unique()),
    default=sorted(df['Club_Join_Date'].unique())
)

filtered_df = df[df['Club_Join_Date'].isin(selected_years)]

# -----------------------------------------------------------------------------
# Module 1: Overview & Dataset
# -----------------------------------------------------------------------------
if navigation == "Overview & Dataset":
    st.markdown('<div class="main-title">🎓 Student Score Analysis & NaviBayes</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Exploration of academic performance metrics and probabilistic classification using Naive Bayes.</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Students", len(filtered_df))
    col2.metric("Avg Math Score", f"{filtered_df['Math_Score'].mean():.1f}")
    col3.metric("Avg Reading Score", f"{filtered_df['Reading_Score'].mean():.1f}")
    col4.metric("Avg Writing Score", f"{filtered_df['Writing_Score'].mean():.1f}")

    st.markdown("---")
    st.subheader("Filtered Dataset Preview")
    st.dataframe(filtered_df, use_container_width=True)

    with st.expander("View Descriptive Statistics"):
        st.write(filtered_df.describe())

# -----------------------------------------------------------------------------
# Module 2: Exploratory Analytics
# -----------------------------------------------------------------------------
elif navigation == "Exploratory Analytics":
    st.markdown('<div class="main-title">📊 Exploratory Analytics</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Correlation Heatmap")
        corr_cols = ['Math_Score', 'Reading_Score', 'Writing_Score', 'Placement_Score']
        corr_matrix = filtered_df[corr_cols].corr()

        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".3f", ax=ax, cbar=True, vmin=-0.2, vmax=1.0)
        ax.set_title("Correlation Heatmap of Scores")
        st.pyplot(fig)

    with col2:
        st.subheader("Score Distributions")
        selected_metric = st.selectbox("Select Score to Plot Distribution", corr_cols)
        
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sns.histplot(filtered_df[selected_metric], kde=True, color="#3B82F6", bins=15, ax=ax)
        ax.set_title(f"Distribution of {selected_metric}")
        ax.set_xlabel("Score")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Pairwise Relationships")
    fig_pair = sns.pairplot(filtered_df[corr_cols], diag_kind='kde', corner=True)
    st.pyplot(fig_pair)

# -----------------------------------------------------------------------------
# Module 3: Naive Bayes Classifier
# -----------------------------------------------------------------------------
elif navigation == "Naive Bayes Classifier":
    st.markdown('<div class="main-title">🤖 Naive Bayes Model Performance</div>', unsafe_allow_html=True)
    st.write("Using Gaussian Naive Bayes to classify high placement performance (`Placement_Score >= 80`).")

    # Feature selection
    features = ['Math_Score', 'Reading_Score', 'Writing_Score']
    X = filtered_df[features]
    y = filtered_df['Placement_Class']

    if len(np.unique(y)) < 2:
        st.warning("Insufficient class diversity in filtered data. Please expand filters in the sidebar.")
    else:
        test_size = st.slider("Test Set Split Ratio", 0.1, 0.4, 0.2, 0.05)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### Model Metrics")
            st.metric("Model Accuracy", f"{acc * 100:.2f}%")
            
            st.markdown("**Parameters Used:**")
            st.json({"Algorithm": "Gaussian Naive Bayes", "Features": features, "Target": "Placement_Class (>=80)"})

        with col2:
            st.markdown("### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                        xticklabels=["Standard", "High Readiness"],
                        yticklabels=["Standard", "High Readiness"], ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        st.markdown("---")
        st.markdown("### Classification Report")
        report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
        st.dataframe(report_df.style.highlight_max(axis=0))

# -----------------------------------------------------------------------------
# Module 4: Interactive Prediction
# -----------------------------------------------------------------------------
elif navigation == "Interactive Prediction":
    st.markdown('<div class="main-title">🔮 Real-Time Student Predictor</div>', unsafe_allow_html=True)
    st.write("Adjust academic scores to predict placement classification using Naive Bayes.")

    features = ['Math_Score', 'Reading_Score', 'Writing_Score']
    X = df[features]
    y = df['Placement_Class']
    
    model = GaussianNB()
    model.fit(X, y)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Academic Scores")
        math_in = st.slider("Math Score", 0, 100, 75)
        reading_in = st.slider("Reading Score", 0, 100, 75)
        writing_in = st.slider("Writing Score", 0, 100, 75)

        input_data = pd.DataFrame([[math_in, reading_in, writing_in]], columns=features)

    with col2:
        st.subheader("Prediction Result")
        prediction = model.predict(input_data)[0]
        probs = model.predict_proba(input_data)[0]

        if prediction == 1:
            st.success(f"**High Placement Probability**: Likely Ready ({probs[1]*100:.1f}% confidence)")
        else:
            st.info(f"**Standard Placement Probability**: Needs Improvement ({probs[0]*100:.1f}% confidence)")

        # Display Probability Breakdown
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.barh(["Standard", "High Readiness"], probs, color=["#9CA3AF", "#2563EB"])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        st.pyplot(fig)

```

---

### 4. `README.md`

```markdown
# NaviBayes: Student Performance Dashboard

An interactive Streamlit application for exploratory data analysis and Naive Bayes performance classification on student academic scores.

## Features
- **Exploratory Data Analysis**: Dynamic score correlation heatmaps, score distributions, and interactive dataset filtering.
- **Naive Bayes Machine Learning Model**: Trains a Gaussian Naive Bayes classifier to identify placement readiness based on Math, Reading, and Writing performance.
- **Interactive Predictor**: Real-time evaluation of hypothetical student score profiles.

## Installation & Setup

1. **Clone or Download the Repository**:
   ```bash
   git clone [https://github.com/your-username/NaviBayes.git](https://github.com/your-username/NaviBayes.git)
   cd NaviBayes

```

2. **Create and Activate a Virtual Environment** (Optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```


3. **Install Dependencies**:
```bash
pip install -r requirements.txt

```


4. **Add Dataset**:
Ensure `Students Performance.csv` is placed inside the `data/` directory or root directory.
5. **Run the Streamlit Application**:
```bash
streamlit run app.py

```



```

```
---

### Project Structure

```text
student_score_app/
│
├── .streamlit/
│   └── config.toml          # Custom theme configuration
├── data/
│   └── Students Performance.csv  # Dataset used for analysis
├── app.py                   # Main Streamlit application
├── requirements.txt         # Dependency management
└── README.md                # Documentation and setup instructions

```

---

### `requirements.txt`

```text
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

```

---

### `app.py`

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff

# Page Configuration
st.set_page_config(
    page_title="Student Score Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        padding: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and cache the student performance dataset."""
    try:
        df = pd.read_csv("data/Students Performance.csv")
        return df
    except FileNotFoundError:
        # Fallback dataset generation for demonstration
        np.random.seed(42)
        n = 399
        df = pd.DataFrame({
            'Math_Score': np.random.randint(60, 101, n),
            'Reading_Score': np.random.randint(60, 101, n),
            'Writing_Score': np.random.randint(60, 101, n),
            'Placement_Score': np.random.randint(60, 101, n),
            'Club_Join_Date': np.random.choice([2018, 2019, 2020, 2021], n)
        })
        return df

df = load_data()

# Sidebar Filters
st.sidebar.title("📌 Filters & Settings")
selected_years = st.sidebar.multiselect(
    "Filter by Club Join Date:",
    options=sorted(df['Club_Join_Date'].unique()),
    default=sorted(df['Club_Join_Date'].unique())
)

filtered_df = df[df['Club_Join_Date'].isin(selected_years)]

score_cols = ['Math_Score', 'Reading_Score', 'Writing_Score', 'Placement_Score']

# Navigation Tabs
tab_overview, tab_eda, tab_corr, tab_predict = st.tabs([
    "📊 Overview", 
    "📈 Exploratory Analysis", 
    "🔥 Correlation Analysis", 
    "🎯 Score Assessment"
])

# ---------------------------------------------------------
# TAB 1: OVERVIEW
# ---------------------------------------------------------
with tab_overview:
    st.title("🎓 Student Performance Analytics Dashboard")
    st.caption("Interactive analysis of student test results and placement scores.")

    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Math Score", f"{filtered_df['Math_Score'].mean():.1f}")
    with col2:
        st.metric("Avg Reading Score", f"{filtered_df['Reading_Score'].mean():.1f}")
    with col3:
        st.metric("Avg Writing Score", f"{filtered_df['Writing_Score'].mean():.1f}")
    with col4:
        st.metric("Avg Placement Score", f"{filtered_df['Placement_Score'].mean():.1f}")

    st.markdown("---")
    
    st.subheader("Raw Dataset Preview")
    st.dataframe(filtered_df, use_container_width=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Dataset Summary Statistics")
        st.dataframe(filtered_df[score_cols].describe().T, use_container_width=True)
    with col_b:
        st.subheader("Distribution by Club Join Date")
        year_counts = filtered_df['Club_Join_Date'].value_counts().reset_index()
        year_counts.columns = ['Year', 'Count']
        fig = px.bar(year_counts, x='Year', y='Count', color='Year', title="Students per Joining Year")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: EXPLORATORY ANALYSIS
# ---------------------------------------------------------
with tab_eda:
    st.header("📈 Score Distribution & Comparison")
    
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        selected_metric = st.selectbox("Select Metric to Visualize:", score_cols)
        chart_type = st.radio("Chart Type:", ["Histogram", "Box Plot", "Violin Plot"])
    
    with col_right:
        if chart_type == "Histogram":
            fig = px.histogram(filtered_df, x=selected_metric, nbins=20, marginal="rug",
                               title=f"Distribution of {selected_metric}")
        elif chart_type == "Box Plot":
            fig = px.box(filtered_df, x='Club_Join_Date', y=selected_metric, color='Club_Join_Date',
                         title=f"{selected_metric} by Club Join Date")
        else:
            fig = px.violin(filtered_df, y=selected_metric, x='Club_Join_Date', color='Club_Join_Date',
                            box=True, points="all", title=f"{selected_metric} Distribution")
        
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Bivariate Relationships")
    x_axis = st.selectbox("Select X Axis:", score_cols, index=0)
    y_axis = st.selectbox("Select Y Axis:", score_cols, index=1)
    
    fig_scatter = px.scatter(
        filtered_df, x=x_axis, y=y_axis, 
        color='Club_Join_Date', hover_data=score_cols,
        trendline="ols", title=f"{x_axis} vs {y_axis}"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: CORRELATION ANALYSIS
# ---------------------------------------------------------
with tab_corr:
    st.header("🔥 Score Correlation Matrix")
    st.write("Examine linear relationships between performance metrics.")

    corr_matrix = filtered_df[score_cols].corr()

    col_c1, col_c2 = st.columns([2, 1])
    
    with col_c1:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".4f", ax=ax)
        plt.title('Correlation Heatmap of Scores')
        st.pyplot(fig)
        
    with col_c2:
        st.subheader("Correlation Values")
        st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=None))

# ---------------------------------------------------------
# TAB 4: SCORE ASSESSMENT & PREDICTION
# ---------------------------------------------------------
with tab_predict:
    st.header("🎯 Student Qualification Evaluator")
    st.write("Simulate student score combinations to estimate overall eligibility.")

    with st.form("score_form"):
        c1, c2 = st.columns(2)
        with c1:
            m_score = st.slider("Math Score", 0, 100, 75)
            r_score = st.slider("Reading Score", 0, 100, 75)
        with c2:
            w_score = st.slider("Writing Score", 0, 100, 75)
            p_score = st.slider("Placement Score Threshold", 0, 100, 70)
        
        submit = st.form_submit_button("Evaluate Performance")

    if submit:
        overall_avg = (m_score + r_score + w_score) / 3
        st.subheader("Results")
        
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric("Overall Academic Average", f"{overall_avg:.2f}")
        with res_col2:
            if overall_avg >= p_score:
                st.success(" Status: Qualified / On Track")
            else:
                st.warning(" Status: Academic Support Recommended")

```

---

### `README.md`

```markdown
# Student Performance Analytics App

An interactive Streamlit web application designed to analyze student test performance, placement outcomes, and metric correlations based on exploratory notebook findings.

## Features
- **Overview Dashboard:** Instant metrics summary and raw dataset exploration.
- **Exploratory Visualizations:** Dynamic histograms, scatter plots, and box plots powered by Plotly.
- **Correlation Matrix:** Seaborn-rendered correlation heatmaps for feature evaluation.
- **Score Evaluator:** Interactive interface to model overall academic thresholds.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-repo/student-performance-app.git](https://github.com/your-repo/student-performance-app.git)
   cd student-performance-app

```

2. **Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```


4. **Add Data:**
Place your `Students Performance.csv` file inside the `data/` directory.
5. **Run the Streamlit app:**
```bash
streamlit run app.py

```



```

```

### Prerequisites & Setup

To run the app locally, install the required packages and execute the run command:

```bash
pip install streamlit pandas numpy plotly matplotlib seaborn
streamlit run app.py
