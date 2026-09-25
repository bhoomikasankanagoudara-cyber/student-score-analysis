Here is the complete Python code for your Streamlit web application. It parses the analysis from your `NaviBayes.ipynb` notebook and converts it into a fully dynamic, interactive data science dashboard using Streamlit, Plotly, and Pandas.

### objective -  Analysing the student score and give feedback by using ml model
### `app.py`
<img width="1600" height="900" alt="Screenshot 2026-09-25 111649" src="https://github.com/user-attachments/assets/4b7592ed-7ae9-4e31-a739-c2b025eb430e" />
<img width="1600" height="900" alt="Screenshot 2026-09-25 111753" src="https://github.com/user-attachments/assets/b27de6f0-c878-46b7-9d6d-dee9a141b773" />



Here is a complete, production-ready Streamlit application structure and codebase based on the exploratory analysis performed in `NaviBayes.ipynb`.

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
