import streamlit as st
import pandas as pd
import tempfile
import os
import pypdf

# ------------------------------------------------------------------------------
# GENAI SDK INITIALIZATION
# ------------------------------------------------------------------------------
try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# --- Page Config & Styling ---
st.set_page_config(
    page_title="COEP Tech - M.Tech AIML Syllabus Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2E1A47, #5B32A8, #7A5CFA);
        padding: 28px;
        border-radius: 14px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.15);
    }
    .stat-card {
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(10px);
        padding: 14px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.2);
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. COMPREHENSIVE SEMESTER 1 SYLLABUS DATABASE
# ==============================================================================
SYLLABUS_DB = {
    "PSMC": {
        "title": "Probability, Statistics and Queuing Theory",
        "code": "PSMC",
        "credits": 4,
        "type": "Program Specific Mathematics Course",
        "units": {
            "Unit 1: Basic Probability Theory [2 Hrs]": {
                "topics": ["Probability Axioms", "Conditional Probability", "Independence of Events", "Bayes' Rule", "Bernoulli Trials"],
                "summary": "Establishes fundamental probability laws, conditional chance evaluation, and updating prior probabilities using Bayes' Theorem.",
                "proofs": "Bayes' Theorem: $P(A_i|B) = \\frac{P(B|A_i)P(A_i)}{\\sum_{j} P(B|A_j)P(A_j)}$",
                "assignments": ["Prove Bayes' theorem from conditional probability axioms.", "Analyze 3 sequential Bernoulli trials for success probabilities."],
                "resource_url": "https://nptel.ac.in/courses/111102111"
            },
            "Unit 2: Random Variables and Expectation [10 Hrs]": {
                "topics": ["Discrete Random Variables & PMF", "Binomial, Poisson, Geometric Distributions", "Indicator Variables", "Continuous Random Variables & PDF", "Exponential, Erlang, Gamma, Normal Distributions", "Moments & Transform Methods"],
                "summary": "Covers discrete/continuous distributions, expectation, variance, moment generating functions (MGF), and joint transforms.",
                "proofs": "Poisson Distribution PMF: $P(X=k) = \\frac{\\lambda^k e^{-\\lambda}}{k!}$ | Normal PDF: $f(x) = \\frac{1}{\\sigma \\sqrt{2\\pi}} e^{-\\frac{1}{2}\\left(\\frac{x-\\mu}{\\sigma}\\right)^2}$",
                "assignments": ["Derive MGF of Normal Distribution.", "Compute expectation of Exponential distribution $E[X] = 1/\\lambda$."],
                "resource_url": "https://ocw.mit.edu/courses/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/"
            },
            "Unit 3: Stochastic Processes [6 Hrs]": {
                "topics": ["Classification of Stochastic Processes", "Bernoulli Process", "Poisson Process", "Renewal Processes"],
                "summary": "Studies indexed collections of random variables over time, modeling arrivals, counts, and memoryless properties.",
                "proofs": "Poisson Arrival Probability: $P(N(t) = k) = \\frac{(\\lambda t)^k e^{-\\lambda t}}{k!}$",
                "assignments": ["Prove that inter-arrival times in a Poisson process are exponentially distributed."],
                "resource_url": "https://nptel.ac.in/courses/111102014"
            },
            "Unit 4: Markov Chains [8 Hrs]": {
                "topics": ["Discrete-Time Markov Chains (DTMC)", "n-step Transition Probabilities", "State Classification & Limiting Probabilities", "M/G/1 Queuing System", "Continuous-Time Markov Chains (CTMC)", "Birth-Death Processes (M/M/1, M/M/m)", "Petri Nets"],
                "summary": "Analyzes state-transition systems with Markov property, steady-state distributions, and queue lengths in computer systems.",
                "proofs": "Chapman-Kolmogorov Equation: $P_{ij}^{(n+m)} = \\sum_{k} P_{ik}^{(n)} P_{kj}^{(m)}$ | M/M/1 Average Queue Length: $L_q = \\frac{\\rho^2}{1-\\rho}$",
                "assignments": ["Solve limiting probabilities for a 3-state DTMC.", "Calculate average wait time for M/M/1 queue with $\\lambda=4, \\mu=6$."],
                "resource_url": "https://nptel.ac.in/courses/111102014"
            },
            "Unit 5: Statistical Inference & Parameter Estimation [8 Hrs]": {
                "topics": ["Sampling Distributions (Normal, Exponential)", "Parameter Estimation (MLE, MAP)", "Markov Chain Estimation", "Hypothesis Testing (Null/Alt, p-values, t-test, Chi-Square)"],
                "summary": "Covers point estimation, maximum likelihood, confidence intervals, and hypothesis testing frameworks for data analysis.",
                "proofs": "Log-Likelihood Optimization: $\\hat{\\theta}_{MLE} = \\arg\\max_{\\theta} \\sum_{i=1}^n \\ln f(x_i | \\theta)$",
                "assignments": ["Derive MLE for parameters $\\mu, \\sigma^2$ of Normal Distribution.", "Perform t-test on sample data."],
                "resource_url": "https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2014/"
            },
            "Unit 6: Regression and Analysis of Variance (ANOVA) [6 Hrs]": {
                "topics": ["Least Square Curve Fitting", "Linear and Non-linear Regression", "Analysis of Variance (One-Way & Two-Way ANOVA)"],
                "summary": "Focuses on linear models, residual sum of squares minimization, correlation coefficients, and hypothesis testing across multiple group means.",
                "proofs": "Least Squares Estimator: $\\hat{\\beta}_1 = \\frac{\\sum (x_i - \\bar{x})(y_i - \\bar{y})}{\\sum (x_i - \\bar{x})^2}$",
                "assignments": ["Fit quadratic regression curve to noisy time-series data.", "Construct One-Way ANOVA table for 3 treatment groups."],
                "resource_url": "https://nptel.ac.in/courses/111105042"
            }
        }
    },
    "PSBC": {
        "title": "Algorithms and Complexity Theory",
        "code": "PSBC",
        "credits": 3,
        "type": "Programme Specific Bridge Course",
        "units": {
            "Unit 1: Mathematical Foundation [4 Hrs]": {
                "topics": ["Asymptotic Notation", "Solving Recurrences", "Master Method"],
                "summary": "Establishes formal bounds on algorithmic growth rates and mathematical tools for recurrence relations.",
                "proofs": "Master Theorem: $T(n) = aT(n/b) + f(n) \\implies T(n) = \\Theta(n^{\\log_b a})$",
                "assignments": ["Solve $T(n) = 2T(n/2) + n \\log n$ using Master Theorem."],
                "resource_url": "https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/"
            },
            "Unit 2: Analysis of Algorithms [4 Hrs]": {
                "topics": ["Time and Space Complexity Analysis", "Worst-case Analysis (Heapsort)", "Average-case Analysis (Quicksort)"],
                "summary": "Analyzes worst-case and expected execution bounds for fundamental search and sort algorithms.",
                "proofs": "Quicksort Average Time: $O(n \\log n)$",
                "assignments": ["Trace Heapsort and count operations."],
                "resource_url": "https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/"
            },
            "Unit 3: Standard Design Techniques-I [6 Hrs]": {
                "topics": ["Divide and Conquer Strategy", "Greedy Method", "Fractional Knapsack", "Huffman Coding"],
                "summary": "Explores greedy choice properties, optimal substructure, and divide-and-conquer paradigms.",
                "proofs": "Greedy Choice Property Proof for Huffman Coding.",
                "assignments": ["Implement Huffman Coding."],
                "resource_url": "https://nptel.ac.in/courses/106102011"
            },
            "Unit 4: Standard Design Techniques-II [6 Hrs]": {
                "topics": ["Dynamic Programming", "Network Flow", "Max-Flow Min-Cut"],
                "summary": "Covers overlapping subproblems, memoization, tabular formulation, and maximum flow in networks.",
                "proofs": "Max-Flow Min-Cut Theorem",
                "assignments": ["Solve Matrix Chain Multiplication."],
                "resource_url": "https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/"
            },
            "Unit 5: Standard Design Techniques-III [6 Hrs]": {
                "topics": ["Backtracking", "Branch-and-Bound"],
                "summary": "Explores state-space tree traversal, pruning non-promising nodes.",
                "proofs": "Bounding Function Formulation for TSP.",
                "assignments": ["Solve 8-Queens problem."],
                "resource_url": "https://nptel.ac.in/courses/106102011"
            },
            "Unit 6: Complexity Theory [4 Hrs]": {
                "topics": ["P, NP, NP-Hard, NP-Complete Classes", "Polynomial-Time Reductions", "SAT"],
                "summary": "Classifies computational problems based on tractability.",
                "proofs": "Cook-Levin Theorem",
                "assignments": ["Prove Independent Set is NP-Complete."],
                "resource_url": "https://ocw.mit.edu/courses/6-045j-automata-computability-and-complexity-spring-2011/"
            }
        }
    },
    "PCC_AI": {
        "title": "Artificial Intelligence",
        "code": "PCC",
        "credits": 4,
        "type": "Programme Core Course",
        "units": {
            "Unit 1: Introduction & Intelligent Agents [8 Hrs]": {
                "topics": ["Foundations & History of AI", "Intelligent Agents & Environments", "Concept of Rationality"],
                "summary": "Introduces AI paradigms and rational decision making.",
                "proofs": "Utility Expectation formulation",
                "assignments": ["Define PEAS for Automated Agent."],
                "resource_url": "https://inst.eecs.berkeley.edu/~cs188/su25/"
            },
            "Unit 2: Solving Problem by Searching [8 Hrs]": {
                "topics": ["Uninformed Search", "Informed Heuristic Search (A*)", "Local Search"],
                "summary": "Covers graph traversal strategies, heuristic search optimality conditions.",
                "proofs": "A* Optimality Proof: $f(n) = g(n) + h(n)$",
                "assignments": ["Implement A* search."],
                "resource_url": "https://inst.eecs.berkeley.edu/~cs188/su25/"
            },
            "Unit 3: Game Solver & Constraint Satisfaction Problems [8 Hrs]": {
                "topics": ["Game Theory", "Alpha-Beta Pruning", "Constraint Propagation"],
                "summary": "Analyzes adversarial decision making in games.",
                "proofs": "Alpha-Beta Pruning Efficiency.",
                "assignments": ["Trace Alpha-Beta pruning."],
                "resource_url": "https://inst.eecs.berkeley.edu/~cs188/su25/"
            },
            "Unit 4: Artificial Neural Networks [8 Hrs]": {
                "topics": ["Perceptron", "Backpropagation", "Gradient Descent Variants"],
                "summary": "Covers foundational deep learning concepts.",
                "proofs": "Backpropagation Delta Rule.",
                "assignments": ["Derive backpropagation equations."],
                "resource_url": "https://cs229.stanford.edu/"
            },
            "Unit 5: Computer Vision [8 Hrs]": {
                "topics": ["Image Formation", "Object Detection", "Modern Vision Architectures"],
                "summary": "Focuses on feature extraction and spatial convolutions.",
                "proofs": "2D Convolution Operation.",
                "assignments": ["Build CNN in PyTorch."],
                "resource_url": "https://cs231n.stanford.edu/"
            }
        }
    },
    "PCC_ML": {
        "title": "Machine Learning",
        "code": "PCC",
        "credits": 4,
        "type": "Programme Core Course",
        "units": {
            "Unit 1: Introduction & ML Systems Engineering [6 Hrs]": {
                "topics": ["AI/ML Basics", "Lifecycle of ML Systems", "Types of ML"],
                "summary": "Covers end-to-end ML production lifecycles.",
                "proofs": "Generalization Error Bound.",
                "assignments": ["Design system architecture."],
                "resource_url": "https://cs229.stanford.edu/"
            },
            "Unit 2: Supervised Learning [10 Hrs]": {
                "topics": ["Linear Regression", "Logistic Regression", "Decision Trees", "SVM"],
                "summary": "Classical algorithms for continuous and categorical targets.",
                "proofs": "Logistic Loss Formulation.",
                "assignments": ["Implement SVM."],
                "resource_url": "https://cs229.stanford.edu/"
            },
            "Unit 3: Ensemble Learning [8 Hrs]": {
                "topics": ["Bagging & Random Forests", "Boosting", "Hyperparameter Tuning"],
                "summary": "Combines weak learners to build robust models.",
                "proofs": "AdaBoost Weight Update.",
                "assignments": ["Train Gradient Boosting model."],
                "resource_url": "https://cs229.stanford.edu/"
            },
            "Unit 4: Unsupervised Learning [8 Hrs]": {
                "topics": ["K-Means Clustering", "DBSCAN", "Gaussian Mixture Models"],
                "summary": "Focuses on partitioning unlabelled data.",
                "proofs": "GMM Expectation-Maximization.",
                "assignments": ["Implement K-Means."],
                "resource_url": "https://cs229.stanford.edu/"
            },
            "Unit 5: Dimensionality Reduction [8 Hrs]": {
                "topics": ["PCA", "t-SNE", "Isolation Forest"],
                "summary": "Explores variance-maximizing projections.",
                "proofs": "PCA Eigenvalue Formulation.",
                "assignments": ["Apply PCA."],
                "resource_url": "https://cs229.stanford.edu/"
            }
        }
    },
    "PCC_DV": {
        "title": "Data Visualization Techniques",
        "code": "PCC",
        "credits": 4,
        "type": "Programme Core Course",
        "units": {
            "Unit 1: Purpose of Data Visualization [8 Hrs]": {
                "topics": ["Data Taxonomy", "Exploratory Data Analysis"],
                "summary": "Establishes human visual perception principles.",
                "proofs": "Lie Factor Calculation.",
                "assignments": ["Perform thorough EDA."],
                "resource_url": "https://www.data-to-viz.com/"
            },
            "Unit 2: Pre-processing and Visualizing [8 Hrs]": {
                "topics": ["Web Scraping", "Aesthetic Mapping", "Coordinate Systems"],
                "summary": "Covers data ingestion pipelines.",
                "proofs": "Linear Scale Mapping.",
                "assignments": ["Build scraper in Python."],
                "resource_url": "https://ggplot2.tidyverse.org/"
            },
            "Unit 3: Distributions [8 Hrs]": {
                "topics": ["Bar Plots", "Histograms & Density Plots"],
                "summary": "Techniques for displaying categorical counts.",
                "proofs": "Kernel Density Estimator.",
                "assignments": ["Generate q-q plots."],
                "resource_url": "https://seaborn.pydata.org/"
            },
            "Unit 4: Advanced Visualization [8 Hrs]": {
                "topics": ["Nested Proportions", "Scatter Plots & Correlograms"],
                "summary": "Multi-variable visual summaries.",
                "proofs": "Pearson Correlation Coefficient.",
                "assignments": ["Construct interactive dashboard."],
                "resource_url": "https://plotly.com/python/"
            },
            "Unit 5: Geospatial Data [8 Hrs]": {
                "topics": ["Time Series Visualization", "Geospatial Data", "Choropleth Mapping"],
                "summary": "Handling temporal data and spatial shapefiles.",
                "proofs": "Exponential Smoothing Formula.",
                "assignments": ["Build Folium map."],
                "resource_url": "https://geopandas.org/"
            }
        }
    },
    "PEC_1": {
        "title": "Department Elective I: Data Analytics",
        "code": "PEC-1",
        "credits": 3,
        "type": "Programme Elective Course",
        "units": {
            "Unit 1: Fundamentals of Data Analytics [8 Hrs]": {
                "topics": ["Data Analytics Basics", "Pre-processing"],
                "summary": "Foundational steps in clean data engineering.",
                "proofs": "Mean Squared Error Imputation.",
                "assignments": ["Clean messy dataset."],
                "resource_url": "https://pandas.pydata.org/docs/"
            },
            "Unit 2: Python Analytics [8 Hrs]": {
                "topics": ["NumPy, Pandas, SciPy", "Web Scraping"],
                "summary": "Vectorized compute operations.",
                "proofs": "Vectorized Dot Product.",
                "assignments": ["Analyze using Pandas."],
                "resource_url": "https://scipy.org/"
            },
            "Unit 3: Correlated Data Analysis [8 Hrs]": {
                "topics": ["ANOVA", "Chi-Square Test", "Logistic Regression"],
                "summary": "Hypothesis testing for independence.",
                "proofs": "Chi-Square Test Statistic.",
                "assignments": ["Execute Chi-Square test."],
                "resource_url": "https://www.statsmodels.org/"
            },
            "Unit 4: Cluster Analysis [8 Hrs]": {
                "topics": ["Decision Tree Construction", "Cluster Analysis"],
                "summary": "Information gain optimization.",
                "proofs": "Gini Impurity.",
                "assignments": ["Build Decision Tree Classifier."],
                "resource_url": "https://scikit-learn.org/stable/"
            },
            "Unit 5: Social Media Analytics [8 Hrs]": {
                "topics": ["Social Network Features", "Association Rule Mining"],
                "summary": "Graph centrality metrics.",
                "proofs": "Association Rule Lift.",
                "assignments": ["Apply Apriori algorithm."],
                "resource_url": "https://networkx.org/"
            }
        }
    },
    "RM_IPR": {
        "title": "Research Methodology and IPR",
        "code": "RM",
        "credits": 3,
        "type": "Research Methodology",
        "units": {
            "Unit 1: Problem Formulation [8 Hrs]": {
                "topics": ["Meaning & Sources of Research Problem", "Scope & Objectives"],
                "summary": "Framing clear research gaps.",
                "proofs": "Methodological Logic Flow.",
                "assignments": ["Draft problem statement."],
                "resource_url": "https://nptel.ac.in/courses/121106007"
            },
            "Unit 2: Literature Studies [8 Hrs]": {
                "topics": ["Literature Survey Approaches", "Design of Experiments"],
                "summary": "Systematic literature review.",
                "proofs": "Taguchi Signal-to-Noise Ratio.",
                "assignments": ["Construct Literature Matrix."],
                "resource_url": "https://nptel.ac.in/courses/121106007"
            },
            "Unit 3: Research Proposal [8 Hrs]": {
                "topics": ["Plagiarism & Research Ethics", "Technical Writing"],
                "summary": "Ethics in publishing.",
                "proofs": "Plagiarism Guidelines.",
                "assignments": ["Write formal research proposal."],
                "resource_url": "https://nptel.ac.in/courses/121106007"
            },
            "Unit 4: Intellectual Property Rights [8 Hrs]": {
                "topics": ["Nature & Importance of IPR", "Objectives of IPR"],
                "summary": "Economic leadership through IP protection.",
                "proofs": "TRIPS Agreement Guidelines.",
                "assignments": ["Analyze a patent dispute."],
                "resource_url": "https://nptel.ac.in/courses/109106137"
            },
            "Unit 5: Types of IPR [8 Hrs]": {
                "topics": ["Patents", "Trademarks, Copyrights"],
                "summary": "Filing procedures and strategy.",
                "proofs": "Patent Claims Structure.",
                "assignments": ["Draft a Provisional Patent."],
                "resource_url": "https://ipindia.gov.in/"
            }
        }
    }
}

# ==============================================================================
# 2. STATE MANAGEMENT & API LOGIC
# ==============================================================================

if 'view' not in st.session_state:
    st.session_state.view = 'home'
if 'selected_course' not in st.session_state:
    st.session_state.selected_course = 'PSMC'
if 'selected_unit' not in st.session_state:
    st.session_state.selected_unit = None
if 'selected_topic' not in st.session_state:
    st.session_state.selected_topic = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'completed_courses' not in st.session_state:
    st.session_state.completed_courses = set()
if 'subject_books' not in st.session_state:
    st.session_state.subject_books = {}
# ADDED: Store array of full-coverage files
if 'additional_sources' not in st.session_state:
    st.session_state.additional_sources = {}

def navigate_to(view_name, course_id=None, unit_name=None, topic_name=None):
    st.session_state.view = view_name
    if course_id:
        st.session_state.selected_course = course_id
    if unit_name:
        st.session_state.selected_unit = unit_name
    if topic_name:
        st.session_state.selected_topic = topic_name

# MODIFIED: Accepts a list of file references
def get_gemini_response(api_key, prompt_text, system_context="", file_refs=None):
    if not HAS_GENAI:
        return "⚠️ Error: `google-genai` library is missing."
    if not api_key:
        return "Please enter your Gemini API key in the sidebar."
    
    try:
        client = genai.Client(api_key=api_key)
        
        contents_payload = []
        if file_refs:
            for f in file_refs:
                contents_payload.append(f)
            
        contents_payload.append(f"System Context: {system_context}\n\nUser Question: {prompt_text}")
        
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=contents_payload
        )
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini API: {str(e)}"

# ==============================================================================
# 3. SIDEBAR CONTROLS
# ==============================================================================

with st.sidebar:
    st.title("⚙️ Control Panel")
    secret_key = ""
    
    try:
        if "GEMINI_API_KEY" in st.secrets:
            secret_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass 
        
    gemini_key = st.text_input("Enter Gemini API Key:", value=secret_key, type="password")
    
    st.divider()
    st.markdown("### 📊 Semester 1 Progress")
    comp_crs = len(st.session_state.completed_courses)
    st.progress(comp_crs / len(SYLLABUS_DB))
    
    st.divider()
    if st.button("🏠 Return to Dashboard", use_container_width=True):
        navigate_to('home')

# ==============================================================================
# 4. VIEW 1: DASHBOARD LANDING PAGE
# ==============================================================================
if st.session_state.view == 'home':
    st.markdown("""
    <div class="main-header">
        <h1 style="margin-top:0px;">M.Tech AIML Syllabus Explorer</h1>
        <p>Search the complete curriculum and practice assignments.</p>
    </div>
    """, unsafe_allow_html=True)
    
    cols = st.columns(3)
    for idx, (c_id, c_data) in enumerate(SYLLABUS_DB.items()):
        with cols[idx % 3]:
            with st.container(border=True):
                st.caption(f"Sem 1 • {c_data['code']}")
                st.markdown(f"#### {c_data['title']}")
                if st.button("Inspect Unit ➡️", key=f"btn_inspect_{c_id}", use_container_width=True):
                    first_unit = list(c_data['units'].keys())[0]
                    navigate_to('detail', c_id, first_unit, c_data['units'][first_unit]['topics'][0])

# ==============================================================================
# 5. VIEW 2: DRILL-DOWN INSPECTION & DYNAMIC PDF SLICER
# ==============================================================================
elif st.session_state.view == 'detail':
    course_id = st.session_state.selected_course
    course_data = SYLLABUS_DB[course_id]
    
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title(course_data['title'])
    with col2:
        if st.button("✖ Dashboard", use_container_width=True):
            navigate_to('home')
            st.rerun()
            
    # --- REDESIGNED COURSE VAULT: SLICER + FULL ADDITIONAL SOURCES ---
    with st.expander(f"📚 Course Vault: Knowledge Base for {course_data['code']}", expanded=True):
        
        st.markdown("#### 1. Main Textbook (Requires Slicing)")
        if course_id in st.session_state.subject_books:
            st.success(f"✅ Textbook slice successfully indexed.")
            if st.button("Replace/Upload New Slice"):
                del st.session_state.subject_books[course_id]
                st.rerun()
        else:
            st.info("Upload the heavy textbook PDF, then select the specific page range for the current unit.")
            uploaded_pdf = st.file_uploader("Choose Textbook PDF", type=["pdf"], key="textbook")
            
            if uploaded_pdf and gemini_key:
                page_col1, page_col2 = st.columns(2)
                with page_col1:
                    start_page = st.number_input("Start Page", min_value=1, value=1)
                with page_col2:
                    end_page = st.number_input("End Page", min_value=1, value=25)
                    
                if st.button("✂️ Slice & Upload Textbook", type="primary"):
                    if start_page > end_page:
                        st.error("Start page must be less than or equal to End page.")
                    else:
                        with st.spinner("Slicing PDF and indexing into Gemini Vault..."):
                            tmp_path = ""
                            try:
                                reader = pypdf.PdfReader(uploaded_pdf)
                                writer = pypdf.PdfWriter()
                                total_pages = len(reader.pages)
                                safe_start = max(0, start_page - 1)
                                safe_end = min(end_page, total_pages)
                                
                                for i in range(safe_start, safe_end):
                                    writer.add_page(reader.pages[i])
                                    
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                    writer.write(tmp)
                                    tmp_path = tmp.name
                                    
                                client = genai.Client(api_key=gemini_key)
                                uploaded_doc = client.files.upload(file=tmp_path)
                                st.session_state.subject_books[course_id] = uploaded_doc
                                
                                st.success(f"Successfully sliced pages {start_page} to {safe_end} and uploaded!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Slicing/Upload failed: {e}")
                            finally:
                                if tmp_path and os.path.exists(tmp_path):
                                    os.remove(tmp_path)
            elif uploaded_pdf and not gemini_key:
                st.warning("Please enter your Gemini API Key in the sidebar first.")
                
        st.divider()
        
        # NEW: Multiple slides/files uploader
        st.markdown("#### 2. Class Slides & Additional Sources (Full Coverage)")
        st.info("Upload class slides, topic notes, or extra PDFs. These will be read in full without page slicing.")
        additional_files = st.file_uploader("Upload Slides/PDFs", type=["pdf"], accept_multiple_files=True, key="add_files")
        
        if additional_files and gemini_key:
            if st.button("Upload Additional Sources"):
                with st.spinner("Indexing additional sources..."):
                    if course_id not in st.session_state.additional_sources:
                        st.session_state.additional_sources[course_id] = []
                        
                    client = genai.Client(api_key=gemini_key)
                    for file in additional_files:
                        tmp_path = ""
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                tmp.write(file.read())
                                tmp_path = tmp.name
                            uploaded_doc = client.files.upload(file=tmp_path)
                            st.session_state.additional_sources[course_id].append({"name": file.name, "doc": uploaded_doc})
                        except Exception as e:
                            st.error(f"Failed to upload {file.name}: {e}")
                        finally:
                            if tmp_path and os.path.exists(tmp_path):
                                os.remove(tmp_path)
                    st.success("Additional sources successfully added to vault!")
                    st.rerun()
                    
        # Display existing extra sources
        if course_id in st.session_state.additional_sources and st.session_state.additional_sources[course_id]:
            st.markdown("**Currently Uploaded Additional Sources:**")
            for idx, src in enumerate(st.session_state.additional_sources[course_id]):
                st.write(f"- 📄 `{src['name']}`")
            if st.button("Clear Additional Sources", type="secondary"):
                del st.session_state.additional_sources[course_id]
                st.rerun()

    st.divider()
    
    # UI Layout
    left_panel, right_panel = st.columns([1.6, 1.0])
    
    # Pre-calculate what exists in the Vault
    has_textbook = course_id in st.session_state.subject_books
    has_additional = course_id in st.session_state.additional_sources and len(st.session_state.additional_sources[course_id]) > 0
    is_grounded = has_textbook or has_additional
    
    with left_panel:
        for unit_title, unit_info in course_data['units'].items():
            with st.expander(f"📖 {unit_title}", expanded=(unit_title == st.session_state.selected_unit)):
                btn_cols = st.columns(len(unit_info['topics']))
                for t_idx, topic_name in enumerate(unit_info['topics']):
                    b_type = "primary" if topic_name == st.session_state.selected_topic else "secondary"
                    if btn_cols[t_idx % len(btn_cols)].button(topic_name, key=f"btn_{unit_title}_{t_idx}", type=b_type):
                        st.session_state.selected_unit = unit_title
                        st.session_state.selected_topic = topic_name
                        st.rerun()

        if st.session_state.selected_topic:
            active_unit = course_data['units'][st.session_state.selected_unit]
            st.markdown(f"### 🔍 Focus: `{st.session_state.selected_topic}`")
            
            t_sum, t_proof, t_assgn, t_video, t_notebook = st.tabs(["📝 Summary", "🧮 Proofs", "🎯 Assignments", "🔗 Videos", "🧠 Textbook Engine"])
            
            with t_sum: 
                st.write(active_unit['summary'])
            with t_proof: 
                st.success(active_unit['proofs'])
            with t_assgn:
                for task in active_unit['assignments']: 
                    st.write(f"- {task}")
            with t_video:
                st.markdown("Click below to open the top-ranked academic video tutorial for this specific module:")
                st.markdown(f"👉 [**Open Official Video Tutorial for {st.session_state.selected_topic}**]({active_unit['resource_url']})")
            
            with t_notebook:
                st.markdown("#### Extract from Uploaded Vault")
                if is_grounded:
                    if st.button(f"Generate Deep Summary for '{st.session_state.selected_topic}'", type="primary"):
                        with st.spinner("Analyzing uploaded vault sources..."):
                            prompt = f"Please read the attached files (textbooks/slides) and provide a detailed summary, key formulas, and concepts specifically regarding '{st.session_state.selected_topic}'."
                            
                            # Gather all uploaded refs
                            all_refs = []
                            if has_textbook: 
                                all_refs.append(st.session_state.subject_books[course_id])
                            if has_additional:
                                for src in st.session_state.additional_sources[course_id]:
                                    all_refs.append(src['doc'])
                                    
                            response = get_gemini_response(gemini_key, prompt, system_context="You are a strict academic summarizer.", file_refs=all_refs)
                            st.markdown(response)
                else:
                    st.warning("Upload a textbook or class slides in the Course Vault above to unlock this feature.")

    # AI Chat Panel (Context Aware + Focus Mode Toggle)
    with right_panel:
        st.subheader("💬 AI Tutor")
        active_t = st.session_state.selected_topic or "General"
        
        # New Feature: Let user lock the Chatbot context
        if is_grounded:
            st.caption(f"✅ Grounded in Course Vault | Topic: `{active_t}`")
            chat_mode = st.radio(
                "Tutor Focus Mode:", 
                ["General + Vault Knowledge", "Strict (Vault Sources Only)"],
                horizontal=True,
                help="Select 'Strict' if you want the tutor to ONLY answer using your uploaded textbooks and slides, ignoring general web knowledge."
            )
        else:
            st.caption(f"Topic: `{active_t}` (Standard Knowledge)")
            chat_mode = "General + Vault Knowledge"
        
        chat_box = st.container(height=360)
        with chat_box:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    
        if user_query := st.chat_input("Ask a question..."):
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            # Prepare contextual instructions based on mode
            if chat_mode == "Strict (Vault Sources Only)":
                sys_ctx = f"Course: {course_data['title']}, Topic: {active_t}. INSTRUCTION: You MUST answer the user's question STRICTLY and ONLY using the information found in the attached syllabus files (slides, textbook). Do not use outside knowledge. If the answer is not in the attached documents, politely state that you cannot answer based on the current resources."
            else:
                sys_ctx = f"You are an expert Professor at COEP Tech teaching M.Tech AIML. Course: {course_data['title']}, Topic: {active_t}."
            
            # Gather references
            all_refs = []
            if has_textbook: 
                all_refs.append(st.session_state.subject_books[course_id])
            if has_additional:
                for src in st.session_state.additional_sources[course_id]:
                    all_refs.append(src['doc'])
                    
            ai_reply = get_gemini_response(gemini_key, user_query, system_context=sys_ctx, file_refs=all_refs)
            
            st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})
            st.rerun()

