
import streamlit as st
import pandas as pd

# ------------------------------------------------------------------------------
# NEW GENAI SDK INITIALIZATION
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
    .topic-cta {
        background-color: #f0f2f6;
        border: 1px solid #d0d7de;
        border-radius: 6px;
        padding: 6px 12px;
        margin: 3px;
        display: inline-block;
        font-size: 0.88rem;
    }
    .unit-card {
        border-left: 4px solid #7A5CFA;
        background-color: #f9f9fb;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. COMPREHENSIVE SEMESTER 1 SYLLABUS DATABASE (COEP TECH M.TECH AIML)
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
                "topics": ["Asymptotic Notation ($O, \\Omega, \\Theta, o, \\omega$)", "Standard Functions & Summations", "Solving Recurrences (Substitution, Recursion Tree, Master Method)"],
                "summary": "Establishes formal bounds on algorithmic growth rates and mathematical tools for recurrence relations.",
                "proofs": "Master Theorem: $T(n) = aT(n/b) + f(n) \\implies T(n) = \\Theta(n^{\\log_b a})$ when $f(n) = O(n^{\\log_b a - \\epsilon})$",
                "assignments": ["Solve $T(n) = 2T(n/2) + n \\log n$ using Master Theorem.", "Prove $3n^2 + 5n = \\Theta(n^2)$ using $c_1, c_2, n_0$ definition."],
                "resource_url": "https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/"
            },
            "Unit 2: Analysis of Algorithms [4 Hrs]": {
                "topics": ["Time and Space Complexity Analysis", "Worst-case Analysis of Data Structures (Heapsort)", "Average-case Analysis of Quicksort"],
                "summary": "Analyzes worst-case and expected execution bounds for fundamental search and sort algorithms.",
                "proofs": "Quicksort Average Time Recurrence: $T(n) = \\frac{2}{n} \\sum_{k=0}^{n-1} T(k) + \\Theta(n) = O(n \\log n)$",
                "assignments": ["Trace Heapsort on array $[4, 10, 3, 5, 1]$ and count operations.", "Derive Quicksort worst-case recurrence."],
                "resource_url": "https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/"
            },
            "Unit 3: Standard Design Techniques-I [6 Hrs]": {
                "topics": ["Divide and Conquer Strategy", "Greedy Method", "Fractional Knapsack", "Huffman Coding", "Job Sequencing"],
                "summary": "Explores greedy choice properties, optimal substructure, and divide-and-conquer paradigms.",
                "proofs": "Greedy Choice Property Proof for Huffman Coding / Fractional Knapsack.",
                "assignments": ["Implement Huffman Coding for string 'ABRACADABRA'.", "Solve Job Sequencing with deadlines."],
                "resource_url": "https://nptel.ac.in/courses/106102011"
            },
            "Unit 4: Standard Design Techniques-II [6 Hrs]": {
                "topics": ["Dynamic Programming (0/1 Knapsack, LCS, Matrix Chain)", "Network Flow (Ford-Fulkerson, Max-Flow Min-Cut)"],
                "summary": "Covers overlapping subproblems, memoization, tabular formulation, and maximum flow in networks.",
                "proofs": "Max-Flow Min-Cut Theorem: Value of max flow equals capacity of minimum cut.",
                "assignments": ["Solve Matrix Chain Multiplication for dimensions $[10, 30, 5, 60]$.", "Implement Ford-Fulkerson."],
                "resource_url": "https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/"
            },
            "Unit 5: Standard Design Techniques-III [6 Hrs]": {
                "topics": ["Backtracking Strategy (N-Queens, Graph Coloring)", "Branch-and-Bound (0/1 Knapsack, TSP)"],
                "summary": "Explores state-space tree traversal, pruning non-promising nodes, bounding functions, and optimization.",
                "proofs": "Bounding Function Formulation for TSP Branch and Bound.",
                "assignments": ["Solve 8-Queens problem using Backtracking.", "Solve TSP for 4 cities using Branch and Bound."],
                "resource_url": "https://nptel.ac.in/courses/106102011"
            },
            "Unit 6: Complexity Theory [4 Hrs]": {
                "topics": ["P, NP, NP-Hard, NP-Complete Classes", "Polynomial-Time Reductions", "SAT, 3SAT, 3VC, Independent Set, Subset Sum, Hamiltonian Circuit"],
                "summary": "Classifies computational problems based on tractability and formal proofs of NP-completeness.",
                "proofs": "Cook-Levin Theorem (SAT is NP-Complete) | Reduction: 3SAT $\\le_P$ Vertex Cover.",
                "assignments": ["Prove Independent Set is NP-Complete by reduction from Vertex Cover."],
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
                "topics": ["Foundations & History of AI", "Intelligent Agents & Environments", "Concept of Rationality", "PEAS Description", "Agent Structures (Reflex, Goal, Utility-based)"],
                "summary": "Introduces AI paradigms, environment properties (deterministic, stochastic, static, dynamic), and rational decision making.",
                "proofs": "Utility Expectation: $a^* = \\arg\\max_a \\sum_{s'} P(s'|s,a) U(s')$",
                "assignments": ["Define PEAS for Automated Medical Diagnosis Agent.", "Compare Simple Reflex vs Utility-Based Agents."],
                "resource_url": "https://inst.eecs.berkeley.edu/~cs188/su25/"
            },
            "Unit 2: Solving Problem by Searching [8 Hrs]": {
                "topics": ["Uninformed Search (BFS, DFS, IDDFS, UCS)", "Informed Heuristic Search (A*, Greedy Best-First)", "Admissibility & Consistency of Heuristics", "Local Search (Hill Climbing, Simulated Annealing)", "Online & Partially Observable Search"],
                "summary": "Covers graph traversal strategies, heuristic search optimality conditions, and local optimization in complex spaces.",
                "proofs": "A* Optimality Proof: $f(n) = g(n) + h(n)$ is optimal if $h(n)$ is admissible ($h(n) \\le h^*(n)$).",
                "assignments": ["Implement A* search for 8-Puzzle problem with Manhattan Distance.", "Prove consistency implies admissibility."],
                "resource_url": "https://inst.eecs.berkeley.edu/~cs188/su25/"
            },
            "Unit 3: Game Solver & Constraint Satisfaction Problems [8 Hrs]": {
                "topics": ["Game Theory & Minimax Algorithm", "Alpha-Beta Pruning", "Monte Carlo Tree Search (MCTS)", "Constraint Propagation & Backtracking (MRV, LCV, AC-3)"],
                "summary": "Analyzes adversarial decision making in games and CSP solving using arc consistency and heuristics.",
                "proofs": "Alpha-Beta Pruning Efficiency: Reduces search complexity from $O(b^m)$ to $O(b^{m/2})$ in best case.",
                "assignments": ["Trace Alpha-Beta pruning on a game tree of depth 4.", "Implement Sudoku solver using AC-3 and Backtracking."],
                "resource_url": "https://inst.eecs.berkeley.edu/~cs188/su25/"
            },
            "Unit 4: Artificial Neural Networks [8 Hrs]": {
                "topics": ["Perceptron & Multilayer Perceptron", "Activation Functions (ReLU, Sigmoid, Softmax)", "Loss Functions & Backpropagation", "Gradient Descent Variants", "Basics of CNNs and RNNs"],
                "summary": "Covers foundational deep learning concepts, automatic differentiation via computational graphs, and loss optimization.",
                "proofs": "Backpropagation Delta Rule: $\\frac{\\partial L}{\\partial w_{ij}} = \\delta_j x_i$ where $\\delta_j = \\frac{\\partial L}{\\partial z_j}$",
                "assignments": ["Derive backpropagation equations for a 2-layer neural network from scratch."],
                "resource_url": "https://cs229.stanford.edu/"
            },
            "Unit 5: Computer Vision [8 Hrs]": {
                "topics": ["Image Formation & Features", "Image Classification & Object Detection", "Modern Vision Architectures (CNNs, Vision Transformers)", "Encoder-Decoder Architectures"],
                "summary": "Focuses on feature extraction, spatial convolutions, bounding box regression, and modern ViT models.",
                "proofs": "2D Convolution Operation: $S(i,j) = (I * K)(i,j) = \\sum_m \\sum_n I(i-m, j-n) K(m,n)$",
                "assignments": ["Build CNN in PyTorch for CIFAR-10 classification.", "Apply transfer learning with ResNet-18."],
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
                "topics": ["AI/ML Basics & Evolution", "ML Systems Engineering", "Lifecycle of ML Systems", "Types of ML (Supervised, Unsupervised, RL)"],
                "summary": "Covers end-to-end ML production lifecycles, data pipelines, model requirements, and system design principles.",
                "proofs": "Generalization Error Bound: $R(f) \\le R_{emp}(f) + \\sqrt{\\frac{h(\\ln(2n/h) + 1) - \\ln(\\delta/4)}{n}}$",
                "assignments": ["Design system architecture for an e-commerce recommendation system."],
                "resource_url": "https://cs229.stanford.edu/"
            },
            "Unit 2: Supervised Learning (Regression & Classification) [10 Hrs]": {
                "topics": ["Linear Regression & Gradient Descent", "Logistic Regression", "Naive Bayes Classifier", "Decision Trees & Impurity Measures", "KNN & Support Vector Machines (SVM)", "Performance Metrics (Confusion Matrix, ROC-AUC, F1-Score)"],
                "summary": "Deep dive into classical parametric and non-parametric algorithms for continuous and categorical targets.",
                "proofs": "Logistic Loss / Binary Cross-Entropy: $J(\\theta) = -\\frac{1}{m}\\sum [y^{(i)}\\ln h_\\theta(x^{(i)}) + (1-y^{(i)})\\ln(1-h_\\theta(x^{(i)}))]$",
                "assignments": ["Derive update rule for Logistic Regression.", "Implement SVM with RBF kernel on Iris dataset."],
                "resource_url": "https://cs229.stanford.edu/"
            },
            "Unit 3: Ensemble Learning [8 Hrs]": {
                "topics": ["Bagging & Random Forests", "Boosting (AdaBoost, Gradient Boosting, XGBoost)", "Model Selection & Hyperparameter Tuning", "Feature Selection"],
                "summary": "Combines weak learners to build robust models, reducing variance (Bagging) or bias (Boosting).",
                "proofs": "AdaBoost Weight Update: $w_{i} \\leftarrow w_{i} \\exp(\\alpha_m y_i g_m(x_i))$ where $\\alpha_m = \\frac{1}{2} \\ln\\frac{1-e_m}{e_m}$",
                "assignments": ["Train Gradient Boosting model on Heart Disease dataset and plot feature importances."],
                "resource_url": "https://cs229.stanford.edu/"
            },
            "Unit 4: Unsupervised Learning [8 Hrs]": {
                "topics": ["K-Means Clustering", "DBSCAN", "Hierarchical Clustering", "Gaussian Mixture Models (GMM)", "Silhouette Score Evaluation"],
                "summary": "Focuses on partitioning unlabelled data, density-based clustering, expectation-maximization (EM), and cluster validation.",
                "proofs": "GMM Expectation-Maximization Update Step for Responsibilities $\\gamma_{ik} = \\frac{\\pi_k \\mathcal{N}(x_i|\\mu_k, \\Sigma_k)}{\\sum_j \\pi_j \\mathcal{N}(x_i|\\mu_j, \\Sigma_j)}$",
                "assignments": ["Implement K-Means from scratch in Python and validate using Silhouette Score."],
                "resource_url": "https://cs229.stanford.edu/"
            },
            "Unit 5: Dimensionality Reduction & Anomaly Detection [8 Hrs]": {
                "topics": ["Principal Component Analysis (PCA)", "Linear Discriminant Analysis (LDA)", "t-SNE", "Isolation Forest", "One-Class SVM"],
                "summary": "Explores variance-maximizing projections, class separability transformations, manifold learning, and outlier detection.",
                "proofs": "PCA Eigenvalue Formulation: $S v_i = \\lambda_i v_i$ where $S$ is sample covariance matrix.",
                "assignments": ["Apply PCA on high-dimensional dataset to reduce 50 features down to 2 principal components."],
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
                "topics": ["Data Taxonomy (1D, 2D, 3D, Temporal, Trees, Graphs)", "Nominal, Ordinal, Quantitative Data", "Goals & Characteristics of Good Visualizations", "Exploratory Data Analysis (EDA)"],
                "summary": "Establishes human visual perception principles, data types, and exploratory inspection frameworks.",
                "proofs": "Lie Factor Calculation: $\\text{Lie Factor} = \\frac{\\text{Size of effect shown in graphic}}{\\text{Size of effect in data}}$",
                "assignments": ["Perform thorough EDA on Titanic dataset with Seaborn."],
                "resource_url": "https://www.data-to-viz.com/"
            },
            "Unit 2: Collecting, Pre-processing and Visualizing Data [8 Hrs]": {
                "topics": ["Web Scraping & Aggregation", "Aesthetic Mapping (Color, Shape, Size)", "Coordinate Systems (Cartesian, Polar, Curved Axes)"],
                "summary": "Covers data ingestion pipelines and mapping continuous/discrete variables onto visual encodings.",
                "proofs": "Linear Scale Mapping: $f(x) = y_{min} + \\frac{x - x_{min}}{x_{max} - x_{min}} (y_{max} - y_{min})$",
                "assignments": ["Build scraper in Python and map scraped data onto a customized polar plot."],
                "resource_url": "https://ggplot2.tidyverse.org/"
            },
            "Unit 3: Visualizing Amounts, Distributions, Proportions [8 Hrs]": {
                "topics": ["Bar Plots & Heatmaps", "Histograms & Density Plots (KDE)", "q-q Plots & Empirical CDFs"],
                "summary": "Techniques for displaying categorical counts, continuous probability distributions, and normality checks.",
                "proofs": "Kernel Density Estimator: $\\hat{f}_h(x) = \\frac{1}{nh} \\sum_{i=1}^n K\\left(\\frac{x - x_i}{h}\\right)$",
                "assignments": ["Generate q-q plots to inspect heavy-tailed continuous data."],
                "resource_url": "https://seaborn.pydata.org/"
            },
            "Unit 4: Advanced Visualization Design [8 Hrs]": {
                "topics": ["Visualizing Distributions along Axes", "Proportions & Nested Proportions (Treemaps)", "Scatter Plots & Correlograms", "Dimension Reduction Visuals"],
                "summary": "Multi-variable visual summaries, correlation matrix heatmaps, interactive pairplots, and treemaps.",
                "proofs": "Pearson Correlation Coefficient: $r = \\frac{\\sum (x_i-\\bar{x})(y_i-\\bar{y})}{\\sqrt{\\sum (x_i-\\bar{x})^2 \\sum (y_i-\\bar{y})^2}}$",
                "assignments": ["Construct interactive Plotly dashboard with linked scatter and bar charts."],
                "resource_url": "https://plotly.com/python/"
            },
            "Unit 5: Time Series, Trends and Geospatial Data [8 Hrs]": {
                "topics": ["Time Series Visualization & Smoothing", "Trend Decomposition", "Geospatial Data, Projections & Choropleth Mapping"],
                "summary": "Handling temporal data, seasonal decomposition, spatial shapefiles, GeoJSON, and choropleths.",
                "proofs": "Exponential Smoothing Formula: $s_t = \\alpha x_t + (1-\\alpha) s_{t-1}$",
                "assignments": ["Build Folium choropleth map of India showing state-level metrics."],
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
                "topics": ["Data Analytics Basics", "Pre-processing & Imputation", "Descriptive, Predictive, Prescriptive Analytics"],
                "summary": "Foundational steps in clean data engineering, missing value imputation, and analytical taxonomy.",
                "proofs": "Mean Squared Error Imputation Loss: $MSE = \\frac{1}{N} \\sum (y_i - \\hat{y}_i)^2$",
                "assignments": ["Clean messy dataset with Pandas and apply KNN imputation."],
                "resource_url": "https://pandas.pydata.org/docs/"
            },
            "Unit 2: Data Analytics with Python [8 Hrs]": {
                "topics": ["NumPy, Pandas, SciPy, Matplotlib", "Statistical Procedures", "Web Scraping in Python"],
                "summary": "Vectorized compute operations, dataframe operations, SciPy statistical hypothesis functions.",
                "proofs": "Vectorized Dot Product: $c = A \\cdot B = \\sum a_i b_i$",
                "assignments": ["Perform web scraping of financial tabular data and analyze using Pandas."],
                "resource_url": "https://scipy.org/"
            },
            "Unit 3: Correlated Data Analysis [8 Hrs]": {
                "topics": ["ANOVA and ANCOVA", "Chi-Square Statistical Test", "Linear and Logistic Regression Analysis"],
                "summary": "Hypothesis testing for independence, contingency tables, and regression diagnostic outputs.",
                "proofs": "Chi-Square Test Statistic: \\chi^2 = \\sum \\frac{(O_i - E_i)^2}{E_i}",
                "assignments": ["Execute Chi-Square test of independence on customer churn survey data."],
                "resource_url": "https://www.statsmodels.org/"
            },
            "Unit 4: Decision Tree and Cluster Analysis [8 Hrs]": {
                "topics": ["Decision Tree Construction & Algorithms", "Cluster Analysis & K-Means", "Advantages/Disadvantages of Clustering"],
                "summary": "Information gain optimization, Gini index, K-Means centroid convergence, and distance metrics.",
                "proofs": "Gini Impurity: $I_G(p) = 1 - \\sum_{i=1}^J p_i^2$",
                "assignments": ["Build Decision Tree Classifier from scratch using Gini impurity."],
                "resource_url": "https://scikit-learn.org/stable/"
            },
            "Unit 5: Social Media & Text Analytics [8 Hrs]": {
                "topics": ["Social Network Features & Graphs", "Association Rule Mining (Apriori)", "Artificial Neural Networks for Web Analytics"],
                "summary": "Graph centrality metrics, support/confidence/lift rules in transaction data, and web analytics.",
                "proofs": "Association Rule Lift: $\\text{Lift}(A \\rightarrow B) = \\frac{P(A \\cap B)}{P(A) P(B)}$",
                "assignments": ["Apply Apriori algorithm on market basket transaction dataset."],
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
            "Unit 1: Research Problem Formulation [8 Hrs]": {
                "topics": ["Meaning & Sources of Research Problem", "Criteria & Characteristics of Good Problem", "Errors in Selecting Problem", "Scope & Objectives"],
                "summary": "Framing clear research gaps, avoiding symptoms vs root cause confusion, defining measurable objectives.",
                "proofs": "Methodological Logic: Problem $\\rightarrow$ Hypothesis $\\rightarrow$ Variables $\\rightarrow$ Measurement $\\rightarrow$ Validation",
                "assignments": ["Draft a 100-word formal problem statement for your M.Tech dissertation topic."],
                "resource_url": "https://nptel.ac.in/courses/121106007"
            },
            "Unit 2: Literature Studies & DoE [8 Hrs]": {
                "topics": ["Effective Literature Survey Approaches", "Design of Experiments (DoE)", "Taguchi Method", "Prototype Build, Test & Redesign"],
                "summary": "Systematic literature review, orthogonal array experiment planning, and iterative prototype testing.",
                "proofs": "Taguchi Signal-to-Noise Ratio (Larger is Better): $S/N = -10 \\log_{10} \\left( \\frac{1}{n} \\sum \\frac{1}{y_i^2} \\right)$",
                "assignments": ["Construct a Literature Review Matrix for 5 IEEE papers in your domain."],
                "resource_url": "https://nptel.ac.in/courses/121106007"
            },
            "Unit 3: Developing a Research Proposal & Ethics [8 Hrs]": {
                "topics": ["Plagiarism & Research Ethics", "Effective Technical Writing", "Format of Research Proposal & Review Committee Assessment"],
                "summary": "Ethics in publishing, Turnitin/Grammarly standards, structuring grant/thesis proposals.",
                "proofs": "Plagiarism Similarity Index Ceiling Guidelines (< 10% threshold).",
                "assignments": ["Write a 2-page formal research proposal following standard IEEE/COEP format."],
                "resource_url": "https://nptel.ac.in/courses/121106007"
            },
            "Unit 4: Intellectual Property Rights [8 Hrs]": {
                "topics": ["Concepts of Property & Intellectual Property", "Nature & Importance of IPR", "Objectives of IPR Frameworks"],
                "summary": "Economic leadership through IP protection, national wealth generation, and patent dynamics.",
                "proofs": "TRIPS Agreement Guidelines & Patent Lifespan Rules (20 Years from filing).",
                "assignments": ["Case Study: Analyze a recent patent dispute in AI software algorithms."],
                "resource_url": "https://nptel.ac.in/courses/109106137"
            },
            "Unit 5: Types of Intellectual Property Rights [8 Hrs]": {
                "topics": ["Patents & Indian Patent Office Administration", "Patent Rights, Licensing & Tech Transfer", "Provisional & Non-Provisional Applications", "Trademarks, Copyrights, Industrial Designs"],
                "summary": "Filing procedures, claims specification drafting, provisional application strategy, and copyright rules.",
                "proofs": "Patent Claims Structure: Preamble + Transition + Body of Claims.",
                "assignments": ["Draft a mock Provisional Patent Application for a novel software pipeline."],
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

def navigate_to(view_name, course_id=None, unit_name=None, topic_name=None):
    st.session_state.view = view_name
    if course_id:
        st.session_state.selected_course = course_id
    if unit_name:
        st.session_state.selected_unit = unit_name
    if topic_name:
        st.session_state.selected_topic = topic_name

def get_gemini_response(api_key, prompt_text, system_context=""):
    if not HAS_GENAI:
        return "⚠️ **Error:** The `google-genai` library is not installed. Please stop the app, run `pip install google-genai` in your terminal, and restart."
    
    if not api_key:
        return f"**[AI Tutor Sandbox]** (No API Key detected).\n\nHere is a structured explanation regarding **{st.session_state.selected_topic or 'your query'}**:\n\n* **Core Concept:** This topic in *{st.session_state.selected_course}* forms a foundational building block for Semester 1.\n* **Key Takeaway:** Ensure you review the formulas and complete the practice assignments listed in the left panel before class.\n\n*(Tip: Paste your Gemini API key in the left sidebar to enable live AI responses!)*"
    
    try:
        # Initialize the new SDK Client
        client = genai.Client(api_key=api_key)
        
        full_prompt = f"System Context: You are an expert Professor at COEP Tech teaching M.Tech AIML. Context: {system_context}\n\nUser Question: {prompt_text}"
        
        # Pointing explicitly to the Pro reasoning model
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=full_prompt
        )
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini API: {str(e)}"

# ==============================================================================
# 3. SIDEBAR CONTROLS & API CONFIG
# ==============================================================================

with st.sidebar:
    st.title("⚙️ Control Panel")
    st.markdown("### 🔑 Gemini AI Integration")
    gemini_key = st.text_input("Enter Gemini API Key:", type="password", help="Get your free key from Google AI Studio")
    
    st.divider()
    st.markdown("### 📊 Semester 1 Progress")
    total_crs = len(SYLLABUS_DB)
    comp_crs = len(st.session_state.completed_courses)
    st.progress(comp_crs / total_crs)
    st.caption(f"Completed {comp_crs} of {total_crs} Courses ({int(comp_crs/total_crs*100)}%)")
    
    st.divider()
    if st.button("🏠 Return to Dashboard", use_container_width=True):
        navigate_to('home')

# ==============================================================================
# 4. VIEW 1: DASHBOARD LANDING PAGE
# ==============================================================================

if st.session_state.view == 'home':
    # Main Header
    st.markdown(f"""
    <div class="main-header">
        <p style="font-size:0.95rem; margin-bottom:2px; opacity:0.9;">COEP Technological University • AY 2025-26</p>
        <h1 style="margin-top:0px; font-size:2.2rem; font-weight:700;">M.Tech AIML Syllabus Explorer</h1>
        <p style="opacity:0.95; max-width:800px;">Search the complete curriculum, inspect all Semester 1 course units, practice assignments, and interact with the AI Tutor.</p>
        <div style="display: flex; gap: 15px; margin-top: 20px;">
            <div class="stat-card" style="flex: 1;"><h2>25</h2><p style="margin:0; font-size:0.85rem;">Total Credits</p></div>
            <div class="stat-card" style="flex: 1;"><h2>7</h2><p style="margin:0; font-size:0.85rem;">Subjects</p></div>
            <div class="stat-card" style="flex: 1;"><h2>38</h2><p style="margin:0; font-size:0.85rem;">Mapped Units</p></div>
            <div class="stat-card" style="flex: 1;"><h2>{int(comp_crs/total_crs*100)}%</h2><p style="margin:0; font-size:0.85rem;">Progress</p></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Search and Filter
    col_search, col_filter = st.columns([3, 1])
    with col_search:
        search_query = st.text_input("🔍 Search courses, units, or specific topics...", placeholder="e.g. Markov Chains, A* Search, Backpropagation")
    with col_filter:
        st.selectbox("Filter Semester", ["Semester 1 (Active)", "Semester 2", "Semester 3", "Semester 4"])
        
    st.markdown("### Semester 1 Courses")
    
    # Render Course Grid (3 Columns)
    cols = st.columns(3)
    for idx, (c_id, c_data) in enumerate(SYLLABUS_DB.items()):
        # Filter logic
        if search_query:
            match = search_query.lower() in c_data['title'].lower() or any(
                search_query.lower() in u.lower() or any(search_query.lower() in t.lower() for t in u_data['topics'])
                for u, u_data in c_data['units'].items()
            )
            if not match:
                continue
                
        with cols[idx % 3]:
            with st.container(border=True):
                st.caption(f"Sem 1 • {c_data['code']}")
                st.markdown(f"#### {c_data['title']}")
                st.write(f"**{c_data['credits']} Credits** • {len(c_data['units'])} Units")
                st.caption(c_data['type'])
                
                c_col1, c_col2 = st.columns([1, 1])
                with c_col1:
                    is_comp = st.checkbox("Completed", key=f"comp_{c_id}", value=(c_id in st.session_state.completed_courses))
                    if is_comp:
                        st.session_state.completed_courses.add(c_id)
                    else:
                        st.session_state.completed_courses.discard(c_id)
                with c_col2:
                    if st.button("Inspect Unit ➡️", key=f"btn_inspect_{c_id}", use_container_width=True):
                        # Default to first unit
                        first_unit = list(c_data['units'].keys())[0]
                        first_topic = c_data['units'][first_unit]['topics'][0]
                        navigate_to('detail', c_id, first_unit, first_topic)

# ==============================================================================
# 5. VIEW 2: DRILL-DOWN INSPECTION & TOPIC CTA EXPLORER
# ==============================================================================

elif st.session_state.view == 'detail':
    course_id = st.session_state.selected_course
    course_data = SYLLABUS_DB[course_id]
    
    # Navigation Header
    head_col1, head_col2 = st.columns([4, 1])
    with head_col1:
        st.caption(f"Semester 1 / {course_data['code']}")
        st.title(course_data['title'])
        st.markdown(f"**Credits:** {course_data['credits']} | **Category:** {course_data['type']}")
    with head_col2:
        if st.button("✖ Close / Dashboard", use_container_width=True, type="primary"):
            navigate_to('home')
            st.rerun()
            
    st.divider()
    
    # Two-Column Layout: Course Content Explorer (Left) | Interactive AI Tutor (Right)
    left_panel, right_panel = st.columns([1.6, 1.0])
    
    with left_panel:
        st.subheader("📖 Course Units & Minute Topics")
        st.info("💡 **Instructions:** Click on any Unit to expand it, then click a **Topic CTA button** to load detailed proofs, practice problems, and video links below!")
        
        # Render all units for selected course
        for unit_title, unit_info in course_data['units'].items():
            is_expanded = (unit_title == st.session_state.selected_unit)
            
            with st.expander(f"📚 {unit_title}", expanded=is_expanded):
                st.markdown("**Click a topic to explore:**")
                
                # Render Topic CTAs as interactive buttons
                btn_cols = st.columns(len(unit_info['topics']))
                for t_idx, topic_name in enumerate(unit_info['topics']):
                    btn_type = "primary" if (topic_name == st.session_state.selected_topic) else "secondary"
                    if btn_cols[t_idx % len(btn_cols)].button(topic_name, key=f"btn_t_{unit_title}_{t_idx}", type=btn_type):
                        st.session_state.selected_unit = unit_title
                        st.session_state.selected_topic = topic_name
                        st.rerun()
                
                st.divider()
                st.caption(f"Unit Summary: {unit_info['summary']}")

        # Render Active Topic Details Box
        if st.session_state.selected_unit and st.session_state.selected_topic:
            active_unit = course_data['units'][st.session_state.selected_unit]
            
            st.markdown("---")
            st.markdown(f"### 🔍 Detailed Focus: `{st.session_state.selected_topic}`")
            
            tab_summary, tab_proofs, tab_assign, tab_video = st.tabs([
                "📝 Concept Summary", 
                "🧮 Proofs & Formulas", 
                "🎯 Lab & Practice Assignments", 
                "🔗 Curated Video Resource"
            ])
            
            with tab_summary:
                st.write(active_unit['summary'])
                st.markdown(f"**Unit Context:** {st.session_state.selected_unit}")
                
            with tab_proofs:
                st.success(active_unit['proofs'])
                
            with tab_assign:
                st.markdown("**Recommended Self-Assessment Problems:**")
                for task in active_unit['assignments']:
                    st.write(f"- {task}")
                    
            with tab_video:
                st.markdown("Click below to open the top-ranked academic video tutorial for this specific module:")
                st.markdown(f"👉 [**Open Official Video Tutorial for {st.session_state.selected_topic}**]({active_unit['resource_url']})")

    # Right Panel: Responsive AI Tutor
    with right_panel:
        st.subheader("💬 AI Tutor (Gemini Powered)")
        active_t = st.session_state.selected_topic or "General Syllabus"
        st.caption(f"Bound Context: **{course_data['title']}** $\\rightarrow$ `{active_t}`")
        
        # Chat Display Container
        chat_box = st.container(height=420)
        with chat_box:
            if not st.session_state.chat_history:
                st.chat_message("assistant").markdown(f"Hello Ayush! I am your AI Tutor for **{course_data['title']}**. Ask me anything about `{active_t}`, request mathematical derivations, or ask for real-world engineering examples!")
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    
        # Chat Input Bar
        if user_query := st.chat_input(f"Ask about {active_t}..."):
            # Append User Query
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            # Generate AI Response
            sys_ctx = f"Course: {course_data['title']}, Unit: {st.session_state.selected_unit}, Active Topic: {st.session_state.selected_topic}"
            ai_reply = get_gemini_response(gemini_key, user_query, system_context=sys_ctx)
            
            st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})
            st.rerun()