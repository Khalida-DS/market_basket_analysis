# Your Target Role Context
Senior DS roles (especially with PL-300, DP-100, AI-102) expect you to demonstrate:

* Production-thinking code, not notebook scripts
* Business impact, not just model accuracy
* End-to-end ownership — data → model → deployment → monitoring

## Step-by-Step Plan for THIS Project
### Step 1 — Refactor Code Quality (Day 1–2)
This is the most visible signal of seniority. We'll transform the flat script into a proper class-based pipeline.
What we'll do:

* Create a MarketBasketAnalyzer class
* Replace all iterrows() loops with vectorized operations
* Add type hints, docstrings, logging instead of print()
* Create a config.py for all thresholds
* Add requirements.txt

Why it matters to interviewers: They will read your code. This alone separates junior from senior.


### Step  2 — Fix the Analysis Bugs & Deepen EDA (Day 2–3)
There are real issues in the current code we need to fix first.
What we'll do:

* Fix the off-by-one bug in poplr_item
* Fix the customer_id groupby (currently sums wrong columns)
* Add proper item co-occurrence heatmap
* Add category-level analysis (not just item-level)

Why it matters: A senior DS catches their own bugs before presenting to stakeholders.

### Step 3— Strengthen the Association Rules Analysis (Day 3–4)
Go beyond running Apriori and printing top 5 rules.
What we'll do:

* Add redundant rule filtering
* Add a rule scatter plot: Lift vs. Confidence, colored by Zhang's metric — this is the standard senior visualization
* Add bootstrapped confidence intervals on top rules to test stability
* Add segment-level rules (e.g., high-frequency buyers vs. one-time buyers)

Why it matters: Zhang's metric is already smart — now we make the statistical rigor visible.

### Step 4 — Build a Recommendation Engine (Day 4–5)
Turn the rules into something that actually does something.
What we'll do:

* Build a recommend(basket: list) → list function
* Given a customer's current basket, return top-N product recommendations with confidence scores
* Handle edge cases (unknown items, empty basket, no rules match)
* Save rules to SQLite to simulate a production database lookup

Why it matters: This is the bridge between "I ran an algorithm" and "I built a product."


### Step 5 — Streamlit Dashboard (Day 5–7)
You already mention it in your project — now actually build it.
What we'll do:

* Page 1: EDA overview (basket size distribution, top items, heatmap)
* Page 2: Rule Explorer (filter by confidence, lift, Zhang — interactive table)
* Page 3: Live Recommender (user picks items → see recommendations in real time)

Why it matters: PL-300 + a Streamlit app shows you can deliver insights to non-technical stakeholders — a core senior DS expectation.

### Step 6 — Documentation & Business Framing (Day 7)
What we'll do:

Write a proper README with problem statement, methodology, results, and business impact
Add a one-page "executive summary" PDF output
Quantify impact with assumptions: "If 5% of rule-triggered transactions result in one additional item at average basket value of $X, projected annual uplift = $Y"

Why it matters: Senior DS roles require you to communicate to business leaders, not just engineers.


## Recommended Order to Work On This Project

1. Step 1 (refactor) — do this first, it changes everything
2. Step 3 (rule analysis) — this is the analytical heart
3. Step 4 (recommender) — this is the engineering proof
4. Step 5 (Streamlit) — this is the portfolio showcase
5. Step 2 (EDA fixes) — polish during step 4–5
6. Step 6 (docs) — always last


## Phase 0: The Senior DS Mindset Before Touching Code
A junior DS opens a notebook and starts coding.
A senior DS asks three questions first:

1. What is the folder structure? (Would a new team member understand this in 5 minutes?)
2. How will I track changes? (Git from day one, not after)
3. How will others run this? (Dependencies, environment, config — all documented)

### Step 1: Professional Folder Structure
This is the industry-standard structure used in real DS teams:
```
market_basket_analysis/
│
├── data/
│   ├── raw/                  # Original, untouched data files
│   └── processed/            # Cleaned, transformed data
│
├── notebooks/
│   └── 01_exploration.ipynb  # Only for exploration, never production logic
│
├── src/                      # All reusable Python code lives here
│   ├── __init__.py
│   ├── config.py             # All thresholds, paths, constants
│   ├── data_loader.py        # Data loading & validation
│   ├── preprocessor.py       # Cleaning & transformation
│   ├── analyzer.py           # Apriori + Zhang's metric logic
│   ├── recommender.py        # Recommendation engine
│   └── visualizer.py         # All chart functions
│
├── tests/                    # Unit tests (senior DS writes tests)
│   ├── test_preprocessor.py
│   └── test_recommender.py
│
├── dashboard/
│   └── app.py                # Streamlit app
│
├── outputs/
│   ├── rules/                # Saved association rules
│   └── figures/              # Saved charts
│
├── .gitignore                # What NOT to push to GitHub
├── requirements.txt          # All dependencies with versions
├── README.md                 # Project documentation
└── main.py                   # Entry point — runs the full pipeline

```

The key rule: notebooks/ is for thinking. src/ is for building. Never put business logic inside a notebook.
### Step 2: Set Up Git Locally First
Open your terminal (or VS Code terminal) and do this step by step.
#### 2.1  — Check Git is installed

```
git --version
```
#### 2.2 — Configure your identity (one time only)

```
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

#### 2.3 — Create the project folder and initialize Git

```
mkdir market_basket_analysis
cd market_basket_analysis
git init
```
git init creates a hidden .git folder — this is your local version history.

#### 2.4 — Create the folder structure

```

mkdir -p data/raw data/processed notebooks src tests dashboard outputs/rules outputs/figures
touch src/__init__.py
touch src/config.py src/data_loader.py src/preprocessor.py src/analyzer.py src/recommender.py src/visualizer.py
touch tests/test_preprocessor.py tests/test_recommender.py
touch dashboard/app.py main.py requirements.txt README.md .gitignore
```

### Step 3: Set Up Your .gitignore
This tells Git what to never track. Very important.

Creating .gitignore for a senior DS Python project
.gitignore
Done

### Step 4: Set Up Your Python Virtual Environment

A senior DS never installs packages globally. Always use a virtual environment per project.
```
# Inside your project folder
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# You'll see (venv) in your terminal — this means it's active
```

 Why this matters: If you install pandas 2.0 for one project and pandas 1.5 for another, they'd conflict globally. Virtual environments isolate each project's dependencies completely.

### Step 5: requirements.txt

Never just write pandas — always pin versions. This ensures anyone can reproduce your exact environment.

```
# Install everything in one command
pip install -r requirements.txt
```
### Step 6: Your First README.md
This is what recruiters and hiring managers read first on GitHub.

### Step 7: Your Git Commit Workflow
This is how senior engineers commit code. Every commit tells a story.
The commit message convention (Conventional Commits):

```
type: short description

Types:
feat:     a new feature
fix:      a bug fix
refactor: code restructure, no behavior change
docs:     documentation only
test:     adding tests
chore:    setup, config, dependencies
```

The first commit

```
# See what's untracked
git status

# Stage everything
git add .

# Your first commit
git commit -m "chore: initialize project structure and configuration"
```

Going forward, commit after each logical step:

```

git commit -m "feat: add data loader with validation"
git commit -m "feat: add vectorized preprocessor replacing iterrows loops"
git commit -m "fix: correct off-by-one in item frequency calculation"
git commit -m "feat: add Zhang metric filtering to association rules"
git commit -m "feat: add recommendation engine with SQLite persistence"
git commit -m "feat: add Streamlit dashboard with rule explorer"
git commit -m "test: add unit tests for preprocessor and recommender"
git commit -m "docs: update README with results and usage instructions"
```

Rule: Never commit broken code. Never commit with message "update" or "fix stuff."

### Step 8: Push to GitHub

```
# 1. Go to github.com → New Repository
# Name: market_basket_analysis
# Description: End-to-end Market Basket Analysis with Apriori + Zhang's Metric
# Set to Public (portfolio visibility)
# Do NOT initialize with README (you already have one)

# 2. Connect your local repo to GitHub
git remote add origin https://github.com/YOUR_USERNAME/market_basket_analysis.git

# 3. Rename default branch to 'main' (modern standard)
git branch -M main

# 4. Push
git push -u origin main
```

### Step 9: Branching Strategy
A senior DS never works directly on main. Here's the workflow:

```

# Create a branch for each feature
git checkout -b feat/data-loader
# ... write code ...
git add .
git commit -m "feat: add data loader with schema validation"
git push origin feat/data-loader

# When done, merge back to main via Pull Request on GitHub
# Then locally:
git checkout main
git pull origin main
git branch -d feat/data-loader  # delete the branch locally
```

##### Branch naming convention:

feat/recommender-engine
fix/item-frequency-bug
refactor/vectorize-preprocessor
docs/update-readme