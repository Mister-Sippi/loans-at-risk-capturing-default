
# Loans at Risk: Capturing Default

### Predicting Loan Default at Origination

In this project, the primary modeling goal is to predict whether a borrower is likely to default **at the time of loan application**, using only information available at origination (income, debt-to-income ratio, credit history, loan amount, employment, and other application-time features).

Predicting default at this stage provides actionable insights for financial institutions. Rather than simply issuing a “yes/no” decision, these predictions inform **risk-based lending and pricing decisions**. For example, a borrower identified as high-risk might be offered a smaller loan, additional collateral, or financial counseling, while low-risk applicants can receive competitive terms.

Interpretability is critical. Decisions cannot rely solely on a black-box model; the most influential factors — such as high debt-to-income ratio or limited credit history — must be communicated clearly. This ensures that lending decisions are **defensible, transparent, and aligned with regulatory requirements**, while also allowing financial institutions to optimize portfolio performance and manage credit risk proactively.

### Dataset

The raw LendingClub loan dataset is publicly available for download [here](https://www.lendingclub.com/info/download-data.action). For this project:

- **Training data:** Loans issued from **2007–2014**, which I clean and preprocess myself.
- **Test data:** Loans issued in **2015**, kept separate to evaluate model performance.
- **Raw data only:** No preprocessed LendingClub datasets are used — all cleaning and transformation are performed as part of this project to demonstrate end-to-end reproducibility.
- **Data dictionary included:** I reference the LendingClub data dictionary to understand each variable, identify post-loan features, and flag columns that could introduce leakage.

### Workflow

The project follows a structured **ETL and analysis workflow**, implemented across four notebooks:

1. **Notebook 1 – ETL:** Load raw CSV data, inspect for missing values, constants, type inconsistencies, and potential leakage columns using the data dictionary; clean and transform the dataset for downstream analysis.
2. **Notebook 2 – Exploratory Data Analysis (EDA):** Investigate distributions, correlations, and relationships in the cleaned dataset to generate insights and inform feature engineering.
3. **Notebook 3 – Feature Engineering & Modeling:** Create predictive features, train and evaluate models, and interpret results to provide actionable recommendations.
4. **Notebook 4 – Validation:** Assess model performance on the test set, ensure generalizability, and validate robustness of predictions.

> Optional extension: Mid-loan default prediction could also be explored, using post-loan information such as payment history or remaining principal, to provide an **early-warning system**. This is beyond the scope of the current ETL workflow but demonstrates awareness of dynamic risk modeling opportunities.
>
