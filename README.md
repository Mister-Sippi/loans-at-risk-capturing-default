# Loans at Risk: Capturing Default

### Predicting Loan Default at Origination

In this project, the primary modeling goal is to predict whether a borrower is likely to default **at the time of loan application**, using only information available at origination (income, debt-to-income ratio, credit history, loan amount, employment, and other application-time features).

Predicting default at this stage provides actionable insights for financial institutions. Rather than simply issuing a “yes/no” decision, these predictions inform **risk-based lending and pricing decisions**. For example, a borrower identified as high-risk might be offered a smaller loan, additional collateral, or financial counseling, while low-risk applicants can receive competitive terms.

Interpretability is critical. Decisions cannot rely solely on a black-box model; the most influential factors — such as high debt-to-income ratio or limited credit history — must be communicated clearly. This ensures that lending decisions are **defensible, transparent, and aligned with regulatory requirements**, while also allowing financial institutions to optimize portfolio performance and manage credit risk proactively.

### Research Question

How well did LendingClub's internal grading system capture borrower risk, and can a model built using only information available at the time of application match or improve that classification?

1. Does LendingClub's grading system successfully separate borrowers into groups with different realized default risk?
2. Do borrower characteristics available at the time of application show systematic differences in realized default outcomes?

### Dataset

The raw LendingClub loan dataset is publicly available for download [here](https://www.lendingclub.com/info/download-data.action). For this project:

- **Training data:** Loans issued from **2007–2014**, which I clean and preprocess myself.
- **Test data:** Loans issued in **2015**, kept separate to evaluate model performance.
- **Raw data only:** No preprocessed LendingClub datasets are used — all cleaning and transformation are performed as part of this project to demonstrate end-to-end reproducibility.
- **Data dictionary included:** I reference the LendingClub data dictionary to understand each variable, identify post-loan features, and flag columns that could introduce leakage.


Dataset source: LendingClub historical loan data (2007–2018)

The dataset originates from LendingClub's historical data releases.
Because the original LendingClub download portal is no longer publicly accessible,
the files used in this project were obtained from a preserved mirror of the
original LendingClub CSV releases.

Dataset: LendingClub Historical Loan Data (2007–2015)

The dataset originates from LendingClub’s historical loan data releases.
The files were obtained from a public mirror of the original dataset.

Files included in the download:

- loan_data_2007-2014.csv
- loan_data_2015.csv
- LCDataDictionary.xlsx
- loan_data_2007-2014_preprocessed.csv

The preprocessed dataset was not used in this project. All data preparation
and feature construction were performed independently in the ETL pipeline
to ensure transparency and reproducibility.

https://web.archive.org/web/20160911194705/https://www.lendingclub.com/info/download-data.action

SOURCE: https://web.archive.org/web/20160911194705/https://www.lendingclub.com/info/download-data.action

### Rejected Loan Applications

LendingClub also provides datasets containing rejected loan applications.
These records were not included in the analysis.

Rejected applications contain information about borrowers who were screened
out during the underwriting process. As such, they can provide insight into
LendingClub’s acceptance criteria and risk tolerance.

However, rejected loans do not have observed repayment outcomes because the
loans were never issued. This means they cannot be used to estimate default
risk directly or to evaluate predictive models.

Including rejected applications would therefore introduce a fundamental
limitation: the model could not be validated against realized outcomes.

For this reason, the analysis focuses on issued loans with known repayment
outcomes. This allows the project to evaluate whether borrower information
available at the time of application can reproduce or improve LendingClub’s
risk classification.

The rejected application dataset remains valuable for studying underwriting
policies and acceptance boundaries, but incorporating it would require
additional techniques such as reject inference, which are outside the scope
of this project.

### Dataset Scope

The LendingClub historical dataset includes loan records through 2016.
However, only the first two quarters of 2016 are available in the archived
releases.

These partial-year records were not included in the analysis. Loans issued
in 2016 would not have sufficient time to reach a terminal repayment outcome
(default or full repayment) within the dataset period, which would introduce
outcome censoring.

To ensure consistent and fully observed loan outcomes, the dataset used in
this project is limited to loans issued between 2007 and 2015.

The data is then split chronologically:

- **Training dataset:** loans issued between 2007 and 2014
- **Testing dataset:** loans issued in 2015



### Raw data files

The archived LendingClub dataset was downloaded with
original filenames such as:

- LoanStats3a.csv
- LoanStats3b.csv
- LoanStats3c.csv
- LoanStats3d.csv

For clarity these files were renamed in the `data/external` directory:

| Original file   | Project filename        |
| --------------- | ----------------------- |
| LoanStats3a.csv | loan_data_2007_2011.csv |
| LoanStats3b.csv | loan_data_2012_2013.csv |
| LoanStats3c.csv | loan_data_2014.csv      |              |                         |
| LoanStats3d.csv | loan_data_2015.csv      |

### Workflow

The project follows a structured **ETL and analysis workflow**, implemented across four notebooks:

1. **Notebook 1 – ETL:** Load raw CSV data, inspect for missing values, constants, type inconsistencies, and potential leakage columns using the data dictionary; clean and transform the dataset for downstream analysis.
2. **Notebook 2 – Exploratory Data Analysis (EDA):** Investigate distributions, correlations, and relationships in the cleaned dataset to generate insights and inform feature engineering.
3. **Notebook 3 – Feature Engineering & Modeling:** Create predictive features, train and evaluate models, and interpret results to provide actionable recommendations.
4. **Notebook 4 – Validation:** Assess model performance on the test set, ensure generalizability, and validate robustness of predictions.

> Optional extension: Mid-loan default prediction could also be explored, using post-loan information such as payment history or remaining principal, to provide an **early-warning system**. This is beyond the scope of the current ETL workflow but demonstrates awareness of dynamic risk modeling opportunities.
