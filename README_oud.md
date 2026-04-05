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

# LOANS AT RISK: CAPTURING DEFAULT

## Table of Contents

- [Project Background](#project-background)
- [Problem &amp; Evaluation Framework](#problem--evaluation-framework)
- [Executive Summary](#executive-summary)
- [Data &amp; Constraints](#data--constraints)
- [Data Governance](#data-governance)
- [Approach](#approach)
- [Results](#results)
  - [Risk Structure](#risk-structure)
  - [Model Performance](#model-performance)
  - [Decision Impact](#decision-impact)
- [What This Means](#what-this-means)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Full Report](#full-report)

## Project Background

This project focuses on credit risk, where borrower-level information is used to assess the likelihood of default and support lending decisions. LendingClub is a U.S.-based lender that originally operated as a peer-to-peer marketplace, connecting individual borrowers with investors. Founded in 2006, the platform facilitated unsecured personal loans and performed underwriting, assigned risk grades, and set loan terms such as interest rates. This marketplace model was its primary mode of operation from its founding through the late 2010s, during which it became one of the largest online consumer lending platforms in the United States. In 2020, LendingClub transitioned to a bank-based model following its acquisition of Radius Bank. Since then, it operates as a digital bank, originating and holding loans on its own balance sheet. The analysis in this project is based on anonymized historical loan data released by LendingClub.

## Problem & Evaluation Framework

The core problem is to evaluate whether borrower-level information available at the time of application is sufficient to improve lending decisions relative to LendingClub’s underwriting system. Specifically, the analysis tests whether a model built on application-time features can better distinguish between borrowers who will repay and those who will default, and whether this improved ranking translates into more effective capital allocation. The evaluation is conducted under realistic decision constraints. Only issued loans are observed, as rejected applications could not be retrieved from the available data sources. In addition, rejected applications do not have realized repayment outcomes. As a result, the analysis reflects improvements within LendingClub’s historical acceptance boundary rather than across the full applicant population. All predictions are restricted to information available at the moment of application. No post-loan or repayment-related variables are used, ensuring that the model operates under the same informational conditions as real underwriting decisions. To translate model outputs into decision outcomes, a constrained proxy economic framework is used. Predictions are mapped back to individual loans using a stable row identifier, allowing acceptance decisions to be aligned with realized outcomes and associated loan amounts. Accepted defaulting loans are treated as loss exposure, while rejected performing loans represent opportunity cost, both proxied using loan amounts. Outcomes are aggregated across the portfolio to compare decision policies under varying acceptance thresholds. The resulting values reflect relative differences in capital allocation between strategies and should be interpreted as directional measures of allocation efficiency rather than a complete representation of lending profitability. The model is evaluated in terms of its impact on lending decisions, not just predictive performance.

## Executive Summary

This analysis shows that borrower-level information available at the time of application can improve lending decisions relative to LendingClub’s underwriting system, but only under specific operating conditions. A gradient boosting model (CatBoost) trained on application-time features improves borrower risk ranking across the full risk spectrum (**AUC 0.723 vs 0.692**), indicating more consistent separation between borrowers who repay and those who default. When translated into decisions, this improved ranking produces economically meaningful effects that depend on the chosen acceptance threshold.

- **Low acceptance (~37%)**: Both the model and the baseline generate negative outcomes. The model reduces losses by approximately **€27M** relative to LendingClub’s underwriting system, reflecting improved rejection of high-risk borrowers rather than profitability.
- **Mid-range acceptance (~66–84%)**: The model consistently outperforms the baseline, delivering its strongest impact in this region. This region aligns with the mid-risk segment of the portfolio, where borrower density is highest and decisions are most uncertain. In this setting, improvements in ranking within subgrades become materially more valuable, as small differences in ordering directly affect acceptance decisions. The model improves proxy economic outcomes by approximately **€11–15M** by better distinguishing marginal borrowers, reducing exposure to defaulting loans while retaining a higher share of performing loans.
- **High acceptance (>~85%)**: Differences between strategies diminish or may reverse as most borrowers are accepted and selection becomes less binding. In this regime, lending volume dominates risk differentiation, limiting the value of improved ranking.

The model does not introduce fundamentally new predictive signal but reorganizes existing information into a structure that enables more precise and controllable capital allocation. Its value is conditional: it improves outcomes where decisions are sensitive to ranking precision, but provides limited advantage when acceptance policies are either highly restrictive or highly permissive.

## Data & Constraints

The analysis is based on anonymized historical loan data released by LendingClub, containing borrower characteristics, loan attributes, and realized repayment outcomes for issued loans. The data reflects actual lending decisions made on the platform and therefore represents a real-world underwriting environment rather than a simulated setting. Only issued loans are observed, as rejected applications could not be retrieved from the available data sources. In addition, rejected applications do not have realized repayment outcomes. As a result, the analysis is conducted within LendingClub's historical acceptance boundary and does not represent the full applicant population. All features are restricted to information available at the time of application, including borrower characteristics such as income, debt-to-income ratio, credit history, and loan request details. Variables that capture post-origination behavior or loan performance are excluded to ensure that the analysis reflects the informational constraints of real underwriting decisions. The evaluation follows a temporal split, with loans issued between 2007–2014 used for model development and loans from 2015 used for out-of-sample testing, preserving the forward-looking nature of the problem and avoiding information leakage across time. Some variables exhibit systematic reporting differences across vintages, particularly in earlier years where certain financial fields are missing or inconsistently recorded. These patterns are treated as reporting artifacts rather than borrower signal and are handled accordingly during preprocessing. The feature space is governed to enforce these constraints and ensure consistency across datasets, as summarized below.

| Category   | Description                                      | Decision Impact                               |
| ---------- | ------------------------------------------------ | --------------------------------------------- |
| Dropped    | Not available at application                     | Prevents leakage                              |
| Excluded   | Forward-looking or post-loan variables           | Maintains temporal validity                   |
| Benchmark  | LendingClub outputs (e.g., grade, interest rate) | Enables comparison with underwriting system   |
| Recency    | Missingness encoded as signal                    | Preserves behavioral information              |
| Geographic | Location-based variables removed                 | Reduces risk of discrimination and proxy bias |

