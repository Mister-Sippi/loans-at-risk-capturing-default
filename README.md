# LOANS AT RISK: CAPTURING DEFAULT

**Full Report (PDF):** [View Report](./reports/loans_at_risk_capturing_default_report.pdf)

---

## Table of Contents

- [Project Background](#project-background)
- [Problem &amp; Evaluation Framework](#problem--evaluation-framework)
- [Executive Summary](#executive-summary)
- [Data &amp; Constraints](#data--constraints)
- [Data Governance](#data-governance)
- [Approach](#approach)
- [Results](#results)
- [Implications](#implications)
- [Repository Structure](#repository-structure)
- [Setup](#setup)

---

## Project Background

This project grew out of a curiosity about how lending systems actually work, how risk gets priced, how capital gets directed, and whether the mechanisms doing that job are as precise as they appear. This led to the question: Can lending decisions be improved using only the information available at the moment a borrower applies—and can this outperform a lender's internal risk grading system? This project evaluates that question using historical data from LendingClub, a U.S.-based consumer lender that originally operated as a marketplace platform. Borrowers submitted loan applications, while capital was provided by retail and institutional investors who selected loans to fund. Loans were originated through a partner bank and then made available to investors, meaning that LendingClub primarily intermediated external capital rather than taking credit risk on its own balance sheet. Within this structure, LendingClub assigned internal risk grades that determined both loan pricing and how investor capital was allocated across borrowers. The grading system therefore functioned not only as a risk assessment tool, but as a mechanism for directing capital under uncertainty. This context is central to the analysis. The project does not evaluate prediction in isolation, but whether application-time information can be structured in a way that leads to **better capital allocation decisions** than the platform's existing grading system.

---

## Problem & Evaluation Framework

The objective is to evaluate whether borrower-level information available at the time of application can improve lending decisions relative to LendingClub's underwriting system. The focus is not prediction in isolation, but whether improved risk ordering translates into better capital allocation. The analysis is conducted under realistic constraints. Only issued loans are observed, as rejected applications could not be retrieved and do not have realized outcomes. The evaluation therefore operates within LendingClub's historical acceptance boundary. All features are restricted to application-time information, ensuring that the model uses the same information set available during underwriting. To evaluate decisions, a constrained economic proxy is used. Predictions are mapped back to individual loans using a stable row identifier, allowing decisions to be aligned with realized outcomes and loan amounts. Accepted defaults represent loss exposure, while rejected performing loans represent opportunity cost. Outcomes are aggregated across the loan portfolio and compared across decision thresholds. These results reflect relative differences in allocation efficiency rather than full profitability. A full economic model is not constructed. Interest income, recovery rates, funding costs, and timing of cash flows are not consistently observable in the available data and would require additional assumptions. The proxy framework therefore prioritizes comparability and avoids introducing unverified modeling assumptions. Results should be interpreted as directional rather than as complete measures of profitability.

---

## Executive Summary

This project evaluates whether borrower-level information available at application can improve lending decisions relative to a lender's internal grading system. A gradient boosting model (CatBoost) improves borrower risk ranking (**AUC 0.723** vs **0.692**), but the impact of this improvement depends entirely on how decisions are made. Three operating regimes emerge:

- **Low acceptance (~37%)** Both the model and the baseline produce negative outcomes. The model reduces losses by **USD ~27 million** by rejecting more high-risk borrowers, but does not create profitability.
- **Mid-range acceptance (~66–84%)** This is where the model matters. In this region, borrower risk is most uncertain and decisions are most sensitive to ranking. The model improves outcomes by **USD ~11–15 million** by reducing default exposure while retaining more performing loans.
- **High acceptance (>~85%)**
  Differences diminish as most borrowers are accepted. Improved ranking has limited effect.

The key result is that the model does not introduce new predictive information. It reorganizes existing borrower data into a structure that enables more precise control over lending decisions. Its value is concentrated in the mid-risk region, where small improvements in ranking translate into meaningful differences in capital allocation. This implies that model deployment is not primarily a modeling problem, but a **policy problem**. The model provides a mechanism to adjust risk tolerance and manage trade-offs between default exposure and lending volume, but its effectiveness depends on selecting an appropriate operating threshold.

---

## Data & Constraints

The analysis uses anonymized historical LendingClub data containing borrower characteristics, loan attributes, and realized outcomes for issued loans. The dataset reflects actual lending decisions and therefore represents a constrained view of the applicant population. Only issued loans are available. Rejected applications could not be retrieved and do not have observable outcomes, limiting the analysis to LendingClub's historical acceptance boundary. This restricts conclusions to relative improvements within accepted loans rather than across the full applicant pool. The dataset spans multiple origination vintages with varying reporting quality. Some variables exhibit systematic missingness patterns across vintages. These patterns align with changes in LendingClub's reporting practices over time rather than underlying borrower behavior. Missingness in these variables is therefore interpreted as a reporting artifact, not as an indicator of credit risk. A temporal split is applied (**2007–2014** for **training**, **2015** for **testing**) to preserve a forward-looking evaluation structure and reflect how models would perform on new vintages. A small subset of issued loans is labeled as not meeting LendingClub's credit policy but was funded nonetheless. Although limited in number, these loans indicate that the underwriting boundary is not strictly enforced and provide an additional view on how risk is handled at the margin.

---

## Data Governance

The feature space is explicitly governed to align with the application-time constraint and avoid leakage or unintended bias. Variables not observable at origination are removed, forward-looking variables are excluded, and LendingClub outputs are retained only as benchmarks. Missingness in credit timing variables is treated as signal rather than discarded where it reflects borrower behavior, while variables affected by reporting changes are not interpreted as risk indicators. Geographic variables are removed to reduce the risk of proxy discrimination.

| Category   | Description                                      | Decision Impact                               |
| ---------- | ------------------------------------------------ | --------------------------------------------- |
| Dropped    | Not available at application                     | Prevents leakage                              |
| Excluded   | Forward-looking or post-loan variables           | Maintains temporal validity                   |
| Benchmark  | LendingClub outputs (e.g., grade, interest rate) | Enables comparison with underwriting system   |
| Recency    | Missingness encoded as signal                    | Preserves behavioral information              |
| Geographic | Location-based variables removed                 | Reduces risk of discrimination and proxy bias |

---

## Approach

The analysis is structured as a controlled evaluation of whether application-time information can support better lending decisions under realistic constraints. The design separates three distinct problems:

1. **Data validity**
   The dataset is validated to ensure that repayment outcomes are well-defined, that the temporal structure supports out-of-sample evaluation, and that reporting artifacts are identified rather than misinterpreted as borrower signal.
2. **Predictive structure**
   Submission-time features are evaluated to determine whether they contain systematic and economically plausible differences in default risk. Multiple model classes are compared to assess whether increased flexibility reveals additional signal or reorganizes existing information.
3. **Decision impact**
   Model outputs are translated into acceptance policies using probability thresholds and evaluated against the existing grading system. Outcomes are assessed in terms of error structure and exposure rather than classification metrics alone.

This structure enforces the application-time constraint, prevents information leakage, and ensures that model performance is evaluated in terms of its effect on lending decisions rather than in isolation.

---

## Results

The model improves capital allocation primarily in the mid-risk region, where LendingClub's grading system is least precise. The dataset provides a stable and interpretable representation of the lending process. Borrower characteristics available at application show consistent relationships with default outcomes, and default risk can be predicted with meaningful but limited precision. Risk ordering is further examined by comparing LendingClub's pricing and policy decisions to observed default outcomes. While higher grades and subgrades are associated with higher default rates, the relationship is not fully consistent. Variation in default outcomes within subgrades is not always matched by differences in interest rates, indicating that pricing does not fully reflect differences in borrower risk at this level. A small subset of loans labeled as not meeting LendingClub's credit policy but issued anyway is also examined. Although limited in number, these loans indicate that the underwriting boundary is not strictly enforced and provide an additional test of how risk is handled at the margin. Model performance improves with flexibility, but gains are inconsistent rather than structural. This suggests that remaining prediction error is driven by information not observable at application rather than by model limitations. The decision analysis shows that improved ranking matters most where borrower risk is uncertain and decisions are marginal. In these regions, better ordering leads to improved allocation outcomes. Outside these regions, the impact diminishes.

---

## Implications

The model does not create new information about borrower risk. It reorganizes existing information into a structure that allows more precise control over lending decisions. Its value is therefore not in prediction alone, but in how it changes decisions under uncertainty. The main benefit emerges in the mid-risk region, where borrower density is highest and decisions are most sensitive to ranking. In this setting, improved ordering directly affects which loans are accepted or rejected, leading to measurable differences in capital allocation. Outside this region, the impact is limited. When acceptance is very low, both strategies lose money and the model mainly reduces losses. When acceptance is very high, most borrowers are accepted and improved ranking has little effect. This implies that model deployment is a policy decision rather than a technical one. The model provides a mechanism to adjust risk tolerance and control trade-offs between default exposure and lending volume, but it does not remove uncertainty. Its usefulness depends on selecting and maintaining an operating threshold that reflects the desired balance between risk and opportunity.

---

## Repository Structure

- `notebooks/` – ETL, EDA, Modeling, and Validation stages
- `src/` – modules for data preprocessing, feature engineering, modeling, and plotting
- `data/` – raw, interim, and processed datasets
- `artifacts/` – figures, tables, and evaluation outputs
- `logs/` – execution logs

---

### Setup

This project requires Python 3.10+.

1. Create a virtual environment:
   python -m venv .venv

2. Activate the environment:
   source .venv/bin/activate  (Mac/Linux)
   .venv\Scripts\activate     (Windows)

3. Install dependencies:

```bash
pip install -r requirements.txt
```

Key libraries include:

pandas, numpy
scikit-learn
catboost
matplotlib, plotly

Run notebooks in order:

1. ETL
2. EDA
3. Modeling
4. Validation
