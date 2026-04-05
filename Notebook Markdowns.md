# Loans at Risk: Capturing Default — Extract · Transform · Load

## Purpose

This notebook implements the ETL layer of the LendingClub default prediction project.  
The training window covers **2007–2014**; the test window covers **2015**.

The objective is not analysis. It is structural control. Raw CSV extracts are converted into datasets that are suitable for downstream modeling under a clearly defined constraint: prediction is made at the moment of application submission.“Structurally reliable” in this context means:

- Identical schema across train and test  
- Explicit treatment of datatypes and null structure  
- Removal of null-only and constant columns  
- Elimination of export artifacts  
- Early enforcement of the submission-time prediction boundary  

No interpretation is performed here. This notebook establishes a clean, governed input layer so that later analytical steps operate on stable ground.
All dependency management is handled at repository level (`README`, `requirements.txt`). This notebook assumes the environment is already configured.

---

## Extract

### Raw Source Normalization

The archived LendingClub CSV releases contain several non-loan rows that are
included as part of the original export format. These rows are not loan-level
records and must be removed before the files can be treated as tabular data.

Typical examples include:

- Prospectus or informational text rows at the top of the file  
- Repeated header rows embedded within the dataset  
- Footer summary rows such as  
  *“Total amount funded in policy code 1”* and  
  *“Total amount funded in policy code 2”*

These rows are artifacts of the original export format rather than actual
observations. They are removed during raw ingestion before any schema
validation or transformation logic is applied.

After normalization, the source files are concatenated into two
canonical raw datasets:

- **2007–2014** → training dataset  
- **2015** → testing dataset  

The resulting files contain loan-level records only and serve as the raw input
layer for the ETL pipeline. 

Extraction preserves the raw state of the data (after source normalization). No filtering, type coercion, or feature selection occurs at this stage. The raw layer is treated as immutable input.

---

## Transform

The transformation phase enforces structural and temporal discipline.

Concretely:

- Schema and datatype validation  
- Identification and removal of null-only and constant columns  
- Detection of mixed types and object-encoded numerics  
- Removal of structural artifacts (e.g., exported index columns)  
- Explicit exclusion of post-origination, underwriting, and pricing variables  
- Deterministic type conversions (including numeric and date normalization)  
- Structured handling of credit-event timing variables, where null represents absence rather than data loss  
- Identical transformation logic across training and test sets  

No feature engineering beyond structural normalization is performed.  
No modeling assumptions are introduced.

The output of this phase is a submission-time compliant dataset.

---

## Load

The transformed datasets are persisted in Parquet format. Two layers are materialized:

- **Clean dataset** → structurally aligned variables  
- **Feature-base dataset** → submission-time eligible variables plus target  

Schema identity between training and test is validated prior to persistence.
These outputs form the controlled input layer for EDA and modeling.

---

## Scope Boundary

Feature engineering, modeling, evaluation, and decision logic are handled in subsequent notebooks.
This notebook exists to ensure that those stages operate on data that is structurally sound, temporally consistent, and reproducible.

---

## Raw Source Normalization

Before we can use the dataset we need to make sure that we actually deal with structural data and that all files share the same schemas. 

Source inspection confirms that all LendingClub export files share the same
non-tabular artifacts and the same header schema. Normalization can therefore
be applied consistently across all source files before concatenation.

## Raw Dataset Construction

The LendingClub data is distributed as several CSV archives covering different time periods. These files are not strict tabular exports. They contain additional rows such as prospectus notices, blank lines, summary totals, and occasional internal headers. Before the data can be treated as a dataset, these structural artifacts must be removed.

### Source normalization

Each archive file is processed line-by-line and written to a staging dataset. Rows that do not belong to the tabular structure are excluded, leaving only loan-level observations with the original column layout preserved. The normalized files are written to the **staging layer**.

### Schema validation

After normalization the column headers of all staged files are compared to verify that the schema is identical across archives. This ensures the files can be combined without structural alignment.

### Raw dataset consolidation

The normalized archives are then consolidated into two canonical datasets.

The project uses a temporal split:

- **2007–2014** → training period  
- **2015** → testing period

The training dataset is created by concatenating the staged files for:

- 2007–2011  
- 2012–2013  
- 2014  

The resulting files are written to the raw data layer:

- `raw_loan_data_2007_2014.csv`  
- `raw_loan_data_2015.csv`

From this point forward all downstream processing operates on these canonical raw datasets rather than on the original archive exports.

# Raw Layer – Structural Governance and Eligibility

The data is provided as two independent source files:

- **2007–2014 vintage (training)**
- **2015 vintage (testing)**

The training vintage defines the admissible feature universe.  
The testing vintage is aligned to it.

> **Note:** Governance decisions below are based on the full audit outputs stored in `artifacts/audit` and the LendingClub Data Dictionary accompanying the dataset. Variable eligibility and lifecycle interpretation follow the documented field definitions.

---

# 1. Initial Data Inspection – Structural Baseline

| Metric | Train (2007–2014) | Test (2015) |
|------|------------------|-------------|
| Rows | **466,287** | **421,095** |
| Columns | **111** | **111** |
| Memory Usage | ~580 MB | ~520 MB |
| Numeric Columns | 87 | 86 |
| Object / String Columns | 24 | 25 |
| Fully Null Columns | **17** | 0 |
| Constant Columns | 2 (`policy_code`, `application_type`) | 1 (`policy_code`) |
| String-like Columns (post dtype scan) | 24 | 25 |
| Columns present only in one split | 0 | 0 |
| Dtype mismatches | 38 | — |

The raw dataset contains **111 variables** across both partitions.

Relative to earlier dataset versions, the wider feature universe primarily reflects **expanded credit bureau reporting fields introduced in later platform vintages**.

The training vintage therefore contains a mixture of:

- stable application variables  
- stable bureau variables  
- later bureau variables with partial historical reporting  
- structural artifacts introduced during export  

Governance rules below formalize which variables remain admissible for modeling.

---

## Fully Null Variables (Training-Defined Constraint)

Seventeen variables are **entirely null in the training data**.

According to the LendingClub Data Dictionary these variables represent later-introduced fields including:

- short-horizon account opening activity windows  
- segmented installment loan utilization measures  
- joint borrower income and verification attributes  
- segmented inquiry counters  

These variables did not exist during the training period. Although a small fraction of observations appear in the testing vintage, they remain **94–99% null**, indicating partial late-period reporting rather than stable availability. A model trained on the 2007–2014 regime cannot estimate variables that did not exist in that regime. All **17 structurally null columns** are therefore removed from both datasets.

---

## Constant Variables and Structural Artifacts

Two columns exhibit zero variance in the training data:

- `policy_code`
- `application_type`

`policy_code` is an internal LendingClub platform designation and contains no variation.

`application_type` indicates whether an application is submitted by an individual borrower or jointly.

In the **2007–2014 training data all applications are individual**.  
Joint applications appear only in the later testing vintage and rely on borrower attributes that are structurally absent from the training data. Including this variable would therefore introduce a borrower structure that never occurs in the training regime. Both variables are removed.

The export artifact `Unnamed: 0` is also removed defensively when present.  
It represents a CSV index column and does not correspond to a documented field in the LendingClub data dictionary.

---

## Credit Timing Variables

The dataset contains several variables measuring the **recency of adverse credit events**:

- `mths_since_last_delinq`
- `mths_since_last_major_derog`
- `mths_since_last_record`
- `mths_since_recent_bc_dlq`
- `mths_since_recent_revol_delinq`

These variables measure the number of months since the borrower last experienced negative credit behavior such as:

- missed payments  
- serious delinquencies  
- public records  

A value of `0` indicates the event occurred within the previous month.  
Higher values indicate events further in the past.

Missing values indicate that **no such event exists in the borrower’s recorded credit history**. High null percentages therefore represent **structural absence** rather than **incomplete reporting**. Dropping these variables would remove meaningful risk signal, while leaving them as `NaN` would treat structural absence as missing data. Handling:

1. Create a binary indicator `has_<variable>`
2. Replace missing values in the original variable with sentinel value **9999**

The sentinel lies well beyond any realistic month count and preserves ordinal interpretation.

---

## Bureau Reporting Expansion

A broader group of bureau variables exhibits **partial historical reporting**. Approximately **15% of observations in the training vintage are missing**, while the same fields are **fully populated in the 2015 testing vintage**. This pattern reflects **expansion of credit bureau reporting coverage over time**, not a transformation error. These variables remain valid predictors because the information they contain exists at application time. To separate **information content** from **reporting availability**, an additional binary indicator is created for each affected variable:

`has_<variable>`

Interpretation:

- `has_* = 0` → field not reported in that vintage  
- `has_* = 1` → value observed  

The original numeric values remain unchanged. This prevents historical reporting gaps from silently introducing a temporal signal into the model.

---

# 2. Submission-Time Boundary

Prediction is defined at the moment the borrower submits the application. The admissible information set therefore consists of:

- borrower-declared application inputs  
- credit bureau data retrieved at submission  

No later platform actions, pricing decisions, funding outcomes, or servicing updates are allowed to enter the training feature space. A variable is eligible only if:

1. It is available at submission  
2. It does not encode underwriting, pricing, funding, or servicing outcomes  
3. It contains usable observations in the training data  

---

# 3. Feature Classification Overview – Application Submission Prediction Point

| Column Group | Category | Action | Rationale |
|------|----------|--------|-----------|
| Application inputs (`loan_amnt`, `term`, `purpose`, `annual_inc`, `dti`<br>`emp_length`, `home_ownership`) | Application input | Keep | Borrower-declared attributes available at submission |
| Credit snapshot (`open_acc`, `total_acc`, `inq_last_6mths`<br>`delinq_2yrs`, `pub_rec`, `collections_12_mths_ex_med`<br>`revol_bal`, `revol_util`, `acc_now_delinq`<br>`tot_cur_bal`, `tot_coll_amt`, `tot_hi_cred_lim`<br>`total_rev_hi_lim`, `total_il_high_credit_limit`) | Credit snapshot | Keep | Credit bureau state at time of application |
| Bureau reporting expansion (`avg_cur_bal`, `mo_sin_old_rev_tl_op`, `mo_sin_rcnt_rev_tl_op`<br>`mo_sin_rcnt_tl`, `num_accts_ever_120_pd`, `num_actv_bc_tl`<br>`num_actv_rev_tl`, `num_bc_tl`, `num_il_tl`<br>`num_op_rev_tl`, `num_rev_tl_bal_gt_0`, `num_tl_30dpd`<br>`num_tl_90g_dpd_24m`, `num_tl_op_past_12m`, `num_rev_accts`) | Credit snapshot | Keep (Indicator Added) | Reporting coverage expanded historically; indicator separates information from reporting availability |
| Credit timing variables (`mths_since_last_delinq`, `mths_since_last_major_derog`, `mths_since_last_record`<br>`mths_since_recent_bc_dlq`, `mths_since_recent_revol_delinq`) | Credit timing | Keep (Transformed) | Structural absence encoded via sentinel and indicator |
| Target (`loan_status`) | Target | Target Only | Defines repayment outcome |
| Platform signals (`grade`, `sub_grade`) | Platform signal | Benchmark Only | LendingClub underwriting classification |
| Pricing outputs (`int_rate`, `installment`) | Pricing output | Benchmark Only | Determined by platform underwriting decision |
| Underwriting outcome (`verification_status`) | Underwriting outcome | Exclude | Determined during verification process |
| Funding decisions (`funded_amnt`, `funded_amnt_inv`) | Funding outcome | Exclude | Reflect investor allocation rather than borrower attributes |
| Lifecycle timestamps (`issue_d`, `initial_list_status`) | Post-submission | Exclude | Determined after application submission |
| Bureau updates (`last_credit_pull_d`) | Post-submission | Exclude | Reflect bureau refresh after origination |
| Servicing variables (`out_prncp`, `out_prncp_inv`, `total_pymnt`, `total_pymnt_inv`<br>`total_rec_prncp`, `total_rec_int`, `total_rec_late_fee`<br>`recoveries`, `collection_recovery_fee`<br>`last_pymnt_d`, `next_pymnt_d`, `pymnt_plan`) | Post-origination | Exclude | Observed only after loan issuance; would introduce outcome leakage |
| Structurally null variables (`annual_inc_joint`, `dti_joint`, `verification_status_joint`<br>`open_acc_6m`, `open_il_6m`, `open_il_12m`, `open_il_24m`<br>`mths_since_rcnt_il`, `total_bal_il`, `il_util`<br>`open_rv_12m`, `open_rv_24m`, `all_util`<br>`inq_fi`, `total_cu_tl`, `inq_last_12m`, `max_bal_bc`) | Vintage dependent | Drop | Absent in training regime |
| Identifiers (`id`, `member_id`, `url`) | Structural | Drop | Non-predictive artifacts |
| Structural variables (`policy_code`, `application_type`) | Structural | Drop | `policy_code` is constant across both datasets; `application_type` has zero variance in the training data and only varies in the testing vintage |
| Free-text (`desc`, `emp_title`, `title`) | Unstructured | Drop | High-cardinality text outside structured modeling scope |
| Geographic proxies (`addr_state`, `zip_code`) | Application input | Drop | Removed to reduce proxy discrimination risk |

---

# 4. String Column Normalization

After structural removal and boundary classification, remaining object-based variables require normalization. Normalization ensures:

- consistent representation across datasets  
- stable typing  
- removal of formatting variation without altering meaning  

Some numeric variables in the raw export are stored as percentage strings rather than numeric values. Two examples are **`int_rate`** (loan interest rate) and **`revol_util`** (revolving credit utilization).  
These appear in the source files in the form `"13.56%"`.

During transformation the percentage symbol is removed and the remaining value is converted to a numeric type so the variables can be treated as numeric quantities during analysis.

| Column | Category | Transformation Action | Training Eligibility | Rationale |
|------|----------|----------------------|---------------------|-----------|
| `term` | Structured categorical | Strip whitespace → extract numeric term (36 / 60) → rename `term_months` → convert to int | Keep | Contractual term |
| `int_rate` | Numeric (percent string) | Remove `%` → convert to float | Benchmark Only | Pricing signal produced by platform underwriting |
| `revol_util` | Numeric (percent string) | Remove `%` → convert to float | Keep | Borrower revolving credit utilization |
| `emp_length` | Ordinal categorical | Map to integer scale 0–10 (`<1` → 0, `10+` → 10) | Keep | Borrower tenure |
| `home_ownership` | Categorical | Normalize casing; map `NONE` → `OTHER` | Keep | Borrower attribute |
| `purpose` | Categorical | Normalize casing | Keep | Loan purpose |
| `grade` | Platform signal | Normalize representation | Benchmark Only | Underwriting output |
| `sub_grade` | Platform signal | Normalize representation | Benchmark Only | Granular signal |
| `verification_status` | Underwriting outcome | Normalize representation | Exclude | Determined after submission |
| `initial_list_status` | Workflow variable | Normalize representation | Exclude | Listing outcome |
| `pymnt_plan` | Servicing indicator | Map `n` → 0, `y` → 1 | Exclude | Post-origination flag |
| `issue_d` | Lifecycle timestamp | Convert to datetime | Exclude | Occurs after submission |
| `last_credit_pull_d` | Lifecycle update | Convert to datetime | Exclude | Bureau refresh |
| `last_pymnt_d` | Servicing timeline | Convert to datetime | Exclude | Post-origination |
| `next_pymnt_d` | Servicing timeline | Convert to datetime | Exclude | Post-origination |
| `loan_status` | Target | No transformation | Target Only | Outcome definition |

---

# 5. Transformation Execution

The governance rules above are implemented in the following sequence.

Execution steps:

- Create a stable internal identifier `row_id`
- Remove structurally ineligible columns
- Align testing data to the training-defined feature universe
- Apply credit timing encoding (`has_*` + sentinel `9999`)
- Create reporting indicators for bureau expansion variables
- Normalize categorical string fields and convert percent-encoded numeric variables
- Convert month-year timestamps to datetime

Outputs are persisted as:

- **`clean`** — governed dataset with benchmark variables retained
- **`feature_base`** — modeling dataset containing only training-eligible predictors and the target

---

# Notebook Conclusion – Data Cleaning and Feature Base Construction

This notebook establishes the transformation layer required for the **Loans at Risk: Capturing Default** project.

The governing constraint is that prediction must rely **only on information available at the moment a borrower submits a loan application**. All transformation decisions follow from that boundary. The raw LendingClub exports contain **111 variables** across both dataset partitions. Through structural governance and transformation, these are reduced to two controlled dataset layers:

- **clean** – a normalized dataset in which variables retain their economic meaning but are stored using consistent datatypes
- **feature_base** – the modeling dataset containing only variables admissible under the submission-time constraint

Key transformations performed in this notebook include:

- removal of structurally ineligible or non-predictive variables  
- encoding of credit timing variables to distinguish structural absence from observed events  
- separation of bureau information from historical reporting availability  
- normalization of categorical string fields  
- conversion of percent-encoded numeric variables to numeric form  
- conversion of month-year timestamps to datetime representations  

The resulting datasets maintain identical schemas across the training and testing partitions and preserve row counts throughout transformation. The **clean dataset** contains **90 variables**, while the **feature_base dataset** contains **66 variables** that satisfy the submission-time eligibility rules.

The transformation layer therefore removes three structural risks before modeling begins: information leakage from post-submission variables, misinterpretation of structurally absent credit events as missing data, and spurious temporal signals introduced by historical reporting expansion. This feature space forms the controlled input layer for the subsequent exploratory analysis and modeling stages of the project.

# Loans at Risk: Capturing Default — Exploratory Data Analysis

### Purpose

This notebook examines the LendingClub loan dataset to determine whether it provides a coherent representation of the lending process and whether the information available at loan application contains observable differences in borrower default outcomes.

Before predictive modeling can be performed, it is necessary to verify that loan outcomes are well defined, that the dataset supports a realistic temporal evaluation design, and that submission-time borrower and loan characteristics exhibit meaningful variation in default rates.

---

### Analytical Approach

The analysis proceeds in two stages.

The first stage evaluates the structural properties of the dataset using the **`clean` dataset**. This dataset retains the loan issue date (`issue_d`), allowing the analysis to examine the temporal evolution of the lending portfolio and verify that the data supports a realistic train–test evaluation design. Diagnostic checks assess loan outcome definitions, the distribution of loan issuance over time, and potential reporting shifts affecting several balance-related variables.

The second stage examines the submission-time feature space using the **`feature_base` dataset**. This dataset contains only variables observable at the moment of loan application and therefore suitable for predictive modeling. The loan issue date (`issue_d`) is excluded because it reflects the loan origination timestamp and would introduce information unavailable at prediction time.

Using this feature-base dataset, borrower credit history, leverage, repayment capacity, and loan structure variables are evaluated to determine whether they correspond to systematic differences in default rates.

---

### Scope

This notebook focuses on dataset diagnostics and submission-time feature exploration. It establishes the modeling population using realized repayment outcomes and evaluates whether borrower and loan characteristics observable at application correspond to differences in default risk.

Predictive modeling is not performed here. Model construction, performance evaluation, and comparison with LendingClub’s grading system are addressed in the subsequent stages of the project.

## Part 1 — Diagnostic Validation (`clean`)

This section evaluates outcome integrity, cohort definition, temporal structure, and reporting stability using the **clean training dataset** and **clean testing dataset**. The objective is to confirm that the outcome variable, temporal structure, and dataset composition behave coherently before any feature-level analysis begins.

The clean dataset retains the full loan record. That makes it the correct layer for validating repayment outcomes, defining the realized-outcome modeling cohort, and examining temporal diagnostics based on `issue_d`.


#### Outcome distribution by vintage

The table is more useful here than a stacked bar chart. The question is not visual composition for its own sake. The question is whether outcome composition shifts across issuance years in a way that affects cohort definition and label maturity.

Early vintages are dominated by realized repayment outcomes. Later periods contain a larger share of loans whose repayment process had not yet concluded at the moment of extraction. That difference must be handled before modeling.


#### Realized-outcome cohort definition

The modeling population is defined by excluding loans whose repayment outcome has not yet been determined. This removes censoring from the label and restricts the analysis to loans where borrower behavior is fully observed.

The resulting cohort is the relevant base for both later feature analysis and downstream modeling.


#### 1.6 Diagnostic summary

The clean dataset confirms three key properties of the lending process representation:

- 1. the outcome variable behaves coherently once non-realized repayment states are excluded. Loans that have completed their repayment cycle fall clearly into repayment or default outcomes, allowing the modeling cohort to be defined using realized borrower behavior.

- 2. the temporal structure of the dataset is consistent with the growth of the lending platform. Loan issuance expands over time while default rates remain broadly stable, supporting the evaluation design in which models are trained on earlier lending cohorts and evaluated on a later period.

- 3. several balance-related variables exhibit a reporting shift in the early years of the platform. These fields show systematic missingness during the earlier cohorts but become consistently populated in later years. This pattern reflects changes in reporting practices rather than borrower behavior and must therefore be treated as a data-quality artifact during modeling.

Taken together, these diagnostics indicate that the dataset provides a stable and interpretable observation of the lending process.

---

#### Outcome Definition and Event Balance

The modeling population is restricted to loans with a realized repayment outcome. Loans that remain active or are in intermediate repayment states (e.g., *current*, *grace period*, or *late*) are excluded because their final credit outcome is not yet observed. Loans labeled *does not meet the credit policy* are collapsed into their corresponding economic outcomes (*charged off* or *fully paid*), ensuring that all observations reflect realized borrower behavior.

After applying these definitions, the modeling population contains **64,514 default events** and **278,443 non-default events**, for a total of **342,957 loans**.

The resulting event rate is:

Event rate = 64,514 / 342,957 ≈ **18.8%**

Methodological guidance in predictive modeling commonly recommends a minimum number of outcome events relative to the number of predictors used in a logistic regression model. A widely cited rule-of-thumb is the **events-per-variable (EPV) criterion**, which suggests approximately **10 outcome events per predictor parameter** to obtain stable coefficient estimates (Peduzzi et al., 1996; Harrell, 2015). Subsequent research has shown that this threshold may in some cases be relaxed, but it remains a useful conservative guideline for assessing sample adequacy (Vittinghoff & McCulloch, 2007).

With **64,514 observed default events**, the dataset provides orders of magnitude more outcome events than required for the number of predictors used in this study. The modeling population therefore satisfies widely accepted minimum sample size requirements for predictive modeling.

With the integrity of the dataset established and the modeling population confirmed to contain a sufficient number of outcome events, the analysis can now move to the submission-time feature space used for model development.

---

## Part 1 — Conclusion

The diagnostic stage confirms that the dataset provides a coherent view of the lending process and that the modeling population can be defined clearly. Loans that have completed their repayment cycle fall into two observable outcomes: repayment or default, allowing the modeling cohort to be defined using realized borrower behavior rather than intermediate loan states.

The dataset’s temporal structure is also consistent with the growth of the lending platform. Loan issuance increases substantially over time while default rates remain broadly stable. This supports the evaluation design used in this project: models are trained on earlier lending cohorts and evaluated on a later period, allowing predictive performance to be measured under realistic out-of-sample conditions.

The analysis also identifies a reporting shift affecting several balance-related variables. These fields contain systematic missing values in the earlier years but become consistently populated in the test period. This pattern reflects a change in reporting practices rather than borrower behavior. Documenting this shift prevents the modeling stage from mistakenly interpreting reporting artifacts as predictive signal.

With unresolved loan states excluded, the resulting modeling population contains a large number of realized default events, ensuring that the dataset is statistically suitable for predictive modeling.

Taken together, these diagnostics indicate that the dataset provides a stable and interpretable record of the lending process. The outcome definition is clear, the temporal structure supports meaningful evaluation, and potential reporting artifacts have been identified. With these issues addressed, the analysis can move to the submission-time feature space to examine whether borrower characteristics contain signals relevant to credit risk.

---

## Part 2 — Decision-Focused Analysis (`feature_base`)

This section analyzes the submission-time feature space used for modeling. The **feature base training dataset** and **feature base testing dataset** contain only variables that satisfy the application-submission prediction boundary.

The feature base dataset does **not** retain `issue_d`. Temporal diagnostics therefore stop here. From this point onward the analysis is strictly cross-sectional: borrower profile, contract terms, balance-sheet burden, credit history, recent credit behavior, and reporting-shift-sensitive fields are evaluated only through information available at application.

### 2.1 Target Definition & Cohort

The feature base dataset uses the same realized-outcome restriction as the clean dataset. This keeps the label definition consistent while limiting the analysis to variables that were available at application submission.

At this stage the relevant question is not temporal drift by issuance date. The relevant question is whether the submission-time feature space is examined on the same realized-outcome population defined earlier.

### 2.2 Application Profile

Application-profile variables describe the borrower at the moment of application. The relevant question is whether these fields separate repayment risk in a way that is both economically interpretable and stable across splits.

### 2.3 Loan Structure

Loan-structure variables describe the requested contract rather than the borrower. The relevant question is whether contract design and stated use are associated with materially different default outcomes.

### 2.4 Debt Burden

Debt-burden variables measure how much financial load the borrower already carries at the moment of application. This is one of the most important sections of the notebook because it speaks directly to repayment capacity.

### 2.5 Credit History

Credit-history variables summarize the depth and condition of the borrower’s credit file. The question is whether longer and cleaner histories are associated with lower realized default rates.

### 2.6 Recent Credit Behavior

Recent-credit-behavior variables capture short-horizon credit activity and the recency of adverse events. The purpose here is to determine whether recent strain or recent credit seeking is associated with materially higher realized default risk.

### 2.7 Reporting Shifts / Data Quality

The reporting-shift-sensitive balance variables identified in the clean dataset remain relevant in the feature base dataset, but they are no longer evaluated over time because `issue_d` is not retained here. The relevant question is whether missingness differs materially between train and test and whether the explicit missingness flags carry risk information.

#### 2.8 Decision EDA Summary — Submission-Time Feature Behavior

The feature-base analysis shows that the submission-time feature space contains observable differences in default outcomes without relying on post-origination information. Borrower profile, loan design, leverage, credit-file depth, recent credit activity, and explicit missingness all produce systematic variation in default rates.

The important point is not that every variable is equally predictive. Rather, multiple dimensions of borrower and loan characteristics display economically plausible differences in repayment outcomes. These patterns are consistent with the mechanisms typically associated with consumer credit risk.

Taken together, the results indicate that the information available at loan application contains sufficient structure to support predictive modeling under the submission-time constraint.

---

## Part 2 Conclusion

The feature-level analysis examines the information available at the moment a borrower applies for a loan and evaluates how that information corresponds to observed differences in default rates. Several patterns emerge consistently across the feature groups. Variables describing existing credit obligations and past credit behavior show the strongest differences in default outcomes. Measures of leverage and credit utilization are generally associated with higher default rates, while indicators of established credit history correspond with lower risk.

Borrower profile variables such as income and employment characteristics also display variation in default rates, although the differences are typically less pronounced. These attributes describe repayment capacity but do not fully capture how borrowers manage their existing credit obligations, so they provide contextual information without dominating the predictive signal.

Loan structure variables introduce an additional dimension. Interest rates and contractual loan characteristics partly reflect LendingClub’s internal underwriting decisions. Their relationship with default risk therefore contains both borrower information and the platform’s attempt to price that risk. This observation connects directly to the central analytical question of the project: whether a model built using the same application-time information can reproduce or improve the ordering implied by LendingClub’s grading system.

Overall, the submission-time feature space exhibits several economically plausible sources of variation in default risk. Taken together, these results indicate that the information available at loan application contains sufficient structure to support predictive modeling. The next stage evaluates how accurately these relationships can be captured using formal predictive models.

---

## Notebook Conclusion

The diagnostic stage confirms that the LendingClub dataset provides a coherent representation of the lending process. Repayment outcomes can be defined using completed loan states, the temporal structure of the data supports a realistic train–test evaluation design, and reporting shifts affecting several balance variables have been identified and documented. These findings establish that the dataset can be used for predictive analysis without relying on information that would not have been available at the moment of loan application.

The submission-time feature space also shows systematic differences in borrower default outcomes. Variables describing leverage, credit utilization, and historical repayment behavior display the strongest variation in default rates. Borrower profile characteristics provide additional context about repayment capacity, while loan structure variables reflect the platform’s own attempt to price borrower risk through interest rates and contract terms.

Three conclusions follow from these observations: 
- 1. The dataset represents a stable lending environment with consistent default behavior across the evaluation window. 
- 2. The information available at application submission contains multiple sources of risk differentiation, particularly in borrower leverage and credit history. 
- 3. The feature space allows a direct comparison between LendingClub’s existing grading system and model-based estimates of borrower risk.

The next stage moves from exploratory analysis to predictive modeling. Using only the submission-time information defined in this notebook, the modeling phase evaluates how accurately borrower default risk can be predicted and whether a statistical model can reproduce or improve the risk ordering implied by LendingClub’s grading system.

# Loans at Risk: Capturing Default — Modeling

## Purpose

This notebook implements the modeling stage of the *Loans at Risk: Capturing Default* project.

The analysis uses the `feature_base` datasets produced in the ETL pipeline:

- **Training set:** loans issued between 2007 and 2014  
- **Testing set:** loans issued in 2015  

The modeling population is restricted to loans with terminal outcomes in order to define a clear prediction target.

The purpose of this notebook is to train and evaluate a small set of representative predictive models on this population and identify the model that produces the most accurate predictions of borrower default risk.

---

## Analytical Approach

Default prediction is formulated as a binary classification problem.

Loans are classified as:

- **Default (1):** Charged Off or Default  
- **Non-default (0):** Fully Paid  

Two policy-related loan status categories are normalized before modeling:

- “Does not meet the credit policy. Status: Fully Paid” → Fully Paid  
- “Does not meet the credit policy. Status: Charged Off” → Charged Off  

This preserves the economic outcome of the loan while removing administrative distinctions that are not relevant to the prediction task.

Model performance is evaluated primarily using **ROC-AUC**, with additional metrics including precision, recall, F1 score, and confusion matrices. Performance is compared between the training and testing datasets to assess predictive accuracy and potential overfitting.

---

## Model Strategy

Three models are evaluated in this analysis:

1. **Logistic Regression**  
2. **Random Forest**  
3. **CatBoost**

These models represent three different approaches to predictive modeling: a classical statistical model, a bagging-based tree ensemble, and a boosting-based ensemble method.

### Logistic Regression

Logistic Regression serves as the baseline model and represents the traditional statistical approach widely used in credit risk modeling (Thomas et al., 2017).

The model combines borrower characteristics into a weighted linear score. Each feature contributes positively or negatively depending on how strongly it is associated with default risk. Because this score can take any value, it is transformed using the **logistic function**, which maps the score onto a value between 0 and 1. The result can therefore be interpreted as the predicted probability that a loan will default. Logistic regression has historically been the dominant methodology used in credit scoring because it produces stable probability estimates and remains relatively interpretable compared to more complex machine learning models. It therefore provides a natural benchmark against which more flexible nonlinear models can be evaluated.

### Random Forest

Random Forest represents the **bagging ensemble paradigm** introduced by Breiman (2001).

A decision tree predicts outcomes by repeatedly dividing the data into smaller groups based on feature values. Each split attempts to separate loans with different outcomes—for example, separating borrowers with high debt ratios from those with lower ones. A single decision tree can be unstable because small changes in the data may lead to very different splits. Random Forest addresses this by building **many trees**, each trained on a slightly different random sample of the data and predictor variables. Each tree produces its own prediction, and the final model output is obtained by averaging the predictions across all trees. Combining many trees in this way reduces the instability of individual trees while allowing the model to capture nonlinear relationships and interactions between variables.

### CatBoost

CatBoost represents the **boosting ensemble paradigm**, where models are trained sequentially rather than independently (Prokhorenkova et al., 2018).

Like Random Forest, CatBoost uses decision trees as its building blocks. However, instead of training many independent trees, boosting trains trees **one after another**. Each new tree focuses on correcting prediction errors made by the previous trees. Over many iterations the model gradually improves its predictions by concentrating on the observations that were most difficult to predict earlier in the training process. CatBoost also includes techniques designed for tabular datasets containing categorical variables and aims to reduce bias during model training. Empirical research shows that tree-based ensembles remain effective approaches for tabular prediction tasks (Grinsztajn et al., 2022).

---

Together these models represent three distinct learning approaches:

- linear statistical modeling  
- bagged tree ensembles  
- boosted tree ensembles  

Their performance is compared in order to identify the model that produces the most accurate predictions of borrower default risk.

---

## Structure

The notebook proceeds in four stages.

1. **Modeling Population**  
   The modeling population is finalized and the binary target variable is constructed.

2. **Feature Engineering**  
   Additional feature transformations are performed to prepare the dataset for model training.

3. **Model Training**  
   Logistic Regression, Random Forest, and CatBoost models are trained on the training dataset.

4. **Model Evaluation**  
   Model performance is evaluated on both the training and testing datasets in order to identify the best-performing model.

---


## Modeling Population – Conclusion

This stage restricts the dataset to loans with **terminal repayment outcomes** and constructs the binary prediction target `target_default`.

Loans in active repayment states (e.g., current, late, or in grace period) are excluded because their final credit outcome is not yet observed. Policy-related loan statuses indicating that a loan did not meet LendingClub’s internal credit policy are interpreted according to their underlying economic outcomes when constructing the target variable.

The resulting datasets therefore contain only loans with **realized repayment outcomes** and a clearly defined binary target suitable for predictive modeling.

Population reduction and target balance are documented in the modeling artifacts produced in this stage.

To establish a clear pipeline boundary, the resulting datasets are persisted as:

- `model_train_data.parquet`
- `model_test_data.parquet`

These files represent the finalized **modeling population** and serve as the input for the next stage of the pipeline.

---

The analysis now proceeds to **feature engineering**, where the submission-time feature space is prepared for model training.

## Feature Transformation Strategy

Feature transformation is defined after reviewing feature type, cardinality, and numeric distribution shape in the modeling population.

Two columns are excluded from the modeling feature space:

- `loan_status` is removed because it reflects repayment outcome and would introduce direct target leakage.
- `row_id` is excluded from the feature set but retained in the dataset to preserve row-level identity across pipeline stages and enable explicit alignment during validation.

---

Categorical transformation is limited to:

- `purpose`  
- `home_ownership`  

These are the only retained borrower-facing categorical features. Both are low-cardinality application-time attributes and can be one-hot encoded without materially increasing dimensionality.

The transformation design is intentionally branch-aware rather than uniform. Three model-input branches are constructed from the same post-drop base:

1. **Shared branch**  
   Selected monetary features are log-transformed, retained categoricals are one-hot encoded, and remaining missing values are imputed. This branch is used for Logistic Regression, Random Forest (shared), and CatBoost (shared) to enable direct comparison across model classes.

2. **Tree no-log branch**  
   The same categorical encoding and imputation steps are applied, but monetary features are left on their original scale. This branch isolates the impact of log transformation on tree-based models.

3. **CatBoost native branch**  
   Categorical features remain in native form, missing values are preserved, and monetary features are left untransformed. This branch evaluates CatBoost under a representation closer to its intended usage.

---

The following binary indicators are retained without transformation:

- `has_mths_since_last_delinq`  
- `has_mths_since_last_major_derog`  
- `has_mths_since_last_record`  
- `has_mths_since_recent_bc_dlq`  
- `has_mths_since_recent_revol_delinq`  

These variables already encode the presence or absence of specific credit events. As structurally binary features, additional transformation would not improve representation.

---

Log transformation is applied selectively to the following numeric features in the **shared branch only**:

- `annual_inc`  
- `loan_amnt`  
- `revol_bal`  
- `tot_coll_amt`  
- `total_bal_ex_mort`  
- `avg_cur_bal`  
- `bc_open_to_buy`  
- `tot_cur_bal`  
- `tot_hi_cred_lim`  
- `total_bc_limit`  
- `total_il_high_credit_limit`  
- `total_rev_hi_lim`  

These variables represent monetary exposure and exhibit substantial right skew. In their raw form, a small number of extreme observations dominate scale without proportionally increasing signal. Log transformation compresses the upper tail while preserving rank order, improving learnability for linear models and stabilizing feature influence.

---

The remaining numeric features are left untransformed for structural reasons:

1. **Count variables**  
   Delinquencies, inquiries, account totals, and public records represent discrete event frequency. Log transformation would reduce interpretability without clear modeling benefit.

2. **Ratio and bounded variables**  
   Variables such as `dti`, `revol_util`, `bc_util`, `percent_bc_gt_75`, and `pct_tl_nvr_dlq` already operate on constrained scales. Their primary structure is not driven by extreme magnitude, so log transformation does not address a meaningful limitation.

3. **Time-since variables**  
   Several variables use sentinel-coded values to represent missing historical events. Their skewness reflects data construction as much as borrower behavior, so log transformation would distort this encoding rather than improve representation.

4. **Lower-range numeric variables**  
   Terms, counts, and basic credit history measures remain directly interpretable and usable across all model classes in their original scale.

---

The resulting feature engineering design prioritizes controlled comparison over automation. Leakage and identifiers are removed from the feature set, while `row_id` is retained for alignment. From this common base, three branches are constructed to isolate the effect of transformation choices on model behavior.

---

## Feature Engineering Conclusion

Feature engineering produced three distinct model-input branches from a common post-drop base.

The **shared branch** applies selective numeric log transformation, one-hot encoding, and median imputation. This branch provides the common comparison space used for Logistic Regression, Random Forest, and the shared CatBoost run.

The **tree no-log branch** applies the same one-hot encoding and median imputation steps, but leaves selected monetary features on their original scale. This branch isolates whether log transformation materially changes tree-model performance under otherwise comparable preprocessing.

The **CatBoost native branch** preserves retained categorical variables in native form, keeps missing values in place, and leaves selected monetary features untransformed. This branch tests CatBoost under a more native tree-based representation.

With these branches in place, the next stage is model training and comparative evaluation across both model class and preprocessing design.

---

## 3. Modeling

With the modeling population defined and feature engineering complete, the next step is to estimate predictive models on the engineered training data and evaluate how well they generalize to the testing period.

The modeling design now combines model-class comparison with preprocessing-branch comparison. Three branches are used:

- a **shared branch** with selective log transformation, one-hot encoding, and imputation
- a **tree no-log branch** with one-hot encoding and imputation but no numeric log transformation
- a **CatBoost native branch** with native categorical handling, native missing-value handling, and no numeric log transformation

Within this design, Logistic Regression is trained on the shared branch as a linear baseline, Random Forest is compared across the shared and tree no-log branches, and CatBoost is compared across the shared branch, the tree no-log branch, and the native branch.

This structure answers three related questions at once: how well default risk can be predicted from application-time information, how much predictive value is gained by moving from a linear baseline to more flexible model classes, and whether tree-based models benefit materially from preprocessing choices such as numeric log transformation and native categorical or missing-value handling.

---

Logistic Regression was trained on a scaled version of the shared imputed feature matrix. This scaling step was applied only to the linear baseline because it supports stable numerical optimization and does not alter the underlying feature set used for model comparison.

## Modeling Conclusion

The modeling stage successfully constructs and trains multiple model classes under the constraint that only application-time information is used. Logistic Regression, Random Forest, and CatBoost models are all implemented using the same feature space, ensuring that differences in performance reflect modeling approach rather than data leakage or inconsistent inputs.

Separate preprocessing branches are used where necessary, but all models operate on the same underlying information set. This allows for a controlled comparison in the evaluation phase.

At this stage, no conclusions are drawn about which model performs best. The purpose of the modeling step is to produce valid, comparable models that can be evaluated in terms of both predictive performance and decision impact.

The next step is therefore to evaluate these models on the test set and analyze their error structure and economic implications.

---

## 4. Evaluation

Model training establishes candidate estimators, but does not by itself answer how well they generalize beyond the training window. Evaluation is therefore conducted on the 2015 testing period in order to assess out-of-sample performance under the temporal split defined for the project.

The comparison includes Logistic Regression on the shared branch, Random Forest on both the shared and tree no-log branches, CatBoost on both the shared and tree no-log branches, and an additional CatBoost native run with native categorical and missing-value handling. This makes it possible to compare model classes under a common input structure while also testing two preprocessing questions directly: whether numeric log transformation materially affects tree-model performance, and whether CatBoost gains materially from a more native representation.

Evaluation focuses on three questions. First, how well do the models rank higher-risk borrowers above lower-risk borrowers? Second, how do their classification outcomes behave on the testing period? Third, how informative and stable are the predicted default probabilities? Together, these results determine which model and preprocessing branch provides the strongest basis for subsequent tuning and validation against LendingClub’s grading system.

---

### Why discrimination metrics are not sufficient

The primary evaluation metric used so far is ROC AUC. This metric measures how well the model ranks borrowers from low risk to high risk across all possible thresholds. A higher ROC AUC indicates that, on average, borrowers who default receive higher predicted probabilities than those who do not.

However, this does not directly translate into better lending decisions. Lending is not concerned with ranking alone, but with specific actions: which loans are approved and which are rejected. Those actions depend on a chosen threshold, and different thresholds produce different combinations of errors.

In particular, two models with very similar ROC AUC values can produce meaningfully different error structures at a given threshold. One model may allow more defaults to pass through (false negatives), while another may reject more good loans (false positives). These differences are not captured by ROC AUC but are central to the economic outcome of the model.

For this reason, model evaluation must go beyond discrimination metrics and examine the structure and cost of prediction errors.

## Evaluation Progress and Next Step

The first stage of evaluation compared model performance on the 2015 testing period using standard classification metrics. This included ROC AUC, accuracy, precision, recall, F1, Brier score, and confusion matrix counts. Together, these results showed how well each model ranked borrower risk, how conservative or aggressive its default classifications were at the default threshold, and how informative its predicted probabilities were.

At this stage, CatBoost remained the strongest overall model family, but the branch comparison now matters as much as the model comparison. The results distinguish between a shared branch designed for direct comparability, a tree no-log branch designed to isolate the effect of numeric log transformation on tree models, and a native CatBoost branch that preserves categorical and missing-value structure more directly. This makes it possible to assess not only which model performs best, but also whether preprocessing choices materially change that conclusion.

These results establish which models perform best in statistical terms, but they do not yet show the full decision meaning of those errors. The confusion matrix counts treat every loan equally, even though a false negative on a small loan and a false negative on a much larger loan do not carry the same practical consequence. In the same way, a model that appears weaker in event counts could still be preferable if the loans it misclassifies are economically less important.

The next step therefore extends evaluation from counts to exposure. For each model, the predicted outcomes will be linked back to raw loan amounts in the testing data so that true positives, true negatives, false positives, and false negatives can also be assessed in monetary terms. This makes it possible to examine not only which model performs best statistically, but also which model performs best when errors are viewed through the size of the loans involved.

---

## Feature Space Assessment

The current results also do not support opening a broader feature-engineering cycle.

Performance gains across model classes are incremental rather than structural. Moving from Logistic Regression to Random Forest and then to CatBoost improves predictive performance, but not by an amount that suggests a large volume of unused signal remains in the existing feature set. This matters because different model classes extract different kinds of structure. Logistic Regression can only capture relatively simple linear relationships. Random Forest can capture non-linear thresholds and interactions. CatBoost is more flexible again and is usually better at exploiting subtle interactions and irregular variable behavior. If major predictive structure were still sitting in the current variables but simply had not been represented properly, the shift from a linear model to stronger tree-based models would normally produce a clearer performance jump. That is not what happens here. Performance improves, but it improves gradually. The models are not revealing a hidden reserve of signal waiting to be unlocked by more elaborate feature construction. This pattern is consistent with established results in statistical learning, where improvements in predictive performance tend to exhibit diminishing returns as model complexity increases (Ng, 2000).

The branch comparisons support the same conclusion. Removing numeric log transformation does not materially change tree-model performance, and CatBoost does not gain meaningfully from a more native representation. That matters because these branches directly test whether current feature handling is suppressing useful information. If the main limitation were a poor representation of the existing variables, these preprocessing changes would be expected to move the results more clearly. Instead, they leave the model ranking largely unchanged and produce only marginal differences in scores. Taken together, this suggests that the feature representation is not the main constraint.

The error structure points in the same direction. False negatives remain present, but not in a pattern that indicates an obvious missing interaction or overlooked relationship that could be recovered through additional feature construction. In practice, that means the models are still missing some defaults, but they are not doing so in a way that clearly identifies a specific blind spot in the current feature space. There is no obvious sign that a small number of ratio features, interaction terms, or alternative transforms would resolve a concentrated weakness. Instead, the remaining errors look more like the result of limited information than of an unmodeled pattern sitting in plain sight.

Under the strict application-time boundary used in this project, that result is plausible. The model only sees borrower and loan information available at the moment of application. It does not see later repayment behavior, servicing developments, changing borrower circumstances, or other post-origination signals that may drive eventual default. That means some uncertainty is structural. It reflects the limits of what can be known at origination, not a failure to engineer the current variables aggressively enough. In statistical learning terms, this corresponds to irreducible error in the data-generating process, where remaining prediction error cannot be eliminated through additional modeling or feature construction (Hastie et al., 2009).

For that reason, further feature work would add complexity without materially improving the decision boundary. The feature space is therefore treated as sufficient. The next step is to lock the strongest model configuration, apply controlled hyperparameter tuning, test whether loan-size weighting improves decision usefulness, and then prepare the final version of the model produced in this notebook before moving the broader threshold and portfolio-level decision analysis into the separate validation notebook.

---

### Interpreting prediction errors in a lending context

The confusion matrix distinguishes between four outcomes, but in a lending setting these outcomes have direct economic meaning.

- **True Negative (TN):** A non-defaulting loan that is correctly approved. This represents successful lending activity.
- **True Positive (TP):** A defaulting loan that is correctly rejected. This represents avoided loss.
- **False Positive (FP):** A non-defaulting loan that is rejected. This represents missed opportunity.
- **False Negative (FN):** A defaulting loan that is approved. This represents realized loss.

Because loan amounts differ across observations, the economic impact of these errors is not uniform. A single false negative on a large loan can outweigh many small errors. For this reason, the analysis aggregates total loan amounts within each outcome category to approximate the financial impact of model decisions.

## Monetary Evaluation

Monetary evaluation extends the confusion-matrix comparison by linking predicted outcomes in the testing period back to raw loan amounts. This shows not only how often each model makes different types of classification errors, but also how much loan exposure is attached to those errors.

This step matters because the statistical comparison alone cannot tell whether a model that looks weaker in event counts is still preferable once loan size is taken into account. A false negative on a small loan and a false negative on a much larger loan count the same in the confusion matrix, but they do not carry the same practical consequence. The same logic applies to false positives: some models may reject more good loans in count terms while still protecting more or less economically important exposure.

The monetary comparison therefore evaluates the same model-and-branch combinations used in the statistical comparison — Logistic Regression on the shared branch, Random Forest on the shared and tree no-log branches, CatBoost on the shared and tree no-log branches, and CatBoost on the native branch — but now in terms of the raw principal amounts attached to true positives, true negatives, false positives, and false negatives.

---

### Interpreting the model trade-off

The comparison between CatBoost variants illustrates an important trade-off.

The native CatBoost configuration achieves the lowest false negative exposure, meaning it allows slightly fewer defaulting loans to pass through. However, the magnitude of this improvement is small relative to the increase in false positive exposure, meaning a larger volume of good loans is rejected.

The shared CatBoost model, while marginally worse in terms of false negative exposure, avoids rejecting as many good loans. This results in a more balanced error profile. The preferred model is therefore not the one that minimizes a single metric, but the one that provides the best balance between avoiding losses and preserving lending opportunity.

---

## 5. Model Selection and Optimization

The statistical comparison identified `catboost_native` as the strongest model on ROC AUC, Brier score, recall, and false-negative count. The monetary comparison, however, slightly changes the decision. While `catboost_native` produces the lowest false-negative exposure, its advantage over `catboost_shared` is negligible in dollar terms, while its false-positive exposure is materially higher. In other words, the native branch catches only a very small additional amount of bad lending, but does so at the cost of rejecting meaningfully more good lending.

For that reason, the shared CatBoost branch is selected as the candidate model for optimization. It remains one of the strongest statistical models, preserves a cleaner and more comparable preprocessing design, and provides the more favorable balance between avoided losses and unnecessarily rejected good loans.

The next step is therefore to optimize `catboost_shared` in two stages. First, hyperparameter tuning is used to improve the untuned candidate without changing the objective. Second, a weighted version of the tuned model is estimated so that larger loans receive more influence during training. This makes it possible to separate predictive optimization from objective redesign.

---

### CatBoost Shared Hyperparameter Tuning

Hyperparameter tuning is applied only to the selected candidate model. The goal is not to reopen model selection, but to improve the chosen CatBoost shared branch under the same predictive objective. A lightweight random search is used on a validation split drawn from the training data, leaving the testing period untouched until final evaluation.

---

### Loan-Amount Weighting

The tuned CatBoost shared model is now re-estimated with sample weights based on raw loan amounts. This does not change the feature space or the tuned parameter values. It changes only the training objective by giving larger loans more influence. The purpose of this step is to test whether the candidate model can be shifted toward better protection against economically larger mistakes without first changing the lending threshold.

### Why loan-amount weighting did not improve the model

The weighted model was designed to give larger loans more influence during training. The intuition behind this approach is that mistakes on larger loans are more costly.

While this adjustment slightly improved overall discrimination metrics, it did not improve the economic outcome relative to the tuned unweighted model. In particular, it did not reduce false negative exposure sufficiently and led to a deterioration in the error balance.

This suggests that the model was already capturing the relevant structure of risk in the data. Increasing the influence of larger loans did not introduce new information, but instead shifted the optimization in a way that did not translate into better decisions.

As a result, the weighted model is not selected.

---


## Model Optimization Conclusion

The CatBoost shared model was selected as the candidate model because it offered the best balance between statistical performance and monetary trade-offs. Hyperparameter tuning then tested whether this candidate could be improved without changing the underlying objective.

The tuning step produced a measurable improvement. The model’s ability to separate higher-risk borrowers from lower-risk borrowers increased, recall improved, and the number of false negatives declined relative to the untuned model. This indicates that the model was not yet fully optimized and that controlled parameter tuning can meaningfully improve performance without altering the feature space.

A weighted variant was then estimated in order to test whether giving larger loans more influence during training would shift the model toward a more economically favorable error profile. While this adjustment led to a slight improvement in overall risk separation, it did not improve the practical lending objective. In particular, it did not reduce false negative exposure sufficiently and, in some cases, worsened the balance between false negatives and false positives.

This result suggests that the model already captures the relevant risk structure present in the data, and that increasing the influence of larger loans does not introduce additional usable signal. Instead, it shifts the optimization in a way that does not translate into better portfolio-level outcomes.

The optimization stage therefore leads to a clear selection: the final model from this notebook is the tuned, unweighted CatBoost shared model. This model is saved as a documented artifact and passed to the validation stage, where predictions are translated into decision thresholds and lending-policy trade-offs rather than treated as a pure classification exercise.

---

## Notebook Conclusion

This notebook evaluates how well default risk can be captured using only application-time borrower information and whether increased modeling complexity improves predictive and decision outcomes.

The results show that default risk is meaningfully predictable, but not to a level that suggests near-perfect separation. Model performance improves as flexibility increases, with CatBoost outperforming both Logistic Regression and Random Forest. However, these improvements are incremental rather than structural, indicating that the available feature set already captures most of the accessible signal.

Model selection cannot be based on statistical metrics alone. While the CatBoost native configuration achieves slightly stronger risk separation and marginally lower false negative exposure, it does so at the cost of rejecting a materially larger volume of good loans. The shared CatBoost model provides a more balanced trade-off between risk reduction and lending opportunity and is therefore selected as the candidate model.

This candidate was then optimized. Hyperparameter tuning produced a measurable improvement, increasing risk separation, improving recall, and reducing the number of false negatives relative to the untuned model. This confirms that the model was not yet fully optimized and that controlled tuning can improve performance without changing the underlying feature space.

A weighted variant was also evaluated in order to prioritize larger loans during training. While this approach slightly improved overall risk separation, it did not improve the economic error profile relative to the tuned unweighted model. In particular, it did not reduce false negative exposure sufficiently and, in some cases, worsened the balance between false negatives and false positives. As a result, the weighted model is not selected.

The modeling stage therefore concludes with a clear outcome: the final model is the tuned, unweighted CatBoost shared model. This model represents the best balance between predictive performance and decision impact under the constraint of application-time information and is saved as a documented artifact.

The next step is to move beyond model performance and evaluate how predictions translate into decisions. This is addressed in the validation stage, where thresholds, trade-offs, and portfolio-level outcomes are analyzed.

# Loans at Risk: Capturing Default — Validation

## Purpose

This notebook implements the validation stage of the *Loans at Risk: Capturing Default* project. The analysis uses aligned datasets derived from the same underlying loan population:

- **Training set:** loans issued between 2007 and 2014  
- **Testing set:** loans issued in 2015  

The modeling population is restricted to loans with terminal outcomes in order to define a clear prediction target. The purpose of this notebook is **not to train or compare predictive models**, but to evaluate whether the selected model can be used to support **lending decisions** and whether it provides an improvement over LendingClub’s internal grading system. Specifically, this notebook assesses how predicted default probabilities translate into **acceptance policies**, **risk outcomes**, and **capital trade-offs**.

---

## Data Coverage and Baseline Scope

The modeling and validation analysis is restricted to loans with **application-time information** and a **clear observed repayment outcome**. This implies that the supervised modeling population consists of loans that were **approved and issued by LendingClub** and later observed to a terminal outcome. In principle, evaluating LendingClub’s broader approval process would require access to rejected applications as well. Archived rejected-loan files appear to exist, but repeated download attempts produced source-side errors, so these data could not be incorporated into the project. As a result, the analysis is conducted on the population of loans that LendingClub chose to accept and that can be observed to completion. This has two implications:

- The model is trained and evaluated on loans with observable terminal outcomes, consistent with the project’s prediction boundary.
- The comparison with LendingClub’s internal grading system is therefore a comparison **within the accepted loan population**, not a full evaluation of LendingClub’s initial approval boundary across all applicants.

Accordingly, the notebook does not claim to reproduce or replace LendingClub’s complete underwriting process. Instead, it evaluates whether default risk can be more effectively ordered and managed within the set of loans that LendingClub chose to accept.

---

## Data Sources

The validation analysis combines two dataset representations derived from the same underlying loan population:

- **Selected model input datasets**  
  The datasets `selected_model_input_train.parquet` and `selected_model_input_test.parquet` contain the fully engineered feature space used by the final selected model. These datasets are loaded into `df_model_input_train` and `df_model_input_test`. The trained model is applied to these datasets within this notebook to generate predicted default probabilities at the time of application. These predictions form the basis for all subsequent validation analyses. The underlying feature engineering and model training steps are not repeated here. This notebook operates on the finalized model and its inputs to produce a consistent validation dataset.

- **`clean` datasets**  
  These datasets retain LendingClub’s internal grading system and contextual variables, including loan amount, that are required for decision-system comparison and capital evaluation.

Both dataset representations share a common row identifier (`row_id`), which is used to align model predictions with LendingClub grades and observed outcomes. This alignment produces a unified validation dataset in which each loan is associated with:

- predicted default probability (model)  
- LendingClub grade (baseline system)  
- observed repayment outcome  
- loan characteristics relevant for decision evaluation  

This structure allows direct comparison between model-based and grade-based decision policies on the same set of loans.

---

## Analytical Framing

The modeling phase established that default risk can be predicted from application-time borrower information and identified a candidate model. Validation shifts the focus from prediction to **decision use**.
A predictive model is only useful if it enables better decisions than existing alternatives. In this context, the relevant benchmark is LendingClub’s internal **grade-based risk classification**, which implicitly defines how borrower risk is evaluated at the time of application. The central question becomes:

> Can a probability-based model improve lending outcomes relative to a grade-based policy when applied at the time of application?

Answering this requires translating model outputs into **operating policies** and evaluating their consequences.

---

## Decision Framework

Predicted default probabilities are converted into lending decisions using **threshold-based policies**. For a given threshold:

- Loans with predicted default probability below the threshold are **accepted**
- Loans with predicted default probability above the threshold are **rejected**

Different thresholds represent different lending strategies, ranging from conservative (low tolerance for default risk) to more permissive (higher acceptance rates with increased risk exposure). Model-based policies are evaluated across a range of thresholds to understand the trade-off between:

- **Default risk among accepted loans**
- **Volume of accepted loans**
- **False positives:** good loans rejected  
- **False negatives:** bad loans accepted  

These outcomes are compared directly to policies derived from **LendingClub grades**, which serve as the baseline decision system.

---

## Evaluation Dimensions

Validation focuses on three complementary dimensions:

#### 1. Risk Separation

- Assessed using ROC curves and AUC  
- Measures how well the model orders borrowers by risk  
- Compared to LendingClub’s grade-based system  
- Indicates ranking quality, not decision usefulness  

#### 2. Calibration

- Compares predicted probabilities to observed default rates  
- Evaluates whether probabilities reflect actual risk levels  
- Required for reliable threshold-based decisions  
- Miscalibration leads to systematic over- or underestimation of risk  

#### 3. Decision Outcomes

- Evaluates lending policies defined by probability thresholds  

For each threshold:

- Acceptance rate  
- Default rate among accepted loans  
- False positives (good loans rejected)  
- False negatives (bad loans accepted)  
- Loan volume and exposure  

Compared to LendingClub grade-based policies to assess whether the model:

- reduces default exposure  
- preserves or improves lending volume  
- improves the risk–opportunity trade-off  

---

## Stress Testing

In addition to standard evaluation, the model is tested on a **policy-exception population** that was excluded from model training. This provides a check on whether the model generalizes beyond the standard underwriting population and whether its decision signals remain stable under distributional shifts.

---

## Structure

The notebook proceeds in five stages.

1. **Model Outputs**  
   Load predictions from the selected model for both training and testing datasets.

2. **Baseline Definition**  
   Define LendingClub grade-based acceptance policies to serve as a benchmark.

3. **Risk Evaluation**  
   Assess risk separation (ROC/AUC) and calibration of predicted probabilities.

4. **Policy Simulation**  
   Evaluate threshold-based lending policies and compare outcomes to grade-based policies.

5. **Decision Synthesis**  
   Identify candidate operating thresholds and summarize trade-offs in terms of risk, volume, and capital exposure.

---

## Outcome

The goal of this notebook is to determine whether a model-based decision policy can **improve lending outcomes relative to existing grading practices**, and to define how such a model would be used in practice. The result is not a statement about model accuracy alone, but a **decision-oriented assessment** of whether and how the model should be deployed.

---

## Clean Population Alignment

The validation stage combines two data representations with different roles. The **model-input datasets** define the population on which the model operates. They are restricted to loans with terminal outcomes and match the population used during modeling. Predicted default probabilities are generated by applying the trained model to these datasets. The **clean datasets** retain the original variables required for interpretation and decision analysis, such as LendingClub grade and raw loan amount. These datasets are not restricted to the modeling population and therefore contain additional observations. Before validation, both representations must be aligned to the same row universe.

---

### Alignment

Validation is performed only on loans for which model predictions are available. The clean dataset is therefore restricted to the set of `row_id`s present in the model-input dataset. All other rows are removed. This produces a dataset that:
- matches the modeled population exactly  
- retains all contextual variables  
- avoids mixing raw and transformed representations  

---

### Target

The model-input datasets contain the binary target. The clean datasets do not. If the target is not present, it is reconstructed from `loan_status` using the same mapping applied during modeling. This ensures consistency between training and validation.

---

### Result

The final validation dataset:
- contains the same loans as the model-input dataset  
- includes the target variable  
- retains LendingClub grades and contextual features  

This dataset is used for all subsequent validation steps.

## 1. Model Outputs

The validation dataset contains outcomes and contextual variables but does not yet include model predictions. The trained model is applied to the engineered model-input datasets to generate predicted default probabilities for each loan. These predictions reflect the estimated probability of default at the time of application. The model-input datasets are used exclusively for scoring, as they contain the transformed feature space required by the model. The resulting predictions are then aligned with the clean validation dataset using `row_id`. This produces a unified dataset in which each loan is associated with:
- predicted default probability  
- observed outcome (`target_default`)  
- LendingClub grade  
- loan characteristics (e.g. loan amount)  

This dataset forms the basis for all subsequent validation analyses.

## Model Outputs — Summary

Predicted default probabilities were generated using the selected model and aligned with the clean validation datasets via `row_id`. The resulting dataset associates each loan with:
- predicted default probability  
- observed outcome (`target_default`)  
- LendingClub grade  
- loan characteristics (e.g. loan amount)  

These datasets are persisted to ensure that all subsequent analysis operates on a stable and reproducible base. The validation datasets now represent the full decision context: model predictions, realized outcomes, and the baseline system.

---

### Next Step

The next section evaluates **risk separation**. This assesses whether the model provides a stronger ordering of borrower risk than LendingClub’s grade-based system, using ROC curves and AUC.

---

## Baseline Structure: Grade and Subgrade System

The grade-system summary shows that `sub_grade` is fully populated across both training and testing datasets. Each grade contains exactly five subgrades, with no missing values. Within each grade, loans are not evenly distributed across subgrades. The distributions are normalized, so values represent the share of loans within each grade. The patterns are asymmetric:

- Higher-quality grades (e.g. A) are skewed toward higher-index subgrades (e.g. `a4`, `a5`), indicating that many loans are close to the boundary with the next grade rather than concentrated at the lowest-risk end (`a1`).
- Lower-quality grades (D–G) show the opposite pattern, with more loans in lower-index subgrades (e.g. `d1`, `g1`), meaning fewer loans occupy the extreme high-risk tail (`g5`).
- Intermediate grades (B and C) are more evenly distributed.

This indicates that the grading system is not symmetric. Higher-quality grades are used more broadly, while lower-quality grades are more selective at the upper-risk end. These patterns are consistent across training and testing datasets, suggesting a stable grading structure rather than a sampling effect. An important implication is that subgrades are not equally sized. Moving a threshold across subgrades does not result in equal changes in loan volume. Overall, `sub_grade` represents a consistent internal ranking system and is used as the baseline for comparison.

## Baseline Structure Conclusion

The LendingClub grade system provides a structured and stable baseline for risk classification. Subgrades are fully populated and consistently applied across both training and testing datasets. The internal distribution of loans within grades is non-uniform and asymmetric, meaning that thresholds based on subgrades do not translate into uniform changes in loan volume. Observed default rates confirm that `sub_grade` represents a meaningful risk ordering. Default risk increases substantially from `a1` to `g5`, and this pattern is consistent across datasets. While the highest-risk subgrades show some variability due to smaller sample sizes, the overall structure remains stable. This indicates that the baseline system already captures a substantial portion of the available risk signal. As a result, the model is not competing against a weak baseline. Any improvement must come from refining the ordering within and across subgrades, particularly in regions where risk is concentrated. The baseline is therefore evaluated at the `sub_grade` level, providing a complete and operationally relevant ranking of borrowers.

---

## 3. Risk Evaluation

This section evaluates the selected model from two complementary perspectives:

1. **Risk separation**  
   → how effectively the model orders borrowers from lower to higher risk relative to LendingClub’s internal ranking  

2. **Calibration**  
   → whether predicted default probabilities correspond to observed default rates  

Together, these determine whether the model produces both a strong risk ordering and usable probability estimates before policy evaluation. Risk separation is evaluated by comparing the model’s predicted probabilities to the subgrade-based baseline. ROC curves and AUC are computed for both systems on training and testing data to assess how well each orders borrowers by default risk.

---

### Risk Separation — Conclusion

The model improves risk separation relative to LendingClub’s grading system, indicating that additional predictive signal is available beyond the subgrade-based baseline. This means borrowers can be ordered more precisely by risk than under the existing grading system. The next step is to determine whether this improved ordering translates into better lending decisions, in terms of reduced default exposure and preserved lending volume.

---

### Calibration

Calibration assesses whether predicted default probabilities correspond to observed default rates. This is required for threshold-based decision-making, as probability estimates determine whether a given acceptance rule reflects the intended level of risk.

---

### Calibration — Conclusion

The model is reasonably well calibrated across the low-to-middle probability range in both training and testing data. Predicted default probabilities track observed default rates closely enough to support threshold-based evaluation. Calibration is less stable in the highest-risk region, where the relationship between predicted and observed risk becomes less consistent across datasets. As a result, decisions targeting the extreme upper tail should be interpreted with additional caution. Overall, the model provides probability estimates that are suitable for decision evaluation, with weaker reliability at the highest predicted risk levels.

---

## Risk Evaluation — Conclusion

The model demonstrates improved risk separation relative to the subgrade-based baseline, indicating that additional predictive signal is available beyond LendingClub’s grading system. This improvement is consistent across training and testing data, though more modest out of sample. Calibration results show that predicted default probabilities broadly align with observed default rates across the low-to-middle probability range. Calibration is less stable in the highest-risk region, where predicted and observed risk diverge more noticeably. Taken together, the model provides both a stronger risk ordering and probability estimates suitable for threshold-based decision-making, with reduced reliability in the upper tail. The next step is to evaluate whether these improvements translate into better lending decisions in terms of default risk and lending volume.

---

## 4. Policy Simulation

This section evaluates how the model and the subgrade-based baseline perform when translated into lending decisions. For a range of acceptance policies, both systems are used to simulate which loans would be approved or rejected. The resulting portfolios are compared in terms of default rates, error types, and loan exposure. This provides a direct assessment of how each approach trades off risk reduction against lending volume, and whether the model’s improvements translate into decision outcomes.

---

### Model Threshold Policies

Model-based lending policies are defined by probability thresholds. Loans with predicted default probability at or below the threshold are accepted, while loans above the threshold are rejected. Thresholds are evaluated on both training and testing data to assess how acceptance rules affect default risk, lending volume, and error trade-offs.

---

The model achieves consistently lower default rates at comparable acceptance levels, indicating stronger risk separation than the baseline across the core operating range.

**Interpretation**

Positive values indicate that the model allocates capital more efficiently than the baseline at comparable acceptance levels, with the strongest advantage in the mid-range. The model achieves consistently lower default rates at comparable acceptance levels, indicating stronger risk separation across the core operating range.

---

## Policy Landscape

Both the model and the baseline define a trade-off between acceptance, risk, and capital allocation. As policies become more permissive, acceptance increases, default rates rise, and opportunity cost declines. This structure is consistent across both systems. The difference lies in how efficiently each system moves along this trade-off. The risk frontier shows that the model achieves lower default rates at comparable acceptance levels across the core operating range. This reflects a more precise ordering of borrower risk, allowing for better separation between higher- and lower-risk applicants at the time of application. This improvement in ranking translates into policy outcomes. At comparable acceptance levels, the model allocates capital more efficiently than the baseline in the mid-range, reducing exposure to defaulting loans while retaining more performing loans. The advantage is not uniform. At very low acceptance levels, the model is more conservative, increasing opportunity cost. At very high acceptance levels, both systems converge as acceptance approaches saturation, reducing the impact of improved ordering. Taken together, the model does not change the structure of the trade-off, but improves the efficiency with which it is navigated. The benefit is concentrated in the region where underwriting decisions are active, rather than at the extremes.

---

## Policy Exceptions and Pricing Consistency

This section examines loans that were issued despite not meeting LendingClub’s stated credit policy. The objective is to assess whether these policy exceptions differ in risk and whether pricing reflects that difference. Policy exception loans represent a small share of the portfolio (~1% in the training data), but their behavior differs materially from standard-issued loans. The observed default rate increases from 18.77% to 27.68% (+8.92 pp), and the model assigns higher predicted risk (18.79% → 24.46%, +5.67 pp). This indicates that both realized outcomes and model-based estimates consistently identify these loans as higher risk. Pricing increases only marginally. The average interest rate rises from 13.78% to 14.15% (+0.37 pp), which is small relative to the increase in default risk. Policy exception loans are also issued at lower average amounts, suggesting partial mitigation through exposure size rather than pricing.

Taken together, this indicates a structural inconsistency. Higher-risk loans are both issued and recognized as such, but pricing does not adjust proportionally to the observed increase in risk. This suggests that policy overrides are systematically associated with higher-risk loans without a corresponding adjustment in pricing or exposure. In the test data, no policy exception loans are observed, indicating that this behavior is confined to earlier vintages. This finding supports the broader validation result: risk can be identified, but is not consistently translated into aligned decision and pricing behavior.

---

## Risk Stratification

Loans are grouped into risk bands based on predicted default probability, and observed default rates are compared across those bands. This establishes whether the model produces a meaningful risk ordering that can serve as a reference for assessing how pricing varies with risk.

---

**Interpretation**

The model produces a clear and monotonic risk ordering. As predicted default probability increases, observed default rates rise consistently across all bands, confirming that borrowers can be reliably ranked by risk. Predicted probabilities are generally higher than observed default rates, particularly in the upper bands. This indicates a conservative bias: the model tends to overestimate risk. While this reduces the likelihood of underestimating defaults, it also requires careful interpretation when applying thresholds. The spread in risk is substantial. Observed default rates increase from approximately 3% in the lowest band to over 40% in the highest band, reflecting a structural shift in borrower quality rather than a marginal increase in risk. This increase is non-linear. Risk rises gradually in the lower bands but accelerates in the upper bands, indicating that default exposure is concentrated in a relatively small portion of the portfolio. As a result, decisions in the higher-risk range have a disproportionate impact. This structure explains the earlier threshold results. Thresholds around 0.35–0.40 align with the point where risk begins to accelerate more sharply. Below this range, risk remains relatively controlled; above it, default exposure increases rapidly. Taken together, the model provides usable risk stratification. It establishes a stable reference for evaluating whether pricing and policy decisions are aligned with the underlying risk structure of the portfolio.

---

## Interpretation — Risk, Pricing, and Portfolio Outcomes

This analysis compares **predicted risk**, **observed default rates**, and **interest rates** across risk bands derived from the model’s predicted default probabilities. Each band represents borrowers with similar predicted risk at the time of application, allowing direct comparison between realized risk and pricing. Within each band, we evaluate:

- **Observed default rate** → realized borrower risk  
- **Interest rate (mean/median)** → pricing applied to that risk  

The objective is to assess whether pricing scales in line with the underlying risk structure. Interest rates increase monotonically across risk bands, indicating that LendingClub differentiates pricing based on risk. However, pricing increases much more gradually than observed default risk:

- Default risk rises from ~3% to ~43%  
- Interest rates rise from ~8% to ~19%  

This gap becomes most pronounced in the upper risk bands, where default rates accelerate sharply while interest rates increase more slowly. Pricing therefore moves in the correct direction, but does not scale proportionally with risk. This reveals a structural mismatch between risk and pricing. Risk is non-linear and concentrated in the upper tail, while pricing follows a more gradual pattern. As a result, pricing appears relatively weak in the segments where default risk increases most rapidly. This does not imply that pricing is incorrect or that the portfolio is unprofitable, but it indicates that pricing is not tightly aligned with realized risk across the full distribution. The policy analysis shows that portfolio-level outcomes can remain favorable even when pricing is not closely aligned with risk in all segments. This is because:

- Lower- and mid-risk loans contribute stable performance  
- Higher-risk segments represent a smaller share of total exposure  

As a result, outcomes are driven not only by pricing, but also by how capital is distributed across the risk spectrum. The key implication is that the system captures differences in risk, but does not fully align pricing with the shape of that risk. The limitation is therefore not the availability of risk information, but how that information is used in decisions. The model provides a clearer view of the risk distribution, particularly in the upper tail, enabling more targeted decisions in segments where risk increases sharply but pricing does not adjust at the same rate. In other words, the model does not just predict default—it makes visible where pricing and decision rules are not fully aligned with the underlying risk structure.

---

## 5. Decision Synthesis

The previous sections establish three key facts: default risk can be estimated at the time of application, risk increases non-linearly across the loan portfolio, and pricing increases with risk but more gradually, particularly in the upper tail. The task is therefore not to identify risk, but to decide how to act on it. The model produces a predicted default probability for each loan. This is converted into a decision using a threshold: loans at or below the threshold are accepted, and loans above it are rejected. The threshold defines how risk is traded against volume. In practice, the threshold jointly determines three outcomes:

- **Acceptance rate** → how many loans are issued  
- **Default exposure** → how many accepted loans will default  
- **Retained performing loans** → how many good loans are kept  

Changing the threshold shifts all three simultaneously. Lower thresholds (e.g. 0.20–0.30) reduce default exposure but reject a larger share of performing loans, resulting in a more conservative policy. Higher thresholds (e.g. 0.45+) increase acceptance and volume, but admit a disproportionate number of high-risk loans, leading to a more aggressive policy. Policy simulations show that the strongest outcomes occur in the mid-range, centered around **0.35–0.40**. In this region, default exposure remains controlled while a substantial share of performing loans is retained, and outcomes improve relative to the baseline at comparable acceptance levels. Outside this range, the trade-off deteriorates: below ~0.30 the policy becomes overly restrictive, while above ~0.45 exposure to high-risk loans increases rapidly.

This result follows directly from the structure of the loan portfolio. Risk increases non-linearly, with sharp escalation in the upper bands, while pricing increases more gradually in that same region. As a result:

> Additional accepted loans in the upper tail contribute disproportionately to downside risk

A threshold around 0.35–0.40 limits exposure to this segment while preserving volume in lower-risk regions. The limitation of the baseline system is not that it fails to capture risk, but that it operates in discrete categories. Subgrades reflect broad differences in risk but cannot distinguish between borrowers within the same category. The model introduces a continuous risk measure that:

- differentiates borrowers within subgrades  
- identifies higher-risk loans more precisely  
- enables selective exclusion of the upper-risk tail  

This allows capital to be allocated more precisely across the risk distribution. A practical decision rule is therefore:

- Accept loans with predicted default probability ≤ **0.35–0.40**  
- Reject loans above this threshold  

This preserves a large share of performing loans while limiting exposure to segments where risk increases rapidly. The model does not remove the trade-off between risk and volume. It makes that trade-off explicit and controllable, allowing decisions to be aligned directly with estimated risk and enabling more effective allocation of capital across the portfolio.

---

## 6. Outcome — Conclusion

This analysis evaluates whether default risk can be captured using only application-time information and whether that information can be used to improve lending decisions. The results show that default risk can be estimated at the time of application. The model produces a stable and monotonic ordering of borrowers, with a substantial spread between low- and high-risk segments, indicating that risk is both predictable and meaningfully differentiable across the loan portfolio. However, the existing LendingClub system does not fully align with this risk structure. While pricing increases with risk, it does so more gradually than the observed increase in default rates, particularly in the upper tail. This creates segments where loans carry disproportionately high risk relative to their pricing. Policy simulations show that this misalignment has direct decision consequences. The subgrade-based system captures broad differences in risk but lacks the precision to control exposure within grades, resulting in continued allocation to segments where risk is concentrated.

The model-based approach improves on this by introducing a continuous risk measure that enables more targeted decisions. By applying threshold-based policies, exposure to the highest-risk segment can be reduced while preserving a substantial share of performing loans. At comparable acceptance levels in the core operating range (~0.25–0.35), the model reduces default rates by approximately **1.0 percentage point**, corresponding to a relative reduction of **~12%** compared to the baseline. Importantly, the economic impact depends on the operating region. At very low acceptance levels, both the model and the baseline generate negative outcomes, and the model’s contribution is best understood as **reducing losses** rather than creating value. The model’s advantage becomes economically meaningful in the mid-range of acceptance thresholds, where outcomes are already positive. In this region, the model improves proxy economic value by approximately **€11–15 million** relative to the baseline, reflecting more efficient allocation of capital at similar or higher acceptance levels. At higher acceptance levels, where most loans are approved, the difference between systems diminishes and can reverse, as outcomes become driven more by volume than by selection. These estimates are directional and do not represent a full economic model. The key implication is that the limitation is not the availability of risk information, but how that information is used. The model does not change the structure of the lending trade-off, but makes it explicit and controllable, enabling more effective allocation of capital under uncertainty.

# Project References

This file contains the methodological and domain references used throughout the project notebooks and report.

---

## References

Breiman, L. (2001).
**Random Forests.**
Machine Learning, 45, 5–32.
[https://doi.org/10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)

Introduces the Random Forest algorithm, an ensemble learning method that improves predictive stability by averaging predictions from multiple decision trees trained on bootstrapped samples.

Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022).
**Why do tree-based models still outperform deep learning on tabular data?**
Advances in Neural Information Processing Systems (NeurIPS).
[https://arxiv.org/abs/2207.08815](https://arxiv.org/abs/2207.08815)

An empirical study demonstrating that tree-based ensemble models remain highly competitive and often outperform deep learning approaches on typical tabular datasets.

Harrell, F. E. (2015).
**Regression Modeling Strategies (2nd ed.).**
Springer.

A comprehensive reference on predictive modeling methodology, including model validation, sample size considerations, and best practices for building reliable statistical prediction models.

Hastie, T., Tibshirani, R., & Friedman, J. (2009).
**The Elements of Statistical Learning (2nd ed.).**
Springer.

A foundational reference in statistical learning theory. Introduces the bias–variance trade-off and the concept of irreducible error, highlighting that a portion of prediction error is inherent to the data-generating process and cannot be eliminated through additional modeling or feature engineering.

LendingClub.
**LendingClub Loan Data Dictionary and Loan Statistics Dataset.**
[LendingClub Data Download Page](https://www.lendingclub.com/info/download-data.action)

The LendingClub data dictionary defines the variables contained in the public loan dataset. All column definitions and loan status semantics used in this project follow the definitions provided in the official LendingClub documentation.

Ng, A. Y. (2000).
**Model Complexity, Goodness of Fit and Diminishing Returns.**
Advances in Neural Information Processing Systems (NeurIPS).

Demonstrates that improvements in model performance tend to diminish as model complexity increases, providing a theoretical basis for observing performance plateaus across increasingly flexible model classes.

Peduzzi, P., Concato, J., Kemper, E., Holford, T. R., & Feinstein, A. R. (1996).
**A simulation study of the number of events per variable in logistic regression analysis.**
Journal of Clinical Epidemiology, 49(12), 1373–1379.
[https://doi.org/10.1016/S0895-4356(96)00236-3](https://doi.org/10.1016/S0895-4356(96)00236-3)

Introduces the widely cited **events-per-variable (EPV)** guideline suggesting approximately 10 outcome events per predictor parameter to ensure stable coefficient estimates in logistic regression models.

Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018).
**CatBoost: Unbiased boosting with categorical features.**
Advances in Neural Information Processing Systems (NeurIPS).
[https://arxiv.org/abs/1706.09516](https://arxiv.org/abs/1706.09516)

Introduces the CatBoost gradient boosting algorithm designed to handle categorical variables while reducing prediction shift during training.

Thomas, L. C., Edelman, D. B., & Crook, J. N. (2017).
**Credit Scoring and Its Applications (2nd ed.).**
SIAM.

A foundational reference for statistical credit risk modeling. Logistic regression remains one of the most widely used models in consumer credit risk prediction due to interpretability and regulatory transparency.

Vittinghoff, E., & McCulloch, C. E. (2007).
**Relaxing the rule of ten events per variable in logistic and Cox regression.**
American Journal of Epidemiology, 165(6), 710–718.
[https://doi.org/10.1093/aje/kwk052](https://doi.org/10.1093/aje/kwk052)

Revisits the EPV guideline and demonstrates that stable models can sometimes be estimated with fewer events per variable, while still emphasizing the importance of adequate sample size in predictive modeling.