# Loans at Risk: Capturing Default — Extract · Transform · Load

## Purpose

This notebook implements the ETL layer of the LendingClub default prediction project.  
The training window covers **2007–2014**; the test window covers **2015**.

The objective is not analysis. It is structural control.

Raw CSV extracts are converted into datasets that are suitable for downstream modeling under a clearly defined constraint: prediction is made at the moment of application submission.

“Structurally reliable” in this context means:

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

The transformed datasets are persisted in Parquet format.

Two layers are materialized:

- **Clean dataset** → structurally aligned variables  
- **Feature-base dataset** → submission-time eligible variables plus target  

Schema identity between training and test is validated prior to persistence.
These outputs form the controlled input layer for EDA and modeling.

---

## Scope Boundary

Feature engineering, modeling, evaluation, and decision logic are handled in subsequent notebooks.
This notebook exists to ensure that those stages operate on data that is structurally sound, temporally consistent, and reproducible.

## Raw Source Normalization

Before we can use the dataset we need to make sure that we actually deal with structural data and that all files share the same schemas. 

Source inspection confirms that all LendingClub export files share the same
non-tabular artifacts and the same header schema. Normalization can therefore
be applied consistently across all source files before concatenation.

## Raw Dataset Construction

The LendingClub data is distributed as several CSV archives covering different time periods. These files are not strict tabular exports. They contain additional rows such as prospectus notices, blank lines, summary totals, and occasional internal headers.

Before the data can be treated as a dataset, these structural artifacts must be removed.

### Source normalization

Each archive file is processed line-by-line and written to a staging dataset. Rows that do not belong to the tabular structure are excluded, leaving only loan-level observations with the original column layout preserved.

The normalized files are written to the **staging layer**.

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

## Initial Data Inspection

This step examines the raw dataset.

The objective is to map its structural characteristics before any transformation rules are applied.

The inspection evaluates:

- Column count and schema alignment  
- Datatype assignments and object-encoded numerics  
- Fully null and constant columns  
- Patterns of missingness, including variables where null reflects structural absence  
- Mixed-type inconsistencies  
- Identifier fields and exported artifacts  

No variables are removed and no values are altered at this stage.

Findings from this inspection determine which columns require exclusion, normalization, or explicit encoding during transformation. Every later modification is grounded in observations documented here.

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

These variables did not exist during the training period.

Although a small fraction of observations appear in the testing vintage, they remain **94–99% null**, indicating partial late-period reporting rather than stable availability.

A model trained on the 2007–2014 regime cannot estimate variables that did not exist in that regime.

All **17 structurally null columns** are therefore removed from both datasets.

---

## Constant Variables and Structural Artifacts

Two columns exhibit zero variance in the training data:

- `policy_code`
- `application_type`

`policy_code` is an internal LendingClub platform designation and contains no variation.

`application_type` indicates whether an application is submitted by an individual borrower or jointly.

In the **2007–2014 training data all applications are individual**.  
Joint applications appear only in the later testing vintage and rely on borrower attributes that are structurally absent from the training data.

Including this variable would therefore introduce a borrower structure that never occurs in the training regime.

Both variables are removed.

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

Missing values indicate that **no such event exists in the borrower’s recorded credit history**.

High null percentages therefore represent **structural absence** rather than **incomplete reporting**.

Dropping these variables would remove meaningful risk signal, while leaving them as `NaN` would treat structural absence as missing data.

Handling:

1. Create a binary indicator `has_<variable>`
2. Replace missing values in the original variable with sentinel value **9999**

The sentinel lies well beyond any realistic month count and preserves ordinal interpretation.

---

## Bureau Reporting Expansion

A broader group of bureau variables exhibits **partial historical reporting**.

Approximately **15% of observations in the training vintage are missing**, while the same fields are **fully populated in the 2015 testing vintage**.

This pattern reflects **expansion of credit bureau reporting coverage over time**, not a transformation error.

These variables remain valid predictors because the information they contain exists at application time.

To separate **information content** from **reporting availability**, an additional binary indicator is created for each affected variable:

`has_<variable>`

Interpretation:

- `has_* = 0` → field not reported in that vintage  
- `has_* = 1` → value observed  

The original numeric values remain unchanged.

This prevents historical reporting gaps from silently introducing a temporal signal into the model.

---

# 2. Submission-Time Boundary

Prediction is defined at the moment the borrower submits the application.

The admissible information set therefore consists of:

- borrower-declared application inputs  
- credit bureau data retrieved at submission  

No later platform actions, pricing decisions, funding outcomes, or servicing updates are allowed to enter the training feature space.

A variable is eligible only if:

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

After structural removal and boundary classification, remaining object-based variables require normalization.

Normalization ensures:

- consistent representation across datasets  
- stable typing  
- removal of formatting variation without altering meaning  

Some numeric variables in the raw export are stored as percentage strings rather than numeric values.

Two examples are **`int_rate`** (loan interest rate) and **`revol_util`** (revolving credit utilization).  
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

# Notebook Conclusion – Data Cleaning and Feature Base Construction

This notebook establishes the transformation layer required for the **Loans at Risk: Capturing Default** project.

The governing constraint is that prediction must rely **only on information available at the moment a borrower submits a loan application**. All transformation decisions follow from that boundary.

The raw LendingClub exports contain **111 variables** across both dataset partitions.  
Through structural governance and transformation, these are reduced to two controlled dataset layers:

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