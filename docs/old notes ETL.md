# Raw Layer – Structural Governance and Eligibility

The data is provided as two independent source files:

- 2007–2014 vintage (train)
- 2015 vintage (test)

The training vintage defines the admissible feature universe.  
The testing vintage is aligned to it.

> **Note:** Governance decisions below are based on the full audit outputs stored in `artifacts/audit` and the LendingClub Data Dictionary accompanying the dataset. Variable eligibility and lifecycle interpretation follow the documented field definitions.

---

## 1. Initial Data Inspection – Structural Baseline

| Metric | Train (2007–2014) | Test (2015) |
|--------|-------------------|-------------|
| Rows | **466,285** | **421,094** |
| Columns | **75** | **74** |
| Memory Usage | ~389 MB | ~323 MB |
| Numeric Columns | 49 | 51 |
| Object / String Columns | 26 | 23 |
| Fully Null Columns | **17** | 0 |
| Constant Columns | 2 (`policy_code`, `application_type`) | 1 (`policy_code`) |
| Columns >50% Null | 4 | Several (overlapping high-missing fields) |
| String-like Columns (post dtype scan) | 22 | 23 |
| Columns present only in one split | 1 (`Unnamed: 0`, train only) | 0 |
| Dtype mismatches | 1 (`verification_status_joint`) | — |

---

### Fully Null Variables (Training-Defined Constraint)

17 columns — approximately **23% of the training feature space** — are entirely null in the training data.
According to the LendingClub Data Dictionary, these fields represent later-introduced bureau segmentation metrics and joint-application variables, including:

- Short-horizon account opening activity (6–24 month windows)
- Installment-specific balance and utilization measures
- Segmented inquiry counts
- Joint borrower income and verification attributes

Their absence is structural: they were not reported during the period covered by the training data.
In the testing data, these columns remain 94–99% null, indicating limited late-period reporting rather than stable feature availability. This reflects reporting expansion rather than a meaningful shift in information availability.
The model is trained exclusively on the training data. A variable that does not exist during training cannot be estimated or validated. Allowing these fields to appear only in the testing data would introduce features the model has never observed.

All 17 columns are therefore removed from both datasets.

---

### Constant Variables and Structural Artifacts

Two columns — `policy_code` and `application_type` — exhibit zero variance in the training data.

`policy_code` reflects an internal platform designation, and `application_type` indicates whether the application is individual or joint. In the training data, `application_type` is exclusively `INDIVIDUAL`, and `policy_code` does not vary. A variable without variation carries no informational content and cannot contribute to model estimation.
Both columns are therefore removed from the datasets.

`Unnamed: 0` appears only in the training file.
It is an exported index column introduced during CSV generation and does not correspond to a documented field in the Data Dictionary.
It is therefore also removed.

---

### Credit Timing Variables

- `mths_since_last_delinq` (Training: **~53.7%** null | Testing: **~48.4%** null)  
- `mths_since_last_major_derog` (Training: **~78.8%** null | Testing: **~70.9%** null)  
- `mths_since_last_record` (Training: **~86.6%** null | Testing: **~82.3%** null)

These variables measure the number of months since the borrower last experienced a negative credit event.  
This includes missed payments, serious credit impairments, or public records such as bankruptcies or legally recorded unpaid debts.
They capture the recency of adverse credit behavior at the time of application.
A value of `0` does not indicate absence. It indicates that the event occurred within the past month. Smaller numeric values represent more recent adverse events.
Missing values (`NaN`) indicate that no such event has occurred in the borrower’s recorded credit history.
The high null percentages in both training and testing data reflect structural absence — many borrowers have no prior adverse events — rather than incomplete reporting.
Recency of prior credit issues is structurally informative. A borrower with a recent missed payment carries different risk implications than a borrower with no prior credit impairments.
Dropping these variables would discard meaningful signal. Leaving them as `NaN` would conflate structural absence with missing data.

Handling:

1. Create a binary indicator: `has_<variable>`  
2. Replace missing values in the original column with sentinel value `9999`

The sentinel value lies well beyond any realistic observed month count. This preserves ordinal interpretation: larger values indicate more distant events, and the sentinel consistently represents “no prior event” without colliding with valid data.

Absence is encoded explicitly while preserving numeric ordering.

---

## 2. Submission-Time Boundary

Prediction is defined at the moment the borrower submits the application.

The admissible information set therefore consists of:

- Borrower-declared application inputs  
- Credit bureau data retrieved at submission  

No later platform actions, pricing decisions, funding outcomes, or servicing updates are allowed to enter the training feature space.

A variable is eligible only if:

1. It is available at submission  
2. It does not encode underwriting, pricing, funding, or servicing outcomes  
3. It contains usable observations in the training data  

The training data defines the admissible feature universe.  
The testing data is aligned to that universe.  
Additional reporting fields that appear in the testing data but are absent in training are removed to preserve temporal consistency.

Variables are excluded from the modeling feature set when they violate this boundary:

- **Underwriting and pricing outputs** (`grade`, `sub_grade`, `int_rate`, `installment`, `verification_status`) reflect LendingClub’s internal risk assessment.  
  Including them would embed prior decision logic into the model, creating circularity rather than independent inference.

- **Funding and allocation fields** (`funded_amnt`, `funded_amnt_inv`) encode platform or investor response after review.  
  These are not borrower characteristics and would introduce selection bias.

- **Servicing and lifecycle variables** (`total_pymnt*`, `recoveries`, `out_prncp*`, `last_pymnt_*`, etc.) are observed only after issuance.  
  Including them would constitute leakage, as they encode post-origination outcomes.

These variables are removed from the `feature_base` dataset used for training.
They are retained in the `clean` dataset when structurally valid, so that benchmarking, diagnostic comparison, and audit checks remain possible without contaminating the modeling input layer.
This boundary governs model training while preserving contextual information and temporal alignment.

---

## 3. Feature Classification Overview – Application Submission Prediction Point

| Column | Category | Action | Rationale |
|--------|----------|--------|-----------|
| `annual_inc`, `dti` | Application input | Keep | Declared at submission |
| `loan_amnt`, `term`, `purpose` | Application input | Keep | Application-time inputs |
| `home_ownership`, `emp_length` | Application input | Keep | Observable at submission |
| `open_acc`, `total_acc` | Credit snapshot | Keep | Credit file state at application |
| `inq_last_6mths` | Credit snapshot | Keep | Recent inquiry activity |
| `delinq_2yrs` | Credit snapshot | Keep | Historical delinquency count |
| `pub_rec` | Credit snapshot | Keep | Public record count |
| `collections_12_mths_ex_med` | Credit snapshot | Keep | Recent collections history |
| `revol_bal`, `revol_util` | Credit snapshot | Keep | Revolving exposure and utilization |
| `acc_now_delinq` | Credit snapshot | Keep | Current delinquency indicator |
| `tot_cur_bal`, `tot_coll_amt` | Credit snapshot | Keep | Aggregate credit balances |
| `mths_since_last_delinq` | Credit timing | Keep (Transformed) | Structural absence encoded explicitly |
| `mths_since_last_major_derog` | Credit timing | Keep (Transformed) | Structural absence encoded explicitly |
| `mths_since_last_record` | Credit timing | Keep (Transformed) | Structural absence encoded explicitly |
| `loan_status` | Target | Target Only | Defines outcome |
| `grade`, `sub_grade` | Platform signal | Benchmark Only | Origination-time assessment |
| `int_rate`, `installment` | Pricing output | Benchmark Only | Determined during underwriting |
| `earliest_cr_line` | Credit timestamp | Exclude | Cannot derive credit age without submission timestamp |
| `verification_status`, `is_inc_v` | Underwriting outcome | Exclude | Determined after submission |
| `funded_amnt`, `funded_amnt_inv` | Funding decision | Exclude | Platform allocation decision |
| `issue_d` | Lifecycle timestamp | Exclude | Occurs after submission |
| `initial_list_status` | Workflow variable | Exclude | Listing outcome |
| Lifecycle updates (`last_credit_pull_d`, `last_fico_range_*`) | Bureau update | Exclude | Updated after submission |
| Servicing variables (`total_pymnt*`, `recoveries`, `out_prncp*`, `last_pymnt_*`, `pymnt_plan`) | Post-origination | Exclude | Observed after issuance |
| Structurally null 2007–2014 variables | Vintage-dependent | Drop | No training observations |
| Identifiers (`id`, `member_id`, `url`, `Unnamed: 0`) | Structural | Drop | Non-predictive artifacts |
| Constants (`policy_code`, `application_type`) | Structural | Drop | Zero variance |
| Free-text (`desc`, `emp_title`, `title`) | Unstructured | Drop | High cardinality; unstructured; Outside structured scope |
| Geographic proxies (`addr_state`, `zip_code`) | Application input | Drop | Removed to reduce proxy discrimination risk |

<br>
<br>

> **Future enhancement:**  
If a verified application submission timestamp becomes available, `earliest_cr_line` can be converted into `credit_age_years` and reconsidered under the temporal availability rule.

---

## 4. String Column Normalization

After structural removal and boundary classification, the remaining object-based fields require normalization.
These columns represent borrower-declared categories, platform classifications retained for benchmarking, and lifecycle timestamps that remain excluded from modeling.

Normalization ensures:

- Consistent representation across training and testing data  
- Stable typing (categorical, numeric, datetime)  
- Removal of formatting variation without altering meaning  

This step standardizes representation only.  
Feature eligibility is governed in the previous section.

The transformations below are applied consistently across both datasets.

---

| Column | Category | Transformation Action | Training Eligibility | Rationale |
|---------|------------|------------------------|----------------------|------------|
| `term` | Structured categorical | Strip whitespace → extract numeric term (36 / 60) → rename to `term_months` → convert to int | Keep | Contractual term declared at submission |
| `emp_length` | Ordinal categorical | Map to integer scale 0–10 (`<1` → 0, `10+` → 10); retain NaN | Keep | Missing indicates unknown tenure, not structural absence |
| `home_ownership` | Categorical | Standardize casing; map `NONE` → `OTHER`; normalize representation (strip/case/mapping) | Keep | Borrower-declared attribute |
| `purpose` | Categorical | Standardize casing; normalize representation (strip/case/mapping) | Keep | Borrower-declared loan purpose |
| `grade` | Platform risk signal | Standardize casing; normalize representation (strip/case/mapping) | Retain (Benchmark Only) | Assigned during underwriting |
| `sub_grade` | Platform risk signal | Standardize casing; normalize representation (strip/case/mapping) | Retain (Benchmark Only) | Granular underwriting signal |
| `verification_status` | Underwriting outcome | Standardize casing; normalize representation (strip/case/mapping) | Exclude | Determined during review process |
| `initial_list_status` | Workflow variable | Standardize casing; normalize representation (strip/case/mapping) | Exclude | Platform listing outcome |
| `pymnt_plan` | Servicing indicator | Map `n` → 0, `y` → 1 | Exclude | Post-origination servicing flag |
| `issue_d` | Lifecycle timestamp | Convert month-year string to datetime | Exclude | Occurs after submission |
| `last_credit_pull_d` | Lifecycle update | Convert month-year string to datetime | Exclude | Subsequent bureau update |
| `last_pymnt_d` | Servicing timeline | Convert month-year string to datetime | Exclude | Post-origination variable |
| `next_pymnt_d` | Servicing timeline | Convert month-year string to datetime | Exclude | Post-origination variable |
| `loan_status` | Target | No transformation at this stage | Keep (Target Only) | Outcome definition; cohort defined later |

---

## 5. Transformation Execution

The following sequence implements the governance rules defined above.

Execution steps:

- Create a stable internal identifier `row_id` during transformation so rows can be traced across the governed outputs (`clean` and `feature_base`) and any downstream audit or evaluation artifacts.
- Remove structurally ineligible columns (fully null in training data, constants, identifiers, free-text fields, geographic proxies, and post-origination variables).
- Align testing data to the training-defined column universe before type normalization.
- Apply credit-timing encoding (`has_*` indicators + `9999` sentinel) for the `mths_since_last_*` variables.
- Normalize remaining string fields using the column-level rules defined in Section 4.
- Convert month-year strings to datetime for lifecycle fields (structural consistency only; these remain excluded from modeling).

Outputs are persisted as:

- `clean` — governed and normalized dataset; benchmark variables retained but not eligible for training.
- `feature_base` — training-eligible predictors plus target; leakage and post-submission fields removed.


LATEST VERSION

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
| Rows | **466,285** | **421,094** |
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

High null percentages therefore represent **structural absence rather than incomplete reporting**.

Dropping these variables would remove meaningful risk signal, while leaving them as `NaN` would conflate structural absence with missing data.

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
| Bureau updates (`last_credit_pull_d`, `last_fico_range_low`, `last_fico_range_high`) | Post-submission | Exclude | Reflect bureau refresh after origination |
| Servicing variables (`out_prncp`, `out_prncp_inv`, `total_pymnt`, `total_pymnt_inv`<br>`total_rec_prncp`, `total_rec_int`, `total_rec_late_fee`<br>`recoveries`, `collection_recovery_fee`<br>`last_pymnt_d`, `next_pymnt_d`, `pymnt_plan`) | Post-origination | Exclude | Observed only after loan issuance; would introduce outcome leakage |
| Structurally null variables (`annual_inc_joint`, `dti_joint`, `verification_status_joint`<br>`open_acc_6m`, `open_il_6m`, `open_il_12m`, `open_il_24m`<br>`mths_since_rcnt_il`, `total_bal_il`, `il_util`<br>`open_rv_12m`, `open_rv_24m`, `all_util`<br>`inq_fi`, `total_cu_tl`, `inq_last_12m`, `max_bal_bc`) | Vintage dependent | Drop | Absent in training regime |
| Identifiers (`id`, `member_id`, `url`) | Structural | Drop | Non-predictive artifacts |
| Structural variables (`policy_code`, `application_type`) | Structural | Drop | `policy_code` is constant across both datasets; `application_type` has zero variance in the training data and only varies in the testing vintage |
| Free-text (`desc`, `emp_title`, `title`) | Unstructured | Drop | High-cardinality text outside structured modeling scope |
| Geographic proxies (`addr_state`, `zip_code`) | Application input | Drop | Removed to reduce proxy discrimination risk |

<br>
<br>

> **Note on `earliest_cr_line`:**  
> This variable records the month and year of the borrower’s earliest credit account and is commonly used to derive **credit history length**, a known predictor of default risk.  
> However, the LendingClub dataset does not provide the **exact application submission timestamp**. Without that reference point, the borrower’s credit age at the moment of application cannot be calculated reliably.   
> Because this project enforces a strict **submission-time information boundary**, `earliest_cr_line` is excluded from the modeling feature space.
> **Future enhancement:**  
> If a verified application submission timestamp becomes available, `earliest_cr_line` can be converted into `credit_age_years` and reconsidered under the temporal availability rule.

---

# 4. String Column Normalization

After structural removal and boundary classification, remaining object-based variables require normalization.

Normalization ensures:

- consistent representation across datasets  
- stable typing  
- removal of formatting variation without altering meaning  

---

| Column | Category | Transformation Action | Training Eligibility | Rationale |
|------|----------|----------------------|---------------------|-----------|
| `term` | Structured categorical | Strip whitespace → extract numeric term (36 / 60) → rename `term_months` → convert to int | Keep | Contractual term |
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
- Normalize categorical string fields
- Convert month-year timestamps to datetime

Outputs are persisted as:

- **`clean`** — governed dataset with benchmark variables retained
- **`feature_base`** — modeling dataset containing only training-eligible predictors and the target
