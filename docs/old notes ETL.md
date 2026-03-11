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
