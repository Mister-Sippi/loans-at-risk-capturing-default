# Loans at Risk: Capturing Default

---

## Executive Summary

This report evaluates whether borrower-level information available at the time of application can improve lending decisions relative to a lender’s internal grading system.

A gradient boosting model (CatBoost) improves borrower risk ranking (AUC 0.723 vs 0.692), indicating that default risk can be partially captured using application-time data. However, this improvement does not translate uniformly into better outcomes. Its value depends on how model outputs are translated into lending decisions.

The key result is that model value is concentrated in the **mid-risk region**, where borrower quality is most uncertain and decisions are most sensitive to ranking. In this region, improved ordering of borrowers leads to meaningful differences in capital allocation. Outside this region, the effect is limited: at low acceptance levels both strategies lose money, while at high acceptance levels most borrowers are accepted and differences diminish.

This leads to a central conclusion: model deployment is not primarily a modeling problem, but a **policy problem**. The model provides a mechanism to control trade-offs between default exposure and lending volume, but its effectiveness depends on selecting an appropriate operating threshold.

---

## Problem Context

Lending decisions must be made under uncertainty using only the information available at the time a borrower applies. The objective is not simply to predict default, but to determine whether available information can be structured in a way that leads to better decisions than the lender’s existing grading system.

The analysis is conducted under realistic constraints. Only issued loans are observed, and rejected applications are not available. This restricts the evaluation to the population of loans that were historically accepted. The goal is therefore not to reproduce the entire underwriting process, but to assess whether risk can be more effectively managed within the accepted population.

All features are restricted to application-time information. This ensures that the model operates under the same information constraints as the original lending decision.

---

## Data and Portfolio Dynamics

The dataset reflects real lending activity across multiple vintages, capturing both borrower characteristics and realized outcomes.

![Portfolio Evolution](portfolio_evolution.png)

**Figure 1. Portfolio evolution over time.**  
Loan issuance increases substantially over the observation period, reflecting the growth of the platform. Default rates exhibit cyclical variation but remain broadly stable relative to the expansion in volume.

This stability is important. It indicates that the dataset represents a consistent lending environment rather than a shifting or unstable process. As a result, differences in outcomes can be interpreted as differences in borrower risk rather than artifacts of changing conditions.

At the same time, several variables exhibit systematic missingness in earlier vintages. These patterns align with changes in reporting practices rather than borrower behavior. Treating these patterns correctly is essential to avoid introducing spurious signals into the model.

---

## Model Performance

The model improves the ability to distinguish between higher-risk and lower-risk borrowers.

![ROC Curve](report_roc_curve_test.png)

**Figure 2. ROC curve comparison.**  
The model consistently outperforms the baseline grading system, indicating stronger separation between defaulting and non-defaulting borrowers.

However, improved ranking alone is not sufficient. Lending decisions depend on calibrated probabilities, not just ordering.

![Calibration Curve](report_calibration_curve_test.png)

**Figure 3. Calibration curve.**  
Predicted probabilities align reasonably well with observed default rates across most of the range. This supports their use in threshold-based decision policies.

Taken together, these results indicate that the model produces meaningful and interpretable probability estimates. However, they do not yet determine whether these estimates lead to better decisions.

---

## Risk Structure

The model organizes borrowers into distinct risk segments that correspond to observed outcomes.

![Risk Stratification](report_risk_stratification_test.png)

**Figure 4. Risk stratification.**  
Default rates increase consistently across predicted risk bands. This confirms that the model captures economically meaningful differences in borrower quality.

The key implication is that the model does not create new information, but reorganizes existing borrower characteristics into a clearer structure. This structure becomes useful only when translated into decisions.

---

## Decision Analysis

To evaluate decision impact, predicted probabilities are converted into acceptance policies using thresholds.

![Policy Frontier](report_policy_frontier_test.png)

**Figure 5. Policy frontier.**  
At comparable acceptance rates, the model consistently achieves lower default rates than the baseline. This indicates that improved ranking can translate into more efficient allocation of capital.

However, the magnitude of this improvement varies across the acceptance spectrum. This highlights that the value of the model is conditional, not uniform.

---

## Economic Interpretation

The practical value of improved decisions depends on their economic consequences.

![Economic Comparison](report_proxy_economic_comparison.png)

**Figure 6. Proxy economic comparison.**  
Model advantage varies across thresholds, with the largest gains concentrated in the mid-range acceptance region.

At low acceptance levels, both strategies lose money because too few loans are approved. The model reduces losses by rejecting more high-risk borrowers, but does not create profitability.

At high acceptance levels, most borrowers are approved, and improved ranking has limited impact. The decision boundary becomes less selective, reducing the influence of model differences.

The most important region is the middle, where decisions are marginal and borrower quality is uncertain.

![Operating Points](report_policy_comparison_table.png)

**Figure 7. Top operating points.**  
In this region, the model produces measurable improvements in capital allocation by reducing default exposure while retaining more performing loans.

---

## Key Insight

The model improves capital allocation primarily in the **mid-risk region**, where the baseline system is least precise.

This region represents the area where:
- borrower quality is ambiguous  
- decisions are sensitive to ranking  
- small improvements have large effects  

Outside this region, either:
- risk is too high to justify lending, or  
- most borrowers are accepted regardless of ranking  

This explains why model improvements are concentrated rather than uniform.

---

## Conclusion

The model does not introduce new information about borrower risk. Instead, it reorganizes existing information into a structure that enables more precise control over lending decisions.

Its value is conditional and depends on how it is used. When applied within the appropriate operating range, it improves capital allocation. Outside that range, its impact is limited.

This implies that successful deployment is not primarily a technical challenge, but a **policy design problem**. The model provides a tool for managing trade-offs between risk and opportunity, but it does not remove uncertainty.

The effectiveness of the system therefore depends on selecting and maintaining an operating threshold that reflects the desired balance between default exposure and lending volume.