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
[LendingClub Data Download Page](https://web.archive.org/web/20160911194705/https://www.lendingclub.com/info/download-data.action)

The LendingClub data dictionary defines the variables contained in the public loan dataset. All column definitions and loan status semantics used in this project follow the definitions provided in the official LendingClub documentation.

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

Seven Pillars Institute. (2019).
**Peer-to-peer lending: Lending Club.**
Seven Pillars Institute.
[https://mail.sevenpillarsinstitute.org/peer-to-peer-lending-lending-club/](https://mail.sevenpillarsinstitute.org/peer-to-peer-lending-lending-club/)

Provides an overview of LendingClub’s peer-to-peer lending model, describing how the platform connects borrowers with investors, assigns risk grades, and facilitates capital allocation without primarily taking credit risk on its own balance sheet.

Thomas, L. C., Edelman, D. B., & Crook, J. N. (2002).
**Credit Scoring and Its Applications (1st ed.).**
SIAM.

A foundational reference for statistical credit risk modeling. Logistic regression remains one of the most widely used models in consumer credit risk prediction due to interpretability and regulatory transparency.

Vittinghoff, E., & McCulloch, C. E. (2007).
**Relaxing the rule of ten events per variable in logistic and Cox regression.**
American Journal of Epidemiology, 165(6), 710–718.
[https://doi.org/10.1093/aje/kwk052](https://doi.org/10.1093/aje/kwk052)

Revisits the EPV guideline and demonstrates that stable models can sometimes be estimated with fewer events per variable, while still emphasizing the importance of adequate sample size in predictive modeling.