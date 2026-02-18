loan_status_counts = (
    df_clean_training_data["loan_status"]
    .value_counts(dropna=False)
    .sort_values(ascending=False)
)

loan_status_counts


loan_status_distribution = (
    df_clean_training_data["loan_status"]
    .value_counts(dropna=False)
    .to_frame(name="count")
)

loan_status_distribution["percentage"] = (
    loan_status_distribution["count"] 
    / loan_status_distribution["count"].sum() 
    * 100
).round(2)

loan_status_distribution