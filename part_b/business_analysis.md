# Part B: Business Analysis - Promotion Effectiveness at a Fashion Retail Chain

## Scenario

A fashion retailer operates 50 stores across urban, semi-urban, and rural locations. Each month, the marketing team runs one of five promotions: Flat Discount, BOGO (Buy-One-Get-One), Free Gift with Purchase, Category-Specific Offer, and Loyalty Points Bonus. Stores vary in size, monthly footfall, local competition density, and customer demographics. The company wants to determine which promotion should be deployed in each store each month to maximise the number of items sold.

---

## B1. Problem Formulation — 8 marks

### (a) — 3 marks

Formulate this as a machine learning problem. State clearly: what is the target variable, what are the candidate input features, and what type of ML problem is this? Justify your choice of problem type.

**Target Variable:** items_sold (number of units sold per store per month)

**Candidate Input Features:** Store attributes (store_size, location_type, store_id), Promotion attributes (promotion_type), Temporal attributes (month, season, is_weekend, is_festival), Market attributes (competition_density, footfall), Historical performance (previous month's sales, rolling average sales)

**Problem Type:** Supervised learning regression problem with contextual bandit/recommendation elements.

**Justification:** Regression because items_sold is continuous. Supervised learning because we have labeled historical data. Recommendation context because we need to select the best promotion for each store-month.

### (b) — 3 marks

The company currently measures performance using total sales revenue. Explain why using items sold (sales volume) is a more reliable target variable for this problem. What broader principle does this illustrate about target variable selection in real-world ML projects?

**Why items sold is more reliable than revenue:** Revenue is distorted by discounts (a 50% discount doubles units sold but revenue may stay flat), price variability across products, cannibalization effects, and seasonal markdowns. Items sold provides a standardized, comparable measure of true demand impact.

**Example:** A BOGO promotion might sell 500 items (high volume) but generate only $2,000 revenue, while a Flat Discount sells 300 items but generates $3,000 revenue. Revenue would incorrectly suggest Flat Discount is better.

**Broader Principle:** "Measure what you actually want to optimize, not what is easiest to measure." This illustrates alignment between business objective and target variable.

### (c) — 2 marks

A junior analyst suggests running one single global model across all 50 stores. Propose and justify an alternative modelling strategy that accounts for the fact that stores in different locations respond very differently to the same promotion.

**Proposed Strategy:** Location-stratified modeling - separate models by location_type (urban/semi-urban/rural), or store-cluster models, or mixed-effects models with store-specific intercepts.

**Justification:** Urban customers respond better to BOGO/loyalty points (value-conscious, competitive); Rural customers respond better to free gifts (relationship-driven); Semi-urban respond best to category offers. Different competitive dynamics and customer demographics require localized models.

---

## B2. Data and EDA Strategy — 10 marks

### (a) — 4 marks

The raw data arrives in four separate tables: transactions, store attributes, promotion details, and a calendar (with weekend and festival flags). Describe how you would join these tables. What is the grain of the final modelling dataset (one row = what?), and what aggregations would you perform before modelling?

**Join Strategy:** transactions LEFT JOIN calendar ON date, then LEFT JOIN store_attributes ON store_id, then LEFT JOIN promotion_details ON promotion_type.

**Final Dataset Grain:** One row = one store for one month

**Aggregations Required:** SUM(items_sold) → monthly_items_sold, COUNT(transaction_id) → transaction_count, MODE(promotion_type) → primary_promotion, COUNT(is_weekend) → weekend_days_in_month, SUM(is_festival) → festival_days_in_month. Also create average items per transaction, promotion frequency per store, and rolling 3-month sales average.

### (b) — 4 marks

Describe the EDA you would perform before building a model. Specify at least four analyses or charts, what you would look for in each, and how the findings would influence your feature engineering or modelling decisions.

**Analysis 1 - Promotion Performance by Location:** Grouped bar chart of items_sold by promotion_type faceted by location_type. Look for which promotion works best in each location. Finding: Urban prefers BOGO/loyalty; Rural prefers free gift. Action: Create interaction feature promotion_type × location_type.

**Analysis 2 - Temporal Patterns:** Line plot of monthly items_sold over time colored by location. Look for December peaks and seasonal patterns. Action: Create cyclical features (month_sin, month_cos) and lag features.

**Analysis 3 - Competition Density Impact:** Scatter plot of competition_density vs items_sold colored by promotion_type. Look for whether promotions work better in high-competition areas. Action: Bin competition_density into categories or create threshold features.

**Analysis 4 - Store Size Interaction:** Box plot of items_sold by store_size faceted by promotion_type. Look whether large stores respond differently. Action: Create store_size × promotion_type interaction.

### (c) — 2 marks

You notice that 80% of transactions in the dataset occurred without any promotion. Describe how this imbalance could affect your model and what steps you would take to address it.

**How imbalance affects model:** Model learns "no promotion" well but struggles to distinguish between promotion types (sparse positive examples). High variance causes unstable rankings. Overfitting to noise in the few promotion examples. Poor generalization to new stores/months.

**Steps to address:** Undersample non-promotion data to match promotion count; Oversample promotion data using SMOTE; Assign higher class weights to promotion examples; Two-stage modeling (first predict if promotion is beneficial, then which promotion); Synthetic data generation; Bayesian methods with informative priors; Use lift-over-baseline metrics instead of absolute RMSE.

---

## B3. Model Evaluation and Deployment — 12 marks

### (a) — 4 marks

You have monthly store-level data spanning three years across 50 stores. Describe how you would set up the train-test split. Why is a random split inappropriate here? Which evaluation metrics would you use, and how would you interpret each in the context of this business problem?

**Train-Test Split:** Temporal split - first 30 months for training (83%), last 6 months for testing (17%). No randomization.

**Why random split is inappropriate:** Temporal leakage (future data would leak into training), seasonality distortion (breaks yearly patterns), violates business realism (we predict future from past, not random periods), fails to detect concept drift.

**Evaluation Metrics:**
- RMSE: Typical prediction error in items sold. Answers "What's our typical error margin?"
- MAE: Average absolute error. Easier to explain to stakeholders.
- MAPE: Percentage error. Useful for comparing across different-sized stores.
- Lift over baseline: (Promotion sales - Baseline sales) / Baseline sales. Answers "What incremental value does our model provide?"

**Recommended primary metric:** Lift over baseline + MAE (MAE for absolute accuracy, Lift for business value).

### (b) — 4 marks

After training, the model recommends the Loyalty Points Bonus for Store 12 in December and the Flat Discount for Store 12 in March. Using the concept of feature importance, explain how you would investigate and communicate to the marketing team why the model makes different recommendations for the same store in different months.

**Investigation Steps:**

1. Compare feature values between December and March: is_festival (1 vs 0), month (12 vs 3), days_since_last_promotion (45 vs 20), competition_promotions (high vs low)

2. Examine feature importance from the model: month (32%), is_festival (28%), competition_density (15%), days_since_last_promo (12%)

3. Generate SHAP/LIME explanations: December - is_festival=1 adds +45 items sold for Loyalty Points; March - month=3 and low competition favor Flat Discount

**Communication to Marketing Team:**

December (Holiday Season): Customers buying gifts for others. Loyalty points appeal to gift-givers earning rewards. High footfall encourages repeat visits. → Recommendation: Loyalty Points Bonus

March (Post-Holiday): Customers shopping for themselves. Flat discounts appeal to value-seekers. Lower footfall needs immediate incentive. → Recommendation: Flat Discount

Key driver: Seasonality (month + is_festival account for 60% of decision change)

### (c) — 4 marks

The trained model needs to generate recommendations at the start of every month for all 50 stores without being retrained each time. Describe the end-to-end deployment process: how you would save the model, how new monthly data would be prepared and fed in, and what monitoring you would put in place to detect when the model's performance has degraded and retraining is needed.

**Model Saving:** Use joblib.dump() to save the full pipeline (preprocessor + model). Save metadata with feature list, promotion options, training date, and version.

**Monthly Data Preparation:** At month start, collect store attributes (static), generate calendar features (month, is_festival, is_weekend), pull competition density from external source, query previous month's sales from database, calculate rolling features. Apply same preprocessing steps: cyclical encoding (month_sin/month_cos), days_since_last_promotion, lag features, scaling using saved scaler, one-hot encoding.

**Recommendation Generation:** For each store, predict items_sold for each promotion type by creating copies with different promotion_type values. Select the promotion with highest predicted sales. Output recommendations dataframe.

**Monitoring & Retraining Triggers:**
- Prediction drift: MAPE > 25% for rolling 3 months → investigate
- Data drift: PSI (Population Stability Index) > 0.2 → retrain
- Business impact: Lift over baseline < 5% for 2 consecutive months → retrain
- Promotion adoption: < 80% of recommendations followed → check operations
- Time-based: 6 months since last training → retrain
- New promotion type introduced → retrain
- Significant market change (new competitor) → retrain

**Retraining Process:** Fetch new transactions since last training, combine with historical data, retrain pipeline, validate on recent holdout, compare metrics, deploy if improved, update version and metadata, send alert.

---

## Summary Table

| Section | Key Takeaways |
|---------|---------------|
| B1 | Regression with target=items_sold; hierarchical models by location; items_sold > revenue for measuring promotion effectiveness |
| B2 | Join to store-month grain; analyze location×promotion, seasonality, competition, store size; address 80% non-promotion imbalance via sampling/weighting |
| B3 | Temporal train-test split; use MAE + lift metrics; explain recommendations via feature importance/SHAP; deploy monthly pipeline with monitoring and retraining triggers |
