# Assignment 2 – Diabetes Classification: Report

## Q1. Understanding the data

**What do class 0 and class 1 mean? Is the target balanced? Why does the class distribution matter when evaluating the model?**

Class 0 means the respondent has no diabetes. Class 1 means they have prediabetes or diabetes.

The target is very imbalanced:

| Class | Count | Share |
|---|---|---|
| 0 – no diabetes | 218,334 | 86.1 % |
| 1 – prediabetes/diabetes | 35,346 | 13.9 % |

This matters for evaluation because a "model" that always predicts 0 would already get 86.1 % accuracy and still miss every person with diabetes. Because of this, accuracy looks good even for a useless model. We need metrics that look at class 1 directly, such as precision, recall and F1. The imbalance is also why we used `stratify=y` in the train/test split and `class_weight="balanced"` in the model.

## Q2. Understanding the features

**Choose three features of different types. What do their values mean? Why is looking only at the numbers stored in the CSV not enough to understand a feature?**

- **BMI (numerical):** body mass index in kg/m², ranging from 12 to 98. Higher means more weight relative to height. Values above about 50 are rare (0.9 %) and some may be reporting errors.
- **HighBP (binary categorical):** 0 means the person has not been told they have high blood pressure, 1 means they have.
- **GenHlth (ordered categorical):** self-rated general health, where 1 is excellent, 2 very good, 3 good, 4 fair and 5 poor.

The CSV alone is not enough to understand these features. Every value is stored as a float (`1.0`, `5.0`, `30.0`), so the CSV doesn't tell you whether a number is a measurement, a yes/no flag or a category code. For example, GenHlth = 5 is *worse* health than 1, not "five times as much health". Age = 9 means the age group 60–64, not 9 years old. Income = 8 means ≥ $75k, not $8. The distances between codes are not necessarily equal either. You need the codebook to know the direction, the units and the type of each feature, and that decides which plots and preprocessing are appropriate.

## Q3. EDA

**Which three to five findings were most informative or interesting? What did you observe? What can you conclude, and what can you not conclude?**

1. **General health shows a strong gradient.** The share of class 1 rises steadily with worse self-rated health: 2.5 % (excellent), 7.2 %, 17.8 %, 31.0 % and 37.9 % (poor).
2. **Age.** The share with diabetes rises from 1.4 % in the youngest group (18–24) to about 22 % around ages 70–79, then falls slightly to 18.5 % for 80+. That dip might come from survivorship bias.
3. **Blood pressure and cholesterol.** 24.4 % of people with HighBP have diabetes, against 6.0 % without it. For HighChol it is 22.0 % against 8.0 %.
4. **BMI.** The median is 31 for class 1 and 27 for class 0, so the whole distribution is shifted upward for people with diabetes.
5. **Income and education show a socioeconomic gradient.** The diabetes share falls from about 24–26 % at the lowest income levels to 8 % at the highest. Education shows a similar fall, from about 27–29 % to 9.7 %.

We also observed that MentHlth and PhysHlth are heavily concentrated at 0 days (69 % and 63 % of rows). AnyHealthcare, Sex, NoDocbcCost, Fruits and Veggies show very weak relationships with the target (|r| < 0.07). CholCheck is extremely unbalanced: 96 % of people had a cholesterol check.

**What we can conclude:** in this survey these features are *associated* with diabetes, and they are likely to be useful predictors for the model.

**What we cannot conclude:** that any of them *causes* diabetes. The direction can even be reversed: people with diabetes may rate their health as poor *because* they have diabetes. Confounders such as age also drive several features at once (age is related to blood pressure, cholesterol and difficulty walking). The data is self-reported cross-sectional survey data, so it shows no time order, and it includes 24,206 duplicate rows.

## Q4. TRAIN and TEST

**Why must TEST remain separate while preprocessing and the model are fitted?**

TEST is supposed to simulate new, unseen data. Anything fitted on TEST leaks information about it into the model. That includes the StandardScaler's mean and standard deviation and the model's coefficients. The evaluation would then be optimistically biased and wouldn't tell us how the model performs on genuinely new people.

So every preprocessing step and the model are fitted on TRAIN only (`model.fit(x_train, y_train)`), and only `transform`/`predict` are applied to TEST. Putting everything in one Pipeline guarantees this. For the same reason TEST must not be resampled or balanced: it should reflect the real class distribution.

## Q5. Classification errors and metrics

**Explain false positives and false negatives for this problem. What do accuracy, precision and recall tell you? Why is accuracy alone not enough?**

Confusion matrix on TEST (n = 50,736) using `model.predict()`, i.e. the default threshold 0.50:

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **True 0** | TN = 31,739 | FP = 11,928 |
| **True 1** | FN = 1,689 | TP = 5,380 |

- **False positive (11,928):** the model flags a person without diabetes as having it. The consequence is unnecessary worry and follow-up tests, but a blood test would clear it up.
- **False negative (1,689):** the model misses a person who has diabetes. This is usually more serious: the condition goes undiagnosed and untreated, with a risk of complications.

| Metric | Value | Meaning |
|---|---|---|
| Accuracy | 0.732 | Share of all predictions that are correct |
| Precision | 0.311 | Of the people flagged as diabetic, 31 % actually are |
| Recall | 0.761 | Of the people who actually have diabetes, the model finds 76 % |
| F1 | 0.441 | Harmonic mean of precision and recall |

Accuracy alone isn't enough because of the class imbalance. Always predicting 0 gives 0.861 accuracy, which is *higher* than our model, yet it has recall 0 and finds nobody with diabetes. Accuracy is dominated by the large class 0 and hides how the model performs on the class we care about.

## Q6. Probability and threshold

**Compare thresholds 0.50 and 0.40. What happened to false positives, false negatives, precision and recall? What did not change?**

**Threshold comparison table**

| Threshold | Accuracy | Precision | Recall | F1 | False positives | False negatives |
|---|---|---|---|---|---|---|
| 0.50 | 0.732 | 0.311 | 0.761 | 0.441 | 11,928 | 1,689 |
| 0.40 | 0.665 | 0.274 | 0.854 | 0.415 | 15,979 | 1,033 |

Lowering the threshold from 0.50 to 0.40 means more people are classified as class 1. As a result:

- **False positives increased** by 4,051 (11,928 → 15,979).
- **False negatives decreased** by 656 (1,689 → 1,033).
- **Recall increased** (0.761 → 0.854): we find more of the people who actually have diabetes.
- **Precision decreased** (0.311 → 0.274): a larger share of the positive predictions are false alarms.

Accuracy (0.732 → 0.665) and F1 (0.441 → 0.415) also dropped, because most of the people who switched to class 1 are really class 0.

**What did not change:** the trained model itself, meaning its coefficients, the predicted probabilities from `predict_proba`, and the ranking of people from lowest to highest risk, so the ROC-AUC stays at 0.82. The data, the train/test split and the preprocessing didn't change either. Changing the threshold only changes how probabilities are turned into class labels.

## Q7. Conclusion

**What did you learn from the data and the Logistic Regression model? Give at least two limitations.**

The EDA showed that general health, age, high blood pressure, high cholesterol, BMI, difficulty walking and income are the features most strongly associated with diabetes. Several lifestyle features (fruits, vegetables, smoking) and healthcare access showed only weak relationships.

Logistic Regression ranks people reasonably well (ROC-AUC 0.82), but because of the class imbalance it can't get both high precision and high recall at once. The threshold is a trade-off decision. For screening, a lower threshold (higher recall) makes sense because missing a person with diabetes is usually costlier than a false alarm.

**Limitations:**

1. **Self-reported, cross-sectional survey data.** Answers can contain reporting errors, and since there is no time order the data shows only associations, not causation.
2. **The target lumps prediabetes and diabetes together,** and the dataset has 24,206 duplicate rows that may end up in both TRAIN and TEST, which can make the results slightly optimistic.
3. **Logistic Regression is linear in the log-odds.** Treating ordered categories such as Age and Income as numbers assumes equal distances between categories, and interactions between features aren't modelled.
4. **Precision is low (about 0.3).** Most positive predictions are false alarms, so the model works as a screening aid, not a diagnostic tool.
5. **Only one random train/test split was used,** with no cross-validation, and the thresholds were not chosen on a separate validation set.

## G1. How was the work divided among group members, and what did each member contribute?

We did not divide the work between us. The entire assignment was completed during one in-person meeting where all group members sat together and worked through every part as a group, from the EDA and preprocessing to the model, the evaluation and the written answers. Nobody worked on anything alone, and everyone contributed equally to the whole solution.

## G2. How did your group make sure that everyone understood the complete solution, not only the part they worked on?

Since we solved the whole assignment together in one meeting, there were no separate parts. Everyone saw every step as it was done, and anyone could step in, ask questions and share their ideas along the way. By discussing each task and sharing our knowledge with each other, we made sure that every member understood the complete solution.
