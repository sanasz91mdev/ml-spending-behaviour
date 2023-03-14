# Predicting Spending Interest Report

[[_TOC_]]
## Data Summary:


|  Data | Value  |
|--|--|
| Problem Type | Multiclass Classification|
| Number of rows on training data | 2000 |
|Training - to - Test Ratio  | 80 to 20 |
| Feature count |  15 |
| Number of categorical features |  5 |
| Number of Numeric features | 10 |
| Target Variable| Target-MCC1|
| Target Variable Type| Categorical|
| Number of classes in Target| 10|
| Values of classes in Target| 4111, 4814, 5411, 5621, 5691, 5812, 5983,7032, 5999, 7832 |

## MCCs Lookup - Definition

| Merchant category code | Description                               |
| --------------------- | ----------------------------------------- |
| 7832                  | Motion Picture Theaters                   |
| 5411                  | Grocery Stores                            |
| 5812                  | Eating places and Restaurant              |
| 7032                  | Sporting and Recreational Camps           |
| 5999                  | Miscellaneous and Specialty Retail Stores |
| 4111                  | Transportation                            |
| 5691                  | Men’s and Women’s Clothing Stores         |
| 5983                  | Fuel Oil, Liquefied Petroleum             |
| 4814                  | Telecommunication services                |
| 5621                  | Women’s Ready-to-Wear Stores              |



## Sample Top 5 Rows

| AgeGroup | Marital Status | Day     | Gender | City | Avg-7832 | Avg-5411 | Avg-5812 | Avg-7032 | Avg-5983 | Avg-4111 | Avg-5999 | Avg-5691 | Avg-4814 | Avg-5621 | Target-MCC1 |
|----------|---------------|---------|--------|------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|-------------|
| 20       | Single        | Weekday | F      | KHI  | 10      | 25      | 20      | 10      | 5       | 5       | 5       | 5       | 15      | 0       | 5411        |
| 20       | Single        | Weekday | F      | KHI  | 0       | 10      | 0       | 65      | 25      | 0       | 0       | 0       | 0       | 0       | 7032        |
| 30       | Single        | Weekday | M      | KHI  | 10      | 5       | 0       | 0       | 32      | 0       | 15      | 0       | 0       | 38      | 5621        |
| 40       | Single        | Weekday | F      | ISL  | 0       | 22      | 10      | 3       | 3       | 5       | 7       | 2       | 0       | 48      | 5621        |
| 30       | Married       | Weekday | F      | ISL  | 10      | 25      | 20      | 10      | 5       | 5       | 5       | 5       | 15      | 0       | 5411        |


## Cardinality:

Refers to the number of possible values that a categorical feature can assume:

|  Feature | Number of Unique Values| Unique Values|
|--|--|--|
| AgeGroup | 3| 20s, 30s, 40s|
|City  | 3 |KHI, LHR, ISL|
| Day |  2 |Weekday, Weekend|
| Gender | 2|Male, Female|

## ML Algorithm

|  Algorithm | Value|
|--|--|
| Problem Type | Multiclass Classification|
| Name| Random Forest| 

|Hyperparameter| Values|
|--|--|
| n_estimators| 100|
| random_state| 42|
| max_depth| none|


## Model Accuracy


| Metric | Value |
|--|--|
|  Model Accuracy| 0.93 |

## Feature Importance

|Feature  | Importance |
|--|--|
|AgeGroup| 0.007335352846323242|
|MaritalStatus| 0.003780795258948365|
|Day| 0.006009442644614128|
|Gender| 0.004859433183447916|
|City| 0.006545369156478366|
|Avg-7832| 0.46691761668599463|
|Avg-5411| 0.03854568023937666|
|Avg-5812| 0.01321235913167839|
|Avg-7032| 0.2247785109812169|
|Avg-5983| 0.01376515795764316|
|Avg-4111| 0.10336592425084284|
|Avg-5999| 0.03103362660596684|
|Avg-5691| 0.016577535325028935|
|Avg-4814| 0.013617798502985055|
|Avg-5621| 0.04965539722945451|


## Classification Report:
              precision    recall  f1-score   support

        4111       0.71      0.55      0.62        22
        4814       1.00      1.00      1.00         1
        5411       0.92      0.95      0.93        58
        5621       0.98      1.00      0.99        89
        5691       0.88      0.95      0.91        56
        5812       0.92      0.85      0.88        40
        5983       1.00      1.00      1.00         7
        7032       0.98      0.96      0.97        55
        7832       0.93      0.94      0.94        72

    accuracy                           0.93       400
    macro avg       0.92      0.91     0.92       400
    weighted avg    0.93      0.93     0.93       400



## Definitions

### Precision
 It is the proportion of true positive predictions out of all the positive predictions. It measures the accuracy of positive predictions.
It Answers the following question: what proportion of predicted Positives is truly Positive?  

`Precision = TP/(TP+FP)`

### Recall: 
It is the proportion of true positive predictions out of all the actual positive cases. It measures the completeness of positive predictions. 
what proportion of actual Positives is correctly classified? recall is 

`Recall = TP/(TP+FN)`

### Accuracy: 
what proportion of photos — both Positive and Negative — were correctly classified?
proportion of correct predictions out of all the predictions made by the model on the test data accuracy is 

`Accuracy = (TP+TN)/(TP+TN+FP+FN)`

### F1-score: 

It is the harmonic mean of precision and recall. It balances both the metrics and gives an overall measure of the model's performance.

`F1-score = 2 × (precision × recall)/(precision + recall)`



### Support: 

It is the number of actual occurrences of the class in the dataset.

The report includes these metrics for each class in the target variable. In this case, there are ten classes represented by the numbers 4111, 4814, 5411, 5621, 5691, 5812, 5983, 7032, 5999, 7832.

- The macro average and weighted average of precision, recall, and F1-score are also provided. 
- Macro average is the unweighted average of each metric across all the classes.
- Weighted average considers the number of samples in each class.

