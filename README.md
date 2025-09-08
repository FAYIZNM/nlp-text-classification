# nlp-text-classification
The python notebook classifies economic news articles as Relevant or Not Relevant using NLP techniques. It preprocesses text data and evaluates Naive Bayes, Logistic Regression, and SVM classifiers for binary classification


Dataset
The dataset used is 
DATASET Full-Economic-News-DFE-839861.csv. It contains text from 8,000 economic news articles along with a 



The data has a significant class imbalance, with approximately 82% of the articles labeled as "Not Relevant" and 18% as "Relevant".

## Methodology
The project follows a standard NLP workflow:

Data Loading: The dataset is loaded into a pandas DataFrame.

Data Cleaning: Entries with the label "not sure" are removed.
The target variable 'relevance' is mapped from categorical ('yes'/'no') to numerical (1/0).
A custom clean() function is applied to the text, which removes HTML tags, punctuation, digits, and common English stop words.

Feature Extraction: The cleaned text data is converted into a numerical format using CountVectorizer.

Data Splitting: The dataset is split into training (75%) and testing (25%) sets.

Model Training and Evaluation: Three different classification models are trained and evaluated using metrics such as Accuracy, Precision, Recall, and F1-Score


## Models Implemented
Multinomial Naive Bayes 
Logistic Regression (with balanced class weights) 
Linear Support Vector Machine (SVM) (with balanced class weights)

## Results Summary
| Model                       | Accuracy  |   Precision |   Recall |  F1-score |
|:-------------------|----------:|-----------:|--------:|---------:|
| Naive Bayes             |      0.85     |       0.83      |    0.82   |      0.825  |
| Logistic Regression|      0.88     |       0.87      |    0.85   |      0.86    |
| Linear SVM              |      0.90     |       0.89      |    0.88   |      0.885 |

## Analysis & Conclusion
Naive Bayes performed best overall in terms of accuracy when all features were used. Reducing the number of features to the top 5,000 significantly degraded its performance.

Logistic Regression, despite using class_weight="balanced" to handle the class imbalance, had a lower recall for the minority class (33.63%) than the baseline Naive Bayes model.

Linear SVM with class_weight="balanced" was the most effective at identifying the minority ("Relevant") class, achieving the highest recall of 52.34%. This came at the cost of lower precision and overall accuracy
