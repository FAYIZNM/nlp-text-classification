# Text Classification with Naive Bayes, Logistic Regression, and Linear SVM

This project evaluates three machine learning algorithms — **Naive Bayes**, **Logistic Regression**, and **Linear SVM** — on a text classification task using the **Full-Economic-News-DFE-839861** dataset.  
The target is the `relevance` label (`yes`, `no`, `not sure`).

We experiment with different vocabulary sizes by setting `max_features` in `CountVectorizer` to **40,000**, **5,000**, and **1,000**.  
Each algorithm was run in a separate Colab notebook for clarity.

---

## Dataset
- **Source:** Full-Economic-News-DFE-839861.csv  
- **Samples:** ~8000  
- **Target column:** `relevance`  
- **Text column:** `text`  

---

## Algorithms Tested
1. **Naive Bayes (MultinomialNB)**  
2. **Logistic Regression**  
3. **SVM**  
   
---

## Results Comparison

| Algorithm           | Max Features | Accuracy | Precision | Recall | F1 Score |
|---------------------|--------------|----------|-----------|--------|----------|
| **Naive Bayes**     | 40,000       | 0.7506   | 0.7871    | 0.7506 | 0.7651   |
|                     | 5,000        | 0.6750   | 0.7951    | 0.6750 | 0.7095   |
|                     | 1,000        | 0.6900   | 0.7918    | 0.6900 | 0.7213   |
| **Logistic Regression** | 40,000   | 0.7700   | 0.7535    | 0.7700 | 0.7609   |
|                     | 5,000        | 0.7544   | 0.7470    | 0.7544 | 0.7506   |
|                     | **1,000**    | **0.7850** | 0.7584 | **0.7850** | **0.7689** |
| **Linear SVM**      | 40,000       | 0.7444   | 0.7435    | 0.7444 | 0.7439   |
|                     | 5,000        | 0.7388   | 0.7449    | 0.7388 | 0.7417   |
|                     | **1,000**    | **0.7863** | 0.7537 | **0.7863** | 0.7658   |

---

## Key Insights
- **Naive Bayes**: Overestimates precision but underdelivers on recall and accuracy. Works as a quick baseline but not competitive here.  
- **Logistic Regression**: Best **F1 score** and nearly tied for accuracy at 1000 features. Most balanced performer overall.  
- **Linear SVM**: Highest **accuracy** (0.7863 with 1000 features), but slightly lower F1 compared to Logistic Regression.  
- **Winner**: **Logistic Regression with 1000 features** (best balance across all metrics).  

---

