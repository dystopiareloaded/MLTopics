# 🎓 University-Style Machine Learning Question Bank (Topic-Wise)

---

## 📅 Week 23: Introduction to ML & Linear Regression

### 🧠 Theory
- Define Machine Learning. Differentiate between Supervised, Unsupervised, Semi-supervised, and Reinforcement Learning.
- Discuss the challenges in designing ML systems with examples.
- Compare Batch vs. Online Learning.
- Describe the ML life cycle with suitable stages.

### 🧮 Derivation
- Derive the Normal Equation for Simple Linear Regression.
- Prove that the least squares solution minimizes the sum of squared errors.

### 🔢 Numerical
- Given a dataset, compute slope and intercept of the best-fit line manually.
- Calculate MAE, MSE, RMSE, R² from actual vs. predicted values.

---

## 📅 Week 24: Gradient Descent

### 🧠 Theory
- Compare Batch, Stochastic, and Mini-Batch Gradient Descent.
- Explain how the learning rate affects convergence.

### 🧮 Derivation
- Derive the parameter update rule in Gradient Descent.
- Show how gradient descent behaves on convex and non-convex functions.

### 🔢 Numerical
- Perform 3 iterations of GD with θ=0, α=0.01, on sample data.
- Compare convergence using two learning rates (0.01 and 0.1).

---

## 📅 Week 25: Regression Analysis

### 🧠 Theory
- Distinguish between prediction and inference in regression.
- List and explain assumptions of linear regression.

### 🧮 Derivation
- Derive the F-statistic using ESS, RSS, TSS.
- Prove that \( R^2 = 1 - \frac{RSS}{TSS} \)

### 🔢 Numerical
- Compute R², F-statistic, Adjusted R² from TSS and RSS.
- Interpret confidence intervals for regression coefficients.

---

## 📅 Week 26: Feature Selection

### 🧠 Theory
- Explain Filter, Wrapper, and Embedded methods.
- Discuss limitations of correlation-based feature selection.

### 🔢 Numerical
- Calculate ANOVA F-values for features.
- Compute VIF to detect multicollinearity.

---

## 📅 Week 27: Regularization

### 🧠 Theory
- Explain the bias-variance tradeoff with a diagram.
- Compare Ridge, Lasso, and ElasticNet.

### 🧮 Derivation
- Derive Ridge Regression cost function and update rule.
- Show how L1 leads to sparsity in Lasso.

### 🔢 Numerical
- Compute Ridge regression coefficients for given λ.
- Plot Lasso coefficient paths across α values.

---

## 📅 Week 28: K-Nearest Neighbors

### 🧠 Theory
- Explain K’s impact on bias-variance in KNN.
- List and compare distance metrics.

### 🔢 Numerical
- Classify a new point using K=3 and a labeled dataset.
- Compute weighted KNN prediction for a test sample.

---

## 📅 Week 29: Principal Component Analysis (PCA)

### 🧠 Theory
- Define curse of dimensionality and PCA’s role.
- Compare PCA with SVD.

### 🧮 Derivation
- Derive PCA using variance maximization.
- Explain eigenvalues/eigenvectors role in PCA.

### 🔢 Numerical
- Perform PCA: standardize, compute covariance, eigen-decompose.
- Project data to the first principal component.

---

## 📅 Week 30: Model Evaluation

### 🧠 Theory
- Explain ROC-AUC and its use.
- Discuss Precision, Recall, F1-score trade-offs.

### 🔢 Numerical
- Calculate confusion matrix metrics from TP/FP/TN/FN.
- Manually compute k-fold CV score for a dataset.

---

## 📅 Week 31: Naive Bayes

### 🧠 Theory
- Explain Naive Bayes assumption.
- Differentiate Gaussian, Multinomial, and Bernoulli NB.

### 🧮 Derivation
- Derive Bayes Theorem and log-probabilities for classification.

### 🔢 Numerical
- Calculate spam probability using NB and Laplace smoothing.
- Build a basic classifier using word frequency data.

---

## 📅 Week 32: Logistic Regression

### 🧠 Theory
- Describe sigmoid function and logistic regression interpretation.
- Explain odds and log-odds in logistic regression.

### 🧮 Derivation
- Derive log-likelihood for logistic regression.
- Apply MLE to optimize coefficients.

### 🔢 Numerical
- Perform one iteration of gradient descent in logistic regression.
- Calculate log-loss for given predictions.

---

## 📅 Week 33: Support Vector Machines

### 🧠 Theory
- Differentiate hard vs. soft margin SVM.
- Describe the role of kernel functions.

### 🧮 Derivation
- Derive optimization for hard-margin SVM.
- Explain KKT conditions in SVM formulation.

### 🔢 Numerical
- Identify support vectors in a given 2D dataset.
- Compute kernel value (RBF or polynomial) between two points.

---

## 📅 Week 34: Decision Trees

### 🧠 Theory
- Explain CART splitting using Gini impurity.
- Define pruning and its importance.

### 🔢 Numerical
- Compute Gini Index and Information Gain for sample splits.
- Construct a small decision tree (depth = 2) manually.

---

## 📅 Week 35: Ensemble Methods

### 🧠 Theory
- Compare Bagging and Boosting intuitively.
- Differences between Random Forest and Gradient Boosting.

### 🔢 Numerical
- Manually simulate 3-tree Random Forest output.
- Perform 2 AdaBoost iterations and update weights.

---

## 📅 Week 36: Gradient Boosting & XGBoost

### 🧠 Theory
- Explain Gradient Boosting intuition and steps.
- How does XGBoost improve on traditional GBM?

### 🧮 Derivation
- Derive function update rule for log-loss classification.
- Explain Taylor expansion and regularization in XGBoost.

### 🔢 Numerical
- Manually perform 1 iteration of Gradient Boosting.
- Compute gain and similarity score in XGBoost split.

---

## 📦 Clustering Algorithms (KMeans, DBSCAN, Hierarchical)

### 🧠 Theory
- Compare Partitional, Hierarchical, Density-based clustering.
- Explain Elbow Method and Silhouette Score.

### 🧮 Derivation
- Derive Lloyd’s update rules for KMeans.

### 🔢 Numerical
- Do 2 iterations of KMeans on 5-point 2D data.
- Identify core/border points in DBSCAN for ε and MinPts.
- Construct dendrogram and find optimal clusters.

---


| Category                        | Included Topics                                                                   |
| ------------------------------- | --------------------------------------------------------------------------------- |
| 📘 **Core Theory**              | Every fundamental concept (definitions, comparisons, assumptions, life cycle)     |
| 🧮 **Mathematical Derivations** | Normal equation, gradient descent, regularization, SVM optimization, PCA, etc.    |
| 🔢 **Numerical Practice**       | Manual iterations, matrix computations, metric calculations, coding logic         |
| 📊 **Model Evaluation**         | ROC-AUC, cross-validation, metrics, bias-variance, hyperparameter tuning          |
| 🧠 **Intuition + Comparison**   | Deep insights into algorithm differences, applications, and visual interpretation |
| 🧪 **Applied Aspects**          | Hands-on manual simulation (e.g., AdaBoost, KMeans, XGBoost steps)                |


---

# 🎯 Top 30 Must-Study Machine Learning Questions  
### (University Exam-Oriented | Weeks 23–36)

---

## 📘 1. Core Theory Questions (10)

1. **Define Machine Learning.** Differentiate between:
   - Supervised
   - Unsupervised
   - Semi-supervised
   - Reinforcement Learning

2. **Explain the key assumptions of Linear Regression:**
   - Linearity
   - Homoscedasticity
   - Independence
   - Multicollinearity

3. **Compare Gradient Descent variants:**
   - Batch Gradient Descent (BGD)
   - Stochastic Gradient Descent (SGD)
   - Mini-Batch Gradient Descent

4. **What is the Bias-Variance Tradeoff?** Illustrate it graphically.  
   How does regularization help reduce overfitting?

5. **Describe the Machine Learning development life cycle.**  
   Include data collection, cleaning, training, validation, testing, and deployment.

6. **Compare Ridge, Lasso, and ElasticNet.**  
   When should each be used?

7. **What is PCA (Principal Component Analysis)?**  
   Discuss its intuition and purpose in dimensionality reduction.

8. **Explain model evaluation metrics:**
   - Precision
   - Recall
   - F1-score
   - ROC-AUC

9. **Differentiate between Hard-Margin and Soft-Margin SVM.**  
   Which one is suitable for linearly non-separable data?

10. **Compare Bagging vs. Boosting.**  
    Explain how each improves performance and reduces error.

---

## 🧮 2. Important Derivations (10)

11. **Derive the Normal Equation** used in linear regression:
   \[
   \theta = (X^TX)^{-1}X^Ty
   \]

12. **Prove that**:
   \[
   R^2 = 1 - \frac{RSS}{TSS}
   \]

13. **Derive the gradient descent update rule** for minimizing MSE:
   \[
   \theta_j := \theta_j - \alpha \cdot \frac{\partial J}{\partial \theta_j}
   \]

14. **Derive the F-statistic** used for overall model significance in regression.

15. **Derive the log-likelihood function** for Logistic Regression.  
    Show how it leads to the logistic loss function.

16. **Derive Ridge Regression’s cost function and modified normal equation.**

17. **Derive PCA via variance maximization.**  
    Explain the role of eigenvalues and eigenvectors.

18. **Formulate the SVM optimization problem.**  
    Explain constraints and use of Lagrange multipliers.

19. **Derive the function update rule in Gradient Boosting.**  
    Use residuals for regression or log-odds for classification.

20. **Explain Taylor expansion in XGBoost.**  
    How is it used to approximate and optimize the loss function?

---

## 🔢 3. Important Numerical Problems (10)

21. **Given a dataset**, manually compute:
   - Slope (m)
   - Intercept (b)
   for Simple Linear Regression.

22. **Perform 2–3 iterations of Gradient Descent** by hand:
   - $\theta = 0$
   - $\alpha = 0.01$
   - Use small dataset

23. **Calculate the following metrics**:
   - MAE
   - MSE
   - RMSE
   - $R^2$

24. **Use ANOVA** to compute F-value for feature importance.

25. **Given a confusion matrix**, calculate:
   - Accuracy
   - Precision
   - Recall
   - F1-score

26. **Classify a point using KNN (K=3)** from a small labeled dataset.

27. **Perform 2 iterations of KMeans** clustering manually on 2D points.

28. **Given a covariance matrix**, compute:
   - Eigenvalues
   - Eigenvectors
   - Project data using PCA

29. **Construct a decision tree of depth 2** using Gini Index or Information Gain.

30. **Calculate similarity score and gain in XGBoost** using:
   - Gradient
   - Hessian
   - Regularization term

---

## ✅ Bonus Tip

- 🧠 Focus on understanding **intuition behind derivations**.
- ✍️ Practice each numerical on **paper, calculator, or Python**.
- 🔁 Revise the theory questions until you can answer each in under 3 minutes.
- 📌 Create flashcards for formulas and metric interpretations.

---

> 📘 *“Don’t just memorize. Understand the 'why' behind the math.”*  
> — For success in university-level ML exams.
