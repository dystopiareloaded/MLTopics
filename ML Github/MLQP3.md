### **Machine Learning Final Examination**  
**Time:** 3 Hours  
**Total Marks:** 100  
**Instructions:**  
- Answer all questions.  
- Assume suitable data if missing.  
- Justify all answers unless specified.  

---

#### **Section A: Theory & Short Answers** (5 × 6 = 30 Marks)  
1. **Overfitting vs. Underfitting**:  
   Define both terms. Provide two techniques to combat overfitting in linear regression and one remedy for underfitting.  

2. **Gradient Descent Variants**:  
   Contrast Batch GD, Stochastic GD, and Mini-Batch GD. Explain how *learning rate schedules* improve Stochastic GD convergence.  

3. **PCA & Eigen Decomposition**:  
   Why is mean-centering critical in PCA? Derive the relationship between eigenvalues and explained variance.  

4. **Bias-Variance Tradeoff**:  
   Mathematically decompose the expected prediction error into bias, variance, and irreducible error.  

5. **SVM Kernels**:  
   Explain how the RBF kernel transforms non-linear data. Provide a use case where Polynomial kernels outperform RBF.  

6. **Ensemble Methods**:  
   Why does Random Forest reduce overfitting compared to a single Decision Tree? Define *OOB error*.  

---

#### **Section B: Derivations & Proofs** (10 × 3 = 30 Marks)  
1. **Ridge Regression Derivation**:  
   Given the loss function \( J(\mathbf{w}) = \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2 + \lambda \|\mathbf{w}\|^2 \),  
   derive the closed-form solution \( \mathbf{w}^* = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y} \).  

2. **Naive Bayes Classifier**:  
   Starting from Bayes’ theorem, derive the log-probability estimation for class \( C_k \) given features \( \mathbf{x} \).  
   Explain the role of Laplace smoothing.  

3. **Logistic Regression MLE**:  
   Prove that maximizing the log-likelihood for binary logistic regression is equivalent to minimizing cross-entropy loss.  
   Show the gradient update rule.  

---

#### **Section C: Numerical Problems** (20 × 2 = 40 Marks)  
1. **Gradient Descent & Regularization**:  
   Dataset:  
