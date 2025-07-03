# 🎓 University Examination: Machine Learning
## 📅 Module: Regression Analysis (Part 1 & 2)
**Duration:** 3 Hours  
**Maximum Marks:** 100  
**Instructions:**
- Attempt all questions.
- Assume reasonable values where necessary and state them clearly.
- Show all calculations and derivations for full credit.

---

### 📘 Part A: Conceptual and Theoretical Questions (30 Marks)

1. **[3 marks]** Define and differentiate between *inference* and *prediction* in the context of regression analysis. Provide an example of each.

2. **[4 marks]** Explain the relationship between TSS, RSS, and ESS. Derive the formula linking them and describe what each term represents in regression analysis.

3. **[4 marks]** What is the purpose of the F-statistic in multiple linear regression? State its mathematical formula and explain how it is used for hypothesis testing.

4. **[4 marks]** Define **goodness-of-fit**. How is it quantitatively measured in linear regression? Discuss the implications of a very high R².

5. **[5 marks]** List and explain the key assumptions of linear regression. For each assumption, mention:
   - How it can be tested
   - The possible consequences of violation
   - Any remedy (transformation, technique, or diagnostic)

6. **[5 marks]** Explain **multicollinearity** in multiple linear regression. Derive how the **Variance Inflation Factor (VIF)** is calculated. What does a high VIF indicate?

7. **[5 marks]** Distinguish between **standard error** and **standard deviation** in the context of regression coefficients. How does standard error impact confidence intervals?

---

### 📘 Part B: Mathematical Derivations and Proofs (30 Marks)

8. **[5 marks]** Given a linear regression model:  
   $$ y = \beta_0 + \beta_1x + \varepsilon, \quad \varepsilon \sim N(0, \sigma^2) $$  
   Derive the formula for the **least squares estimates** of \( \beta_0 \) and \( \beta_1 \) using the method of minimizing the residual sum of squares (RSS).

9. **[5 marks]** Prove that in simple linear regression:
   $$ R^2 = \text{cor}(X, Y)^2 $$
   where \( R^2 \) is the coefficient of determination.

10. **[5 marks]** Derive the formula for the **confidence interval** of a regression coefficient \( \beta_j \). Clearly state the assumptions required for the derivation.

11. **[5 marks]** Explain why multicollinearity inflates the variance of the OLS estimates. Use matrix notation and the variance-covariance matrix of \( \hat{\beta} \).

12. **[5 marks]** Show how the F-statistic for overall model significance in multiple regression is derived from TSS and RSS.

13. **[5 marks]** Derive the condition number of a matrix and explain how it can be used to detect multicollinearity in linear regression.

---

### 📘 Part C: Applied and Numerical Problems (40 Marks)

14. **[8 marks]** You are given the following dataset:
   | x | y |
   |---|---|
   | 1 | 2 |
   | 2 | 4 |
   | 3 | 6 |
   | 4 | 8 |
   | 5 | 11 |

   a) Fit a simple linear regression model.  
   b) Calculate the coefficients \( \hat{\beta}_0 \) and \( \hat{\beta}_1 \).  
   c) Calculate TSS, RSS, and R².  
   d) Comment on the model fit.

15. **[6 marks]** A regression model gives the following summary:
   ```
   Coefficients:
   Intercept: 1.5 (SE = 0.2)
   X1: 3.2 (SE = 0.4)
   ```
   a) Construct a 95% confidence interval for \( \beta_1 \).  
   b) Perform a t-test to check if \( \beta_1 \) is significantly different from zero. Use \( t_{0.025, df=28} \approx 2.048 \)

16. **[6 marks]** A regression model has RSS = 150, ESS = 450.  
   a) Calculate TSS and R².  
   b) If the model has 3 predictors and 100 observations, compute the F-statistic.  
   c) Interpret the result in terms of model usefulness.

17. **[8 marks]** A dataset with two highly correlated predictors (X1 and X2) gives the following VIF values:
   - VIF(X1) = 12.5
   - VIF(X2) = 10.8

   a) What does this indicate about multicollinearity?  
   b) Suggest two techniques to reduce multicollinearity.  
   c) Simulate (or assume) a new model after removing X2 and check the change in R².  
   d) Discuss the trade-off between removing multicollinearity and retaining explanatory power.

18. **[6 marks]** Fit a **polynomial regression** model of degree 2 on the dataset:  
   | x | y |
   |---|---|
   | 1 | 2 |
   | 2 | 5 |
   | 3 | 10 |
   | 4 | 17 |

   a) Write the polynomial model.  
   b) Use matrix formulation to solve for coefficients.  
   c) Predict the value of \( y \) when \( x = 5 \).

19. **[6 marks]** Explain the impact of **autocorrelation** in regression models:
   a) How can it be detected?  
   b) Simulate a scenario (time-series based) where autocorrelation leads to misleading regression results.  
   c) Propose at least two remedies.

---

## 📘 Bonus Question (Optional – 5 Marks)

20. **[5 marks]** Suppose you build a model with high R² and low RMSE on training data, but the test RMSE increases drastically.  
   a) What is this phenomenon called?  
   b) How can it be addressed in the context of **polynomial regression**?

---

### 🔍 End of Question Paper



## 💻 Practical Section: Python-based Regression Analysis (30 Marks)

> 🔧 You may use `NumPy`, `Pandas`, `scikit-learn`, `statsmodels`, and `matplotlib`/`seaborn` for these tasks. Attach code + output + interpretation wherever applicable.

---

### Q1. [6 Marks] Basic Linear Regression

You are given the following dataset:

```python
import pandas as pd

data = pd.DataFrame({
    'TV': [230, 44, 17, 151, 180, 8, 57, 120],
    'Radio': [37, 39, 45, 41, 10, 32, 29, 17],
    'Sales': [22.1, 10.4, 9.3, 18.5, 12.9, 7.2, 11.8, 13.2]
})
```

**Tasks:**
a) Fit a multiple linear regression model to predict `Sales` using `TV` and `Radio`.  
b) Print the regression coefficients and R² score.  
c) Interpret the coefficients in context.  

---

### Q2. [6 Marks] Polynomial Regression & Visualization

You are given:

```python
import numpy as np
x = np.array([1, 2, 3, 4, 5, 6])
y = np.array([2, 5, 10, 17, 26, 37])
```

**Tasks:**
a) Fit a **polynomial regression of degree 2**.  
b) Plot the original points and the predicted curve.  
c) Print the coefficients and predict \( y \) for \( x = 7 \).

---

### Q3. [6 Marks] Multicollinearity Detection and VIF Calculation

Use the following dataset:

```python
import pandas as pd

df = pd.DataFrame({
    'X1': [1, 2, 3, 4, 5, 6, 7, 8],
    'X2': [2, 4, 6, 8, 10, 12, 14, 16],
    'X3': [1, 3, 5, 7, 9, 11, 13, 15],
    'Y': [5, 11, 17, 23, 29, 35, 41, 47]
})
```

**Tasks:**
a) Fit a regression model predicting `Y` using `X1`, `X2`, and `X3`.  
b) Compute the **VIF** for each predictor.  
c) Identify and explain the presence of multicollinearity.

---

### Q4. [6 Marks] Model Diagnostics – Residuals and Homoscedasticity

Use the `TV`, `Radio`, `Sales` dataset from Q1.

**Tasks:**
a) Fit a linear regression model using `TV` and `Radio`.  
b) Plot the residuals vs fitted values.  
c) Use **Breusch-Pagan test** to check for **heteroscedasticity**. Interpret the result.

---

### Q5. [6 Marks] Confidence Intervals & Inference

Use `statsmodels` to refit the model from Q1.

**Tasks:**
a) Display the full regression summary using `ols`.  
b) Extract and interpret the 95% confidence interval for the TV coefficient.  
c) Conduct a t-test to check if the `Radio` coefficient is significantly different from 0.

---

📝 **Note**: Each question is marked on correctness (2), code structure (2), and interpretation (2).

---

### 🔚 End of Practical Section

