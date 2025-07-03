# 🎓 University Question Bank: Regression Analysis (Part 1 & 2)

## 📅 Syllabus Coverage
- Definition, Purpose, Statistical Links  
- Inference vs. Prediction  
- TSS, RSS, ESS, F-statistic, Degrees of Freedom  
- Goodness-of-fit, Confidence Intervals  
- Polynomial Regression: Formulation, Use Cases  
- Assumptions: Linearity, Normality, Homoscedasticity, Autocorrelation, Multicollinearity  
- Detection (VIF, Condition Number), Remedies, Standard Error  
- Python-based implementation & diagnostics  

---

## 📘 Section A: Conceptual / Short Answer (3–5 Marks Each)

1. Define linear regression. What are its primary objectives in statistical modeling?
2. Differentiate between inference and prediction. Provide one practical use-case for each.
3. What is the relationship between TSS, RSS, and ESS? Explain each term.
4. What does the F-statistic measure in the context of regression?
5. Define "goodness-of-fit." How is it interpreted using R²?
6. List the key assumptions of linear regression and briefly explain each.
7. What is multicollinearity? Why is it considered problematic?
8. Explain the purpose of calculating confidence intervals for regression coefficients.
9. What is standard error in regression? How is it different from standard deviation?
10. State the consequences of violating the assumption of homoscedasticity.

---

## 📘 Section B: Long Answer / Derivation-Based (5–8 Marks Each)

11. Derive the least squares estimators for simple linear regression from first principles.
12. Show that \( R^2 = \text{cor}(x, y)^2 \) in simple linear regression.
13. Derive the confidence interval for a regression coefficient \( \beta_j \).
14. Explain and derive the formula for the F-statistic used in multiple regression analysis.
15. Prove that multicollinearity inflates the variance of the estimated regression coefficients using matrix notation.
16. Derive the formula for the Variance Inflation Factor (VIF).
17. What is the condition number of a matrix? How can it be used to detect multicollinearity?
18. Explain the Breusch-Pagan test and how it helps in detecting heteroscedasticity.

---

## 📘 Section C: Numerical / Applied Questions (6–8 Marks Each)

19. Given a dataset of 6 data points, calculate the regression line manually and compute R².
20. From a regression output:
   - Coefficients: Intercept = 1.5 (SE = 0.2), X = 3.2 (SE = 0.4)
   - t-critical = 2.045  
   Construct the 95% confidence interval for \( \beta_X \) and test its significance.
21. A regression model has RSS = 80, TSS = 160. Calculate R² and interpret the result.
22. In a model with 2 predictors and 50 observations:
   - RSS = 120, TSS = 300  
   Calculate the F-statistic and assess the model's overall significance.
23. A dataset has X1, X2 highly correlated with VIFs of 12 and 15. Discuss the implications and suggest remedial steps.
24. Fit a polynomial regression model (degree 2) on:
   ```
   x = [1, 2, 3, 4], y = [3, 6, 11, 18]
   ```
   Solve for coefficients and predict y when x = 5.

---

## 💻 Section D: Python-Based Practical (5–8 Marks Each)

25. Load the following dataset:

   ```python
   data = pd.DataFrame({
       'TV': [230, 44, 17, 151, 180, 8, 57, 120],
       'Radio': [37, 39, 45, 41, 10, 32, 29, 17],
       'Sales': [22.1, 10.4, 9.3, 18.5, 12.9, 7.2, 11.8, 13.2]
   })
   ```

   a) Fit a linear regression model using `TV` and `Radio` as predictors.  
   b) Print coefficients, R² score, and interpret them.

26. Given `x = [1,2,3,4,5]` and `y = [2,5,10,17,26]`, fit a **polynomial regression** of degree 2 using `sklearn`. Plot and interpret.

27. Using a synthetic dataset with multicollinearity:
   ```python
   df = pd.DataFrame({
       'X1': [1,2,3,4,5,6,7,8],
       'X2': [2,4,6,8,10,12,14,16],
       'Y': [3,6,9,12,15,18,21,24]
   })
   ```
   a) Fit a regression model using X1 and X2.  
   b) Calculate VIFs and interpret.

28. Fit a model on the `TV` and `Radio` dataset.  
   a) Plot residuals vs fitted values.  
   b) Check for heteroscedasticity using the **Breusch-Pagan test**.

29. Use `statsmodels` to refit the Advertising model.  
   a) Print the OLS summary.  
   b) Extract and interpret the 95% CI for the TV coefficient.

30. Using time-series data (or simulated), detect autocorrelation using:
   - Residual plots  
   - Durbin-Watson statistic  
   Suggest remedies if autocorrelation is present.

---

## 🧠 Bonus Practice Questions (Open-Ended)

31. Can a model have a high R² and still be useless for prediction? Explain with an example.
32. In polynomial regression, why does test error increase after a certain degree?
33. Describe a real-world example where multicollinearity could arise and how you'd detect it.
34. Suppose you're given a dataset with 50 predictors. How would you approach feature selection while maintaining model interpretability?

---

### ✅ Tip for Study
Break this question bank across 7 days:
- Day 1–2: Concepts + Short Theories
- Day 3–4: Derivations + Numericals
- Day 5–6: Python Practicals
- Day 7: Self-test using sample paper or mock viva

---

📦 Let me know if you'd like:
- PDF / LaTeX version of this question bank  
- A sample **exam paper generated from this bank**  
- Jupyter notebook template for practical questions  
