# Code Template: Linear Regression
#
# This template demonstrates how to perform a linear regression analysis
# using the `statsmodels` library. The agent can use this as a reference
# for understanding how to build a model and interpret its summary output.

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

def linear_regression_example(dataframe: pd.DataFrame, dependent_var: str, independent_vars: list[str]):
    """
    Performs a linear regression and prints the model summary.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        dependent_var (str): The name of the dependent (outcome) variable.
        independent_vars (list[str]): A list of names of the independent (predictor) variables.
    """
    print(f"--- Linear Regression Example: Predicting '{dependent_var}' ---")

    # 1. Construct the model formula from the list of independent variables
    # The `+` operator is used to add predictors to the model.
    # We use Q("") to handle column names that might have spaces or special characters.
    formula = f'Q("{dependent_var}") ~ ' + ' + '.join([f'Q("{var}")' for var in independent_vars])
    
    print(f"\nModel Formula: {formula}")

    # 2. Fit the Ordinary Least Squares (OLS) model
    model = ols(formula, data=dataframe).fit()

    # 3. Print the full model summary
    # The summary contains a wealth of information including R-squared, coefficients,
    # p-values, and assumption test statistics (like Durbin-Watson).
    print("\nModel Summary:")
    print(model.summary())

    # 4. Interpret the key results from the summary
    print("\nKey Interpretations:")
    
    # R-squared
    print(f"- Adjusted R-squared: {model.rsquared_adj:.3f}")
    print(f"  This means that approximately {model.rsquared_adj:.1%} of the variance in '{dependent_var}' can be explained by the predictor variables in the model.")

    # Global F-test
    f_pvalue = model.f_pvalue
    alpha = 0.05
    if f_pvalue < alpha:
        print(f"- Global F-test p-value: {f_pvalue:.4f}. The model as a whole is statistically significant.")
    else:
        print(f"- Global F-test p-value: {f_pvalue:.4f}. The model as a whole is not statistically significant.")

    # Coefficients
    print("- Coefficients:")
    for i, var in enumerate(independent_vars):
        coef = model.params[f'Q("{var}")']
        p_val = model.pvalues[f'Q("{var}")']
        
        significance = "is statistically significant" if p_val < alpha else "is not statistically significant"
        
        print(f"  - {var}: Coef = {coef:.3f}, P-value = {p_val:.3f}. This variable {significance}.")

# Example Usage:
if __name__ == '__main__':
    # Create a sample DataFrame
    data = {
        'Salary': [70000, 50000, 120000, 95000, 85000, 65000, 110000],
        'YearsExperience': [5, 2, 10, 7, 6, 4, 9],
        'EducationLevel': [16, 14, 18, 16, 16, 12, 18] # e.g., years of schooling
    }
    df = pd.DataFrame(data)

    # Run the example function
    linear_regression_example(
        dataframe=df,
        dependent_var='Salary',
        independent_vars=['YearsExperience', 'EducationLevel']
    ) 