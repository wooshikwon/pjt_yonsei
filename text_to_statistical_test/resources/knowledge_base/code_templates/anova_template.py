# Code Template: One-Way ANOVA
#
# This template demonstrates how to perform a One-Way Analysis of Variance (ANOVA)
# using the `statsmodels` library. The agent can use this as a reference
# for understanding the formula-based approach and the subsequent post-hoc tests.

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def anova_example(dataframe: pd.DataFrame, group_column: str, value_column: str):
    """
    Performs a One-Way ANOVA and a Tukey's HSD post-hoc test.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        group_column (str): The name of the column that defines the groups (3 or more).
        value_column (str): The name of the column with the numerical values to compare.
    """
    print(f"--- ANOVA Example: Comparing '{value_column}' across '{group_column}' groups ---")

    # 1. Fit the ANOVA model using a formula
    # The formula `value_column ~ C(group_column)` tells statsmodels to model
    # the `value_column` as a function of the `group_column`, treating the latter
    # as a categorical variable `C()`.
    model = ols(f'Q("{value_column}") ~ C(Q("{group_column}"))', data=dataframe).fit()
    
    # 2. Perform the ANOVA and get the anova table
    anova_table = sm.stats.anova_lm(model, typ=2)

    print("\nANOVA Results:")
    print(anova_table)

    # 3. Interpret the ANOVA results
    p_value = anova_table['PR(>F)'][0]
    alpha = 0.05

    if p_value < alpha:
        print(f"\nConclusion: The p-value ({p_value:.4f}) is less than {alpha}.")
        print("There is a statistically significant difference in the means of '{value_column}' across the '{group_column}' groups.")
        print("Proceeding to Tukey's HSD post-hoc test to find out which groups differ.")

        # 4. Perform Tukey's HSD post-hoc test if ANOVA is significant
        tukey_results = pairwise_tukeyhsd(
            endog=dataframe[value_column],
            groups=dataframe[group_column],
            alpha=alpha
        )
        
        print("\nTukey's HSD Results:")
        print(tukey_results)
        
        print("\nInterpretation of Tukey's HSD:")
        print("The `reject=True` column indicates pairs of groups with statistically significant differences.")

    else:
        print(f"\nConclusion: The p-value ({p_value:.4f}) is greater than or equal to {alpha}.")
        print(f"There is not enough evidence to claim a significant difference in the means of '{value_column}' across the groups.")
        print("Post-hoc test is not necessary.")

# Example Usage:
if __name__ == '__main__':
    # Create a sample DataFrame
    data = {
        'Fertilizer': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
        'Yield': [25, 28, 26, 35, 32, 33, 22, 24, 23]
    }
    df = pd.DataFrame(data)

    # Run the example function
    # Note: Using Q() to quote column names with special characters or spaces
    anova_example(dataframe=df, group_column='Fertilizer', value_column='Yield') 