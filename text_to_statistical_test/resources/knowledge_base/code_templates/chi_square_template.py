# Code Template: Chi-Square Test of Independence
#
# This template demonstrates how to perform a Chi-Square test of independence
# using `scipy.stats` and `pandas`. The agent can use this as a reference
# for understanding how a contingency table is created and used in the test.

import pandas as pd
from scipy.stats import chi2_contingency

def chi_square_example(dataframe: pd.DataFrame, variable1: str, variable2: str):
    """
    Performs a Chi-Square test of independence and prints the results.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        variable1 (str): The name of the first categorical variable.
        variable2 (str): The name of the second categorical variable.
    """
    print(f"--- Chi-Square Test Example: Analyzing association between '{variable1}' and '{variable2}' ---")

    # 1. Create a contingency table (crosstab)
    # This table shows the frequency of each combination of categories.
    contingency_table = pd.crosstab(dataframe[variable1], dataframe[variable2])
    
    print("\nContingency Table (Observed Frequencies):")
    print(contingency_table)

    # 2. Perform the Chi-Square test
    # The `chi2_contingency` function returns the chi2 statistic, p-value,
    # degrees of freedom (dof), and the expected frequencies table.
    chi2_stat, p_value, dof, expected_freq = chi2_contingency(contingency_table)

    print("\nResults:")
    print(f"Chi-Square Statistic: {chi2_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Degrees of Freedom: {dof}")
    
    print("\nExpected Frequencies Table:")
    print(pd.DataFrame(expected_freq, index=contingency_table.index, columns=contingency_table.columns).round(2))

    # 3. Check the assumption of expected frequencies
    if (expected_freq < 5).any().any():
        print("\nWarning: Some cells have an expected frequency of less than 5.")
        print("The Chi-Square test results may not be accurate. Consider using Fisher's Exact Test instead.")

    # 4. Interpret the results
    alpha = 0.05
    if p_value < alpha:
        print(f"\nConclusion: The p-value is less than {alpha}, so we reject the null hypothesis.")
        print(f"There is a statistically significant association between '{variable1}' and '{variable2}'.")
    else:
        print(f"\nConclusion: The p-value is greater than or equal to {alpha}, so we fail to reject the null hypothesis.")
        print(f"There is no statistically significant association between '{variable1}' and '{variable2}'.")

# Example Usage:
if __name__ == '__main__':
    # Create a sample DataFrame
    data = {
        'Region': ['North', 'North', 'South', 'South', 'East', 'East', 'West', 'West'],
        'Smoker': ['Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'No']
    }
    df = pd.DataFrame(data)
    df = pd.concat([df]*5, ignore_index=True) # Increase sample size to meet assumptions

    # Run the example function
    chi_square_example(dataframe=df, variable1='Region', variable2='Smoker') 