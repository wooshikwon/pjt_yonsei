# Code Template: Independent Samples T-test
#
# This template demonstrates how to perform an independent samples t-test
# using the `scipy.stats` library. The agent can use this as a reference
# for understanding the logic and parameters of the `t_test` tool.

import pandas as pd
from scipy.stats import ttest_ind

def independent_t_test_example(dataframe: pd.DataFrame, group_column: str, value_column: str):
    """
    Performs an independent samples t-test and prints the results.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        group_column (str): The name of the column that defines the two groups.
        value_column (str): The name of the column with the numerical values to compare.
    """
    print(f"--- Independent T-Test Example: Comparing '{value_column}' between '{group_column}' groups ---")

    # 1. Separate the data into two groups based on the group column
    groups = dataframe[group_column].unique()
    if len(groups) != 2:
        print(f"Error: The group column '{group_column}' must have exactly two unique values.")
        return

    group1_data = dataframe[dataframe[group_column] == groups[0]][value_column].dropna()
    group2_data = dataframe[dataframe[group_column] == groups[1]][value_column].dropna()

    print(f"Group 1 ('{groups[0]}'): n={len(group1_data)}, Mean={group1_data.mean():.2f}")
    print(f"Group 2 ('{groups[1]}'): n={len(group2_data)}, Mean={group2_data.mean():.2f}")

    # 2. Perform the t-test
    # The `ttest_ind` function returns the t-statistic and the p-value.
    # `equal_var=True` assumes homogeneity of variance (checked by Levene's test).
    # If Levene's test fails, `equal_var=False` should be used to perform Welch's t-test.
    t_statistic, p_value = ttest_ind(group1_data, group2_data, equal_var=True)

    print(f"\nResults:")
    print(f"T-statistic: {t_statistic:.4f}")
    print(f"P-value: {p_value:.4f}")

    # 3. Interpret the results
    alpha = 0.05
    if p_value < alpha:
        print(f"\nConclusion: The p-value is less than {alpha}, so we reject the null hypothesis.")
        print(f"There is a statistically significant difference in '{value_column}' between the two groups.")
    else:
        print(f"\nConclusion: The p-value is greater than or equal to {alpha}, so we fail to reject the null hypothesis.")
        print(f"There is not enough evidence to claim a significant difference in '{value_column}' between the two groups.")

# Example Usage:
if __name__ == '__main__':
    # Create a sample DataFrame
    data = {
        'Treatment': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'],
        'Score': [85, 90, 88, 92, 75, 78, 80, 77]
    }
    df = pd.DataFrame(data)

    # Run the example function
    independent_t_test_example(dataframe=df, group_column='Treatment', value_column='Score') 