import numpy as np
import pandas as pd

from pyDOE2 import fullfact, gsd


def create_fact_plan(df: pd.DataFrame, method_choice: str = "fullfact", reduction_ratio: int = 2) -> pd.DataFrame:

    r""" This function accepts a DataFrame summarizing the possible levels of different features \
    and returns a DataFrame corresponding to a Design of Experiment.

    The DoE can be rendered under its full form (fullfact method) or with a reduction ratio (gsd) \
    inherited from Generalized Subset Designs

    Parameters:
    ----------
    df: pd.DataFrame
        DataFrame summarizing the possible levels of each feature:

        |   Feat1   |   Feat2   |
        +-----------+-----------+
        | min level | min level |
        | max level | max level |

        or

        |   Feat1   |   Feat2   |   Feat3   |    ...    |
        +-----------+-----------+-----------+-----------+
        | min level | min level | min level |    ...    |
        | avg level | avg level | avg level |    ...    |
        | max level | max level | max level |    ...    |


    method_choice: str
        Name of the DoE creation method selected ("fullfact" or "gsd").

    reduction_ratio: int
        Reduction ratio used by the Generalized Subset Designs ("gsd") method.

    Returns:
    -------
    df : pd.DataFrame
        A DataFrame with the combination of levels by features.

        Output for a 2 x 2 input DataFrame with min/max levels:

        |   Feat1   |   Feat2   |
        +-----------+-----------+
        | min level | min level |
        | min level | max level |
        | max level | min level |
        | max level | max level |

    Raises:
    ------
    - Exception if the DoE method differs from possible options ("fullfact" or "gsd")

    """

    method_options = ["fullfact", "gsd"]

    # Check if the method_choice is part of the possible options
    assert method_choice in method_options, "method_choice should be \"fullfact\" or \"gsd\"."

    assert isinstance(reduction_ratio, int), "reduction_ratio should be an integer."

    # Create a list levels number x features
    # Ex. for a 2 features x 2 levels DataFrame: [2, 2]
    # Ex. for a 2 features x 3 levels DataFrame: [3, 3]
    # Ex. for a 4 features x 3 levels DataFrame: [3, 3, 3, 4]
    fact_dim = [df.shape[0] for _ in range(df.shape[1])]

    # Create DoE (NumPy) according to the chosen method
    if method_choice == "fullfact":
        F = fullfact(fact_dim)
    else:
        F = gsd(fact_dim, reduction_ratio)

    # Create/extract indexes and columns names
    plan_idx = [i for i in range(F.shape[0])]
    plan_col = df.columns

    # Iniate the DoE plan as DataFrame
    plan = pd.DataFrame(index=plan_idx, columns=plan_col)

    # Populate values within the DoE plan DataFrame
    for row_idx, row in enumerate(F):

        for col_idx in range(df.shape[1]):

            if df.iloc[int(row[col_idx]), col_idx]!= np.nan:
                plan.iloc[row_idx, col_idx] = df.iloc[int(row[col_idx]), col_idx]

    # Reassign the original DataFrame types to the DoE plan
    for column in plan.columns:
        plan[column] = plan[column].astype(df.dtypes[column])

    return plan
