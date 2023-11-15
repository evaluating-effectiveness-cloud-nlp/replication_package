import pandas as pd

def divide_dataframe(df: pd.DataFrame, unique_columns_names=True):
    rows = int(len(df)/2)
    part1 = df.iloc[:rows,:]
    part2 = df.iloc[rows:,:]
    part1 = part1.reset_index(drop=True)
    part2 = part2.reset_index(drop=True)
    if unique_columns_names:
        part2.columns = [f'{c}__1' for c in part2.columns]

    df = pd.concat([part1, part2],axis=1)
    df = df.reset_index(drop=True)

    return df