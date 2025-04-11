import numpy as np
import pandas as pd


def load_dataframe(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    
    Args:
        file_path (str): The path to the CSV file.
    
    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    return pd.read_csv(file_path, sep=';')


def merge_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Merge two DataFrames on the 'student_id' column.
    
    Args:
        df1 (pd.DataFrame): The first DataFrame.
        df2 (pd.DataFrame): The second DataFrame.
    
    Returns:
        pd.DataFrame: The merged DataFrame.
    """
    on = ["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"]
    return pd.merge(df1, df2, on=on, suffixes=('_mat', '_por'))


def save_dataframe(df: pd.DataFrame, file_path: str) -> None:
    """
    Save a DataFrame to a CSV file.
    
    Args:
        df (pd.DataFrame): The DataFrame to save.
        file_path (str): The path to the output CSV file.
    
    Returns:
        None
    """
    df.to_csv(file_path, sep=';', index=False)
    
    
def main():
    # Load the data
    df1 = load_dataframe('./data/student-mat.csv')
    df2 = load_dataframe('./data/student-por.csv')

    # Merge the dataframes
    merged_df = merge_dataframes(df1, df2)

    # Save the merged dataframe
    save_dataframe(merged_df, './data/merged.csv')
    
    
if __name__ == "__main__":
    main()