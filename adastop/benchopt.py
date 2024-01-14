import pandas as pd


def process_benchopt(file):
    """
    For now, suppose that there is only one dataset
    """
    df = pd.read_parquet(file)
    df= df[["solver_name",'objective_value','idx_rep']]
    df_ret = { name : [] for name in df["solver_name"].unique()}
    for rep in df["idx_rep"].unique():
        for solver in df["solver_name"].unique():
            df_rep_solver = df.loc[ (df["solver_name"]==solver) & (df["idx_rep"]==rep)]
            df_ret[solver].append(df_rep_solver['objective_test_loss'].iloc[-1])
    return pd.DataFrame(df_ret)
