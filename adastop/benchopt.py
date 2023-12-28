from benchopt import run_benchmark
from benchopt.benchmark import Benchmark
import pandas as pd


def run_benchopt(solver, dataset, n_repetitions, output_name, forced_solvers, timeout=100, max_runs=10): 
    # load benchmark
    BENCHMARK_PATH = "./"
    benchmark = Benchmark(BENCHMARK_PATH)

    # run benchmark
    run_benchmark(
        benchmark,
        solver_names=solver,
        dataset_names=dataset,
        n_repetitions=n_repetitions,
        timeout=timeout,
        max_runs=max_runs,
        output_name=output_name,
        forced_solvers = forced_solvers
    )

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
            df_ret[solver].append(df_rep_solver['objective_value'].iloc[-1])
    return pd.DataFrame(df_ret)
