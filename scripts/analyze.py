import argparse
import csv
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", type=str, default=None, help="Kernel name of kernel_agent results to analyze")
    args = parser.parse_args()

    if args.kernel:
        return args.kernel

    return None

def get_csv_files(kernel):
    seed_dirs = list(Path(f"results/{kernel}/").glob("seed_*/"))
    csv_files = []
    for seed_dir in seed_dirs: 
        csv_files.extend(seed_dir.glob("*.csv"))

    return csv_files




if __name__ == "__main__":
    kernel = parse_args()
    csv_files = get_csv_files(kernel)

    dfs = []
    for file in csv_files:
        seed = int(file.parent.name.split("_")[1])
        df = pd.read_csv(file)
        df['seed'] = seed
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    print(df)

    # first plot showing the gap over timesteps
    timestep_results = df.groupby('timestep')['gap'].agg(['mean', 'std'])
    plt.plot(timestep_results.index, timestep_results['mean'])
    plt.fill_between(timestep_results.index, (timestep_results['mean'] - timestep_results['std']).clip(lower=0), timestep_results['mean'] + timestep_results['std'], alpha=0.3)
    plt.xlabel('timestep')
    plt.ylabel('gap (%)')
    plt.title('Gap between optimal and median execution time per timestep')
    plt.show()

    # second plot
    last_timestep = df[df['timestep'] == df['timestep'].max()]

    results = last_timestep.groupby('gpu')['gap'].agg(['mean', 'std'])

    plt.bar(results.index, results['mean'], yerr=results['std'])
    plt.xlabel('gpu')
    plt.ylabel('gap (%)')
    plt.title('Final gap per gpu file')
    plt.show()


    # third plot
    holdout_results = df[df['is_holdout'] == True]
    train_results = df[df['is_holdout'] == False]

    holdout_results = holdout_results.groupby('timestep')['gap'].agg(['mean', 'std'])
    train_results = train_results.groupby('timestep')['gap'].agg(['mean', 'std'])

    plt.plot(holdout_results.index, holdout_results['mean'], label='holdouts')
    plt.fill_between(holdout_results.index, (holdout_results['mean'] - holdout_results['std']).clip(lower=0), holdout_results['mean'] + holdout_results['std'], alpha=0.3)

    plt.plot(train_results.index, train_results['mean'], label='train files')
    plt.fill_between(train_results.index, (train_results['mean'] - train_results['std']).clip(lower=0), train_results['mean'] + train_results['std'], alpha=0.3)
    plt.xlabel('timestep')
    plt.ylabel('gap (%)')
    plt.title('Gap between optimal and median execution time per timestep for both holdout and train gpus')
    plt.legend()
    plt.show()
