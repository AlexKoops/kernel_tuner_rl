import argparse
import os
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

    os.makedirs(f"plots/{kernel}/", exist_ok=True)

    # first plot showing the gap over timesteps
    timestep_results = df.groupby('timestep')['gap'].agg(['mean', 'std'])
    plt.plot(timestep_results.index, timestep_results['mean'])
    plt.fill_between(timestep_results.index, (timestep_results['mean'] - timestep_results['std']).clip(lower=0), timestep_results['mean'] + timestep_results['std'], alpha=0.3)
    plt.xlabel('timestep')
    plt.ylabel('gap (%)')
    plt.title('Gap between optimal and median execution time per timestep')
    plt.savefig(f"plots/{kernel}/gap_over_timesteps.png")
    plt.close()

    # second plot
    last_timestep = df[df['timestep'] == df['timestep'].max()]

    results = last_timestep.groupby('gpu')['gap'].agg(['mean', 'std'])

    plt.bar(results.index, results['mean'], yerr=results['std'])
    plt.xlabel('gpu')
    plt.ylabel('gap (%)')
    plt.title('Final gap per gpu file')
    plt.savefig(f"plots/{kernel}/final_gap_per_gpu.png")
    plt.close()


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
    plt.savefig(f"plots/{kernel}/training_and_holdout_gaps_over_timesteps.png")
    plt.close()


    # Plot 4: Training diagnostics over time
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    diag = df[df['is_holdout'] == False].groupby('timestep').first().reset_index()

    metrics = [
        ('entropy',       'Entropy',       axes[0, 0]),
        ('policy_loss',   'Policy Loss',   axes[0, 1]),
        ('value_loss',    'Value Loss',    axes[1, 0]),
        ('clip_fraction', 'Clip Fraction', axes[1, 1]),
    ]

    for col, label, ax in metrics:
        ax.plot(diag['timestep'], diag[col])
        ax.set_xlabel('Timestep')
        ax.set_ylabel(label)
        ax.set_title(f'{label} over Training')

    plt.tight_layout()
    plt.savefig(f"plots/{kernel}/training_diagnostics.png")
    plt.close()

# Plot 5: Mean episode return over time (across seeds)
    ep_return = df[df['is_holdout'] == False].groupby('timestep')['mean_ep_return'].agg(['mean', 'std'])
    plt.plot(ep_return.index, ep_return['mean'], label='mean episode return')
    plt.fill_between(ep_return.index,
        (ep_return['mean'] - ep_return['std']).clip(lower=0),
        ep_return['mean'] + ep_return['std'], alpha=0.3)
    plt.xlabel('Timestep')
    plt.ylabel('Return')
    plt.title('Mean Episode Return over Training')
    plt.savefig(f"plots/{kernel}/mean_episode_return.png")
    plt.close()
