import os
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from root import rootpath

DATA_PATH = f"{rootpath}/data/pcg_data/"
ALGO = "ppo"

def plot_mean_std(x, y, std=None, label=None, log_scale=False, rolling_avg=False):
    if log_scale:
        plt.yscale("log")
    plt.plot(x, y, label=label)
    if rolling_avg:
        y_rolling_avg = y.rolling(5).mean()
        plt.plot(x, y_rolling_avg, label="rolling avg")
    if std is not None:
        plt.fill_between(x, y-std, y+std, alpha=0.2)

def save_graph(xlabel, ylabel, path, name):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(path + name)
    plt.clf()

def create_exps_graph(exps, xs, ys, log_scale, xlabel, ylabel, graph_path, name, stds=None, rolling_avg=False):
    for i in range(len(exps)):
        y = ys[i].rolling(10).mean() if rolling_avg else ys[i]
        std = None if stds is None else stds[i]
        plot_mean_std(xs[i], y, std=std, label=",".join(exps[i]["hyperparameters"]), log_scale=log_scale)
    print(graph_path + name)
    save_graph(xlabel, ylabel, graph_path, name)

def process_pcg_exp_data(path, n_steps, num_envs, exp_name):
    # steps_per_rollout = n_steps * num_envs
    df = pd.read_csv(path + "data.csv", index_col=0)
    x = df.index # * steps_per_rollout + steps_per_rollout

    avg_rew_sum = df.iloc[:, 0]
    plot_mean_std(x, avg_rew_sum, df.iloc[:, 1], rolling_avg=True)
    save_graph("Rollouts", "Avg cumulative reward", path, f"rew_sum_{exp_name}.png")

    avg_p_per_ep = df.iloc[:, 2]
    std_p_per_ep = df.iloc[:, 3]
    plot_mean_std(x, avg_p_per_ep, std_p_per_ep, rolling_avg=True)
    save_graph("Rollouts", "Avg Playability reward per episode", path, f"avg_p_{exp_name}.png")

    avg_d_per_ep = df.iloc[:, 4]
    plot_mean_std(x, avg_d_per_ep, df.iloc[:, 5], rolling_avg=True)
    save_graph("Rollouts", "Avg Diversity reward per episode", path, f"avg_d_{exp_name}.png")

    avg_n_per_ep = df.iloc[:, 6]
    plot_mean_std(x, avg_n_per_ep, df.iloc[:, 7], rolling_avg=True)
    save_graph("Rollouts", "Avg Novelty reward per episode", path, f"avg_n_{exp_name}.png")

    rolling_avg_non_p_count = df.iloc[:, 8].rolling(10).mean()
    plt.plot(x, df.iloc[:, 8])
    plt.plot(x, rolling_avg_non_p_count, label="rolling avg")
    save_graph("Rollouts", "Avg number of non-playable segments per episode", path, f"avg_non_p_count_{exp_name}.png")
    return [x, avg_rew_sum, avg_p_per_ep, avg_d_per_ep, avg_n_per_ep, rolling_avg_non_p_count, std_p_per_ep], df.shape[0] # * steps_per_rollout

def process_pcg_exps_data(path, exps, use_min_x=True):
    new_path = f"{path}all/"
    data = []
    min_rollouts = float("inf")
    # max_index = -1
    for exp in exps:
        new_path += exp["name"] + "_"
        temp_data, temp_rollouts = process_pcg_exp_data(f"{path}{exp['name']}/", exp["n_steps"], exp["num_envs"], exp["name"])
        data.append(temp_data)
        if temp_rollouts < min_rollouts:
            min_rollouts = temp_rollouts
            # max_index = min_steps // (exp["n_steps"] * exp["num_envs"])
    new_path = f"{new_path[:-1]}/"
    print(new_path)
    os.makedirs(new_path, exist_ok=True)
    if use_min_x:
        data = [[data[i][j][data[i][0] <= min_rollouts] for j in range(len(data[i]))] for i in range(len(data))]
    xs = [data[i][0] + 1 for i in range(len(data))] # [data[i][0][:max_index] for i in range(len(data))]
    create_exps_graph(exps, xs, [data[i][1] for i in range(len(data))], False, "Rollouts", "Avg cumulative reward", new_path, "rew_sum_all.png", rolling_avg=True)
    create_exps_graph(exps, xs, [data[i][2] for i in range(len(data))], False, "Rollouts", "Avg Playability reward per episode", new_path, "avg_p_all.png", stds=[data[i][6] for i in range(len(data))] ,rolling_avg=True)
    create_exps_graph(exps, xs, [data[i][3] for i in range(len(data))], False, "Rollouts", "Avg Diversity reward per episode", new_path, "avg_d_all.png", rolling_avg=True)
    create_exps_graph(exps, xs, [data[i][4] for i in range(len(data))], False, "Rollouts", "Avg Novelty reward per episode", new_path, "avg_n_all.png", rolling_avg=True)
    create_exps_graph(exps, xs, [data[i][5] for i in range(len(data))], False, "Rollouts", "Avg number of non-playable segments per episode", new_path, "avg_non_p_count_all.png", rolling_avg=True)

if __name__ == "__main__":
    exps = [{"name":"static_one", "n_steps":16, "num_envs":1, "hyperparameters":["static", "pop=1"]},
            {"name":"static_four", "n_steps":16, "num_envs":4, "hyperparameters":["static", "pop=4"]},
            {"name":"coop_one", "n_steps":16, "num_envs":1, "hyperparameters":["coop", "pop=1"]},
            {"name":"coop_four", "n_steps":16, "num_envs":4, "hyperparameters":["coop", "pop=4"]},
            {"name":"hybrid_four", "n_steps":16, "num_envs":4, "hyperparameters":["hybrid", "pop=4"]},
            ]
    # process_pcg_exps_data(f"{DATA_PATH}{ALGO}/", [exps[0],exps[2]], use_min_x=True)
    process_pcg_exps_data(f"{DATA_PATH}{ALGO}/", [exps[2],exps[3],exps[4]], use_min_x=True)
