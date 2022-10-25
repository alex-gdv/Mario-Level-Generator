import os
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from root import rootpath

DATA_PATH = f"{rootpath}/data/"
ALGO = "ppo"

def plot_mean_std(x, y, std=None, label=None, log_scale=False):
    if log_scale:
        plt.yscale("log")
    plt.plot(x, y, label=label)
    if std is not None:
        plt.fill_between(x, y-std, y+std, alpha=0.2)

def save_graph(xlabel, ylabel, path, name):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(path + name)
    plt.clf()

def process_play_exp_data(path, label):
    df = pd.read_csv(path + "data.csv", index_col=0)
    plot_mean_std(df.index, df.iloc[:, 0], df.iloc[:, 1], label)
    save_graph("Rollouts", "Avg cumulative reward", path, "rew_sum.png")
    plot_mean_std(df.index, df.iloc[:, 4], df.iloc[:, 5], label)
    save_graph("Rollouts", "Avg actions per pixel", path, "actions_per_pixel.png")
    plot_mean_std(df.index, df.iloc[:, 2], df.iloc[:, 3], label)
    save_graph("Rollouts", "Avg progress through level (%)", path, "level_progress.png")
    rolling_avg_win_ratio = df.iloc[:, 6].rolling(10).mean()
    plt.plot(df.index, df.iloc[:, 6])
    plt.plot(df.index, rolling_avg_win_ratio)
    save_graph("Rollouts", "Avg win ratio (%)", path, "win_ratio.png")
    return rolling_avg_win_ratio

def create_exps_graph(data_path, exps, y_index, std_index, log_scale, xlabel, ylabel, graph_path, name):
    for exp in exps:
        df = pd.read_csv(f"{data_path}{exp['name']}/{ALGO}/data.csv", index_col=0)
        rolling_avg = df.iloc[:, y_index].rolling(10).mean()
        plot_mean_std(df.index, rolling_avg, label=",".join(exp["hyperparameters"]), log_scale=log_scale)
    plt.legend()
    save_graph(xlabel, ylabel, graph_path, name)

def process_play_exps_data(path, exps):
    new_path = f"{path}"
    rolling_avg_win_ratio_lst = []
    for exp in exps:
        new_path += exp["name"] + "_"
        print(f"{path}{exp['name']}/{ALGO}/")
        temp = process_play_exp_data(f"{path}{exp['name']}/{ALGO}/", exp["hyperparameters"][0])
        rolling_avg_win_ratio_lst.append(temp)
    new_path = f"{new_path[:-1]}/{ALGO}/"
    os.makedirs(new_path, exist_ok=True)
    create_exps_graph(path, exps, 0, 1, False, "Rollouts", "Avg cumulative reward", new_path, "rew_sum_group.png")
    create_exps_graph(path, exps, 4, 5, False, "Rollouts", "Avg actions per pixel", new_path, "actions_per_pixel_group.png")
    create_exps_graph(path, exps, 2, 3, False, "Rollouts", "Avg progress through level (%)", new_path, "level_progress_group.png")
    for i in range(len(exps)):
        plt.plot(rolling_avg_win_ratio_lst[i], label=",".join(exps[i]["hyperparameters"]))
    plt.legend()
    save_graph("Rollouts", "Avg win ratio (%)", new_path, "win_ratio_group.png")

if __name__ == "__main__":
    exps = [{"name":"play16", "hyperparameters":["skip=30"]},
            {"name":"play17", "hyperparameters":["skip=60"]},
            {"name":"play18", "hyperparameters":["skip=10"]}]        
    process_play_exps_data(DATA_PATH, exps)
