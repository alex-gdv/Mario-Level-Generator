from stable_baselines3.common.callbacks import BaseCallback
import os
import math
import pandas as pd

def mean_std(lst):
    avg = sum(lst) / len(lst)
    std = math.sqrt(sum([(num-avg)**2 for num in lst])/len(lst))
    return avg, std

class Play_Callback(BaseCallback):
    def __init__(self, model_save_freq, model_path, data_save_freq, data_path, verbose=0):
        super(Play_Callback, self).__init__(verbose)
        self.model_save_freq = model_save_freq
        self.model_path = model_path
        self.data_save_freq = data_save_freq
        self.data_path = data_path
        self.data = []
        self.indices = []

    def _init_callback(self):
        if self.model_path is not None:
            os.makedirs(self.model_path, exist_ok=True)
        if self.data_path is not None:
            os.makedirs(self.data_path, exist_ok=True)

    def _on_step(self):
        if self.num_timesteps % self.model_save_freq == 0:
            print("Saving model...")
            self.model.save(f"{self.model_path}best_model_{self.num_timesteps}")
        if self.num_timesteps % self.data_save_freq == 0:
            print("Saving data...")
            df = pd.DataFrame(self.data)
            df.to_csv(f"{self.data_path}data.csv")
        return True

    def _on_rollout_end(self):
        if len(self.indices) == 0:
            self.indices = [0 for _ in range(len(self.locals["infos"]))]
        rew_sum = []
        pct_comp = []
        acts_dist_ratio = []
        wins = 0
        for i in range(len(self.locals["infos"])):
            for j in range(self.indices[i], len(self.locals["infos"][i]["data"])):
                rew_sum.append(self.locals["infos"][i]["data"][j][0])
                pct_comp.append(self.locals["infos"][i]["data"][j][1] / self.locals["infos"][i]["data"][j][3])
                if self.locals["infos"][i]["data"][j][1] != 0:
                    acts_dist_ratio.append(self.locals["infos"][i]["data"][j][2] / self.locals["infos"][i]["data"][j][1])
                if self.locals["infos"][i]["data"][j][4]:
                    wins += 1
            self.indices[i] = len(self.locals["infos"][i]["data"])
        if len(rew_sum) > 0:
            avg_rew_sum, std_rew_sum = mean_std(rew_sum)
            avg_pct_comp, std_pct_comp = mean_std(pct_comp)
            avg_acts_dist_ratio, std_acts_dist_ratio = mean_std(acts_dist_ratio)
            win_ratio = wins / len(rew_sum)
            self.data.append((avg_rew_sum, std_rew_sum, avg_pct_comp, std_pct_comp, avg_acts_dist_ratio, std_acts_dist_ratio, win_ratio))

    def _on_training_end(self):
        df = pd.DataFrame(self.data)
        df.to_csv(f"{self.data_path}data.csv")

class PCG_Callback(BaseCallback):
    def __init__(self, model_save_freq, model_path, data_save_freq, data_path, verbose=0):
        super(PCG_Callback, self).__init__(verbose)
        self.model_save_freq = model_save_freq
        self.model_path = model_path
        self.data_save_freq = data_save_freq
        self.data_path = data_path
        self.data = []
        self.indices = []

    def _init_callback(self):
        if self.model_path is not None:
            os.makedirs(self.model_path, exist_ok=True)
        if self.data_path is not None:
            os.makedirs(self.data_path, exist_ok=True)

    def _on_step(self):
        if self.num_timesteps % self.model_save_freq == 0:
            print("Saving model...")
            self.model.save(f"{self.model_path}best_model_{self.num_timesteps}")
        return True
    
    def _on_rollout_end(self):
        if len(self.indices) == 0:
            self.indices = [0 for _ in range(len(self.locals["infos"]))]
        rew_sum = []
        p_per_ep = []
        d_per_seg = []
        n_per_seg = []
        non_playable_count = 0
        for i in range(len(self.locals["infos"])):
            for j in range(self.indices[i], len(self.locals["infos"][i]["data"])):
                rew_sum.append(self.locals["infos"][i]["data"][j][4])
                p_per_ep.append(self.locals["infos"][i]["data"][j][0])
                d_per_seg += self.locals["infos"][i]["data"][j][1]
                n_per_seg += self.locals["infos"][i]["data"][j][2]
                if not self.locals["infos"][i]["data"][j][3]:
                    non_playable_count += 1
            self.indices[i] = len(self.locals["infos"][i]["data"])
        if len(rew_sum) > 0:
            avg_rew_sum, std_rew_sum = mean_std(rew_sum)
            avg_p_per_ep, std_p_per_ep = mean_std(p_per_ep)
            avg_d_per_seg, std_d_per_seg = mean_std(d_per_seg)
            avg_n_per_seg, std_n_per_seg = mean_std(n_per_seg)
            self.data.append((avg_rew_sum, std_rew_sum, avg_p_per_ep, std_p_per_ep, avg_d_per_seg, std_d_per_seg,
                                avg_n_per_seg, std_n_per_seg, non_playable_count))
            print("Saving data...")
            df = pd.DataFrame(self.data)
            df.to_csv(f"{self.data_path}data.csv")

    def _on_training_end(self):
        df = pd.DataFrame(self.data)
        df.to_csv(f"{self.data_path}data.csv")
