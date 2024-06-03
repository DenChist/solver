import gymnasium as gym
from gymnasium import spaces
import numpy as np

from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy

class FFSSEnv(gym.Env):
    def __init__(self, n_jobs, n_stages, machines, processing_times, due_dates):
        super(FFSSEnv, self).__init__()
        self.n_jobs = n_jobs
        self.n_stages = n_stages
        self.machines = machines
        self.processing_times = processing_times
        self.due_dates = due_dates
        
        self.action_space = spaces.Discrete(14)
        
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(n_jobs, n_stages + 1), dtype=np.float32)
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.job_completion_times = np.zeros((self.n_jobs, self.n_stages), dtype=np.float32)
        self.machine_available_times = np.zeros((self.n_stages, max(self.machines)), dtype=np.float32)
        self.current_time = 0
        return self.get_state(), {}

    def get_state(self):
        return np.hstack((self.job_completion_times, self.due_dates[:, None].astype(np.float32)))

    def step(self, action):
        job_selection_rule = action // 2
        machine_selection_rule = action % 2
        
        job = self.select_job(job_selection_rule)
        stage = self.select_stage(job)
        machine = self.select_machine(machine_selection_rule, stage)
        
        start_time = max(self.machine_available_times[stage][machine], self.current_time)
        processing_time = self.processing_times[job][stage]
        completion_time = start_time + processing_time
        
        self.job_completion_times[job][stage] = completion_time
        self.machine_available_times[stage][machine] = completion_time
        self.current_time = completion_time
        
        Cmax = np.max(self.job_completion_times[:, -1])
        T = np.sum(np.maximum(self.job_completion_times[:, -1] - self.due_dates, 0))
        reward = -1.0 * (Cmax + T)
        
        done = np.all(self.job_completion_times[:, -1] > 0)
        terminated = bool(done)
        truncated = False
        
        return self.get_state(), reward, terminated, truncated, {}
    
    def select_stage(self, job):
        for stage in range(self.n_stages):
            if self.job_completion_times[job][stage] == 0:
                return stage
        return self.n_stages - 1
    

    
    def select_job(self, rule):
        remaining_processing_times = np.sum(self.processing_times - self.job_completion_times, axis=1)
        next_stage_processing_times = self.processing_times[:, np.argmin(self.job_completion_times, axis=1)]
        
        if rule == 0:  # SPT
            job = np.argmin(remaining_processing_times)
        elif rule == 1:  # LPT
            job = np.argmax(remaining_processing_times)
        elif rule == 2:  # EDD
            job = np.argmin(self.due_dates)
        elif rule == 3:  # ODD
            operation_due_dates = self.due_dates - remaining_processing_times
            job = np.argmin(operation_due_dates)
        elif rule == 4:  # SRP
            job = np.argmin(self.processing_times.sum(axis=1))
        elif rule == 5:  # LNP
            job = np.argmax(next_stage_processing_times)
        elif rule == 6:  # SNP
            job = np.argmin(next_stage_processing_times)
        
        job = min(max(job, 0), self.n_jobs - 1)
        return job

    def select_machine(self, rule, stage):
        if rule == 0:  # FCFS
            machine = np.argmin(self.machine_available_times[stage])
        elif rule == 1:  # WINQ
            machine = np.argmin(np.sum(self.machine_available_times, axis=1))
        
        machine = min(max(machine, 0), len(self.machine_available_times[stage]) - 1)
        return machine



n_jobs = 5
n_stages = 3
machines = [2, 2, 2]
processing_times = np.random.randint(1, 10, (n_jobs, n_stages)).astype(np.float32)
due_dates = np.random.randint(10, 20, n_jobs).astype(np.float32)
env = FFSSEnv(n_jobs, n_stages, machines, processing_times, due_dates)

check_env(env)

model = DQN(MlpPolicy, env, verbose=1, learning_rate=1e-3, buffer_size=50000, learning_starts=1000, batch_size=32, tau=1.0, gamma=0.99, train_freq=4, gradient_steps=1, target_update_interval=100, exploration_fraction=0.1, exploration_final_eps=0.02, max_grad_norm=10)
model.learn(total_timesteps=100000)

model.save("d3qn_ffss")

model = DQN.load("d3qn_ffss")

obs, _ = env.reset()
for _ in range(n_jobs * n_stages):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, terminated, truncated, _ = env.step(action)
    if terminated:
        break

print("Finished with reward:", rewards)