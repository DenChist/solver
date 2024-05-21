import numpy as np
import gym
from stable_baselines3 import PPO

class ProductionEnv(gym.Env):
    def __init__(self, n_tasks, n_stages, machines_per_stage, processing_times, deadlines):
        super(ProductionEnv, self).__init__()
        self.n_tasks = n_tasks
        self.n_stages = n_stages
        self.machines_per_stage = machines_per_stage
        self.processing_times = processing_times
        self.deadlines = deadlines
        
        self.action_space = gym.spaces.Discrete(14)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(n_tasks, n_stages, max(machines_per_stage)), dtype=np.float32)
        
        self.current_stage = 0

    def reset(self):
        self.state = np.zeros((self.n_tasks, self.n_stages, max(self.machines_per_stage)))
        self.current_time = 0
        self.task_completion = np.zeros(self.n_tasks)
        self.machine_availability = np.zeros((self.n_stages, max(self.machines_per_stage)))
        self.current_stage = 0
        return self.state

    def step(self, action):
        task, stage, machine = self.select_task_and_machine(action)
        reward, done = self.process_task(task, stage, machine)
        
        self.state[task, stage, machine] = 1
        self.current_time = max(self.current_time, self.machine_availability[stage, machine]) + self.processing_times[task, stage]
        self.task_completion[task] = max(self.task_completion[task], self.current_time)
        self.machine_availability[stage, machine] = self.current_time
        
        if np.all(self.state[:, stage, :] == 1):
            self.current_stage = min(self.current_stage + 1, self.n_stages - 1)
        
        done = self.current_stage == self.n_stages - 1 and np.all(self.state[:, -1, :] == 1)
        return self.state, reward, done, {}

    def select_task_and_machine(self, action):
        task_rule = action // 2
        machine_rule = action % 2
        task = self.select_task(self.current_stage, task_rule)
        machine = self.select_machine(self.current_stage, machine_rule)
        return task, self.current_stage, machine

    def select_task(self, stage, rule_index):
        if rule_index == 0:
            return np.argmin(self.processing_times[:, stage])  # SPT
        elif rule_index == 1:
            return np.argmax(self.processing_times[:, stage])  # LPT
        elif rule_index == 2:
            return np.argmin(self.deadlines)  # EDD
        elif rule_index == 3:
            return np.argmin(self.deadlines / self.processing_times.sum(axis=1))  # ODD
        elif rule_index == 4:
            return np.argmin(self.processing_times.sum(axis=1))  # SRP
        elif rule_index == 5:
            return np.argmax(self.processing_times[:, (stage + 1) % self.n_stages])  # LNP
        elif rule_index == 6:
            return np.argmin(self.processing_times[:, (stage + 1) % self.n_stages])  # SNP
        else:
            raise ValueError("Invalid rule index")

    def select_machine(self, stage, rule_index):
        if rule_index == 0:
            return np.argmin(self.machine_availability[stage, :self.machines_per_stage[stage]])  # FCFS
        elif rule_index == 1:
            return np.argmin(self.machine_availability[stage, :self.machines_per_stage[stage]].sum(axis=0))  # WINQ
        else:
            raise ValueError("Invalid rule index")

    def process_task(self, task, stage, machine):
        processing_time = self.processing_times[task, stage]
        self.machine_availability[stage, machine] += processing_time
        reward = - (np.max(self.task_completion) + np.sum(np.maximum(self.task_completion - self.deadlines, 0)))
        done = np.all(self.task_completion > 0)
        return reward, done

n_tasks = 5
n_stages = 3
machines_per_stage = [3, 2, 3]
processing_times = np.random.randint(20, 100, (n_tasks, n_stages))
deadlines = np.random.randint(200, 300, n_tasks)

env = ProductionEnv(n_tasks, n_stages, machines_per_stage, processing_times, deadlines)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    print(f"Action: {action}, Reward: {rewards}, Done: {done}")
