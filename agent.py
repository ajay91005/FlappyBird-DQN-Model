import gymnasium as gym
import flappy_bird_gymnasium
import torch
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import random
import argparse
import os

if torch.cuda.is_available():
    device="cuda"
else:
    device="cpu"

RUNS_DIR='runs'
os.makedirs(RUNS_DIR, exist_ok=True)


class Agent:
    def __init__(self, param_set):
        self.param_set=param_set

        with open("parameters.yaml","r") as f:
            all_param=yaml.safe_load(f)
            params=all_param[param_set]
        
        self.epsilon_init=params["epsilon_init"]
        self.epsilon_min=params["epsilon_min"]
        self.epsilon_decay=params["epsilon_decay"]
        self.replay_memory_size=params["replay_memory_size"]
        self.mini_batch_size=params["mini_batch_size"]
        self.network_sync_rate=params["network_sync_rate"]
        self.alpha=params["alpha"]
        self.gamma=params["gamma"]
        self.reward_threshold=params["reward_threshold"]

        self.loss=nn.MSELoss()
        self.optimizer=None

        self.LOG_FILE=os.path.join(RUNS_DIR, f"{self.param_set}.log")
        self.MODEL_FILE=os.path.join(RUNS_DIR, f"{self.param_set}.pt")


    def run(self, is_training=True, render=False ):
        env=gym.make("FlappyBird-v0", render_mode="human" if render else None)

        input_state=env.observation_space.shape[0] #input_dimension
        output_state=env.action_space.n  #output dimension

        policy_dqn=DQN(input_state, output_state).to(device)


        if is_training:
            memory=ReplayMemory(self.replay_memory_size)
            epsilon=self.epsilon_init

            target_dqn=DQN(input_state, output_state).to(device)

            #copy params from policy to target
            target_dqn.load_state_dict(policy_dqn.state_dict())

            steps=0
            self.optimizer=optim.Adam(policy_dqn.parameters(), lr=self.alpha)

            best_reward=float("-inf")

        else:
            #load the model from saved one
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        for episode in itertools.count():
            state,_=env.reset()
            state=torch.tensor(state, dtype=torch.float, device=device)

            episode_rewards=0
            terminated=False

            while (not terminated and episode_rewards<self.reward_threshold):

                if is_training and random.random()<epsilon:
                    action=env.action_space.sample()
                    action=torch.tensor(action, dtype=torch.long, device=device)
                
                else:
                    with torch.no_grad():
                        action=policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()
            

                next_state, reward, terminated, _, info =env.step(action.item())

                next_state=torch.tensor(next_state, dtype=torch.float, device=device)
                reward=torch.tensor(reward, dtype=torch.float, device=device)

                if is_training:
                    memory.append(state, action, next_state, reward, terminated)
                    steps+=1

                    # if len(memory)>self.mini_batch_size:
                    #     #get sample
                    #     mini_batch=memory.sample(self.mini_batch_size)

                    #     self.optimize(mini_batch, policy_dqn, target_dqn)

                    #     # epsilon=max(epsilon*self.epsilon_decay, self.epsilon_min)

                    #     #sync the network
                    #     if steps>self.network_sync_rate:
                    #         target_dqn.load_state_dict(policy_dqn.state_dict())
                    #         steps=0
                    
                state=next_state
                episode_rewards+=reward.item()

            print(f"For episode={episode+1}, reward={episode_rewards} & epsilon")

            if is_training:
                #epsilon_decay
                epsilon=max(epsilon*self.epsilon_decay, self.epsilon_min)

                #Saving the model
                if episode_rewards>best_reward:
                    log_msg=f"best reward={episode_rewards} for episode={episode+1}"

                    with open(self.LOG_FILE, "a") as f:
                        f.write(log_msg + "\n")

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward=episode_rewards

            if is_training and len(memory)>self.mini_batch_size:
                #get sample
                mini_batch=memory.sample(self.mini_batch_size)

                self.optimize(mini_batch, policy_dqn, target_dqn)

                #sync the network
                if steps>self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    steps=0
            # # env.close()

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        #get batch of experiences
        states,actions,next_states,rewards,terminations =zip(*mini_batch)

        states=torch.stack(states)
        actions=torch.stack(actions)
        next_states=torch.stack(next_states)
        rewards=torch.stack(rewards)
        terminations=torch.tensor(terminations).float().to(device)

        #Calculating target Q-values-if terminations=true=>0

        with torch.no_grad():
            target_q=rewards+(1-terminations)*self.gamma*target_dqn(next_states).max(dim=1)[0]

        #calculation predicted q-values
        current_q=policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()


        #Loss
        loss=self.loss(current_q,target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__== "__main__":
    #parse commandline inputs
    parser=argparse.ArgumentParser(description='Train or Test Model')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args=parser.parse_args()

    dql=Agent(param_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    
    else:
        dql.run(is_training=False, render=True)

    