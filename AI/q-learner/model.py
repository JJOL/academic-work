import numpy as np
import gym
import random
import time
# from IPython.display import clear_output


def initModel(envName):
    env = gym.make(envName)

    action_space_size = env.action_space.n
    state_space_size = env.observation_space.n

    q_table = np.zeros((state_space_size, action_space_size))

    return q_table, env


def trainModel(model, world, *params):
    q_table = model
    env = world
    n_episodes, max_steps_per_episode, learning_rate, discount_rate, min_exploration_rate, exploration_decay_rate = params

    # Hyper Parameters
    max_exploration_rate = 1
    exploration_rate = 1
    rewards_all_episodes = []

    debug_episode_count = int(n_episodes / 10)
    
    # Q-Learning
    for episode in range(n_episodes):
        if episode % debug_episode_count == 0:
            print("Training episode %d..." % (episode,))
        state = env.reset()
        
        done = False
        rewards_current_episode = 0
        
        for step in range(max_steps_per_episode):
            
            exploration_rate_threshold = random.uniform(0, 1)
            if (exploration_rate_threshold > exploration_rate):
                action = np.argmax(q_table[state, :])
            else:
                action = env.action_space.sample()
                
            new_state, reward, done, info = env.step(action)
            
            # Update Q-table for Q(s,a)
            q_table[state, action] = (1 - learning_rate) * q_table[state, action] + \
                learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))
            
            state = new_state
            rewards_current_episode += reward
            
            if done == True:
                break
                
        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)
                
        rewards_all_episodes.append(rewards_current_episode)

    rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), n_episodes/1000)
    count = 1000
    print("********Average reward per thousand episodes*******\n")
    for r in rewards_per_thousand_episodes:
        print(count, ": ", str(sum(r/1000)))
        count += 1000

    print("\n\n*******Q-table**************\n")
    print(q_table)


# Play Game
def playModel(model, world, max_steps_per_episode, victory_reward):
    q_table = model
    env = world
    for episode in range(3):
        state = env.reset()
        done = False
        print("*******Episode %d*******\n\n\n" % (episode+1,))
        time.sleep(1)
        
        for step in range(max_steps_per_episode):
            # clear_output(wait=True)
            env.render()
            time.sleep(0.3)
            
            action = np.argmax(q_table[state, :])
            new_state, reward, done, info = env.step(action)
            
            if done:
                # clear_output(wait=True)
                env.render()
                if (reward == victory_reward):
                    print("****Agent reached the goal!****")
                    time.sleep(3)
                else:
                    print("****You fell through a hole!****")
                    time.sleep(3)
                # clear_output(waiclt=True)
                break
                
            state = new_state
            
    env.close()