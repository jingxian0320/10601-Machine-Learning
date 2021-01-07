from environment import MountainCar
import sys
import numpy as np
import random
from scipy.sparse import csr_matrix


N_ACTION = 3 #(0) pushingthe car left, (1) doing nothing, and (2) pushing the car right.

def initialize(mode):
    if mode == "raw":
        return np.zeros((2, N_ACTION)), 0
    else:
        return np.zeros((2048, N_ACTION)), 0


def state_to_matrix(env, state):
    n = len(state.values())
    return csr_matrix((list(state.values()), (list(state.keys()), np.zeros(n))), shape=(env.state_space, 1))

def receive_sample(env, state, epsilon, w, b):
    if random.random() < epsilon:
        action = random.randint(0,2)
    else:
        q = state.T.dot(w) + b
        action = np.argmax(q)
    state,reward,done = env.step(action)
    return action, state,reward,done
        
        
    
def train(mode, episodes, max_interations, epsilon, gamma, learning_rate):
    returns = []
    env = MountainCar(mode)
    w, b = initialize(mode)
    for e in range(episodes):
        state = state_to_matrix(env, env.reset())
        sum_reward = 0
        for i in range(max_interations):
            action,next_state,reward,done = receive_sample(env, state, epsilon, w, b)
            sum_reward += reward
            
            next_state = state_to_matrix(env, next_state)
            old_q = (state.T.dot(w[:,action]) + b)[0]
            next_max = np.max(next_state.T.dot(w) + b)
            delta_w = np.zeros(w.shape)
            delta_w[:,action] = state.T.toarray()
            #print (delta_w)
            w = w - learning_rate  * (old_q - (reward + gamma * next_max)) * delta_w
            b = b - learning_rate  * (old_q - (reward + gamma * next_max))
            
            state = next_state
            if done:
                break
            
        returns.append(sum_reward)
    #print (returns)
    return w,b,returns
            
def output_weight(w, b, out_file):
    with open(out_file, "w") as output:
        output.write(str(b)+'\n')
        for row in w:
            for e in row:
                output.write(str(e)+'\n')
                
def output_returns(returns, out_file):
    with open(out_file, "w") as output:
        for i in returns:
            output.write(str(i)+'\n')


if __name__ == '__main__':
    mode = sys.argv[1] #‘raw’’or‘‘tile’’
    weight_out = sys.argv[2]
    returns_out = sys.argv[3]
    #the number of episodes your program should train the agent for.  
    #One episodeis a sequence of states, actions and rewards, 
    #which ends with terminal state or ends when the maximum episode length has been reached.
    episodes = int(sys.argv[4]) 
    max_iterations = int(sys.argv[5]) #maximum  of  the  length  of  an  episode.
    epsilon = float(sys.argv[6])
    gamma = float(sys.argv[7]) #the discount factor
    learning_rate  = float(sys.argv[8])
    
    w,b,returns = train(mode, episodes, max_iterations, epsilon, gamma, learning_rate)
    output_weight(w,b,weight_out)
    output_returns(returns,returns_out)
    