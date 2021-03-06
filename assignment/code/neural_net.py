# import libraries
from types import MethodDescriptorType
import numpy as np
from tqdm.notebook import tqdm
import os
import json
import time
import random
from collections import namedtuple, deque

# import from files
from Chess_env import *



# ===== Epsilon-greedy Policy =====

def EpsilonGreedy_Policy(Qvalues, allowed_a, epsilon):
    """
    returns: tuple
        an action in form of a one-hot encoded vector with the same shapeensions as Qvalues.
        an action as decimal integer (0-based)

    Assumes only a single state, i.e. online learning and NOT (mini-)batch learning.
    """
    # get the Qvalues and the indices (relative of all Qvalues) for the allowed actions
    allowed_a_ind = np.where(allowed_a==1)[0]
    Qvalues_allowed = Qvalues[allowed_a_ind]
    

    # ------------ epsilon greedy ------------

    # draw a random number and compare it to epsilon
    rand_value = np.random.uniform(0, 1, 1)

    if rand_value < epsilon:  # if the random number is smaller than epsilon, draw a random action
        action_taken_ind_of_allwed_only = np.random.randint(0, len(allowed_a_ind))
    else:  # greedy action
        action_taken_ind_of_allwed_only = np.argmax(Qvalues_allowed)

    # get index of the action that was chosen (relative to all actions, not only allowed)
    ind_of_action_taken = allowed_a_ind[action_taken_ind_of_allwed_only]


    # ------------ create usable output ------------

    # get the shapeensions of the Qvalues
    N_a, N_samples = np.shape(Qvalues)  # N_samples must be 1

    # initialize all actions of binary mask to 0
    A_binary_mask = np.zeros((N_a,N_samples))
    # set the action that was chosen to 1
    A_binary_mask[ind_of_action_taken,:] = 1

    return A_binary_mask, ind_of_action_taken



# ===== activation functions and it's derivatives ======

# relu and its derivative
def relu(x):
    return np.maximum(0,x)

def heaviside(x):
    return np.heaviside(x,0)

# sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gradient_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

# tanh and its derivative
def tanh(x):
    return np.tanh(x)

def gradient_tanh(x):
    return 1 - np.tanh(x)**2

# identity and its derivative
def identity(x):
    return x

def const(x):
    return np.ones(x.shape)
    

def act_f_and_gradient(activation_function="relu"):
    if activation_function == "relu":
        return relu, heaviside
    elif activation_function == "sigmoid":
        return sigmoid, gradient_sigmoid
    elif activation_function == "tanh":
        return tanh, gradient_tanh
    else:  # identity and constant 1
        return identity, const



# ===== Replay Memory for Experience Replay (with DQN) =====

Transition = namedtuple('Transition', ("state", "action", "reward", "next_state", "done"))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        # if less data than batch size, return all data
        if len(self) < batch_size:
            batch_size = len(self)
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)




# ===== Neural Network ======

class NeuralNetwork(object):

    def __init__(self, N_in, N_h, N_a, activation_function_1="relu", activation_function_2=None, method="qlearning", seed=None, capacity=100_000, C=100):
        """
        activation functions: "relu", "sigmoid", "tanh", None
        methods: "qlearning", "sarsa", "dqn"
        """
        self.D = N_in  # input dimension (without bias)
        self.K = N_h   # nr hidden neurons (without bias)
        self.O = N_a   # nr output neurons (letter O, not digit 0)

        # store method and seed
        self.method = method
        self.seed = seed

        if self.method == "dqn":
            self.capacity = capacity
            self.replay_memory = ReplayMemory(capacity)
            self.C = C

        # set activation function and gradient function
        self.act_f_1_name = activation_function_1
        self.act_f_2_name = activation_function_2
        self.act_f_1, self.grad_act_f_1 = act_f_and_gradient(activation_function_1)
        self.act_f_2, self.grad_act_f_2 = act_f_and_gradient(activation_function_2)


        # initialize the weights and biases and set grobal seed
        np.random.seed(self.seed)

        # self.W1 = np.random.randn(self.K+1, self.D+1)/np.sqrt(self.D+1)  # standard normal distribution, shape: (K+1, D+1)
        # glorot/xavier normal initialization
        # self.W1 = np.random.randn(self.K+1, self.D+1)*np.sqrt(2/ (self.D+1 + self.K+1))  # standard normal distribution, shape: (K+1, D+1)
        self.W1 = np.random.standard_normal((self.K+1, self.D+1))*np.sqrt(2/ (self.D+1 + self.K+1))  # standard normal distribution, shape: (K+1, D+1)
        # self.W1 = np.random.randn(self.K+1, self.D+1)  # standard normal distribution, shape: (K+1, D+1)

        # self.W2 = np.random.randn(self.O, self.K+1)/np.sqrt(self.K+1)  # standard normal distribution, shape: (O, K+1)
        # glorot/xavier normal initialization
        self.W2 = np.random.standard_normal((self.O, self.K+1))*np.sqrt(2/ (self.K+1 + self.O))  # standard normal distribution, shape: (O, K+1)
        # self.W2 = np.random.randn(self.O, self.K+1)  # standard normal distribution, shape: (O, K+1)

        if self.method == "dqn":
            self.W1_target = np.copy(self.W1)
            self.W2_target = np.copy(self.W2)


    def forward(self, x, target=False):
        """
        x has shape: (D+1, 1) (constant bias 1 must be added beforehand added)
        target: if True, use the weights of the target network

        returns:
            last logits (i.e. Qvalues) of shape (O, 1)
        """

        if target == True:
            W1 = np.copy(self.W1_target)
            W2 = np.copy(self.W2_target)
        else:
            W1 = np.copy(self.W1)
            W2 = np.copy(self.W2)

        # forward pass/propagation
        a1 = W1 @ x
        h1 = self.act_f_1(a1)
        h1[0,:] = 1  # set first row (bias to second layer) to 1 (this ignores the weights for the k+1th hidden neuron, because this should not exist; this allows to only use matrix multiplication and simplify the gradients as we only need 2 instead of 4)
        a2 = W2 @ h1
        h2 = self.act_f_2(a2)
        return a1, h1, a2, h2


    def backward(self, R, x, Qvalues, Q_prime, a1, h1, a2, gamma, future_reward, action_binary_mask):
        """
        backward for methods "qlearning" and "sarsa"

        x has shape (D+1, 1) (constant bias 1 must be added beforehand)
        set future_reward=True for future reward with gamma>0, False for immediate reward.
        Q_prime must be chosen according to the method on x_prime (on- or off-policy)
        """

        # backward pass/backpropagation
        # compute the gradient of the square loss with respect to the parameters
        
        # ===== compute TD error (aka delta) =====

        # make reward of shape (O, 1)
        R_rep = np.tile(R, (self.O, 1))
        if future_reward:  # future reward
            delta = R_rep + gamma*Q_prime - Qvalues  # -> shape (O, 1)
        else:  # immediate reward
            delta = R_rep - Qvalues  # -> shape (O, 1)
        
        # update only action that was taken, i.e. all rows apart from the one corresponding to the action taken (action index) are 0
        delta = delta*action_binary_mask
        

        self.compute_gradients(delta, a1, h1, a2, x)
        self.update_parameters(self.eta)

    
    def backward_dqn(self, batch, gamma):
        """
        backward for method "dqn"
        """

        # ===== compute targets y and feature matrix X =====

        # turn batch into individual tuples, numpy arrays, or lists
        states = batch.state
        rewards = np.array(list(batch.reward))
        actions = np.array(list(batch.action))
        next_states = list(batch.next_state)
        dones = np.array(list(batch.done))

        # compute targets y and feature matrix X
        y = np.zeros((self.O, len(dones)))
        for j in np.arange(len(dones)):
            if dones[j]:  # if done, set y_j = r_j
                y[actions[j], j] = rewards[j]
            else:
                # compute Q_prime
                Q_target = self.forward(next_states[j], target=True)[-1]
                y[actions[j], j] = rewards[j] + gamma*np.max(Q_target)


        # convert states to feature matrix X
        X = np.hstack((states))


        # ===== compute TD error (aka delta) =====

        a1, h1, a2, Qvalues = self.forward(X)
        delta = y - Qvalues  # -> shape (O, batch_size)

        self.compute_gradients(delta, a1, h1, a2, X)
        self.update_parameters(self.eta)


    def compute_gradients(self, delta, a1, h1, a2, x):
        # ===== compute gradient of the loss with respect to the weights =====

        # common part of the gradient  TODO: check dimensions
        self.dL_da2 = delta * self.grad_act_f_2(a2) 

        # gradient of loss wrt W2
        self.dL_dW2 = self.dL_da2 @ h1.T

        # gradient of loss wrt W1
        self.dL_dW1 = ( (self.W2.T @ self.dL_da2) * self.grad_act_f_1(a1) ) @ x.T



    def update_parameters(self, eta):

        # gradient clipping

        # dL_dW1_norm = np.linalg.norm(self.dL_dW1)
        # if dL_dW1_norm >= self.gradient_clip:
        #     self.dL_dW1 = self.gradient_clip * self.dL_dW1 / dL_dW1_norm

        # dL_dW2_norm = np.linalg.norm(self.dL_dW2)
        # if dL_dW2_norm >= self.gradient_clip:
        #     self.dL_dW2 = self.gradient_clip * self.dL_dW2 / dL_dW2_norm

        # update W1 and W2 
        self.W2 = self.W2 + eta * self.dL_dW2
        self.W1 = self.W1 + eta * self.dL_dW1




    def train(self, env, N_episodes, eta, epsilon_0, beta, gamma, alpha=0.001, gradient_clip=1, batch_size=32, run_number=None):
        """
        alpha is used as weight for the exponential moving average displayed during training.
        batch_size is only used for the DQN method.
        """

        # add training hyper parameters
        self.N_episodes = N_episodes
        self.eta = eta
        self.epsilon_0 = epsilon_0
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.gradient_clip = gradient_clip
        self.batch_size = batch_size


        training_start = time.time()

        try:

            # initialize histories for important metrics
            self.R_history = np.full([self.N_episodes, 1], np.nan)
            self.N_moves_history = np.full([self.N_episodes, 1], np.nan)
            self.dL_dW1_norm_history = np.full([self.N_episodes, 1], np.nan)
            self.dL_dW2_norm_history = np.full([self.N_episodes, 1], np.nan)

            # progress bar
            episodes = tqdm(np.arange(self.N_episodes), unit="episodes")
            ema_previous = 0

            n_steps = 0

            for n in episodes:

                epsilon_f = self.epsilon_0 / (1 + beta * n)   ## DECAYING EPSILON
                Done = 0                                   ## SET DONE TO ZERO (BEGINNING OF THE EPISODE)
                i = 1                                    ## COUNTER FOR NUMBER OF ACTIONS
                
                S, X, allowed_a = env.Initialise_game()      ## INITIALISE GAME
                X = np.expand_dims(X, axis=1)                     ## MAKE X A TWO DIMENSIONAL ARRAY
                X = np.copy(np.vstack((np.array([[1]]), X)))  # add bias term

                if self.method == "sarsa":
                    # compute Q values for the given state
                    a1, h1, a2, Qvalues = self.forward(X)  # -> shape (O, 1)

                    # choose an action A using epsilon-greedy policy
                    A_binary_mask, A_ind = EpsilonGreedy_Policy(Qvalues, allowed_a, epsilon_f)  # -> shape (O, 1)


                while Done==0:                           ## START THE EPISODE

                    if (self.method == "qlearning") or (self.method == "dqn"):
                        # compute Q values for the given state
                        a1, h1, a2, Qvalues = self.forward(X)  # -> shape (O, 1)

                        # choose an action A using epsilon-greedy policy
                        A_binary_mask, A_ind = EpsilonGreedy_Policy(Qvalues, allowed_a, epsilon_f)  # -> shape (O, 1)


                    # take action and observe reward R and state S_prime
                    S_prime, X_prime, allowed_a_prime, R, Done = env.OneStep(A_ind)
                    X_prime = np.expand_dims(X_prime, axis=1)
                    X_prime = np.copy(np.vstack((np.array([[1]]), X_prime)))  # add bias term

                    n_steps += 1

                    if self.method == "dqn":

                        # store the transition in memory
                        self.replay_memory.push(X, A_ind, R, X_prime, Done)

                        # sample a batch of transitions
                        transactions = self.replay_memory.sample(self.batch_size)
                        # turn list of transactions into transaction of lists
                        batch = Transition(*zip(*transactions))

                        # backward step and parameter update
                        self.backward_dqn(batch, self.gamma)
                    
                    # update Q values indirectly by updating the weights and biases directly

                    if Done==1:  # THE EPISODE HAS ENDED, UPDATE...BE CAREFUL, THIS IS THE LAST STEP OF THE EPISODE

                        if (self.method == "qlearning") or (self.method == "sarsa"):
                            # compute gradients and update weights
                            self.backward(R, X, Qvalues, None, a1, h1, a2, None, future_reward=False, action_binary_mask=A_binary_mask)

                        # store history
                        # todo: record max possible reward per episode
                        self.R_history[n] = np.copy(R)  # reward per episode
                        self.N_moves_history[n] = np.copy(i)  # nr moves per episode

                        # store norm of gradients
                        self.dL_dW1_norm_history[n] = np.linalg.norm(self.dL_dW1)
                        self.dL_dW2_norm_history[n] = np.linalg.norm(self.dL_dW2)

                        # compute exponential moving average (EMA) to display during training
                        ema = alpha*R + (1-alpha)*ema_previous
                        if n == 0: # first episode
                            ema = R
                        ema_previous = ema
                        if run_number is not None:
                            episodes.set_description(f"Run = {run_number}; EMA Reward = {ema:.2f}")
                        else:
                            episodes.set_description(f"EMA Reward = {ema:.2f}")

                        break

                    else:  # IF THE EPISODE IS NOT OVER...

                        if self.method == "qlearning":
                            # chose next action off-policy
                            Q_prime = np.max(self.forward(X_prime)[-1])

                        elif self.method == "sarsa":
                            # chose next action on-policy

                            a1_prime, h1_prime, a2_prime, Qvalues_prime = self.forward(X_prime)  # -> shape (N_a, 1)

                            # chose next action and save it
                            A_binary_mask_prime, A_ind_prime = EpsilonGreedy_Policy(Qvalues_prime, allowed_a_prime, epsilon_f)

                            # get Qvalue of next action
                            Q_prime = Qvalues_prime[A_ind_prime]


                        if (self.method == "qlearning") or (self.method == "sarsa"):
                            # backpropagation and weight update
                            self.backward(R, X, Qvalues, Q_prime, a1, h1, a2, self.gamma, future_reward=True, action_binary_mask=A_binary_mask)

                        
                        # NEXT STATE AND CO. BECOME ACTUAL STATE...     
                        if self.method == "sarsa":
                            A_binary_mask = np.copy(A_binary_mask_prime)
                            A_ind = np.copy(A_ind_prime)
                            a1 = np.copy(a1_prime)
                            h1 = np.copy(h1_prime)
                            a2 = np.copy(a2_prime)
                            Qvalues = np.copy(Qvalues_prime)
                        S = np.copy(S_prime)
                        X = np.copy(X_prime)
                        allowed_a = np.copy(allowed_a_prime)
                        
                        i += 1  # UPDATE COUNTER FOR NUMBER OF ACTIONS
                    
                    if (self.method == "dqn") and (n_steps % self.C == 0):
                        # update target network every C steps
                        self.W1_target = np.copy(self.W1)
                        self.W2_target = np.copy(self.W2)


            training_end = time.time()
            self.training_time_in_seconds = training_end - training_start

            return None

        
        except KeyboardInterrupt as e:
            # return nothing
            training_end = time.time()
            self.training_time_in_seconds = training_end - training_start

            return None

    
    def save(self, name_extension=None):
        # create directory for the model
        name = f"{self.method}_{self.act_f_1_name}_{self.act_f_2_name}"
        if name_extension is not None:
            name += f"_{name_extension}"

        path = f"models/{name}"
        if not os.path.isdir(path): os.mkdir(path)
        print(f"saving to: {path}")

        # save weights
        np.save(f"{path}/W1.npy", self.W1)
        np.save(f"{path}/W2.npy", self.W2)

        # save training history
        np.save(f"{path}/training_history_R.npy", self.R_history)
        np.save(f"{path}/training_history_N_moves.npy", self.N_moves_history)
        np.save(f"{path}/training_history_dL_dW1_norm.npy", self.dL_dW1_norm_history)
        np.save(f"{path}/training_history_dL_dW2_norm.npy", self.dL_dW2_norm_history)

        # save training parameters and other general info
        params = {
            "method": self.method,
            "N_episodes": self.N_episodes,
            "eta": self.eta,
            "epsilon_0": self.epsilon_0,
            "beta": self.beta,
            "gamma": self.gamma,
            "alpha": self.alpha,
            # "gradient_clip": self.gradient_clip,
            "seed": self.seed,
            "D": self.D,
            "K": self.K,
            "O": self.O,
            "training_time_in_seconds": self.training_time_in_seconds
        }
        if self.method == "dqn":
            params["capacity"] = self.capacity
            params["batch_size"] = self.batch_size
            params["C"] = self.C
        with open(f"{path}/training_parameters.json", "w") as f:
            json.dump(params, f)

    
def load_from(method, act_f_1, act_f_2, name_extension=None):

    # read values and store in neural network instance
    name = f"{method}_{act_f_1}_{act_f_2}"
    if name_extension is not None:
        name += f"_{name_extension}"

    path = f"models/{name}"
    # print(f"loading from: {path}")

    # initialize neural network
    nn = NeuralNetwork(0,0,0, activation_function_1=act_f_1, activation_function_2=act_f_2, method=method)

    # network weights
    nn.W1 = np.load(f"{path}/W1.npy")
    nn.W2 = np.load(f"{path}/W2.npy")

    # network training history
    nn.R_history = np.load(f"{path}/training_history_R.npy")
    nn.N_moves_history = np.load(f"{path}/training_history_N_moves.npy")
    nn.dL_dW1_norm_history = np.load(f"{path}/training_history_dL_dW1_norm.npy")
    nn.dL_dW2_norm_history = np.load(f"{path}/training_history_dL_dW2_norm.npy")

    # network training parameters
    with open(f"{path}/training_parameters.json", "r") as f:
        params = json.load(f)

        # set parameters to the network instance
        nn.method = params["method"]
        nn.N_episodes = int(params["N_episodes"])
        nn.eta = float(params["eta"])
        nn.epsilon_0 = float(params["epsilon_0"])
        nn.beta = float(params["beta"])
        nn.gamma = float(params["gamma"])
        nn.alpha = float(params["alpha"])
        # nn.gradient_clip = float(params["gradient_clip"])
        try:
            nn.seed = int(params["seed"])
        except:
            nn.seed = params["seed"]
        nn.D = int(params["D"])
        nn.K = int(params["K"])
        nn.O = int(params["O"])
        nn.training_time_in_seconds = float(params["training_time_in_seconds"])

        if nn.method == "dqn":
            nn.capacity = int(params["capacity"])
            nn.batch_size = int(params["batch_size"])
            nn.C = int(params["C"])


    if nn.method == "dqn":
        nn.W1_target = np.copy(nn.W1)
        nn.W2_target = np.copy(nn.W2)

    return nn