import random
import time

import numpy

import tensorflow as tf
import keras._tf_keras.keras.layers as layers
import keras._tf_keras.keras.models as models
import keras

import numpy as np

# from keras import layers
from Map import Map

import matplotlib.pyplot as plt

import csv

class ReplayMemory:
    def __init__(self, maxSize=5000):
        self.buffer = []
        self.maxSize = maxSize

    def add(self, memory):
        if len(self.buffer) >= self.maxSize:
            self.buffer.pop()

        self.buffer.insert(0, memory)

    def sample(self, size):
        return random.sample(self.buffer, size)

    def size(self):
        return len(self.buffer)

class Agent:
    def __init__(self, env=None, epsilon=1.0, epsilonDecay=0.0001, epsilonMin=0.1, isTraining=True, gamma=0.9, learningRate=0.003, batchSize = 32, trainFreq=5, targetUpdateFreq=50):
        if not env:
            self.env = Map(mapName="BackAndForth")
        else:
            self.env = env

        # use this if we want deep q learning
        self.model = self.createAgent(learningRate)

        self.targetModel = self.createAgent(learningRate)

        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay
        self.epsilonMin = epsilonMin
        self.isTraining = isTraining

        self.gamma = gamma
        self.learningRate = learningRate

        self.batchSize = batchSize

        self.trainFreq = trainFreq

        self.targetUpdateFreq = targetUpdateFreq

        self.updateCount = 0

        # # shape of q value array is size of the states and actions possible
        # self.qValues = np.zeros(shape=(self.env.size * self.env.size * 6, 5))

    # don't worry about this unless we want to try deep Q learning
    def createAgent(self, learningRate):
        model = tf.keras.Sequential([
            layers.Dense(64, input_dim=1, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(5, activation="linear")
        ])

        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=learningRate))

        return model

    # update weights
    def updateTarget(self):
        self.targetModel.set_weights(self.model.get_weights())


    def preprocessState(self, state):
        state = tuple(tuple(row) for row in state)

        hashVal = hash(state)

        # turn tuple into hash (I hope this works)
        hashVal = abs(hashVal % (self.env.size * self.env.size * 6))

        return hashVal

        #return state

    def isFinished(self):
        return self.env.goalReached

    def chooseAction(self, state):
        # if we are training...
        if self.isTraining:

            # decay the epsilon value
            if self.epsilon > self.epsilonMin:
                self.epsilon -= self.epsilonDecay

            # roll a random number
            randnum = random.uniform(0, 1)

            # choose randomly if smaller than epsilon
            if randnum < self.epsilon:
                randaction = np.random.randint(0, 5)
                print(randaction)
                return randaction

        # get the maximum value
        qVals = self.model.predict(np.array([state]))
        return np.argmax(qVals[0])

    def step(self, action):
        reward = self.env.makeAction(action)
        state = self.env.getGridLayout()
        done = self.isFinished()

        self.updateCount += 1

        # update the target every once in a while
        if self.updateCount >= self.targetUpdateFreq:
            self.updateCount = 0
            self.updateTarget()

        return reward, state, done


    def evaluate(self, numSteps=None, numEpisodes=1, sleep=False):

        Rewards = []
        Times = []

        self.isTraining = False
        for j in range(numEpisodes):
            self.env.reset()
            state = self.env.getGridLayout()
            state = self.preprocessState(state)
            stepCount = 0
            reward = 0
            while not self.isFinished():
                action = self.chooseAction(state)
                reward, nextState, done = self.step(action)

                nextState = self.preprocessState(nextState)

                if not sleep:
                    self.env.printGrid(0.1)

                state = nextState

                if numSteps:
                    if numSteps <= stepCount:
                        print("Agent Failed to reach goal in time with a final score of {}".format(reward))
                        if not sleep:
                            time.sleep(1)
                        break
                    else:
                        stepCount += 1

            Times.append(self.env.time)
            print("Episode {} ended in {} units of time with a final score of {}".format(j, self.env.time, reward))

            if not sleep:
                time.sleep(1)

            Rewards.append(reward)

        return sum(Rewards)/len(Rewards), sum(Times)/len(Times)

    def runQLearning(self, memory, numSteps=None, numEpisodes=1):
        self.isTraining = True
        lossVals = []
        for j in range(numEpisodes):
            self.env.reset()
            state = self.env.getGridLayout()
            state = self.preprocessState(state)
            stepCount = 0
            reward = 0
            totalReward = 0
            count = 0

            while not self.isFinished():
                action = self.chooseAction(state)
                reward, nextState, done = self.step(action)

                nextState = self.preprocessState(nextState)

                memory.add((state, action, reward, nextState, done))

                self.env.printGrid(0)
                state = nextState
                totalReward += reward

                if memory.size() > self.batchSize and self.trainFreq < count:
                    count = 0
                    batch = memory.sample(self.batchSize)

                    states, actions, rewards, nextStates, dones = zip(*batch)

                    states = np.array(states)
                    actions = np.array(actions)
                    rewards = np.array(rewards)
                    nextStates = np.array(nextStates)
                    dones = np.array(dones)

                    target = rewards + self.gamma * np.max(self.targetModel.predict(nextStates), axis=1) * (1-dones)
                    qVals = self.model.predict(states)

                    for k in range(self.batchSize):
                        qVals[k][actions[k]] = target[k]

                    history = self.model.fit(states, qVals, epochs=1, verbose=0)

                    lossVals.append(history.history['loss'][0])

                count += 1

                if numSteps:
                    if numSteps < stepCount:
                        print("Agent Failed to reach goal in time with a final score of {}".format(reward))
                        # time.sleep(1)
                        break

                stepCount += 1

            print("Episode {} ended in {} units of time with a final score of {}".format(j, self.env.time, reward))
            print("Epsilon Value: {}".format(self.epsilon))

        return sum(lossVals)/len(lossVals)

# CHANGE ME TO CHANGE ENV
envName = "FinalMap"
env = Map(10, envName)

memory = ReplayMemory()

qLearningAgent = Agent(env=env, epsilonDecay=0.00005, learningRate=0.0005, batchSize=128, trainFreq=5)

# let's see what happens after 100 episodes on this small instance
rewards = []

losses = []

times = []

for i in range(50):
    losses.append(qLearningAgent.runQLearning(memory, numEpisodes=10, numSteps=500))
    reward, evaltime = qLearningAgent.evaluate(numEpisodes=5, numSteps=500, sleep=True)
    rewards.append(reward)
    times.append(evaltime)

figure1 = plt.figure("Rewards")
plt.plot(rewards)
plt.ylabel("Reward")
plt.xlabel("Trained Episode Count (10s of Episodes)")
plt.title("DDQN Learning Reward over Time {}".format(envName))
plt.savefig("DDQNResults/" + envName + ".png")

figure2 = plt.figure("Losses")
plt.plot(losses)
plt.ylabel("Loss")
plt.xlabel("Episode Count (10s of Episodes)")
plt.title("DDQN Average Loss over Time {}".format(envName))
plt.savefig("DDQNResults/" + envName + "Loss.png")

figure3 = plt.figure("Time")
plt.plot(times)
plt.ylabel("Units of Time")
plt.xlabel("Trained Episode Count (10s of Episodes)")
plt.title("DDQN Average Time of episode completion over Time {}".format(envName))
plt.savefig("DDQNResults/" + envName + "Time.png")

print("BEGIN FINAL EVALUATION")
input("Press Enter to begin: ")
reward, evaltime = qLearningAgent.evaluate(numEpisodes=1, numSteps=500)

rewards.append(reward)
times.append(evaltime)

with open("DDQNResults/" + envName + ".csv", 'w') as f:
    write = csv.writer(f)

    write.writerow(['Reward'])

    for row in rewards:
        row = [row]
        write.writerow(row)

with open("DDQNResults/" + envName + "Loss.csv", 'w') as f:
    write = csv.writer(f)

    write.writerow(['Loss'])

    for row in losses:
        row = [row]
        write.writerow(row)

with open("DDQNResults/" + envName + "Time.csv", 'w') as f:
    write = csv.writer(f)

    write.writerow(['Time'])

    for row in times:
        row = [row]
        write.writerow(row)


