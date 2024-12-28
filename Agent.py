import random
import time

import numpy
# import tensorflow as tf
# import keras

import numpy as np

# from keras import layers
from Map import Map

import matplotlib.pyplot as plt

import csv


class Agent:
    def __init__(self, env=None, epsilon=1.0, epsilonDecay=0.0001, epsilonMin=0.1, isTraining=True, gamma=0.9, learningRate=0.003):
        if not env:
            self.env = Map(mapName="BackAndForth")
        else:
            self.env = env

        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay
        self.epsilonMin = epsilonMin
        self.isTraining = isTraining

        self.gamma = gamma
        self.learningRate = learningRate

        # shape of q value array is size of the states and actions possible
        self.qValues = np.zeros(shape=(self.env.size * self.env.size * 6, 5))

    def preprocessState(self, state):
        state = tuple(tuple(row) for row in state)

        hashVal = hash(state)

        # turn tuple into hash (I hope this works)
        hashVal = abs(hashVal % (self.env.size * self.env.size * 6))

        return hashVal

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
        qVals = np.array(self.qValues[state])
        return np.argmax(qVals)

    def step(self, action):
        reward = self.env.makeAction(action)
        state = self.env.getGridLayout()

        return reward, state


    def evaluate(self, numSteps=None, numEpisodes=1, sleep=False):

        Rewards = []
        Times = []

        self.isTraining = False
        for i in range(numEpisodes):
            self.env.reset()
            state = self.env.getGridLayout()
            state = self.preprocessState(state)
            stepCount = 0
            reward = 0
            while not self.isFinished():
                action = self.chooseAction(state)
                reward, nextState = self.step(action)

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
            print("Episode {} ended in {} units of time with a final score of {}".format(i, self.env.time, reward))

            if not sleep:
                time.sleep(1)

            Rewards.append(reward)

        return sum(Rewards)/len(Rewards), sum(Times)/len(Times)

    def runQLearning(self, numSteps=None, numEpisodes=1):

        Rewards = []

        lossValues = []

        self.isTraining = True
        for i in range(numEpisodes):
            self.env.reset()
            state = self.env.getGridLayout()
            state = self.preprocessState(state)
            stepCount = 0
            reward = 0

            while not self.isFinished():
                action = self.chooseAction(state)
                reward, nextState = self.step(action)

                nextState = self.preprocessState(nextState)
                nextBestAction = np.argmax(np.array(self.qValues[nextState]))

                target = reward + self.gamma * self.qValues[nextState][nextBestAction]

                error = target - self.qValues[state][action]

                lossValues.append((error * error))

                self.qValues[state][action] += self.learningRate * error

                self.env.printGrid(0)

                state = nextState

                if numSteps:
                    if numSteps <= stepCount:
                        print("Agent Failed to reach goal in time with a final score of {}".format(reward))
                        # time.sleep(1)
                        break
                    else:
                        stepCount += 1

            print("Episode {} ended in {} units of time with a final score of {}".format(i, self.env.time, reward))
            print("Epsilon Value: {}".format(self.epsilon))

            Rewards.append(reward)

            # time.sleep(1)

        return (sum(lossValues) / len(lossValues))

# CHANGE ME TO CHANGE ENV
envName = "FinalMap"
env = Map(10, envName)

qLearningAgent = Agent(env=env, epsilonDecay=0.000005, learningRate=0.0005, epsilonMin=0.1)

# let's see what happens after 100 batches of episodes on this small instance
rewards = []

losses = []

times = []

for i in range(50):
    losses.append(qLearningAgent.runQLearning(numEpisodes=10, numSteps=500))
    reward, evalTime = qLearningAgent.evaluate(numEpisodes=10, numSteps=500, sleep=True)
    rewards.append(reward)
    times.append(evalTime)

figure1 = plt.figure("Rewards")
plt.plot(rewards)
plt.ylabel("Reward")
plt.xlabel("Trained Episode Count (10s of Episodes)")
plt.title("Q Learning Reward over Time {}".format(envName))
plt.savefig("QLearningResults/" + envName + ".png")


figure2 = plt.figure("Losses")
plt.plot(losses)
plt.ylabel("Reward")
plt.xlabel("Trained Episode Count (10s of Episodes)")
plt.title("Q Learning Loss over Time {}".format(envName))
plt.savefig("QLearningResults/" + envName + "Loss.png")

figure3 = plt.figure("Times")
plt.plot(times)
plt.ylabel("Time")
plt.xlabel("Trained Episode Count (10s of Episodes)")
plt.title("Q Learning Times per Episode over Trained Episode Count {}".format(envName))
plt.savefig("QLearningResults/" + envName + "Time.png")


print("BEGIN FINAL EVALUATION")
input("Press Enter to Continue:")
reward, evalTime = qLearningAgent.evaluate(numEpisodes=1, numSteps=1000)

rewards.append(reward)
times.append(evalTime)

with open("QLearningResults/" + envName + ".csv", 'w') as f:
    write = csv.writer(f)

    write.writerow(['Reward'])

    for row in rewards:
        row = [row]
        write.writerow(row)

with open("QLearningResults/" + envName + "Loss.csv", 'w') as f:
    write = csv.writer(f)

    write.writerow(['Loss'])

    for row in losses:
        row = [row]
        write.writerow(row)

with open("QLearningResults/" + envName + "Time.csv", 'w') as f:
    write = csv.writer(f)

    write.writerow(['Time'])

    for row in times:
        row = [row]
        write.writerow(row)

