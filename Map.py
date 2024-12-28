# Changing Map by Robert Walsh
'''
This object exists as a way to train and test a different style of deep reinforcement learning pathfinding AI
maps can be generated on the fly or called from map.json if there is a specific layout that is desired.
Agent is capable of making at most 5 actions at any given time: Left, Right, Up, Down, or Nothing
There exists 6 possible states that a tile in the map can have: Impassible (-1), Occupied by Agent (0), Goal (1),
Normal (2), Slow (3), Very Slow (4)
Impassible tiles cannot be entered under any circumstances, and normal, slow, and very slow tiles take time to enter
Additionally, the map will keep track of a "bonus" value which will be rewarded upon reaching the goal.
Faster times means a bigger bonus. It is our prediction that it could help the agent avoid slower tiles
To help motivate the agent, there will be rewards for getting closer to the goal, and penalties for moving away
'''

import json
import os
import random
import time
import copy
from colorama import init
from termcolor import colored

init()


class Map:
    def __init__(self, size=5, mapName=None):

        # we need 2d array to hold map data
        self.grid = []
        self.backupGrid = []

        # this size will determine how big the array will be. size x size
        self.size = size

        # finally initialize the starting position of the agent
        self.startingPosition = [0, 0]
        self.position = [0, 0]
        self.goalPosition = [0, 0]

        # swap the agent and current position values
        self.holder = 0

        # keep track of time for bonus and changing environment
        self.time = 0

        # how much reward is there for the current position, how close/far are we from goal?
        self.currentReward = 0

        # if a name is provided, then just use the data
        if mapName:
            self.loadGrid(mapName)
        else:
            # we then want to generate our grid with this
            self.generateGrid()

        # make a bonus equal to size cubed
        self.bonus = self.size * self.size * self.size

        # have we reached a goal?
        self.goalReached = False

        self.startDistance = self.getCurrDistance()

        #random.seed(27)

    def generateGrid(self):
        # generate a 2d array with random values for a grid
        for i in range(0, self.size):
            self.grid.append([])
            for j in range(0, self.size):
                self.grid[i].append(2)

        self.backupGrid = copy.deepcopy(self.grid)

    def reset(self):
        self.grid = copy.deepcopy(self.backupGrid)
        self.position = copy.deepcopy(self.startingPosition)
        self.holder = self.grid[self.position[0]][self.position[1]]
        self.grid[self.position[0]][self.position[1]] = 0
        self.goalReached = False
        self.time = 0
        #random.seed(27)

    def loadGrid(self, name):
        with open("map.json", "r") as file:
            data = json.load(file)

        if name in data.keys():
            self.grid = data[name][0]
            self.backupGrid = copy.deepcopy(self.grid)
            self.startingPosition = data[name][1]
            self.position = copy.deepcopy(self.startingPosition)
            self.goalPosition = data[name][2]

            # hold onto the starting position data and swap grid data with agent
            self.holder = self.grid[self.position[0]][self.position[1]]
            self.grid[self.position[0]][self.position[1]] = 0
            self.size = len(self.grid)

        else:
            print("map of name does not exist")

    def printGrid(self, sleepTime):
        os.system("clear")
        # print results like this so it's easier to understand
        for array in self.grid:
            rowstring =""
            for element in array:
                if element == 1:
                    element = colored(str(element), "green")
                elif element == 0:
                    element = colored(str(element), "green")
                elif element == 2:
                    element = colored(str(element), "light_yellow")
                elif element == 3:
                    element = colored(str(element), "yellow")
                elif element == 4:
                    element = colored(str(element), "red")

                if len(str(element)) == 2:
                    rowstring += " " + str(element)
                else:
                    rowstring += "  " + str(element)
            print(rowstring)

        time.sleep(sleepTime)

    def getGridLayout(self):
        return self.grid

    # change the grid randomly, a representation of changing road conditions
    def changeGrid(self):
        # so we don't have to have 3 if checks, just see if value falls in here
        values = [2, 3, 4]

        # go through each value in the grid
        for i in range(0, self.size):
            for j in range(0, self.size):
                # if the value is traversable and random chance says "change it"...
                if self.grid[i][j] in values and random.randint(0, 99) < 10:
                    # roll again to see if conditions worsen or get better
                    if random.randint(0, 99) > 50:
                        self.grid[i][j] = min(4, self.grid[i][j] + 1)
                    else:
                        self.grid[i][j] = max(2, self.grid[i][j] - 1)

                if self.grid[i][j] == 0 and random.randint(0, 99) < 10:
                    # roll again to see if conditions worsen or get better
                    if random.randint(0, 99) > 50:
                        self.holder = min(4, self.holder + 1)
                    else:
                        self.holder = max(2, self.holder - 1)


    # update reward to be based on distance and penalties
    def updateReward(self):
        currentDistance = self.getCurrDistance()

        self.currentReward = self.startDistance - currentDistance

    # gets the distance of agent to the goal
    def getCurrDistance(self):
        distance = abs(self.goalPosition[0] - self.position[0]) + abs(self.goalPosition[1] - self.position[1])
        return distance

    # once a valid action has been selected, we can now move the agent
    # return map to original form, place agent in new position
    # adjust time based on speed of new tile
    def move(self, pos):
        self.grid[self.position[0]][self.position[1]] = self.holder
        self.position = [pos[0], pos[1]]
        self.holder = self.grid[self.position[0]][self.position[1]]
        self.grid[self.position[0]][self.position[1]] = 0

        # if holder is 1, congrats you reached the goal!
        # end the episode
        if self.holder == 1:
            # give a bonus if applicable
            finalBonus = max(10, self.bonus - self.time)
            self.goalReached = True
            return self.currentReward + finalBonus

        # otherwise increment time and give reward
        self.time += (self.holder - 1) * (self.holder - 1)

        self.updateReward()
        # change the grid for each time change
        for i in range(0, self.holder - 1):
            self.changeGrid()
        return self.currentReward

    def makeAction(self, action):
        # 0 = no action, 1=up, 2=left, 3=down, 4=right
        # if we make no action, just increase the time
        if action == 0:
            # print("no action")
            self.time += 3
            self.changeGrid()
            return self.currentReward

        # go up
        elif action == 1:
            # print("up")
            # no out of bounds here
            if self.position[0] == 0:
                self.time += 3
                self.changeGrid()
                return self.currentReward
            # also no going to invalid spots
            elif self.grid[self.position[0] - 1][self.position[1]] == -1:
                self.time += 3
                self.changeGrid()
                return self.currentReward
            # allow movement
            else:
                return self.move([self.position[0] - 1, self.position[1]])

        # go left
        elif action == 2:
            # print("left")
            # no out of bounds
            if self.position[1] == 0:
                self.time += 3
                self.changeGrid()
                return self.currentReward
            # also no going to invalid spots
            elif self.grid[self.position[0]][self.position[1] - 1] == -1:
                self.time += 3
                self.changeGrid()
                return self.currentReward
            else:
                return self.move([self.position[0], self.position[1]-1])


        # go down
        elif action == 3:
            # print("down")
            # no out of bounds
            if self.position[0] == self.size-1:
                self.time += 3
                self.changeGrid()
                return self.currentReward
            # also no going to invalid spots
            elif self.grid[self.position[0] + 1][self.position[1]] == -1:
                self.time += 3
                self.changeGrid()
                return self.currentReward
            else:
                return self.move([self.position[0] + 1, self.position[1]])

        # go right
        else:
            # print("right")
            # no out of bounds
            if self.position[1] == self.size-1:
                self.time += 3
                self.changeGrid()
                return self.currentReward
            # also no going to invalid spots
            elif self.grid[self.position[0]][self.position[1] + 1] == -1:
                self.time += 3
                self.changeGrid()
                return self.currentReward
            else:
                return self.move([self.position[0], self.position[1]+1])

    # make sure grid is working correctly
    def testing(self):
        self.reset()
        while not self.goalReached:
            self.printGrid()
            print("Current Reward: ", self.currentReward)
            print("Time Passed: ", self.time)
            print("Possible Bonus: ", self.bonus)
            # value = int(input("Insert number for movement on map: "))
            value = random.randint(0, 4)
            result = self.makeAction(value)
        print("Map Complete with final reward (With bonus): ", result)

#Map = Map(mapName="BackAndForth")
#Map.testing()

