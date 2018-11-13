import operator
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from city import City
from fitness import Fitness


def createRoute(cityList):  #ftaxnei ena route/individual px athina - thessaloniki - larisa - athina
    route = random.sample(cityList, len(cityList))
    return route

def initialPopulation(popSize, cityList): # ftiaxnei ena population --> me tosa routes mesa oso to popSize aka population size
    population = [] #auti i lista einai ENA population . kathe stoixeio tis listas einai ena route
    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population


def rankRoutes(population): #kanoume rank ta routes/inidividuals TOU population
    fitnessResults = {} #edo mesa exei ta routes taksinomimena
    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

def selection(popRanked, eliteSize):  #dinoume to  ouput apo to rankRoutes kai posa tha paroun golden buzz (elitesize)
    selectionResults = [] #poia routes tha perasoun stin epomeni fasi ta IDs

    #create the weighted distribution
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(0, eliteSize): #edo pernao ta prota elitesize
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize): #gia kathe route pou exei meinei...
        pick = 100 * random.random() #epilegoume ena random arithmo % ...
        for i in range(0, len(popRanked)):  #gia kathe taksinomimeno route pali...
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults

def matingPool(population, selectionResults): #kseskartarisma, epistrefei to poulation mono me ta selected routes, tora pia legete mono matingpool
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1)) #random cities
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child


def breedPopulation(matingpool, eliteSize): #dimiourgoume ena population pou exei ftiaxtei apo ola ta komena kai rammena routes
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize): #sta paidia einai pali sigoura ta elites, xoris kopsimo kai rapsimo
        children.append(matingpool[i])

    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool) - i - 1]) #anaparagogi me tin texniki kospimatos kai rapsimatos
        children.append(child)
    return children

def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual


def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration


def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))

    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)

    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    return bestRoute

#######################################################################

def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    progress = []
    progress.append(1 / rankRoutes(pop)[0][1])

    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        progress.append(1 / rankRoutes(pop)[0][1])

    plt.interactive(False)
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    print 'ok'
    plt.savefig('wow.png')
    plt.show(block=True)

cityList = []

for i in range(0,25):
    cityList.append(City(x=int(random.random() * 200), y=int(random.random() * 200)))

geneticAlgorithmPlot(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=700)
