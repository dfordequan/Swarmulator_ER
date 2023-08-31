"""
Optimize the swarm foraging task using the Artificial Bee Colony algorithm.
@author: Dequan Ou, 2023

@article{karaboga2005idea,
    title={An idea based on honey bee swarm for numerical optimization},
    author={Karaboga, Dervis and Basturk, Bahriye},
    journal={Technical report-tr06},
    volume={2006},
    number={01},
    year={2005}
}
"""
import random
import numpy as np
from deap import base, creator, tools
import pickle
import matplotlib.pyplot as plt

class evolution_abc:
    '''Wrapper around the DEAP package to run the ABC algorithm'''

    def __init__(self):
        '''Initialize the DEAP wrapper'''
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        self.g = 0

    def setup(self, fitness_function_handle, constraint=None, GENOME_LENGTH=20, POPULATION_SIZE=100, LIMIT=100):
        '''Set up the parameters'''
        self.GENOME_LENGTH = GENOME_LENGTH
        self.POPULATION_SIZE = POPULATION_SIZE
        self.LIMIT = LIMIT

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", random.random)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, self.GENOME_LENGTH)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", fitness_function_handle)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        if constraint is not None:
            self.toolbox.decorate("evaluate", tools.DeltaPenalty(constraint, 7))
        self.stats = []

    def mutate_bee(self, bee):
        '''ABC-specific mutation for a bee'''
        k = random.randint(0, len(bee)-1)
        phi = random.uniform(-1, 1)
        bee[k] += phi * (bee[k] - random.choice(self.pop)[k])
        return bee,

    def evolve(self, generations=100, verbose=False):
        '''Run the ABC algorithm'''
        random.seed()
        self.pop = self.toolbox.population(n=self.POPULATION_SIZE)
        if verbose: print('{:=^40}'.format(' Start of ABC '))

        # Evaluate the initial population
        fitnesses = list(map(self.toolbox.evaluate, self.pop))
        for bee, fit in zip(self.pop, fitnesses):
            bee.fitness.values = fit

        trials = [0] * len(self.pop)

        for gen in range(generations):
            self.g = gen
            # Employed Bee Phase
            for i, bee in enumerate(self.pop):
                new_bee = self.toolbox.clone(bee)
                self.mutate_bee(new_bee)
                new_bee.fitness.values = self.toolbox.evaluate(new_bee)
                if new_bee.fitness > bee.fitness:
                    self.pop[i] = new_bee
                    trials[i] = 0
                else:
                    trials[i] += 1

            # Onlooker Bee Phase
            bee_probs = [bee.fitness.values[0] for bee in self.pop]
            bee_probs = [prob/sum(bee_probs) for prob in bee_probs]


            selected_indices = np.random.choice(np.arange(len(self.pop)), size=len(self.pop), p=bee_probs)
            selected_bees = [self.pop[i] for i in selected_indices]

            for i, bee in enumerate(selected_bees):
                new_bee = self.toolbox.clone(bee)
                self.mutate_bee(new_bee)
                new_bee.fitness.values = self.toolbox.evaluate(new_bee)
                if new_bee.fitness > bee.fitness:
                    selected_bees[i] = new_bee
                    trials[i] = 0
                else:
                    trials[i] += 1

            # Scout Bee Phase
            for i, trial in enumerate(trials):
                if trial >= self.LIMIT:
                    self.pop[i] = self.toolbox.individual()
                    self.pop[i].fitness.values = self.toolbox.evaluate(self.pop[i])
                    trials[i] = 0

            self.stats.append(self.store_stats(self.pop, gen))
            if verbose: self.disp_stats(gen)

        self.best_ind = self.get_best()

        if verbose:
            print('{:=^40}'.format(' End of ABC '))
            print("Best individual is %s, %s" % (self.best_ind, self.best_ind.fitness.values))

        return self.pop
    
    def get_best(self):
        '''Returns the best individual'''
        return tools.selBest(self.pop, 1)[0]
    
    def save(self, filename, pop=None, gen=None, stats=None):
        '''Save the evolution to a file'''
        if pop is None: pop = self.pop
        if gen is None: gen = self.g
        if stats is None: stats = self.stats
        pickle.dump({'pop':pop,'gen':gen,'stats':stats}, open(filename, "wb"))

    def load(self, filename):
        '''Load the evolution from a file'''
        data = pickle.load(open(filename, "rb"))
        self.pop = data['pop']
        self.g = data['gen']
        self.stats = data['stats']
        return data
    
    def store_stats(self, population, iteration=0):
        '''Store the current stats and return a dict'''
        fitnesses = [individual.fitness.values[0] for individual in population]
        return {
            'g': iteration,
            'mu': np.mean(fitnesses),
            'std': np.std(fitnesses),
            'max': np.max(fitnesses),
            'min': np.min(fitnesses)
        }

    def disp_stats(self, iteration=0):
        '''Print the current stats'''
        print(">> gen = %i, mu = %.2f, std = %.2f, max = %.2f, min = %.2f" % 
        (self.stats[iteration]['g'],
        self.stats[iteration]['mu'],
        self.stats[iteration]['std'],
        self.stats[iteration]['max'],
        self.stats[iteration]['min']))

    def plot_evolution(self, figurename=None):
        '''Plot the evolution of the fitness'''
        plt.figure()
        plt.plot([s['mu'] for s in self.stats], label="Average fitness")
        plt.plot([s['max'] for s in self.stats], label="Maximum fitness")
        plt.plot([s['min'] for s in self.stats], label="Minimum fitness")
        plt.xlabel('Generations')
        plt.ylabel('Fitness')
        plt.legend()
        if figurename:
            plt.savefig(figurename)
        else:
            plt.show()

