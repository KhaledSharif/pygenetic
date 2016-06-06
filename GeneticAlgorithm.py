import numpy as np
from numba import jit

class Human:
    attributes = None

    def __init__(self):
        self.attributes = None
    @jit
    def create_uniform(self, minimum, maximum, size):
        self.attributes = np.random.uniform(minimum, maximum, size)
    @jit
    def create_from_other(self, attributes: np.array):
        self.attributes = attributes

    def mutate(self, std, probability, min, max):
        change = np.array([(1 if x < probability else 0) for x in (np.random.rand(len(self.attributes), 1))])
        self.attributes += np.random.normal(0, std, len(self.attributes)) * change
        for i in range(len(self.attributes)):
            if self.attributes[i] > max: self.attributes[i] = max
            elif self.attributes[i] < min: self.attributes[i] = min
    @jit
    def mate(self, other):
        baby = Human()
        baby.create_from_other(0.5 * self.attributes + 0.5 * other.get_attributes())
        return baby

    def get_attributes(self):
        return self.attributes


class Population:
    humans = None
    att_min, att_max = None, None

    def __init__(self):
        self.humans = None

    def get_population(self):
        return self.humans
    @jit
    def create_population(self, attributes_min, attributes_max, attributes_size, population_size):
        self.humans = []
        self.att_min, self.att_max = attributes_min, attributes_max
        for i in range(population_size):
            self.humans.append(Human())
            self.humans[-1].create_uniform(attributes_min, attributes_max, attributes_size)

    def mutate_population(self, standard_deviation, individual_probability, population_probability):
        for i in range(len(self.humans)):
            if np.random.rand(1, 1) < population_probability:
                self.humans[i].mutate(standard_deviation, individual_probability, self.att_min, self.att_max)

    @jit
    def crossover_population(self):
        length = range(len(self.humans))
        for i in length:
            for j in length:
                if i >= j: continue
                self.humans.append(self.humans[i].mate(self.humans[j]))
    @jit
    def sort_population(self, function, maximize=False):
        self.humans.sort(key=function, reverse=maximize)
    @jit
    def evolve_population(self, number_to_keep):
        self.humans = self.humans[0:number_to_keep]
