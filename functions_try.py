# Class of home (so people in a given house are always part of the same house)
class HH:
    def __init__(self):
        self._hh_individuals = [] # initialising an empty list of individuals that can belong to a Household object (hence no input value required)

    def add_individual(self, individual):
        self._hh_individuals.append(individual) # keep appending "individual" passed to the function AddIndividual to the empty list created above

    def count_hh_individuals(self):
        return len(self._hh_individuals) # Getter function to get # individuals in the household

    def count_hh_susceptible(self):
        n_susceptible = 0 # start the count with 0 susceptible individuals in the household
        for i in self._hh_individuals: # iterating through all individuals (objects) in the list
            n_susceptible += i.state() == "S" # checking if the attribute state for the individual object is equal to S
        return n_susceptible

    def count_hh_infected(self):
        n_infected = 0 # start the count with 0 infected individuals in the household
        for i in self._hh_individuals: # iterating through all individuals (objects) in the list
            n_infected += i.state() == "I" # checking if the attribute state for the individual object is equal to I
        return n_infected

    def count_hh_recovered(self):
        n_recovered = 0 # start the count with 0 recovered individuals in the household
        for i in self._hh_individuals: # iterating through all individuals (objects) in the list
            n_recovered += i.state() == "R" # checking if the attribute state for the individual object is equal to R
        return n_recovered

    def get_individuals(self):
        return self._hh_individuals


# Class of individual (so that properties of an individual stay connected)
# Make methods to perform any changes that can happen to an individual
class Individual:
    def __init__(self, state, hh, ID, type_of_hh, hh_size, time_of_infection, infector_ID, infector_hh, infector_type_of_hh): # intialise an individual with these details
        self._state = state
        self._household = hh
        self._ID = ID
        self._type_of_hh = type_of_hh
        self._hh_size = hh_size
        self._time_of_infection = time_of_infection
        self._infector_ID = infector_ID
        self._infector_hh = infector_hh
        self._infector_type_of_hh = infector_type_of_hh
        hh.add_individual(self) # Now add this individual to houshold it belongs to by using the method from Household class

    def StoI(self):
        if self._state == "I":
            raise RuntimeError("Individual is already infected")
        self._state = "I" # if this method is used, change the state of the individual from S to I

    def ItoR(self):
        if self._state == "R":
            raise RuntimeError("Individual is already recovered")
        self._state = "R" # if this method is used, change the state of the individual from I to R

    def add_time_of_infection(self, new_time_of_infection):
        self._time_of_infection = new_time_of_infection

    def add_infector_ID(self, new_infector_ID): # if this method is used update the person infecting this Susceptible person
        if self._state != "S":
            raise RuntimeError("Individual cannot be infected if the current state is not Susceptible")
        if (self._infector_ID == "InitialInfection") or (self._infector_ID != "NotInfected"):
            raise RuntimeError("Individual either classified as Infected initially or already infected")
        self._infector_ID = new_infector_ID
        
    def add_infector_hh(self, new_infector_hh): # if this method is used update the person infecting this Susceptible person
        if self._state != "S":
            raise RuntimeError("Individual cannot be infected if the current state is not Susceptible")
        if (self._infector_ID == "InitialInfection") or (self._infector_ID != "NotInfected"):
            raise RuntimeError("Individual either classified as Infected initially or already infected")
        self._infector_hh = new_infector_hh 
        
    def add_infector_tyoe_of_hh(self, new_infector_type_of_hh): # if this method is used update the person infecting this Susceptible person
        if self._state != "S":
            raise RuntimeError("Individual cannot be infected if the current state is not Susceptible")
        if (self._infector_ID == "InitialInfection") or (self._infector_ID != "NotInfected"):
            raise RuntimeError("Individual either classified as Infected initially or already infected")
        self._infector_type_of_hh = new_infector_type_of_hh         

    def get_state(self):
        return self._state # Make state a public attribute, so it can called outside

    def get_hh(self):
        return self._hh # Make state a public attribute, so it can called outside

    def get_ID(self):
        return self._ID # Make state a public attribute, so it can called outside

    def get_type_of_hh(self):
        return self._type_of_hh # Make state a public attribute, so it can called outside
    
    def get_hh_size(self):
        return self._hh_size # Make state a public attribute, so it can called outside    
    
    def get_time_of_infection(self):
        return self._time_of_infection # Make state a public attribute, so it can called outside
    
    def get_infector_ID(self):
        return self._infector_ID # Make state a public attribute, so it can called outside
    
    def get_infector_hh(self):
        return self._infector_hh # Make state a public attribute, so it can called outside

    def get_infector_type_of_hh(self):
        return self._infector_type_of_hh # Make state a public attribute, so it can called outside




# Creating undirected graph structures for capturing who can contact whom
class Graph:
    def __init__(self, num_of_nodes, directed=False):
        self.m_num_of_nodes = num_of_nodes
        self.m_nodes = range(self.m_num_of_nodes)

        self.m_directed = directed

        self.m_adj_list = {node: set() for node in self.m_nodes}      

    def add_edge(self, node1, node2, weight=1):
        self.m_adj_list[node1].add((node2, weight))
        
        if not self.m_directed:
            self.m_adj_list[node2].add((node1, weight))

    def print_adj_list(self):
        for key in self.m_adj_list.keys():
            print("node", key, ": ", self.m_adj_list[key])    
    
        
    
    
    

from random import seed
from random import random 
import numpy as np
# Function to create households for one iteration (or one run of the simulation)
# DONT ALLOW SIZE 0 HH
def create_hh(n_hh, type_of_hh_array, prob_type_of_hh_array, mean_hh_size_array, initial_prob_I_array):
    list_hh = [] # Create an empty list of all possible households
    list_ind = [] # Create an empty list of all individuals
    id_tick = -1 # so that first ID is 0
    for i in range(n_hh):
        hh = HH()
        r_type_of_hh = np.random.choice(type_of_hh_array,  p = prob_type_of_hh_array) # to get type of hh for this realisation
        mean_hh_size_temp = mean_hh_size_array[r_type_of_hh] # this gives mean hh size corresponding to type of hh
        n_hh_size_temp = np.random.poisson(mean_hh_size_temp,1) # this gives a random number from Poisson with given mean
        # now create individuals in this hh 
        for j in range(n_hh_size_temp[0]):
            id_tick = id_tick+1
            r_inf = random() # to assign initial infectious state
            if r_inf <= initial_prob_I_array[r_type_of_hh]: # because each type of hh has initial no. of infecteds
                ind = Individual("I", hh, id_tick, r_type_of_hh, n_hh_size_temp, "initial_I","initial_I", "initial_I","initial_I")
            else:
                ind = Individual("S", hh, id_tick, r_type_of_hh, n_hh_size_temp, "not_I","not_I", "not_I","not_I") 
            list_ind.append(ind)
        list_hh.append(hh)      
    return(list_hh, list_ind)
    

    
    
    
    