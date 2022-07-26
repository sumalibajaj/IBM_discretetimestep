# First letter of a Class name and function (method) name is Capitalised
# All letters of a class attribute are lower case

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
        self._hh = hh
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
        if (self._infector_ID == "InitialInfection") or (self._infector_ID != "not_I"):
            raise RuntimeError("Individual either classified as Infected initially or already infected")
        self._infector_ID = new_infector_ID
        
    def add_infector_hh(self, new_infector_hh): # if this method is used update the person infecting this Susceptible person
        self._infector_hh = new_infector_hh 
        
    def add_infector_type_of_hh(self, new_infector_type_of_hh): # if this method is used update the person infecting this Susceptible person
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



import random
import numpy as np
# Function to create households for one iteration (or one run of the simulation)
# DONT ALLOW SIZE 0 HH
def create_hh(n_hh, type_of_hh_array, prob_type_of_hh_array, mean_hh_size_array, initial_prob_I_array):
    list_hh = [] # Create an empty list of all possible households
    list_ind = [] # Create an empty list of all individuals
    id_tick = -1 # so that first ID is 0
    
    type_of_hh_count = len(type_of_hh_array)
    list_ind_by_type_of_hh = [[] for i in range(type_of_hh_count)]
    for i in range(n_hh):
        hh = HH()
        r_type_of_hh = np.random.choice(type_of_hh_array,  p = prob_type_of_hh_array) # to get type of hh for this realisation
        mean_hh_size_temp = mean_hh_size_array[r_type_of_hh] # this gives mean hh size corresponding to type of hh
        n_hh_size_temp = np.random.poisson(mean_hh_size_temp,1) # this gives a random number from Poisson with given mean
        # now create individuals in this hh 
        for j in range(n_hh_size_temp[0]):
            id_tick = id_tick+1
            r_inf = random.random() # to assign initial infectious state
            if r_inf <= initial_prob_I_array[r_type_of_hh]: # because each type of hh has initial no. of infecteds
                ind = Individual("I", hh, id_tick, r_type_of_hh, n_hh_size_temp, 0,"initial_I", "initial_I","initial_I")
            else:
                ind = Individual("S", hh, id_tick, r_type_of_hh, n_hh_size_temp, "not_I","not_I", "not_I","not_I") 
            list_ind.append(ind)
            list_ind_by_type_of_hh[r_type_of_hh].append(ind)
        list_hh.append(hh)      
    return(list_hh, list_ind, list_ind_by_type_of_hh)
    
    
    
# Creating undirected graph structures for capturing who can contact whom
class Graph:
    def __init__(self, list_of_ind):
        self.list_of_ind = list_of_ind

        self.m_adj_list = {ind: set() for ind in self.list_of_ind}   # here dict keys are individuals  

    def add_edge(self, ind1, ind2, category=0):
        self.m_adj_list[ind1].add((ind2, category))
        self.m_adj_list[ind2].add((ind1, category))

    def print_adj_list(self):
        for key in self.m_adj_list.keys():
            print("node", key.get_ID(), ": ", self.m_adj_list[key])    
            
    def return_adj_list(self):
        return self.m_adj_list    
    
    

    
    
    
import itertools

# Function to create initial adjacency list 
def create_adjacency_list(list_hh_ind_input, n_hh, n_ind, type_of_hh_array, mean_n_contacts_within_area, mean_n_contacts_outside_area):
    # Initialising the adjacency list
    graph = Graph(list_hh_ind_input[1])

    # For connecting people in the same hh
    count_within_hh_contacts = 0
    for i in range(n_hh):
        hh_inds_temp = list_hh_ind_input[0][i].get_individuals()
        hh_size_temp = len(hh_inds_temp)
        list_contacts_temp = hh_inds_temp
        # now making combinations of 2s from the list of contacts and adding them as edges
        for contacts in itertools.combinations(list_contacts_temp, 2):
            graph.add_edge(contacts[0], contacts[1], "same_hh") 



    # For connecting people from different hh (type of hh also mentioned as area)
    for i in range(n_ind):
        ind1_temp = list_hh_ind_input[1][i] # choosing ind1
        ind1_type_of_hh_temp = ind1_temp.get_type_of_hh() # checking ind1's area
        list_contacts_non_hh_temp = [] # empty list of all contacts ind1 can have

        for area_list_index in range(len(type_of_hh_array)): # going through list of all areas     
            # when looking at same area which have individuals
            # append contacts within same area to the list
            if (area_list_index == ind1_type_of_hh_temp) & (len(list_hh_ind_input[2][area_list_index])>0): 
                n_contacts_within_area = np.random.poisson(mean_n_contacts_within_area[area_list_index],1)
                k=0
                while k <(int(n_contacts_within_area)): # will pick how many contacts can ind1 have within their area
                    ind2_temp = random.sample(list_hh_ind_input[2][area_list_index], 1)[0] # picking random ind2
                    if ind2_temp.get_hh() != ind1_temp.get_hh():
                        list_contacts_non_hh_temp.append(ind2_temp)
                        k += 1     

            # when looking at other area which have individuals
            # append contacts outside area to the list
            elif (area_list_index != ind1_type_of_hh_temp) & (len(list_hh_ind_input[2][area_list_index])>0): 
                n_contacts_outside_area = np.random.poisson(mean_n_contacts_outside_area[area_list_index],1)
                for l in range(int(n_contacts_outside_area)): # will pick how many contacts can ind1 have in this area
                    ind2_temp = random.sample(list_hh_ind_input[2][area_list_index], 1)[0] # picking random ind2
                    list_contacts_non_hh_temp.append(ind2_temp)

        # now making combinations of 2s from the list of contacts for ind1 and adding them as edges 
        for ind2_index in list_contacts_non_hh_temp:
            if ind1_temp.get_type_of_hh() == ind2_index.get_type_of_hh():
                graph.add_edge(ind1_temp, ind2_index, "same_area")  
            else:
                graph.add_edge(ind1_temp, ind2_index, "outside_area")              
 
    return(graph)    
    
        
    
    
import scipy.stats as stats 
    
# calculating probability of infecting someone upon contact given the no. of days since infection.
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7201952/
# https://www.science.org/doi/10.1126/science.abb6936

def cal_prob_of_inf(p_max, mean_gamma, sd_gamma, t_inf_max):
    shape_gamma = (mean_gamma/sd_gamma)**2
    scale_gamma = sd_gamma**2/mean_gamma

    mode_gamma = (shape_gamma-1)*scale_gamma
  
    t = np.linspace (0, t_inf_max, t_inf_max+1) 

    #calculate pdf of Gamma distribution for each t-value
    pdf_t = stats.gamma.pdf(t, a = shape_gamma, scale=scale_gamma)

    # pdf of mode
    pdf_mode = stats.gamma.pdf(mode_gamma, a = shape_gamma, scale=scale_gamma)
    
    # scaled probability of infection for each t
    p_t = p_max * (pdf_t/pdf_mode)
    
    return(p_t)

    
    
    
    
    
    
    
    
    
    
    
    