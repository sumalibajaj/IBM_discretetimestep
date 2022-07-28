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
    def __init__(self, state, hh, ID, type_of_hh, hh_size, time_of_infection, infector_ID, infector_hh, infector_type_of_hh,\
                n_within_area_contacts = 0, n_outside_area_contacts = 0): # intialise an individual with these details
        self._state = state
        self._hh = hh
        self._ID = ID
        self._type_of_hh = type_of_hh
        self._hh_size = hh_size
        self._time_of_infection = time_of_infection
        self._infector_ID = infector_ID
        self._infector_hh = infector_hh
        self._infector_type_of_hh = infector_type_of_hh
        self._n_within_area_contacts = n_within_area_contacts
        self._n_outside_area_contacts = n_outside_area_contacts
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
        
    def add_n_within_area_contacts(self, new_n_within_area_contacts):
        self._n_within_area_contacts = new_n_within_area_contacts
        
    def add_n_outside_area_contacts(self, new_n_outside_area_contacts):
        self._n_outside_area_contacts = new_n_outside_area_contacts        

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
    
    def get_n_within_area_contacts(self):
        return self._n_within_area_contacts
    
    def get_n_outside_area_contacts(self):
        return self._n_outside_area_contacts



import random
import numpy as np
# Function to create households for one iteration (or one run of the simulation)
# DONT ALLOW SIZE 0 HH
def create_hh(n_hh_input, type_of_hh_array_input, prob_type_of_hh_array_input, \
              mean_hh_size_array_input, initial_prob_I_array_input):
    list_hh_temp = [] # Create an empty list of all possible households
    list_ind_temp = [] # Create an empty list of all individuals
    id_tick = -1 # so that first ID is 0
    
    type_of_hh_count = len(type_of_hh_array_input)
    list_ind_by_type_of_hh_temp = [[] for i in range(type_of_hh_count)]
    for i in range(n_hh_input):
        hh = HH()
        r_type_of_hh = np.random.choice(type_of_hh_array_input,  p = prob_type_of_hh_array_input) # to get type of hh for this realisation
        mean_hh_size_temp = mean_hh_size_array_input[r_type_of_hh] # this gives mean hh size corresponding to type of hh
#         n_hh_size_temp = np.random.poisson(mean_hh_size_temp,1) # this gives a random number from Poisson with given mean
#         # now create individuals in this hh 
        n_hh_size_temp = int(mean_hh_size_temp) # this gives a random number from Poisson with given mean
        # now create individuals in this hh         
        for j in range(n_hh_size_temp):
            id_tick = id_tick+1
            r_inf = random.random() # to assign initial infectious state
            if r_inf <= initial_prob_I_array_input[r_type_of_hh]: # because each type of hh has initial no. of infecteds
                ind = Individual("I", hh, id_tick, r_type_of_hh, n_hh_size_temp, 0,"initial_I", "initial_I","initial_I")
            else:
                ind = Individual("S", hh, id_tick, r_type_of_hh, n_hh_size_temp, "not_I","not_I", "not_I","not_I") 
            list_ind_temp.append(ind)
            list_ind_by_type_of_hh_temp[r_type_of_hh].append(ind)
        list_hh_temp.append(hh)      
    return(list_hh_temp, list_ind_temp, list_ind_by_type_of_hh_temp)
    
    
    
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
# def create_adjacency_list(list_hh_ind_input, n_hh_input, n_ind_input, type_of_hh_array_input, mean_n_contacts_within_area_input, mean_n_contacts_outside_area_input):
    
# #     # Check conditions:
# #     for i range(len(type_of_hh_array_input)):
# #         mean_n_contacts_outside_area_input[i]*len(list_hh_ind_input[2][i]) >= 
    
#     # Initialising the adjacency list
#     graph_temp = Graph(list_hh_ind_input[1])

#     # For connecting people in the same hh
#     count_within_hh_contacts_input = 0
#     for i in range(n_hh_input):
#         hh_inds_temp = list_hh_ind_input[0][i].get_individuals()
#         hh_size_temp = len(hh_inds_temp)
#         list_contacts_temp = hh_inds_temp
#         # now making combinations of 2s from the list of contacts and adding them as edges
#         for contacts in itertools.combinations(list_contacts_temp, 2):
#             graph_temp.add_edge(contacts[0], contacts[1], "same_hh") 

#     # For connecting people from different hh (type of hh also mentioned as area)
#     # starting with list of individuals in area 0
#     for area_list_index in range(len(list_hh_ind_input[2])):
#         area_list_inds = list_hh_ind_input[2][area_list_index]
#         # looping through individuals in a given area
#         for ind1_temp in area_list_inds:
#             n_contacts_within_area = mean_n_contacts_within_area_input[area_list_index]
#             k=ind1_temp
#             while k <(int(n_contacts_within_area)): # will pick how many contacts can ind1 have within their area
#                 ind2_temp = random.sample(area_list_inds, 1)[0] # picking random ind2
#                 if ind2_temp.get_hh() != ind1_temp.get_hh():
#                     list_contacts_non_hh_temp.append(ind2_temp)
#                     graph_temp.add_edge(ind1_temp, ind2_temp, "same_area") 
#                     k += 1  
            

#     # For connecting people from different hh (type of hh also mentioned as area)
#     for i in range(n_ind_input):
#         ind1_temp = list_hh_ind_input[1][i] # choosing ind1
#         ind1_type_of_hh_temp = ind1_temp.get_type_of_hh() # checking ind1's area
#         list_contacts_non_hh_temp = [] # empty list of all contacts ind1 can have

#         for area_list_index in range(len(type_of_hh_array_input)): # going through list of all areas     
#             # when looking at same area which have individuals
#             # append contacts within same area to the list
#             if (area_list_index == ind1_type_of_hh_temp) & (len(list_hh_ind_input[2][area_list_index])>0): 
# #                 n_contacts_within_area = np.random.poisson(mean_n_contacts_within_area_input[area_list_index],1)
#                 n_contacts_within_area = mean_n_contacts_within_area_input[area_list_index]
#                 k=0
#                 while k <(int(n_contacts_within_area)): # will pick how many contacts can ind1 have within their area
#                     ind2_temp = random.sample(list_hh_ind_input[2][area_list_index], 1)[0] # picking random ind2
#                     if ind2_temp.get_hh() != ind1_temp.get_hh():
#                         list_contacts_non_hh_temp.append(ind2_temp)
#                         k += 1     

#             # when looking at other area which have individuals
#             # append contacts outside area to the list
#             elif (area_list_index != ind1_type_of_hh_temp) & (len(list_hh_ind_input[2][area_list_index])>0): 
# #                 n_contacts_outside_area = np.random.poisson(mean_n_contacts_outside_area_input[area_list_index],1)
#                 n_contacts_outside_area = mean_n_contacts_outside_area_input[area_list_index]
#                 for l in range(int(n_contacts_outside_area)): # will pick how many contacts can ind1 have in this area
#                     ind2_temp = random.sample(list_hh_ind_input[2][area_list_index], 1)[0] # picking random ind2
#                     list_contacts_non_hh_temp.append(ind2_temp)

#         # now making combinations of 2s from the list of contacts for ind1 and adding them as edges 
#         for ind2_index in list_contacts_non_hh_temp:
#             if ind1_temp.get_type_of_hh() == ind2_index.get_type_of_hh():
#                 graph_temp.add_edge(ind1_temp, ind2_index, "same_area")  
#             else:
#                 graph_temp.add_edge(ind1_temp, ind2_index, "outside_area")              
 
#     return(graph_temp)    
    
        
    
    
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

    

    
# running simulations    


def sim(n_iter_input, n_hh_input, type_of_hh_array_input, prob_type_of_hh_array_input, mean_hh_size_array_input, \
        initial_prob_I_array_input, mean_n_contacts_within_area_input, mean_n_contacts_outside_area_input, \
        t_max_input, max_recovery_t_input, p_t_input):    

    # open a csv file to write outputs
    f = open('test.csv', 'w')
    f.write("infector_ID,infector_type_of_hh,infectee_ID,infectee_type_of_hh,time,iter \n")

    g = open('sim.csv', 'w')
    g.write("S,I,R,time,iter \n")     
    
    # start iterations
    iter_count = 1
    
    # running n_iter_input number of simulations (i.e creating n_iter_input no. of hypothetical populations)
    while iter_count <= n_iter_input:
        
        # creating apopulation and assigning households and attributes
        list_hh_ind_input = create_hh(n_hh_input, type_of_hh_array_input, prob_type_of_hh_array_input, \
                                      mean_hh_size_array_input, initial_prob_I_array_input)
        
        # calculating total number of individuals created
        n_ind_input = len(list_hh_ind_input[1])

        # creating the initial adjacency list for this population
        graph_input = create_adjacency_list(list_hh_ind_input, n_hh_input, n_ind_input, type_of_hh_array_input, \
                                            mean_n_contacts_within_area_input, mean_n_contacts_outside_area_input)

        # starting the clock at time = 0
        t = 0
        
        # calculating number of infected, susceptible and recovered individuals at time = 0 and writing to SIR output file
        I_temp = len([i for i in list_hh_ind_input[1] if i.get_state() == "I"])
        S_temp = n_ind_input - I_temp
        R_temp = 0    
        g.write(str(S_temp)+","+str(I_temp)+","+str(R_temp)+","+str(t)+","+str(iter_count)+"\n")

        # running the simulation for this population till maximum time specified (t_max_input)
        while t <= t_max_input:
            inds = list(graph_input.m_adj_list.keys()) # list of all individuals in the population
            infs = [ind for ind in inds if ind.get_state() == "I"] # list of all infected individuals in the population
            for inf in infs:
                time_since_infection = t - inf.get_time_of_infection()
                if time_since_infection >= max_recovery_t_input:
                    inf.ItoR()
                    I_temp -= 1
                    R_temp += 1
                else:
                    connections = graph_input.m_adj_list[inf] # set of all connections for a given infected individual
                    sus = [con for con in connections if con[0].get_state() == "S"]
                    n_to_infect = int(np.random.binomial(len(sus), p_t_input[time_since_infection], 1)) # this depends on how long inf individual has been infectious
                    sus_selected = random.sample(sus, n_to_infect) # selecting susceptibles who will get infected
                    for sus_sel in sus_selected:
                        # Do the following actions now that is person is getting infected
                        sus_sel[0].add_time_of_infection(t)
                        sus_sel[0].add_infector_ID(inf.get_ID())            
                        sus_sel[0].add_infector_hh(inf.get_hh())          
                        sus_sel[0].add_infector_type_of_hh(inf.get_type_of_hh())           
                        sus_sel[0].StoI()           

                        # temporarily saving details of the infection that occured
                        infector_ID_temp = str(inf.get_ID())
                        infector_type_of_hh_temp = str(inf.get_type_of_hh())            
                        infectee_ID_temp = str(sus_sel[0].get_ID())
                        infectee_type_of_hh_temp = str(sus_sel[0].get_type_of_hh())            
                        time_of_infection_temp = str(t)
                        
                        # writing details about who got infected by whom and type of hhs to the output file
                        f.write(infector_ID_temp+","+infector_type_of_hh_temp+","+infectee_ID_temp+","\
                                +infectee_type_of_hh_temp+","+time_of_infection_temp+","+str(iter_count)+"\n")

                        # update S, I, R counts
                        S_temp -= 1
                        I_temp += 1

            # increase time by one day
            t += 1
            
            # writing to SIR output file
            g.write(str(S_temp)+","+str(I_temp)+","+str(R_temp)+","+str(t)+","+str(iter_count)+"\n")
        
        print("Simulation ", iter_count, " is complete.")
        # the simulation for this population is complete, so increase iter_count by 1
        iter_count += 1
        
    f.close()
    g.close()
        
    
    
    
    
    
    