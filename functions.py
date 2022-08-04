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
                n_within_area_contacts = 0, n_outside_area_contacts = 0, n_outside_area_temp = 0): # intialise an individual with these details
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
        self._n_outside_area_temp = n_outside_area_temp
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
        
    def add_n_outside_area_temp(self, new_n_outside_area_temp):
        self._n_outside_area_temp = new_n_outside_area_temp        

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
    
    def get_n_outside_area_temp(self):
        return self._n_outside_area_temp



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
        n_hh_size_temp = int(np.random.poisson(mean_hh_size_temp,1)) # this gives a random number from Poisson with given mean
#         # now create individuals in this hh 
#         n_hh_size_temp = int(mean_hh_size_temp) # this gives a random number from Poisson with given mean
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
import networkx as nx
import matplotlib as plt

# Function to create initial adjacency list 
def create_adjacency_list(list_hh_ind_input, n_hh_input, n_ind_input, type_of_hh_array_input, mean_n_contacts_outside_hh_input):

    ############################################################
    # creating connections between members of the same household
    ############################################################
    
    # initialising graph
    graph = nx.Graph()

    for hh in range(n_hh_input):
        inds_temp = list_hh_ind_input[0][hh].get_individuals()
        graph.add_nodes_from(inds_temp)
        graph.add_edges_from(itertools.combinations(inds_temp, 2))


    # Adding attributes about how many contacts will people have within and outside their area
    for ind in list_hh_ind_input[1]:
        area_temp = ind.get_type_of_hh()    
        temp_within_area_contacts = np.random.poisson(mean_n_contacts_outside_hh_input[area_temp][area_temp], 1)[0]
        ind.add_n_within_area_contacts(temp_within_area_contacts)

    ############################################################
    # creating connections between members of the same area  
    ############################################################
    
    for area in range(len(type_of_hh_array_input)):
    # for area in range(1):    
        ind_tick = 0
        list_inds_available = list_hh_ind_input[2][area]
        for ind1 in list_hh_ind_input[2][area]:   
            # choosing ind1 from the list of individuals in a given area
            ind1_degree_hh = ind1.get_hh_size() - 1 # this is fixed number for an individual
            ind1_degree_hh_within_area = ind1_degree_hh + ind1.get_n_within_area_contacts() # this is fixed number for an individual

            # check if any individuals are available to be connected to
            if len(list_inds_available) == 0:
                # forcing ind1's within area contacts = 0 if no individual is left
                ind1.add_n_within_area_contacts(0) # BEST THING TO DO?

            else:
                # don't run the following if the maximum possible connections for ind1 has been reached
                while graph.degree[ind1] < ind1_degree_hh_within_area:    
                    # drawing a potential contact from within the area
                    # if there are available individuals, continue and draw one randomly
                    if len(list_inds_available) != 0:
                        ind2  = random.sample(list_inds_available, 1)[0]           
                        # checking how many contacts this person can have (in hh and within area)
                        ind2_degree_hh = ind2.get_hh_size() - 1
                        ind2_degree_hh_within_area = ind2_degree_hh + ind2.get_n_within_area_contacts()

                        # check if this person is from same hh, or already connected, or has reached maximum possible connections setup                
                        if (ind2.get_hh() == ind1.get_hh()) | (ind2 in graph[ind1]) | (graph.degree[ind2] == ind2_degree_hh_within_area):
                            list_inds_available = list(set(list_inds_available) - set([ind2]))
                        else:
                            graph.add_edge(ind1, ind2)
                    else:
                        # correct within area contacts to how many were possible
                        ind1.add_n_within_area_contacts(graph.degree[ind1] - (ind1.get_hh_size() - 1)) 
    #                     print("breaking from the while loop")
    #                     print("setting within area connections for id", ind1.get_ID(), " to ", ind1.get_n_within_area_contacts())
                        break
                # remove ind1 from available list of someone's potential contact within the area (because all of ind1's contacts are exhausted now)
                list_inds_available = list(set(list_inds_available) - set([ind1])) 

    ############################################################
    # creating connections between members of different areas 
    ############################################################
    
    for i in range(len(type_of_hh_array_input)-1):  

        # finding individuals to connect to in the next areas j
        for j in range(i+1, len(type_of_hh_array_input)):   

            # assigning how many contacts individuals in area i will have with area j
            ig_list1 = [x.add_n_outside_area_temp(np.random.poisson(mean_n_contacts_outside_hh_input[i][j], 1)) for x in list_hh_ind_input[2][i]]
    #         ig_list1 = [x.add_n_outside_area_temp(mean_n_contacts_outside_hh_input[i][j]) for x in list_hh_ind_input[2][i]]

            # create a copy of individuals in area j and later remove individuals who are no longer available to be connected
            list_inds_available = list_hh_ind_input[2][j]

            for ind1 in list_hh_ind_input[2][i]:
    #             print("")
    #             print("finding connections for id", ind1.get_ID(), " of area", i, " in area", j)

                # assign how many contacts will individuals (ind2s) in area j have with area i
                ig_list2 = [x.add_n_outside_area_contacts(np.random.poisson(mean_n_contacts_outside_hh_input[j][i], 1)) for x in list_inds_available]
    #             ig_list2 = [x.add_n_outside_area_contacts(mean_n_contacts_outside_hh_input[j][i]) for x in list_inds_available]

                # calculate how many contacts can ind1 have in area j
    #             ind1_degree_hh = ind1.get_hh_size() - 1 # this is fixed number for an individual
    #             ind1_degree_hh_within_area = ind1_degree_hh + ind1.get_n_within_area_contacts() # this is fixed number for an individual
                ind1_degree_hh_within_area = graph.degree[ind1]
                ind1_degree_area_i_j = ind1.get_n_outside_area_temp()         
                ind1_degree_hh_within_outside_area_temp = ind1_degree_hh_within_area + ind1_degree_area_i_j # this is fixed number for an individual
    #             print("connections needed for id", ind1.get_ID(), "=", ind1_degree_hh_within_outside_area_temp, "and connections acheived=", graph.degree[ind1])

                  # check if any individuals are available to be connected to
                if len(list_inds_available) == 0:

                    # forcing ind1's within area contacts = 0 if no individual is left
                    ind1.add_n_outside_area_contacts(0)
    #                 print("setting 0 contacts in area", j, " for id", ind1.get_ID())

                else:          

                    # don't run the following if the maximum possible connections for ind1 has been reached
                    while graph.degree[ind1] < ind1_degree_hh_within_outside_area_temp:
    #                     print("no. of available ind2s ", len(list_inds_available))

                        # if there are available individuals, continue and draw one randomly
                        if len(list_inds_available) != 0:

                            # drawing a potential contact from outside the area
                            ind2  = random.sample(list_inds_available, 1)[0]
                            # checking how many contacts this person can have (in hh and within area)
                            ind2_degree_hh = ind2.get_hh_size() - 1
                            ind2_degree_hh_within_area = ind2_degree_hh + ind2.get_n_within_area_contacts()
                            ind2_degree_hh_within_outside_area = ind2_degree_hh_within_area + ind2.get_n_outside_area_contacts() # this is fixed number for an individual

                            # check if this person is laready connected to ind1 or has reached maximum possible connections setup                
                            if (ind2 in graph[ind1]) | (graph.degree[ind2] == ind2_degree_hh_within_outside_area):

                                list_inds_available = list(set(list_inds_available) - set([ind2]))
    #                             print("Rejecting this id", ind2.get_ID())

                            else:

                                graph.add_edge(ind1, ind2)
    #                             print("connected id", ind1.get_ID(), " to id", ind2.get_ID(), " who has", graph.degree[ind2], "connections out of ", ind2_degree_hh_within_outside_area)

    #                         print("id", ind1.get_ID(), "now has ", graph.degree[ind1], " contacts") 

                        else:

                            print("breaking from the while loop")
                            break 

    print("adjacency graph created")                
    return(graph)            
            
            
import pandas as pd

def view_input_observed_contact_matrices(graph_input, type_of_hh_array_input, mean_n_contacts_outside_hh_input, mean_hh_size_array_input): 

    obs_contact_within_hh = []
    obs_contact_matrix_area = []

    for ind in list(graph_input.nodes):
        contacts = list(graph_input.adj[ind])

        # first column is the area of the individual
        # creating an empty row for number of contacts in each area for ind
        contact_within_hh_row = [0]*2
        contact_within_hh_row[0] = ind.get_type_of_hh()
        count_within_hh = 0

        contact_area_row = [0] * (len(type_of_hh_array_input) + 1)    
        contact_area_row[0] = ind.get_type_of_hh()

        for contact in contacts:
            # if contact is in the same hh, increase count_within_hh by 1
            if ind.get_hh() == contact.get_hh():
                count_within_hh = count_within_hh + 1

            # otherwise increase the column about the area of the contact
            else:
                area_of_contact = contact.get_type_of_hh()
                # area of the contact is column number + 1 (indexing starts from 0)
                contact_area_row[area_of_contact + 1] = contact_area_row[area_of_contact + 1] + 1

            contact_within_hh_row[1] = count_within_hh

        obs_contact_within_hh.append(contact_within_hh_row)
        obs_contact_matrix_area.append(contact_area_row)            

    print("user input and simulated contact matrices for contacts within household and in areas are: ")
    
    # user input for contacts within hh
    input_contact_within_hh = [x-1 for x in mean_hh_size_array_input]
    input_contact_within_hh = pd.DataFrame(input_contact_within_hh)
    input_contact_within_hh.columns = ['within_hh']
    print(input_contact_within_hh) 
    
    # simulated contacts within hh    
    cm1 = pd.DataFrame(obs_contact_within_hh)
    cm1.columns = ["", 'within_hh']
    print(cm1.groupby('')[['within_hh']].mean())

    print("")
    
    # user input for contacts in areas
    input_contact_matrix_area = pd.DataFrame(mean_n_contacts_outside_hh_input)
    print(input_contact_matrix_area)
    
    # simulated contacts in areas    
    cm2 = pd.DataFrame(obs_contact_matrix_area)
    cm2.columns = ["", "0", "1", "2"]
    print(cm2.groupby('')[['0', '1', '2']].mean())


    
    
    
    
# dropping contacts according to probability specified when this function is called    
def drop_contacts(graph_input, p_drop_contact_input):
    for node in list(graph_input.nodes):
        connections = list(graph_input.adj[node])
        node_area = node.get_type_of_hh()
        for con in connections:
            r = random.random()
            if r < p_drop_contact_input[node_area]:
                graph_input.remove_edge(node, con)    
    
    
    
    
    
    
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
        initial_prob_I_array_input, mean_n_contacts_outside_hh_input, \
        t_max_input, max_recovery_t_input, p_t_input, \
        p_drop_contact_input, t_drop_contact_input):    

    # open a csv file to write outputs
    f = open('test.csv', 'w')
    f.write("infector_ID,infector_type_of_hh,infectee_ID,infectee_type_of_hh,time,iter \n")

    g = open('sim.csv', 'w')
    g.write("S,I,R,time,iter \n") 
    
    infs_file = open('infs.csv', 'w')
    infs_file.write("area_0,area_1,area_2,time,iter \n")
    
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
                                            mean_n_contacts_outside_hh_input)

        # starting the clock at time = 0
        t = 0
        
        # calculating number of infected, susceptible and recovered individuals at time = 0 and writing to SIR output file
        I_temp = len([i for i in list_hh_ind_input[1] if i.get_state() == "I"])
        S_temp = n_ind_input - I_temp
        R_temp = 0    
        g.write(str(S_temp)+","+str(I_temp)+","+str(R_temp)+","+str(t)+","+str(iter_count)+"\n")
        
        I_0 = len([i for i in list_hh_ind_input[1] if (i.get_state() == "I") & (i.get_type_of_hh() == 0)])
        I_1 = len([i for i in list_hh_ind_input[1] if (i.get_state() == "I") & (i.get_type_of_hh() == 1)])
        I_2 = len([i for i in list_hh_ind_input[1] if (i.get_state() == "I") & (i.get_type_of_hh() == 2)])
        infs_file.write(str(I_0)+","+str(I_1)+","+str(I_2)+","+str(t)+","+str(iter_count)+"\n")

        # running the simulation for this population till maximum time specified (t_max_input)
        while t <= t_max_input:
            
            # if t = t_drop_contact_input (time when NPIs are imposed), then update the adjacency graph:
            if t == t_drop_contact_input:
                drop_contacts(graph_input, p_drop_contact_input)
                
                
#             print(t , sum([graph_input.degree[x] for x in list(graph_input.nodes)]))
            inds = list(graph_input.nodes) # list of all individuals in the population
            infs = [ind for ind in inds if ind.get_state() == "I"] # list of all infected individuals in the population
            for inf in infs:
                time_since_infection = t - inf.get_time_of_infection()
                
                # if recovery time for this individual has been reached, then recover this individual
                if time_since_infection >= max_recovery_t_input:
                    inf.ItoR()
                    I_temp -= 1
                    R_temp += 1
                    if inf.get_type_of_hh() == 0:
                        I_0 -=1
                    elif inf.get_type_of_hh() == 1:
                        I_1 -= 1
                    else:
                        I_2 -= 1
                
                # else check if this individual will infect their contacts
                else:                    
                    # this parts run at all times (the section above only happens at time = t_drop_contact_input)
                    connections = list(graph_input.adj[inf]) # set of all connections for a given infected individual
                    sus = [con for con in connections if con.get_state() == "S"]
                    n_to_infect = int(np.random.binomial(len(sus), p_t_input[time_since_infection], 1)) # this depends on how long inf individual has been infectious
                    sus_selected = random.sample(sus, n_to_infect) # selecting susceptibles who will get infected
                    for sus_sel in sus_selected:
                        # Do the following actions now that is person is getting infected
                        sus_sel.add_time_of_infection(t)
                        sus_sel.add_infector_ID(inf.get_ID())            
                        sus_sel.add_infector_hh(inf.get_hh())          
                        sus_sel.add_infector_type_of_hh(inf.get_type_of_hh())           
                        sus_sel.StoI()           

                        # temporarily saving details of the infection that occured
                        infector_ID_temp = str(inf.get_ID())
                        infector_type_of_hh_temp = str(inf.get_type_of_hh())            
                        infectee_ID_temp = str(sus_sel.get_ID())
                        infectee_type_of_hh_temp = str(sus_sel.get_type_of_hh())            
                        time_of_infection_temp = str(t)
                        
                        # writing details about who got infected by whom and type of hhs to the output file
                        f.write(infector_ID_temp+","+infector_type_of_hh_temp+","+infectee_ID_temp+","\
                                +infectee_type_of_hh_temp+","+time_of_infection_temp+","+str(iter_count)+"\n")

                        # update S, I, R counts
                        S_temp -= 1
                        I_temp += 1
                        
                        if inf.get_type_of_hh() == 0:
                            I_0 +=1
                        elif inf.get_type_of_hh() == 1:
                            I_1 += 1
                        else:
                            I_2 += 1

            # increase time by one day
            t += 1
            
            # writing to SIR output file
            g.write(str(S_temp)+","+str(I_temp)+","+str(R_temp)+","+str(t)+","+str(iter_count)+"\n")
            infs_file.write(str(I_0)+","+str(I_1)+","+str(I_2)+","+str(t)+","+str(iter_count)+"\n")
        
        print("Simulation ", iter_count, " is complete.")
        # the simulation for this population is complete, so increase iter_count by 1
        iter_count += 1
        
    f.close()
    g.close()
        
    
    
    
    
    
    