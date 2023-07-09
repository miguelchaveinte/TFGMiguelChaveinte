import math
import random

import numpy as np

from src.utilsAlgoritmo.func_population import initial_population

def crossover(parent1,parent2,upper_bound,lower_bound):
    '''
    This function performs the crossover between two routes parents of the population to get two new routes.

    Parameters
    ----------
    parent1 : array
        array of the coordinates of the first parent route of the population
    parent2 : array
        array of the coordinates of the second parent route of the population

    Returns 
    -------
    child1 : array
        array of the coordinates of the first child route of the population
    child2 : array
        array of the coordinates of the second child route of the population
    '''

    number_waypoint_new=0

    while(number_waypoint_new<=1):

        index1=math.floor(random.random()*(len(parent1[0])-1))
        index2=min(len(parent2[0])-2,index1+1+len(parent2[0])-math.floor(random.random()*(len(parent2[0])-index1))) #empezamos por el final de la ruta y cogemos o el minimo entre el waypoint antes de la llegada o un número superior a index1+1 seguro



        long1,lat1=parent1[0][index1],parent1[1][index1]
        long2,lat2=parent2[0][index2],parent2[1][index2]

        number_waypoint_new=index2-index1+1


    new_routes_based_great_circle=initial_population(2,[lat1,long1],[lat2,long2],upper_bound,lower_bound,number_waypoint_new)

    new_route=new_routes_based_great_circle[1]# [1:len(new_routes_based_great_circle[1])-2]


    new_route_long=new_route[0][1:len(new_route[0])-1] # asi no cojo el punto de salida y el de llegada que viene dado por los padres
    new_route_lat=new_route[1][1:len(new_route[1])-1]


    child1=[np.concatenate((parent1[0][:index1+1],np.array(new_route_long),parent2[0][index2:]),axis=None),np.concatenate((parent1[1][:index1+1],np.array(new_route_lat),parent2[1][index2:]),axis=None)]


    # ahora child2

    long1,lat1=parent2[0][index1],parent2[1][index1]
    long2,lat2=parent1[0][index2],parent1[1][index2]


    new_routes_based_great_circle=initial_population(2,[lat1,long1],[lat2,long2],upper_bound,lower_bound,number_waypoint_new)

    new_route=new_routes_based_great_circle[1]# [1:len(new_routes_based_great_circle[1])-2]


    new_route_long=new_route[0][1:len(new_route[0])-1] # asi no cojo el punto de salida y el de llegada que viene dado por los padres
    new_route_lat=new_route[1][1:len(new_route[1])-1]


    child2=[np.concatenate((parent2[0][:index1+1],np.array(new_route_long),parent1[0][index2:]),axis=None),np.concatenate((parent2[1][:index1+1],np.array(new_route_lat),parent1[1][index2:]),axis=None)]


    return np.array(child1),np.array(child2)






    