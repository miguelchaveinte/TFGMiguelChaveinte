import math
import random

import numpy as np

from src.utilsAlgoritmo.func_population import initial_population

def mutation(route,upper_bound,lower_bound):
    """
    mutation of one route, so changes a random part of it
    Parameters->
    route: array, [lon_coord,lat_coord]
        the coordinates of the route
    upper_bound: int
        the upper bound of the part of the mutation of the route
    lower_bound: int
        the lower bound of the part of the mutation of the route

    Returns->
    route_mutate: array, [lon_coord,lat_coord]
        the coordinates of the mutationroute
    """

    number_waypoint_new=0

    while(number_waypoint_new<=1):

        index1=math.floor(random.random()*(len(route[0])-1))
        index2=min(len(route[0])-2,index1+1+len(route[0])-math.floor(random.random()*(len(route[0])-index1))) #empezamos por el final de la ruta y cogemos o el minimo entre el waypoint antes de la llegada o un número superior a index1+1 seguro


        long1,lat1=route[0][index1],route[1][index1]
        long2,lat2=route[0][index2],route[1][index2]

        number_waypoint_new=index2-index1+1


    new_routes_based_great_circle=initial_population(2,[lat1,long1],[lat2,long2],upper_bound,lower_bound,number_waypoint_new)

    new_route=new_routes_based_great_circle[1]# [1:len(new_routes_based_great_circle[1])-2]


    new_route_long=new_route[0][1:len(new_route[0])-1] # asi no cojo el punto de salida y el de llegada que viene dado por los padres
    new_route_lat=new_route[1][1:len(new_route[1])-1]


    route_mutate=[np.concatenate((route[0][:index1+1],np.array(new_route_long),route[0][index2:]),axis=None),np.concatenate((route[1][:index1+1],np.array(new_route_lat),route[1][index2:]),axis=None)]

    return route_mutate