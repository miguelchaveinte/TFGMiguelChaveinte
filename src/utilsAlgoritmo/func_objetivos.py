from datetime import datetime, timedelta
import numpy as np


def get_closest(array, value):

    """
    get the closest value to a given value in an array

    Parameters->
    array: array
        array of values
    value: float
        value to compare
    
    Returns->   
    np.abs(array - value).argmin(): integer
        the index of the closest value to a given value in an array
    """
        
    return np.abs(array - value).argmin()

def calculate_bearing(route):
    """
    calculates the bearing for each cell in the grid

    Parameters->
    route: array
        route defining by [gridCooridinateX, gridCoordinateY, speed]

    Returns->
    route: array, [gridCooridinateX, gridCoordinateY,speed, bearing]
        array of the coordinates of the route, speed and the bearing of each coordinate of cell in the grid
    """

    bearing=[]

    route=route.tolist()

    for i in range (len(route[0])-1):

        #todo: construir una nueva lista 

        if route[1][i]<route[1][i+1]: #up direction
            bearing.append(0)
        elif route[1][i]>route[1][i+1]: #down direction
            bearing.append(1)
        elif route[0][i]<route[0][i+1]: #right direction
            bearing.append(2)
        elif route[0][i]>route[0][i+1]: #left direction
            bearing.append(3)
        else:
            bearing.append("ERROR BEARING")

    route.append(bearing)

    return route


def calculate_time_grid(route,grids_time,dat_long,dat_lat):
    """
    calculates the time required for a route based on grid cell time

    Parameters->
    route: array
        route defining by [gridCooridinateX, gridCoordinateY, speed]
    grid_time: array
        the time of the grid

    Returns->
    hours: integer
        the time required for the route
    """

    time_sum=0
    #comprobar que route es un array 
    route_bearing=calculate_bearing(route)
    for i in range(1,len(route_bearing[0])-1):  #comprobar esto bien
        bearing=route_bearing[2][i] # seria coordinates[3] -> OJO!! COMPROBAR
        grid_time=grids_time[bearing]
        long_point=get_closest(dat_long,route_bearing[0][i])
        lat_point=get_closest(dat_lat,route_bearing[1][i])
        time_sum=time_sum+(grid_time[2041-lat_point][long_point])/2 # usar get_closest . probar antes 

    hours=timedelta(minutes=time_sum)
    return hours




def time_differences(routes,start_time,end_time,grids_time,dat_long,dat_lat):
    """
    calculates the time differences between the routes of the population taking into account the start time, end time  and time required for the grid cell

    Parameters->
    routes: array
        array of the coordinates of each route of the population
    start_time: datetime
        the start time of the route
    end_time: datetime
        the end time of the route
    grid_time: array
        array of the time required for the grid cell.
        [direction of the grid: N,S,E,W ; speed_grid ; bearing_grid] -> OJO!!! COMPROBAR QUE LO PASO ASÍ 

    Returns->
    time_diff: array, [time_diff]
        array of the time differences between the routes of the population
    """

    # revisar que los start_time y end_time nos venga como datetime. si viene como str: datetime.strptime(start_time, "%d.%m.%Y %H:%M" )
    time_diff = []
    for route in routes:
        duration_time_grids=calculate_time_grid(route,grids_time,dat_long,dat_lat) # route comprobar-> esto por la poblacion
        duration_time_route=duration_time_grids+start_time #ruta
        difference_time= end_time- duration_time_route
        total_min=difference_time.total_seconds()/60
        time_diff.append(float(total_min) ** 2)

    return time_diff  #np.array(time_diff) ??






def calculate_fuel(route,grids_time,dat_long,dat_lat):
    """
    calculates the fuel use in liters for a route taking into account the time required for the grid cell

    Parameters->
    route: array
        array of the coordinates of each route of the population
    grid_time: array
        array of the time required for the grid cell.
        [direction of the grid: N,S,E,W ; speed_grid ; bearing_grid]

    Returns->
    use_fuel: integer
        the fuel use in liters for a route
    """

    use_fuel=0
    route_bearing=calculate_bearing(route)
    for i in range(1,len(route_bearing[0])-1):  #comprobar esto bien
        bearing=route_bearing[2][i] # seria coordinates[3] -> OJO!! COMPROBAR
        grid_time=grids_time[bearing]

        long_point=get_closest(dat_long,route_bearing[0][i])
        lat_point=get_closest(dat_lat,route_bearing[1][i])

        #assuming 70% of engine power, needs 154g/kwh and the engine needs 33200kw -> https://www.wingd.com/en/documents/general/papers/engine-selection-for-very-large-container-vessels.pdf/
        use_fuel=use_fuel+(grid_time[2041-lat_point][long_point]/60 * 154*33200)/2   # usar get_closest . probar antes 

    return use_fuel


def fuel_use(routes,grids_time,dat_long,dat_lat):
    """
    calculates the fuel use in liters between the routes of the population taking into account the time required for the grid cell

    Parameters->
    routes: array
        array of the coordinates of each route of the population
    grid_time: array
        array of the time required for the grid cell.
        [direction of the grid: N,S,E,W ; speed_grid ; bearing_grid]

    Returns->
      
    """

    all_fuel_use = []
    for route in routes:
        use_fuel=calculate_fuel(route,grids_time,dat_long,dat_lat)
        use_fuel_tons=use_fuel/1000000
        all_fuel_use.append(use_fuel_tons)

    return all_fuel_use