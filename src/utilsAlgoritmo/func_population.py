import pyproj
import numpy as np


def population_form(form):
    population_size= int(form['population'])
    start_point=[float(form['latSalida']),float(form['lonSalida'])]
    end_point=[float(form['latDestino']),float(form['lonDestino'])]
    upper_bound=int(form['upperBound'])
    lower_bound=int(form['lowerBound'])

    routes=initial_population(population_size, start_point, end_point, upper_bound, lower_bound,10)
    return routes

def great_circle(start_point, end_point, n_points=1000):
    """
    reference route: great circle between star_point and end_point by Pyproj library 

    Parameters->
    start_point: array, [start_lat,start_long]
        the startedpoint of the route
    end_point: array, [end_lat,end_long]
        the endpoint of the route
    n_points: int
        the number of coordinates that define the route (waypoints)

    Returns->
    [x,y]: array, [lon_coord,lat_coord]
        the coordinates of the great circle route
    """

    # Pyproj documentation: https://pyproj4.github.io/pyproj/stable/api/geod.html

    g = pyproj.Geod(ellps='WGS84', f=0)

    (az12, az21, dist) = g.inv(start_point[1], start_point[0], end_point[1], end_point[0])

    # calculate the latitude and longitude
    lonlats = g.npts(start_point[1], start_point[0], end_point[1],
                     end_point[0], n_points, initial_idx=0, terminus_idx=0)

    x = [punto[0] for punto in lonlats]
    y = [punto[1] for punto in lonlats]

    return [x, y]


def initial_population(population_size, start_point, end_point, upper_bound, lower_bound, n_points=1000):
    """
    creates initial population based on great circle route. 

    Parameters->
    population_size: int
        the number of routes in the population
    start_point: array, [start_lat,start_long]
        the startedpoint of the route
    end_point: array, [end_lat,end_long]
        the endpoint of the route
    upper_bound: int
        the upper bound of the latitude of the waypoints
    lower_bound: int
        the lower bound of the latitude of the waypoints
    n_points: int
        the number of coordinates that define the route (waypoints)

    Returns->
    routes: array, [[lon_coord,lat_coord]]
        the coordinates of each route of the initial population
    """

    routes = []
    great_circle_route = great_circle(start_point, end_point, n_points)
    routes.append(great_circle_route)
    route_upper = [[x + upper_bound if i == 1 and 0 < j < len(row) - 1 else x for j, x in enumerate(row)] for i, row in enumerate(great_circle_route)]
    route_lower = [[x - lower_bound if i == 1 and 0 < j < len(row) - 1 else x for j, x in enumerate(row)] for i, row in enumerate(great_circle_route)]
    for i in range(population_size-1):
        waypoints_lat=[]
        #waypoints_lat.append(great_circle_route[1][0]) #añadimos el start_lat 
        # -> lo quito pq ya en route_upper y lower lo he tenido en cuenta que empiece y termine igual y por tanto random da la coordenada.
        for j in range(n_points):
            lat_bound = np.random.uniform(route_lower[1][j], route_upper[1][j])
            waypoints_lat.append(lat_bound)
        #waypoints_lat.append(great_circle_route[1][n_points-1]) #añadimos el end_lat
        waypoints_lon = great_circle_route[0]
        route = [waypoints_lon,waypoints_lat]
        routes.append(route)
    return routes





