import numpy as np
import datetime
from joblib import load

from src.predict_sog import *

from pymoo.algorithms.moo.nsga2 import NSGA2

from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.termination import get_termination


from pymoo.optimize import minimize


from src.func_objetivos import time_differences
from src.func_objetivos import fuel_use

from src.initial_population import initial_population

from src.func_crossover import crossover
from pymoo.core.population import Population
from pymoo.core.variable import Real, get

from src.func_mutation import mutation

from time import process_time  #ELIMINAR FINAL

def get_closest(array, value): #eliminar al final
    
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

def algorithm_form(form):

    t_start = process_time()
    population_size= int(form['population'])
    upper_bound=int(form['upperBound'])
    lower_bound=int(form['lowerBound'])
    waypoints=int(form['waypoints'])
    offsprings=int(form['offsprings'])
    prob_mutacion=float(form['tasaMutacion'])
    generations=int(form['generations'])

    start_point=[float(form['latSalida']),float(form['lonSalida'])]
    end_point=[float(form['latDestino']),float(form['lonDestino'])]

    start_time=form['fechaSalida']
    end_time=form['fechaLlegada']

    startTime_object = datetime.strptime(start_time, "%Y-%m-%dT%H:%M")
    endTime_object = datetime.strptime(end_time, "%Y-%m-%dT%H:%M")

    shipLength=float(form['shipLength'])
    shipWidth=float(form['shipWidth'])
    shipDraft=float(form['shipDraft'])

    #shipResist=shipLength*shipWidth*shipDraft


    #1-OBTENEMOS LAS CONDICIOENS. FILTRAMOS POR LA FECHA LAS CONDICIONES. CALCULAMOS EL GRIDTIME

    ds_wav,ds_phy=get_condiciones()

    #decididir que fecha escoger -> cogemos la de salida por la imposibilidad de calcular todos los dias

    dat_wav=ds_wav.sel(time=startTime_object,method='nearest')
    dat_phy=ds_phy.sel(time=startTime_object,method='nearest')

    dat_long=dat_phy.longitude.values
    dat_lat=dat_phy.latitude.values

    model=load('./models/DTR_model.joblib')

    [SOG_N,SOG_E,SOG_S,SOG_W]=grid_preparation(dat_wav,dat_phy,shipLength,shipWidth,shipDraft,model)

    gridsTime=calculateGridTime(SOG_N,SOG_E,SOG_S,SOG_W)

    #2-EJECUTAMOS EL ALGORITMO
    res=runNSGA2Algorithm(start_point, end_point,startTime_object,endTime_object,population_size,upper_bound, lower_bound, waypoints,gridsTime,offsprings,generations,dat_long,dat_lat,prob_mutacion)


    #3-RETORNAMOS LOS RESULTADOS QUE NOS INTERESARN
    results = res.F
    routes = res.X

    print(results)
    print('-------------')
    print(routes)

    print('**********************')
    route_minTime = routes[np.argmin(results[:,0], axis=0)]
    route_minfuelUSe = routes[np.argmin(results[:,1], axis=0)]

    print(route_minTime)
    print(route_minfuelUSe)

    t_stop = process_time()
    time_dt = t_stop-t_start
    print("Elapsed time in seconds INSIDE: ", time_dt)

    route_minTime=route_minTime.tolist()
    cost=[]
    for i in range(len(route_minTime[0])-1):  #comprobar esto bien
        long_point=get_closest(dat_wav.longitude.data,route_minTime[0][i])
        lat_point=get_closest(dat_wav.latitude.data,route_minTime[1][i])
        cost.append(gridsTime[0][2041-lat_point][long_point]) # usar get_closest . probar antes 
    route_minTime.append(cost)

    route_minfuelUSe=route_minfuelUSe.tolist()
    cost=[]
    for i in range(len(route_minfuelUSe[0])-1):  #comprobar esto bien
        long_point=get_closest(dat_wav.longitude.data,route_minfuelUSe[0][i])
        lat_point=get_closest(dat_wav.latitude.data,route_minfuelUSe[1][i])
        cost.append(gridsTime[0][2041-lat_point][long_point]) # usar get_closest . probar antes 
    route_minfuelUSe.append(cost)

    print(route_minTime)

    X, Y = np.meshgrid(dat_long[1200:2160], dat_lat[1341:1641])

    # Convertir la matriz en un vector
    X = X.ravel()
    Y = Y.ravel()

    # Crear un DataFrame con las columnas longitude y latitude
    df = pd.DataFrame({'longitude': X, 'latitude': Y})


    #routes=runNSGA2Algorithm(start_point, end_point,start_time,end_time,population_size,upper_bound, lower_bound, waypoints,gridTime,offsprings,generations,dat_long,dat_lat)
    return [route_minTime,route_minfuelUSe],gridsTime,df['latitude'].tolist(),df['longitude'].tolist()



def runNSGA2Algorithm(start_point, end_point,start_time,end_time,population_size,upper_bound, lower_bound, waypoints,gridTime,offsprings,generations,dat_long,dat_lat,prob_mutation):

    class MyProblem(Problem):
            
            def __init__(self):
                super().__init__(n_var=2,
                                n_obj=2,
                                n_constr=0,
                                xl=0.0,
                                xu=1.0)
    
            def _evaluate(self, X, out, *args, **kwargs):
                f1 = time_differences(X[:],start_time,end_time,gridTime,dat_long,dat_lat)
                f2 = fuel_use(X[:],gridTime,dat_long,dat_lat)
                out["F"] = np.column_stack([f1, f2])

    class MySampling(Sampling):
        # population_size, start_point, end_point, upper_bound, lower_bound, n_points=1000
        def __init__(self,start_point, end_point, upper_bound, lower_bound, waypoints,var_type=float, default_dir=None) -> None:
            super().__init__()
            self.var_type = var_type
            self.default_dir = default_dir
            self.start_point=start_point
            self.end_point=end_point
            self.upper_bound=upper_bound
            self.lower_bound=lower_bound
            self.waypoints=waypoints

        def _do(self, problem, n_samples, **kwargs):
            routes= initial_population(n_samples,self.start_point,self.end_point,self.upper_bound,self.lower_bound,self.waypoints)
            return routes
        

    class MyCrossover(Crossover):
        def __init__(self,  upper_bound, lower_bound,waypoints,**kwargs) -> None:
            super().__init__(2, 2,1.0)  # (n_parents,n_offsprings,probability)
            self.upper_bound=upper_bound
            self.lower_bound=lower_bound
            self.waypoints=waypoints

        def do(self, problem, pop, parents=None, **kwargs):

            # if a parents with array with mating indices is provided -> transform the input first
            if parents is not None:
                pop = [pop[mating] for mating in parents]

            # get the dimensions necessary to create in and output
            n_parents, n_offsprings = self.n_parents, self.n_offsprings
            n_matings, n_var = len(pop), problem.n_var

            # get the actual values from each of the parents
            X = np.swapaxes(np.array([[parent.get("X") for parent in mating] for mating in pop]), 0, 1)
            if self.vtype is not None:
                X = X.astype(self.vtype)

            # the array where the offsprings will be stored to
            Xp = np.empty(shape=(n_offsprings, n_matings, n_var,self.waypoints), dtype=X.dtype)  #modify shape to adapt to my problem -> waypoints of the coordinates route. #n_var pq igual que como dividimos las coordenadas 

            # the probability of executing the crossover
            prob = get(self.prob, size=n_matings)

            # a boolean mask when crossover is actually executed
            cross = np.random.random(n_matings) < prob

            # the design space from the parents used for the crossover
            if np.any(cross):

                # we can not prefilter for cross first, because there might be other variables using the same shape as X
                Q = self._do(problem, X, **kwargs)
                assert Q.shape == (n_offsprings, n_matings, problem.n_var,self.waypoints), "Shape is incorrect of crossover impl." #modify shape to adapt to my problem -> waypoints of the coordinates route
                Xp[:, cross] = Q[:, cross]

            #DED
            for k in np.flatnonzero(~cross):
                if n_offsprings < n_parents:
                    s = np.random.choice(np.arange(self.n_parents), size=n_offsprings, replace=False)
                elif n_offsprings == n_parents:
                    s = np.arange(n_parents)
                else:
                    s = []
                    while len(s) < n_offsprings:
                        s.extend(np.random.permutation(n_parents))
                    s = s[:n_offsprings]

                Xp[:, k] = np.copy(X[s, k])

            # flatten the array to become a 2d-array
            #Xp = Xp.reshape(-1, X.shape[-1])
            Xp = Xp.reshape(n_offsprings*n_matings,2,self.waypoints)  #TODO:ADAPT TO N_OFFSPRINGS

            # create a population object
            off = Population.new("X", Xp)

            return off

        def _do(self, problem, X, **kwargs):
            _,n_matings,n_coord,n_waypoints= X.shape

            # The output owith the shape (n_offsprings, n_matings, n_coord,n_waypoints)
            # Because there the number of parents and offsprings are equal it keeps the shape of X
            Y = np.full_like(X, None, dtype=object)

            # for each mating provided
            for k in range(n_matings):
                parent1,parent2=X[0, k,:,:], X[1, k,:,:]
                child1,child2=crossover(parent1,parent2,self.upper_bound,self.lower_bound)
                Y[0, k,:,:], Y[1, k,:,:]=child1,child2

            return Y


    
    class MyMutation(Mutation):
        def __init__(self,upper_bound, lower_bound, prob=None, **kwargs):
            super().__init__()
            self.prob = prob
            self.upper_bound=upper_bound
            self.lower_bound=lower_bound

        def _do(self, problem, X, **kwargs):
            offsprings=[]

            for route in X:
                #mutation action with probability
                if np.random.uniform(0, 1) < self.prob:
                    route_mutated=mutation(route,self.upper_bound,self.lower_bound)
                    offsprings.append(route_mutated)
                else:
                    offsprings.append(route)
            
            offsprings=np.array(offsprings)
            return offsprings

    
        



    problem=MyProblem()

    print(problem)

    algorithm = NSGA2(
        pop_size=population_size,
        n_offsprings=offsprings,
        sampling=MySampling(start_point, end_point, upper_bound, lower_bound, waypoints),
        crossover=MyCrossover(upper_bound/4, lower_bound/4,waypoints), 
        mutation=MyMutation(upper_bound/4, lower_bound/4,prob_mutation),  #TODO: TRAER LA PROBABILIDAD DEL USUARIO 
        eliminate_duplicates=False
    )

    termination=get_termination("n_gen", generations)

    res = minimize(problem,
                algorithm,
                termination, # ('n_gen', generations) -> tb asi se puede poner
                seed=1,
                save_history=True,
                verbose=True)
    

    return res

'''


import random

gridTimeRandom=[[[random.random() for i in range(4320)] for j in range(2041)]for x in range (5)]
print(gridTimeRandom[0][0][3])

import xarray as xr
import warnings
warnings.simplefilter("ignore")
from datetime import datetime
'''

'''instantiate the connection to the OPeNDAP server thanks to a local 
function copernicusmarine_datastore(): '''

#https://marine.copernicus.eu/news/access-data-opendap-erddap-api
#https://nrt.cmems-du.eu/thredds/dodsC/cmems_mod_glo_wav_anfc_0.083deg_PT3H-i.html
#https://help.marine.copernicus.eu/en/articles/4683022-what-are-the-advantages-of-the-file-transfer-protocol-ftp-data-access-service#h_3a5f09adf9

#! /usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Copernicus Marine User Support Team"
__copyright__ = "(C) 2021 E.U. Copernicus Marine Service Information"
__credits__ = ["E.U. Copernicus Marine Service Information"]
__license__ = "MIT License - You must cite this source"
__version__ = "202104"
__maintainer__ = "D. Bazin, E. DiMedio, C. Giordan"
__email__ = "servicedesk dot cmems at mercator hyphen ocean dot eu"


'''



def copernicusmarine_datastore(dataset, username, password):
    from pydap.client import open_url
    from pydap.cas.get_cookies import setup_session
    cas_url = 'https://cmems-cas.cls.fr/cas/login'
    session = setup_session(cas_url, username, password)
    session.cookies.set("CASTGC", session.cookies.get_dict()['CASTGC'])
    database = ['my', 'nrt']
    url = f'https://{database[0]}.cmems-du.eu/thredds/dodsC/{dataset}'
    try:
        data_store = xr.backends.PydapDataStore(open_url(url, session=session))
    except:
        url = f'https://{database[1]}.cmems-du.eu/thredds/dodsC/{dataset}'
        data_store = xr.backends.PydapDataStore(open_url(url, session=session))
    return data_store


USERNAME = 'mchaveinte'
PASSWORD = 'Cuchiagosto23'
DATASET_ID ='cmems_mod_glo_phy_anfc_0.083deg_P1D-m'

data_phy_prueba=copernicusmarine_datastore(DATASET_ID, USERNAME, PASSWORD)

dataset_phy_prueba= xr.open_dataset(data_phy_prueba)
import random

gridTimeRandom= [[[random.random() for i in range(4320)] for j in range(2041)] for x in range (4)]

print(np.shape(gridTimeRandom[0]))

startTime_object = datetime.strptime('20.06.2021 12:00', "%d.%m.%Y %H:%M" )
endTime_object = datetime.strptime('30.06.2021 12:00', "%d.%m.%Y %H:%M" )

print(startTime_object)

import pandas as pd

print(dataset_phy_prueba.sel(time=startTime_object,method='nearest'))

#TODO: comprobar offspring mayor que 10

res=runNSGA2Algorithm([40.7128,-74.0060],[38.7223,-9.1393],startTime_object,endTime_object,10,8,8,10,gridTimeRandom,10,20,dataset_phy_prueba.longitude.data,dataset_phy_prueba.latitude.data,0.5)

results = res.F
routes = res.X

print(results)
print('-------------')
print(routes)

print('**********************')
route_minTime = routes[np.argmin(results[:,0], axis=0)]
route_minfuelUSe = routes[np.argmin(results[:,1], axis=0)]

print(route_minTime)
print(route_minfuelUSe)


'''
