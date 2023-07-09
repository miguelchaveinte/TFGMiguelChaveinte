import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime
import ftplib
import time

import xarray as xr


'''
def download(url,file_path,filename,user_cmems,pass_cmems):
    """
    download the data from the CMEMS

    Parameters->
    url: string
        url of the CMEMS
    path: string
        path of the CMEMS
    filename: string
        name of the file to download
    user_cmems: string
        user of the CMEMS
    pass_cmems: string
        password of the CMEMS
    
    Returns->   
    None
    """

    with ftplib.FTP(url) as ftp:
        try:
            ftp.login(user_cmems,pass_cmems)
            ftp.cwd(file_path)
            if(os.path.isfile(filename)):
                print("File already exists: ",filename)
            else:
                print("Downloading... :",filename)
                ftp.retrbinary("RETR " + filename ,open(filename, 'wb').write)
        except ftplib.all_errors as e:
            print('FTP error:', e)

def get_condiciones(startTime,endTime):
    """
    get the conditions of the weather for a given time

    Parameters->
    startTime: datetime
        the start time of the route
    endTime: datetime
        the end time of the route
    
    Returns->   
    condiciones: array
        array of the conditions of the weather for a given time
    """

    load_dotenv()
    user_cmems=os.getenv("USER_CMEMS")
    pass_cmems=os.getenv("PASS_CMEMS")

    date = startTime.strftime("%Y%m%d")

    path_date = date[0:4] + "/" + date[4:6]

    url = 'nrt.cmems-du.eu'
    path_wav = 'Core/GLOBAL_ANALYSISFORECAST_WAV_001_027/cmems_mod_glo_wav_anfc_0.083deg_PT3H-i/' + path_date
    path_phy = ['Core/GLOBAL_ANALYSISFORECAST_PHY_001_024/cmems_mod_glo_phy_anfc_0.083deg_P1D-m/' + path_date,'Core/GLOBAL_ANALYSISFORECAST_PHY_001_024/cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m/' + path_date,'Core/GLOBAL_ANALYSISFORECAST_PHY_001_024/cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m/' + path_date,'Core/GLOBAL_ANALYSISFORECAST_PHY_001_024/cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m/' + path_date]

    # get the data from the CMEMS
    with ftplib.FTP(url) as ftp:
        try:
            ftp.login(user_cmems,pass_cmems)
            ftp.cwd(path_wav)
            files = ftp.nlst()
            files = [i for i in files if date in i]
            filename_wav = files[0]

            download(url,path_wav,filename_wav,user_cmems,pass_cmems)

            filename_phy=[]
            for i in range (len(path_phy)):
                ftp.cwd('/')
                ftp.cwd(path_phy[i])
                files = ftp.nlst()
                files = [i for i in files if date in i]
                filename_phy_i = files[0]
    
                download(url, path_phy[i],filename_phy_i,user_cmems,pass_cmems)
                filename_phy.append(filename_phy_i)
            
        except ftplib.all_errors as e:
            print('FTP error:', e)

'''

'''instantiate the connection to the OPeNDAP server thanks to a local 
function copernicusmarine_datastore(): '''

#! /usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Copernicus Marine User Support Team"
__copyright__ = "(C) 2021 E.U. Copernicus Marine Service Information"
__credits__ = ["E.U. Copernicus Marine Service Information"]
__license__ = "MIT License - You must cite this source"
__version__ = "202104"
__maintainer__ = "D. Bazin, E. DiMedio, C. Giordan"
__email__ = "servicedesk dot cmems at mercator hyphen ocean dot eu"

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

'''
# utils to convert dates 
str_to_date = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
date_to_str = lambda x: x.strftime('%Y-%m-%dT%H:%M:%SZ')
str_to_date2 = lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ')
'''


def get_condiciones():
    """
    get the conditions of the weather for a given time


    Returns->   
    condiciones: array
        array of the conditions of the weather for a given time
    """

    load_dotenv()
    user_cmems=os.getenv("USER_CMEMS")
    pass_cmems=os.getenv("PASS_CMEMS")

    dataset_wav='cmems_mod_glo_wav_anfc_0.083deg_PT3H-i'
    dataset_phy=['cmems_mod_glo_phy_anfc_0.083deg_P1D-m','cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m','cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m','cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m',]


    # get the data from the CMEMS
    data_store_wav = copernicusmarine_datastore(dataset_wav, user_cmems, pass_cmems)
    data_store_phy = []
    for i in range (len(dataset_phy)):
        time.sleep(0.01)
        data_store_phy.append(copernicusmarine_datastore(dataset_phy[i], user_cmems, pass_cmems))

    # get the data from the CMEMS
    ds_wav = xr.open_dataset(data_store_wav)
    ds_phy_array = []
    for i in range (len(data_store_phy)):
        ds_phy_array.append(xr.open_dataset(data_store_phy[i]))

    # TODO: join ds_phy
    ds_phy = xr.merge(ds_phy_array)

    # TODO: save in paht -> not possible due to size of files

    #retorno los datasets y luego ya cojo las fechas que necesite.
    return (ds_wav,ds_phy)



def relative_direction_wind(ship_dir,vmdr_ww):
  vmdr_ww_360=vmdr_ww
  vmdr_ww_360[vmdr_ww_360 < 0] = 360 + vmdr_ww_360[vmdr_ww_360 < 0]

  
  if ship_dir == "N":
      dir_4 = np.full((len(vmdr_ww_360), 1), 2)
      dir_4[(vmdr_ww_360 < 45) | (vmdr_ww_360 > 315)] = 1
      dir_4[(vmdr_ww_360 > 135) & (vmdr_ww_360 < 225)] = 3
  elif ship_dir == "E":
      dir_4 = np.full((len(vmdr_ww_360), 1), 2)
      dir_4[(vmdr_ww_360 > 45) & (vmdr_ww_360 < 135)] = 1
      dir_4[(vmdr_ww_360 > 225) & (vmdr_ww_360 < 315)] = 3
  elif ship_dir == "W":
      dir_4 = np.full((len(vmdr_ww_360), 1), 2)
      dir_4[(vmdr_ww_360 > 45) & (vmdr_ww_360 < 135)] = 3
      dir_4[(vmdr_ww_360 > 225) & (vmdr_ww_360 < 315)] = 1
  else: # ship_dir == "S"
      dir_4 = np.full((len(vmdr_ww_360), 1), 2)
      dir_4[(vmdr_ww_360 < 45) | (vmdr_ww_360 > 315)] = 3
      dir_4[(vmdr_ww_360 > 135) & (vmdr_ww_360 < 225)] = 1

  '''

  if ship_dir == "N":
      dir_4 = np.full((len(vmdr_ww), 1), 2)
      dir_4[(vmdr_ww < 45) | (vmdr_ww > 315)] = 1
      dir_4[(vmdr_ww > 135) & (vmdr_ww < 225)] = 3
  elif ship_dir == "E":
      dir_4 = np.full((len(vmdr_ww), 1), 2)
      dir_4[(vmdr_ww > 45) & (vmdr_ww < 135)] = 1
      dir_4[(vmdr_ww > 225) & (vmdr_ww < 315)] = 3
  elif ship_dir == "W":
      dir_4 = np.full((len(vmdr_ww), 1), 2)
      dir_4[(vmdr_ww > 45) & (vmdr_ww < 135)] = 3
      dir_4[(vmdr_ww > 225) & (vmdr_ww < 315)] = 1
  else: # ship_dir == "S"
      dir_4 = np.full((len(vmdr_ww), 1), 2)
      dir_4[(vmdr_ww < 45) | (vmdr_ww > 315)] = 3
      dir_4[(vmdr_ww > 135) & (vmdr_ww < 225)] = 1

  '''



  return dir_4



def join_cmems(ds_wav,ds_phy,ship_length,ship_width,ship_draft,vhm0,l):

    #!!!! OJO !!! : TIENEN QUE VENIR LOS DS_WAV Y DS_PHY FILTRADOS YA POR LA FECHA

    #data_array = (np.flipud(ds_wav["VHM0"][:, :]).data)  # extract data from CMEMS
    #dim = data_array.shape
    #l = np.prod(dim)  # get number of "pixel"

    # extract parameters from cmems dataset and reshape to array with dimension of 1 x number of pixel
    vhm0_ww=np.nan_to_num((np.flipud(ds_wav["VHM0_WW"][:, :])).reshape(l, 1),nan=-32767)
    vmdr_sw2=np.nan_to_num((np.flipud(ds_wav["VMDR_SW2"][:, :])).reshape(l, 1),nan=-32767)
    vmdr_sw1=np.nan_to_num((np.flipud(ds_wav["VMDR_SW1"][:, :])).reshape(l, 1),nan=-32767)
    vtm10=np.nan_to_num((np.flipud(ds_wav["VTM10"][:, :])).reshape(l, 1),nan=-32767)
    vmdr_ww=np.nan_to_num((np.flipud(ds_wav["VMDR_WW"][:, :])).reshape(l, 1),nan=-32767)
    vtm01_sw2=np.nan_to_num((np.flipud(ds_wav["VTM01_SW2"][:, :])).reshape(l, 1),nan=-32767)
    vhm0_sw1=np.nan_to_num((np.flipud(ds_wav["VHM0_SW1"][:, :])).reshape(l, 1),nan=-32767)
    vsdx=np.nan_to_num((np.flipud(ds_wav["VSDX"][:, :])).reshape(l, 1),nan=-32767)
    vsdy=np.nan_to_num((np.flipud(ds_wav["VSDY"][:, :])).reshape(l, 1),nan=-32767)
    #vhm0=np.nan_to_num((np.flipud(ds_wav["VHM0"][:, :])).reshape(l, 1),nan=-32767)
    vhm0_sw2=np.nan_to_num((np.flipud(ds_wav["VHM0_SW2"][:, :])).reshape(l, 1),nan=-32767)

    resist=np.full((l, 1), ship_length*ship_width*ship_draft) 
    draft=np.full((l, 1),ship_draft) 

    thetao=np.nan_to_num((np.flipud(ds_phy["thetao"][1,:, :])).reshape(l, 1),nan=9.96921e+36)  #np.nan_to_num(so,nan=9.96921e+36)
    so=np.nan_to_num((np.flipud(ds_phy["so"][1,:, :])).reshape(l, 1),nan=9.96921e+36)
    uo=np.nan_to_num((np.flipud(ds_phy["uo"][1,:, :])).reshape(l, 1),nan=9.96921e+36)
    vo=np.nan_to_num((np.flipud(ds_phy["vo"][1,:, :])).reshape(l, 1),nan=9.96921e+36)
    zos=np.nan_to_num((np.flipud(ds_phy["zos"][:, :])).reshape(l, 1),nan=9.96921e+36)
    mlotst=np.nan_to_num((np.flipud(ds_phy["mlotst"][:, :])).reshape(l, 1),nan=9.96921e+36)
    bottomT=np.nan_to_num((np.flipud(ds_phy["tob"][:, :])).reshape(l, 1),nan=9.96921e+36)

    dir_possibilities=["N","E","S","W"]

    X_predict=[]

    for dir in dir_possibilities:
        dir_4= relative_direction_wind(dir,vmdr_ww)

        data_dir = np.concatenate((resist, draft, vhm0_ww, vmdr_sw2, vmdr_sw1, vtm10,vmdr_ww,vtm01_sw2,vhm0_sw1,vsdx,vsdy,vhm0,vhm0_sw2,thetao,so,uo,vo,zos,mlotst,bottomT,dir_4), axis=1)

        X_predict_dir = pd.DataFrame(data=data_dir,    # values
            index=list(range(0, l)),    # 1st column as index
            columns=["resist","Draft", "VHM0_WW", "VMDR_SW2", "VMDR_SW1", "VTM10", "VMDR_WW", "VTM01_SW2","VHM0_SW1","VSDX","VSDY","VHM0","VHM0_SW2","thetao","so","uo","vo","zos","mlotst","bottomT","dir_4"])  # 1st row as the column names
   
        X_predict.append(X_predict_dir)

    return X_predict



def grid_preparation(ds_wav,ds_phy,ship_length,ship_width,ship_draft,model):

    data_cmems_vhm0=np.nan_to_num((np.flipud(ds_wav["VHM0"][:, :])),nan=-32767)
    dim_data=data_cmems_vhm0.shape
    l = np.prod(dim_data)  # get number of "pixel"


    X_predict_array=join_cmems(ds_wav,ds_phy,ship_length,ship_width,ship_draft,data_cmems_vhm0.reshape(l, 1),l)

    #print(X_predict)

    #print(np.unique(X_predict['dir_4']))

    SOG_return=[]


    for X_predict in X_predict_array:
        
        SOG_prediction=model.predict(X_predict)

        SOG_prediction = SOG_prediction.reshape(dim_data)  # reshape to 'coordinates'

        SOG_prediction[data_cmems_vhm0 < -30000] = -5  # -32767.0 # mask data with negative value

        SOG_return.append(SOG_prediction)


    return SOG_return



def calculateGridTime(SOG_N,SOG_E,SOG_S,SOG_W):
    kmGridEW = np.load("./models/gridLengthEW.npy") 
    kmGridNS = np.load("./models/gridLengthNS.npy")

    timeGridE = SOG_E
    timeGridE = np.where(timeGridE < 0, 50000, (kmGridEW * 1000) / (timeGridE * 30.87))

    timeGridN = SOG_N
    timeGridN = np.where(timeGridN < 0, 50000, (kmGridNS * 1000) / (timeGridN * 30.87))
   
    timeGridS = SOG_S
    timeGridS = np.where(timeGridS < 0, 50000, (kmGridNS * 1000) / (timeGridS * 30.87))
  
    timeGridW = SOG_W
    timeGridW = np.where(timeGridW < 0, 50000, (kmGridEW * 1000) / (timeGridW * 30.87))

    timeGrids = [timeGridN, timeGridS, timeGridE, timeGridW]

    return timeGrids



    





    




