from src.modelo.predict_sog import get_condiciones,grid_preparation,calculateGridTime

from joblib import load


def condiciones_mar(startTime_object,shipLength,shipWidth,shipDraft):
    ds_wav,ds_phy=get_condiciones()

    #decididir que fecha escoger -> cogemos la de salida por la imposibilidad de calcular todos los dias

    dat_wav=ds_wav.sel(time=startTime_object,method='nearest')
    dat_phy=ds_phy.sel(time=startTime_object,method='nearest')


    model=load('./models/DTR_model.joblib')

    [SOG_N,SOG_E,SOG_S,SOG_W]=grid_preparation(dat_wav,dat_phy,shipLength,shipWidth,shipDraft,model)

    gridsTime=calculateGridTime(SOG_N,SOG_E,SOG_S,SOG_W)

    return gridsTime,dat_wav,dat_phy
