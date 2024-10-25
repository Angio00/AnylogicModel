import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
def chooseTheBest(X,y):
    kf = KFold(n_splits=6)
    kf.get_n_splits(X)
    lista_modelli = []
    lista_score = []
    for train_index, test_index in kf.split(X):
        modello = RandomForestRegressor(n_jobs= -1)
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index ]
        y_test = y[test_index]
        modello.fit(X_train, y_train)
        score = mean_squared_error(y_test, modello.predict(X_test))
        lista_modelli.append(modello)
        lista_score.append(score)
    index_best = lista_score.index(min(lista_score))
    return lista_modelli[index_best]


def PrevediDomanda(periodo: int, point_sales: str, type: str, mese, anno, day_month):
    demand_prec = pd.read_csv(f"DomandeRetailers//Domanda{point_sales}.txt", sep = " ", names= ["ValueA", "ValueB", "data"])
    demand_simul = pd.read_csv(f"DOmandeRetailersnew//Domanda{point_sales}new.txt", sep = " ", names = ["ValueA", "ValueB", "data"])
    demand_prec['data'] = pd.to_datetime(demand_prec['data'])
    demand_simul['data'] = pd.to_datetime(demand_simul['data'])
    demand_prec.set_index('data', inplace= True)
    demand_simul.set_index('data', inplace=True)
    week = f'{periodo}W'
    demand_prec = demand_prec.resample(week).sum()
    demand_simul = demand_simul.resample(week).sum()
    grandezza_campione = demand_prec.shape[0]
    combined_demand = pd.concat([demand_prec, demand_simul])
    combined_demand['anno'] = combined_demand.index.year
    combined_demand['mese'] = combined_demand.index.month
    combined_demand['day'] = combined_demand.index.day
    
    if type == "A":
        X_train = combined_demand.drop(columns=['ValueA', 'ValueB']).values[-grandezza_campione:]
        y_train = combined_demand['ValueA'].values[-grandezza_campione:]
    else:
        X_train = combined_demand.drop(columns=['ValueA', 'ValueB']).values[-grandezza_campione:]
        y_train = combined_demand['ValueB'].values[-grandezza_campione:]

    x_prevision = np.array([anno, mese, day_month])
    x_prevision_reshaped = x_prevision.reshape(1,-1)
    Regressor = chooseTheBest(X_train, y_train)
    return int(Regressor.predict(x_prevision_reshaped)[0])


