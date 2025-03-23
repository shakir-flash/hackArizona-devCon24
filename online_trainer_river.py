# online_trainer_river.py
from river import linear_model, preprocessing

zone_models = {}

def initialize_online_model(zone_name):
    model = preprocessing.StandardScaler() | linear_model.LinearRegression()
    zone_models[zone_name] = model
    return model

def update_online_model(zone_name, X_dict, y):
    model = zone_models.get(zone_name)
    if model:
        model.learn_one(X_dict, y)

def predict_online_model(zone_name, X_dict):
    model = zone_models.get(zone_name)
    if model:
        return model.predict_one(X_dict)
    return None
