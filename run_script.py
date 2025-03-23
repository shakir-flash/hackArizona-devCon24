from ml_trainer import train_zone_model, predict_zone_model
train_zone_model(df, "LEO-W_CO2", target_col="co2")
predictions = predict_zone_model(df, "LEO-W_CO2", target_col="co2")
