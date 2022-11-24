import numpy as np
import pandas as pd
import os

input_data_path = "../input_data/"
weights_path = "../weights/"
output_data_path= "../output_data/"



def preprocessing( df ):
    temp = np.expand_dims( df.pop("Temperatura").values + 273., axis=1)**-(1)
    y = np.expand_dims(  df.pop("Riv_Misurato").values , axis=1)
    data = np.log( df.values )
    x = np.concatenate( [ temp, data ], axis=1 )
    return x,y


def preprocessing_opt(df):
    one_on_t = np.expand_dims(df["Temperatura"].values + 273., axis=1) ** (-1)
    log_h = np.log(np.expand_dims(df["Altezza"].values, axis=1))
    log_s = np.log(np.expand_dims(df["Velocità"].values, axis=1))
    log_p = np.log(np.expand_dims(df["Pressione_PID"].values, axis=1))
    log_t = np.log(np.expand_dims(df["Distanza_PID"].values, axis=1))
    x = np.concatenate([one_on_t, log_h, log_s, log_p, log_t], axis=1)
    y = np.expand_dims( df["Rivestimento_PID"].values, axis=1 )
    return x, y

def load_join_shuffle_validation_set():
    validation = pd.read_excel(input_data_path + "prep_validation.xlsx")
    validation_opt = pd.read_excel(input_data_path + "prep_validation_opt.xlsx")

    x_val_opt, y_val_opt = preprocessing_opt(validation_opt)
    x_val, y_val = preprocessing(validation)
    x_val_joined = np.concatenate([x_val, x_val_opt], axis=0)
    y_val_joined = np.concatenate([y_val, y_val_opt], axis=0)

    joined = np.concatenate([x_val_joined,y_val_joined], axis=1)

    np.random.seed(42)
    np.random.shuffle(joined)

    x = joined[:,0:-1]
    y = np.expand_dims( joined[:,-1],axis=1)

    #to check the randomness, see the same data at the same shuffled position.
    # print( x[1500] )
    #print( y[1500] )

    return x,y
