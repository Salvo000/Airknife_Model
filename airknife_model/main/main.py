import numpy as np
import pandas as pd
from code.utils import input_data_path, preprocessing
from code.controller_net import preprocessing as preprocessing_controller
from code.coating_net import load_coating_net
from code.controller_net import load_controller_net,get_riv_from_tuple

#example of external usage of the coating net
model = load_coating_net()
df = pd.read_excel( input_data_path+"shuffled_test.xlsx")
x,y = preprocessing(df=df.copy())
riv_preds = model.predict(x)
df["Riv_Predetto"] = np.exp(riv_preds)
print( df )


#example of external usage of the controller net
controller = load_controller_net()
test = pd.read_excel( input_data_path+"shuffled_test.xlsx" )
x_test,y_test = preprocessing_controller( data=test.copy() )

res = controller.predict(x_test)
log_p_gen, log_d_gen = np.split(res, 2, axis=1)

test["Pres_Gen"] = np.exp(log_p_gen)
test["Dist_Gen"] = np.exp(log_d_gen)
test["Riv_Pred"] = np.exp(get_riv_from_tuple(model,log_p_gen, log_d_gen, test))

print( test )



#esempio di codice per le vecchie metriche del opt sul controller.

# codice di vecchie metriche calcolate su opt
# dev_set = validation.copy()
# dev_set["rmse_pressure_preset_pid"] = np.round( np.sqrt( ( dev_set["Pressione_PID"].values - dev_set["Pressione_Preset"].values ) ** 2), decimals=3)
# dev_set["rmse_pressure_model_pid" ] = np.round( np.sqrt( ( dev_set["Pressione_PID"].values - dev_set["Pres_Gen"].values ) ** 2), decimals=3)
# dev_set["rmse_distance_preset_pid"] = np.round( np.sqrt( ( dev_set["Distanza_PID"].values - dev_set["Distanza_Preset"].values ) ** 2), decimals=3 )
# dev_set["rmse_distance_model_pid"] = np.round( np.sqrt((dev_set["Distanza_PID"].values - dev_set["Dist_Gen"].values) ** 2), decimals=3 )
# err_list = dev_set[["rmse_pressure_preset_pid", "rmse_pressure_model_pid", "rmse_distance_preset_pid", "rmse_distance_model_pid"]].mean().values
#
#
# print(
#     f"errore pressione preset: \t{err_list[0]}\n"
#     f"errore pressione modello:\t{err_list[1]}\n"
#     f"errore distanza preset:  \t{err_list[2]}\n"
#     f"errore distanza modello: \t{err_list[3]}\n"
# )


