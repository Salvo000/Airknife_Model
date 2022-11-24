import os

import numpy as np
import pandas as pd
df1 = pd.read_excel("prep_training.xlsx")
df2 = pd.read_excel("prep_validation_opt.xlsx")
df3 = pd.read_excel("prep_validation.xlsx")
df4 = pd.read_excel("prep_testing.xlsx")

#some preprocessing for opt
df2 = df2[df2["training"]==False]
df2.pop("Rotolo")
df2.pop("training")
df2.pop("Pressione_Preset")
df2.pop("Distanza_Preset")
df2.pop("Riv_Misurato_Preset")
df2.pop("Rivestimento_Target")
df2.rename(columns = {'Pressione_PID':'Pressione', "Distanza_PID":"Distanza","Rivestimento_PID":"Riv_Misurato"}, inplace = True)
df2 = df2[["Temperatura","Altezza","Velocità","Riv_Misurato","Pressione","Distanza"]]
dataset = pd.DataFrame( np.concatenate([df1,df2,df3,df4],axis=0), columns=df2.columns )

#ci sono solo 3 elementi che superano i 146
dataset = dataset[dataset["Riv_Misurato"]<146.]

#from 40 to 216.35
#ogni 8 grammi aggiungo uno valore alla classe.
#perchè (40-216)/25(num classi che voglio)=7.0 grammi.

#parto con i dati ordinati e con num_classe=0 e th=val_min+7 grammi.
#se il valore che incontro supera, allora sommo +1 al valore della classe e

dataset.sort_values(by=['Riv_Misurato'], inplace=True)
th = dataset["Riv_Misurato"].min() + 8
id_class = 0
classes = []
for riv in dataset["Riv_Misurato"].values:
    if riv > th:
        id_class+=1
        th+=7.
    classes.append(id_class)

classes = np.expand_dims(classes, axis=1)

print("Classes frequencies")
num,freq = np.unique( classes, return_counts=True)
print(np.asarray((num, freq)).T)


from sklearn.model_selection import train_test_split
x_data, x_test, classes_data, classes_test = train_test_split(
                        dataset.values, classes,
                        stratify=classes,
                        random_state=42, test_size=0.20)

print( f"dim test: {x_test.shape}" )
print( f"dim data before split: {x_data.shape}" )

x_train, x_val, y_train, y_val = train_test_split(
                        x_data, classes_data,
                        stratify=classes_data, random_state=42, test_size=0.25)

print( f"dim training: {x_train.shape}" )
print( f"dim validation: {x_val.shape}" )


def write_in_excel(nparray,filename):
    df = pd.DataFrame(nparray, columns=df1.columns, )
    df.to_excel(f"{filename}",index=False)

write_in_excel(x_test,"shuffled_test.xlsx")
write_in_excel(x_val,"shuffled_val.xlsx")
write_in_excel(x_train,"shuffled_train.xlsx")




