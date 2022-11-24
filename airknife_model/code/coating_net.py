import os

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import Model, layers, initializers, optimizers, losses, callbacks, regularizers
from code.utils import input_data_path,weights_path, preprocessing_opt,preprocessing,load_join_shuffle_validation_set
from code.linear_model import coat_loss, create_linear_model
from code.linear_model import MODEL_NAME as lm_name


def create_coating_net(linear_model):

    init = initializers.RandomUniform(minval=0.045, maxval=0.065, seed=42 )
    reg = regularizers.l2(0.00001)
    inputs = layers.Input(shape=([5,]))
    out2 = layers.Dense(32, activation="swish", kernel_initializer=init, kernel_regularizer=reg)(inputs)
    out2 = layers.Dense(32, activation="swish", kernel_initializer=init, kernel_regularizer=reg)(out2)
    out2 = layers.Dense(32, activation="swish", kernel_initializer=init, kernel_regularizer=reg)(out2)
    out2 = layers.Dense(1, kernel_initializer=init, kernel_regularizer=reg)(out2)
    #out2 = layers.Lambda(lambda x: x*2.5 )(out2)
    additive_block = keras.Model(inputs,out2)

    out1 = linear_model(inputs)
    out2 = additive_block(inputs)
    added = layers.Add()([out1,out2])
    model = keras.Model(inputs,added)

    return model,additive_block

def generate_coating_model(from_external=False):
    lm = create_linear_model()
    path = weights_path+lm_name
    lm.load_weights(path)
    lm.trainable=False
    return create_coating_net(lm)

def load_coating_net():
    model, _ = generate_coating_model()
    model.load_weights(weights_path + MODEL_NAME)
    return model



MODEL_NAME = "cnet.h5"
if __name__ == '__main__':
    pd.options.display.max_columns = None
    load_weights = True
    do_training = False
    do_validation = True
    do_testing = True

    EPOCHS = 500
    BATCH_SIZE = 512
    LR = 0.001

    model, additive_model = generate_coating_model()

    if load_weights:
        model.load_weights(weights_path + MODEL_NAME)

    model.compile(optimizer=optimizers.Adam(learning_rate=LR), loss=coat_loss)

    if do_training:
        x, y = preprocessing( df=pd.read_excel(input_data_path + "shuffled_train.xlsx"))
        validation = pd.read_excel(input_data_path + "shuffled_val.xlsx")
        x_val, y_val = preprocessing( df=validation.copy() )

        #early_stopping = callbacks.EarlyStopping(monitor='val_loss', mode="min", patience=10, restore_best_weights=True)
        history = model.fit(x=x,y=y,shuffle=True, validation_data=(x_val, y_val), epochs=EPOCHS, validation_batch_size=BATCH_SIZE,  batch_size=BATCH_SIZE)#, callbacks=early )
        model.save_weights(filepath=weights_path+MODEL_NAME)

        print( history.history.keys() )

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()

        print("Additive numbers after training on training set")
        additive_numbers = additive_model.predict(x)
        print(f"mean: {np.exp(np.mean(additive_numbers))} - std: {np.exp(np.std(additive_numbers))}")


    if do_validation:
        if do_training is False:
            validation = pd.read_excel(input_data_path + "shuffled_val.xlsx")
            x_val,y_val = preprocessing( df=validation.copy() )

        print("Doing Validation")
        print(f"Validation: Coating loss is {np.round(np.sqrt(model.evaluate(x_val, y_val)), 3)} g/m^2 for face")

        print("Additive numbers for validation are:\n")
        corrections = additive_model.predict(x_val)
        print(f"mean: {np.exp(np.mean(corrections))} - std: {np.exp(np.std(corrections))}")

    if do_testing:
        test = pd.read_excel( input_data_path+"shuffled_test.xlsx")
        x,y = preprocessing(test.copy())
        print(f"test coating loss is {np.round(np.sqrt(model.evaluate(x, y)), 3)} g/m^2 for face")
        test["Riv_Predetto"] = np.exp(model.predict(x))
        print(test)






