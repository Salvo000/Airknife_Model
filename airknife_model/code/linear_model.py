import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model, layers, initializers, optimizers, losses, callbacks, regularizers
from code.utils import input_data_path, weights_path, output_data_path,preprocessing,preprocessing_opt,load_join_shuffle_validation_set

def create_linear_model():
    init = initializers.GlorotNormal(seed=42)
    inputs = layers.Input([5,], dtype=tf.float32 )
    outputs = layers.Dense(1, use_bias=True, kernel_initializer=init, dtype=tf.float32 )(inputs)
    return Model( inputs, outputs)


def coat_loss( y_true, y_preds):
    return tf.reduce_mean( (y_true - tf.math.exp( y_preds ) )**2 , axis=-1)


MODEL_NAME = "lm.h5"
if __name__ == '__main__':
    pd.options.display.max_columns = None
    load_weights = True
    do_training = False
    do_validation=True
    do_testing = True

    #training parameters
    EPOCHS = 3500 #3500 epoche, poi procedi piano.
    BATCH_SIZE = 1024
    LR = 0.001

    lm = create_linear_model()

    if load_weights:
        lm.load_weights(weights_path+MODEL_NAME)

    lm.compile( optimizer=optimizers.Adam(learning_rate=LR), loss=coat_loss )

    if do_training:
        training = pd.read_excel(input_data_path + "shuffled_train.xlsx")
        x, y = preprocessing(training)
        validation = pd.read_excel( input_data_path+"shuffled_val.xlsx")
        x_val,y_val = preprocessing(validation)

        # early = callbacks.EarlyStopping( patience=15 )
        lm.fit(x=x,y=y,shuffle=True, validation_data=(x_val, y_val), epochs=EPOCHS, validation_batch_size=BATCH_SIZE,  batch_size=BATCH_SIZE)#, callbacks=early )
        lm.save_weights(filepath=weights_path+MODEL_NAME)

    if do_validation:
        print("Validation output\n")

        if do_training is False:
            validation = pd.read_excel(input_data_path + "shuffled_val.xlsx")
            x_val, y_val = preprocessing(df=validation.copy())

        print(f"Coating loss is {np.round(np.sqrt(lm.evaluate(x_val, y_val)), 3)} g/m^2 ")
        validation["Riv_Predetto"] = np.exp( lm.predict(x_val))
        print(validation)

    if do_testing:
        print("Testing output\n")
        test = pd.read_excel( input_data_path+"shuffled_test.xlsx")
        x,y = preprocessing(df=test.copy())
        print(f"Testing: Coating loss is {np.round(np.sqrt(lm.evaluate(x, y)), 3)} g/m^2 for face")
        test["Riv_Predetto"] =np.exp( lm.predict(x))
        print( test )

