import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import Model, layers, initializers, optimizers, losses, callbacks, regularizers
from code.coating_net import load_coating_net
from code.utils import weights_path, input_data_path


def create_controller_model():
    type = tf.float64
    init = initializers.RandomUniform(minval=0.045, maxval=0.065, seed=42 )
    reg = regularizers.l2(0.00001)

    inputs = layers.Input(shape=([4]), dtype=type)
    out = layers.Dense(32, activation="swish", kernel_initializer=init, bias_initializer="ones", kernel_regularizer=reg, dtype=type)(inputs)
    out = layers.Dense(32, activation="swish", kernel_initializer=init, bias_initializer="ones", kernel_regularizer=reg, dtype=type)(out)
    out = layers.Dense(2,  activation="swish", kernel_initializer=init, bias_initializer="ones", dtype=type )(out)

    speed = tf.exp( tf.gather(inputs, 2, axis=1) )

    threshold = 125.
    x = tf.convert_to_tensor( 250., dtype=tf.float64)
    y = tf.convert_to_tensor( 160., dtype=tf.float64)

    p_min = tf.where( speed >= threshold, x, y )
    d_min = tf.fill( tf.shape( speed ) , 7.5 )

    log_d_min = tf.expand_dims( tf.math.log( d_min ), axis=1 )
    log_p_min = tf.expand_dims( tf.math.log( p_min ), axis=1 )

    min_out = layers.concatenate([log_p_min,log_d_min], axis=1 , dtype=type )

    out = layers.Maximum( dtype=type )([out, min_out])

    return Model(inputs,out)


def get_loss(alpha,gamma):

    d_min = tf.convert_to_tensor( 7.5 , dtype=tf.float64 )
    log_d_min= tf.math.log( d_min )

    def loss(y_true,y_preds):

        mse = lambda x, y: tf.reduce_mean( (tf.exp(x) - tf.exp(y)) ** 2 )
        one_on_t, log_h, log_v, log_r_trues, log_p, log_d = tf.split( y_true, 6, axis=1)

        log_p_gen, log_d_gen  = tf.split( y_preds, 2, axis=1)

        p_min = tf.where( tf.exp( log_v ) >= 125. ,
                    tf.convert_to_tensor( 250. , dtype=tf.float64),
                    tf.convert_to_tensor( 160. , dtype=tf.float64)
            )

        log_p_min = tf.math.log( p_min )

        fivefold = tf.concat([one_on_t, log_h, log_v, log_p_gen, log_d_gen], axis=1 ) # i valori sono tutti log eccetto t^-1
        log_r_preds =  tf.cast( cnet_model( fivefold ), dtype=tf.float64 )

        coat_loss = mse( log_r_trues, log_r_preds )
        dist_loss = alpha * tf.sqrt( mse( log_d_gen, log_d_min) )
        pres_loss = gamma * tf.sqrt( mse( log_p_gen, log_p_min) )

        return coat_loss + dist_loss + pres_loss

    return loss


def get_metrics():

    def mse_coating(y_true,y_preds):
        y_true = tf.cast( y_true, dtype=tf.float64)
        y_preds = tf.cast(y_preds, dtype=tf.float64)

        mse = lambda x, y: tf.reduce_mean((tf.exp(x) - tf.exp(y)) ** 2 )
        one_on_t, log_h, log_v, log_r_trues,log_p, log_d = tf.split( y_true, 6, axis=1)

        log_p_gen, log_d_gen  = tf.split( y_preds, 2, axis=1)

        fivefold = tf.concat([one_on_t, log_h, log_v, log_p_gen, log_d_gen], axis=1)# i valori sono tutti log eccetto t^-1
        log_r_preds =  tf.cast( cnet_model( fivefold ), dtype=tf.float64 )

        coat_loss = mse( log_r_trues, log_r_preds )

        return coat_loss

    return mse_coating

#optional to use
def preprocessing_opt(data):
    log_p = np.log(np.expand_dims(data.pop("Pressione_PID").values, axis=1))
    log_d = np.log(np.expand_dims(data.pop("Distanza_PID").values, axis=1))

    data["Temperatura"] = (data["Temperatura"] + 273.) ** (-1)
    one_on_temp = np.expand_dims(data.pop("Temperatura").values, axis=1)

    log_x_data = np.log(data[["Altezza", "Velocità", "Rivestimento_Target"]].values)
    x = np.concatenate([one_on_temp, log_x_data], axis=1)
    y = np.concatenate([one_on_temp, log_x_data, log_p, log_d], axis=1)

    x = tf.cast(x, dtype=tf.float64)
    y = tf.cast(y, dtype=tf.float64)

    # x:<(1/t),log_h,log_vlog_r>
    # y:<(1/t),log_h,log_v,log_r, log_p, log_d>
    return x, y

def preprocessing(data):
    log_p = np.log(np.expand_dims(data.pop("Pressione").values, axis=1))
    log_d = np.log(np.expand_dims(data.pop("Distanza").values, axis=1))

    data["Temperatura"] = data["Temperatura"] + 273.
    temp = data.pop("Temperatura").values
    temp = np.expand_dims(temp, axis=1)
    temp = temp ** (-1)

    x_data = np.log(data.values)
    x = np.concatenate([temp, x_data], axis=1)
    y = np.concatenate([temp, x_data, log_p, log_d], axis=1)

    x = tf.cast(x, dtype=tf.float64)
    y = tf.cast(y, dtype=tf.float64)

    # x:<(1/t),log_h,log_v,log_r>
    # y:<(1/t),log_h,log_v,log_r, log_p, log_d>
    return x, y

def load_controller_net():
    model = create_controller_model()
    model.load_weights(weights_path + MODEL_NAME)
    return model


def get_riv_from_tuple(cnet_model, log_p_gen, log_d_gen, set):
    one_on_t = np.expand_dims((set["Temperatura"].values + 273.) ** (-1), axis=1)
    log_h = np.expand_dims(np.log(set["Altezza"].values), axis=1)
    log_v = np.expand_dims(np.log(set["Velocità"].values), axis=1)
    fivefold = np.concatenate([one_on_t, log_h, log_v, log_p_gen, log_d_gen], axis=1)
    return cnet_model(fivefold)


MODEL_NAME="controller.h5"
if __name__ == '__main__':
    pd.options.display.max_columns = None

    load_weights = True
    do_training = False
    print_training_history = False

    do_validation = True
    do_testing = True

    EPOCHS = 2000
    BATCH_SIZE = 512
    LR = 0.0001
    alpha_val = (0.75)*7 #dist
    gamma_val = 0.3      #pres #da alcune analisi, il peso della pressione sembra troppo basso..

    cnet_model = load_coating_net()

    model = create_controller_model()

    if load_weights:
        model.load_weights(weights_path + MODEL_NAME)


    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR),
        loss=get_loss(alpha=alpha_val, gamma=gamma_val),
        metrics=get_metrics()
    )


    if do_training:
        training = pd.read_excel(input_data_path + "shuffled_train.xlsx")
        validation = pd.read_excel(input_data_path + "shuffled_val.xlsx")

        x, y = preprocessing( data=training)
        x_val, y_val = preprocessing(data=validation.copy())

        print(f"training size (80%): x:{x.shape}, y: {y.shape}")
        print(f"validation size (80%): x:{x_val.shape}, y: {y_val.shape}")

        # early_stopping = callbacks.EarlyStopping(monitor='val_loss', mode="min", patience=10, restore_best_weights=True)
        history = model.fit(x=x, y=y, validation_data=(x_val,y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, validation_batch_size=BATCH_SIZE)#, callbacks=early_stopping )
        model.save_weights(weights_path+MODEL_NAME)

        if print_training_history:
            print(history.history.keys())
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper right')
            plt.show()

    if do_validation:
        if do_training is False:
            validation = pd.read_excel(input_data_path + "shuffled_val.xlsx")
            x_val, y_val = preprocessing(data=validation.copy())

        print( "Doing Validation ")

        res = model.predict(x_val)
        log_p_gen, log_d_gen = np.split(res, 2, axis=1)
        validation["Pres_Gen"] = np.exp(log_p_gen)
        validation["Dist_Gen"] = np.exp(log_d_gen)
        validation["Riv_Pred"] = np.exp( get_riv_from_tuple(cnet_model,log_p_gen, log_d_gen, validation) )
        print( validation )
        # validation.to_excel(data_output_path + "training_output.xlsx", index=False)

    if do_testing:
        print("Do testing")

        testing = pd.read_excel(input_data_path + "shuffled_test.xlsx")
        x,y = preprocessing(data=testing.copy())
        res = model.predict(x)
        log_p_gen, log_d_gen = np.split(res, 2, axis=1)

        testing["Pres_Gen"] = np.exp(log_p_gen)
        testing["Dist_Gen"] = np.exp(log_d_gen)
        testing["Riv_Pred"] = np.exp( get_riv_from_tuple(cnet_model,log_p_gen, log_d_gen, testing) )

        print(testing)





