'''
Adopted from
https://www.kaggle.com/competitions/icr-identify-age-related-conditions/discussion/430843

Further reading:
https://towardsdatascience.com/how-to-write-a-custom-keras-model-so-that-it-can-be-deployed-for-serving-7d81ace4a1f8
'''

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, train_test_split

import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, ProgbarLogger
from tensorflow.keras import regularizers as R
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras import layers as L
from tensorflow.keras import optimizers as O
from tensorflow.keras import constraints as C
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy, sparse_categorical_crossentropy, Loss
from tensorflow.keras.metrics import AUC
import tensorflow_addons as tfa
from tensorflow_addons.metrics import F1Score
tf.keras.utils.set_random_seed(722)

def balanced_log_loss_np(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1-1e-15)
    nc = np.bincount(y_true)
    balanced_log_loss_score = (-1/nc[0]*(np.sum(np.where(y_true==0,1,0) * np.log(1-y_pred))) - 1/nc[1]*(np.sum(np.where(y_true!=0,1,0) * np.log(y_pred)))) / 2
    return balanced_log_loss_score



try:
    @tf.keras.utils.register_keras_serializable()
    def smish(x):
        return x * K.tanh(K.log(1 + K.sigmoid(x)))


    @tf.keras.utils.register_keras_serializable()
    class GatedLinearUnit(L.Layer):
        def __init__(self, units, **kwargs):
            super().__init__(**kwargs)
            self.linear = L.Dense(units)
            self.sigmoid = L.Dense(units, activation="sigmoid")
            self.units = units

        def get_config(self):
            config = super().get_config()
            config['units'] = self.units
            return config

        def call(self, inputs):
            return self.linear(inputs) * self.sigmoid(inputs)


    @tf.keras.utils.register_keras_serializable()
    class GatedResidualNetwork(L.Layer):
        def __init__(self, units, dropout_rate, **kwargs):
            super().__init__(**kwargs)
            self.units = units
            self.dropout_rate = dropout_rate
            self.relu_dense = L.Dense(units, activation=smish)
            self.linear_dense = L.Dense(units)
            self.dropout = L.Dropout(dropout_rate)
            self.gated_linear_unit = GatedLinearUnit(units)
            self.layer_norm = L.LayerNormalization()
            self.project = L.Dense(units)

        def get_config(self):
            config = super().get_config()
            config['units'] = self.units
            config['dropout_rate'] = self.dropout_rate
            return config

        def call(self, inputs):
            x = self.relu_dense(inputs)
            x = self.linear_dense(x)
            x = self.dropout(x)
            if inputs.shape[-1] != self.units:
                inputs = self.project(inputs)
            x = inputs + self.gated_linear_unit(x)
            x = self.layer_norm(x)
            return x


    @tf.keras.utils.register_keras_serializable()
    class VariableSelection(L.Layer):
        def __init__(self, num_features, units, dropout_rate, **kwargs):
            super().__init__(**kwargs)
            self.grns = list()
            # Create a GRN for each feature independently
            for idx in range(num_features):
                grn = GatedResidualNetwork(units, dropout_rate)
                self.grns.append(grn)
            # Create a GRN for the concatenation of all the features
            self.grn_concat = GatedResidualNetwork(units, dropout_rate)
            self.softmax = L.Dense(units=num_features, activation="softmax")
            self.num_features = num_features
            self.units = units
            self.dropout_rate = dropout_rate

        def get_config(self):
            config = super().get_config()
            config['num_features'] = self.num_features
            config['units'] = self.units
            config['dropout_rate'] = self.dropout_rate
            return config

        def call(self, inputs):
            v = L.concatenate(inputs)
            v = self.grn_concat(v)
            v = tf.expand_dims(self.softmax(v), axis=-1)

            x = []
            for idx, input_ in enumerate(inputs):
                x.append(self.grns[idx](input_))
            x = tf.stack(x, axis=1)

            outputs = tf.squeeze(tf.matmul(v, x, transpose_a=True), axis=1)
            return outputs


    @tf.keras.utils.register_keras_serializable()
    class VariableSelectionFlow(L.Layer):
        def __init__(self, num_features, units, dropout_rate, dense_units=None, **kwargs):
            super().__init__(**kwargs)
            self.variableselection = VariableSelection(num_features, units, dropout_rate)
            self.split = L.Lambda(lambda t: tf.split(t, num_features, axis=-1))
            self.dense = dense_units
            if dense_units:
                self.dense_list = [L.Dense(dense_units, \
                                           activation='linear') \
                                   for _ in tf.range(num_features)
                                  ]
            self.num_features = num_features
            self.units = units
            self.dropout_rate = dropout_rate
            self.dense_units = dense_units

        def get_config(self):
            config = super().get_config()
            config['num_features'] = self.num_features
            config['units'] = self.units
            config['dropout_rate'] = self.dropout_rate
            config['dense_units'] = self.dense_units
            return config        

        def call(self, inputs):
            split_input = self.split(inputs)
            if self.dense:
                l = [self.dense_list[i](split_input[i]) for i in range(len(self.dense_list))]
            else:
                l = split_input
            return self.variableselection(l)        
except:
    pass

blls = []

batch_size = 64

units_1 = 32
drop_1 = 0.75
dense_units = 8

units_2 = 16
drop_2 = 0.5

units_3 = 8
drop_3 = 0.25

first_run=1

NFOLDS=10
cv = StratifiedKFold(n_splits = NFOLDS, shuffle=True, random_state=722)
for n, (train_idx, val_idx) in enumerate(cv.split(X, (tgt + 1) * (tgt2 - 3))):
    for k in range( NFOLDS ):
        print(f'______fold {n+1}______, ________repeat {k+1}__________')

        inputs_1 = tf.keras.Input(shape=(56,))
        
        features_1 = VariableSelectionFlow(56, units_1, drop_1, dense_units=dense_units)(inputs_1)
        features_2 = VariableSelectionFlow(units_1, units_2, drop_2)(features_1)         
        features_3 = VariableSelectionFlow(units_2, units_3, drop_3)(features_2)         

        outputs = L.Dense(1, activation="sigmoid")(features_3)

        model = Model(inputs=inputs_1, outputs=outputs)      

        if first_run:
            first_run=0; print( model.summary() )
        
        opt = O.Adam(1e-3, epsilon=1e-7)
        loss = binary_crossentropy
        
        saver=tf.keras.callbacks.ModelCheckpoint(
            '../working/{epoch:02d}.hd5f',
            monitor = "val_loss",
            verbose = 1,
            save_best_only= False,
            save_weights_only= False,
            mode = "auto",
            save_freq="epoch",
            options=None,
            initial_value_threshold=None,
        )

        lr = ReduceLROnPlateau(monitor="val_loss", mode='min', factor=0.95, patience=1, verbose=1)
        es = EarlyStopping(monitor='val_loss', mode='min', patience=25, verbose=1, restore_best_weights=True)

        model.compile(optimizer=opt, 
                        loss=loss
                     )

        history = model.fit(x=X[train_idx],
                          y=tgt[train_idx],
                          batch_size=batch_size,
                          epochs=200,
                          validation_data=(X[val_idx], tgt[val_idx]),
                              callbacks=[lr,es,saver]
                )
                
        probs = model.predict( X[val_idx] )[:,0]
        bll = balanced_log_loss_np(train_df.Class.values[val_idx], probs)
        blls.append(bll)
        val_loss = np.asarray(history.history['val_loss'])
        train_loss = np.asarray(history.history['loss'])
        min_val_loss = val_loss.min()
        min_train_loss = train_loss[val_loss.argmin()]
        print(f'{min_train_loss:.4f}, {min_val_loss:.4f}, {bll:.4f}')  
        
        model.save_weights(f'final_model_f{n}_r{k}_tr{min_train_loss:.4f}_val{min_val_loss:.4f}.h5')
        
print(np.mean(blls))
