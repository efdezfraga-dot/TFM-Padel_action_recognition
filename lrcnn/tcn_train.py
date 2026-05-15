"""
Train the Temporal Convolutional Network (TCN) model. 
Parameters are set in params.json file in the relevant directory under "/experiments".
"""

import tensorflow as tf 
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers.schedules import ExponentialDecay
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Conv1D, GlobalAveragePooling1D, Masking, Add, Activation, BatchNormalization
from data_utils import DataSet
import glob
from keras.callbacks import Callback

import numpy as np
import os
import argparse
import pickle
from utils import Params

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/tcn_model',
                    help="Directory containing params.json")

# --- BLOQUE TCN PERSONALIZADO ---
def tcn_block(x, filters, kernel_size, dilation_rate, dropout_rate):
    """
    Construye un bloque residual TCN con convoluciones dilatadas causales.
    """
    # Primera convolución 1D
    conv1 = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, 
                   padding='causal', activation='relu')(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(dropout_rate)(conv1)

    # Segunda convolución 1D
    conv2 = Conv1D(filters=filters, kernel_size=kernel_size, dilation_rate=dilation_rate, 
                   padding='causal', activation='relu')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Dropout(dropout_rate)(conv2)

    # Conexión Residual (Si el número de filtros cambia, proyectamos la entrada)
    if x.shape[-1] != filters:
        x = Conv1D(filters=filters, kernel_size=1, padding='same')(x)

    # Sumamos la entrada original con la salida de las convoluciones
    res = Add()([x, conv2])
    return Activation('relu')(res)

class KeepLastNCheckpoints(Callback):
    def __init__(self, directory, keep=15):
        super().__init__()
        self.directory = directory
        self.keep = keep

    def on_epoch_end(self, epoch, logs=None):
        # Busca todos los archivos .keras en la carpeta de checkpoints
        archivos = glob.glob(os.path.join(self.directory, '*.keras'))
        
        # Ordena los archivos por fecha de modificación (el más viejo primero)
        archivos.sort(key=os.path.getmtime)
        
        # Si hay más archivos del límite permitido, borramos los viejos
        if len(archivos) > self.keep:
            para_borrar = archivos[:-self.keep] # Seleccionamos los sobrantes
            for archivo in para_borrar:
                try:
                    os.remove(archivo)
                except OSError as e:
                    print(f"\n⚠️ No se pudo borrar {archivo}: {e}")

# --- CONSTRUCTOR DEL MODELO TCN ---
def build_tcn(num_features=2048, num_filters=128, kernel_size=3, dropout_rate=0.3, seq_length=150, num_classes=9):
    inputs = Input(shape=(seq_length, num_features))
    
    # 0. Masking para ignorar el relleno (padding)
    x = Masking(mask_value=0.0)(inputs)
    
    # 1. Reducimos dimensiones iniciales para no saturar la RAM (de 2048 a 256)
    x = Dense(256, activation='relu')(x)
    
    # 2. Apilamos bloques TCN aumentando la dilatación exponencialmente
    # Dilataciones: 1, 2, 4, 8, 16. Esto permite al modelo ver un "campo receptivo" muy amplio.
    dilations = [1, 2, 4, 8, 16]
    for dilation in dilations:
        x = tcn_block(x, filters=num_filters, kernel_size=kernel_size, 
                      dilation_rate=dilation, dropout_rate=dropout_rate)
    
    # 3. Agrupación temporal (Pooling)
    x = GlobalAveragePooling1D()(x)
    
    # 4. Clasificador Final
    x = Dense(128, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs, outputs)
    return model

# --- FUNCIÓN DE ENTRENAMIENTO ---
def train(model_dir, cnn_model, saved_model=None, 
          learning_rate=1e-4, train_size=0.8, seq_length=150,
          num_filters=128, kernel_size=3, dropout_rate=0.3, 
          num_classes=9, batch_size=16, nb_epoch=100):

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    dataset = DataSet(cnn_model, seq_length)
    
    # Generadores
    generator = dataset.frame_generator(batch_size, 'train')
    val_generator = dataset.frame_generator(batch_size, 'validation')
    
    steps_per_epoch = int((len(dataset.data) * train_size) // batch_size)
    val_size = len(dataset.data) * (1 - train_size)
    steps_per_epoch_val = max(1, int(val_size // batch_size))

    checkpointer = ModelCheckpoint(
        filepath=os.path.join(checkpoints_dir, 'tcn_weights.epoch_{epoch:04d}.keras'),
        verbose=1, save_best_only=False, save_freq='epoch') 
    
    cleaner = KeepLastNCheckpoints(checkpoints_dir, keep=15)
    tb = TensorBoard(log_dir=model_dir)

    # Cargar o construir modelo
    if saved_model:
        tcn_model = load_model(saved_model) # TCN no necesita custom_objects si usamos la API funcional
    else:
        tcn_model = build_tcn(
            num_features=2048, num_filters=num_filters, kernel_size=kernel_size, 
            dropout_rate=dropout_rate, seq_length=seq_length, num_classes=num_classes
        )
    
    lr_schedule = ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=10000, decay_rate=0.9)
    optimizer = Adam(learning_rate=lr_schedule)
    
    tcn_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])
    print(tcn_model.summary())

    history = tcn_model.fit(
        generator,
        steps_per_epoch=steps_per_epoch,
        epochs=nb_epoch,
        verbose=1,
        callbacks=[tb, checkpointer, cleaner],
        validation_data=val_generator,
        validation_steps=steps_per_epoch_val
    )

    return history

if __name__ == '__main__':
    args = parser.parse_args()
    params = Params(os.path.join(args.model_dir, 'params.json'))

    # --- DUMMY MODEL PARA LEER DEL DISCO DURO (.npy) ---
    a = Input(shape=(1,))
    b = Dense(1)(a)
    cnn_model = Model(inputs=a, outputs=b)

    # Ejecutar Entrenamiento
    train(args.model_dir, cnn_model, 
          saved_model=None if params.saved_model == "None" else params.saved_model,
          learning_rate=params.learning_rate, train_size=params.train_size, 
          seq_length=params.seq_length, num_filters=getattr(params, 'num_filters', 128),
          kernel_size=getattr(params, 'kernel_size', 3), dropout_rate=params.dropout_rate, 
          num_classes=params.num_classes, batch_size=params.batch_size, nb_epoch=params.nb_epoch)