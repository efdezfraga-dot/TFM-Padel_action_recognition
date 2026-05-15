"""
Train the Video Transformer model using pre-extracted features (Dummy Model).
"""

import tensorflow as tf 
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers.schedules import ExponentialDecay
from keras.optimizers import Adam
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Embedding, Masking
from data_utils import DataSet

import numpy as np
import os
import argparse
import pickle

# Importar la configuración de parámetros personalizada
from utils import Params

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/transformer_model',
                    help="Directorio que contiene el archivo params.json")

# --- CAPAS PERSONALIZADAS (Custom Layers) ---

@tf.keras.utils.register_keras_serializable()
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.output_dim = output_dim
        self.position_embeddings = Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )

    def call(self, inputs):
        length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_positions = self.position_embeddings(positions)
        return inputs + embedded_positions

    def get_config(self):
        config = super().get_config()
        config.update({"sequence_length": self.sequence_length, "output_dim": self.output_dim})
        return config

@tf.keras.utils.register_keras_serializable()
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = Sequential([
            Dense(dense_dim, activation="gelu"),
            Dropout(dropout_rate),
            Dense(embed_dim),
        ])
        self.layernorm_1 = LayerNormalization()
        self.layernorm_2 = LayerNormalization()
        self.dropout_1 = Dropout(dropout_rate)

    def call(self, inputs, training=False):
        # Pre-norm para mayor estabilidad
        x = self.layernorm_1(inputs)
        attention_output = self.attention(x, x)
        x = inputs + self.dropout_1(attention_output, training=training)
        
        y = self.layernorm_2(x)
        y = self.dense_proj(y)
        return x + y

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate
        })
        return config

# --- CONSTRUCTOR DEL MODELO ---

def build_transformer(num_features=2048, embed_dim=512, dense_dim=1024, num_heads=8, num_layers=4, dropout_rate=0.4, seq_length=150, num_classes=9):
    
    inputs = Input(shape=(seq_length, num_features))
    
    # 0. Masking para ignorar el padding de ceros
    x = Masking(mask_value=0.0)(inputs)
    
    # 1. Proyección de rasgos
    x = Dense(embed_dim, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    
    # 2. Embedding posicional
    x = PositionalEmbedding(sequence_length=seq_length, output_dim=embed_dim)(x)
    
    # 3. Bloques Transformer apilados
    for _ in range(num_layers):
        x = TransformerEncoder(embed_dim=embed_dim, dense_dim=dense_dim, num_heads=num_heads, dropout_rate=dropout_rate)(x)
    
    # 4. Pooling Global
    x = GlobalAveragePooling1D()(x)
    
    # 5. Clasificador Final
    x = Dense(256, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs, outputs)
    return model

# --- FUNCIÓN DE ENTRENAMIENTO ---

def train(model_dir, cnn_model, saved_model=None, 
          learning_rate=1e-4, train_size=0.8, seq_length=150,
          embed_dim=512, dense_dim=1024, num_heads=8, num_layers=4, 
          dropout_rate=0.4, num_classes=9, batch_size=16, nb_epoch=100):

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
        filepath=os.path.join(checkpoints_dir, 'transformer_weights.epoch_{epoch:04d}.keras'),
        verbose=1, save_best_only=False, save_freq='epoch') 
    
    tb = TensorBoard(log_dir=model_dir)

    # Cargar o construir modelo
    if saved_model:
        transformer_model = load_model(saved_model, custom_objects={
            'PositionalEmbedding': PositionalEmbedding, 'TransformerEncoder': TransformerEncoder
        })
    else:
        transformer_model = build_transformer(
            num_features=2048, embed_dim=embed_dim, dense_dim=dense_dim, 
            num_heads=num_heads, num_layers=num_layers, dropout_rate=dropout_rate,
            seq_length=seq_length, num_classes=num_classes
        )
    
    lr_schedule = ExponentialDecay(initial_learning_rate=learning_rate, decay_steps=10000, decay_rate=0.9)
    optimizer = Adam(learning_rate=lr_schedule)
    
    transformer_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])
    print(transformer_model.summary())

    history = transformer_model.fit(
        generator,
        steps_per_epoch=steps_per_epoch,
        epochs=nb_epoch,
        verbose=1,
        callbacks=[tb, checkpointer],
        validation_data=val_generator,
        validation_steps=steps_per_epoch_val
    )

    return history

if __name__ == '__main__':
    args = parser.parse_args()
    params = Params(os.path.join(args.model_dir, 'params.json'))

    # --- DUMMY MODEL PARA EVITAR EXTRACCIÓN EN VIVO ---
    a = Input(shape=(1,))
    b = Dense(1)(a)
    cnn_model = Model(inputs=a, outputs=b)

    # Ejecutar Entrenamiento
    train(args.model_dir, cnn_model, 
          saved_model=None if params.saved_model == "None" else params.saved_model,
          learning_rate=params.learning_rate, train_size=params.train_size, 
          seq_length=params.seq_length, embed_dim=params.embed_dim, 
          dense_dim=params.dense_dim, num_heads=params.num_heads, 
          num_layers=getattr(params, 'num_layers', 4),
          dropout_rate=params.dropout_rate, num_classes=params.num_classes,
          batch_size=params.batch_size, nb_epoch=params.nb_epoch)