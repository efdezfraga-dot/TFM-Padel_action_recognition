import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import load_model, Model
from keras.layers import Input, Dense
import os
import argparse
from data_utils import DataSet
from utils import Params

# Importamos las capas customizadas de tu script de entrenamiento
from transformer_train import PositionalEmbedding, TransformerEncoder

# 1. Definir las etiquetas de tus clases
class_names = ['bandeja', 'derecha', 'remate', 'reves', 'salidaD', 'salidaR', 'vibora', 'voleaD', 'voleaR']

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/tcn_model', help="Directorio del modelo")
# Aquí pon el nombre exacto del archivo .keras que se haya guardado en tu carpeta checkpoints
parser.add_argument('--checkpoint', default='checkpoints/transformer_weights.epoch_0100.keras', help="Archivo de pesos")

args = parser.parse_args()

def evaluate():
    json_path = os.path.join(args.model_dir, 'params.json')
    params = Params(json_path)

    # 1. Crear el modelo Dummy para que data_utils no falle al buscar las características
    a = Input(shape=(1,))
    b = Dense(1)(a)
    cnn_model = Model(inputs=a, outputs=b)

    # 2. Cargar los datos de Test
    print("\nCargando datos de Test...")
    # Asegúrate de usar el mismo seq_length con el que extrajiste las características
    seq_length = getattr(params, 'seq_length', 150) 
    dataset = DataSet(cnn_model, seq_length)
    
    # Cambia 'test' por 'validation' si tu data_utils no tiene un conjunto de test explícito
    try:
        X_test, y_test = dataset.generate_data('test')
    except:
        print("No se encontró conjunto 'test', usando 'validation' para la evaluación...")
        X_test, y_test = dataset.generate_data('validation')

    print(f"Forma de X_test: {X_test.shape}")

    # 3. Cargar el modelo entrenado con las Custom Objects
    model_path = os.path.join(args.model_dir, args.checkpoint)
    print(f"\nCargando modelo desde: {model_path}")
    
    modelo_cargado = load_model(model_path, custom_objects={
        'PositionalEmbedding': PositionalEmbedding,
        'TransformerEncoder': TransformerEncoder
    })

    # 4. Hacer predicciones
    print("\nGenerando predicciones...")
    y_pred_prob = modelo_cargado.predict(X_test)
    
    y_pred_classes = np.argmax(y_pred_prob, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # 5. Imprimir el Reporte de Clasificación
    print("\n" + "="*50)
    print(" REPORTE DE CLASIFICACIÓN - TRANSFORMER")
    print("="*50)
    print(classification_report(y_true_classes, y_pred_classes, target_names=class_names))
    
    # 6. Dibujar y guardar la Matriz de Confusión
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='rocket', 
                xticklabels=class_names, yticklabels=class_names,
                cbar=True, square=True)
    
    plt.title('Matriz de Confusión - Transformer Padel', fontsize=16)
    plt.ylabel('Etiqueta Real (True Label)', fontsize=12)
    plt.xlabel('Etiqueta Predicha (Predicted Label)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Guarda la imagen en la carpeta del modelo
    save_fig_path = os.path.join(args.model_dir, 'matriz_confusion_transformer.png')
    plt.savefig(save_fig_path, dpi=300)
    print(f"\nMatriz de confusión guardada en: {save_fig_path}")
    
    # Mostrar la matriz en pantalla
    plt.show()

if __name__ == '__main__':
    evaluate()