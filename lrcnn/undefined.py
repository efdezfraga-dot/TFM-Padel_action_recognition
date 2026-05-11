# Ejecuta este código para encontrar archivos problemáticos
import numpy as np
import os

folder_path = r"c:/Users/efdez/Desktop/TFG/tennis_action_recognition/lrcnn/data/sequences/voleaR"
bad_files = []

for file_name in os.listdir(folder_path):
    if file_name.endswith(".npy"):
        data = np.load(os.path.join(folder_path, file_name))
        if data.shape != (16, 2048):
            print(f"Archivo corrupto: {file_name} - Forma: {data.shape}")
            bad_files.append(file_name)

print(f"\nArchivos con problemas: {bad_files if bad_files else 'Ninguno'}")