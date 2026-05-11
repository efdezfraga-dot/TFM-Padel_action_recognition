import cv2
import numpy as np
import os
from pathlib import Path

# Carpeta de entrada y salida
input_folder = Path("C:/Users/efdez/Desktop/VIDEO_RGB/voleaD")
output_folder = input_folder.parent / "VIDEO_RGB_ROTATED_voleaD"
output_folder.mkdir(exist_ok=True)

# Obtener todos los archivos de video (puedes ajustar la extensión si es necesario)
video_files = list(input_folder.glob("*.mp4"))  # Cambia a *.avi o *.mov si es necesario

for video_path in video_files:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"No se pudo abrir el video: {video_path}")
        continue

    # Obtener propiedades del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Cambiar codec si es necesario

    # Nombre del archivo de salida
    output_path = output_folder / f"{video_path.stem}_rotated.mp4"
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Rotación aleatoria para todo el video
    angle = np.random.uniform(-15, 15)
    print(f"Rotando '{video_path.name}' con ángulo de {angle:.2f}°")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Matriz de rotación
        M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        rotated_frame = cv2.warpAffine(frame, M, (width, height))

        out.write(rotated_frame)

    cap.release()
    out.release()

print("Procesamiento completado.")
