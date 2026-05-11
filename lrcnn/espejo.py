import os
import cv2

def mirror_videos(input_path):
    # Listar todos los archivos en el directorio
    for filename in os.listdir(input_path):
        # Verificar si es un archivo de video
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
            file_path = os.path.join(input_path, filename)
            
            # Crear nombre para el video espejado
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_flipped{ext}"
            output_path = os.path.join(input_path, output_filename)
            
            # Capturar el video original
            cap = cv2.VideoCapture(file_path)
            
            # Obtener propiedades del video
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            
            # Crear VideoWriter para el video espejado
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Procesar cada frame
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Espejar el frame horizontalmente
                flipped_frame = cv2.flip(frame, 1)
                
                # Escribir el frame espejado
                out.write(flipped_frame)
            
            # Liberar recursos
            cap.release()
            out.release()
            print(f"Video espejado creado: {output_filename}")

if __name__ == "__main__":
    video_path = r"C:/Users/efdez/Desktop/VIDEO_RGB/remate" \
    ""  # Cambia esta ruta al directorio de tus videos
    mirror_videos(video_path)
    print("Proceso completado!")