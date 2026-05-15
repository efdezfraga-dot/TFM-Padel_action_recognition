import numpy as np

# 1. Pon aquí la ruta exacta a uno de los archivos que acabas de generar
# Ejemplo: 'data/bandeja/nombre_del_video-150-features.npy'
ruta_archivo = 'c:/Users/efdez/Desktop/Carrera/TFG/tennis_action_recognition/lrcnn/data/sequences/bandeja/Anton00001901_rotated-150-features.npy'

try:
    # 2. Cargar el archivo en memoria
    caracteristicas = np.load(ruta_archivo)
    
    # 3. Mostrar la información crucial
    print("=== INSPECCIÓN DEL ARCHIVO .NPY ===")
    print(f"Ruta: {ruta_archivo}")
    
    # Esta es la métrica más importante (Shape)
    print(f"\nDimensión de la matriz (Shape): {caracteristicas.shape}")
    print(f"Tipo de dato: {caracteristicas.dtype}")
    
    # 4. Comprobación automática para tu Transformer
    if len(caracteristicas.shape) == 2:
        frames = caracteristicas.shape[0]
        features_por_frame = caracteristicas.shape[1]
        
        print("\n--- DIAGNÓSTICO ---")
        if frames == 150:
            print("✅ ¡ÉXITO! El vídeo tiene exactamente 150 fotogramas.")
        else:
            print(f"⚠️ ATENCIÓN: El vídeo tiene {frames} fotogramas en lugar de 150.")
            
        if features_por_frame == 2048:
            print("✅ ¡ÉXITO! InceptionV3 ha extraído las 2048 características correctamente.")
        else:
            print(f"⚠️ ATENCIÓN: Hay {features_por_frame} características en lugar de 2048.")
            
    else:
        print("\n⚠️ La matriz no tiene 2 dimensiones (Fotogramas x Características). Revisa la extracción.")

except FileNotFoundError:
    print(f"❌ Error: No se ha encontrado el archivo en la ruta:\n{ruta_archivo}")
except Exception as e:
    print(f"❌ Ha ocurrido un error inesperado: {e}")