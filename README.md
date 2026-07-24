# Reconocimiento de acciones en pádel con deep learning

Trabajo de Fin de Máster: adaptación de un reconocedor de golpes de **tenis** a un
reconocedor de golpes de **pádel**, usando una arquitectura LRCN (Long-term Recurrent
Convolutional Network).

```
clip de vídeo → InceptionV3 (features por frame) → secuencia de 16 pasos → LSTM → clase de golpe
```

## Basado en trabajo previo

Este repositorio **parte de [chow-vincent/tennis_action_recognition](https://github.com/chow-vincent/tennis_action_recognition)**
(Vincent Chow y Ohi Dibua, proyecto CS230 de Stanford, licencia MIT), que resuelve la misma
tarea sobre el dataset de tenis THETIS. El historial de git conserva sus commits originales.

Se mantiene su licencia MIT (ver [LICENSE](LICENSE)).

## Qué aporta este TFM

- **Dataset propio de pádel.** 567 clips anotados sobre partidos reales, sustituyendo por
  completo al dataset THETIS de tenis.
- **Nueva taxonomía de 9 clases** específicas de pádel, frente a las 6 de tenis del original:
  `bandeja`, `derecha`, `revés`, `remate`, `víbora`, `salidaD`, `salidaR`, `voleaD`, `voleaR`.
  Golpes como la bandeja y la víbora no tienen equivalente en tenis.
- **Aumento de datos** (`espejo.py`, `rotation.py`): espejado horizontal y rotación, que
  llevan el conjunto de 567 a 2268 muestras.
- **Reentrenamiento y ajuste de hiperparámetros** del LRCN sobre los datos de pádel
  (LSTM de 512 unidades, capa densa de 256, secuencias de 16 frames, 300 épocas).
- Adaptación del pipeline de extracción de características y de los scripts de
  entrenamiento y evaluación (`extract_and_save_sequences.ipynb`, `data_to_csv.ipynb`,
  `data_utils.py`, `lrcnn_train.py`, `lrcnn_evaluate.py`).

El directorio [`optical_flow/`](optical_flow) es código **del proyecto original sin
modificar**, conservado para referencia. La aportación de este TFM está en [`lrcnn/`](lrcnn).

## ⚠️ Limitaciones conocidas

**Fuga de datos en la partición.** El aumento de datos se aplicó *antes* de dividir en
train / validation / test, de modo que variantes espejadas o rotadas de un mismo clip base
acabaron en conjuntos distintos: 331 de los 567 clips base (58 %) aparecen en más de un
split. El modelo puede así ser evaluado sobre una transformación de un clip que ya vio
durante el entrenamiento.

**Consecuencia: las métricas de test obtenidas están infladas y no reflejan la capacidad
real de generalización.** Por eso no se publican aquí.

La corrección pasa por particionar por **clip base** (el nombre de fichero identifica el
clip original) y aplicar el aumento únicamente al conjunto de entrenamiento. Idealmente la
partición debería ser además por partido o jugador, para medir generalización a jugadores
no vistos. Queda pendiente regenerar `lrcnn/data/data_file.csv` con ese criterio y
reentrenar.

## Estructura

```
├── lrcnn/          # Modelo LRCN: extracción de features + LSTM (aportación del TFM)
│   ├── data/       # data_file.csv con los splits; las features .npy no se versionan
│   ├── espejo.py   # Aumento por espejado horizontal
│   ├── rotation.py # Aumento por rotación
│   └── ...         # Entrenamiento, evaluación y búsqueda de hiperparámetros
└── optical_flow/   # Código original de flujo óptico (sin modificar)
```

Ver [`lrcnn/README.md`](lrcnn/README.md) para las instrucciones de ejecución.

## Datos

Ni los vídeos ni las features extraídas (`.npy`) se incluyen en el repositorio por tamaño.
El pipeline para regenerarlas desde los vídeos está descrito en `lrcnn/README.md`.

## Licencia

MIT — ver [LICENSE](LICENSE). Copyright original de Vincent Chow y Ohi Dibua (2018);
modificaciones de Eduardo Fernández Fraga.
