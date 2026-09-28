# Laboratorio 6. Transfer Learning y Fine-Tuning

CC3092: Deep Learning y Sistemas Inteligentes

Comparación de tres estrategias para clasificar CIFAR-10: una CNN entrenada desde cero, VGG-16 preentrenada
en ImageNet como extractor de características (backbone congelado) y VGG-16 con fine-tuning de los bloques
convolucionales superiores. Incluye medición de costo computacional (parámetros, MACs/FLOPs, latencia, tiempo
y memoria de GPU) y un experimento con el 10 % de los datos de entrenamiento.

## Contenido

- `Laboratorio_6_Transfer_Learning.ipynb`: notebook completo y comentado (carga y preparación de datos,
  investigación de transfer learning en PyTorch, definición de los tres modelos, 16 iteraciones de
  entrenamiento, medición de recursos, experimento de cantidad de datos y evaluación final en test).
- `results/`: resultado (historial de loss, métricas, tiempos, memoria) de cada iteración en formato JSON.
  El notebook reutiliza estos archivos si la configuración de la iteración no cambió; para reentrenar todo
  basta con borrar la carpeta o poner `CONFIG["reuse_cache"] = False`.
- `figuras/`: gráficas generadas por el notebook (curvas de loss, matrices de confusión, comparaciones).

## Requisitos

```
pip install torch torchvision torchinfo scikit-learn pandas matplotlib jupyter
```

Se requiere una GPU con CUDA (se midió con una NVIDIA GeForce RTX 3060 Ti de 8 GB). CIFAR-10 y los pesos de
VGG-16 se descargan automáticamente en la primera ejecución.

## Uso

Abrir `Laboratorio_6_Transfer_Learning.ipynb` en Jupyter y ejecutar las celdas en orden. El entrenamiento
completo toma aproximadamente una hora en la GPU indicada; los checkpoints de la mejor iteración de cada modelo
se guardan temporalmente en `checkpoints/` (no se versiona).
