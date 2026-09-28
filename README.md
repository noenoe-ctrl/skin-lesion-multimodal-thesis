# Clasificación multimodal de lesiones cutáneas

Investigación de tesis sobre el aporte incremental de representaciones
visuales congeladas de CLIP y DINOv2 cuando ya se dispone de metadata clínica.
La evaluación utiliza PAD-UFES-20, clasificación binaria y de seis clases,
cinco holdouts repetidos con separación por paciente y preprocesamiento
ajustado exclusivamente con train. ConvNeXt es un comparador visual adicional.

## Pregunta de investigación

¿Cuánto aportan los modelos fundacionales de imágenes a la clasificación de
lesiones cutáneas cuando ya disponemos de metadata clínica, y qué ganancia
adicional produce fusionar ambas fuentes?

Se comparan metadata sola, imágenes solas y fusión, con baselines clínicos
MLP, logística y boosting; controles de pérdida/capacidad; evaluación por
lesión y análisis exploratorio de incertidumbre por paciente. El avance
actual contiene 175 resultados finales por partición/tarea/configuración.
La ganancia binaria frente a referencias clínicas fuertes es pequeña y no
concluyente; la fusión muestra una ganancia mayor en multiclase. El alcance
se limita al dataset y a los modelos evaluados.

## Dónde empezar

- [Protocolo, respuesta provisional para el profesor y referencias](experiments/patient_split_v1/README.md).
- [Resultados completos y tablas para revisar](results/patient_split_v1/README.md).
- [Plan de la extensión de metadata sola](experiments/patient_split_v1/METADATA_BASELINE_TESIS.md).
- `dataset/01_...ipynb` a `13_...ipynb`: notebooks históricos, conservados
  como antecedentes. Sus resultados no deben mezclarse con el protocolo corregido.
- `experiments/patient_split_v1/14_Protocolo_Paciente_TrainOnly.ipynb`:
  entrada nueva para Colab.

## Datos y dependencias

El repositorio contiene código, documentación, notebooks y resultados
agregados. Las imágenes, metadata original, caches visuales, filas clínicas,
predicciones por imagen, checkpoints y entornos locales se mantienen fuera
de Git. La tesis PDF tampoco se incluye en esta selección.

Obtener PAD-UFES-20 desde su distribución original y respetar sus condiciones:
[artículo del dataset](https://doi.org/10.1016/j.dib.2020.106221).
La entrada clínica es `dataset/PAD-UFES-20/metadata.csv` original, sin imputar.

Para ejecutar el flujo actual también hacen falta los caches congelados de
CLIP, DINOv2 y ConvNeXt en `dataset/PAD-UFES-20/resultados/`, junto con
`split_train.csv`, `split_val.csv` y `split_test.csv` históricos que identifican
el orden de las filas de esos caches. Los notebooks históricos contienen los
experimentos de extracción. El protocolo nuevo lee esos splits únicamente
para alinear embeddings con `img_id`; no los usa como particiones de evaluación.

La copia publicada de tablas no reemplaza esos datos ni los artefactos
completos necesarios para reproducir las corridas. Python 3.12 y dependencias
principales fijadas en [requirements.txt](experiments/patient_split_v1/requirements.txt).

## Reproducir el protocolo

Desde la raíz del repositorio en PowerShell:

```powershell
python -m venv experiments/patient_split_v1/.venv
& ./experiments/patient_split_v1/.venv/Scripts/python.exe -m pip install -r experiments/patient_split_v1/requirements.txt
& ./experiments/patient_split_v1/.venv/Scripts/python.exe experiments/patient_split_v1/run_experiment.py prepare
& ./experiments/patient_split_v1/.venv/Scripts/python.exe experiments/patient_split_v1/run_experiment.py verify
& ./experiments/patient_split_v1/.venv/Scripts/python.exe experiments/patient_split_v1/run_experiment.py smoke --run-id smoke_local
```

Después de verificar el smoke y disponer de los datos/caches:

```powershell
& ./experiments/patient_split_v1/.venv/Scripts/python.exe experiments/patient_split_v1/run_experiment.py full --run-id paper_baseline_v1
& ./experiments/patient_split_v1/.venv/Scripts/python.exe experiments/patient_split_v1/run_metadata_baseline.py full --run-id metadata_only_v1
& ./experiments/patient_split_v1/.venv/Scripts/python.exe experiments/patient_split_v1/complete_thesis_experiments.py
& ./experiments/patient_split_v1/.venv/Scripts/python.exe experiments/patient_split_v1/run_empirical_prior.py
& ./experiments/patient_split_v1/.venv/Scripts/python.exe experiments/patient_split_v1/analyze_thesis_completion.py
```

El entrenamiento local adicional usa CPU. Los identificadores existentes no
se sobrescriben. Los artefactos completos se generan localmente en
`experiments/patient_split_v1/artifacts/`, que está excluido de Git.
Para comprobar las pruebas, ejecutar `python -m unittest test_protocol
test_metadata_baseline test_thesis_completion -v` desde ese directorio con
el Python del entorno creado.

## Estado de investigación

Los resultados y sus intervalos son exploratorios y condicionales a los
modelos entrenados. La evaluación externa y la incertidumbre con
reentrenamiento son ampliaciones pendientes. El proyecto no constituye una
herramienta validada para decisiones clínicas.
