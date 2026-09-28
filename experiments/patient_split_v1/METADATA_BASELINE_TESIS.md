# Aporte incremental de las imágenes sobre la metadata clínica

Extensión para la pregunta central de la tesis, 28/09/2026. No se plantea
todavía como un resultado de paper ni se asume superioridad de una modalidad.
Los resultados existentes ya se inspeccionaron: este análisis adicional es
exploratorio; no es una hipótesis registrada antes de observarlos.

Actualización: los 10 entrenamientos completos de metadata MLP ya se
ejecutaron y verificaron. Véase
`artifacts/comparisons/metadata_only_v1/RESULTADOS_METADATA_COMPLETOS.md`
para las tablas actualizadas y las ganancias pareadas por semilla.

Actualización posterior: también se completaron las referencias logística,
boosting y priors, los controles de pérdida/capacidad y el análisis por
paciente. El informe vigente es
`artifacts/comparisons/thesis_completion_v2/RESULTADOS_TESIS_COMPLETOS.md`.
La bibliografía y los comandos completos están en `README.md`. Las secciones
de pendientes de este documento registran el plan inicial de la extensión;
no describen el estado actual de esos experimentos ya terminados.

## Pregunta y alcance

¿Cuánto aportan los modelos fundacionales de imágenes a la clasificación de
lesiones cutáneas cuando ya disponemos de metadata clínica, y qué ganancia
adicional produce fusionar ambas fuentes?

La evaluación empírica de esta pregunta se realiza en PAD-UFES-20, con
representaciones congeladas de CLIP/DINOv2, clasificación binaria y de seis
clases, y separación por paciente. El dataset pertenece a la metodología;
las conclusiones deben limitarse a la evidencia obtenida con este diseño.

Se evalúan imágenes con separación por paciente, no un diagnóstico único por
paciente. La metadata incluye características del paciente **y de la lesión**
(síntomas, diámetros, localización). No es solamente información demográfica.
Binario: BCC/MEL/SCC malignos frente a ACK/NEV/SEK. Multiclase: esas seis clases.
ConvNeXt es un comparador visual preentrenado; no se atribuye automáticamente
la misma condición de modelo fundacional que a CLIP/DINOv2. Con estos datos
no se separa causalmente arquitectura, tamaño y tipo de preentrenamiento.

Para una métrica S, una partición r, metadata M, imágenes V y fusión F:

- Comparación unimodal: delta_visual(r) = S(V,r) - S(M,r).
- Aporte visual dado el contexto clínico: delta_visual_condicional(r) = S(F,r) - S(M,r).
- Aporte de metadata a las imágenes: delta_metadata(r) = S(F,r) - S(V,r).
- Complementariedad: delta_mejor_unimodal(r) = S(F,r) - max(S(M,r), S(V,r)).

La última es un contraste descriptivo; no sirve para escoger un modelo a
desplegar usando test. Para seleccionar el mejor baseline, usar validación.
El contraste visual versus metadata **no** mide por sí solo el aporte visual
condicionado a disponer de metadata. Las diferencias describen pipelines
predictivos, no una cantidad causal de información presente en una modalidad.

## Hipótesis contrastables

H1: las representaciones visuales y la metadata sola pueden diferir en
desempeño. H0: delta_visual = 0; alternativa bilateral: distinto de cero.

H2 (principal): añadir las imágenes a la metadata puede cambiar el desempeño.
H0: delta_visual_condicional = 0; alternativa bilateral: distinto de cero.
Solo se hablará de mejora si la diferencia observada es positiva y la
incertidumbre la respalda. Es posible observar equivalencia o empeoramiento;
una diferencia no significativa no demuestra equivalencia.

H3: añadir metadata a las imágenes puede cambiar el desempeño.
H0: delta_metadata = 0; alternativa bilateral: distinto de cero.

H4 (secundaria): A y B pueden diferir al fijar datos, tarea y pérdida.
No comparar arquitecturas como si BCE y focal fueran el mismo tratamiento.
La complementariedad positiva frente a ambas modalidades es una expectativa
a evaluar, no un supuesto del diseño.

## Antecedentes y baselines

- Pacheco y Krohling (2020), *The impact of patient clinical information on
  automated skin cancer detection*: estudian incorporar datos clínicos a
  imágenes. Justifica medir la contribución clínica, no asumir que nuestras
  imágenes fundacionales aportan más que metadata.
  https://arxiv.org/abs/1909.12912
- Ou et al. (2022), *A deep learning based multimodal fusion model for skin
  lesion diagnosis using smartphone collected clinical images and metadata*:
  combinan encoders visuales/clínicos y atención intra/intermodal, comparando
  métodos de fusión. Sirve como antecedente para A/B y ablaciones.
  https://pubmed.ncbi.nlm.nih.gov/36268206/
- Deng et al. (2025), *Development of a Transfer Learning-Based, Multimodal
  Neural Network for Identifying Malignant Dermatological Lesions From
  Smartphone Images*: compara red clínica, DenseNet-121 visual y red
  multimodal en PAD-UFES-20 binario; incluye discriminación y calibración.
  Es un antecedente directo de la comparación de tres modalidades.
  https://pubmed.ncbi.nlm.nih.gov/40568418/
- CLIP (Radford et al., 2021): https://arxiv.org/abs/2103.00020
- DINOv2 (Oquab et al.): https://arxiv.org/abs/2304.07193

Estos artículos motivan el diseño. Sus cifras no son comparables directamente
con las nuestras sin verificar split por paciente, etiquetas, preprocesamiento,
selección y evaluación. No se atribuye aquí a sus splits una auditoría que
no se haya realizado.

| Baseline clínico | Objetivo | Estado |
|---|---|---|
| MLP 512 -> 256 -> salida | Ablación comparable con la cabeza de A y los modelos visuales existentes | 10 entrenamientos completos y verificados |
| Regresión logística L2 / multinomial | Referencia simple y regularizada | Propuesto, pendiente |
| Gradient boosting tabular | Referencia no lineal que evita depender de una única MLP | Propuesto, pendiente |
| Prior de clases de train (DummyClassifier) | Referencia sin información de entrada | Propuesto, pendiente |

Para regresión logística proponemos C en {0.01,0.1,1,10}; elegir con el mismo
criterio de validación, sin reajustar el preprocesador con val. Para boosting,
fijar una grilla pequeña y el mismo presupuesto antes de correrla. No elegir
entre baselines clínicos usando test. La primera MLP mantiene hiperparámetros
fijos de las corridas anteriores; igualdad de procedimiento no implica
igualdad de número de parámetros porque cambia la dimensión de entrada.
Un baseline débil de metadata podría exagerar el aporte aparente de imágenes;
de ahí las referencias lineal y tabular pendientes.

## Qué muestran las corridas actuales

Fuente: `artifacts/runs/paper_baseline_v1/summary.csv`, 90 entrenamientos,
5 holdouts por paciente x 2 tareas x 9 modelos. Métricas reconciliadas contra
predicciones en `results_verification.json`. Medias siguientes; consultar el
CSV para desviaciones. Los valores de metadata provienen de
`artifacts/runs/metadata_only_v1/summary.csv` (10 entrenamientos nuevos).
Son diferencias descriptivas, sin significancia probada.

| Modelo | Entrada | AUC binaria | BACC binaria | AUC macro OvR | BACC multiclase | Estado |
|---|---|---:|---:|---:|---:|---|
| Metadata MLP | M | 0.9410 | 0.8692 | 0.9086 | 0.5993 | 5 semillas completas |
| CLIP | V | 0.9023 | 0.8283 | 0.9183 | 0.6509 | 5 semillas completas |
| DINOv2 | V | 0.8818 | 0.7921 | 0.8981 | 0.6280 | 5 semillas completas |
| ConvNeXt | V | 0.8662 | 0.7930 | 0.8729 | 0.5648 | 5 semillas completas |
| CLIP + DINOv2 | V | 0.9046 | 0.8110 | 0.9186 | 0.6441 | 5 semillas completas |
| CLIP + metadata | V + M | 0.9497 | 0.8872 | 0.9346 | 0.7095 | 5 semillas completas |
| DINOv2 + metadata | V + M | 0.9473 | 0.8753 | 0.9290 | 0.6807 | 5 semillas completas |
| ConvNeXt + metadata | V + M | 0.9451 | 0.8750 | 0.9207 | 0.6494 | 5 semillas completas |
| A: CLIP + DINOv2 + metadata | V + M | 0.9506 | 0.8828 | 0.9381 | 0.7111 | 5 semillas completas |
| B: fusión intermedia | V + M | 0.9555 | 0.8731 | 0.9383 | 0.7174 | 5 semillas completas; focal binaria |

CLIP+metadata frente a CLIP: +0.0475 AUC binaria y +0.0586 BACC multiclase.
A frente a CLIP+DINOv2: +0.0460 AUC binaria y +0.0671 BACC multiclase.
Son mejoras de fusión respecto a imágenes. Frente a metadata MLP, A mejora
la AUC binaria en 0.0097 y B en 0.0146; la BACC multiclase aumenta 0.1119
con A y 0.1182 con B. Son resultados descriptivos. B tiene mayor AUC binaria pero no mayor BACC
que A; no existe un ganador único entre métricas.

| Contraste pareado, misma semilla y tarea | Qué responde | Qué falta |
|---|---|---|
| CLIP / DINOv2 / ConvNeXt / CLIP+DINOv2 menos M | Diferencia entre modalidades solas | Incertidumbre por paciente |
| CLIP+M menos M y menos CLIP | Aporte de cada modalidad | Incertidumbre por paciente |
| DINOv2+M menos M y menos DINOv2 | Aporte de cada modalidad | Incertidumbre por paciente |
| ConvNeXt+M menos M y menos ConvNeXt | Control visual convencional | Incertidumbre por paciente |
| A menos M y menos CLIP+DINOv2 | Ganancia de concatenar ambas fuentes | Incertidumbre por paciente |
| B menos M y menos CLIP+DINOv2 | Ganancia del pipeline B | M focal para binario; controla también arquitectura |
| B menos A con igual pérdida | Efecto de estrategia de fusión | B+BCE y A+focal binarios |

## Implementación y ejecución

Archivos nuevos: `run_metadata_baseline.py`, `test_metadata_baseline.py`,
`verify_metadata_run.py` y este documento. El runner registra `metadata_mlp`
solo en memoria y reutiliza `run_experiment.train_one`; no edita el runner
original ni añade modelos a su matriz de nueve modelos por defecto.

Reutiliza los cinco manifiestos y los cinco `preprocessor.json` existentes.
No crea splits ni calcula estadísticas globales. El verificador vuelve a
ajustar estadísticas en train para auditarlas; entrenar usa el estado ya
guardado. Usa 21 variables clínicas, excluyendo IDs, etiquetas y biopsed.
La cabeza recibe únicamente `processor.transform(filas)`; el diccionario
visual es vacío. La verificación de integridad sí audita los caches existentes.

Configuración: MLP con capas ocultas 512/256, ReLU, dropout 0.3, Adam
lr=0.001, weight_decay=0.0001, batch=64, hasta 50 épocas, paciencia=12,
ReduceLROnPlateau sobre loss de validación. Binario BCE, checkpoint por
AUC de validación, umbral 0.5. Multiclase focal ponderada por frecuencias de
train, checkpoint por BACC de validación, argmax. Son exactamente los
procedimientos compartidos de A y de las cabezas visuales anteriores.
`--binary-loss focal` permite una corrida clínica controlada con la pérdida
de B; no se ejecuta automáticamente.

Desde PowerShell, en `C:\Users\huari\tesis\experiments\patient_split_v1`:

```powershell
& ./.venv/Scripts/python.exe -m unittest test_protocol test_metadata_baseline -v
& ./.venv/Scripts/python.exe run_metadata_baseline.py smoke --run-id metadata_only_smoke_v1 --device cpu
& ./.venv/Scripts/python.exe verify_metadata_run.py --run-dir artifacts/smoke/metadata_only_smoke_v1
```

El smoke usa semilla 42, 128 imágenes de train, 64 de validación y 2 épocas
en cada tarea. No entrena ni selecciona con test. El preprocesamiento viene
del train completo de esa partición; esto es prueba funcional, no evaluación
de aprendizaje con pocas muestras. El run-id ya utilizado se rechaza: para
repetir el smoke, elegir otro identificador.

Comandos **ya ejecutados** para completar y verificar las 10 corridas M
(el run-id existente se conserva y no permite sobrescribir):

```powershell
& ./.venv/Scripts/python.exe run_metadata_baseline.py full --run-id metadata_only_v1
& ./.venv/Scripts/python.exe verify_metadata_run.py --run-dir artifacts/runs/metadata_only_v1
```

Control clínico focal adicional, 5 corridas binarias, **no ejecutado**:

```powershell
& ./.venv/Scripts/python.exe run_metadata_baseline.py full --tasks binary --binary-loss focal --run-id metadata_only_focal_v1
```

Controles de arquitectura existentes, 20 corridas binarias, **no ejecutados**:

```powershell
& ./.venv/Scripts/python.exe run_experiment.py full --tasks binary --models clip_dino_meta fusion --factorial --run-id fusion_loss_control_v1
```

Ese comando repite A+BCE y B+focal y añade A+focal y B+BCE. Solo 10 de las
20 combinaciones son nuevas; la CLI existente incluye las cuatro sin
sobrescribir las anteriores. Para valorar B estrictamente, también conviene
una ablación de sus ramas con/sin metadata y capacidad comparable; pendiente.

## Métricas e inferencia para la tesis

- Mantener métricas y selección previas para comparar sin cambiar el criterio
  después de observar test. Principal binaria: ROC-AUC; principal multiclase:
  BACC. Reportar accuracy, BACC, precisión, recall, F1 y AUC macro OvR.
- Recall binario mide sensibilidad a malignidad. Multiclase usa macro-F1,
  macro-precisión y macro-recall; este último coincide con BACC aquí, por lo
  que no son dos evidencias independientes.
- Reportar por semilla, media +/- desviación poblacional (ddof=0, convención
  existente) y diferencias pareadas por semilla; no mezclar splits históricos
  por imagen con este protocolo. Los resultados completos M ya están verificados.
- Añadir especificidad, AP (definirla como average precision, sin confundirla
  con integración trapezoidal de PR), Brier y calibración como análisis
  secundarios de predicciones guardadas. Si se elige otro umbral o se calibra,
  hacerlo solo con validación y aplicar después a test.
- Los cinco holdouts tienen pacientes compartidos entre repeticiones. No
  tratar cinco métricas como muestras independientes ni hacer bootstrap por
  imágenes como si cada foto fuera un paciente diferente. Para intervalos
  de diferencias usar bootstrap pareado por paciente dentro de cada test,
  incluyendo todas sus imágenes, y reportar por repetición. Un intervalo
  conjunto necesita tratar también la dependencia entre repeticiones; no
  concatenar todas las predicciones como observaciones independientes.
  Fijar semilla/número de remuestreos y tratar remuestreos sin todas las clases.
- Predefinir contrastes principales A/B frente a M y corrección por múltiples
  comparaciones si se informan p-valores; no seleccionar solo mejoras.
- Auditar tamaños por clase, errores/confusiones y pacientes con múltiples
  imágenes. Evaluación agregada por lesión compuesta es un análisis de
  sensibilidad pendiente, con regla de agregación fijada antes de medir.
- Reservar nuevos datos o evaluación externa para conclusiones confirmatorias
  si se sigue eligiendo diseños a partir de estos tests ya observados.

## Pendiente y límites

Las 10 corridas M y sus deltas pareados frente a los 90 resultados existentes
ya están completos y verificados. Falta cuantificar incertidumbre por paciente
e incorporar regresión logística, boosting y prior; fijar ajuste por validación
y comparar contra el mejor baseline clínico elegido sin test. Ejecutar los
controles de pérdidas/capacidad antes de atribuir diferencias a A/B.

Esta tesis medirá el aporte de **estas representaciones congeladas y estos
clasificadores en PAD-UFES-20**. No demostrará que todos los modelos
fundacionales mejoran, que la fusión siempre gana, ni utilidad clínica o
generalización externa. La investigación para un paper podrá ampliar después
controles, datos externos y análisis de robustez sin confundir esas etapas.
