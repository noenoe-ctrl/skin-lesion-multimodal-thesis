# PAD-UFES-20: protocolo por paciente v1

## Diseño fijado antes de entrenar

- Cinco holdouts agrupados repetidos; semillas 42, 123, 456, 789 y 2024.
- Unidad de muestra: imagen; unidad de separación: `patient_id`.
- Cada paciente queda completo en train, validación o test. Para auditar
  lesiones se usa `(patient_id, lesion_id)`, pues `lesion_id` no es globalmente único.
- Estratificación por las seis clases, compartida entre binario y multiclase
  y entre todos los modelos. No se reduce el paciente a una clase mayoritaria.
- `StratifiedGroupKFold(n_splits=10, shuffle=False)`: bloque 0 para test,
  bloque 1 para validación, ocho bloques para train. Se recodifican aleatoriamente
  los grupos con la semilla para desempatar el algoritmo de forma reproducible.
  No se seleccionan particiones por desempeño del modelo.
- Objetivo aproximado 80/10/10; en los cinco splits verificados se obtuvieron
  1838/230/230 imágenes. Las proporciones por clase no son matemáticamente exactas.
- Estos son holdouts repetidos, **no** validación cruzada de cinco folds.
  Un paciente puede aparecer en conjuntos distintos entre repeticiones; nunca
  dentro de una misma repetición. Las métricas entre repeticiones no son independientes.

## Preprocesamiento dentro de cada partición

1. Limpieza fija por valor: vacíos y `UNK`/`UNKNOWN` pasan a faltantes;
   las categorías se normalizan a mayúsculas. No se usa información del conjunto
   para esta limpieza.
2. Medianas por variable numérica, calculadas únicamente con las filas de train.
   Son estadísticas ponderadas por filas/imágenes, sin deduplicar pacientes.
   No se condicionan por diagnóstico. Una columna numérica completamente vacía
   en train provoca un error explícito.
3. Moda por variable categórica en train, con desempate alfabético. Si no hay
   observaciones, se usa la categoría explícita `__UNKNOWN__`.
4. Min-Max de edad, fototipo y diámetros, seleccionado por nombre y ajustado
   sobre train imputado. No se recorta val/test a [0,1]; valores fuera del rango
   de train pueden producir valores fuera de ese intervalo.
5. One-hot para todas las variables categóricas, vocabulario de train y una
   categoría reservada para valores nuevos. Se mantienen nombres y orden de columnas.
6. Validación y test solo reciben `transform`: las estadísticas y el vocabulario
   nunca se vuelven a aprender. Todos los modelos de una semilla comparten el
   mismo preprocesador.

`patient_id`, `lesion_id`, `img_id`, `diagnostic`, `label` y `biopsed` no son
predictores. Las etiquetas solo se utilizan para estratificar, entrenar y evaluar.
Los CSV `*_imputed.csv` conservan esos campos para trazabilidad, pero el modelo
solo recibe las columnas de `META` definidas en `protocol.py`.

## Archivos y artefactos

- `protocol.py`: particiones, preprocesador serializable, auditoría, hashes y alineación de embeddings.
- `run_experiment.py`: CLI separada para preparar, verificar, probar y entrenar.
- `test_protocol.py`: cinco pruebas de regresión de imputación, independencia
  de etiquetas, escalamiento por nombre, desconocidos y columnas vacías.
- `requirements.txt`: versiones principales usadas en la verificación local.
- `14_Protocolo_Paciente_TrainOnly.ipynb`: entrada para Colab; entrenamiento completo desactivado.
- `artifacts/protocol.json`: diseño, versiones y hashes de datos/embeddings.
- `artifacts/split_summary.csv`: imágenes, pacientes, lesiones y seis clases por conjunto.
- `artifacts/splits/seed_*/`: `train_raw.csv`, `val_raw.csv`, `test_raw.csv`, sus
  versiones imputadas, matrices numéricas, `manifest.csv`, `audit.json` y `preprocessor.json`.
- `artifacts/historical_sha256.json`: hashes de notebooks y archivos históricos
  comprobados para verificar que se conservaron.
- `artifacts/smoke/`: únicamente prueba técnica; no son resultados científicos.
- `artifacts/runs/`: se crea **solo** al solicitar el entrenamiento completo.

No se sobrescriben artefactos preparados ni identificadores de ejecución existentes.
Para una nueva preparación se utiliza otro `--output`; para otra ejecución, otro `--run-id`.

## Ejecución local en PowerShell

Desde `C:\Users\huari\tesis`, el entorno aislado `.venv` ya contiene las
dependencias para la prueba CPU. No modifica el entorno de los notebooks anteriores.

```powershell
# Verificación de los splits y de los archivos históricos, sin entrenar
& '.\experiments\patient_split_v1\.venv\Scripts\python.exe' '.\experiments\patient_split_v1\run_experiment.py' verify

# Pruebas pequeñas de preprocesamiento
& '.\experiments\patient_split_v1\.venv\Scripts\python.exe' '.\experiments\patient_split_v1\test_protocol.py'

# Repetir la prueba técnica con un ID nuevo
& '.\experiments\patient_split_v1\.venv\Scripts\python.exe' '.\experiments\patient_split_v1\run_experiment.py' smoke --run-id smoke_02 --device cpu

# EXPERIMENTO COMPLETO: 9 modelos x 2 tareas x 5 particiones = 90 entrenamientos
& '.\experiments\patient_split_v1\.venv\Scripts\python.exe' '.\experiments\patient_split_v1\run_experiment.py' full --run-id paper_baseline_v1

# Alternativa: añadir el control arquitectura x pérdida en binario (100 entrenamientos)
& '.\experiments\patient_split_v1\.venv\Scripts\python.exe' '.\experiments\patient_split_v1\run_experiment.py' full --run-id paper_factorial_v1 --factorial
```

Los dos últimos comandos son alternativas: no hace falta ejecutar ambos.
La instalación local es CPU. Para GPU se puede usar Colab o instalar una versión
CUDA compatible de PyTorch en otro entorno; el comando elige CUDA si está disponible.
Puede limitarse una ejecución con `--models clip_dino_meta fusion` y
`--tasks binary`, o usar `--epochs` para otra prueba, guardada con un ID distinto.

Para preparar artefactos nuevos, sin sustituir los existentes:

```powershell
& '.\experiments\patient_split_v1\.venv\Scripts\python.exe' '.\experiments\patient_split_v1\run_experiment.py' prepare --output '.\experiments\patient_split_v1\artifacts_v2'
```

En otra máquina: Python 3.12, crear un entorno y ejecutar
`python -m pip install -r requirements.txt`. Para PyTorch CPU se puede instalar
`torch==2.8.0` desde `https://download.pytorch.org/whl/cpu`. La lista de paquetes del
entorno de prueba se conserva en `environment_verified.txt`; puede contener la
variante `torch==2.8.0+cpu`, que requiere ese índice.

## Entrenamiento y comparación

Los nueve modelos son `clip`, `dino`, `convnext`, `clip_dino`, `clip_meta`,
`dino_meta`, `convnext_meta`, `clip_dino_meta` y `fusion`. Se reutilizan los
backbones congelados; no se hace fine-tuning.

- Cabezas de concatenación: entrada → 512 → 256 → salida; dropout 0.3.
- Fusión intermedia: ramas visuales a 256, metadata → 128 → 256;
  concatenación → 256 → salida.
- Adam: lr 0.001, weight decay 0.0001, lotes de 64, máximo 50 épocas,
  early stopping 12 y ReduceLROnPlateau por pérdida de validación.
- Selección: AUC de validación en binario, balanced accuracy en multiclase.
- Binario: BCE para concatenaciones y Focal(alpha=0.25, gamma=2) para fusión.
  `--factorial` añade concatenación completa con Focal y fusión con BCE.
- Multiclase: Focal ponderada con pesos inversos calculados solo en train.
  Se corrige la fórmula: el peso queda fuera del factor de probabilidad focal.
  Esta corrección y el one-hot de todas las categóricas son diferencias
  explícitas respecto a los notebooks históricos.
- Predicción: umbral binario 0.5 o argmax multiclase, sin calibración en test.
- Semillas reiniciadas por experimento; operaciones deterministas habilitadas.
  No se garantiza igualdad bit a bit entre versiones/hardware diferentes.

La prueba `smoke` hace cuatro entrenamientos: dos modelos (concatenación completa
y fusión), dos tareas, primera semilla, 128 imágenes train y 64 de validación,
dos épocas. Su preprocesador es el del train completo de esa partición.
La verificación lee test para auditar el split; la prueba **no evalúa test ni
genera sus predicciones**. Sus muestras son equilibradas por clase y sus métricas
de validación solo prueban funcionamiento, no estiman desempeño final.

Cada entrenamiento completo guarda checkpoint, configuración, historial,
probabilidades por imagen y métricas individuales por semilla. Al terminar se
genera `summary.csv` con media y desviación estándar poblacional (ddof=0),
para mantener la convención anterior. Los resultados por semilla permanecen disponibles.
Las figuras posteriores deben consumir esas predicciones, sin reentrenar.
Las inferencias estadísticas deben considerar pacientes y la dependencia entre holdouts.

Los resultados históricos y nuevos no deben mezclarse en un único ranking:
cambian split, imputación, codificación, escalamiento y la fórmula multiclase.
Los hashes comprueban correspondencia e integridad del caché, pero no pueden
probar retrospectivamente cómo se extrajo cada embedding histórico.

Referencias de implementación: [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html),
[preprocesamiento sin fuga](https://scikit-learn.org/1.8/common_pitfalls.html),
[reproducibilidad en PyTorch](https://docs.pytorch.org/docs/stable/notes/randomness.html).

## Pregunta central de la tesis y comparación completa

**¿Cuánto aportan los modelos fundacionales de imágenes a la clasificación de
lesiones cutáneas cuando ya disponemos de metadata clínica, y qué ganancia
adicional produce fusionar ambas fuentes?**

PAD-UFES-20 pertenece a la metodología con la que evaluamos la pregunta.
Las conclusiones se limitan a estas representaciones congeladas, estos
clasificadores y estos pacientes; no se presupone mejora visual o multimodal.
La metadata incluye atributos clínicos del paciente y de la lesión.

Se comparan **todos** los modelos ejecutados, organizados en tres grupos:

1. Metadata sola: MLP, regresión logística L2/multinomial, gradient boosting,
   prior empírico de train y prior ponderado (uniforme en multiclase).
2. Imágenes solas: CLIP, DINOv2, ConvNeXt y CLIP+DINOv2.
3. Fusión: CLIP+metadata, DINOv2+metadata, ConvNeXt+metadata, A y B.

A y B son los contrastes principales de fusión; los otros modelos son
ablaciones y comparadores necesarios para identificar qué fuente aporta.
ConvNeXt se mantiene como comparador visual preentrenado convencional, sin
atribuirle automáticamente la misma condición fundacional que CLIP/DINOv2.

La referencia clínica principal se elige **con validación** en cada semilla
y tarea entre MLP, logística y boosting; no se selecciona con test. Se reporta
también cada baseline clínico individual para que la selección sea auditable.
Las grillas clínicas tienen cuatro candidatos por familia: C={0.01,0.1,1,10}
para logística y árboles={100,200} x hojas={7,15} para HistGradientBoosting.
Boosting desactiva su split interno de early stopping; todo se aprende en
train y solo el score de la validación por paciente decide la configuración.
En multiclase ambos usan pesos balanceados calculados en train. Los métodos
tabulares minimizan log-loss, no focal; igualdad de datos/selección no significa
que todos los algoritmos deban usar el mismo optimizador o loss.

Los controles nuevos añaden A+focal, B+BCE y MLP clínica+focal en binario.
Además, B solo metadata y B solo imágenes conservan la arquitectura y el
número de parámetros de B completo en ambas tareas. Las ramas sin información
reciben ceros constantes; sus sesgos permanecen entrenables. Son ablaciones
con capacidad igual, no arquitecturas más pequeñas propuestas para despliegue.

Archivos nuevos de esta ampliación:

- `complete_thesis_experiments.py`: grillas clínicas y controles; guarda el
  diseño antes de entrenar y rechaza sobrescribir ejecuciones existentes.
- `run_empirical_prior.py`: referencia con frecuencias originales de train,
  sin ponderación por clases.
- `thesis_metrics.py`: métricas, calibración descriptiva y estadísticas
  ponderadas para remuestreo por paciente.
- `test_thesis_completion.py`: pruebas de selección, ausencia de información
  en ramas anuladas y equivalencia del AUC/BACC ponderado con sklearn.
- `analyze_thesis_completion.py`: reconciliación de predicciones y métricas,
  selección clínica, diferencias pareadas e incertidumbre por paciente.
- `artifacts/runs/thesis_completion_v1/`: 65 referencias/controles nuevos.
- `artifacts/runs/metadata_empirical_prior_v1/`: 10 referencias de prior.
- `artifacts/comparisons/thesis_completion_v2/RESULTADOS_TESIS_COMPLETOS.md`:
  tabla completa y contrastes; `analysis.json` contiene matrices de confusión,
  sensibilidad por lesión, tablas de calibración e intervalos por partición.

Son 175 resultados finales por partición/tarea/modelo: 100 anteriores y 75
adicionales. No confundirlos con 175 modelos distintos ni con el número de
ajustes de candidatos; las grillas tabulares requieren ajustes adicionales.

Desde este directorio, los comandos de la ampliación son:

```powershell
& ./.venv/Scripts/python.exe -m unittest test_thesis_completion -v
& ./.venv/Scripts/python.exe complete_thesis_experiments.py
& ./.venv/Scripts/python.exe run_empirical_prior.py
& ./.venv/Scripts/python.exe analyze_thesis_completion.py
```

Las ejecuciones existentes se conservan y esos comandos rechazan sobrescribir
su salida. En otra máquina, copiar el código junto con los datos/artefactos
verificados para una reproducción independiente.

## Respuesta provisional a la pregunta de tesis: texto para mejorar

**Pregunta:** ¿cuánto aportan los modelos fundacionales de imágenes a la
clasificación de lesiones cutáneas cuando ya disponemos de metadata clínica,
y qué ganancia adicional produce fusionar ambas fuentes?

**Respuesta provisional:** en los experimentos realizados, el aporte visual
adicional depende de la tarea y de la referencia clínica utilizada. En
clasificación binaria, la metadata sola puede alcanzar un desempeño cercano
al de la fusión y todavía no observamos evidencia consistente de una ganancia
adicional frente a referencias clínicas fuertes. En clasificación de seis
clases, las fusiones A y B muestran una ganancia mayor frente a metadata sola.
La evidencia actual apoya investigar esa complementariedad, con las
limitaciones de evaluación descritas abajo; no demuestra que los modelos
fundacionales siempre mejoren el desempeño.

### Contexto de investigación

La incorporación de información clínica a las imágenes tiene antecedentes.
[Pacheco y Krohling (2020)](https://doi.org/10.1016/j.compbiomed.2019.103545)
compararon modelos visuales con y sin un mecanismo de agregación clínica.
[Ou et al. (2022)](https://doi.org/10.3389/fsurg.2022.1029991) estudiaron
encoders de ambas modalidades y fusión mediante atención, comparando métodos
de integración. Estos trabajos motivan evaluar la complementariedad de las
fuentes, pero añadir metadata a imágenes no responde por sí solo cuánto
aportan las imágenes cuando ya existe información clínica.

Un antecedente especialmente pertinente es
[Deng et al. (2025)](https://doi.org/10.1177/11769351251349891), que compararon
redes clínica, visual basada en DenseNet-121 y multimodal en PAD-UFES-20
binario, evaluando discriminación y calibración. Por tanto, la comparación
entre las tres modalidades ya tiene precedentes. En esta tesis se examina el
aporte incremental de representaciones congeladas de CLIP y DINOv2 con
baselines clínicos de distinta complejidad y controles explícitos. Las cifras
publicadas en esos trabajos no se comparan directamente con las nuestras,
pues requerirían verificar equivalencia de datos, particiones y protocolos.

### Evidencia obtenida hasta ahora

La evaluación utiliza PAD-UFES-20, cinco holdouts repetidos con separación por
`patient_id` y el mismo test para todos los modelos de cada repetición.
Imputación, escalamiento y codificación se ajustan únicamente en train.
La selección de checkpoints e hiperparámetros utiliza validación. Se
comparan metadata sola, imágenes solas y todas las fusiones, incluyendo A y B;
se agregan controles con pérdidas iguales y ablaciones de B con igual número
de parámetros. Hay 175 resultados finales verificados por
partición/tarea/configuración, no 175 modelos diferentes.

**Clasificación binaria.** La AUC media de metadata sola es 0.9410 con MLP,
0.9367 con logística y 0.9509 con boosting. A alcanza 0.9506; B original con
focal alcanza 0.9555 y B con BCE 0.9541. Por ejemplo, A prácticamente iguala
a boosting clínico: su diferencia media es -0.03 puntos porcentuales de AUC.
La pequeña ventaja media de B sobre boosting tampoco basta para afirmar una
mejora consistente. Comparar únicamente contra la MLP clínica habría dado
una impresión más favorable del aporte visual.

**Clasificación multiclase.** La BACC media es 0.5993 para metadata MLP,
0.6491 para logística y 0.5573 para boosting, frente a 0.7111 para A y 0.7174
para B. Respecto a logística clínica, las ganancias medias son +6.20 y +6.83
puntos porcentuales, respectivamente. Esto muestra que el tamaño de la
ganancia depende del baseline; las imágenes solas y las fusiones de cada
encoder también deben permanecer en la comparación. Las diferencias frente
a cada referencia individual son análisis secundarios exploratorios.

Además, se eligió una referencia clínica entre MLP, logística y boosting
**con validación en cada partición**, sin escogerla por desempeño en test.
Su resultado no coincide necesariamente con el mayor promedio de test entre
las familias clínicas. Los contrastes principales frente a ese procedimiento
de selección son los siguientes:

| Tarea y métrica | Fusión frente a clínica seleccionada | Ganancia media (puntos porcentuales) | IC95 simultáneo por paciente |
|---|---|---:|---:|
| Binaria, AUC | A, BCE | +0.81 | [-0.76, +2.38] |
| Binaria, AUC | B, BCE | +1.16 | [-0.32, +2.64] |
| Multiclase, BACC | A, focal | +11.26 | [+5.13, +17.39] |
| Multiclase, BACC | B, focal | +11.89 | [+5.57, +18.21] |

Los intervalos binarios incluyen cero; los multiclase quedan por encima de
cero en este análisis. Se calcularon mediante 2000 remuestreos por paciente,
con pesos compartidos entre sus apariciones en distintos tests y ajuste
simultáneo para esos cuatro contrastes. Son intervalos exploratorios
**condicionales a los modelos entrenados y a la selección clínica observada**:
no incluyen toda la incertidumbre de reentrenar ni la dependencia causada
por train compartido. No constituyen una demostración confirmatoria de
superioridad ni una prueba de equivalencia cuando incluyen cero.

El análisis también examina calibración de probabilidades originales,
confusiones por clase y agregación por lesión. Un mayor AUC no garantiza una
mejor sensibilidad, BACC o calibración. Tampoco hay evidencia clara aquí de
superioridad de B sobre A al igualar pérdidas. Los resultados y controles
completos están en
[RESULTADOS_TESIS_COMPLETOS.md](artifacts/comparisons/thesis_completion_v2/RESULTADOS_TESIS_COMPLETOS.md)
y los valores por semilla, intervalos y hashes de predicciones en
`artifacts/comparisons/thesis_completion_v2/analysis.json`.

### Conclusión para este avance y siguiente contribución

La conclusión defendible por ahora es que **una referencia clínica sólida es
indispensable para medir el aporte visual: la ganancia binaria observada es
pequeña y no concluyente, mientras que la fusión presenta una ganancia mayor
en multiclase**. Esto responde parcialmente la pregunta de tesis para estas
representaciones y este dataset. Aún no permite identificar causalmente qué
información visual explica la ganancia, atribuirla exclusivamente al carácter
fundacional de los encoders ni generalizar a otros centros o poblaciones.

Para un futuro paper buscamos una contribución original y demostrable más
allá de reproducir un ranking de modelos. Una dirección que surge de este
avance es estudiar **cuándo conviene incorporar la imagen y cómo fusionarla
cuando la metadata es incompleta o poco informativa**, mediante un método
adaptativo y una evaluación que contemple costo y robustez. Es una propuesta
para discutir, todavía no implementada ni validada como
novedad frente a la literatura o estado del arte. Su eventual contribución requerirá revisar
trabajos específicos, fijar hipótesis y controles antes de medir, y validar
en nuevos pacientes o datos externos.

## Incertidumbre, calibración y alcance

Métricas principales: AUC binaria y BACC multiclase. Se mantienen accuracy,
precisión, recall y F1; se agregan AP (average precision), especificidad binaria,
Brier y ECE de diez intervalos. En multiclase AUC/AP/F1 son macro; Brier es la
suma del error cuadrático por clase y ECE usa confianza de la clase predicha.
Se evalúan las probabilidades originales: no se ajusta un calibrador con test.
La sensibilidad por lesión usa la media de probabilidades por
`(patient_id, lesion_id)` y la misma regla de decisión.

Bootstrap: 2000 remuestreos, semilla 20260928. Se remuestrean pacientes y se
conservan todas sus imágenes. Dentro de cada partición se generan intervalos
pareados. Para la media de los cinco holdouts se remuestrea la unión de
pacientes de test, aplicando el mismo peso a sus apariciones en distintos
holdouts. Se rechazan remuestreos sin todas las clases. Los cuatro contrastes
principales (A/B frente a clínica seleccionada, en cada tarea, con BCE en
binario) tienen intervalos simultáneos mediante la máxima desviación absoluta
estandarizada del bootstrap; los secundarios se señalan como exploratorios.

Estos intervalos son **condicionales a los modelos entrenados y la selección
clínica observada**. Tratan la agrupación de fotos y la repetición de pacientes
en test, pero no toda la variación de reentrenar ni la dependencia debida al
solapamiento de train. El análisis no convierte estos holdouts ya observados
en validación confirmatoria. Evaluación externa/nuevos pacientes e inferencia
con reentrenamiento quedan como ampliaciones, no como resultados demostrados.

## Referencias de investigación

Las fuentes siguientes justifican el diseño y sus comparadores. No se usan
sus cifras como un ranking directamente comparable con el nuestro, ni se
afirma haber auditado sus splits a partir de los resúmenes.

1. Pacheco, A. G. C., et al. (2020). **PAD-UFES-20: A skin lesion dataset
   composed of patient data and clinical images collected from smartphones**.
   *Data in Brief*, 32, 106221.
   [DOI: 10.1016/j.dib.2020.106221](https://doi.org/10.1016/j.dib.2020.106221).
   Fuente primaria del dataset, imágenes clínicas y variables asociadas.
2. Pacheco, A. G. C., y Krohling, R. A. (2020). **The impact of patient clinical
   information on automated skin cancer detection**. *Computers in Biology
   and Medicine*, 116, 103545.
   [DOI: 10.1016/j.compbiomed.2019.103545](https://doi.org/10.1016/j.compbiomed.2019.103545).
   Antecedente para comparar imágenes con/sin información clínica.
3. Ou, C., et al. (2022). **A deep learning based multimodal fusion model for
   skin lesion diagnosis using smartphone collected clinical images and
   metadata**. *Frontiers in Surgery*, 9, 1029991.
   [DOI: 10.3389/fsurg.2022.1029991](https://doi.org/10.3389/fsurg.2022.1029991).
   Antecedente de encoders por modalidad y comparación de métodos de fusión.
4. Deng, J., Guo, E., Zhao, H. J., Venugopal, K., y Moskalyk, M. (2025).
   **Development of a Transfer Learning-Based, Multimodal Neural Network for
   Identifying Malignant Dermatological Lesions From Smartphone Images**.
   *Cancer Informatics*, 24.
   [DOI: 10.1177/11769351251349891](https://doi.org/10.1177/11769351251349891).
   Antecedente directo de comparar red clínica, visual y multimodal en binario,
   incluyendo discriminación y calibración.
5. Radford, A., et al. (2021). **Learning Transferable Visual Models From
   Natural Language Supervision**. *Proceedings of ICML*, PMLR 139.
   [Artículo CLIP](https://arxiv.org/abs/2103.00020).
   Fuente del modelo visual; en esta tesis se utiliza su representación de
   imágenes, sin agregar una rama de texto clínico.
6. Oquab, M., et al. **DINOv2: Learning Robust Visual Features without
   Supervision**. Preprint inicial de 2023.
   [Artículo DINOv2](https://arxiv.org/abs/2304.07193).
   Fuente de las representaciones visuales autosupervisadas.
7. Liu, Z., et al. (2022). **A ConvNet for the 2020s**. *CVPR*.
   [Artículo ConvNeXt](https://arxiv.org/abs/2201.03545).
   Fuente del comparador convolucional preentrenado.
8. Deen, M., y de Rooij, M. (2020). **ClusterBootstrap: An R package for the
   analysis of hierarchical data using generalized linear models with the
   cluster bootstrap**. *Behavior Research Methods*, 52, 572–590.
   [Registro del artículo](https://pubmed.ncbi.nlm.nih.gov/31089956/).
   Apoya remuestrear el sujeto completo cuando existen observaciones
   agrupadas; no valida por sí mismo nuestra extensión a holdouts compartidos.

Documentación de los baselines: [LogisticRegression, sklearn 1.7](https://scikit-learn.org/1.7/modules/generated/sklearn.linear_model.LogisticRegression.html)
y [HistGradientBoostingClassifier, sklearn 1.7](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html).
