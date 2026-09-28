# Investigación de tesis: comparación completa

¿Cuánto aportan los modelos fundacionales de imágenes cuando ya disponemos de metadata clínica, y qué ganancia adicional produce fusionar ambas fuentes?

Evaluación en PAD-UFES-20: mismos cinco holdouts por paciente, estadísticas solo de train y selección con validación. Se conservaron los 100 entrenamientos previos y se completaron 75 referencias/controles nuevos (65 en thesis_completion_v1 y 10 priors empíricos).

## Qué debe compararse

Se incluyen todos los modelos. A y B son contrastes principales; los encoders individuales, sus fusiones y las referencias clínicas explican el origen de la ganancia. La referencia clínica principal se elige entre MLP, logística y boosting con validación en cada semilla/tarea; nunca por resultados de test.

El prior ponderado usa pesos balanceados en multiclase (por ello es uniforme); el prior empírico adicional usa las frecuencias originales de train sin ponderar.

## Selección clínica por validación

| Semilla | Tarea | Modelo elegido |
|---|---|---|
| 42 | binary | Metadata MLP |
| 123 | binary | Metadata boosting |
| 456 | binary | Metadata MLP |
| 789 | binary | Metadata boosting |
| 2024 | binary | Metadata boosting |
| 42 | multi | Metadata MLP |
| 123 | multi | Metadata logística |
| 456 | multi | Metadata logística |
| 789 | multi | Metadata MLP |
| 2024 | multi | Metadata MLP |

## Binaria: todos los modelos

Media ± desviación poblacional; umbral 0.5/argmax. AP=average precision. Brier multiclase=sumatoria por clase, sin dividir por seis.

| Modelo | AUC/OvR | BACC | F1 | AP | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| CLIP | 0.9023 ± 0.0196 | 0.8283 ± 0.0137 | 0.8231 ± 0.0138 | 0.8741 ± 0.0187 | 0.1274 ± 0.0162 | 0.0753 ± 0.0170 |
| DINOv2 | 0.8818 ± 0.0194 | 0.7921 ± 0.0108 | 0.7868 ± 0.0145 | 0.8573 ± 0.0354 | 0.1494 ± 0.0119 | 0.1063 ± 0.0107 |
| ConvNeXt | 0.8662 ± 0.0190 | 0.7930 ± 0.0362 | 0.7918 ± 0.0290 | 0.8314 ± 0.0304 | 0.1543 ± 0.0151 | 0.0938 ± 0.0090 |
| CLIP + DINOv2 | 0.9046 ± 0.0201 | 0.8110 ± 0.0237 | 0.7996 ± 0.0321 | 0.8832 ± 0.0287 | 0.1346 ± 0.0128 | 0.0916 ± 0.0109 |
| CLIP + metadata | 0.9497 ± 0.0062 | 0.8872 ± 0.0134 | 0.8847 ± 0.0127 | 0.9290 ± 0.0172 | 0.0910 ± 0.0105 | 0.0687 ± 0.0216 |
| DINOv2 + metadata | 0.9473 ± 0.0144 | 0.8753 ± 0.0303 | 0.8707 ± 0.0310 | 0.9252 ± 0.0259 | 0.0952 ± 0.0237 | 0.0742 ± 0.0262 |
| ConvNeXt + metadata | 0.9451 ± 0.0130 | 0.8750 ± 0.0216 | 0.8710 ± 0.0218 | 0.9218 ± 0.0237 | 0.1002 ± 0.0218 | 0.0821 ± 0.0301 |
| A (BCE binaria) | 0.9506 ± 0.0144 | 0.8828 ± 0.0226 | 0.8791 ± 0.0219 | 0.9286 ± 0.0284 | 0.0933 ± 0.0201 | 0.0755 ± 0.0275 |
| B (focal binaria) | 0.9555 ± 0.0112 | 0.8731 ± 0.0203 | 0.8615 ± 0.0261 | 0.9342 ± 0.0297 | 0.1011 ± 0.0076 | 0.1275 ± 0.0296 |
| Metadata MLP | 0.9410 ± 0.0157 | 0.8692 ± 0.0246 | 0.8661 ± 0.0244 | 0.9155 ± 0.0262 | 0.0970 ± 0.0175 | 0.0641 ± 0.0175 |
| Metadata logística | 0.9367 ± 0.0202 | 0.8695 ± 0.0248 | 0.8644 ± 0.0245 | 0.9194 ± 0.0280 | 0.0980 ± 0.0157 | 0.0626 ± 0.0152 |
| Metadata boosting | 0.9509 ± 0.0140 | 0.8790 ± 0.0201 | 0.8741 ± 0.0219 | 0.9292 ± 0.0218 | 0.0853 ± 0.0131 | 0.0605 ± 0.0088 |
| Prior ponderado en multiclase | 0.5000 ± 0.0000 | 0.5000 ± 0.0000 | 0.0000 ± 0.0000 | 0.4722 ± 0.0021 | 0.2492 ± 0.0001 | 0.0021 ± 0.0025 |
| B solo metadata, misma capacidad | 0.9468 ± 0.0137 | 0.8592 ± 0.0198 | 0.8482 ± 0.0239 | 0.9207 ± 0.0296 | 0.1174 ± 0.0066 | 0.1531 ± 0.0223 |
| B solo imágenes, misma capacidad | 0.9084 ± 0.0198 | 0.7788 ± 0.0518 | 0.7314 ± 0.0926 | 0.8867 ± 0.0312 | 0.1508 ± 0.0151 | 0.1483 ± 0.0359 |
| A focal | 0.9540 ± 0.0114 | 0.8830 ± 0.0174 | 0.8780 ± 0.0185 | 0.9318 ± 0.0236 | 0.0876 ± 0.0096 | 0.0823 ± 0.0326 |
| B BCE | 0.9541 ± 0.0135 | 0.8912 ± 0.0140 | 0.8863 ± 0.0144 | 0.9282 ± 0.0349 | 0.0837 ± 0.0133 | 0.0659 ± 0.0209 |
| Metadata MLP focal | 0.9443 ± 0.0151 | 0.8517 ± 0.0248 | 0.8376 ± 0.0295 | 0.9162 ± 0.0329 | 0.1193 ± 0.0076 | 0.1554 ± 0.0309 |
| Prior empírico de train | 0.5000 ± 0.0000 | 0.5000 ± 0.0000 | 0.0000 ± 0.0000 | 0.4722 ± 0.0021 | 0.2492 ± 0.0001 | 0.0021 ± 0.0025 |
| Metadata elegida con validación | 0.9425 ± 0.0160 | 0.8699 ± 0.0186 | 0.8665 ± 0.0176 | 0.9186 ± 0.0218 | 0.0945 ± 0.0182 | 0.0698 ± 0.0144 |

### Contrastes principales frente a la referencia clínica elegida

Diferencias e intervalos en puntos porcentuales. IC simultáneo: familia de cuatro contrastes; bootstrap por paciente con pesos compartidos entre repeticiones.

| Comparación | Ganancia media | IC95 simultáneo | Repeticiones positivas / 5 |
|---|---:|---:|---:|
| A (BCE binaria) − clínica elegida | +0.81 | [-0.76, +2.38] | 2 |
| B BCE − clínica elegida | +1.16 | [-0.32, +2.64] | 3 |

### Controles de pérdida y capacidad

| Contraste | Diferencia media | IC95 por paciente (sin ajuste de multiplicidad) |
|---|---:|---:|
| B (focal binaria) − B solo metadata, misma capacidad | +0.87 | [-0.06, +1.76] |
| B (focal binaria) − B solo imágenes, misma capacidad | +4.71 | [+2.98, +6.52] |
| B BCE − A (BCE binaria) | +0.35 | [-0.22, +0.99] |
| B (focal binaria) − A focal | +0.15 | [-0.52, +0.90] |
| A focal − A (BCE binaria) | +0.34 | [-0.13, +0.82] |
| B (focal binaria) − B BCE | +0.14 | [-0.35, +0.70] |
| Metadata MLP focal − Metadata MLP | +0.33 | [-0.04, +0.71] |
| CLIP + metadata − Metadata MLP | +0.87 | [-0.18, +1.89] |
| CLIP + metadata − Metadata logística | +1.30 | [+0.17, +2.65] |
| CLIP + metadata − Metadata boosting | -0.12 | [-1.40, +1.03] |
| DINOv2 + metadata − Metadata MLP | +0.63 | [-0.48, +1.75] |
| DINOv2 + metadata − Metadata logística | +1.06 | [-0.00, +2.24] |
| DINOv2 + metadata − Metadata boosting | -0.36 | [-1.58, +0.70] |
| ConvNeXt + metadata − Metadata MLP | +0.41 | [-0.65, +1.45] |
| ConvNeXt + metadata − Metadata logística | +0.84 | [-0.29, +2.07] |
| ConvNeXt + metadata − Metadata boosting | -0.58 | [-1.78, +0.51] |
| A (BCE binaria) − Metadata MLP | +0.97 | [-0.26, +2.23] |
| A (BCE binaria) − Metadata logística | +1.40 | [+0.25, +2.77] |
| A (BCE binaria) − Metadata boosting | -0.03 | [-1.24, +1.14] |
| B (focal binaria) − Metadata MLP | +1.46 | [+0.48, +2.46] |
| B (focal binaria) − Metadata logística | +1.89 | [+0.78, +3.23] |
| B (focal binaria) − Metadata boosting | +0.46 | [-0.65, +1.46] |
| A focal − Metadata MLP | +1.31 | [+0.20, +2.48] |
| A focal − Metadata logística | +1.73 | [+0.64, +3.08] |
| A focal − Metadata boosting | +0.31 | [-0.78, +1.40] |
| B BCE − Metadata MLP | +1.32 | [+0.19, +2.49] |
| B BCE − Metadata logística | +1.75 | [+0.69, +3.03] |
| B BCE − Metadata boosting | +0.32 | [-0.73, +1.38] |

### Sensibilidad al evaluar por lesión

Promedio aritmético de probabilidades de sus imágenes, agrupando (patient_id, lesion_id), sin seleccionar la regla con test.

| Modelo | Métrica principal por imagen | Métrica principal por lesión |
|---|---:|---:|
| CLIP | 0.9023 | 0.9068 |
| DINOv2 | 0.8818 | 0.8798 |
| ConvNeXt | 0.8662 | 0.8694 |
| CLIP + DINOv2 | 0.9046 | 0.9055 |
| A (BCE binaria) | 0.9506 | 0.9541 |
| B (focal binaria) | 0.9555 | 0.9579 |
| B BCE | 0.9541 | 0.9569 |
| Metadata elegida con validación | 0.9425 | 0.9474 |

## Multiclase: todos los modelos

Media ± desviación poblacional; umbral 0.5/argmax. AP=average precision. Brier multiclase=sumatoria por clase, sin dividir por seis.

| Modelo | AUC/OvR | BACC | F1 | AP | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|
| CLIP | 0.9183 ± 0.0101 | 0.6509 ± 0.0292 | 0.6081 ± 0.0199 | 0.6968 ± 0.0190 | 0.4856 ± 0.0278 | 0.1278 ± 0.0429 |
| DINOv2 | 0.8981 ± 0.0117 | 0.6280 ± 0.0398 | 0.6105 ± 0.0212 | 0.6659 ± 0.0281 | 0.4865 ± 0.0231 | 0.0880 ± 0.0341 |
| ConvNeXt | 0.8729 ± 0.0209 | 0.5648 ± 0.0475 | 0.5237 ± 0.0586 | 0.5905 ± 0.0583 | 0.5518 ± 0.0484 | 0.0809 ± 0.0255 |
| CLIP + DINOv2 | 0.9186 ± 0.0094 | 0.6441 ± 0.0339 | 0.6204 ± 0.0233 | 0.7015 ± 0.0229 | 0.4532 ± 0.0324 | 0.1165 ± 0.0246 |
| CLIP + metadata | 0.9346 ± 0.0082 | 0.7095 ± 0.0149 | 0.6760 ± 0.0173 | 0.7395 ± 0.0452 | 0.3928 ± 0.0216 | 0.0915 ± 0.0265 |
| DINOv2 + metadata | 0.9290 ± 0.0105 | 0.6807 ± 0.0442 | 0.6550 ± 0.0358 | 0.7232 ± 0.0307 | 0.3966 ± 0.0422 | 0.1064 ± 0.0268 |
| ConvNeXt + metadata | 0.9207 ± 0.0138 | 0.6494 ± 0.0350 | 0.6177 ± 0.0566 | 0.6815 ± 0.0468 | 0.4226 ± 0.0530 | 0.0765 ± 0.0369 |
| A (BCE binaria) | 0.9381 ± 0.0077 | 0.7111 ± 0.0336 | 0.6870 ± 0.0296 | 0.7511 ± 0.0312 | 0.3769 ± 0.0178 | 0.0916 ± 0.0626 |
| B (focal binaria) | 0.9383 ± 0.0062 | 0.7174 ± 0.0353 | 0.6884 ± 0.0442 | 0.7408 ± 0.0499 | 0.3831 ± 0.0194 | 0.1070 ± 0.0187 |
| Metadata MLP | 0.9086 ± 0.0118 | 0.5993 ± 0.0519 | 0.5632 ± 0.0497 | 0.6475 ± 0.0474 | 0.4877 ± 0.0260 | 0.1165 ± 0.0344 |
| Metadata logística | 0.8995 ± 0.0088 | 0.6491 ± 0.0302 | 0.5900 ± 0.0125 | 0.6434 ± 0.0359 | 0.4703 ± 0.0324 | 0.0864 ± 0.0379 |
| Metadata boosting | 0.9091 ± 0.0106 | 0.5573 ± 0.0184 | 0.5567 ± 0.0196 | 0.6497 ± 0.0216 | 0.4187 ± 0.0156 | 0.0722 ± 0.0140 |
| Prior ponderado en multiclase | 0.5000 ± 0.0000 | 0.1667 ± 0.0000 | 0.0821 ± 0.0035 | 0.1667 ± 0.0000 | 0.8333 ± 0.0000 | 0.1603 ± 0.0191 |
| B solo metadata, misma capacidad | 0.8907 ± 0.0197 | 0.5549 ± 0.0572 | 0.5039 ± 0.0744 | 0.5919 ± 0.0651 | 0.5427 ± 0.0600 | 0.1108 ± 0.0358 |
| B solo imágenes, misma capacidad | 0.9203 ± 0.0071 | 0.6343 ± 0.0340 | 0.6181 ± 0.0498 | 0.7069 ± 0.0232 | 0.4577 ± 0.0484 | 0.0927 ± 0.0348 |
| Prior empírico de train | 0.5000 ± 0.0000 | 0.1667 ± 0.0000 | 0.0895 ± 0.0004 | 0.1667 ± 0.0000 | 0.7351 ± 0.0013 | 0.0027 ± 0.0004 |
| Metadata elegida con validación | 0.9084 ± 0.0112 | 0.5985 ± 0.0471 | 0.5679 ± 0.0406 | 0.6490 ± 0.0460 | 0.4646 ± 0.0273 | 0.0821 ± 0.0459 |

### Contrastes principales frente a la referencia clínica elegida

Diferencias e intervalos en puntos porcentuales. IC simultáneo: familia de cuatro contrastes; bootstrap por paciente con pesos compartidos entre repeticiones.

| Comparación | Ganancia media | IC95 simultáneo | Repeticiones positivas / 5 |
|---|---:|---:|---:|
| A (BCE binaria) − clínica elegida | +11.26 | [+5.13, +17.39] | 5 |
| B (focal binaria) − clínica elegida | +11.89 | [+5.57, +18.21] | 5 |

### Controles de pérdida y capacidad

| Contraste | Diferencia media | IC95 por paciente (sin ajuste de multiplicidad) |
|---|---:|---:|
| B (focal binaria) − B solo metadata, misma capacidad | +16.26 | [+11.42, +21.68] |
| B (focal binaria) − B solo imágenes, misma capacidad | +8.31 | [+2.40, +15.72] |
| B (focal binaria) − A (BCE binaria) | +0.63 | [-2.88, +4.43] |
| CLIP + metadata − Metadata MLP | +11.02 | [+6.88, +15.85] |
| CLIP + metadata − Metadata logística | +6.04 | [+1.02, +11.36] |
| CLIP + metadata − Metadata boosting | +15.22 | [+10.26, +21.20] |
| DINOv2 + metadata − Metadata MLP | +8.15 | [+2.26, +13.18] |
| DINOv2 + metadata − Metadata logística | +3.16 | [-1.40, +7.61] |
| DINOv2 + metadata − Metadata boosting | +12.34 | [+6.68, +18.06] |
| ConvNeXt + metadata − Metadata MLP | +5.01 | [+0.37, +9.97] |
| ConvNeXt + metadata − Metadata logística | +0.03 | [-5.91, +5.63] |
| ConvNeXt + metadata − Metadata boosting | +9.21 | [+3.11, +15.79] |
| A (BCE binaria) − Metadata MLP | +11.19 | [+6.12, +16.01] |
| A (BCE binaria) − Metadata logística | +6.20 | [+1.06, +11.54] |
| A (BCE binaria) − Metadata boosting | +15.38 | [+10.25, +20.95] |
| B (focal binaria) − Metadata MLP | +11.82 | [+7.03, +16.83] |
| B (focal binaria) − Metadata logística | +6.83 | [+1.67, +12.22] |
| B (focal binaria) − Metadata boosting | +16.01 | [+10.70, +22.10] |

### Sensibilidad al evaluar por lesión

Promedio aritmético de probabilidades de sus imágenes, agrupando (patient_id, lesion_id), sin seleccionar la regla con test.

| Modelo | Métrica principal por imagen | Métrica principal por lesión |
|---|---:|---:|
| CLIP | 0.6509 | 0.6511 |
| DINOv2 | 0.6280 | 0.6192 |
| ConvNeXt | 0.5648 | 0.5609 |
| CLIP + DINOv2 | 0.6441 | 0.6541 |
| A (BCE binaria) | 0.7111 | 0.7148 |
| B (focal binaria) | 0.7174 | 0.7232 |
| Metadata elegida con validación | 0.5985 | 0.6001 |

## Verificación, incertidumbre y límites

Además del selector clínico, analysis.json conserva contrastes exploratorios contra MLP, logística y boosting individualmente. Elegir por validación puede no coincidir con el mejor score de test: no debe presentarse al selector como el mejor modelo de test ni ocultarse el desempeño de las referencias individuales.

Se recalcularon seis métricas desde todas las predicciones; IDs/etiquetas y preprocesadores coinciden con los mismos splits. Los modelos tabulares guardan candidatos y configuración seleccionada; se verificaron sus predicciones al recargar el modelo. Los controles B tienen exactamente el mismo número de parámetros que B completo; las ramas excluidas reciben ceros constantes y sus sesgos siguen siendo entrenables.

Bootstrap: 2000 remuestreos, semilla 20260928. Dentro de cada split se remuestrean pacientes con todas sus imágenes. Para la media de repeticiones se remuestrea la unión de pacientes de test y se aplica el mismo peso a cada aparición del paciente. Se rechazan remuestreos sin alguna de las seis clases y se registran los rechazos.

Los intervalos son condicionales a los modelos ya entrenados y a la selección clínica ya realizada. Capturan agrupación por paciente y sus apariciones en test; no capturan la incertidumbre adicional de reentrenar ni la dependencia inducida por train compartido. No equivalen a una evaluación confirmatoria independiente. No se usaron p-valores ni se declaró equivalencia.

Calibración: Brier, ECE y tablas de confiabilidad de probabilidades originales, no calibradores ajustados con test. Binario evalúa probabilidad de malignidad; multiclase evalúa confianza top-label. Las matrices de confusión, recalls por clase, especificidad binaria y métricas por lesión están en analysis.json.

Queda como ampliación de investigación validar en nuevos datos/pacientes y estudiar incertidumbre de reentrenamiento con un diseño externo o anidado. No se afirma utilidad clínica ni generalización a otros datasets.
