# Resultados publicados del protocolo por paciente

Esta carpeta conserva una copia de tablas agregadas y métricas por semilla,
sin filas clínicas ni predicciones por imagen/paciente. No modifica las
corridas originales guardadas localmente.

- [Informe completo](RESULTADOS_TESIS_COMPLETOS.md).
- [Tabla Excel](Resultados_Tesis_Todos_Modelos.xlsx): 175 entrenamientos y
  10 selecciones clínicas derivadas, identificadas por separado.
- `analysis_summary.json`: comparación, intervalos por paciente y selección
  clínica, sin rutas personales ni hashes de archivos de predicciones.
- `*_summary.csv`: medias y desviaciones poblacionales por configuración.
- `*_metrics_per_seed.csv`: métricas de cada entrenamiento final.

Fuentes: las corridas `paper_baseline_v1` (90), `metadata_only_v1` (10),
`thesis_completion_v1` (65) y `metadata_empirical_prior_v1` (10).
Son 175 resultados por partición/tarea/configuración. Las grillas clínicas
incluyen ajustes de candidatos adicionales; no son 175 modelos distintos.

El informe copiado menciona `analysis.json`, checkpoints y predicciones de
la ejecución local. En esta copia pública, el resumen disponible es
`analysis_summary.json`; los artefactos por imagen y checkpoints no se incluyen.
Para reconstruirlos se requieren los datos y caches originales y ejecutar
el protocolo descrito en el README de la raíz.

Los intervalos son exploratorios y condicionales a modelos ya entrenados.
Consultar las limitaciones en el informe antes de interpretar diferencias.
