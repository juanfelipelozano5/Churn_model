# Modelo de Predicción de Churn — Clientes Bancarios

## Contexto

Prueba técnica para el rol de Analista Jr. de Ciencia de Datos en una compañía financiera que ofrece ahorro, inversión, pensiones y seguros. El área comercial necesita anticipar clientes con mayor probabilidad de fuga para priorizar acciones de retención.

## Dataset

**Bank Customer Churn Modelling** (Kaggle) — 10,000 registros de clientes de un banco europeo con variables demográficas, financieras y de comportamiento.

- **Variable objetivo:** `Exited` (1 = cliente abandonó, 0 = se quedó)
- **Tasa de churn:** ~20.4% (clasificación desbalanceada)

## Estructura del Notebook

| Sección | Descripción |
|---------|-------------|
| 1. Entendimiento del problema | Formulación del objetivo analítico y pregunta de negocio |
| 2. Lectura, limpieza y preparación | Inspección de nulos, duplicados, outliers (IQR), tipos de datos y codificación (One-Hot Encoding) |
| 3. Datos semiestructurados (JSON) | Archivo `business_context.json` con reglas de negocio: segmentos de edad, costos de adquisición/retención, niveles de vinculación y segmento de vida financiera |
| 4. Feature Engineering | Creación de variables: `BalanceSalaryRatio`, `TenureAgeRatio`, `HasBalance`, `NivelVinculacion`, `VidaFinanciera` (con percentiles por género desde JSON) |
| 5. EDA | Análisis de churn por geografía, edad, productos, actividad y correlaciones, orientado a preguntas de negocio |
| 6. Preparación para modelado | Split estratificado 70/15/15 (train/val/test), escalado con StandardScaler sin data leakage |
| 7. Modelado | Regresión Logística (interpretable) y Random Forest (mayor desempeño), ambos con `class_weight='balanced'` |
| 8. Métricas y evaluación | AUC-ROC, KS, precision, recall, F1, matriz de confusión, análisis de umbral (0.3 sugerido por costos de negocio) |
| 9. Estabilidad (PSI) | Population Stability Index entre train y test para validar que el modelo no presenta drift |
| 10. Interpretabilidad (SHAP) | TreeExplainer sobre Random Forest: beeswarm plot y ranking de variables más influyentes |
| 11. Recomendaciones de negocio | Segmentos prioritarios, acciones comerciales concretas y estimación de valor económico |
| Anexo | Ejemplo de despliegue como API REST con FastAPI (no ejecutable en Colab) |

## Tecnologías

- **Python 3.12** en Google Colab
- **Librerías:** pandas, numpy, matplotlib, seaborn, scikit-learn, shap
- **Formato de datos:** CSV (dataset), JSON (contexto de negocio)

## Modelos

| Modelo | Tipo | Propósito |
|--------|------|-----------|
| Regresión Logística | Interpretable (baseline) | Coeficientes directos, explicable al negocio |
| Random Forest | Ensamble (200 árboles, max_depth=10) | Mayor capacidad predictiva, captura no-linealidades |

## Métricas clave

- **AUC-ROC** y **KS** para capacidad discriminativa
- **Precision / Recall / F1** con análisis de umbral
- **PSI < 0.10** para estabilidad del modelo
- **Umbral de clasificación: 0.3** — prioriza recall porque perder un cliente ($500) cuesta 6x más que una campaña de retención innecesaria ($80)

## Estructura del repositorio

```
├── README.md                  # Este archivo
├── requirements.txt           # Dependencias del proyecto
├── curn_model.ipynb           # Notebook principal (Google Colab)
├── business_context.json      # Reglas de negocio en formato JSON
├── Churn_Modelling.csv        # Dataset (descargar de Kaggle)
├── api/                       # Bonus: microservicio
│   └── api.py                 # Endpoint /predict con FastAPI
├── docs/
│   ├── componente_teorico.pdf # Respuestas del componente teórico
│   └── presentacion.pdf       # Presentación ejecutiva (8 slides)
```

## Cómo ejecutar

### Notebook (Google Colab)
1. Abrir `curn_model.ipynb` en Google Colab
2. Subir el archivo `Churn_Modelling.csv` a `/content/`
3. Ejecutar todas las celdas en orden secuencial (de arriba hacia abajo)

### Microservicio (local, opcional)
```bash
pip install -r requirements.txt
cd api/
uvicorn api:app --reload
# Probar: curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"features": [619, 42, 2, 0.0, 1, 1, 1, 101348.88, ...]}'
```

## Uso de IA

### Herramientas utilizadas
- **Claude (Anthropic):** asistencia en estructuración del notebook, explicación de conceptos, debugging de errores y redacción del README.

### Partes asistidas
- Estructura general del notebook y orden de las secciones.
- Debugging de errores de ejecución (KeyError en drop de columnas, ValueError en SHAP, timeout de TreeExplainer).
- Redacción del archivo README.md.
- Ejemplo de código para el microservicio FastAPI.

### Validaciones realizadas
- Cada celda fue ejecutada y verificada manualmente en Google Colab.
- Los resultados de métricas (AUC, precision, recall, F1) fueron interpretados y contrastados con la lógica de negocio.
- El código sugerido por IA fue adaptado al contexto específico del dataset y las reglas de negocio definidas en el JSON.
- Los errores fueron reproducidos y corregidos uno a uno, no copiados sin revisión.

## Uso seguro de IA

### ¿Qué partes fueron asistidas y cuáles validadas manualmente?
La estructura del notebook, el debugging y la documentación fueron asistidos por IA. La ejecución, la interpretación de resultados, las decisiones de negocio (umbral 0.3, segmentos prioritarios) y la validación de métricas fueron realizadas manualmente. Cada celda fue ejecutada en Colab y los resultados verificados antes de avanzar.

### ¿Cómo evitar que instrucciones maliciosas en datos afecten el análisis o un asistente RAG?
- Sanitizar toda entrada de texto antes de procesarla: no ejecutar ni interpretar contenido de campos de texto como instrucciones.
- En un sistema RAG, separar estrictamente el contexto del sistema (system prompt) de los documentos recuperados, tratando estos últimos como datos no confiables.
- Aplicar filtros de entrada que detecten patrones de prompt injection ("ignora las instrucciones", "olvida todo lo anterior") y los descarten o marquen para revisión.
- Validar las salidas del LLM antes de mostrarlas al usuario final, verificando que no contengan información fuera de alcance.

### ¿Qué controles aplicaría para no exponer datos sensibles?
- No incluir datos personales reales en el dataset ni en los entregables (este proyecto usa un dataset público sintético).
- En producción, aplicar anonimización o pseudonimización antes de alimentar cualquier modelo o sistema RAG.
- Restringir acceso a los endpoints del microservicio mediante autenticación (API keys, OAuth).
- No registrar datos de clientes en logs de la API; loguear solo metadata (timestamp, status code, latencia).
- Cumplir con políticas de retención y eliminación de datos según la normativa vigente (Habeas Data en Colombia).

## Autor

Prueba técnica — Analista Jr. de Ciencia de Datos
