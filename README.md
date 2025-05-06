# Fine-tuning de Sentence Transformers para el dominio Oil & Gas

Este proyecto realiza el fine-tuning de un modelo de embeddings basado en Sentence Transformers (all-MiniLM-L6-v2) para tareas de similitud semántica en el dominio de Oil & Gas. El pipeline incluye entrenamiento, evaluación, exportación a ONNX, cuantización y despliegue automático en Hugging Face Model Hub mediante CI/CD con GitHub Actions.

## Características principales
- **Entrenamiento supervisado** usando pares de textos y la métrica de similitud coseno.
- **Evaluación automática** de la calidad de los embeddings antes y después del fine-tuning y la cuantización.
- **Exportación a ONNX** y cuantización INT8 para despliegue eficiente.
- **Registro de experimentos** y métricas con MLflow.
- **CI/CD**: Entrenamiento y despliegue automático en Hugging Face Model Hub usando GitHub Actions.

## Estructura del repositorio
- `fine_tune_ampliado.py`: Script principal de entrenamiento, evaluación, exportación y logging.
- `requirements.txt` y `requirements_linux.txt`: Dependencias para Windows y Linux/CI.
- `training_data_quality.jsonl`: Dataset de pares de textos para entrenamiento.
- `fine_tuned_model_oil_gas/`: Carpeta generada con el modelo fine-tuned y archivos auxiliares.
- `model/`: Carpeta con backups versionados de modelos fine-tuned y ONNX cuantizados.
- `.github/workflows/mlflow_hf.yml`: Workflow de GitHub Actions para CI/CD y despliegue en Hugging Face.

## Cómo entrenar y exportar el modelo
1. Instala las dependencias (Linux):
   ```bash
   pip install -r requirements_linux.txt
   ```
2. Ejecuta el script principal:
   ```bash
   python fine_tune_ampliado.py
   ```
3. El modelo fine-tuned y el modelo ONNX cuantizado se guardarán en las carpetas `fine_tuned_model_oil_gas/` y `model/` respectivamente.

## CI/CD y despliegue en Hugging Face
- El workflow de GitHub Actions (`.github/workflows/mlflow_hf.yml`) entrena el modelo desde cero y sube automáticamente los modelos generados al repositorio de modelos de Hugging Face: [jacgandres/fine-tunning-embeddings](https://huggingface.co/jacgandres/fine-tunning-embeddings)
- El token de acceso a Hugging Face debe estar configurado como secreto `HF` en el repositorio de GitHub.

## Requisitos
- Python 3.11+
- Dependencias especificadas en `requirements_linux.txt`
- Dataset en formato JSONL con los campos `text1` y `text2` para cada ejemplo

## Ejemplo de uso del modelo en Hugging Face
Puedes cargar el modelo fine-tuned directamente desde Hugging Face usando:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('jacgandres/fine-tunning-embeddings')
embeddings = model.encode(["Texto de ejemplo 1", "Texto de ejemplo 2"])
```

## Créditos
- Basado en [Sentence Transformers](https://www.sbert.net/)
- Exportación y cuantización con [Optimum](https://huggingface.co/docs/optimum/index)
- CI/CD con [GitHub Actions](https://github.com/features/actions)
- Despliegue en [Hugging Face Model Hub](https://huggingface.co/)

---

**Autor:** jacgandres

**Fecha de última actualización:** 2025-05-05
