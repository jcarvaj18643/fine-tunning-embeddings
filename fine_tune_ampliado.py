import logging
import os
import numpy as np
import torch
import joblib
import shutil
from datetime import date
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
import onnxruntime as ort

# --- Configuración ---
model_name = "all-MiniLM-L6-v2"
train_dataset_path = "training_data_quality.jsonl"
output_path = "fine_tuned_model_oil_gas"
num_train_epochs = 10
batch_size = 64

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Usando dispositivo: {device}")

# Borrar el folder de salida si existe antes de iniciar el fine-tuning
if os.path.exists(output_path):
    shutil.rmtree(output_path)
    logging.info(f"Carpeta '{output_path}' eliminada antes de iniciar el fine-tuning.")

# --- Carga y fine-tuning ---
model = SentenceTransformer(model_name, device=device)
dataset = load_dataset("json", data_files=train_dataset_path, split="train")
train_examples = [
    InputExample(texts=[ex["text1"], ex["text2"]], label=1.0)
    for ex in dataset
    if ex.get("text1") and ex.get("text2")
]
logging.info(f"Ejemplos válidos para entrenamiento: {len(train_examples)}")

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size, collate_fn=lambda x: x)
train_loss = losses.CosineSimilarityLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_train_epochs,
    warmup_steps=int(0.1 * len(train_dataloader) * num_train_epochs),
    output_path=output_path,
    use_amp=torch.cuda.is_available(),
    show_progress_bar=True
)
model.save(output_path)
logging.info("Modelo fine-tuned guardado.")

# --- Evaluación embeddings base vs. fine-tuned ---
eval_texts = train_examples[:100]
texts1 = [ex.texts[0] for ex in eval_texts]
texts2 = [ex.texts[1] for ex in eval_texts]

# Original
orig_model = SentenceTransformer(model_name, device=device)
emb1_orig = orig_model.encode(texts1, convert_to_numpy=True, batch_size=32, show_progress_bar=False)
emb2_orig = orig_model.encode(texts2, convert_to_numpy=True, batch_size=32, show_progress_bar=False)
cos_sims_orig = [cosine_similarity([a], [b])[0,0] for a,b in zip(emb1_orig, emb2_orig)]

# Fine-tuned
tuned_model = SentenceTransformer(output_path, device=device)
emb1_tuned = tuned_model.encode(texts1, convert_to_numpy=True, batch_size=32, show_progress_bar=False)
emb2_tuned = tuned_model.encode(texts2, convert_to_numpy=True, batch_size=32, show_progress_bar=False)
cos_sims_tuned = [cosine_similarity([a], [b])[0,0] for a,b in zip(emb1_tuned, emb2_tuned)]

print("\n--- Métricas antes de cuantizar ---")
print(f"Original   → coseno promedio: {np.mean(cos_sims_orig):.4f}")
print(f"Fine-tuned → coseno promedio: {np.mean(cos_sims_tuned):.4f}")

# --- Exportar a ONNX y cuantizar ---
onnx_dir = os.path.join(output_path, "onnx")
os.makedirs(onnx_dir, exist_ok=True)

# Export ONNX
tokenizer = AutoTokenizer.from_pretrained(output_path)
ort_model = ORTModelForFeatureExtraction.from_pretrained(
    output_path, export=True, provider="CPUExecutionProvider"
)
ort_model.save_pretrained(onnx_dir)
tokenizer.save_pretrained(onnx_dir)

# Cuantización (especificando el nombre del fichero original)
quantizer = ORTQuantizer.from_pretrained(onnx_dir, file_name="model.onnx")
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
quantizer.quantize(save_dir=onnx_dir, quantization_config=qconfig)

quantized_path = os.path.join(onnx_dir, "model_quantized.onnx")
if not os.path.exists(quantized_path):
    raise FileNotFoundError(f"No se encontró {quantized_path} tras cuantizar.")

# --- Función de embeddings ONNX con mean pooling ---
def get_onnx_embeddings(texts, model_path, tokenizer_path):
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    inputs = tok(texts, padding=True, truncation=True, return_tensors="np")
    mask = inputs["attention_mask"][..., None]
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    ort_inputs = {k: v for k, v in inputs.items() if k in {i.name for i in session.get_inputs()}}
    last_hidden = session.run(None, ort_inputs)[0]  # (batch, seq_len, hidden)
    summed = (last_hidden * mask).sum(axis=1)
    lengths = mask.sum(axis=1)
    return summed / lengths  # (batch, hidden)

# --- Evaluación modelo cuantizado ---
emb1_quant = get_onnx_embeddings(texts1, quantized_path, onnx_dir)
emb2_quant = get_onnx_embeddings(texts2, quantized_path, onnx_dir)
cos_sims_quant = [cosine_similarity([a], [b])[0,0] for a,b in zip(emb1_quant, emb2_quant)]

print("\n--- Métricas después de cuantizar ---")
print(f"Fine-tuned cuantizado INT8 → coseno promedio: {np.mean(cos_sims_quant):.4f}")

# --- Comparación final y guardado del mejor ---
results = {
    "original": np.mean(cos_sims_orig),
    "fine-tuned": np.mean(cos_sims_tuned),
    "cuantizado INT8": np.mean(cos_sims_quant),
}

cos_sims_orig = np.array(cos_sims_orig)
cos_sims_tuned = np.array(cos_sims_tuned)
cos_sims_quant = np.array(cos_sims_quant)

print("\n--- Métricas finales de comparación ---")
print(f"Similitud coseno promedio modelo original: {cos_sims_orig.mean():.4f}")
print(f"Similitud coseno promedio modelo fine-tuned: {cos_sims_tuned.mean():.4f}")
print(f"Similitud coseno promedio modelo fine-tuned cuantizado INT8: {cos_sims_quant.mean():.4f}")

best_name, best_score = max(results.items(), key=lambda x: x[1])
print(f"\n🏆 Veredicto: '{best_name}' con similitud {best_score:.4f}")

# --- Guardar siempre la versión ONNX cuantizada en un folder con fecha ---
dest_dir = f"model/onnx-oil-gas-int8-{date.today().strftime('%Y%m%d')}"
if os.path.exists(dest_dir):
    shutil.rmtree(dest_dir)
shutil.copytree(onnx_dir, dest_dir)
logging.info(f"ONNX cuantizado guardado en {dest_dir}")
# Ahora el modelo está listo para ser subido a Hugging Face o expuesto por CI/CD.

# --- Guardar también el modelo fine-tuned completo en un folder con fecha ---
tuned_dest_dir = f"model/tuned-oil-gas-{date.today().strftime('%Y%m%d')}"
if os.path.exists(tuned_dest_dir):
    shutil.rmtree(tuned_dest_dir)
shutil.copytree(output_path, tuned_dest_dir)
logging.info(f"Modelo fine-tuned guardado en {tuned_dest_dir}")
