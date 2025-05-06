import logging
import os
import numpy as np
import torch
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
import mlflow
import mlflow.pyfunc

# --- Configuración ---
def get_config():
    return {
        "model_name": "all-MiniLM-L6-v2",
        "train_dataset_path": "training_data_quality.jsonl",
        "output_path": "fine_tuned_model_oil_gas",
        "num_train_epochs": 20,
        "batch_size": 64
    }

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        logging.info(f"Carpeta '{path}' eliminada antes de iniciar el fine-tuning.")

def prepare_train_examples(dataset_path):
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    train_examples = [
        InputExample(texts=[ex["text1"], ex["text2"]], label=1.0)
        for ex in dataset
        if ex.get("text1") and ex.get("text2")
    ]
    logging.info(f"Ejemplos válidos para entrenamiento: {len(train_examples)}")
    return train_examples

def train_and_save_model(model_name, train_examples, output_path, num_train_epochs, batch_size, device):
    model = SentenceTransformer(model_name, device=device)
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
    return model

def evaluate_embeddings(model, texts1, texts2):
    emb1 = model.encode(texts1, convert_to_numpy=True, batch_size=32, show_progress_bar=False)
    emb2 = model.encode(texts2, convert_to_numpy=True, batch_size=32, show_progress_bar=False)
    cos_sims = [cosine_similarity([a], [b])[0,0] for a,b in zip(emb1, emb2)]
    return np.array(cos_sims)

def export_and_quantize_onnx(output_path):
    onnx_dir = os.path.join(output_path, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(output_path)
    ort_model = ORTModelForFeatureExtraction.from_pretrained(
        output_path, export=True, provider="CPUExecutionProvider"
    )
    ort_model.save_pretrained(onnx_dir)
    tokenizer.save_pretrained(onnx_dir)
    quantizer = ORTQuantizer.from_pretrained(onnx_dir, file_name="model.onnx")
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
    quantizer.quantize(save_dir=onnx_dir, quantization_config=qconfig)
    quantized_path = os.path.join(onnx_dir, "model_quantized.onnx")
    if not os.path.exists(quantized_path):
        raise FileNotFoundError(f"No se encontró {quantized_path} tras cuantizar.")
    return onnx_dir, quantized_path

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

def save_with_date(src, prefix):
    dest_dir = f"model/{prefix}-{date.today().strftime('%Y%m%d')}"
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    shutil.copytree(src, dest_dir)
    logging.info(f"Guardado en {dest_dir}")
    return dest_dir

def main():
    setup_logging()
    config = get_config()
    device = get_device()
    logging.info(f"Usando dispositivo: {device}")
    clean_folder(config["output_path"])
    train_examples = prepare_train_examples(config["train_dataset_path"])
    
    mlflow.set_experiment("fine_tune_oil_gas")
    with mlflow.start_run():
        mlflow.log_params({
            "model_name": config["model_name"],
            "num_train_epochs": config["num_train_epochs"],
            "batch_size": config["batch_size"]
        })
        model = train_and_save_model(
            config["model_name"], train_examples, config["output_path"],
            config["num_train_epochs"], config["batch_size"], device
        )
        eval_texts = train_examples[:100]
        texts1 = [ex.texts[0] for ex in eval_texts]
        texts2 = [ex.texts[1] for ex in eval_texts]
        # Original
        orig_model = SentenceTransformer(config["model_name"], device=device)
        cos_sims_orig = evaluate_embeddings(orig_model, texts1, texts2)
        # Fine-tuned
        tuned_model = SentenceTransformer(config["output_path"], device=device)
        cos_sims_tuned = evaluate_embeddings(tuned_model, texts1, texts2)
        mlflow.log_metrics({
            "cosine_similarity_original": float(cos_sims_orig.mean()),
            "cosine_similarity_finetuned": float(cos_sims_tuned.mean())
        })
        print("\n--- Métricas antes de cuantizar ---")
        print(f"Original   → coseno promedio: {cos_sims_orig.mean():.4f}")
        print(f"Fine-tuned → coseno promedio: {cos_sims_tuned.mean():.4f}")
        # Exportar y cuantizar
        onnx_dir, quantized_path = export_and_quantize_onnx(config["output_path"])
        emb1_quant = get_onnx_embeddings(texts1, quantized_path, onnx_dir)
        emb2_quant = get_onnx_embeddings(texts2, quantized_path, onnx_dir)
        cos_sims_quant = np.array([cosine_similarity([a], [b])[0,0] for a,b in zip(emb1_quant, emb2_quant)])
        mlflow.log_metric("cosine_similarity_quantized", float(cos_sims_quant.mean()))
        print("\n--- Métricas después de cuantizar ---")
        print(f"Fine-tuned cuantizado INT8 → coseno promedio: {cos_sims_quant.mean():.4f}")
        # Comparación final
        results = {
            "original": cos_sims_orig.mean(),
            "fine-tuned": cos_sims_tuned.mean(),
            "cuantizado INT8": cos_sims_quant.mean(),
        }
        print("\n--- Métricas finales de comparación ---")
        print(f"Similitud coseno promedio modelo original: {cos_sims_orig.mean():.4f}")
        print(f"Similitud coseno promedio modelo fine-tuned: {cos_sims_tuned.mean():.4f}")
        print(f"Similitud coseno promedio modelo fine-tuned cuantizado INT8: {cos_sims_quant.mean():.4f}")
        best_name, best_score = max(results.items(), key=lambda x: x[1])
        print(f"\n🏆 Veredicto: '{best_name}' con similitud {best_score:.4f}")
        # Guardar modelos
        save_with_date(onnx_dir, "onnx-oil-gas-int8")
        save_with_date(config["output_path"], "tuned-oil-gas")
        # Log artifacts
        mlflow.log_artifacts(config["output_path"], artifact_path="finetuned_model")
        mlflow.log_artifacts(onnx_dir, artifact_path="onnx_model")

if __name__ == "__main__":
    main()
