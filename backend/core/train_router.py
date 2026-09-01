import os
import threading
import uuid
import logging
from pathlib import Path
from flask import Blueprint, jsonify, request

from src.data_collection.label_registry import LabelRegistry
from src.data_collection.collection_stats import CollectionStats
from src.data_collection.video_processor import VideoProcessor
from src.data_collection.dataset_builder import DatasetBuilder
from src.training.trainer import train, TrainingConfig

train_blueprint = Blueprint("train", __name__, url_prefix="/api/train")
logger = logging.getLogger(__name__)

# Global state for training jobs
# For production, this should ideally be handled by Redis/Celery or a DB.
training_jobs = {}

@train_blueprint.route("/labels", methods=["GET"])
def get_labels():
    try:
        registry = LabelRegistry()
        categories = registry.get_categories()
        data = {}
        for cat in categories:
            data[cat] = registry.get_labels(cat)
        return jsonify({"status": "success", "categories": data})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@train_blueprint.route("/stats", methods=["GET"])
def get_stats():
    try:
        stats = CollectionStats()
        counts = stats.get_class_counts()
        return jsonify({"status": "success", "counts": counts})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@train_blueprint.route("/upload_video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"status": "error", "message": "No video file provided"}), 400
    
    category = request.form.get("category")
    label = request.form.get("label")
    
    if not category or not label:
        return jsonify({"status": "error", "message": "category and label are required"}), 400
        
    video_file = request.files["video"]
    # Save the file temporarily
    temp_path = f"temp_{video_file.filename}"
    video_file.save(temp_path)
    
    try:
        with VideoProcessor() as processor:
            saved_path = processor.process_and_save(temp_path, category, label)
        
        # Cleanup temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return jsonify({"status": "success", "saved_path": str(saved_path)})
    except Exception as e:
        # Cleanup on error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"status": "error", "message": str(e)}), 500


def _run_training_job(job_id: str, category: str):
    """Background thread to handle dataset building and training."""
    training_jobs[job_id]["status"] = "running"
    training_jobs[job_id]["progress"] = "Building dataset..."
    
    try:
        # 1. Build dataset (merges all collected .npy files into one .pt file)
        builder = DatasetBuilder(output_path="data/merged_dataset.pt")
        builder.build_dataset(mode="all") 
        
        # 2. Get registry to filter label IDs
        registry = LabelRegistry()
        label_names = registry.get_labels(category)
        label_ids = []
        for name in label_names:
            lid = registry.get_label_id(name, category)
            if lid is not None:
                label_ids.append(lid)
                
        if not label_ids:
             raise ValueError(f"No labels found for category {category}")
             
        # 3. Train the model
        training_jobs[job_id]["progress"] = f"Training model for {category}..."
        
        output_dir = f"artifacts/models/{category.lower()}"
        config = TrainingConfig(
            dataset_path="data/merged_dataset.pt",
            output_dir=output_dir,
            model_name=f"{category.lower()}_lstm",
            category=category,
            label_ids=label_ids,
            epochs=50,
            batch_size=8
        )
        
        results = train(config)
        
        training_jobs[job_id]["status"] = "completed"
        training_jobs[job_id]["progress"] = "Training finished"
        training_jobs[job_id]["results"] = {
            "best_val_accuracy": results.get("best_val_accuracy"),
            "model_path": str(Path(output_dir) / f"{category.lower()}_lstm_best.pt")
        }
        
    except Exception as e:
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["error"] = str(e)
        logger.error(f"Training job {job_id} failed: {e}")


@train_blueprint.route("/start", methods=["POST"])
def start_training():
    data = request.json or {}
    category = data.get("category")
    
    if not category:
         return jsonify({"status": "error", "message": "category is required"}), 400
         
    job_id = str(uuid.uuid4())
    training_jobs[job_id] = {
        "status": "pending",
        "category": category,
        "progress": "Starting job..."
    }
    
    # Start training in a background thread
    thread = threading.Thread(target=_run_training_job, args=(job_id, category))
    thread.daemon = True
    thread.start()
    
    return jsonify({"status": "success", "job_id": job_id})

@train_blueprint.route("/status/<job_id>", methods=["GET"])
def get_job_status(job_id):
    if job_id not in training_jobs:
        return jsonify({"status": "error", "message": "Job not found"}), 404
        
    return jsonify({"status": "success", "job": training_jobs[job_id]})
