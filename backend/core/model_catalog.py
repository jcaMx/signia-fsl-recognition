import json
from pathlib import Path
from typing import Dict, Any, List
from src.data_collection.label_registry import LabelRegistry

class ModelCatalog:
    def __init__(self):
        self.project_dir = Path(__file__).resolve().parents[2]
        self.artifacts_models_dir = self.project_dir / "artifacts" / "models"
        
        try:
            self.registry = LabelRegistry()
            self.labels_map = self._build_labels_map()
        except Exception as e:
            print(f"Failed to load LabelRegistry: {e}")
            self.labels_map = {}

    def _build_labels_map(self) -> Dict[int, str]:
        labels = {}
        if hasattr(self.registry, "_df"):
            for _, row in self.registry._df.iterrows():
                try:
                    labels[int(row["id"])] = str(row["label"])
                except ValueError:
                    pass
        return labels

    def get_catalog(self) -> Dict[str, Any]:
        """
        Scans artifacts/models and returns:
        {
            "categories": {
                "color": ["RED", "BLUE", ...], ...
            },
            "models": {
                "color": "artifacts/models/color/color_lstm_best.pt"
            }
        }
        """
        categories = {}
        models = {}
        
        if not self.artifacts_models_dir.exists():
            return {"categories": categories, "models": models}
            
        for category_dir in self.artifacts_models_dir.iterdir():
            if not category_dir.is_dir():
                continue
                
            category_name = category_dir.name
            
            # Find the .pt file and .json file
            pt_files = list(category_dir.glob("*.pt"))
            json_files = list(category_dir.glob("*.json"))
            
            if not pt_files or not json_files:
                continue
                
            best_pt = None
            for pt in pt_files:
                if "best" in pt.name:
                    best_pt = pt
                    break
            if not best_pt:
                best_pt = pt_files[0]
                
            json_file = json_files[0]
            
            try:
                with json_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    classes = data.get("classes", [])
                    label_names = []
                    for c in classes:
                        c_int = int(c)
                        if c_int in self.labels_map:
                            label_names.append(self.labels_map[c_int])
                        else:
                            label_names.append(f"Class {c}")
                    
                    categories[category_name] = label_names
                    models[category_name] = str(best_pt.relative_to(self.project_dir).as_posix())
            except Exception as e:
                print(f"Failed to read model metadata for {category_name}: {e}")
                
        return {
            "categories": categories,
            "models": models
        }
