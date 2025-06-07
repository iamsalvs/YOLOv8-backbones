import torch
import warnings
import time
from pathlib import Path
import pandas as pd
from ultralytics.utils import LOGGER
from distill_model import YOLOv8Distillation

# ==================== GLOBAL SETTINGS ====================

# Student model configs to try (YOLOv8 YAMLs)
MODEL_CONFIGS = [
    r"D:\YOLOV8-tomatod\parser\community\cfg\detect\yolov8n.yaml",
    # "yolov8s.yaml",
    # "yolov8m.yaml",
]

# Dataset configs + matching teacher checkpoint
DATASET_CONFIGS = [
    {"data": r"D:\YOLOV8-tomatod\datasets\tomatOD_yolo\data.yaml",  "name": "tomatod",  "teacher": r"D:\YOLOV8-tomatod\[FINAL] Baseline Results\YOLOv8l_Results\runs6\detect6\tomatOD_run\weights\best.pt"},
    # {"data": "camocrops.yaml","name": "camocrops","teacher": "pt_files/camocrops_best.pt"},
    # {"data": "ccrop.yaml",    "name": "ccrop",    "teacher": "pt_files/ccrop_best.pt"},
]

# Default training args
TRAINING_PARAMS = {
    "epochs":     5,
    "imgsz":      320,
    "batch":      4,
    "lr0":        1e-3,
    "device":     0 if torch.cuda.is_available() else "cpu",
    "project":    "YOLOv8_KD",
    "save_period":5,
    "val":        True,
    "plots":      True,
    "save":       True,
    "exist_ok":   True,
    "workers": 0, 
}

# Hyperparam sweep
EPOCHS_SWEEP      = 5
DISTILL_WEIGHTS   = [0.1, 0.3, 0.5, 0.7, 0.9]
TEMPERATURES      = [2.0, 4.0, 6.0, 8.0]


# ==================== HELPERS ====================

def get_model_name(yaml_path):
    return Path(yaml_path).stem

def get_epoch_weights(weights_dir, save_period, total_epochs):
    wd = Path(weights_dir)
    files = []
    for e in range(save_period, total_epochs+1, save_period):
        p = wd / f"epoch{e}.pt"
        if p.exists(): files.append((e, str(p)))
    for tag in ["best","last"]:
        p = wd / f"{tag}.pt"
        if p.exists(): files.append((tag,str(p)))
    return files

def test_model_weights(model_cfg, weight_path, epoch_id, dw, temp, ds_cfg):
    try:
        ds_name = ds_cfg["name"]
        mdl = YOLOv8Distillation(weight_path)
        results = mdl.val(data=ds_cfg["data"])
        mAP   = results.box.map
        mAP50 = results.box.map50
        mAP75 = results.box.map.map75 if hasattr(results.box,"map75") else 0.0

        out = {
            "dataset":       ds_name,
            "backbone":      get_model_name(model_cfg),
            "epoch":         epoch_id,
            "distill_weight":dw,
            "temperature":   temp,
            "teacher":       ds_cfg["teacher"],
            "mAP":           mAP,
            "mAP50":         mAP50,
            "mAP75":         mAP75,
            "weights":       weight_path,
            "status":        "success",
        }
        del mdl; torch.cuda.empty_cache()
        return out

    except Exception as e:
        return {
            "dataset":       ds_cfg["name"],
            "backbone":      get_model_name(model_cfg),
            "epoch":         epoch_id,
            "distill_weight":dw,
            "temperature":   temp,
            "teacher":       ds_cfg["teacher"],
            "mAP":           0.0,
            "mAP50":         0.0,
            "mAP75":         0.0,
            "weights":       weight_path,
            "status":        f"failed: {e}",
        }

def save_results(row, ds_name, mdl_name, phase="sweep"):
    base = Path.cwd() / TRAINING_PARAMS["project"] / ds_name / mdl_name / phase
    base.mkdir(parents=True, exist_ok=True)
    csv = base / f"{phase}_results.csv"
    df = pd.DataFrame([row])
    df.to_csv(csv, mode="a", header=not csv.exists(), index=False)
    return csv

def check_teachers():
    missing=[]
    for d in DATASET_CONFIGS:
        if not Path(d["teacher"]).exists(): missing.append(d["teacher"])
    if missing:
        LOGGER.error(f"Missing teacher weights: {missing}")
        return False
    return True


# ==================== PHASE 1: HYPERPARAMETER SWEEP ====================

def find_best_params(model_cfg, ds_cfg):
    LOGGER.info(f"🔍 Sweep for {model_cfg} on {ds_cfg['name']}")
    best = None
    for dw in DISTILL_WEIGHTS:
        for temp in TEMPERATURES:
            LOGGER.info(f" Testing dw={dw}, temp={temp}")
            mdl = YOLOv8Distillation(model_cfg)
            args = {
                **TRAINING_PARAMS,
                "data":           ds_cfg["data"],
                "name":           f"{ds_cfg['name']}/{get_model_name(model_cfg)}/sweep/dw{dw}_t{temp}",
                "epochs":        EPOCHS_SWEEP,
                "distill_weight":dw,
                "temperature":   temp,
                "teacher_weights":ds_cfg["teacher"],
            }
            mdl.train(**args)
            wdir   = Path(args["project"])/args["name"]/"weights"/"best.pt"
            if wdir.exists():
                res = test_model_weights(model_cfg, str(wdir), "best", dw, temp, ds_cfg)
                save_results(res, ds_cfg["name"], get_model_name(model_cfg), "sweep")
                if res["status"]=="success" and (best is None or res["mAP"]>best["mAP"]):
                    best = res
            del mdl; torch.cuda.empty_cache()
    return best

# ==================== PHASE 2: FULL TRAINING ====================

def full_train(model_cfg, ds_cfg, best):
    LOGGER.info(f"🚀 Full train {model_cfg} on {ds_cfg['name']} w/ dw={best['distill_weight']}, temp={best['temperature']}")
    mdl = YOLOv8Distillation(model_cfg)
    args = {
        **TRAINING_PARAMS,
        "data":            ds_cfg["data"],
        "name":            f"{ds_cfg['name']}/{get_model_name(model_cfg)}/final",
        "distill_weight":  best["distill_weight"],
        "temperature":     best["temperature"],
        "teacher_weights": ds_cfg["teacher"],
    }
    start=time.time()
    mdl.train(**args)
    duration = (time.time()-start)/60

    wdir = Path(args["project"])/args["name"]/"weights"
    epochs = get_epoch_weights(wdir, TRAINING_PARAMS["save_period"], TRAINING_PARAMS["epochs"])
    all_results=[]
    for eid, w in epochs:
        r = test_model_weights(model_cfg, w, eid, best["distill_weight"], best["temperature"], ds_cfg)
        r["train_time_min"]=duration
        save_results(r, ds_cfg["name"], get_model_name(model_cfg), "final")
        all_results.append(r)
    del mdl; torch.cuda.empty_cache()
    return all_results

# ==================== ORCHESTRATOR ====================

def run_pipeline():
    if not check_teachers(): raise FileNotFoundError("Missing teacher files")
    summary=[]
    total = len(MODEL_CONFIGS)*len(DATASET_CONFIGS)
    n=0

    LOGGER.info(f"▶️ Starting pipeline: {total} combos")
    for m in MODEL_CONFIGS:
        for d in DATASET_CONFIGS:
            n+=1
            LOGGER.info(f"{n}/{total} → Model={m}, Dataset={d['name']}")
            best = find_best_params(m,d)
            if best is None:
                LOGGER.error(" Sweep failed, skipping full train.")
                continue
            final = full_train(m,d,best)
            summary += final

    # save summary CSV
    df = pd.DataFrame(summary)
    out = Path.cwd()/TRAINING_PARAMS["project"]/"summary"
    out.mkdir(exist_ok=True)
    df.to_csv(out/"all_results.csv", index=False)
    LOGGER.info("✅ Pipeline complete. Results in %s", out)

if __name__ == "__main__":
    run_pipeline()
