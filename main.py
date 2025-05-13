import os
import xml.etree.ElementTree as ET
from PIL import Image
import random
import shutil
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from ultralytics import YOLO

def run_yolo_training():

    # SETTINGS: update the DATA path to your new dataset YAML file
    MODEL = 'yolov8'  # choose your model variant
    NAME = 'tomatOD_run'
    DATA = r"D:\YOLOV8-tomatod\CCROP_Dataset\dataset.yaml" # updated dataset path
    EPOCHS = 100
    BATCH = 32
    IMGSZ = 320
    LR0 = 0.01
    LRF = 0.0001
    MOMENTUM = 0.9
    OPTIMIZER = 'SGD'
    SAVE_PERIOD = 50
    VAL = True

    # Update here to use YAML config
    CFG = r"D:\YOLOV8-tomatod\parser\community\cfg\detect\yolov8l.yaml"
    model = YOLO(CFG) 

    # Train the model
    train_results = model.train(
        data=DATA,
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        workers=0,
        save_period=SAVE_PERIOD,
        device=0,
        name=NAME,
        optimizer=OPTIMIZER,
        lr0=LR0,
        lrf=LRF,
        momentum=MOMENTUM,
        val=VAL,
        project="runs/detect" 
    )
    
    # Validate checkpoints if available
    weights_path = os.path.join('runs', 'detect', NAME, 'weights')
    start_epoch = 1
    end_epoch = EPOCHS
    interval = 1
    imgsz_val = IMGSZ
    batch_val = 1
    save_json = True
    conf = 0.01
    iou = 0.5
    max_det = 50

    def write_results(name, metrics, split="val"):
        output_dir = os.path.join('runs', 'detect', name)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'{split}_ap50.txt'), 'a') as file:
            for ap50 in metrics.box.ap50:
                file.write(str(ap50) + '\n')
            file.write('\n')
            file.write(str(metrics.box.map50))
        with open(os.path.join(output_dir, f'{split}_maps.txt'), 'a') as file:
            for m in metrics.box.maps:
                file.write(str(m) + '\n')
            file.write('\n')
            file.write(str(metrics.box.map))
    
    # Loop over epochs and validate only if the checkpoint exists
    for epoch in range(start_epoch, end_epoch, interval):
        weight_file = os.path.join(weights_path, f'epoch{epoch}.pt')
        if not os.path.exists(weight_file):
            print(f"Weight file {weight_file} does not exist, skipping validation for epoch {epoch}.")
            continue
        model = YOLO(weight_file)
        run_label = f"{NAME}_epoch_{epoch}"
        
        # Validate on validation set
        val_metrics = model.val(
            imgsz=imgsz_val,
            batch=batch_val,
            workers=0,
            save_json=save_json,
            conf=conf,
            iou=iou,
            max_det=max_det,
            name=f"{run_label}_val",
            split="val",
            project="runs/detect" 
        )
        write_results(run_label, val_metrics, split="val")

        # Validate on test set
        test_metrics = model.val(
            imgsz=imgsz_val,
            batch=batch_val,
            workers=0,
            save_json=save_json,
            conf=conf,
            iou=iou,
            max_det=max_det,
            name=f"{run_label}_test",
            split="test",
            project="runs/detect" 
        )
        write_results(run_label, test_metrics, split="test")

    # Validate the final model (last.pt) if available
    weight_file = os.path.join(weights_path, 'last.pt')
    if os.path.exists(weight_file):
        model = YOLO(weight_file)
        run_label = f"{NAME}_epoch_{EPOCHS}"
        
        # Final validation
        val_metrics = model.val(
            imgsz=imgsz_val,
            batch=batch_val,
            workers=0,
            save_json=save_json,
            conf=conf,
            iou=iou,
            max_det=max_det,
            name=f"{run_label}_val",
            split="val",
            project="runs/detect" 
            
        )
        write_results(run_label, val_metrics, split="val")

        # Final test set evaluation
        test_metrics = model.val(
            imgsz=imgsz_val,
            batch=batch_val,
            workers=0,
            save_json=save_json,
            conf=conf,
            iou=iou,
            max_det=max_det,
            name=f"{run_label}_test",
            split="test",
            project="runs/detect" 
        )
        write_results(run_label, test_metrics, split="test")
    
    else:
        print("Final model weight file 'last.pt' does not exist.")

if __name__ == "__main__":
    run_yolo_training()
