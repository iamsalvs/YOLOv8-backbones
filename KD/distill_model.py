# distill_model.py
from ultralytics import YOLO
from ultralytics.engine.predictor import BasePredictor as YOLOPredictor
from ultralytics.engine.validator import BaseValidator as YOLOValidator
from ultralytics.nn.tasks import DetectionModel
from distill_train import YOLOv8DistillationTrainer

class YOLOv8Distillation(YOLO):
    """
    YOLOv8 model with knowledge distillation support.

    This class extends the standard YOLOv8 model to use a teacher-student
    distillation trainer when you call `.train(...)`.
    """

    def __init__(self, model='yolov8s.pt'):
        super().__init__(model=model)

    @property
    def task_map(self):
        # override the detect task to use our distillation trainer
        return {
            'detect': {
                'predictor': YOLOPredictor,
                'validator': YOLOValidator,
                'trainer':   YOLOv8DistillationTrainer,
                'model':     DetectionModel,
            }
        }
