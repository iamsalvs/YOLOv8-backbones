import torch
import torch.nn.functional as F
import time
import warnings
import math
import numpy as np
from pathlib import Path
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import autocast

# allow unset_deterministic when missing
try:
    from ultralytics.utils.torch_utils import unset_deterministic
except ImportError:
    unset_deterministic = lambda: None

class YOLOv8DistillationTrainer(DetectionTrainer):
    def __init__(self, cfg=None, overrides=None, callbacks=None, _callbacks=None):
        # merge callbacks / _callbacks
        if callbacks is None and _callbacks is not None:
            callbacks = _callbacks
        # ensure config and overrides are defined
        from ultralytics.utils import DEFAULT_CFG
        if cfg is None:
            cfg = DEFAULT_CFG
        if overrides is None:
            overrides = {}
        # pop distillation args from overrides
        self.distill_weight  = overrides.pop('distill_weight', 0.5)
        self.temperature     = overrides.pop('temperature', 4.0)
        self.teacher_weights = overrides.pop('teacher_weights', None)
        super().__init__(cfg, overrides, callbacks)

        # prepare teacher
        self.teacher_model = None
        if self.teacher_weights:
            self._load_teacher(self.teacher_weights)

    def _load_teacher(self, weights_path):
        if not Path(weights_path).exists():
            LOGGER.warning(f"Teacher weights not found at {weights_path}, skipping KD.")
            return
        LOGGER.info(f"📚 Loading teacher YOLOv8 from {weights_path}")
        self.teacher_model = YOLO(weights_path)
        self.teacher_model.model.eval()
        for p in self.teacher_model.model.parameters():
            p.requires_grad = False
        self.teacher_model.model.to(self.device)

    def _compute_kl_loss(self, student_logits, teacher_logits):
        if student_logits.shape != teacher_logits.shape:
            LOGGER.warning("Shape mismatch for distillation logits, skipping KD.")
            return torch.tensor(0., device=self.device)
        s_log = F.log_softmax(student_logits / self.temperature, dim=-1)
        t_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        return F.kl_div(s_log, t_soft, reduction='batchmean') * (self.temperature ** 2)

    def train_one_epoch(self, epoch):
        self.run_callbacks("on_train_epoch_start")
        self.model.train()
        pbar = enumerate(self.train_loader)
        for i, batch in pbar:
            imgs, labels = batch['img'], batch['cls']
            with autocast(self.amp):
                preds, loss, loss_items = self._train_step(batch)
                if self.teacher_model:
                    with torch.no_grad():
                        t_preds = self.teacher_model.model(batch['img'])
                    s_logits = preds[0]
                    t_logits = t_preds[0]
                    kl_loss = self._compute_kl_loss(s_logits, t_logits)
                    loss = (1 - self.distill_weight) * loss + self.distill_weight * kl_loss
                    loss_items = torch.cat([loss_items, kl_loss.unsqueeze(0)])

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            if i % self.log_interval == 0:
                desc = f"Epoch {epoch+1}/{self.epochs} | " + \
                       " ".join(f"{k}:{v:.3f}" for k, v in zip(self.metrics_names, loss_items.tolist()))
                pbar.set_description(desc)

        self.run_callbacks("on_train_epoch_end")

    # Use base class train logic to handle setup, loops, and teardown
    def train(self):
        super().train()
        unset_deterministic()
        self.run_callbacks("on_train_end")
