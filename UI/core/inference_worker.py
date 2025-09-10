from __future__ import annotations
import csv, time
from pathlib import Path
from typing import Optional

import cv2
from PyQt6.QtCore import QThread, pyqtSignal

from .utils import FPSMeter, ensure_dir, sanitize
from .roi import ROI


class InferenceWorker(QThread):
    frame_ready = pyqtSignal(object)   # QImage (as object)
    stats = pyqtSignal(dict)
    log = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, src, adapter, output_dir: Path, save_frames: bool,
                 csv_path: Path, roi: Optional[ROI], metadata: dict | None):
        super().__init__()
        self.src = src
        self.adapter = adapter
        self.output_dir = Path(output_dir)
        self.save_frames = save_frames
        self.csv_path = Path(csv_path)
        self.roi = roi
        self.metadata = metadata or {}
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        cap = cv2.VideoCapture(self.src)
        if not cap.isOpened():
            self.log.emit(f"[ERROR] Cannot open source: {self.src}")
            self.finished.emit()
            return

        fpsm = FPSMeter()
        frame_idx = 0

        ensure_dir(self.csv_path.parent)
        f = open(self.csv_path, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(f, fieldnames=[
            "ts", "src", "frame", "roi", "pred_class", "pred_prob", "probs",
            "model", "saving", "fps"
        ])
        writer.writeheader()

        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    break

                draw = frame.copy()
                # Crop
                if self.roi:
                    x, y, w, h = self.roi.as_tuple()
                    x2, y2 = x + w, y + h
                    x, y = max(0, x), max(0, y)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    crop = frame[y:y2, x:x2]
                else:
                    crop = frame

                # Predict
                pred_class, pred_prob, probs = self.adapter.predict(
                    crop, self.metadata if self.adapter.use_hn else None
                )

                # Overlay
                txt = f"{pred_class}  {pred_prob*100:.1f}%"
                cv2.rectangle(draw, (10, 10), (10 + max(140, len(txt)*11), 44), (0, 0, 0), -1)
                cv2.putText(draw, txt, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                if self.roi:
                    x, y, w, h = self.roi.as_tuple()
                    cv2.rectangle(draw, (x, y), (x + w, y + h), (255, 180, 0), 2)

                # Emit frame
                qimg = self.cv_to_qimage(draw)
                self.frame_ready.emit(qimg)

                # Save
                saved_path = None
                if self.save_frames:
                    cls_dir = ensure_dir(self.output_dir / sanitize(pred_class))
                    fn = f"{frame_idx:08d}.jpg"
                    saved_path = str(cls_dir / fn)
                    cv2.imwrite(saved_path, crop if self.roi else frame)

                # Log CSV
                fps = fpsm.tick()
                writer.writerow({
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "src": str(self.src),
                    "frame": frame_idx,
                    "roi": self.roi.as_tuple() if self.roi else None,
                    "pred_class": pred_class,
                    "pred_prob": round(float(pred_prob), 5),
                    "probs": ";".join(f"{float(p):.5f}" for p in (probs.tolist() if hasattr(probs, "tolist") else probs)),
                    "model": type(self.adapter.model).__name__,
                    "saving": self.save_frames,
                    "fps": round(fps, 2)
                })
                f.flush()

                # Stats
                self.stats.emit({
                    "frame": frame_idx,
                    "saved": bool(saved_path),
                    "fps": round(fps, 2),
                    "pred": pred_class,
                    "prob": float(pred_prob)
                })

                frame_idx += 1
        finally:
            cap.release()
            f.close()
            self.finished.emit()

    @staticmethod
    def cv_to_qimage(cv_img):
        from PyQt6.QtGui import QImage
        h, w, ch = cv_img.shape
        bytes_per_line = ch * w
        return QImage(cv_img.data, w, h, bytes_per_line, QImage.Format.Format_BGR888).copy()
