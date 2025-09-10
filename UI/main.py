from __future__ import annotations
import sys
from pathlib import Path

from PyQt6.QtCore import Qt, QRect, pyqtSignal
from PyQt6.QtGui import QPixmap, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QComboBox, QCheckBox, QPlainTextEdit,
    QTableWidget, QTableWidgetItem, QGroupBox, QMessageBox
)

from core.model_manager import ModelRegistry, build_adapter
from core.inference_worker import InferenceWorker
from core.roi import ROI


class VideoView(QLabel):
    roi_changed = pyqtSignal(object)   # ROI or None

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 360)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pix = None
        self._drawing = False
        self._start = None
        self._rect = None
        self._enable_roi = False

    def setImage(self, qimg):
        self._pix = QPixmap.fromImage(qimg)
        self.update()

    def paintEvent(self, e):
        super().paintEvent(e)
        if self._pix:
            p = QPainter(self)
            scaled = self._pix.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            p.drawPixmap(
                (self.width()-scaled.width())//2,
                (self.height()-scaled.height())//2,
                scaled
            )
            if self._rect:
                pen = QPen()
                pen.setWidth(2)
                pen.setColor(Qt.GlobalColor.yellow)
                p.setPen(pen)
                p.drawRect(self._rect)

    def enable_roi(self, en: bool):
        self._enable_roi = en
        if not en:
            self._rect = None
            self.update()

    def mousePressEvent(self, ev):
        if not self._enable_roi or not self._pix:
            return
        if ev.button() == Qt.MouseButton.LeftButton:
            self._drawing = True
            self._start = ev.position().toPoint()
            self._rect = QRect(self._start, self._start)
            self.update()

    def mouseMoveEvent(self, ev):
        if self._drawing and self._enable_roi:
            cur = ev.position().toPoint()
            self._rect = QRect(self._start, cur).normalized()
            self.update()

    def mouseReleaseEvent(self, ev):
        if not self._enable_roi:
            return
        if ev.button() == Qt.MouseButton.LeftButton and self._drawing:
            self._drawing = False
            self.update()

    def get_roi_in_frame_coords(self) -> ROI | None:
        if not self._pix or not self._rect:
            return None
        scaled = self._pix.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        x0 = (self.width()-scaled.width())//2
        y0 = (self.height()-scaled.height())//2
        rx = self._rect.translated(-x0, -y0)

        sx = self._pix.width()/scaled.width()
        sy = self._pix.height()/scaled.height()
        x = max(0, int(rx.x()*sx))
        y = max(0, int(rx.y()*sy))
        w = max(1, int(rx.width()*sx))
        h = max(1, int(rx.height()*sy))
        return ROI(x, y, w, h)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DeepCutQC-UI")
        self.resize(1200, 720)

        self.registry = ModelRegistry()
        self.adapter = None
        self.worker: InferenceWorker | None = None
        self.current_src = None
        self.current_roi: ROI | None = None
        self.out_dir = Path("outputs")

        # --- UI widgets ---
        self.video = VideoView()

        btn_open_file = QPushButton("Open Video…")
        btn_open_cam = QPushButton("Open Webcam")
        btn_start = QPushButton("Start")
        btn_stop = QPushButton("Stop")

        self.model_combo = QComboBox()
        names = self.registry.get_names()
        self.model_combo.addItems(names)

        self.save_chk = QCheckBox("Save Frames")
        self.btn_out = QPushButton("Choose Output Folder…")

        self.btn_set_roi = QPushButton("Set ROI")
        self.btn_confirm_roi = QPushButton("Confirm ROI")
        self.btn_clear_roi = QPushButton("Clear ROI")

        self.log = QPlainTextEdit(); self.log.setReadOnly(True)

        # Metadata panel (only for HN)
        self.meta_group = QGroupBox("Metadata (for HyperNetwork)")
        self.meta_table = QTableWidget(0, 2)
        self.meta_table.setHorizontalHeaderLabels(["Key", "Value"])
        v_meta = QVBoxLayout(); v_meta.addWidget(self.meta_table); self.meta_group.setLayout(v_meta)
        self.meta_group.setVisible(False)

        # Layout
        left = QVBoxLayout()
        left.addWidget(QLabel("Model:"))
        left.addWidget(self.model_combo)
        left.addWidget(self.meta_group)
        left.addWidget(self.save_chk)
        left.addWidget(self.btn_out)
        left.addSpacing(8)
        left.addWidget(self.btn_set_roi)
        left.addWidget(self.btn_confirm_roi)
        left.addWidget(self.btn_clear_roi)
        left.addStretch(1)
        left.addWidget(btn_open_file)
        left.addWidget(btn_open_cam)
        left.addWidget(btn_start)
        left.addWidget(btn_stop)

        right = QVBoxLayout()
        right.addWidget(self.video, stretch=5)
        right.addWidget(QLabel("Status / Log"))
        right.addWidget(self.log, stretch=2)

        root = QHBoxLayout()
        side = QWidget(); side.setLayout(left)
        main = QWidget(); main.setLayout(right)
        root.addWidget(side, 0)
        root.addWidget(main, 1)
        central = QWidget(); central.setLayout(root)
        self.setCentralWidget(central)

        # Signals
        btn_open_file.clicked.connect(self.on_open_file)
        btn_open_cam.clicked.connect(self.on_open_cam)
        btn_start.clicked.connect(self.on_start)
        btn_stop.clicked.connect(self.on_stop)
        self.btn_out.clicked.connect(self.on_choose_out)
        self.btn_set_roi.clicked.connect(lambda: self.video.enable_roi(True))
        self.btn_confirm_roi.clicked.connect(self.on_confirm_roi)
        self.btn_clear_roi.clicked.connect(self.on_clear_roi)
        self.model_combo.currentTextChanged.connect(self.on_model_changed)

        # Initialize metadata panel (if models found)
        if self.model_combo.count() == 0:
            QMessageBox.warning(self, "No Models", "No model YAMLs found in configs/models.")
        else:
            self.on_model_changed(self.model_combo.currentText())

    # ----- Slots -----
    def on_model_changed(self, name: str):
        try:
            cfg = self.registry.get_cfg(name)
        except Exception as e:
            self.meta_group.setVisible(False)
            self.log.appendPlainText(f"[ERROR] Cannot load config for model '{name}': {e}")
            return
        use_hn = bool(cfg.get("use_hypernetwork", False))
        self.meta_group.setVisible(use_hn)
        self.meta_table.setRowCount(0)
        if use_hn:
            schema = cfg.get("metadata", {}).get("schema", [])
            for row, field in enumerate(schema):
                self.meta_table.insertRow(row)
                self.meta_table.setItem(row, 0, QTableWidgetItem(field.get("key", "")))
                self.meta_table.setItem(row, 1, QTableWidgetItem(str(field.get("default", ""))))

    def collect_metadata(self) -> dict:
        md = {}
        for r in range(self.meta_table.rowCount()):
            k = self.meta_table.item(r, 0).text().strip() if self.meta_table.item(r, 0) else ""
            v = self.meta_table.item(r, 1).text().strip() if self.meta_table.item(r, 1) else ""
            md[k] = v
        return md

    def on_open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", "Video Files (*.mp4 *.avi *.mkv *.mov)"
        )
        if path:
            self.current_src = path
            self.log.appendPlainText(f"[INFO] Selected file: {path}")

    def on_open_cam(self):
        self.current_src = 0
        self.log.appendPlainText("[INFO] Using webcam (index 0)")

    def on_choose_out(self):
        d = QFileDialog.getExistingDirectory(self, "Choose Output Folder", str(self.out_dir))
        if d:
            self.out_dir = Path(d)
            self.log.appendPlainText(f"[INFO] Output folder: {self.out_dir}")

    def on_confirm_roi(self):
        roi = self.video.get_roi_in_frame_coords()
        if roi:
            self.current_roi = roi
            self.video.enable_roi(False)
            self.log.appendPlainText(f"[INFO] ROI set to: {roi.as_tuple()}")
        else:
            self.log.appendPlainText("[WARN] No ROI selected.")

    def on_clear_roi(self):
        self.current_roi = None
        self.video.enable_roi(False)
        self.log.appendPlainText("[INFO] ROI cleared.")

    def on_start(self):
        if self.worker is not None:
            self.log.appendPlainText("[WARN] Inference already running.")
            return
        if self.current_src is None:
            self.log.appendPlainText("[ERROR] Choose a video file or webcam first.")
            return

        try:
            cfg = self.registry.get_cfg(self.model_combo.currentText())
            self.adapter = build_adapter(cfg)
        except Exception as e:
            self.log.appendPlainText(f"[ERROR] Failed to load model: {e}")
            return

        csv_path = self.out_dir / "predictions_log.csv"
        metadata = self.collect_metadata() if bool(cfg.get("use_hypernetwork", False)) else None

        self.worker = InferenceWorker(
            src=self.current_src,
            adapter=self.adapter,
            output_dir=self.out_dir,
            save_frames=self.save_chk.isChecked(),
            csv_path=csv_path,
            roi=self.current_roi,
            metadata=metadata
        )
        self.worker.frame_ready.connect(self.on_frame)
        self.worker.stats.connect(self.on_stats)
        self.worker.log.connect(self.log.appendPlainText)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()
        self.log.appendPlainText("[INFO] Inference started.")

    def on_stop(self):
        if self.worker:
            self.worker.stop()

    def on_finished(self):
        self.log.appendPlainText("[INFO] Inference stopped.")
        self.worker = None

    def on_frame(self, qimg):
        self.video.setImage(qimg)

    def on_stats(self, d: dict):
        self.statusBar().showMessage(
            f"Frame: {d['frame']} | Saved: {d['saved']} | FPS: {d['fps']} | "
            f"Pred: {d['pred']} ({d['prob']*100:.1f}%)"
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    style_path = Path("assets/styles.qss")
    if style_path.exists():
        app.setStyleSheet(style_path.read_text(encoding="utf-8"))
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
