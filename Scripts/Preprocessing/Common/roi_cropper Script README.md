# ✂️ ROI Cropping Utility for Tissue Video

## 🧠 Overview

This script automates the extraction of **Regions of Interest (ROI)** from frames in a **tissue video**. It is used to crop and save meaningful image regions that can be used later for classification or analysis (e.g., detecting cell states such as `Good Cut` or `Bad Cut`).

---

## 📽️ Application

Originally designed to:
- Process tissue video frame-by-frame
- Detect and crop ROI automatically or by fixed coordinates
- Save cropped images for training deep learning models

---

## ⚙️ Features

- Extracts frames from a tissue video.
- Crops a specified or dynamically detected ROI.
- Saves cropped regions as image files.
- Supports video-to-image processing automation.

---

## 🐍 Suggested Script Names

| Suggested Name               | Reason |
|-----------------------------|--------|
| `video_roi_cropper.py`      | Describes cropping ROIs from video. |
| `tissue_frame_cropper.py`   | Tailored to tissue videos. |
| `frame_roi_extractor.py`    | General video frame ROI tool. |
| `cell_roi_crop_video.py`    | If focused on cell region detection in tissue videos. |

---

## 📦 Requirements

```bash
pip install opencv-python numpy
```

---

## 🚀 Usage

```bash
python tissue_frame_cropper.py --video_path path/to/video.mp4 --output_dir path/to/crops
```

Optional arguments:
- `--roi_x`, `--roi_y`, `--roi_width`, `--roi_height` – fixed region coordinates.
- `--frame_interval` – number of frames to skip between extractions.
- `--visualize` – view detected ROI on frames.

---

## 📤 Output

```
cropped_output/
├── frame_001.jpg
├── frame_005.jpg
...
```

---

## 📌 Notes

- This script is ideal for preprocessing tissue videos for deep learning pipelines.
- Adjust ROI logic to match cross-target, color threshold, or custom detection logic.

---

## 👤 Author

Majid Tavakoli