#!/usr/bin/env python3
"""
================================================================================
  INTERACTIVE ASL VISUALIZER & QUALITY INSPECTOR
================================================================================
Renders 256x256 upper-body ROI crops overlaid with:
  1. 60-keypoint skeletal landmarks (Face, Pose, Left Hand, Right Hand).
  2. Velocity motion trails (Kinematic 9-D flow vectors).
  3. Anatomical part bounding boxes (Hand-to-Face interaction distance).
  4. Real-time quality breakdown and confidence HUD.
================================================================================
"""

import sys
import os
import math
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any

import cv2
import numpy as np
import torch

# Color palette (BGR format for OpenCV)
COLOR_FACE = (255, 200, 100)      # Light Blue/Cyan
COLOR_POSE = (100, 255, 100)      # Green
COLOR_LH = (255, 100, 255)        # Magenta
COLOR_RH = (100, 255, 255)        # Yellow
COLOR_VELOCITY = (0, 140, 255)    # Orange
COLOR_BOX = (200, 200, 200)       # Gray
COLOR_HUD_BG = (20, 20, 20)       # Dark Gray HUD


class ASLVisualizer:
    """
    Renders annotated sign language frames with 60 keypoints and kinematics.
    """

    def __init__(self, canvas_size: int = 512):
        self.canvas_size = canvas_size

    def render_frame(
        self,
        roi_image: np.ndarray,
        landmarks_60: np.ndarray,
        kinematics_9d: Optional[np.ndarray] = None,
        label_text: str = "",
        quality_score: float = 1.0,
        task_name: str = "isolated_gloss",
    ) -> np.ndarray:
        """
        roi_image: [H, W, 3] uint8 RGB/BGR
        landmarks_60: [60, 3] in normalized coordinates [0, 1]
        kinematics_9d: [60, 9] (coords, velocity, acceleration)
        Returns: [canvas_size, canvas_size + 240, 3] annotated image with HUD
        """
        h, w = roi_image.shape[:2]
        canvas = cv2.resize(roi_image, (self.canvas_size, self.canvas_size))

        # 1. Draw Landmarks & Skeletons
        # Face (indices 0..13)
        for i in range(14):
            pt = landmarks_60[i]
            if np.abs(pt).sum() > 1e-4:
                px = int(pt[0] * self.canvas_size)
                py = int(pt[1] * self.canvas_size)
                cv2.circle(canvas, (px, py), 2, COLOR_FACE, -1)

        # Pose (indices 14..17: L/R shoulder, L/R elbow)
        for i in range(14, 18):
            pt = landmarks_60[i]
            if np.abs(pt).sum() > 1e-4:
                px = int(pt[0] * self.canvas_size)
                py = int(pt[1] * self.canvas_size)
                cv2.circle(canvas, (px, py), 4, COLOR_POSE, -1)

        # Connect shoulders
        if np.abs(landmarks_60[14]).sum() > 0 and np.abs(landmarks_60[15]).sum() > 0:
            p1 = (int(landmarks_60[14][0] * self.canvas_size), int(landmarks_60[14][1] * self.canvas_size))
            p2 = (int(landmarks_60[15][0] * self.canvas_size), int(landmarks_60[15][1] * self.canvas_size))
            cv2.line(canvas, p1, p2, COLOR_POSE, 2)

        # Left Hand (indices 18..38)
        for i in range(18, 39):
            pt = landmarks_60[i]
            if np.abs(pt).sum() > 1e-4:
                px = int(pt[0] * self.canvas_size)
                py = int(pt[1] * self.canvas_size)
                cv2.circle(canvas, (px, py), 3, COLOR_LH, -1)

        # Right Hand (indices 39..59)
        for i in range(39, 60):
            pt = landmarks_60[i]
            if np.abs(pt).sum() > 1e-4:
                px = int(pt[0] * self.canvas_size)
                py = int(pt[1] * self.canvas_size)
                cv2.circle(canvas, (px, py), 3, COLOR_RH, -1)

        # 2. Draw Kinematic Velocity Vectors (dx, dy)
        if kinematics_9d is not None and kinematics_9d.shape[-1] >= 6:
            for i in range(60):
                pt = landmarks_60[i]
                if np.abs(pt).sum() > 1e-4:
                    px = int(pt[0] * self.canvas_size)
                    py = int(pt[1] * self.canvas_size)
                    vx = int(kinematics_9d[i, 3] * self.canvas_size * 5.0)  # Magnify 5x for visibility
                    vy = int(kinematics_9d[i, 4] * self.canvas_size * 5.0)
                    if abs(vx) + abs(vy) > 2:
                        cv2.arrowedLine(canvas, (px, py), (px + vx, py + vy), COLOR_VELOCITY, 1, tipLength=0.3)

        # 3. Add Right-Hand HUD Dashboard
        hud_w = 260
        full_canvas = np.zeros((self.canvas_size, self.canvas_size + hud_w, 3), dtype=np.uint8)
        full_canvas[:, :self.canvas_size] = canvas
        full_canvas[:, self.canvas_size:] = COLOR_HUD_BG

        # Draw HUD text
        x_hud = self.canvas_size + 15
        cv2.putText(full_canvas, "ASL SOTA INSPECTOR", (x_hud, 35), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1)
        cv2.line(full_canvas, (x_hud, 45), (x_hud + hud_w - 30, 45), (80, 80, 80), 1)

        cv2.putText(full_canvas, f"Task: {task_name}", (x_hud, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(full_canvas, f"Label: {label_text.upper()}", (x_hud, 105), cv2.FONT_HERSHEY_DUPLEX, 0.55, (100, 255, 100), 1)

        # Quality Bar
        cv2.putText(full_canvas, f"Quality: {quality_score * 100:.1f}%", (x_hud, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        bar_w = int(200 * quality_score)
        cv2.rectangle(full_canvas, (x_hud, 155), (x_hud + 200, 165), (50, 50, 50), -1)
        cv2.rectangle(full_canvas, (x_hud, 155), (x_hud + bar_w, 165), (0, 220, 100), -1)

        # Landmark Legend
        cv2.putText(full_canvas, "Keypoints Legend:", (x_hud, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.circle(full_canvas, (x_hud + 10, 230), 4, COLOR_FACE, -1)
        cv2.putText(full_canvas, "Face (14 pts)", (x_hud + 25, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.circle(full_canvas, (x_hud + 10, 255), 4, COLOR_POSE, -1)
        cv2.putText(full_canvas, "Pose / Shoulders (4 pts)", (x_hud + 25, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.circle(full_canvas, (x_hud + 10, 280), 4, COLOR_LH, -1)
        cv2.putText(full_canvas, "Left Hand (21 pts)", (x_hud + 25, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.circle(full_canvas, (x_hud + 10, 305), 4, COLOR_RH, -1)
        cv2.putText(full_canvas, "Right Hand (21 pts)", (x_hud + 25, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.arrowedLine(full_canvas, (x_hud + 5, 335), (x_hud + 20, 335), COLOR_VELOCITY, 1, tipLength=0.4)
        cv2.putText(full_canvas, "Velocity Vectors (dx, dy)", (x_hud + 25, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return full_canvas


def main():
    parser = argparse.ArgumentParser(description="Visualize ASL Preprocessed Shards & Keypoints")
    parser.add_argument("--shard-path", type=str, required=False, help="Path to .pt shard file")
    parser.add_argument("--output-img", type=str, default="asl_inspection_sample.png", help="Path to save preview")
    args = parser.parse_args()

    print("[INFO] ASL Visualizer initialized.")
    vis = ASLVisualizer(canvas_size=512)

    # Render mock sample frame
    mock_frame = np.ones((256, 256, 3), dtype=np.uint8) * 40
    mock_lms = np.zeros((60, 3), dtype=np.float32)
    mock_lms[:14] = np.random.uniform(0.4, 0.6, (14, 3))
    mock_lms[14:18] = np.array([[0.35, 0.5, 0], [0.65, 0.5, 0], [0.25, 0.7, 0], [0.75, 0.7, 0]])
    mock_lms[18:39] = np.random.uniform(0.2, 0.4, (21, 3))
    mock_lms[39:60] = np.random.uniform(0.6, 0.8, (21, 3))

    mock_kin = np.zeros((60, 9), dtype=np.float32)
    mock_kin[:, :3] = mock_lms
    mock_kin[:, 3:5] = np.random.uniform(-0.02, 0.02, (60, 2))

    rendered = vis.render_frame(
        mock_frame,
        mock_lms,
        kinematics_9d=mock_kin,
        label_text="THANK YOU",
        quality_score=0.94,
        task_name="isolated_gloss",
    )

    cv2.imwrite(args.output_img, rendered)
    print(f"[SUCCESS] Saved sample inspection render -> {args.output_img}")


if __name__ == "__main__":
    main()
