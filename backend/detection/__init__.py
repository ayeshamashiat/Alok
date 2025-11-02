"""
Detection module for YOLO-based object detection with voice output
File: backend/detection/__init__.py
"""

from .live_detect import ObjectDetector, VoiceAnnouncer, start_live_detection, CONFIG
from .model import YOLOModel, create_model
from .utils import (
    calculate_box_area,
    calculate_box_center,
    calculate_distance_from_center,
    draw_proximity_zone,
    format_detection_info,
    apply_nms,
    resize_frame,
    save_detection_snapshot
)

__all__ = [
    'ObjectDetector',
    'VoiceAnnouncer',
    'start_live_detection',
    'CONFIG',
    'YOLOModel',
    'create_model',
    'calculate_box_area',
    'calculate_box_center',
    'calculate_distance_from_center',
    'draw_proximity_zone',
    'format_detection_info',
    'apply_nms',
    'resize_frame',
    'save_detection_snapshot'
]