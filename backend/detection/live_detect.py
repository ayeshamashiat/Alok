"""
Live object detection with proximity filtering and voice announcements
File: backend/detection/live_detect.py
"""

import cv2
from ultralytics import YOLO
import pyttsx3
import time
import os

# Configuration
CONFIG = {
    'model_path': 'yolov8n.pt',
    'min_confidence': 0.5,
    'min_box_area': 5000,  # Minimum bounding box area for proximity
    'min_announcement_interval': 2,  # Seconds between announcements
    'camera_width': 640,
    'camera_height': 480,
    'camera_fps': 30,
    'detection_interval': 3  # Process every Nth frame
}

class ObjectDetector:
    """Handles YOLO object detection with proximity filtering"""
    
    def __init__(self, config):
        self.config = config
        self.model = YOLO(config['model_path'])
        self.last_detected_objects = set()
        self.last_announcement_time = 0
        
    def filter_by_proximity(self, boxes, confidences, class_ids):
        """Filter detections by bounding box size (proximity indicator)"""
        filtered_boxes = []
        filtered_confidences = []
        filtered_classes = []
        
        min_area = self.config['min_box_area']
        
        for box, conf, cls in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            
            if area >= min_area:  # Object is close enough
                filtered_boxes.append(box)
                filtered_confidences.append(conf)
                filtered_classes.append(cls)
        
        return filtered_boxes, filtered_confidences, filtered_classes
    
    def process_frame(self, frame):
        """Process a single frame and return detection results"""
        results = self.model(frame, conf=self.config['min_confidence'], verbose=False)
        
        # Extract detection data
        boxes = []
        confidences = []
        class_ids = []
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                boxes.append(box.xyxy[0].cpu().numpy())
                confidences.append(float(box.conf[0]))
                class_ids.append(int(box.cls[0]))
        
        # Filter by proximity
        filtered_boxes, filtered_confs, filtered_classes = self.filter_by_proximity(
            boxes, confidences, class_ids
        )
        
        # Get object names
        detected_objects = set()
        for cls_id in filtered_classes:
            detected_objects.add(self.model.names[cls_id])
        
        # Draw annotations
        annotated_frame = results[0].plot()
        
        # Draw green rectangles for close objects
        for box in filtered_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Display info
        cv2.putText(annotated_frame, f"Close objects: {len(filtered_boxes)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated_frame, detected_objects
    
    def should_announce(self, detected_objects, current_time):
        """Check if we should make an announcement"""
        objects_changed = detected_objects != self.last_detected_objects
        time_elapsed = current_time - self.last_announcement_time >= self.config['min_announcement_interval']
        
        return objects_changed and time_elapsed


class VoiceAnnouncer:
    """Handles text-to-speech announcements"""
    
    @staticmethod
    def speak(text):
        """Blocking speech function that reinitializes engine each time"""
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1.0)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine
        except Exception as e:
            print(f"Speech error: {e}")
    
    @staticmethod
    def generate_announcement(objects_list):
        """Generate natural language announcement from object list"""
        if len(objects_list) == 1:
            return f"I see a {objects_list[0]}"
        elif len(objects_list) == 2:
            return f"I see a {objects_list[0]} and a {objects_list[1]}"
        else:
            return f"I see {', '.join(objects_list[:-1])}, and a {objects_list[-1]}"


def start_live_detection():
    """Main function to start live detection"""
    detector = ObjectDetector(CONFIG)
    announcer = VoiceAnnouncer()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not found!")
        return

    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['camera_width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['camera_height'])
    cap.set(cv2.CAP_PROP_FPS, CONFIG['camera_fps'])

    print("Starting live detection... Press 'q' to quit.")
    print(f"Settings: Min confidence={CONFIG['min_confidence']}, Min box area={CONFIG['min_box_area']}")

    frame_count = 0
    annotated_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        frame_count += 1
        
        # Process every Nth frame for performance
        if frame_count % CONFIG['detection_interval'] == 0:
            annotated_frame, detected_objects = detector.process_frame(frame)
            
            current_time = time.time()
            
            # Check if we should announce
            if detector.should_announce(detected_objects, current_time):
                if detected_objects:
                    objects_list = list(detected_objects)
                    announcement = announcer.generate_announcement(objects_list)
                    
                    print(f"Detected nearby: {announcement}")
                    announcer.speak(announcement)
                    detector.last_announcement_time = current_time
                else:
                    print("No close objects detected")
                
                detector.last_detected_objects = detected_objects.copy()
        
        # Display frame
        if annotated_frame is not None:
            cv2.imshow("Live Detection (Green = Close Objects)", annotated_frame)
        else:
            cv2.imshow("Live Detection (Green = Close Objects)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_live_detection()