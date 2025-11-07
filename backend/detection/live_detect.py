"""
Live object detection with proximity filtering and voice announcements
AUTOMATIC DUAL MODEL DETECTION - Detects with BOTH models simultaneously!
File: backend/detection/live_detect.py
"""

import cv2
from ultralytics import YOLO
from gtts import gTTS
import time
import os
import tempfile
from pathlib import Path
from playsound import playsound

# Try to import config, fallback to manual path resolution
try:
    from config import get_model_path
    MODEL_PATH = get_model_path('best.pt')
except ImportError:
    # Fallback: look for model in same directory as this script
    current_dir = Path(__file__).parent.absolute()
    model_file = current_dir / 'best.pt'
    MODEL_PATH = str(model_file) if model_file.exists() else 'best.pt'

# Bangla translations for common objects
BANGLA_TRANSLATIONS = {
    # Common objects from pretrained model
    'person': 'মানুষ',
    'bicycle': 'সাইকেল',
    'car': 'গাড়ি',
    'motorcycle': 'মোটরসাইকেল',
    'airplane': 'বিমান',
    'bus': 'বাস',
    'train': 'ট্রেন',
    'truck': 'ট্রাক',
    'boat': 'নৌকা',
    'traffic light': 'ট্রাফিক লাইট',
    'fire hydrant': 'ফায়ার হাইড্রান্ট',
    'stop sign': 'স্টপ সাইন',
    'parking meter': 'পার্কিং মিটার',
    'bench': 'বেঞ্চ',
    'bird': 'পাখি',
    'cat': 'বিড়াল',
    'dog': 'কুকুর',
    'horse': 'ঘোড়া',
    'sheep': 'ভেড়া',
    'cow': 'গরু',
    'elephant': 'হাতি',
    'bear': 'ভালুক',
    'zebra': 'জেব্রা',
    'giraffe': 'জিরাফ',
    'backpack': 'ব্যাকপ্যাক',
    'umbrella': 'ছাতা',
    'handbag': 'হ্যান্ডব্যাগ',
    'tie': 'টাই',
    'suitcase': 'স্যুটকেস',
    'frisbee': 'ফ্রিসবি',
    'skis': 'স্কি',
    'snowboard': 'স্নোবোর্ড',
    'sports ball': 'খেলার বল',
    'kite': 'ঘুড়ি',
    'baseball bat': 'বেসবল ব্যাট',
    'baseball glove': 'বেসবল গ্লাভস',
    'skateboard': 'স্কেটবোর্ড',
    'surfboard': 'সার্ফবোর্ড',
    'tennis racket': 'টেনিস র‍্যাকেট',
    'bottle': 'বোতল',
    'wine glass': 'ওয়াইন গ্লাস',
    'cup': 'কাপ',
    'fork': 'কাঁটা চামচ',
    'knife': 'ছুরি',
    'spoon': 'চামচ',
    'bowl': 'বাটি',
    'banana': 'কলা',
    'apple': 'আপেল',
    'sandwich': 'স্যান্ডউইচ',
    'orange': 'কমলা',
    'broccoli': 'ব্রকলি',
    'carrot': 'গাজর',
    'hot dog': 'হট ডগ',
    'pizza': 'পিজা',
    'donut': 'ডোনাট',
    'cake': 'কেক',
    'chair': 'চেয়ার',
    'couch': 'সোফা',
    'potted plant': 'গাছের টব',
    'bed': 'বিছানা',
    'dining table': 'খাবার টেবিল',
    'toilet': 'টয়লেট',
    'tv': 'টিভি',
    'laptop': 'ল্যাপটপ',
    'mouse': 'মাউস',
    'remote': 'রিমোট',
    'keyboard': 'কিবোর্ড',
    'cell phone': 'মোবাইল ফোন',
    'microwave': 'মাইক্রোওয়েভ',
    'oven': 'ওভেন',
    'toaster': 'টোস্টার',
    'sink': 'সিঙ্ক',
    'refrigerator': 'ফ্রিজ',
    'book': 'বই',
    'clock': 'ঘড়ি',
    'vase': 'ফুলদানি',
    'scissors': 'কাঁচি',
    'teddy bear': 'টেডি বিয়ার',
    'hair drier': 'হেয়ার ড্রায়ার',
    'toothbrush': 'টুথব্রাশ',
    
    # Add your custom model classes here
    'USD_bill': 'মার্কিন ডলার',
    'EUR_bill': 'ইউরো',
    'BDT_bill': 'টাকা',
    'coin': 'মুদ্রা',
    # Add more custom classes as needed
}

# Configuration
CONFIG = {
    'custom_model_path': MODEL_PATH,
   'pretrained_model_path': str(Path(__file__).parent / 'yolo11n.pt'),
    'min_confidence': 0.5,
    'min_box_area': 5000,
    'min_announcement_interval': 2,
    'camera_width': 640,
    'camera_height': 480,
    'camera_fps': 30,
    'detection_interval': 3,
    'use_dual_models': True,
    'language': 'bn',  # Bangla
}

class DualModelDetector:
    """Handles YOLO object detection with BOTH custom and pretrained models"""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.last_detected_objects = set()
        self.last_announcement_time = 0
        
        # Load custom model
        try:
            print(f"Loading CUSTOM model from {config['custom_model_path']}...")
            self.models['custom'] = YOLO(config['custom_model_path'])
            custom_classes = len(self.models['custom'].names)
            print(f"✓ Custom model loaded ({custom_classes} classes)")
        except Exception as e:
            print(f"✗ Could not load custom model: {e}")
        
        # Load pretrained model
        try:
            print(f"Loading PRETRAINED model from {config['pretrained_model_path']}...")
            self.models['pretrained'] = YOLO(config['pretrained_model_path'])
            pretrained_classes = len(self.models['pretrained'].names)
            print(f"✓ Pretrained model loaded ({pretrained_classes} classes)")
        except Exception as e:
            print(f"✗ Could not load pretrained model: {e}")
        
        if not self.models:
            raise RuntimeError("No models could be loaded!")
        
        print(f"→ Loaded {len(self.models)} model(s) for detection")
        
    def filter_by_proximity(self, boxes, confidences, class_ids):
        """Filter detections by bounding box size (proximity indicator)"""
        filtered_boxes = []
        filtered_confidences = []
        filtered_classes = []
        
        min_area = self.config['min_box_area']
        
        for box, conf, cls in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            
            if area >= min_area:
                filtered_boxes.append(box)
                filtered_confidences.append(conf)
                filtered_classes.append(cls)
        
        return filtered_boxes, filtered_confidences, filtered_classes
    
    def process_frame_dual(self, frame):
        """Process frame with BOTH models and merge results"""
        all_detections = {
            'boxes': [],
            'confidences': [],
            'class_names': [],
            'source_models': []
        }
        
        annotated_frame = frame.copy()
        
        # Process with custom model
        if 'custom' in self.models:
            custom_results = self.models['custom'](
                frame, 
                conf=self.config['min_confidence'], 
                verbose=False
            )
            
            if custom_results[0].boxes is not None and len(custom_results[0].boxes) > 0:
                boxes = []
                confidences = []
                class_ids = []
                
                for box in custom_results[0].boxes:
                    boxes.append(box.xyxy[0].cpu().numpy())
                    confidences.append(float(box.conf[0]))
                    class_ids.append(int(box.cls[0]))
                
                filtered_boxes, filtered_confs, filtered_classes = self.filter_by_proximity(
                    boxes, confidences, class_ids
                )
                
                for box, conf, cls in zip(filtered_boxes, filtered_confs, filtered_classes):
                    all_detections['boxes'].append(box)
                    all_detections['confidences'].append(conf)
                    all_detections['class_names'].append(self.models['custom'].names[cls])
                    all_detections['source_models'].append('custom')
                    
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    label = f"{self.models['custom'].names[cls]} ({conf:.2f}) [CUSTOM]"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Process with pretrained model
        if 'pretrained' in self.models:
            pretrained_results = self.models['pretrained'](
                frame, 
                conf=self.config['min_confidence'], 
                verbose=False
            )
            
            if pretrained_results[0].boxes is not None and len(pretrained_results[0].boxes) > 0:
                boxes = []
                confidences = []
                class_ids = []
                
                for box in pretrained_results[0].boxes:
                    boxes.append(box.xyxy[0].cpu().numpy())
                    confidences.append(float(box.conf[0]))
                    class_ids.append(int(box.cls[0]))
                
                filtered_boxes, filtered_confs, filtered_classes = self.filter_by_proximity(
                    boxes, confidences, class_ids
                )
                
                for box, conf, cls in zip(filtered_boxes, filtered_confs, filtered_classes):
                    is_duplicate = False
                    if 'custom' in self.models:
                        for i, existing_box in enumerate(all_detections['boxes']):
                            if all_detections['source_models'][i] == 'custom':
                                iou = self.calculate_iou(box, existing_box)
                                if iou > 0.5:
                                    is_duplicate = True
                                    break
                    
                    if not is_duplicate:
                        all_detections['boxes'].append(box)
                        all_detections['confidences'].append(conf)
                        all_detections['class_names'].append(self.models['pretrained'].names[cls])
                        all_detections['source_models'].append('pretrained')
                        
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        label = f"{self.models['pretrained'].names[cls]} ({conf:.2f}) [PRETRAINED]"
                        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        detected_objects = set(all_detections['class_names'])
        
        custom_count = sum(1 for s in all_detections['source_models'] if s == 'custom')
        pretrained_count = sum(1 for s in all_detections['source_models'] if s == 'pretrained')
        
        cv2.putText(annotated_frame, f"Custom: {custom_count} | Pretrained: {pretrained_count}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Total close objects: {len(all_detections['boxes'])}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(annotated_frame, "BLUE=Custom | GREEN=Pretrained", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return annotated_frame, detected_objects, all_detections
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union of two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def should_announce(self, detected_objects, current_time):
        """Check if we should make an announcement"""
        objects_changed = detected_objects != self.last_detected_objects
        time_elapsed = current_time - self.last_announcement_time >= self.config['min_announcement_interval']
        
        return objects_changed and time_elapsed


class VoiceAnnouncer:
    """Handles text-to-speech announcements in Bangla using Google TTS"""
    
    @staticmethod
    def speak(text):
        """Google TTS for Bangla"""
        try:
            # Create Bangla audio
            tts = gTTS(text=text, lang='bn', slow=False)
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                temp_file = fp.name
                tts.save(temp_file)
            
            # Play audio
            playsound(temp_file)
            
            # Clean up
            os.remove(temp_file)
        except Exception as e:
            print(f"Speech error: {e}")
    
    @staticmethod
    def generate_announcement(objects_list):
        """Generate Bangla announcement from object list"""
        # Translate objects to Bangla
        bangla_objects = []
        for obj in objects_list:
            bangla_name = BANGLA_TRANSLATIONS.get(obj, obj)
            bangla_objects.append(bangla_name)
        
        if len(bangla_objects) == 0:
            return "কিছু পাওয়া যায়নি"
        elif len(bangla_objects) == 1:
            return f"আমি একটি {bangla_objects[0]} দেখছি"
        elif len(bangla_objects) == 2:
            return f"আমি একটি {bangla_objects[0]} এবং একটি {bangla_objects[1]} দেখছি"
        else:
            items = ', '.join([f"একটি {obj}" for obj in bangla_objects[:-1]])
            return f"আমি {items} এবং একটি {bangla_objects[-1]} দেখছি"


def start_live_detection():
    """Main function to start live detection with BOTH models"""
    
    print("\n" + "="*60)
    print("  স্বয়ংক্রিয় ডুয়াল মডেল লাইভ ডিটেকশন")
    print("="*60)
    print("  নীল বক্স   = কাস্টম মডেল (আপনার অবজেক্ট)")
    print("  সবুজ বক্স  = প্রি-ট্রেইন্ড মডেল (সাধারণ অবজেক্ট)")
    print("="*60 + "\n")
    
    detector = DualModelDetector(CONFIG)
    announcer = VoiceAnnouncer()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not found!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['camera_width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['camera_height'])
    cap.set(cv2.CAP_PROP_FPS, CONFIG['camera_fps'])

    print("লাইভ ডিটেকশন শুরু হচ্ছে... 'q' চাপুন বন্ধ করতে।")
    print(f"Settings: Min confidence={CONFIG['min_confidence']}, Min box area={CONFIG['min_box_area']}")
    print(f"Models active: {len(detector.models)}")
    print()

    frame_count = 0
    annotated_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        frame_count += 1
        
        if frame_count % CONFIG['detection_interval'] == 0:
            annotated_frame, detected_objects, all_detections = detector.process_frame_dual(frame)
            
            current_time = time.time()
            
            if detector.should_announce(detected_objects, current_time):
                if detected_objects:
                    objects_list = list(detected_objects)
                    announcement = announcer.generate_announcement(objects_list)
                    
                    custom_objects = [all_detections['class_names'][i] 
                                    for i, s in enumerate(all_detections['source_models']) 
                                    if s == 'custom']
                    pretrained_objects = [all_detections['class_names'][i] 
                                        for i, s in enumerate(all_detections['source_models']) 
                                        if s == 'pretrained']
                    
                    print(f"\n>>> সনাক্ত করা হয়েছে:")
                    if custom_objects:
                        print(f"    কাস্টম মডেল: {', '.join(set(custom_objects))}")
                    if pretrained_objects:
                        print(f"    প্রি-ট্রেইন্ড মডেল: {', '.join(set(pretrained_objects))}")
                    print(f"    ঘোষণা: {announcement}\n")
                    
                    announcer.speak(announcement)
                    detector.last_announcement_time = current_time
                else:
                    print("কোন কাছাকাছি অবজেক্ট সনাক্ত করা যায়নি")
                
                detector.last_detected_objects = detected_objects.copy()
        
        if annotated_frame is not None:
            cv2.imshow("Dual Model Live Detection (Press 'q' to quit)", annotated_frame)
        else:
            cv2.imshow("Dual Model Live Detection (Press 'q' to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nলাইভ ডিটেকশন শেষ হয়েছে।")


if __name__ == "__main__":
    start_live_detection()