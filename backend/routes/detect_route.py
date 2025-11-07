"""
Detection routes for FastAPI with AUTOMATIC dual-model detection
File: backend/routes/detect_route.py
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import cv2
import numpy as np
import io
from pathlib import Path

# Import detection components
from detection.live_detect import start_live_detection
from detection.model import YOLOModel, create_model

# Create router
router = APIRouter()

# Initialize BOTH models globally (loads once when server starts)
models = {}

# Load custom model
try:
    detection_dir = Path(__file__).parent.parent / 'detection'
    custom_model_path = detection_dir / 'best.pt'
    
    if custom_model_path.exists():
        print(f"✓ Loading CUSTOM model from: {custom_model_path}")
        models['custom'] = YOLOModel(model_path=str(custom_model_path))
        print(f"✓ Custom model loaded with {len(models['custom'].get_all_classes())} classes")
    else:
        print(f"✗ Custom model not found at {custom_model_path}")
except Exception as e:
    print(f"✗ Error loading custom model: {e}")

# Load pre-trained YOLOv11 model

try:
    yolo11_path = Path(__file__).parent.parent / 'detection' / 'yolo11n.pt'
    print(f"✓ Loading PRE-TRAINED YOLOv11n model from {yolo11_path}...")
    models['pretrained'] = YOLOModel(model_path=str(yolo11_path))
    print(f"✓ Pre-trained model loaded with {len(models['pretrained'].get_all_classes())} classes")
except Exception as e:
    print(f"✗ Error loading pre-trained model: {e}")

# Set default model
if 'custom' in models:
    default_model = models['custom']
    print("→ Default model: CUSTOM (best.pt)")
elif 'pretrained' in models:
    default_model = models['pretrained']
    print("→ Default model: PRE-TRAINED (yolov11n)")
else:
    raise RuntimeError("No models could be loaded!")

# Response models
class DetectionResult(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    source_model: str  # Which model detected this

class DetectionResponse(BaseModel):
    success: bool
    detections: List[DetectionResult]
    total_objects: int
    models_used: List[str]  # Which models were used
    custom_objects: int  # Count from custom model
    pretrained_objects: int  # Count from pretrained model

# Helper function to merge detections from both models
def merge_detections(custom_results, pretrained_results, iou_threshold=0.5):
    """
    Merge detections from both models, removing duplicates
    
    Args:
        custom_results: Results from custom model
        pretrained_results: Results from pretrained model
        iou_threshold: IOU threshold for considering boxes as duplicates
        
    Returns:
        List of merged detection results
    """
    def calculate_iou(box1, box2):
        """Calculate Intersection over Union of two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    merged = []
    
    # Add all custom detections first (they have priority)
    for i in range(len(custom_results['class_names'])):
        merged.append({
            'class_name': custom_results['class_names'][i],
            'confidence': custom_results['confidences'][i],
            'bbox': custom_results['boxes'][i].tolist(),
            'source_model': 'custom'
        })
    
    # Add pretrained detections, but skip if overlapping with custom
    for i in range(len(pretrained_results['class_names'])):
        pretrained_box = pretrained_results['boxes'][i]
        
        # Check if this box overlaps significantly with any custom detection
        is_duplicate = False
        for custom_detection in merged:
            if custom_detection['source_model'] == 'custom':
                custom_box = custom_detection['bbox']
                iou = calculate_iou(custom_box, pretrained_box)
                if iou > iou_threshold:
                    is_duplicate = True
                    break
        
        # If not a duplicate, add it
        if not is_duplicate:
            merged.append({
                'class_name': pretrained_results['class_names'][i],
                'confidence': pretrained_results['confidences'][i],
                'bbox': pretrained_box.tolist(),
                'source_model': 'pretrained'
            })
    
    return merged

# Helper function to get model
def get_model(model_type: str = 'default'):
    """Get the specified model or default"""
    if model_type == 'default':
        return default_model, 'custom' if 'custom' in models else 'pretrained'
    elif model_type == 'custom' and 'custom' in models:
        return models['custom'], 'custom'
    elif model_type == 'pretrained' and 'pretrained' in models:
        return models['pretrained'], 'pretrained'
    elif model_type == 'both':
        return None, 'both'  # Special case for dual detection
    else:
        return default_model, 'custom' if 'custom' in models else 'pretrained'

# Existing endpoints
@router.get("/live")
def live_detect():
    """Start live detection with voice announcements"""
    start_live_detection()
    return {"status": "Live detection ended."}

@router.get("/")
def detection_home():
    """Detection module home"""
    available_models = list(models.keys())
    return {
        "message": "Detection API is running",
        "available_models": available_models + ['both'],
        "default_model": 'custom' if 'custom' in models else 'pretrained',
        "endpoints": {
            "/live": "Start live camera detection with voice",
            "/image": "Upload image for detection (supports model selection)",
            "/image/dual": "AUTOMATIC: Detect with BOTH models simultaneously",
            "/image/annotated": "Get annotated image (supports model selection)",
            "/classes": "Get available detection classes for a model",
            "/models": "List all available models",
            "/health": "Check detection service health"
        }
    }

@router.get("/models")
def list_models():
    """List all available models and their info"""
    model_info = {}
    for name, model in models.items():
        model_info[name] = {
            "name": name,
            "classes_count": len(model.get_all_classes()),
            "device": model.device,
            "model_path": model.model_path,
            "classes": model.get_all_classes()[:10]  # Show first 10 classes
        }
    return {
        "success": True,
        "total_models": len(models),
        "default_model": 'custom' if 'custom' in models else 'pretrained',
        "models": model_info
    }

# NEW ENDPOINT: AUTOMATIC DUAL DETECTION
@router.post("/image/dual", response_model=DetectionResponse)
async def detect_image_dual(
    file: UploadFile = File(...),
    confidence: Optional[float] = 0.5,
    merge_duplicates: bool = Query(True, description="Remove overlapping detections")
):
    """
    🎯 AUTOMATIC: Detect with BOTH models simultaneously!
    
    This endpoint automatically runs detection with both your custom model
    AND the pre-trained model, then combines the results.
    
    Perfect for detecting:
    - Your custom objects (e.g., currencies) from custom model
    - Common objects (e.g., bottle, person) from pre-trained model
    
    All in ONE request!
    
    Args:
        file: Image file (jpg, png, etc.)
        confidence: Confidence threshold (0.0 to 1.0)
        merge_duplicates: Remove overlapping boxes (recommended: True)
    
    Returns:
        Combined detection results from both models
    """
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run detection with BOTH models
        models_used = []
        custom_results = None
        pretrained_results = None
        
        # Detect with custom model
        if 'custom' in models:
            custom_results = models['custom'].predict_and_parse(image, conf_threshold=confidence)
            models_used.append('custom')
        
        # Detect with pretrained model
        if 'pretrained' in models:
            pretrained_results = models['pretrained'].predict_and_parse(image, conf_threshold=confidence)
            models_used.append('pretrained')
        
        # Merge results
        if custom_results and pretrained_results and merge_duplicates:
            merged_detections = merge_detections(custom_results, pretrained_results)
        else:
            # Just combine without merging
            merged_detections = []
            
            if custom_results:
                for i in range(len(custom_results['class_names'])):
                    merged_detections.append({
                        'class_name': custom_results['class_names'][i],
                        'confidence': custom_results['confidences'][i],
                        'bbox': custom_results['boxes'][i].tolist(),
                        'source_model': 'custom'
                    })
            
            if pretrained_results:
                for i in range(len(pretrained_results['class_names'])):
                    merged_detections.append({
                        'class_name': pretrained_results['class_names'][i],
                        'confidence': pretrained_results['confidences'][i],
                        'bbox': pretrained_results['boxes'][i].tolist(),
                        'source_model': 'pretrained'
                    })
        
        # Format response
        detections = []
        custom_count = 0
        pretrained_count = 0
        
        for det in merged_detections:
            detection = DetectionResult(
                class_name=det['class_name'],
                confidence=det['confidence'],
                bbox=det['bbox'],
                source_model=det['source_model']
            )
            detections.append(detection)
            
            if det['source_model'] == 'custom':
                custom_count += 1
            else:
                pretrained_count += 1
        
        return DetectionResponse(
            success=True,
            detections=detections,
            total_objects=len(detections),
            models_used=models_used,
            custom_objects=custom_count,
            pretrained_objects=pretrained_count
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Original image endpoint (with model selection)
@router.post("/image", response_model=DetectionResponse)
async def detect_image(
    file: UploadFile = File(...),
    confidence: Optional[float] = 0.5,
    model_type: str = Query('both', description="Model to use: 'custom', 'pretrained', or 'both'")
):
    """
    Detect objects in uploaded image
    
    Args:
        file: Image file (jpg, png, etc.)
        confidence: Confidence threshold (0.0 to 1.0)
        model_type: Which model(s) to use ('custom', 'pretrained', or 'both')
    
    Returns:
        Detection results with bounding boxes and classes
    """
    # If 'both' is selected, use the dual detection endpoint
    if model_type == 'both':
        return await detect_image_dual(file, confidence)
    
    try:
        # Get the specified model
        model, model_used = get_model(model_type)
        
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run detection
        results = model.predict_and_parse(image, conf_threshold=confidence)
        
        # Format response
        detections = []
        for i in range(len(results['class_names'])):
            detection = DetectionResult(
                class_name=results['class_names'][i],
                confidence=results['confidences'][i],
                bbox=results['boxes'][i].tolist(),
                source_model=model_used
            )
            detections.append(detection)
        
        return DetectionResponse(
            success=True,
            detections=detections,
            total_objects=len(detections),
            models_used=[model_used],
            custom_objects=len(detections) if model_used == 'custom' else 0,
            pretrained_objects=len(detections) if model_used == 'pretrained' else 0
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/image/annotated")
async def detect_and_annotate(
    file: UploadFile = File(...),
    confidence: Optional[float] = 0.5,
    model_type: str = Query('both', description="Model to use: 'custom', 'pretrained', or 'both'")
):
    """
    Detect objects and return annotated image
    
    Args:
        file: Image file
        confidence: Confidence threshold
        model_type: Which model(s) to use ('custom', 'pretrained', or 'both')
    
    Returns:
        Annotated image with bounding boxes (color-coded by model)
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        annotated = image.copy()
        
        # If both models, run detection with both
        if model_type == 'both' and 'custom' in models and 'pretrained' in models:
            # Custom model detections (BLUE boxes)
            custom_results = models['custom'].predict(image, conf_threshold=confidence)
            custom_parsed = models['custom'].predict_and_parse(image, conf_threshold=confidence)
            
            for i, box in enumerate(custom_parsed['boxes']):
                x1, y1, x2, y2 = map(int, box)
                label = f"{custom_parsed['class_names'][i]} ({custom_parsed['confidences'][i]:.2f})"
                
                # Draw BLUE box for custom model
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(annotated, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            # Pretrained model detections (GREEN boxes)
            pretrained_results = models['pretrained'].predict(image, conf_threshold=confidence)
            pretrained_parsed = models['pretrained'].predict_and_parse(image, conf_threshold=confidence)
            
            for i, box in enumerate(pretrained_parsed['boxes']):
                x1, y1, x2, y2 = map(int, box)
                label = f"{pretrained_parsed['class_names'][i]} ({pretrained_parsed['confidences'][i]:.2f})"
                
                # Draw GREEN box for pretrained model
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Add legend
            cv2.putText(annotated, "BLUE = Custom Model", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(annotated, "GREEN = Pretrained Model", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        else:
            # Single model
            model, model_used = get_model(model_type)
            results = model.predict(image, conf_threshold=confidence)
            annotated = model.annotate_image(image, results)
            
            # Add text showing which model was used
            cv2.putText(annotated, f"Model: {model_used}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Convert to bytes
        _, buffer = cv2.imencode('.jpg', annotated)
        io_buf = io.BytesIO(buffer)
        
        return StreamingResponse(io_buf, media_type="image/jpeg")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/classes")
def get_classes(model_type: str = Query('both', description="Model to use: 'custom', 'pretrained', or 'both'")):
    """Get all available detection classes for specified model(s)"""
    try:
        if model_type == 'both':
            result = {}
            if 'custom' in models:
                result['custom'] = models['custom'].get_all_classes()
            if 'pretrained' in models:
                result['pretrained'] = models['pretrained'].get_all_classes()
            
            return {
                "success": True,
                "models": result,
                "total_custom_classes": len(result.get('custom', [])),
                "total_pretrained_classes": len(result.get('pretrained', []))
            }
        else:
            model, model_used = get_model(model_type)
            classes = model.get_all_classes()
            return {
                "success": True,
                "model_used": model_used,
                "total_classes": len(classes),
                "classes": classes
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
def health_check():
    """Check if detection service is healthy"""
    try:
        health_status = {
            "status": "healthy",
            "models_loaded": len(models),
            "available_models": list(models.keys()) + ['both'],
            "default_model": 'custom' if 'custom' in models else 'pretrained',
            "dual_detection_available": len(models) >= 2
        }
        
        # Add details for each model
        for name, model in models.items():
            health_status[f"{name}_model"] = {
                "loaded": True,
                "classes": len(model.get_all_classes()),
                "device": model.device,
                "info": str(model)
            }
        
        return health_status
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }