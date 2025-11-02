"""
Detection routes for FastAPI
File: backend/routes/detect_route.py
"""

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import cv2
import numpy as np
import io

# Import detection components
from detection.live_detect import start_live_detection
from detection.model import YOLOModel, create_model

# Create router
router = APIRouter()

# Initialize model globally (loads once when server starts)
model = create_model('nano')

# Response models
class DetectionResult(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]

class DetectionResponse(BaseModel):
    success: bool
    detections: List[DetectionResult]
    total_objects: int

# Your existing working endpoint
@router.get("/live")
def live_detect():
    """Start live detection with voice announcements"""
    start_live_detection()
    return {"status": "Live detection ended."}

# Additional useful endpoints below

@router.get("/")
def detection_home():
    """Detection module home"""
    return {
        "message": "Detection API is running",
        "endpoints": {
            "/live": "Start live camera detection with voice",
            "/image": "Upload image for detection",
            "/image/annotated": "Get annotated image",
            "/classes": "Get available detection classes",
            "/health": "Check detection service health"
        }
    }

@router.post("/image", response_model=DetectionResponse)
async def detect_image(
    file: UploadFile = File(...),
    confidence: Optional[float] = 0.5
):
    """
    Detect objects in uploaded image
    
    Args:
        file: Image file (jpg, png, etc.)
        confidence: Confidence threshold (0.0 to 1.0)
    
    Returns:
        Detection results with bounding boxes and classes
    """
    try:
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
                bbox=results['boxes'][i].tolist()
            )
            detections.append(detection)
        
        return DetectionResponse(
            success=True,
            detections=detections,
            total_objects=len(detections)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/image/annotated")
async def detect_and_annotate(
    file: UploadFile = File(...),
    confidence: Optional[float] = 0.5
):
    """
    Detect objects and return annotated image
    
    Args:
        file: Image file
        confidence: Confidence threshold
    
    Returns:
        Annotated image with bounding boxes
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run detection
        results = model.predict(image, conf_threshold=confidence)
        annotated = model.annotate_image(image, results)
        
        # Convert to bytes
        _, buffer = cv2.imencode('.jpg', annotated)
        io_buf = io.BytesIO(buffer)
        
        return StreamingResponse(io_buf, media_type="image/jpeg")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/classes")
def get_classes():
    """Get all available detection classes"""
    try:
        classes = model.get_all_classes()
        return {
            "success": True,
            "total_classes": len(classes),
            "classes": classes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
def health_check():
    """Check if detection service is healthy"""
    try:
        # Test model
        test_classes = model.get_all_classes()
        return {
            "status": "healthy",
            "model_loaded": True,
            "total_classes": len(test_classes),
            "model_info": str(model)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }