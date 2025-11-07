from fastapi import FastAPI
from routes.detect_route import router as detect_router

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Backend is running successfully!"}

# Include YOLO detection routes
app.include_router(detect_router, prefix="/detect", tags=["Detection"])
