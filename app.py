"""
Traffic Detection API Service

This FastAPI service provides vehicle detection and lane assignment
for Maseru Bridge traffic analysis.

Direction is determined by GEOMETRY, not language inference.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import logging
import os
import json
import urllib.request

from detector import get_detector, LaneCount

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Traffic Detection Service",
    description="Deterministic vehicle detection and lane assignment for Maseru Bridge",
    version="1.0.0"
)


@app.on_event("startup")
async def load_remote_config():
    """
    On startup, fetch lane_config.json from LANE_CONFIG_URL (R2) if set.
    This allows polygon updates without Docker rebuilds — just upload to R2.
    Falls back to bundled lane_config.json if URL is not set or fetch fails.
    """
    config_url = os.environ.get("LANE_CONFIG_URL")
    if not config_url:
        logger.info("ℹ️  LANE_CONFIG_URL not set — using bundled lane_config.json")
        return

    try:
        logger.info(f"🌐 Fetching lane config from: {config_url}")
        with urllib.request.urlopen(config_url, timeout=10) as response:
            remote_config = json.loads(response.read().decode())

        with open("lane_config.json", "w") as f:
            json.dump(remote_config, f, indent=2)

        logger.info("✅ Remote lane_config.json loaded and saved successfully")
    except Exception as e:
        logger.warning(f"⚠️  Failed to fetch remote config: {e} — using bundled lane_config.json")

# Enable CORS for Node.js server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class AnalyzeRequest(BaseModel):
    image: str  # Base64 encoded image
    camera_view: str = "bridge"  # "bridge", "canopy", or "engen"


class AnalyzeResponse(BaseModel):
    success: bool
    SA_to_LS: int
    LS_to_SA: int
    unassigned: int
    total: int
    direction_uncertain: bool
    message: Optional[str] = None
    vehicles: Optional[List[Dict]] = None
    breakdown: Optional[Dict] = None


class MultiAnalyzeRequest(BaseModel):
    frames: List[Dict]  # List of {image: str, camera_view: str}


class MultiAnalyzeResponse(BaseModel):
    success: bool
    combined: Dict
    by_view: Dict
    message: Optional[str] = None


class BurstFrame(BaseModel):
    image: str
    timestamp_ms: int


class BurstAnalyzeRequest(BaseModel):
    """A burst of consecutive frames (oldest first) from one camera view."""
    frames: List[BurstFrame]
    camera_view: str = "bridge"


class BurstAnalyzeResponse(BaseModel):
    success: bool
    SA_to_LS: int
    LS_to_SA: int
    unassigned: int
    total: int
    direction_uncertain: bool
    message: Optional[str] = None
    vehicles: Optional[List[Dict]] = None
    breakdown: Optional[Dict] = None
    flow_metrics: Optional[Dict] = None


class DebugRequest(BaseModel):
    image: str
    camera_view: str = "bridge"


class DebugResponse(BaseModel):
    success: bool
    annotated_image: str  # Base64 encoded debug image
    counts: Dict


class CalibrationRequest(BaseModel):
    camera_view: str
    lane_name: str
    polygon: List[List[int]]


class CalibrationResponse(BaseModel):
    success: bool
    message: str


# Endpoints
@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "Traffic Detection Service",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    try:
        detector = get_detector()
        return {
            "status": "healthy",
            "model_loaded": detector.model is not None,
            "config_loaded": detector.config is not None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_traffic(request: AnalyzeRequest):
    """
    Analyze a single image for traffic direction.
    
    Direction is determined by geometry (point-in-polygon), not language inference.
    
    Returns:
        - SA_to_LS: Count of vehicles heading from South Africa to Lesotho
        - LS_to_SA: Count of vehicles heading from Lesotho to South Africa
        - unassigned: Vehicles detected but not in any defined lane
        - direction_uncertain: True if too many vehicles are unassigned
    """
    try:
        detector = get_detector()
        result = detector.analyze_traffic(request.image, request.camera_view)
        
        message = None
        if result.direction_uncertain:
            message = "Direction uncertain — reporting total vehicles only"
        
        return AnalyzeResponse(
            success=True,
            SA_to_LS=result.SA_to_LS,
            LS_to_SA=result.LS_to_SA,
            unassigned=result.unassigned,
            total=result.total,
            direction_uncertain=result.direction_uncertain,
            message=message,
            vehicles=result.vehicles,
            breakdown=result.to_dict().get("breakdown")
        )
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-multi", response_model=MultiAnalyzeResponse)
async def analyze_multiple_frames(request: MultiAnalyzeRequest):
    """
    Analyze multiple frames and combine results.
    
    Useful for getting a more accurate count from multiple camera angles.
    """
    try:
        detector = get_detector()
        
        combined = {
            "SA_to_LS": 0,
            "LS_to_SA": 0,
            "unassigned": 0,
            "total": 0,
            "direction_uncertain": False
        }
        
        by_view = {}
        
        for frame in request.frames:
            image = frame.get("image")
            camera_view = frame.get("camera_view", "bridge")
            
            if not image:
                continue
            
            result = detector.analyze_traffic(image, camera_view)
            
            # Store per-view results
            by_view[camera_view] = result.to_dict()
            
            # Note: We don't simply add counts as same vehicle might appear in multiple views
            # For now, we take the max from each direction across views
            combined["SA_to_LS"] = max(combined["SA_to_LS"], result.SA_to_LS)
            combined["LS_to_SA"] = max(combined["LS_to_SA"], result.LS_to_SA)
            combined["unassigned"] += result.unassigned
            combined["total"] = combined["SA_to_LS"] + combined["LS_to_SA"] + combined["unassigned"]
            
            if result.direction_uncertain:
                combined["direction_uncertain"] = True
        
        message = None
        if combined["direction_uncertain"]:
            message = "Direction uncertain in one or more views"
        
        return MultiAnalyzeResponse(
            success=True,
            combined=combined,
            by_view=by_view,
            message=message
        )
    except Exception as e:
        logger.error(f"Multi-analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze-burst", response_model=BurstAnalyzeResponse)
async def analyze_burst(request: BurstAnalyzeRequest):
    """
    Analyze a burst of consecutive frames (oldest first, ~1s apart).

    Direction per vehicle is derived from the motion vector projected onto the
    configured flow axis when speed >= motion_threshold_px_per_sec; otherwise
    from the entry-zone direction prior tagged when the track was first seen.

    Tracker state persists per camera_view across calls so vehicles parked in
    a queue across multiple capture cycles retain their direction.
    """
    try:
        detector = get_detector()

        # Convert Pydantic frames to plain dicts the detector expects.
        frames = [{"image": f.image, "timestamp_ms": f.timestamp_ms}
                  for f in request.frames]

        result = detector.analyze_burst(frames, request.camera_view)

        message = None
        if result.direction_uncertain:
            message = "Direction uncertain — reporting total vehicles only"

        return BurstAnalyzeResponse(
            success=True,
            SA_to_LS=result.SA_to_LS,
            LS_to_SA=result.LS_to_SA,
            unassigned=result.unassigned,
            total=result.total,
            direction_uncertain=result.direction_uncertain,
            message=message,
            vehicles=result.vehicles,
            breakdown=result.to_dict().get("breakdown"),
            flow_metrics=getattr(result, "flow_metrics", None),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Burst analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug", response_model=DebugResponse)
async def debug_detection(request: DebugRequest):
    """
    Generate a debug image with lane polygons and detected vehicles visualized.
    
    Useful for calibrating lane polygons.
    
    NOTE: Detection runs ONCE and returns both the annotated image AND counts.
    """
    try:
        detector = get_detector()
        
        # Generate annotated image AND get counts in one call
        # This ensures the counts match what's drawn on the image
        annotated_image, result = detector.draw_debug_image(request.image, request.camera_view)
        
        return DebugResponse(
            success=True,
            annotated_image=annotated_image,
            counts=result.to_dict()
        )
    except Exception as e:
        logger.error(f"Debug failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/track/{camera_view}/{track_id}")
async def get_track_state(camera_view: str, track_id: int):
    """
    Return the current state of one tracked vehicle. Powers the admin
    live-follower: a human picks a track_id off the debug overlay and the
    UI polls this every few seconds to watch the vehicle's progress
    (entry edge, direction, elapsed, speed) until it crosses an exit zone.
    """
    try:
        detector = get_detector()
        t = detector.get_track_state(camera_view, track_id)
        if t is None:
            return {"success": True, "track": None,
                    "message": "Track not currently active in this view"}
        return {"success": True, "track": t}
    except Exception as e:
        logger.error(f"track-status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/calibrate", response_model=CalibrationResponse)
async def calibrate_lane(request: CalibrationRequest):
    """
    Update a lane polygon for calibration.
    
    Use the /debug endpoint to visualize current polygons,
    then use this endpoint to adjust them.
    """
    try:
        detector = get_detector()
        detector.update_lane_polygon(
            request.camera_view,
            request.lane_name,
            request.polygon
        )
        
        return CalibrationResponse(
            success=True,
            message=f"Updated {request.lane_name} polygon for {request.camera_view} view"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config")
async def get_config():
    """Get current lane configuration."""
    try:
        detector = get_detector()
        return {
            "success": True,
            "config": detector.config
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run with: uvicorn app:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
