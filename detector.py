"""
Vehicle Detection and Lane Assignment Module

This module uses YOLOv8 for vehicle detection and geometric lane assignment
to deterministically identify traffic direction on Maseru Bridge.

Direction is NEVER inferred by language - it's computed from geometry.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from shapely.geometry import Point, Polygon
from ultralytics import YOLO
import cv2
from PIL import Image
import io
import base64
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StationaryTracker:
    """
    Tracks vehicle positions across frames to identify stationary/parked vehicles.
    A vehicle at the same location for >= N frames is considered parked and excluded
    from directional counts. This handles the parked truck problem at Maseru Bridge.
    """

    def __init__(self, iou_threshold: float = 0.6, frames_to_mark_stationary: int = 4):
        self.iou_threshold = iou_threshold
        self.frames_to_mark_stationary = frames_to_mark_stationary
        # Dict: bbox_key -> consecutive_frame_count
        self._tracked: Dict[str, int] = {}
        self._stationary: set = set()

    def _bbox_key(self, bbox: Tuple[int, int, int, int]) -> str:
        """Snap bbox to a coarse grid to handle minor pixel jitter."""
        x1, y1, x2, y2 = bbox
        snap = 15  # pixels
        return f"{x1//snap},{y1//snap},{x2//snap},{y2//snap}"

    def _iou(self, a: Tuple, b: Tuple) -> float:
        """Compute intersection over union between two bboxes."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter)

    def update(self, vehicles: List['DetectedVehicle'], camera_view: str) -> List['DetectedVehicle']:
        """
        Update tracker with new frame's vehicles.
        Returns vehicles with stationary ones flagged.
        Stationary vehicles still appear in the debug image but are excluded from counts.
        """
        current_keys = set()

        for v in vehicles:
            key = self._bbox_key(v.bbox)

            # Check if this vehicle matches any existing tracked bbox via IoU
            matched_key = None
            for tracked_key in list(self._tracked.keys()):
                # Parse tracked bbox from key
                parts = list(map(int, tracked_key.split(',')))
                snap = 15
                approx_bbox = (parts[0]*snap, parts[1]*snap, parts[2]*snap, parts[3]*snap)
                if self._iou(v.bbox, approx_bbox) >= self.iou_threshold:
                    matched_key = tracked_key
                    break

            if matched_key:
                self._tracked[matched_key] += 1
                current_keys.add(matched_key)
                if self._tracked[matched_key] >= self.frames_to_mark_stationary:
                    self._stationary.add(matched_key)
                    v.is_stationary = True
                    logger.info(f"🅿️  PARKED vehicle detected at {v.center} (frame #{self._tracked[matched_key]}) - excluded from counts")
                else:
                    v.is_stationary = False
            else:
                self._tracked[key] = 1
                current_keys.add(key)
                v.is_stationary = False

        # Remove tracked vehicles that disappeared (drove away)
        gone_keys = set(self._tracked.keys()) - current_keys
        for k in gone_keys:
            del self._tracked[k]
            self._stationary.discard(k)
            logger.info(f"🚗 Vehicle left the scene (key: {k})")

        return vehicles


@dataclass
class DetectedVehicle:
    """Represents a detected vehicle with its properties."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]  # cx, cy
    confidence: float
    class_id: int
    class_name: str
    assigned_lane: Optional[str] = None
    is_stationary: bool = False  # True = parked/blocked truck, excluded from counts


@dataclass
class LaneCount:
    """Traffic count results for a single analysis."""
    SA_to_LS: int = 0
    LS_to_SA: int = 0
    unassigned: int = 0
    total: int = 0
    direction_uncertain: bool = False
    vehicles: List[Dict] = None
    # Vehicle type breakdown per lane
    SA_to_LS_cars: int = 0
    SA_to_LS_trucks: int = 0
    SA_to_LS_buses: int = 0
    LS_to_SA_cars: int = 0
    LS_to_SA_trucks: int = 0
    LS_to_SA_buses: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "SA_to_LS": self.SA_to_LS,
            "LS_to_SA": self.LS_to_SA,
            "unassigned": self.unassigned,
            "total": self.total,
            "direction_uncertain": self.direction_uncertain,
            "vehicles": self.vehicles or [],
            "breakdown": {
                "SA_to_LS": {
                    "cars": self.SA_to_LS_cars,
                    "trucks": self.SA_to_LS_trucks,
                    "buses": self.SA_to_LS_buses
                },
                "LS_to_SA": {
                    "cars": self.LS_to_SA_cars,
                    "trucks": self.LS_to_SA_trucks,
                    "buses": self.LS_to_SA_buses
                }
            }
        }


class LaneDetector:
    """
    Deterministic lane-based vehicle detection and direction assignment.
    
    Direction is computed using point-in-polygon geometry, NOT language inference.
    """
    
    def __init__(self, config_path: str = "lane_config.json"):
        """Initialize the detector with lane configuration."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.model = None
        self._load_model()
        # Per-view stationary trackers (one per camera angle)
        self._stationary_trackers: Dict[str, StationaryTracker] = {}
        for view_name, view_cfg in self.config["camera_views"].items():
            n_frames = view_cfg.get("stationary_frames_threshold", 4)
            self._stationary_trackers[view_name] = StationaryTracker(
                iou_threshold=0.6,
                frames_to_mark_stationary=n_frames
            )
        
    def _load_config(self) -> Dict:
        """Load lane configuration from JSON file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Lane config not found: {self.config_path}")
    
    def _load_model(self):
        """Load YOLOv8 model for vehicle detection."""
        try:
            # Use YOLOv8n (nano) for speed, or YOLOv8s/m for better accuracy
            self.model = YOLO('yolov8n.pt')
            logger.info("✅ YOLOv8 model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO model: {e}")
            raise
    
    def _get_lane_polygons(self, camera_view: str, actual_width: int, actual_height: int) -> Dict[str, Polygon]:
        """
        Get Shapely polygon objects for the specified camera view.
        Scales polygons to match actual image dimensions.
        """
        if camera_view not in self.config["camera_views"]:
            raise ValueError(f"Unknown camera view: {camera_view}")
        
        view_config = self.config["camera_views"][camera_view]
        
        # Get configured dimensions
        config_width = view_config.get("image_width", 1196)
        config_height = view_config.get("image_height", 735)
        
        # Calculate scale factors
        scale_x = actual_width / config_width
        scale_y = actual_height / config_height
        
        logger.info(f"📐 Scaling polygons: config {config_width}x{config_height} → actual {actual_width}x{actual_height} (scale: {scale_x:.3f}, {scale_y:.3f})")
        
        polygons = {}
        
        for lane_name, lane_data in view_config.get("lanes", {}).items():
            # Scale each coordinate
            scaled_coords = [
                (int(x * scale_x), int(y * scale_y)) 
                for x, y in lane_data["polygon"]
            ]
            polygons[lane_name] = Polygon(scaled_coords)
            logger.info(f"   → {lane_name}: {scaled_coords}")
        
        return polygons
    
    def _get_scaled_polygon_coords(self, camera_view: str, actual_width: int, actual_height: int) -> Dict[str, List]:
        """Get scaled polygon coordinates (for debug drawing)."""
        if camera_view not in self.config["camera_views"]:
            raise ValueError(f"Unknown camera view: {camera_view}")
        
        view_config = self.config["camera_views"][camera_view]
        
        # Get configured dimensions
        config_width = view_config.get("image_width", 1196)
        config_height = view_config.get("image_height", 735)
        
        # Calculate scale factors
        scale_x = actual_width / config_width
        scale_y = actual_height / config_height
        
        scaled_polygons = {}
        
        for lane_name, lane_data in view_config.get("lanes", {}).items():
            scaled_coords = [
                [int(x * scale_x), int(y * scale_y)] 
                for x, y in lane_data["polygon"]
            ]
            scaled_polygons[lane_name] = {
                "polygon": scaled_coords,
                "color": lane_data.get("color", [255, 255, 0])
            }
        
        return scaled_polygons
    
    def _assign_lane(self, center: Tuple[int, int], polygons: Dict[str, Polygon], 
                     dead_zone_px: int = 0) -> Optional[str]:
        """
        Assign a vehicle to a lane using point-in-polygon geometry.
        
        dead_zone_px: if > 0, vehicles whose center is within this many pixels of
        a polygon boundary are treated as unassigned to avoid wrong-lane errors
        caused by overtaking near parked trucks.
        """
        point = Point(center)
        
        for lane_name, polygon in polygons.items():
            if polygon.contains(point):
                # Check if we're too close to the boundary (dead zone)
                if dead_zone_px > 0:
                    dist_to_boundary = polygon.exterior.distance(point)
                    if dist_to_boundary < dead_zone_px:
                        logger.info(f"   ⚠️  Vehicle at {center} is within {dead_zone_px}px dead zone (dist={dist_to_boundary:.1f}px) → UNASSIGNED")
                        return None
                return lane_name
        
        return None
    
    def _decode_image(self, image_data: str) -> np.ndarray:
        """Decode base64 image to numpy array."""
        logger.info(f"📥 Decoding image, data length: {len(image_data)} chars")
        
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
            logger.info("   → Removed data URL prefix")
        
        try:
            image_bytes = base64.b64decode(image_data)
            logger.info(f"   → Decoded {len(image_bytes)} bytes")
            
            image = Image.open(io.BytesIO(image_bytes))
            logger.info(f"   → Image format: {image.format}, mode: {image.mode}, size: {image.size}")
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
                logger.info(f"   → Converted to RGB")
            
            arr = np.array(image)
            logger.info(f"   → NumPy array shape: {arr.shape}, dtype: {arr.dtype}")
            
            return arr
        except Exception as e:
            logger.error(f"❌ Image decode error: {e}")
            raise
    
    def detect_vehicles(self, image: np.ndarray, camera_view: str = None) -> List[DetectedVehicle]:
        """
        Detect vehicles in the image using YOLOv8.
        Uses per-view confidence threshold if camera_view is provided.
        """
        settings = self.config["detection_settings"]
        vehicle_classes = settings["vehicle_classes"]
        # Per-view threshold takes priority over global default
        if camera_view and camera_view in self.config["camera_views"]:
            confidence_threshold = self.config["camera_views"][camera_view].get(
                "confidence_threshold", settings["confidence_threshold"]
            )
        else:
            confidence_threshold = settings["confidence_threshold"]
        min_size = settings.get("min_vehicle_size", 30)
        class_names = settings["class_names"]
        
        logger.info(f"🚗 Running YOLO detection...")
        logger.info(f"   → Image shape: {image.shape}")
        logger.info(f"   → Vehicle classes: {vehicle_classes}")
        logger.info(f"   → Confidence threshold: {confidence_threshold}")
        logger.info(f"   → Min size: {min_size}")
        
        # Run YOLO detection
        results = self.model(image, verbose=False)[0]
        
        # Log ALL raw detections
        all_boxes = len(results.boxes)
        logger.info(f"🤖 YOLO raw detections: {all_boxes}")
        
        vehicles = []
        filtered_class = 0
        filtered_conf = 0
        filtered_size = 0
        
        for i, box in enumerate(results.boxes):
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width = x2 - x1
            height = y2 - y1
            
            # Log every detection
            logger.info(f"   [{i}] class={class_id}, conf={confidence:.3f}, box=({x1},{y1},{x2},{y2}), size={width}x{height}")
            
            # Filter by vehicle classes
            if class_id not in vehicle_classes:
                filtered_class += 1
                logger.info(f"       → FILTERED: not a vehicle class")
                continue
            
            # Filter by confidence
            if confidence < confidence_threshold:
                filtered_conf += 1
                logger.info(f"       → FILTERED: low confidence")
                continue
            
            # Filter by minimum size
            if width < min_size or height < min_size:
                filtered_size += 1
                logger.info(f"       → FILTERED: too small")
                continue
            
            # Calculate center point
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            logger.info(f"       → KEPT: center=({cx},{cy})")
            
            vehicles.append(DetectedVehicle(
                bbox=(x1, y1, x2, y2),
                center=(cx, cy),
                confidence=confidence,
                class_id=class_id,
                class_name=class_names.get(str(class_id), "vehicle")
            ))
        
        logger.info(f"📊 Detection summary: {filtered_class} non-vehicle, {filtered_conf} low-conf, {filtered_size} too-small, {len(vehicles)} kept")
        
        return vehicles
    
    def analyze_traffic(self, image_data: str, camera_view: str = "bridge") -> LaneCount:
        """
        Main analysis function - detects vehicles and assigns to lanes.
        
        Args:
            image_data: Base64 encoded image
            camera_view: Which camera view ("bridge", "canopy", "engen")
        
        Returns:
            LaneCount with deterministic direction counts
        """
        logger.info(f"═══════════════════════════════════════")
        logger.info(f"🔍 ANALYZE TRAFFIC - camera_view: {camera_view}")
        logger.info(f"═══════════════════════════════════════")
        
        # Decode image
        image = self._decode_image(image_data)
        
        # Get actual image dimensions
        actual_height, actual_width = image.shape[:2]
        
        logger.info(f"📸 Image dimensions: {actual_width}x{actual_height}")
        
        # Get per-view dead zone
        view_cfg = self.config["camera_views"].get(camera_view, {})
        dead_zone_px = view_cfg.get("boundary_dead_zone_px", 0)
        
        # Get lane polygons SCALED to actual image size
        polygons = self._get_lane_polygons(camera_view, actual_width, actual_height)
        
        # Detect vehicles (using per-view confidence threshold)
        vehicles = self.detect_vehicles(image, camera_view=camera_view)
        
        # Update stationary tracker — flags parked/blocked trucks
        tracker = self._stationary_trackers.get(camera_view)
        if tracker:
            vehicles = tracker.update(vehicles, camera_view)
        
        logger.info(f"🎯 Assigning {len(vehicles)} vehicles to lanes (dead_zone={dead_zone_px}px)...")
        
        # Assign each vehicle to a lane using geometry
        result = LaneCount(vehicles=[])
        
        for vehicle in vehicles:
            lane = self._assign_lane(vehicle.center, polygons, dead_zone_px=dead_zone_px)
            vehicle.assigned_lane = lane
            
            logger.info(f"   → {vehicle.class_name} at {vehicle.center} → {lane or 'UNASSIGNED'} {'[PARKED]' if vehicle.is_stationary else ''}")
            
            # Skip stationary/parked vehicles from directional counts
            if vehicle.is_stationary:
                result.unassigned += 1  # Track them but don't count direction
                result.vehicles.append({
                    "bbox": vehicle.bbox,
                    "center": vehicle.center,
                    "confidence": round(vehicle.confidence, 3),
                    "class": vehicle.class_name,
                    "lane": "PARKED",
                    "stationary": True
                })
                continue
            
            # Count by lane
            if lane == "SA_to_LS":
                result.SA_to_LS += 1
                if vehicle.class_name == "car":
                    result.SA_to_LS_cars += 1
                elif vehicle.class_name == "truck":
                    result.SA_to_LS_trucks += 1
                elif vehicle.class_name == "bus":
                    result.SA_to_LS_buses += 1
            elif lane == "LS_to_SA":
                result.LS_to_SA += 1
                if vehicle.class_name == "car":
                    result.LS_to_SA_cars += 1
                elif vehicle.class_name == "truck":
                    result.LS_to_SA_trucks += 1
                elif vehicle.class_name == "bus":
                    result.LS_to_SA_buses += 1
            else:
                result.unassigned += 1
            
            # Add to vehicle list
            result.vehicles.append({
                "bbox": vehicle.bbox,
                "center": vehicle.center,
                "confidence": round(vehicle.confidence, 3),
                "class": vehicle.class_name,
                "lane": lane,
                "stationary": False
            })
        
        result.total = len(vehicles)
        
        logger.info(f"═══════════════════════════════════════")
        logger.info(f"✅ FINAL RESULT: SA→LS={result.SA_to_LS} ({result.SA_to_LS_cars}c/{result.SA_to_LS_trucks}t/{result.SA_to_LS_buses}b), LS→SA={result.LS_to_SA} ({result.LS_to_SA_cars}c/{result.LS_to_SA_trucks}t/{result.LS_to_SA_buses}b), unassigned={result.unassigned}, total={result.total}")
        logger.info(f"═══════════════════════════════════════")
        
        # Safety guard: if too many unassigned, flag as uncertain
        unassigned_threshold = self.config["detection_settings"]["unassigned_threshold"]
        if result.total > 0:
            unassigned_ratio = result.unassigned / result.total
            if unassigned_ratio > unassigned_threshold:
                result.direction_uncertain = True
                logger.warning(f"⚠️ Direction uncertain: {unassigned_ratio:.1%} unassigned > {unassigned_threshold:.0%} threshold")
        
        return result
    
    def draw_debug_image(self, image_data: str, camera_view: str = "bridge") -> Tuple[str, LaneCount]:
        """
        Create a debug image showing lane polygons and detected vehicles.
        Useful for calibration.
        
        Returns tuple of (base64 encoded annotated image, LaneCount results)
        """
        logger.info(f"═══════════════════════════════════════")
        logger.info(f"🎨 DRAW DEBUG IMAGE - camera_view: {camera_view}")
        logger.info(f"═══════════════════════════════════════")
        
        # Decode image
        image = self._decode_image(image_data)
        
        # Get actual image dimensions
        actual_height, actual_width = image.shape[:2]
        
        # Get per-view dead zone
        view_cfg = self.config["camera_views"].get(camera_view, {})
        dead_zone_px = view_cfg.get("boundary_dead_zone_px", 0)
        
        # Get SCALED polygon coordinates for drawing
        scaled_polygons = self._get_scaled_polygon_coords(camera_view, actual_width, actual_height)
        
        # Get SCALED Shapely polygons for lane assignment
        polygons = self._get_lane_polygons(camera_view, actual_width, actual_height)
        
        # IMPORTANT: Detect vehicles FIRST on the CLEAN image (before drawing polygons)
        vehicles = self.detect_vehicles(image, camera_view=camera_view)
        
        # Update stationary tracker
        tracker = self._stationary_trackers.get(camera_view)
        if tracker:
            vehicles = tracker.update(vehicles, camera_view)
        
        # Now assign lanes and count
        result = LaneCount(vehicles=[])
        
        for vehicle in vehicles:
            lane = self._assign_lane(vehicle.center, polygons, dead_zone_px=dead_zone_px)
            vehicle.assigned_lane = lane
            
            logger.info(f"   → {vehicle.class_name} at {vehicle.center} → {lane or 'UNASSIGNED'} {'[PARKED]' if vehicle.is_stationary else ''}")
            
            if vehicle.is_stationary:
                result.unassigned += 1
                result.vehicles.append({
                    "bbox": vehicle.bbox, "center": vehicle.center,
                    "confidence": round(vehicle.confidence, 3),
                    "class": vehicle.class_name, "lane": "PARKED", "stationary": True
                })
                continue
            
            if lane == "SA_to_LS":
                result.SA_to_LS += 1
                if vehicle.class_name == "car": result.SA_to_LS_cars += 1
                elif vehicle.class_name == "truck": result.SA_to_LS_trucks += 1
                elif vehicle.class_name == "bus": result.SA_to_LS_buses += 1
            elif lane == "LS_to_SA":
                result.LS_to_SA += 1
                if vehicle.class_name == "car": result.LS_to_SA_cars += 1
                elif vehicle.class_name == "truck": result.LS_to_SA_trucks += 1
                elif vehicle.class_name == "bus": result.LS_to_SA_buses += 1
            else:
                result.unassigned += 1
            
            result.vehicles.append({
                "bbox": vehicle.bbox, "center": vehicle.center,
                "confidence": round(vehicle.confidence, 3),
                "class": vehicle.class_name, "lane": lane, "stationary": False
            })
        
        result.total = len(vehicles)
        
        logger.info(f"✅ DEBUG RESULT: SA→LS={result.SA_to_LS} ({result.SA_to_LS_cars}c/{result.SA_to_LS_trucks}t/{result.SA_to_LS_buses}b), LS→SA={result.LS_to_SA} ({result.LS_to_SA_cars}c/{result.LS_to_SA_trucks}t/{result.LS_to_SA_buses}b)")
        
        # NOW draw polygons (after detection)
        for lane_name, lane_data in scaled_polygons.items():
            coords = np.array(lane_data["polygon"], np.int32)
            color = tuple(lane_data.get("color", [255, 255, 0]))
            cv2.polylines(image, [coords], True, color, 2)
            
            # Add label
            centroid = coords.mean(axis=0).astype(int)
            cv2.putText(image, lane_name, tuple(centroid), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw vehicles on top
        for vehicle in vehicles:
            x1, y1, x2, y2 = vehicle.bbox
            cx, cy = vehicle.center
            lane = vehicle.assigned_lane
            
            # Color based on lane assignment
            if vehicle.is_stationary:
                color = (0, 165, 255)  # Orange for parked/stationary
            elif lane == "SA_to_LS":
                color = (0, 255, 0)  # Green
            elif lane == "LS_to_SA":
                color = (0, 0, 255)  # Red (BGR)
            else:
                color = (0, 255, 255)  # Yellow for unassigned
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            cv2.circle(image, (cx, cy), 5, color, -1)
            
            # Label
            if vehicle.is_stationary:
                label = f"{vehicle.class_name} (PARKED)"
            else:
                label = f"{vehicle.class_name} ({lane or 'unassigned'})"
            cv2.putText(image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Encode back to base64
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buffer).decode('utf-8'), result
    
    def update_lane_polygon(self, camera_view: str, lane_name: str, polygon: List[List[int]]):
        """Update a lane polygon in the configuration."""
        if camera_view not in self.config["camera_views"]:
            raise ValueError(f"Unknown camera view: {camera_view}")
        
        if lane_name not in self.config["camera_views"][camera_view]["lanes"]:
            raise ValueError(f"Unknown lane: {lane_name}")
        
        self.config["camera_views"][camera_view]["lanes"][lane_name]["polygon"] = polygon
        
        # Save updated config
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        return True


# Singleton instance
_detector = None

def get_detector() -> LaneDetector:
    """Get or create the singleton detector instance."""
    global _detector
    if _detector is None:
        _detector = LaneDetector()
    return _detector
