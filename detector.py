"""
Vehicle Detection and Lane Assignment Module

This module uses YOLOv8 for vehicle detection and geometric lane assignment
to deterministically identify traffic direction on Maseru Bridge.

Direction is NEVER inferred by language - it's computed from geometry.
"""

import json
import math
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
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


# =============================================
# VELOCITY TRACKER (Phase 1)
# =============================================
# Tracks each vehicle across consecutive frames so direction can be derived
# from the actual motion vector (and from where the vehicle entered the scene)
# instead of from static polygon membership. Lifecycle-based: a track keeps
# its entry-edge direction prior even while it's sitting stationary in a
# queue, which fixes the bridge-congestion → canopy-spillover misattribution.

@dataclass
class Track:
    """One tracked vehicle across multiple frames."""
    track_id: int
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    class_name: str
    confidence: float
    first_seen_ts: float                  # seconds since epoch
    last_seen_ts: float
    bbox_history: List[Tuple[float, Tuple[int, int, int, int]]] = field(default_factory=list)
    entry_edge: Optional[str] = None      # zone name where the track first appeared
    entry_direction: Optional[str] = None # 'LS_to_SA' | 'SA_to_LS' from entry zone
    direction: Optional[str] = None       # final assignment for the current frame
    direction_source: str = "unassigned"  # 'motion' | 'entry' | 'unassigned'
    speed_along_axis: float = 0.0         # signed px/s along the flow axis
    speed_px_per_sec: float = 0.0         # |velocity| magnitude


class VelocityTracker:
    """
    Stateful IoU-based tracker. Matches new detections to existing tracks by
    bbox overlap; tags entry-edge once on first appearance; computes per-track
    velocity from bbox-history endpoints. State persists across analyze calls
    (per camera view) so queue continuity survives between capture cycles.
    """

    def __init__(
        self,
        iou_match_threshold: float = 0.2,
        max_centroid_distance_px: float = 120.0,
        max_age_seconds: float = 300.0,
        history_size: int = 8,
    ):
        self.iou_threshold = iou_match_threshold
        # Fallback when IoU < threshold (e.g. fast-moving vehicle whose bbox in
        # the next frame doesn't overlap with the prior — at 1 Hz burst a car
        # at 50 km/h moves ~140 px on the bridge, well past any IoU match).
        self.max_centroid_distance_px = max_centroid_distance_px
        self.max_age_seconds = max_age_seconds
        self.history_size = history_size
        self._tracks: Dict[int, Track] = {}
        self._next_id: int = 1

    @staticmethod
    def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / max(1e-6, area_a + area_b - inter)

    def _match(
        self,
        detection_bbox: Tuple[int, int, int, int],
        detection_center: Tuple[int, int],
        used: set,
    ) -> Optional[int]:
        # Pass 1: IoU — preferred when bboxes overlap (slow/queued vehicles).
        best_iou = self.iou_threshold
        best_id_iou = None
        for tid, track in self._tracks.items():
            if tid in used:
                continue
            iou = self._iou(track.bbox, detection_bbox)
            if iou > best_iou:
                best_iou = iou
                best_id_iou = tid
        if best_id_iou is not None:
            return best_id_iou

        # Pass 2: nearest centroid within max_centroid_distance_px.
        # Catches fast-moving vehicles whose bboxes don't overlap between frames.
        best_dist = self.max_centroid_distance_px
        best_id_dist = None
        dcx, dcy = detection_center
        for tid, track in self._tracks.items():
            if tid in used:
                continue
            tcx, tcy = track.center
            d = math.hypot(tcx - dcx, tcy - dcy)
            if d < best_dist:
                best_dist = d
                best_id_dist = tid
        return best_id_dist

    def _evict_stale(self, now_ts: float):
        stale = [tid for tid, t in self._tracks.items()
                 if now_ts - t.last_seen_ts > self.max_age_seconds]
        for tid in stale:
            del self._tracks[tid]

    def update_frame(
        self,
        detections: List["DetectedVehicle"],
        timestamp_s: float,
        entry_zones: List[Dict[str, Any]],
        scene_polygons: List[Polygon],
    ) -> List[Track]:
        """
        Update the tracker with detections from a single frame.

        entry_zones: list of {name, polygon (shapely), direction}
        scene_polygons: list of Polygon — vehicles whose center is outside ALL
            of these AND outside all entry zones are treated as off-scene and dropped.

        Returns the list of Track objects matched/created in this frame.
        """
        self._evict_stale(timestamp_s)

        used_ids: set = set()
        active: List[Track] = []

        for det in detections:
            center_pt = Point(det.center)

            # Off-scene filter: vehicle must be inside a lane polygon or an
            # entry zone to be tracked at all. Keeps us from chasing cars on
            # the riverside road, embankment, etc.
            on_scene = any(p.contains(center_pt) for p in scene_polygons)
            in_entry = any(z["polygon"].contains(center_pt) for z in entry_zones)
            if not on_scene and not in_entry:
                continue

            tid = self._match(det.bbox, det.center, used_ids)
            if tid is not None:
                track = self._tracks[tid]
                track.bbox = det.bbox
                track.center = det.center
                track.confidence = det.confidence
                track.class_name = det.class_name
                track.last_seen_ts = timestamp_s
                track.bbox_history.append((timestamp_s, det.bbox))
                if len(track.bbox_history) > self.history_size:
                    track.bbox_history = track.bbox_history[-self.history_size:]
                used_ids.add(tid)
                active.append(track)
            else:
                # New track. Tag entry edge if center is in an entry zone now.
                entry_edge = None
                entry_direction = None
                for zone in entry_zones:
                    if zone["polygon"].contains(center_pt):
                        entry_edge = zone["name"]
                        entry_direction = zone["direction"]
                        break

                new_id = self._next_id
                self._next_id += 1
                new_track = Track(
                    track_id=new_id,
                    bbox=det.bbox,
                    center=det.center,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    first_seen_ts=timestamp_s,
                    last_seen_ts=timestamp_s,
                    bbox_history=[(timestamp_s, det.bbox)],
                    entry_edge=entry_edge,
                    entry_direction=entry_direction,
                )
                self._tracks[new_id] = new_track
                used_ids.add(new_id)
                active.append(new_track)

        return active

    def compute_velocity(self, track: Track) -> Tuple[float, float]:
        """Average velocity in px/s from oldest→newest bbox in history."""
        hist = track.bbox_history
        if len(hist) < 2:
            return (0.0, 0.0)
        t0, b0 = hist[0]
        t1, b1 = hist[-1]
        dt = t1 - t0
        if dt < 0.05:  # < 50ms, too noisy
            return (0.0, 0.0)
        c0 = ((b0[0] + b0[2]) / 2.0, (b0[1] + b0[3]) / 2.0)
        c1 = ((b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0)
        return ((c1[0] - c0[0]) / dt, (c1[1] - c0[1]) / dt)


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

        # Per-view VELOCITY trackers — Phase 1.
        # Used by analyze_burst(). State persists across calls so a queue
        # parked through multiple capture cycles keeps its entry-edge
        # direction prior even when motion drops to zero.
        self._velocity_trackers: Dict[str, VelocityTracker] = {}
        for view_name, view_cfg in self.config["camera_views"].items():
            self._velocity_trackers[view_name] = VelocityTracker(
                iou_match_threshold=0.3,
                max_age_seconds=float(view_cfg.get("track_max_age_seconds", 300)),
                history_size=8,
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

    # =========================================================
    # PHASE 1 — BURST ANALYSIS WITH MOTION + ENTRY-EDGE DIRECTION
    # =========================================================

    def _get_view_runtime(
        self, camera_view: str, actual_width: int, actual_height: int
    ) -> Dict[str, Any]:
        """
        Build per-frame runtime state for a view: scaled flow axis, scaled
        entry zones, scaled scene mask (union of existing lane polygons),
        motion threshold, and pixels-per-meter.
        """
        if camera_view not in self.config["camera_views"]:
            raise ValueError(f"Unknown camera view: {camera_view}")

        view = self.config["camera_views"][camera_view]
        config_width = view.get("image_width", 800)
        config_height = view.get("image_height", 450)
        sx = actual_width / config_width
        sy = actual_height / config_height

        # Flow axis (unit vector pointing LS_to_SA).
        axis_cfg = view.get("flow_axis")
        if axis_cfg:
            f = axis_cfg["from"]; t = axis_cfg["to"]
            ax = (t[0] - f[0]) * sx
            ay = (t[1] - f[1]) * sy
            mag = math.hypot(ax, ay) or 1.0
            axis_unit = (ax / mag, ay / mag)
        else:
            # Default: down the image (works as a no-op when unconfigured).
            axis_unit = (0.0, 1.0)

        # Entry zones (each {name, polygon (Shapely), direction}).
        entry_zones: List[Dict[str, Any]] = []
        for zone in view.get("entry_zones", []):
            scaled = [(int(x * sx), int(y * sy)) for x, y in zone["polygon"]]
            entry_zones.append({
                "name": zone["name"],
                "polygon": Polygon(scaled),
                "direction": zone["direction"],
            })

        # Scene mask = union of all configured lane polygons. Vehicles whose
        # center is outside this AND outside every entry zone are dropped
        # (off-scene — riverside road, embankment, etc.).
        scene_polygons: List[Polygon] = []
        for lane_data in view.get("lanes", {}).values():
            scaled = [(int(x * sx), int(y * sy)) for x, y in lane_data["polygon"]]
            scene_polygons.append(Polygon(scaled))

        return {
            "axis_unit": axis_unit,
            "entry_zones": entry_zones,
            "scene_polygons": scene_polygons,
            "motion_threshold_px_per_sec": float(
                view.get("motion_threshold_px_per_sec", 5.0)
            ),
            "pixels_per_meter": view.get("pixels_per_meter"),
        }

    def _assign_direction_to_track(
        self,
        track: Track,
        axis_unit: Tuple[float, float],
        motion_threshold: float,
        tracker: VelocityTracker,
    ) -> None:
        """
        Set track.direction using motion projection onto the flow axis when
        the vehicle is meaningfully moving; otherwise fall back to the entry-
        edge prior captured when the track was first seen.
        """
        vx, vy = tracker.compute_velocity(track)
        speed_along_axis = vx * axis_unit[0] + vy * axis_unit[1]
        track.speed_along_axis = speed_along_axis
        track.speed_px_per_sec = math.hypot(vx, vy)

        if abs(speed_along_axis) >= motion_threshold:
            track.direction = "LS_to_SA" if speed_along_axis > 0 else "SA_to_LS"
            track.direction_source = "motion"
            return

        if track.entry_direction:
            track.direction = track.entry_direction
            track.direction_source = "entry"
            return

        track.direction = None
        track.direction_source = "unassigned"

    @staticmethod
    def _add_class_to_breakdown(result: LaneCount, class_name: str, direction: str) -> None:
        if direction == "LS_to_SA":
            if class_name == "car":   result.LS_to_SA_cars += 1
            elif class_name == "truck": result.LS_to_SA_trucks += 1
            elif class_name == "bus":   result.LS_to_SA_buses += 1
        elif direction == "SA_to_LS":
            if class_name == "car":   result.SA_to_LS_cars += 1
            elif class_name == "truck": result.SA_to_LS_trucks += 1
            elif class_name == "bus":   result.SA_to_LS_buses += 1

    def analyze_burst(
        self,
        frames: List[Dict[str, Any]],
        camera_view: str = "bridge",
    ) -> LaneCount:
        """
        Analyze a burst of consecutive frames captured ~1s apart.

        frames: list of {'image': base64_str, 'timestamp_ms': int}, oldest first.

        Direction for each tracked vehicle is:
          1. Sign of motion projected onto the flow axis, if speed >= threshold
          2. Otherwise the entry-zone prior tagged when the track was created
          3. Otherwise unassigned

        Tracker state persists per camera_view across calls — a vehicle parked
        in the same spot across multiple capture cycles keeps its direction.
        """
        logger.info("═══════════════════════════════════════")
        logger.info(f"🔍 ANALYZE BURST - view: {camera_view}, frames: {len(frames)}")
        logger.info("═══════════════════════════════════════")

        if not frames:
            return LaneCount(vehicles=[])

        # Decode first frame to learn actual dimensions.
        first_img = self._decode_image(frames[0]["image"])
        actual_height, actual_width = first_img.shape[:2]

        runtime = self._get_view_runtime(camera_view, actual_width, actual_height)
        axis_unit = runtime["axis_unit"]
        entry_zones = runtime["entry_zones"]
        scene_polygons = runtime["scene_polygons"]
        motion_threshold = runtime["motion_threshold_px_per_sec"]

        tracker = self._velocity_trackers.get(camera_view)
        if tracker is None:
            # View added at runtime — instantiate lazily.
            tracker = VelocityTracker(max_age_seconds=300.0)
            self._velocity_trackers[camera_view] = tracker

        # Walk the burst in order. We hold last_active so we count on the
        # newest frame; tracker state carries motion history across frames.
        last_active: List[Track] = []
        for i, frame_data in enumerate(frames):
            img = first_img if i == 0 else self._decode_image(frame_data["image"])
            ts_ms = frame_data.get("timestamp_ms")
            timestamp_s = (ts_ms / 1000.0) if ts_ms is not None else time.time()

            detections = self.detect_vehicles(img, camera_view=camera_view)
            last_active = tracker.update_frame(
                detections=detections,
                timestamp_s=timestamp_s,
                entry_zones=entry_zones,
                scene_polygons=scene_polygons,
            )

        # Count direction on the newest-frame snapshot.
        result = LaneCount(vehicles=[])
        for track in last_active:
            self._assign_direction_to_track(track, axis_unit, motion_threshold, tracker)

            if track.direction == "LS_to_SA":
                result.LS_to_SA += 1
                self._add_class_to_breakdown(result, track.class_name, "LS_to_SA")
            elif track.direction == "SA_to_LS":
                result.SA_to_LS += 1
                self._add_class_to_breakdown(result, track.class_name, "SA_to_LS")
            else:
                result.unassigned += 1

            result.vehicles.append({
                "bbox": list(track.bbox),
                "center": list(track.center),
                "confidence": round(track.confidence, 3),
                "class": track.class_name,
                "lane": track.direction or "unassigned",
                "track_id": track.track_id,
                "direction_source": track.direction_source,
                "speed_along_axis_px_per_sec": round(track.speed_along_axis, 2),
                "speed_px_per_sec": round(track.speed_px_per_sec, 2),
                "entry_edge": track.entry_edge,
                "stationary": track.direction_source != "motion",
            })

        result.total = len(last_active)

        if result.total > 0:
            unassigned_ratio = result.unassigned / result.total
            threshold = self.config["detection_settings"]["unassigned_threshold"]
            if unassigned_ratio > threshold:
                result.direction_uncertain = True
                logger.warning(
                    f"⚠️ Direction uncertain: {unassigned_ratio:.1%} unassigned "
                    f"> {threshold:.0%} threshold"
                )

        n_motion = sum(1 for t in last_active if t.direction_source == "motion")
        n_entry  = sum(1 for t in last_active if t.direction_source == "entry")
        logger.info(
            f"✅ BURST RESULT: SA→LS={result.SA_to_LS}, LS→SA={result.LS_to_SA}, "
            f"unassigned={result.unassigned}, total={result.total} "
            f"(by motion: {n_motion}, by entry-edge: {n_entry})"
        )

        # Attach speed + transit-time metrics as a flow_metrics dict on the
        # vehicles list. The LaneCount dataclass itself stays back-compat;
        # extras ride on the response via to_dict() in the endpoint.
        ppm = runtime.get("pixels_per_meter")
        flow_metrics = self._compute_flow_metrics(
            last_active, axis_unit, motion_threshold, ppm, camera_view
        )
        # Store on the result via a stash attribute the endpoint will pick up.
        result.flow_metrics = flow_metrics  # type: ignore[attr-defined]
        return result

    def _compute_flow_metrics(
        self,
        tracks: List[Track],
        axis_unit: Tuple[float, float],
        motion_threshold: float,
        pixels_per_meter: Optional[float],
        camera_view: str,
    ) -> Dict[str, Any]:
        """
        Aggregate per-track speeds into directional means and transit-time
        estimates. Returns a dict suitable for direct JSON serialization.
        """
        ls_speeds = [t.speed_along_axis for t in tracks
                     if t.direction == "LS_to_SA" and t.direction_source == "motion"]
        sa_speeds = [t.speed_along_axis for t in tracks
                     if t.direction == "SA_to_LS" and t.direction_source == "motion"]

        def avg(xs):
            return (sum(xs) / len(xs)) if xs else 0.0

        ls_mean_px = avg(ls_speeds)
        sa_mean_px = avg([abs(v) for v in sa_speeds])

        ls_kmh = sa_kmh = None
        ls_transit_s = sa_transit_s = None
        view_cfg = self.config["camera_views"].get(camera_view, {})
        segment_m = float(view_cfg.get("segment_length_m", 50.0))
        # Per-view processing pace used as the wait-time signal when motion
        # is zero (queued / stalled). Border bridge ~30s/car, customs ~45s/car.
        queue_sec_per_vehicle = float(view_cfg.get("queue_seconds_per_vehicle", 30.0))

        if pixels_per_meter and pixels_per_meter > 0:
            mps_per_px = 1.0 / pixels_per_meter
            ls_mps = ls_mean_px * mps_per_px
            sa_mps = sa_mean_px * mps_per_px
            if ls_mps > 0: ls_kmh = round(ls_mps * 3.6, 1)
            if sa_mps > 0: sa_kmh = round(sa_mps * 3.6, 1)
            if ls_mps > 0: ls_transit_s = round(segment_m / ls_mps, 1)
            if sa_mps > 0: sa_transit_s = round(segment_m / sa_mps, 1)

        # Queue length per direction. A track counted here is either:
        #   - direction_source='entry' (stationary, direction came from the
        #     entry-edge prior — typical for a queued vehicle), or
        #   - direction_source='motion' (still moving — included so vehicles
        #     ahead of you in a slow-moving queue also contribute).
        # We deliberately exclude 'unassigned' tracks because we can't say
        # which queue they're part of.
        ls_total = sum(1 for t in tracks if t.direction == "LS_to_SA")
        sa_total = sum(1 for t in tracks if t.direction == "SA_to_LS")
        ls_queued_only = sum(1 for t in tracks
                             if t.direction == "LS_to_SA" and t.direction_source == "entry")
        sa_queued_only = sum(1 for t in tracks
                             if t.direction == "SA_to_LS" and t.direction_source == "entry")

        # Wait-time estimate = queue_clear + free_flow_drive_through.
        # When everyone is stopped, free_flow component is 0/None and the
        # estimate is dominated by queue_length * sec_per_vehicle.
        def estimate_seconds(total_count, transit_s):
            queue_part = total_count * queue_sec_per_vehicle
            transit_part = transit_s if transit_s else 0
            total = queue_part + transit_part
            return round(total, 1) if total > 0 else None

        ls_wait_s = estimate_seconds(ls_total, ls_transit_s)
        sa_wait_s = estimate_seconds(sa_total, sa_transit_s)

        return {
            "pixels_per_meter": pixels_per_meter,
            "segment_length_m": segment_m,
            "queue_seconds_per_vehicle": queue_sec_per_vehicle,
            "LS_to_SA": {
                "moving_count": len(ls_speeds),
                "queued_count": ls_queued_only,
                "total_in_lane": ls_total,
                "mean_speed_kmh": ls_kmh,
                "free_flow_transit_seconds": ls_transit_s,
                "estimated_wait_seconds": ls_wait_s,
            },
            "SA_to_LS": {
                "moving_count": len(sa_speeds),
                "queued_count": sa_queued_only,
                "total_in_lane": sa_total,
                "mean_speed_kmh": sa_kmh,
                "free_flow_transit_seconds": sa_transit_s,
                "estimated_wait_seconds": sa_wait_s,
            },
        }

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
