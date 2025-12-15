# Traffic Detection Service

A deterministic vehicle detection and lane assignment service for Maseru Bridge traffic analysis.

## Key Principle

**Direction is determined by GEOMETRY, not language inference.**

- Uses YOLOv8 for vehicle detection
- Uses point-in-polygon geometry for lane assignment
- Returns exact counts: `{SA_to_LS: X, LS_to_SA: Y, unassigned: Z}`

## API Endpoints

### `POST /analyze`
Analyze a single image for traffic direction.

**Request:**
```json
{
  "image": "<base64_encoded_image>",
  "camera_view": "bridge"  // or "canopy", "engen"
}
```

**Response:**
```json
{
  "success": true,
  "SA_to_LS": 1,
  "LS_to_SA": 0,
  "unassigned": 0,
  "total": 1,
  "direction_uncertain": false,
  "vehicles": [
    {
      "bbox": [640, 300, 750, 400],
      "center": [695, 350],
      "confidence": 0.85,
      "class": "truck",
      "lane": "SA_to_LS"
    }
  ]
}
```

### `POST /analyze-multi`
Analyze multiple frames from different camera angles.

### `POST /debug`
Generate an annotated debug image showing lane polygons and detected vehicles.

### `POST /calibrate`
Update lane polygon coordinates for a specific camera view.

### `GET /config`
Get current lane configuration.

## Deployment on Render

### Option 1: Docker (Recommended)

1. Create a new Web Service on Render
2. Connect to GitHub repo containing this code
3. Choose "Docker" as environment
4. Set plan to "Starter" ($7/month) - free tier won't work for ML
5. Deploy

### Option 2: Manual

1. Create a new Web Service
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Add buildpack for OpenCV dependencies

## Calibration

The lane polygons need to be calibrated for accurate direction detection.

### Step 1: Get a debug image

```bash
curl -X POST https://your-service.onrender.com/debug \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64>", "camera_view": "bridge"}'
```

This returns an annotated image showing current lane polygons.

### Step 2: Identify correct polygon coordinates

Looking at the bridge camera:
- The orange pole is on the **RIGHT** side of the image
- SA→LS lane is on the **RIGHT** (next to orange pole)
- LS→SA lane is on the **LEFT** (away from orange pole)

Polygon format: `[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]`

### Step 3: Update polygons

```bash
curl -X POST https://your-service.onrender.com/calibrate \
  -H "Content-Type: application/json" \
  -d '{
    "camera_view": "bridge",
    "lane_name": "SA_to_LS",
    "polygon": [[700, 200], [1100, 200], [1200, 650], [750, 650]]
  }'
```

## Integration with Node.js Server

Update your `server.js` to call this service:

```javascript
const DETECTOR_URL = process.env.DETECTOR_URL || 'https://traffic-detector.onrender.com';

async function getTrafficCounts(frameBase64, cameraView) {
  const response = await fetch(`${DETECTOR_URL}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image: frameBase64,
      camera_view: cameraView
    })
  });
  
  return await response.json();
}

// Then use the counts in your Claude prompt:
// "There are X vehicles heading SA→LS and Y vehicles heading LS→SA"
// Claude generates human-readable text, but NEVER decides direction
```

## Safety Guard

If more than 25% of detected vehicles are unassigned (outside defined lanes):
- `direction_uncertain` becomes `true`
- The chatbot should say: "Direction uncertain — reporting total vehicles only"

This prevents publishing incorrect direction information.

## Files

- `app.py` - FastAPI application
- `detector.py` - YOLO + ROI detection logic
- `lane_config.json` - Lane polygon definitions
- `requirements.txt` - Python dependencies
- `Dockerfile` - Container definition
- `render.yaml` - Render deployment config
