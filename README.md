# UB Route Finder – Traffic Flow Visualization

A web-based route finding application that compares different pathfinding algorithms (Dijkstra, BFS, DFS) with traffic simulation capabilities.

## Features

- 🗺️ Interactive map interface using Leaflet
- 🔄 Multiple pathfinding algorithms (Dijkstra, BFS, DFS)
- 🚦 Traffic flow simulation
- 📊 Performance comparison charts
- ⚡ Real-time route visualization with directional arrows

## Setup

### 1. Install Dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
or 
.\.venv\Scripts\Activate

pip install -r requirements.txt



pip install flask geopandas shapely pyproj
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
geopandas>=0.13.0
shapely>=2.0.0
pyproj>=3.5.0
```

### 2. Configure Shapefile Path

The application needs OpenStreetMap road data in shapefile format. You have several options:

#### Option A: Place file in project directory (Recommended for GitHub)

```
your-project/
├── app.py
├── map/
│   └── gis_osm_roads_free_1.shp
│   └── gis_osm_roads_free_1.shx
│   └── gis_osm_roads_free_1.dbf
│   └── ... (other shapefile components)
```

#### Option B: Set environment variable

```bash
# Windows
set OSM_SHP=D:\path\to\your\gis_osm_roads_free_1.shp

# Linux/Mac
export OSM_SHP=/path/to/your/gis_osm_roads_free_1.shp
```

#### Option C: Update config.py

Edit the `absolute_paths` list in `config.py` to include your local path.

### 3. Run the Application

```bash
python app.py
```

Then open your browser to: http://localhost:5000

## Usage

1. **Select Start Point**: Click on the map to place a green marker (🟢)
2. **Select End Point**: Click again to place a red marker (🔴)
3. **Choose Algorithm**: Select from dropdown (Dijkstra, BFS, or DFS)
4. **Enable Traffic** (optional): Check the traffic toggle for traffic simulation
5. **Run**: Click "Дүйцэтгэх" to find a route, or "Харьцуулах" to compare all algorithms

## Algorithm Comparison

- **Dijkstra**: Finds shortest weighted path (optimal for distance)
- **BFS**: Finds path with fewest edges (optimal for hops)
- **DFS**: Finds any path (may not be optimal)

## Downloading Map Data

Download OpenStreetMap shapefiles from:
- [Geofabrik](https://download.geofabrik.de/) - Free OSM extracts
- [BBBike](https://extract.bbbike.org/) - Custom extracts

For Ulaanbaatar, Mongolia:
```
https://download.geofabrik.de/asia/mongolia-latest-free.shp.zip
```

## File Structure

```
route-finder/
├── app.py                 # Main Flask application
├── algorithms.py          # Pathfinding algorithms
├── build_graph.py         # Graph construction from shapefile
├── config.py              # Configuration settings
├── templates/
│   └── index.html        # Web interface
├── static/
│   └── style.css         # Styling
└── map/                  # Place your shapefiles here (gitignored)
```

## License

MIT License - feel free to use and modify!


# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py

# Or use uvicorn directly with auto-reload
uvicorn app:app --host 0.0.0.0 --port 5000 --reload
```

## ✨ New Features with FastAPI

### 1. **Automatic API Documentation**
- Swagger UI: http://localhost:5000/docs
- ReDoc: http://localhost:5000/redoc

### 2. **Better Performance**
- Async support ready (can be added later)
- Built on Starlette (fast ASGI framework)
- Pydantic data validation

### 3. **Type Safety**
- Automatic request validation
- Better error messages
- IDE autocomplete support

### 4. **Health Check Endpoint**
```
GET http://localhost:5000/health
```
Returns: `{"status": "healthy", "nodes": 12345}`

## 📊 File Structure (No Changes)
```
route-finder/
├── app.py              ✅ Updated to FastAPI
├── algorithms.py       ✅ No changes
├── build_graph.py      ✅ No changes
├── config.py          ✅ No changes
├── requirements.txt    ✅ Updated
├── README.md          ✅ Updated docs
├── .gitignore         ✅ No changes
├── templates/
│   └── index.html     ✅ Minor update
├── static/
│   └── style.css      ✅ No changes
└── map/
    └── (your shapefiles)
