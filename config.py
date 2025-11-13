import os

# Configuration for the route finder application

# Shapefile path configuration
# Priority: 
# 1. Environment variable OSM_SHP
# 2. Relative path (for GitHub/deployment)
# 3. Absolute path (for local development)

def get_shapefile_path():
    """
    Get the shapefile path from various sources.
    This allows the app to work in different environments without code changes.
    """
    # Try environment variable first
    env_path = os.environ.get("OSM_SHP")
    if env_path and os.path.exists(env_path):
        return env_path
    
    # Try relative paths (for GitHub/deployment)
    relative_paths = [
        os.path.join(os.path.dirname(__file__), "map", "gis_osm_roads_free_1.shp"),
        os.path.join(os.path.dirname(__file__), "data", "gis_osm_roads_free_1.shp"),
        "map/gis_osm_roads_free_1.shp",
        "data/gis_osm_roads_free_1.shp",
    ]
    
    for path in relative_paths:
        if os.path.exists(path):
            return path
    
    # Try absolute path (for local development)
    # Add your local path here
    absolute_paths = [
        "D:/3r_kurs/4-2/map/map/gis_osm_roads_free_1.shp",
        "D:/3r_kurs/4-2/map/gis_osm_roads_free_1.shp",
    ]
    
    for path in absolute_paths:
        if os.path.exists(path):
            return path
    
    # If nothing found, raise error with helpful message
    raise FileNotFoundError(
        "\n❌ Shapefile not found!\n"
        "Please either:\n"
        "  1. Set environment variable: OSM_SHP=/path/to/shapefile.shp\n"
        "  2. Place shapefile in: ./map/gis_osm_roads_free_1.shp\n"
        "  3. Update absolute paths in config.py\n"
    )

# Other configuration options
DEFAULT_PORT = 5000
DEBUG_MODE = True
TARGET_EPSG = 3857  # Web Mercator projection