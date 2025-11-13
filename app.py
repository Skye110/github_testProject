from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from build_graph import build_graph_from_shp
from algorithms import bfs_shortest, dfs_path_safe, dijkstra
from pyproj import Transformer
import os
import logging
import time
import copy
from pathlib import Path

# Import configuration
try:
    from config import get_shapefile_path
except ImportError:
    def get_shapefile_path():
        paths = [
            os.environ.get("OSM_SHP"),
            os.path.join(os.path.dirname(__file__), "map", "gis_osm_roads_free_1.shp"),
            "map/gis_osm_roads_free_1.shp",
            "D:/3r_kurs/4-2/map/map/gis_osm_roads_free_1.shp",
        ]
        for path in paths:
            if path and os.path.exists(path):
                return path
        raise FileNotFoundError("Shapefile not found. Set OSM_SHP env var or place in map/ folder.")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("routefinder")

# Initialize FastAPI app
app = FastAPI(
    title="UB Route Finder",
    description="Traffic Flow Visualization with Pathfinding Algorithms",
    version="2.0.0"
)

# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Load shapefile
try:
    SHP_PATH = get_shapefile_path()
    log.info("Loading graph from %s", SHP_PATH)
    G, graph_crs = build_graph_from_shp(SHP_PATH)
    log.info("Graph loaded: nodes=%d edges=%d", 
             len(G.node_coords), sum(len(v) for v in G.adj.values()))
except FileNotFoundError as e:
    log.error(str(e))
    raise

# Create transformer for coordinate conversion
transformer_to_graph = Transformer.from_crs("EPSG:4326", graph_crs.to_string(), always_xy=True)


def nearest_node(graph, x, y):
    """Find nearest graph node to given coordinates."""
    best, best_d = None, float("inf")
    for nid, (nx, ny) in graph.node_coords.items():
        d = (nx - x)**2 + (ny - y)**2
        if d < best_d:
            best, best_d = nid, d
    return best


def apply_traffic_to_graph(graph, traffic_multiplier=2.0):
    """
    Apply traffic flow simulation to graph edges.
    Higher traffic = higher weight (slower travel).
    
    Traffic is simulated based on:
    - Random variation (simulating congestion)
    - Edge weight (longer roads tend to have more traffic)
    """
    import random
    random.seed(42)  # For reproducible traffic patterns
    
    for node in graph.adj:
        for i, (neighbor, weight, meta) in enumerate(graph.adj[node]):
            # Simulate traffic: random multiplier between 1.0 and traffic_multiplier
            traffic_factor = 1.0 + (random.random() * (traffic_multiplier - 1.0))
            new_weight = weight * traffic_factor
            graph.adj[node][i] = (neighbor, new_weight, meta)
    
    log.info(f"Applied traffic simulation with max multiplier {traffic_multiplier}x")


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main HTML page."""
    templates_path = Path(__file__).parent / "templates" / "index.html"
    if not templates_path.exists():
        raise HTTPException(status_code=404, detail="Template not found")
    
    with open(templates_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    # Replace Flask url_for with direct paths
    html_content = html_content.replace("{{ url_for('static', filename='style.css') }}", "/static/style.css")
    
    return HTMLResponse(content=html_content)


@app.get("/route")
async def route(
    src: str = Query(..., description="Source coordinates (lon,lat)"),
    dst: str = Query(..., description="Destination coordinates (lon,lat)"),
    alg: str = Query("dijkstra", description="Algorithm: dijkstra, bfs, or dfs"),
    traffic: str = Query("false", description="Enable traffic simulation: true or false"),
    time_hour: int = Query(None, description="Hour of day for simulation (0-23)", ge=0, le=23)
):
    """Main route finding endpoint - supports BFS, DFS, and Dijkstra."""
    alg = alg.lower()
    use_traffic = traffic.lower() == "true"

    try:
        lon1, lat1 = map(float, src.split(","))
        lon2, lat2 = map(float, dst.split(","))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"bad src/dst format: {e}")

    # Transform coordinates to graph CRS
    try:
        gx1, gy1 = transformer_to_graph.transform(lon1, lat1)
        gx2, gy2 = transformer_to_graph.transform(lon2, lat2)
    except Exception:
        gx1, gy1, gx2, gy2 = lon1, lat1, lon2, lat2

    # Find nearest nodes
    s = nearest_node(G, gx1, gy1)
    t = nearest_node(G, gx2, gy2)
    
    if s is None or t is None:
        raise HTTPException(status_code=400, detail="nearest node not found")

    # Apply traffic if requested
    graph_to_use = G
    if use_traffic:
        graph_to_use = copy.deepcopy(G)
        apply_traffic_to_graph(graph_to_use, traffic_multiplier=2.0)

    try:
        path_nodes = None
        dist = None
        used_alg = ""
        
        # Run selected algorithm with timing
        start_time = time.perf_counter()
        
        if alg == "bfs":
            log.info(f"Running BFS from node {s} to {t}")
            path_nodes = bfs_shortest(graph_to_use, s, t)
            used_alg = "BFS"
            
        elif alg == "dfs":
            log.info(f"Running DFS from node {s} to {t}")
            path_nodes = dfs_path_safe(graph_to_use, s, t, max_nodes=500000, max_depth=10000)
            used_alg = "DFS"
            
        elif alg == "dijkstra":
            log.info(f"Running Dijkstra from node {s} to {t}")
            path_nodes, dist = dijkstra(graph_to_use, s, t)
            used_alg = "Dijkstra"
            
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Unknown algorithm: {alg}. Use 'bfs', 'dfs', or 'dijkstra'"
            )
        
        elapsed_time = time.perf_counter() - start_time
        
        # Check if path was found
        if not path_nodes:
            log.warning(f"No path found using {used_alg}")
            return JSONResponse({
                "algorithm": used_alg,
                "error": "no path found",
                "message": "Could not find route between points",
                "time": round(elapsed_time * 1000, 2),
                "traffic": use_traffic
            })
        
        # Calculate actual distance (use original graph weights, not traffic-modified)
        actual_distance = 0.0
        for i in range(len(path_nodes) - 1):
            node_a, node_b = path_nodes[i], path_nodes[i + 1]
            for neigh, w, _ in G.adj.get(node_a, []):
                if neigh == node_b:
                    actual_distance += w
                    break
        
        # Calculate travel time with traffic (if enabled)
        travel_time = 0.0
        if use_traffic and dist is not None:
            travel_time = dist  # This is already the traffic-weighted distance
        
        # Convert to coordinates
        coords = [G.node_lonlat[n] for n in path_nodes]
        
        # Build response
        result = {
            "algorithm": used_alg,
            "path": coords,
            "steps": len(path_nodes),
            "distance": round(actual_distance, 3),
            "time": round(elapsed_time * 1000, 2),
            "traffic": use_traffic
        }
        
        if use_traffic and travel_time > 0:
            result["travel_time"] = round(travel_time, 3)
        
        log.info(f"{used_alg} found path with {len(path_nodes)} steps, " +
                f"{round(actual_distance, 3)}m distance in {round(elapsed_time * 1000, 2)}ms" +
                (f" (traffic-weighted: {round(travel_time, 3)})" if use_traffic else ""))
        
        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as ex:
        log.exception("Error processing route")
        raise HTTPException(status_code=500, detail=f"processing error: {str(ex)}")


@app.get("/compare")
async def compare(
    src: str = Query(..., description="Source coordinates (lon,lat)"),
    dst: str = Query(..., description="Destination coordinates (lon,lat)"),
    traffic: str = Query("false", description="Enable traffic simulation: true or false"),
    time_hour: int = Query(None, description="Hour of day (0-23)", ge=0, le=23)
):
    """Compare all three algorithms."""
    use_traffic = traffic.lower() == "true"
    
    try:
        lon1, lat1 = map(float, src.split(","))
        lon2, lat2 = map(float, dst.split(","))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"bad src/dst format: {e}")

    # Transform coordinates
    gx1, gy1 = transformer_to_graph.transform(lon1, lat1)
    gx2, gy2 = transformer_to_graph.transform(lon2, lat2)
    s = nearest_node(G, gx1, gy1)
    t = nearest_node(G, gx2, gy2)
    
    if s is None or t is None:
        raise HTTPException(status_code=400, detail="nearest node not found")

    # Apply traffic if requested
    graph_to_use = G
    if use_traffic:
        graph_to_use = copy.deepcopy(G)
        apply_traffic_to_graph(graph_to_use, traffic_multiplier=2.0)

    results = []
    
    # 1. Dijkstra
    try:
        start_time = time.perf_counter()
        path_nodes, dist = dijkstra(graph_to_use, s, t)
        elapsed_time = time.perf_counter() - start_time
        
        if path_nodes:
            # Calculate actual distance from original graph
            actual_distance = 0.0
            for i in range(len(path_nodes) - 1):
                node_a, node_b = path_nodes[i], path_nodes[i + 1]
                for neigh, w, _ in G.adj.get(node_a, []):
                    if neigh == node_b:
                        actual_distance += w
                        break
            
            coords = [G.node_lonlat[n] for n in path_nodes]
            result = {
                "algorithm": "Dijkstra",
                "path": coords,
                "distance": round(actual_distance, 3),
                "steps": len(path_nodes),
                "time": round(elapsed_time * 1000, 2),
                "traffic": use_traffic
            }
            if use_traffic:
                result["travel_time"] = round(dist, 3)
            results.append(result)
        else:
            results.append({
                "algorithm": "Dijkstra", 
                "error": "no path found",
                "time": round(elapsed_time * 1000, 2),
                "traffic": use_traffic
            })
    except Exception as ex:
        log.exception("Error in Dijkstra comparison")
        results.append({"algorithm": "Dijkstra", "error": str(ex)})
    
    # 2. BFS
    try:
        start_time = time.perf_counter()
        path_nodes = bfs_shortest(graph_to_use, s, t)
        elapsed_time = time.perf_counter() - start_time
        
        if path_nodes:
            # Calculate actual distance from original graph
            actual_distance = 0.0
            for i in range(len(path_nodes) - 1):
                node_a, node_b = path_nodes[i], path_nodes[i + 1]
                for neigh, w, _ in G.adj.get(node_a, []):
                    if neigh == node_b:
                        actual_distance += w
                        break
            
            coords = [G.node_lonlat[n] for n in path_nodes]
            results.append({
                "algorithm": "BFS",
                "path": coords,
                "distance": round(actual_distance, 3),
                "steps": len(path_nodes),
                "time": round(elapsed_time * 1000, 2),
                "traffic": use_traffic
            })
        else:
            results.append({
                "algorithm": "BFS", 
                "error": "no path found",
                "time": round(elapsed_time * 1000, 2),
                "traffic": use_traffic
            })
    except Exception as ex:
        log.exception("Error in BFS comparison")
        results.append({"algorithm": "BFS", "error": str(ex)})
    
    # 3. DFS
    try:
        start_time = time.perf_counter()
        path_nodes = dfs_path_safe(graph_to_use, s, t, max_nodes=500000, max_depth=10000)
        elapsed_time = time.perf_counter() - start_time
        
        if path_nodes:
            # Calculate actual distance from original graph
            actual_distance = 0.0
            for i in range(len(path_nodes) - 1):
                node_a, node_b = path_nodes[i], path_nodes[i + 1]
                for neigh, w, _ in G.adj.get(node_a, []):
                    if neigh == node_b:
                        actual_distance += w
                        break
            
            coords = [G.node_lonlat[n] for n in path_nodes]
            results.append({
                "algorithm": "DFS",
                "path": coords,
                "distance": round(actual_distance, 3),
                "steps": len(path_nodes),
                "time": round(elapsed_time * 1000, 2),
                "traffic": use_traffic
            })
        else:
            results.append({
                "algorithm": "DFS", 
                "error": "no path found",
                "time": round(elapsed_time * 1000, 2),
                "traffic": use_traffic
            })
    except Exception as ex:
        log.exception("Error in DFS comparison")
        results.append({"algorithm": "DFS", "error": str(ex)})
    
    return JSONResponse(results)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "nodes": len(G.node_coords),
        "edges": sum(len(v) for v in G.adj.values())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")