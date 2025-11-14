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
        raise FileNotFoundError("Shapefile not found")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("routefinder")

app = FastAPI(title="UB Route Finder", version="2.0")

static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

try:
    SHP_PATH = get_shapefile_path()
    log.info("Loading graph from %s", SHP_PATH)
    G, graph_crs = build_graph_from_shp(SHP_PATH)
    log.info("Graph loaded: nodes=%d edges=%d", 
             len(G.node_coords), sum(len(v) for v in G.adj.values()))
except FileNotFoundError as e:
    log.error(str(e))
    raise

transformer_to_graph = Transformer.from_crs("EPSG:4326", graph_crs.to_string(), always_xy=True)


def nearest_node(graph, x, y):
    """Find nearest graph node"""
    best, best_d = None, float("inf")
    for nid, (nx, ny) in graph.node_coords.items():
        d = (nx - x)**2 + (ny - y)**2
        if d < best_d:
            best, best_d = nid, d
    return best


def apply_traffic(graph, multiplier=2.0):
    """Apply traffic simulation to edges"""
    import random
    random.seed(42)
    
    for node in graph.adj:
        for i, (neighbor, weight, meta) in enumerate(graph.adj[node]):
            traffic_factor = 1.0 + (random.random() * (multiplier - 1.0))
            new_weight = weight * traffic_factor
            graph.adj[node][i] = (neighbor, new_weight, meta)


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve main HTML page"""
    templates_path = Path(__file__).parent / "templates" / "index.html"
    if not templates_path.exists():
        raise HTTPException(status_code=404, detail="Template not found")
    
    with open(templates_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    html_content = html_content.replace("{{ url_for('static', filename='style.css') }}", "/static/style.css")
    return HTMLResponse(content=html_content)


@app.get("/route")
async def route(
    src: str = Query(...),
    dst: str = Query(...),
    alg: str = Query("dijkstra"),
    traffic: str = Query("false"),
    time_hour: int = Query(None, ge=0, le=23)
):
    """Main route finding endpoint"""
    alg = alg.lower()
    use_traffic = traffic.lower() == "true"

    try:
        lon1, lat1 = map(float, src.split(","))
        lon2, lat2 = map(float, dst.split(","))
    except:
        raise HTTPException(status_code=400, detail="Invalid coordinates")

    gx1, gy1 = transformer_to_graph.transform(lon1, lat1)
    gx2, gy2 = transformer_to_graph.transform(lon2, lat2)
    s = nearest_node(G, gx1, gy1)
    t = nearest_node(G, gx2, gy2)
    
    if s is None or t is None:
        raise HTTPException(status_code=400, detail="Nearest node not found")

    graph_to_use = G
    if use_traffic:
        graph_to_use = copy.deepcopy(G)
        apply_traffic(graph_to_use, 2.0)

    try:
        start_time = time.perf_counter()
        
        if alg == "bfs":
            path_nodes = bfs_shortest(graph_to_use, s, t)
            used_alg = "BFS"
        elif alg == "dfs":
            path_nodes = dfs_path_safe(graph_to_use, s, t, max_depth=10000)
            used_alg = "DFS"
        elif alg == "dijkstra":
            path_nodes, _ = dijkstra(graph_to_use, s, t)
            used_alg = "Dijkstra"
        else:
            raise HTTPException(status_code=400, detail="Unknown algorithm")
        
        elapsed_time = time.perf_counter() - start_time
        
        if not path_nodes:
            return JSONResponse({
                "algorithm": used_alg,
                "error": "no path found",
                "message": "Could not find route",
                "time": round(elapsed_time * 1000, 2),
                "traffic": use_traffic
            })
        
        # Calculate distance using original graph
        actual_distance = 0.0
        for i in range(len(path_nodes) - 1):
            node_a, node_b = path_nodes[i], path_nodes[i + 1]
            for neigh, w, _ in G.adj.get(node_a, []):
                if neigh == node_b:
                    actual_distance += w
                    break
        
        coords = [G.node_lonlat[n] for n in path_nodes]
        
        result = {
            "algorithm": used_alg,
            "path": coords,
            "steps": len(path_nodes),
            "distance": round(actual_distance, 3),
            "time": round(elapsed_time * 1000, 2),
            "traffic": use_traffic
        }
        
        log.info(f"{used_alg} found path: {len(path_nodes)} steps, {round(actual_distance, 3)}m, {round(elapsed_time * 1000, 2)}ms")
        return JSONResponse(result)

    except Exception as ex:
        log.exception("Error processing route")
        raise HTTPException(status_code=500, detail=str(ex))


@app.get("/compare")
async def compare(
    src: str = Query(...),
    dst: str = Query(...),
    traffic: str = Query("false"),
    time_hour: int = Query(None, ge=0, le=23)
):
    """Compare all algorithms"""
    use_traffic = traffic.lower() == "true"
    
    try:
        lon1, lat1 = map(float, src.split(","))
        lon2, lat2 = map(float, dst.split(","))
    except:
        raise HTTPException(status_code=400, detail="Invalid coordinates")

    gx1, gy1 = transformer_to_graph.transform(lon1, lat1)
    gx2, gy2 = transformer_to_graph.transform(lon2, lat2)
    s = nearest_node(G, gx1, gy1)
    t = nearest_node(G, gx2, gy2)
    
    if s is None or t is None:
        raise HTTPException(status_code=400, detail="Nearest node not found")

    graph_to_use = G
    if use_traffic:
        graph_to_use = copy.deepcopy(G)
        apply_traffic(graph_to_use, 2.0)

    results = []
    
    # Dijkstra
    try:
        start_time = time.perf_counter()
        path_nodes, _ = dijkstra(graph_to_use, s, t)
        elapsed_time = time.perf_counter() - start_time
        
        if path_nodes:
            actual_distance = 0.0
            for i in range(len(path_nodes) - 1):
                node_a, node_b = path_nodes[i], path_nodes[i + 1]
                for neigh, w, _ in G.adj.get(node_a, []):
                    if neigh == node_b:
                        actual_distance += w
                        break
            
            coords = [G.node_lonlat[n] for n in path_nodes]
            results.append({
                "algorithm": "Dijkstra",
                "path": coords,
                "distance": round(actual_distance, 3),
                "steps": len(path_nodes),
                "time": round(elapsed_time * 1000, 2),
                "traffic": use_traffic
            })
        else:
            results.append({"algorithm": "Dijkstra", "error": "no path"})
    except Exception as ex:
        results.append({"algorithm": "Dijkstra", "error": str(ex)})
    
    # BFS
    try:
        start_time = time.perf_counter()
        path_nodes = bfs_shortest(graph_to_use, s, t)
        elapsed_time = time.perf_counter() - start_time
        
        if path_nodes:
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
            results.append({"algorithm": "BFS", "error": "no path"})
    except Exception as ex:
        results.append({"algorithm": "BFS", "error": str(ex)})
    
    # DFS
    try:
        start_time = time.perf_counter()
        path_nodes = dfs_path_safe(graph_to_use, s, t, max_depth=10000)
        elapsed_time = time.perf_counter() - start_time
        
        if path_nodes:
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
            results.append({"algorithm": "DFS", "error": "no path"})
    except Exception as ex:
        results.append({"algorithm": "DFS", "error": str(ex)})
    
    return JSONResponse(results)


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "nodes": len(G.node_coords),
        "edges": sum(len(v) for v in G.adj.values())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")