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

# Config
def get_shapefile_path():
    paths = [
        os.environ.get("OSM_SHP"),
        "map/gis_osm_roads_free_1.shp",
        "D:/3r_kurs/4-2/map/map/gis_osm_roads_free_1.shp",
    ]
    for path in paths:
        if path and os.path.exists(path):
            return path
    raise FileNotFoundError("Shapefile not found")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("routefinder")

app = FastAPI(title="UB Route Finder")

# Static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Load graph
try:
    SHP_PATH = get_shapefile_path()
    log.info("Loading: %s", SHP_PATH)
    G, graph_crs = build_graph_from_shp(SHP_PATH)
    log.info("Nodes: %d, Edges: %d", len(G.node_coords), sum(len(v) for v in G.adj.values()))
except Exception as e:
    log.error(str(e))
    raise

transformer_to_graph = Transformer.from_crs("EPSG:4326", graph_crs.to_string(), always_xy=True)


def nearest_node(graph, x, y):
    """Find nearest node"""
    best, best_d = None, float("inf")
    for nid, (nx, ny) in graph.node_coords.items():
        d = (nx - x)**2 + (ny - y)**2
        if d < best_d:
            best, best_d = nid, d
    return best


def apply_traffic(graph, multiplier=2.0):
    """Apply traffic simulation"""
    import random
    random.seed(42)
    for node in graph.adj:
        for i, (neighbor, weight, meta) in enumerate(graph.adj[node]):
            factor = 1.0 + (random.random() * (multiplier - 1.0))
            graph.adj[node][i] = (neighbor, weight * factor, meta)


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve HTML"""
    html_path = Path(__file__).parent / "templates" / "index.html"
    if not html_path.exists():
        raise HTTPException(404, "Template not found")
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content.replace("{{ url_for('static', filename='style.css') }}", "/static/style.css"))


@app.get("/route")
async def route(
    src: str = Query(...),
    dst: str = Query(...),
    alg: str = Query("dijkstra"),
    traffic: str = Query("false")
):
    """Find route"""
    alg = alg.lower()
    use_traffic = traffic.lower() == "true"

    try:
        lon1, lat1 = map(float, src.split(","))
        lon2, lat2 = map(float, dst.split(","))
    except:
        raise HTTPException(400, "Invalid coordinates")

    gx1, gy1 = transformer_to_graph.transform(lon1, lat1)
    gx2, gy2 = transformer_to_graph.transform(lon2, lat2)
    s = nearest_node(G, gx1, gy1)
    t = nearest_node(G, gx2, gy2)
    
    if s is None or t is None:
        raise HTTPException(400, "Node not found")

    graph_to_use = G
    if use_traffic:
        graph_to_use = copy.deepcopy(G)
        apply_traffic(graph_to_use)

    try:
        start_time = time.perf_counter()
        
        if alg == "bfs":
            path_nodes = bfs_shortest(graph_to_use, s, t)
            used_alg = "BFS"
        elif alg == "dfs":
            path_nodes = dfs_path_safe(graph_to_use, s, t)
            used_alg = "DFS"
        elif alg == "dijkstra":
            path_nodes, _ = dijkstra(graph_to_use, s, t)
            used_alg = "Dijkstra"
        else:
            raise HTTPException(400, f"Unknown algorithm: {alg}")
        
        elapsed = (time.perf_counter() - start_time) * 1000
        
        if not path_nodes:
            log.warning(f"No path: {used_alg}")
            return JSONResponse({
                "algorithm": used_alg,
                "error": "no path found",
                "message": "Could not find route between points",
                "time": round(elapsed, 2),
                "traffic": use_traffic
            })
        
        # Calculate distance
        distance = 0.0
        for i in range(len(path_nodes) - 1):
            a, b = path_nodes[i], path_nodes[i + 1]
            for neigh, w, _ in G.adj.get(a, []):
                if neigh == b:
                    distance += w
                    break
        
        coords = [G.node_lonlat[n] for n in path_nodes]
        
        return JSONResponse({
            "algorithm": used_alg,
            "path": coords,
            "steps": len(path_nodes),
            "distance": round(distance, 3),
            "time": round(elapsed, 2),
            "traffic": use_traffic
        })

    except HTTPException:
        raise
    except Exception as ex:
        log.exception("Error")
        raise HTTPException(500, str(ex))


@app.get("/compare")
async def compare(
    src: str = Query(...),
    dst: str = Query(...),
    traffic: str = Query("false")
):
    """Compare all algorithms"""
    use_traffic = traffic.lower() == "true"
    
    try:
        lon1, lat1 = map(float, src.split(","))
        lon2, lat2 = map(float, dst.split(","))
    except:
        raise HTTPException(400, "Invalid coordinates")

    gx1, gy1 = transformer_to_graph.transform(lon1, lat1)
    gx2, gy2 = transformer_to_graph.transform(lon2, lat2)
    s = nearest_node(G, gx1, gy1)
    t = nearest_node(G, gx2, gy2)
    
    if s is None or t is None:
        raise HTTPException(400, "Node not found")

    graph_to_use = G
    if use_traffic:
        graph_to_use = copy.deepcopy(G)
        apply_traffic(graph_to_use)

    results = []
    
    # Dijkstra
    try:
        start_time = time.perf_counter()
        path_nodes, _ = dijkstra(graph_to_use, s, t)
        elapsed = (time.perf_counter() - start_time) * 1000
        
        if path_nodes:
            distance = 0.0
            for i in range(len(path_nodes) - 1):
                a, b = path_nodes[i], path_nodes[i + 1]
                for neigh, w, _ in G.adj.get(a, []):
                    if neigh == b:
                        distance += w
                        break
            coords = [G.node_lonlat[n] for n in path_nodes]
            results.append({
                "algorithm": "Dijkstra",
                "path": coords,
                "distance": round(distance, 3),
                "steps": len(path_nodes),
                "time": round(elapsed, 2),
                "traffic": use_traffic
            })
        else:
            results.append({"algorithm": "Dijkstra", "error": "no path", "time": round(elapsed, 2)})
    except Exception as ex:
        results.append({"algorithm": "Dijkstra", "error": str(ex)})
    
    # BFS
    try:
        start_time = time.perf_counter()
        path_nodes = bfs_shortest(graph_to_use, s, t)
        elapsed = (time.perf_counter() - start_time) * 1000
        
        if path_nodes:
            distance = 0.0
            for i in range(len(path_nodes) - 1):
                a, b = path_nodes[i], path_nodes[i + 1]
                for neigh, w, _ in G.adj.get(a, []):
                    if neigh == b:
                        distance += w
                        break
            coords = [G.node_lonlat[n] for n in path_nodes]
            results.append({
                "algorithm": "BFS",
                "path": coords,
                "distance": round(distance, 3),
                "steps": len(path_nodes),
                "time": round(elapsed, 2),
                "traffic": use_traffic
            })
        else:
            results.append({"algorithm": "BFS", "error": "no path", "time": round(elapsed, 2)})
    except Exception as ex:
        results.append({"algorithm": "BFS", "error": str(ex)})
    
    # DFS
    try:
        start_time = time.perf_counter()
        path_nodes = dfs_path_safe(graph_to_use, s, t)
        elapsed = (time.perf_counter() - start_time) * 1000
        
        if path_nodes:
            distance = 0.0
            for i in range(len(path_nodes) - 1):
                a, b = path_nodes[i], path_nodes[i + 1]
                for neigh, w, _ in G.adj.get(a, []):
                    if neigh == b:
                        distance += w
                        break
            coords = [G.node_lonlat[n] for n in path_nodes]
            results.append({
                "algorithm": "DFS",
                "path": coords,
                "distance": round(distance, 3),
                "steps": len(path_nodes),
                "time": round(elapsed, 2),
                "traffic": use_traffic
            })
        else:
            results.append({"algorithm": "DFS", "error": "no path", "time": round(elapsed, 2)})
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