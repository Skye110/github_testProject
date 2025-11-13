from flask import Flask, request, jsonify, render_template
from build_graph import build_graph_from_shp
from algorithms import bfs_shortest, dfs_path_safe, dijkstra
from pyproj import Transformer
import os
import logging
import time

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("routefinder")

app = Flask(__name__, template_folder="templates", static_folder="static")

# Load shapefile
SHP_PATH = os.environ.get("OSM_SHP", "D:/3r_kurs/4-2/map/gis_osm_roads_free_1.shp")
log.info("Main: loading graph from %s", SHP_PATH)
G, graph_crs = build_graph_from_shp(SHP_PATH)
log.info("Main: graph loaded: nodes=%d adj_entries=%d", 
         len(G.node_coords), sum(len(v) for v in G.adj.values()))

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


def apply_traffic_to_graph(graph, traffic_multiplier=1.5):
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


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/route")
def route():
    """Main route finding endpoint - supports BFS, DFS, and Dijkstra."""
    src = request.args.get("src")
    dst = request.args.get("dst")
    alg = (request.args.get("alg") or "dijkstra").lower()
    use_traffic = request.args.get("traffic", "false").lower() == "true"

    if not src or not dst:
        return jsonify(error="src and dst required (lon,lat)"), 400
    
    try:
        lon1, lat1 = map(float, src.split(","))
        lon2, lat2 = map(float, dst.split(","))
    except Exception as e:
        return jsonify(error=f"bad src/dst format: {e}"), 400

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
        return jsonify(error="nearest node not found"), 400

    # Apply traffic if requested
    graph_to_use = G
    if use_traffic:
        import copy
        from build_graph import Graph
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
            path_nodes = dfs_path_safe(graph_to_use, s, t, max_nodes=1000000, max_depth=5000)
            used_alg = "DFS"
            
        elif alg == "dijkstra":
            log.info(f"Running Dijkstra from node {s} to {t}")
            path_nodes, dist = dijkstra(graph_to_use, s, t)
            used_alg = "Dijkstra"
            
        else:
            return jsonify(error=f"Unknown algorithm: {alg}. Use 'bfs', 'dfs', or 'dijkstra'"), 400
        
        elapsed_time = time.perf_counter() - start_time
        
        # Check if path was found
        if not path_nodes:
            log.warning(f"No path found using {used_alg}")
            return jsonify({
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
        
        return jsonify(result)

    except Exception as ex:
        log.exception("Error processing route")
        return jsonify(error=f"processing error: {str(ex)}"), 500


@app.route("/compare")
def compare():
    """Compare all three algorithms."""
    src = request.args.get("src")
    dst = request.args.get("dst")
    use_traffic = request.args.get("traffic", "false").lower() == "true"
    
    if not src or not dst:
        return jsonify(error="src and dst required (lon,lat)"), 400
    
    try:
        lon1, lat1 = map(float, src.split(","))
        lon2, lat2 = map(float, dst.split(","))
    except Exception as e:
        return jsonify(error=f"bad src/dst format: {e}"), 400

    # Transform coordinates
    gx1, gy1 = transformer_to_graph.transform(lon1, lat1)
    gx2, gy2 = transformer_to_graph.transform(lon2, lat2)
    s = nearest_node(G, gx1, gy1)
    t = nearest_node(G, gx2, gy2)
    
    if s is None or t is None:
        return jsonify(error="nearest node not found"), 400

    # Apply traffic if requested
    graph_to_use = G
    if use_traffic:
        import copy
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
        path_nodes = dfs_path_safe(graph_to_use, s, t, max_nodes=1000000, max_depth=5000)
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
    
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True, port=5000, threaded=True)