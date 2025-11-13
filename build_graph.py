# build_graph.py
from shapely.geometry import LineString
import geopandas as gpd
from collections import defaultdict
from pyproj import Transformer

SNAP_TOLERANCE = 1e-6

def round_coord(coord, tol=SNAP_TOLERANCE):
    """Round coordinate to tolerance for node deduplication."""
    return (round(coord[0]/tol)*tol, round(coord[1]/tol)*tol)

class Graph:
    """
    Graph data structure for road network.
    Uses adjacency list representation with edge weights.
    """
    def __init__(self):
        self.adj = defaultdict(list)  # node_id -> [(neighbor_id, weight, metadata)]
        self.coord_to_node = {}  # rounded_coord -> node_id
        self.node_coords = {}  # node_id -> (x, y) in graph CRS
        self.node_lonlat = {}  # node_id -> (lon, lat) in WGS84
        self.next_node_id = 0

    def get_node(self, coord_graph):
        """Get or create node ID for given coordinates."""
        rc = round_coord(coord_graph)
        if rc in self.coord_to_node:
            return self.coord_to_node[rc]
        nid = self.next_node_id
        self.next_node_id += 1
        self.coord_to_node[rc] = nid
        self.node_coords[nid] = rc
        return nid

    def add_edge(self, a_coord_graph, b_coord_graph, weight=1.0, meta=None, one_way=False):
        """Add edge between two coordinates (creates nodes if needed)."""
        a = self.get_node(a_coord_graph)
        b = self.get_node(b_coord_graph)
        self.adj[a].append((b, float(weight), meta))
        if not one_way:
            self.adj[b].append((a, float(weight), meta))

def build_graph_from_shp(shp_path, target_epsg=3857):
    """
    Build graph from shapefile containing road network.
    
    Args:
        shp_path: Path to .shp file
        target_epsg: Target CRS EPSG code (default 3857 for Web Mercator)
    
    Returns:
        (Graph, CRS): Graph object and its coordinate reference system
    """
    print("📂 Reading shapefile:", shp_path)
    gdf = gpd.read_file(shp_path)
    print("✅ Loaded rows:", len(gdf))

    if gdf.crs is None:
        raise ValueError("Input shapefile has no CRS defined.")

    # Reproject if needed
    if target_epsg and getattr(gdf.crs, "to_epsg", lambda: None)() != target_epsg:
        print(f"🔄 Reprojecting from {gdf.crs} to EPSG:{target_epsg}")
        gdf = gdf.to_crs(epsg=target_epsg)

    graph_crs = gdf.crs
    to_wgs84 = Transformer.from_crs(graph_crs.to_string(), "EPSG:4326", always_xy=True)

    G = Graph()
    processed_segments = 0
    
    # Build graph from geometries
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None:
            continue
        
        # Handle both LineString and MultiLineString
        segments = [geom] if geom.geom_type == "LineString" else list(geom.geoms)
        
        for seg in segments:
            coords = list(seg.coords)
            
            # Create edges between consecutive points
            for i in range(len(coords)-1):
                a = coords[i]
                b = coords[i+1]
                
                # Calculate edge weight (Euclidean distance in graph CRS)
                length = LineString([a, b]).length
                
                # Add edge (bidirectional unless one_way flag is set)
                G.add_edge(
                    (float(a[0]), float(a[1])), 
                    (float(b[0]), float(b[1])), 
                    weight=float(length)
                )
                processed_segments += 1
        
        # Progress indicator for large datasets
        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx + 1}/{len(gdf)} features...")

    # Transform all node coordinates to WGS84 for output
    print("🌍 Converting coordinates to WGS84...")
    for nid, (x, y) in G.node_coords.items():
        lon, lat = to_wgs84.transform(x, y)
        G.node_lonlat[nid] = (float(lon), float(lat))

    print(f"✅ Graph built successfully:")
    print(f"   - Nodes: {len(G.node_coords):,}")
    print(f"   - Edges: {sum(len(v) for v in G.adj.values()):,}")
    print(f"   - Segments processed: {processed_segments:,}")
    
    return G, graph_crs