from collections import deque
import heapq

def bfs_shortest(graph, start, goal):
    """BFS - finds path with fewest edges."""
    if start == goal:
        return [start]
    
    visited = {start}
    q = deque([(start, [start])])
    
    while q:
        node, path = q.popleft()
        
        for neigh, _, _ in graph.adj.get(node, []):
            if neigh in visited:
                continue
            if neigh == goal:
                return path + [neigh]
            visited.add(neigh)
            q.append((neigh, path + [neigh]))
    
    return None


def dfs_path_safe(graph, start, goal, max_nodes=1000000, max_depth=5000):
    """Iterative DFS with safety limits."""
    if start == goal:
        return [start]
    
    stack = [(start, [start])]
    visited_nodes = 0
    
    while stack:
        node, path = stack.pop()
        visited_nodes += 1
        
        if visited_nodes > max_nodes:
            return None
        
        if len(path) > max_depth:
            continue
        
        if node == goal:
            return path
        
        for neigh, _, _ in graph.adj.get(node, []):
            if neigh in path:
                continue
            stack.append((neigh, path + [neigh]))
    
    return None


def dijkstra(graph, start, goal):
    """Dijkstra's algorithm - finds shortest weighted path."""
    dist = {start: 0.0}
    prev = {}
    pq = [(0.0, start)]
    
    while pq:
        d, node = heapq.heappop(pq)
        
        if d > dist.get(node, float("inf")):
            continue
        
        if node == goal:
            break
        
        for neigh, w, _ in graph.adj.get(node, []):
            nd = d + w
            if nd < dist.get(neigh, float("inf")):
                dist[neigh] = nd
                prev[neigh] = node
                heapq.heappush(pq, (nd, neigh))
    
    if goal not in prev and start != goal:
        return None, float("inf")
    
    path = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = prev.get(cur)
        if cur is None:
            return None, float("inf")
    path.append(start)
    path.reverse()
    
    return path, dist.get(goal, 0.0)


def find_alternative_paths(graph, start, goal, num_alternatives=5, penalty_factor=1.5):
    """
    Find diverse alternative paths using penalty method.
    
    This algorithm finds high-quality alternatives by:
    1. Finding the shortest path
    2. Penalizing edges used in previous paths
    3. Finding new paths that avoid heavily-used edges
    
    Args:
        graph: Graph object
        start: Start node
        goal: Goal node
        num_alternatives: Number of alternative paths to find (default 5)
        penalty_factor: How much to penalize reused edges (default 1.5x)
    
    Returns:
        List of (path_nodes, total_weight) tuples
    """
    if start == goal:
        return [([start], 0.0)]
    
    results = []
    edge_penalties = {}  # (node_a, node_b) -> penalty_multiplier
    
    for attempt in range(num_alternatives * 2):  # Try more attempts to find diverse paths
        # Find shortest path with current penalties
        dist = {start: 0.0}
        prev = {}
        pq = [(0.0, start)]
        
        while pq:
            d, node = heapq.heappop(pq)
            
            if d > dist.get(node, float("inf")):
                continue
            
            if node == goal:
                break
            
            for neigh, w, _ in graph.adj.get(node, []):
                # Apply penalty if this edge was used before
                edge_key = (min(node, neigh), max(node, neigh))
                penalty = edge_penalties.get(edge_key, 1.0)
                penalized_weight = w * penalty
                
                nd = d + penalized_weight
                if nd < dist.get(neigh, float("inf")):
                    dist[neigh] = nd
                    prev[neigh] = node
                    heapq.heappush(pq, (nd, neigh))
        
        # Reconstruct path
        if goal not in prev and start != goal:
            break  # No more paths found
        
        path = []
        cur = goal
        while cur != start:
            path.append(cur)
            cur = prev.get(cur)
            if cur is None:
                break
        
        if cur is None:
            break
            
        path.append(start)
        path.reverse()
        
        # Calculate actual weight (without penalties)
        actual_weight = 0.0
        for i in range(len(path) - 1):
            node_a, node_b = path[i], path[i + 1]
            for neigh, w, _ in graph.adj.get(node_a, []):
                if neigh == node_b:
                    actual_weight += w
                    break
        
        # Check if path is too similar to existing paths
        is_duplicate = False
        for existing_path, _ in results:
            # Calculate similarity (percentage of shared edges)
            shared_edges = 0
            total_edges = min(len(path), len(existing_path)) - 1
            
            if total_edges == 0:
                continue
            
            for i in range(len(path) - 1):
                edge = (path[i], path[i + 1])
                for j in range(len(existing_path) - 1):
                    if edge == (existing_path[j], existing_path[j + 1]):
                        shared_edges += 1
                        break
            
            # If more than 70% edges are shared, consider it duplicate
            if shared_edges / total_edges > 0.7:
                is_duplicate = True
                break
        
        if not is_duplicate:
            results.append((path, actual_weight))
            
            # Stop if we have enough diverse paths
            if len(results) >= num_alternatives:
                break
        
        # Penalize edges in this path for next iteration
        for i in range(len(path) - 1):
            edge_key = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
            current_penalty = edge_penalties.get(edge_key, 1.0)
            edge_penalties[edge_key] = current_penalty * penalty_factor
        
        # Stop if path is getting too long (more than 2.5x shortest)
        if results and actual_weight > results[0][1] * 2.5:
            break
    
    return results