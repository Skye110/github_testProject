"""
Pathfinding Algorithms - Fixed and Optimized
============================================

Algorithm Characteristics:
- BFS: Finds path with FEWEST EDGES (hops), explores level by level
- DFS: Finds ANY path quickly, may not be optimal
- Dijkstra: Finds SHORTEST WEIGHTED PATH (actual distance)
"""

from collections import deque
import heapq


def bfs_shortest(graph, start, goal):
    """
    Breadth-First Search - Finds path with FEWEST EDGES.
    
    Characteristics:
    - Explores all neighbors at current depth before going deeper
    - Guarantees path with minimum number of hops/edges
    - Does NOT consider edge weights
    - Good for unweighted graphs or finding "simplest" route
    
    Returns: List of node IDs forming the path, or None if no path exists
    """
    if start == goal:
        return [start]
    
    visited = {start}
    queue = deque([(start, [start])])
    
    while queue:
        node, path = queue.popleft()
        
        # Explore all neighbors
        for neighbor, _, _ in graph.adj.get(node, []):
            if neighbor in visited:
                continue
                
            if neighbor == goal:
                return path + [neighbor]
            
            visited.add(neighbor)
            queue.append((neighbor, path + [neighbor]))
    
    return None


def dfs_path_safe(graph, start, goal, max_nodes=500000, max_depth=10000):
    """
    Depth-First Search - Finds ANY path quickly (iterative version).
    
    Characteristics:
    - Explores as far as possible along each branch before backtracking
    - Finds A path but NOT necessarily the best one
    - Fast to find initial path
    - May find very long/inefficient paths
    
    Safety limits:
    - max_nodes: Maximum nodes to visit (prevents infinite loops)
    - max_depth: Maximum path length (prevents extremely long paths)
    
    Returns: List of node IDs forming the path, or None if no path exists
    """
    if start == goal:
        return [start]
    
    # Use iterative DFS with stack to avoid recursion limits
    # Stack contains: (current_node, path_so_far, visited_set)
    stack = [(start, [start], {start})]
    total_visited = 0
    
    while stack:
        node, path, visited_in_path = stack.pop()
        total_visited += 1
        
        # Safety check: too many nodes visited
        if total_visited > max_nodes:
            # Try to return partial path if we got close
            return None
        
        # Safety check: path too deep
        if len(path) > max_depth:
            continue
        
        # Check if we reached goal
        if node == goal:
            return path
        
        # Get neighbors and sort them to make DFS more deterministic
        neighbors = list(graph.adj.get(node, []))
        
        # Add neighbors to stack (in reverse order so first neighbor is explored first)
        for neighbor, _, _ in reversed(neighbors):
            if neighbor not in visited_in_path:
                new_visited = visited_in_path.copy()
                new_visited.add(neighbor)
                stack.append((neighbor, path + [neighbor], new_visited))
    
    return None


def dijkstra(graph, start, goal):
    """
    Dijkstra's Algorithm - Finds SHORTEST WEIGHTED PATH.
    
    Characteristics:
    - Considers edge weights (actual distances)
    - Guarantees optimal path based on total weight/distance
    - More computationally expensive than BFS/DFS
    - Best for finding actual shortest physical distance
    
    Returns: (path, distance) tuple where path is list of nodes and distance is total weight
             Returns (None, inf) if no path exists
    """
    if start == goal:
        return [start], 0.0
    
    # Distance from start to each node
    dist = {start: 0.0}
    # Previous node in optimal path
    prev = {}
    # Priority queue: (distance, node)
    pq = [(0.0, start)]
    # Track visited nodes to avoid reprocessing
    visited = set()
    
    while pq:
        current_dist, node = heapq.heappop(pq)
        
        # Skip if we already found a better path to this node
        if node in visited:
            continue
        
        visited.add(node)
        
        # Check if we reached the goal
        if node == goal:
            break
        
        # Skip if this distance is outdated
        if current_dist > dist.get(node, float("inf")):
            continue
        
        # Explore neighbors
        for neighbor, weight, _ in graph.adj.get(node, []):
            if neighbor in visited:
                continue
            
            new_dist = current_dist + weight
            
            # If we found a shorter path to neighbor
            if new_dist < dist.get(neighbor, float("inf")):
                dist[neighbor] = new_dist
                prev[neighbor] = node
                heapq.heappush(pq, (new_dist, neighbor))
    
    # Reconstruct path from goal to start
    if goal not in prev and start != goal:
        return None, float("inf")
    
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = prev.get(current)
        if current is None:
            return None, float("inf")
    
    path.append(start)
    path.reverse()
    
    return path, dist.get(goal, 0.0)


def dfs_path_bidirectional(graph, start, goal, max_iterations=100000):
    """
    Bidirectional DFS - Search from both start and goal simultaneously.
    This is more likely to find a path than regular DFS in large graphs.
    
    Returns: List of node IDs forming the path, or None if no path exists
    """
    if start == goal:
        return [start]
    
    # Forward search from start
    forward_stack = [(start, [start], {start})]
    forward_visited = {start: [start]}
    
    # Backward search from goal
    backward_stack = [(goal, [goal], {goal})]
    backward_visited = {goal: [goal]}
    
    iterations = 0
    
    while forward_stack and backward_stack and iterations < max_iterations:
        iterations += 1
        
        # Expand forward search
        if forward_stack:
            node, path, visited_in_path = forward_stack.pop()
            
            # Check if we met the backward search
            if node in backward_visited:
                backward_path = backward_visited[node]
                backward_path.reverse()
                return path + backward_path[1:]
            
            for neighbor, _, _ in graph.adj.get(node, []):
                if neighbor not in visited_in_path:
                    new_path = path + [neighbor]
                    new_visited = visited_in_path.copy()
                    new_visited.add(neighbor)
                    forward_stack.append((neighbor, new_path, new_visited))
                    if neighbor not in forward_visited:
                        forward_visited[neighbor] = new_path
        
        # Expand backward search
        if backward_stack:
            node, path, visited_in_path = backward_stack.pop()
            
            # Check if we met the forward search
            if node in forward_visited:
                forward_path = forward_visited[node]
                path.reverse()
                return forward_path + path[1:]
            
            for neighbor, _, _ in graph.adj.get(node, []):
                if neighbor not in visited_in_path:
                    new_path = path + [neighbor]
                    new_visited = visited_in_path.copy()
                    new_visited.add(neighbor)
                    backward_stack.append((neighbor, new_path, new_visited))
                    if neighbor not in backward_visited:
                        backward_visited[neighbor] = new_path
    
    return None


# Export main functions
__all__ = ['bfs_shortest', 'dfs_path_safe', 'dijkstra', 'dfs_path_bidirectional']


if __name__ == "__main__":
    # Test with a simple graph
    from build_graph import Graph
    
    print("=" * 60)
    print("Algorithm Comparison Test")
    print("=" * 60)
    
    # Create a simple test graph
    #     1
    #    /|\
    #   0 | 3---5
    #    \|/
    #     2---4
    
    G = Graph()
    
    # Add edges with weights
    edges = [
        ((0, 0), (1, 1), 1.4),   # 0-1: short
        ((0, 0), (2, 0), 2.0),   # 0-2: medium
        ((1, 1), (2, 0), 1.4),   # 1-2: short
        ((1, 1), (3, 1), 2.0),   # 1-3: medium
        ((2, 0), (3, 1), 1.4),   # 2-3: short
        ((2, 0), (4, 0), 3.0),   # 2-4: long
        ((3, 1), (5, 1), 2.0),   # 3-5: medium
    ]
    
    for a, b, w in edges:
        G.add_edge(a, b, weight=w)
    
    # Test pathfinding from node 0 to node 5
    start = G.get_node((0, 0))
    goal = G.get_node((5, 1))
    
    print(f"\nFinding path from node {start} to node {goal}")
    print("-" * 60)
    
    # Test BFS
    bfs_path = bfs_shortest(G, start, goal)
    print(f"\nBFS (fewest edges):")
    print(f"  Path: {bfs_path}")
    print(f"  Steps: {len(bfs_path) if bfs_path else 0}")
    
    # Test DFS
    dfs_path = dfs_path_safe(G, start, goal)
    print(f"\nDFS (any path):")
    print(f"  Path: {dfs_path}")
    print(f"  Steps: {len(dfs_path) if dfs_path else 0}")
    
    # Test Dijkstra
    dijkstra_path, dist = dijkstra(G, start, goal)
    print(f"\nDijkstra (shortest weighted):")
    print(f"  Path: {dijkstra_path}")
    print(f"  Steps: {len(dijkstra_path) if dijkstra_path else 0}")
    print(f"  Distance: {dist:.2f}")
    
    print("\n" + "=" * 60)
    print("Key Differences:")
    print("-" * 60)
    print("BFS:      Minimizes NUMBER OF EDGES (hops)")
    print("DFS:      Finds ANY path (may be long)")
    print("Dijkstra: Minimizes TOTAL WEIGHT (actual distance)")
    print("=" * 60)