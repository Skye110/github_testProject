"""
Pathfinding Algorithms - All find FIRST path (not necessarily shortest)
"""
from collections import deque
import heapq


def bfs_shortest(graph, start, goal):
    """Breadth-First Search - finds FIRST path level by level"""
    if start == goal:
        return [start]
    
    visited = {start}
    queue = deque([(start, [start])])
    
    while queue:
        node, path = queue.popleft()
        
        for neighbor, _, _ in graph.adj.get(node, []):
            if neighbor in visited:
                continue
            
            if neighbor == goal:
                return path + [neighbor]
            
            visited.add(neighbor)
            queue.append((neighbor, path + [neighbor]))
    
    return None


def dfs_path_safe(graph, start, goal, max_depth=5000):
    """Depth-First Search - finds FIRST path quickly (iterative)"""
    if start == goal:
        return [start]
    
    # Use simple iterative DFS with visited set
    stack = [(start, [start])]
    visited = set()
    nodes_explored = 0
    max_nodes = 50000
    
    while stack and nodes_explored < max_nodes:
        node, path = stack.pop()
        
        if node in visited:
            continue
            
        visited.add(node)
        nodes_explored += 1
        
        if len(path) > max_depth:
            continue
        
        if node == goal:
            return path
        
        # Add neighbors to stack
        for neighbor, _, _ in graph.adj.get(node, []):
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))
    
    return None


def dijkstra(graph, start, goal):
    """Dijkstra - finds FIRST path using priority queue (shortest weighted)"""
    if start == goal:
        return [start], 0.0
    
    dist = {start: 0.0}
    prev = {}
    pq = [(0.0, start)]
    visited = set()
    
    while pq:
        current_dist, node = heapq.heappop(pq)
        
        if node in visited:
            continue
        
        visited.add(node)
        
        # Found goal - return immediately
        if node == goal:
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
        
        if current_dist > dist.get(node, float("inf")):
            continue
        
        for neighbor, weight, _ in graph.adj.get(node, []):
            if neighbor in visited:
                continue
            
            new_dist = current_dist + weight
            
            if new_dist < dist.get(neighbor, float("inf")):
                dist[neighbor] = new_dist
                prev[neighbor] = node
                heapq.heappush(pq, (new_dist, neighbor))
    
    return None, float("inf")