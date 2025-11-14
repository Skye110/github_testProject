"""
Pathfinding Algorithms - Fixed Version
All algorithms find FIRST path, not necessarily optimal
"""

from collections import deque
import heapq


def bfs_shortest(graph, start, goal):
    """BFS - Breadth-First Search"""
    if start == goal:
        return [start]
    
    visited = {start}
    queue = deque([(start, [start])])
    
    while queue:
        node, path = queue.popleft()
        
        for neighbor, _, _ in graph.adj.get(node, []):
            if neighbor in visited:
                continue
            
            # Return first path found
            if neighbor == goal:
                return path + [neighbor]
            
            visited.add(neighbor)
            queue.append((neighbor, path + [neighbor]))
    
    return None


def dfs_path_safe(graph, start, goal, max_nodes=100000):
    """DFS - Depth-First Search (Fixed iterative version)"""
    if start == goal:
        return [start]
    
    # Stack: (node, path)
    stack = [(start, [start])]
    visited = set()
    nodes_checked = 0
    
    while stack and nodes_checked < max_nodes:
        node, path = stack.pop()
        
        if node in visited:
            continue
        
        visited.add(node)
        nodes_checked += 1
        
        # Check goal
        if node == goal:
            return path
        
        # Add neighbors to stack
        neighbors = graph.adj.get(node, [])
        for neighbor, _, _ in neighbors:
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))
    
    return None


def dijkstra(graph, start, goal):
    """Dijkstra - Returns first path found (early termination)"""
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
        
        # Return immediately when goal is reached
        if node == goal:
            path = []
            current = goal
            while current != start:
                path.append(current)
                current = prev[current]
            path.append(start)
            path.reverse()
            return path, dist[goal]
        
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