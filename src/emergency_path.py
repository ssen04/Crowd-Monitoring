# src/emergency_path.py
"""
Emergency Path Finder

Finds the safest path from video edges to surge hotspot.
Uses A* pathfinding on combined density + surge cost map.

INTEGRATES WITH EXISTING MODULES:
  - Uses density_map from DensityEstimator (CSRNet)
  - Uses surge_map from TemporalSurgeDetector
  - Applies YOUR risk formula: alpha * density + beta * surge
    (same as milestone4_split_view.py risk calculation)

Officials can use this path to reach emergency areas with minimal crowd resistance.
"""

import cv2
import numpy as np
from queue import PriorityQueue


class EmergencyPathFinder:
    def __init__(self, grid_resolution=20):
        """
        Parameters:
          grid_resolution: divide frame into NxN grid for pathfinding
                          (smaller = more precise but slower)
        """
        self.grid_resolution = grid_resolution
        # optimal caching for previously computed paths
        self.path_cache = None

    def find_safe_path(self, frame, density_map, surge_map, alpha=0.6, beta=0.4):
        """
        Find safest path from nearest edge to surge hotspot.

        Uses YOUR existing risk calculation logic (alpha * density + beta * surge)

        Parameters:
          frame: the current video frame (H x W x 3)
          density_map: CSRNet predicted crowd density map
          surge_map: Temporal surge map (change in density)
          alpha: weight for density
          beta: weight for surge/motion

        Returns:
          - path: list of (x, y) coordinates
          - path_vis: visualization overlay
        """
        # extract height and width of frame
        H, W = frame.shape[:2]

        # resize density and surge to match frame size
        density_resized = cv2.resize(density_map, (W, H), interpolation=cv2.INTER_LINEAR)
        surge_resized = cv2.resize(surge_map, (W, H), interpolation=cv2.INTER_LINEAR)

        # create crowd mask - only areas with density > threshold are walkable
        # to prevent paths through sky/empty areas
        density_threshold = np.percentile(density_resized, 10)  # Bottom 10% is "empty"
        crowd_mask = (density_resized > density_threshold).astype(np.float32)

        # Use risk formula: cost = alpha * density + beta * surge
        cost_map = alpha * density_resized + beta * surge_resized

        # crowd mask: make non-crowd areas VERY expensive to traverse
        # to force path to stay within crowd boundaries
        cost_map = np.where(crowd_mask > 0, cost_map, cost_map.max() * 100)

        # normalise cost map to 0-1 for consistent pathfinding
        cost_map_norm = cv2.normalize(cost_map, None, 0, 1, cv2.NORM_MINMAX)

        # find surge hotspot (target location)
        surge_max_loc = np.unravel_index(np.argmax(surge_resized), surge_resized.shape)
        target_y, target_x = surge_max_loc

        # downsample cost map to grid for faster pathfinding
        grid_h = self.grid_resolution
        grid_w = self.grid_resolution

        cost_grid = cv2.resize(cost_map_norm, (grid_w, grid_h), interpolation=cv2.INTER_AREA)

        # convert target to grid coordinates
        target_grid_x = int(target_x * grid_w / W)
        target_grid_y = int(target_y * grid_h / H)
        target_grid = (target_grid_x, target_grid_y)

        # find best starting point on edges (lowest cost)
        start_grid = self._find_best_entry_point(cost_grid)

        # run A* pathfinding
        path_grid = self._astar(cost_grid, start_grid, target_grid)

        if not path_grid:
            # no path found, return empty
            return [], frame.copy()

        # convert grid path back to pixel coordinates
        path_pixels = []
        for gx, gy in path_grid:
            px = int(gx * W / grid_w)
            py = int(gy * H / grid_h)
            path_pixels.append((px, py))

        # create visualization
        path_vis = self._visualize_path(frame, path_pixels, (target_x, target_y), cost_map_norm)

        return path_pixels, path_vis

    def _find_best_entry_point(self, cost_grid):
        """
        Find the edge point with lowest cost
        """
        h, w = cost_grid.shape

        best_cost = float('inf')
        best_point = (0, 0)

        # check all 4 edges
        edges = []

        # top edge
        for x in range(w):
            edges.append((x, 0, cost_grid[0, x]))

        # bottom edge
        for x in range(w):
            edges.append((x, h - 1, cost_grid[h - 1, x]))

        # left edge
        for y in range(h):
            edges.append((0, y, cost_grid[y, 0]))

        # right edge
        for y in range(h):
            edges.append((w - 1, y, cost_grid[y, w - 1]))

        # find minimum cost edge point
        for x, y, cost in edges:
            if cost < best_cost:
                best_cost = cost
                best_point = (x, y)

        return best_point

    def _astar(self, cost_grid, start, goal):
        """
        A* pathfinding algorithm
        """
        h, w = cost_grid.shape

        def heuristic(a, b):
            # manhattan distance heuristic for A*
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        def get_neighbors(pos):
            """
            Return 8-connected neighbors within grid boundaries.
            """
            x, y = pos
            neighbors = []
            # 8-directional movement
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h:
                        neighbors.append((nx, ny))
            return neighbors

        # priority queue: (f_score, counter, position)
        counter = 0
        open_set = PriorityQueue()
        open_set.put((0, counter, start))
        counter += 1

        # dictionary for path reconstruction
        came_from = {}
        # cost from start to current node
        g_score = {start: 0}
        # g + heuristic
        f_score = {start: heuristic(start, goal)}
        # track elements in open_set for quick check
        open_set_hash = {start}

        while not open_set.empty():
            # pop node with lowest f_score
            current = open_set.get()[2]
            open_set_hash.remove(current)

            if current == goal:
                # reconstruct path from goal -> start
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            for neighbor in get_neighbors(current):
                # cost to move to neighbor (base cost + terrain cost)
                nx, ny = neighbor
                # get the cost of this neighbor cell from the cost map
                # higher cost = more crowded or high surge area
                terrain_cost = cost_grid[ny, nx]

                # Higher cost = avoid this area - penalize high-cost cells
                # Compute movement cost to move to this neighbor
                #  Base movement cost = 1.0 (distance); add penalty for high cost terrain
                move_cost = 1.0 + terrain_cost * 10.0
                # tentative g_score = cost from start to neighbor through current
                tentative_g = g_score[current] + move_cost

                # if neighbor not visited before OR found a cheaper path to neighbor
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # update path and scores
                    came_from[neighbor] = current
                    # update g_score: cost from start to neighbor
                    g_score[neighbor] = tentative_g
                    # update f_score: g_score + heuristic to goal
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)

                    # if neighbor is not already in the priority queue
                    if neighbor not in open_set_hash:
                        # add neighbor to open set with priority = f_score
                        open_set.put((f_score[neighbor], counter, neighbor))
                        # ensure unique entries in priority queue
                        counter += 1
                        # track that neighbor is in the open set
                        open_set_hash.add(neighbor)

        # if the loop finishes and goal was never reached = no path found = return empty path
        return []

    def _visualize_path(self, frame, path, target, cost_map):
        """
        Create visualization of safe path
        Overlay the safe path on the original frame for visualization.

        Draws:
          - Semi-transparent cost map
          - Green line along path
          - Start point (ENTRY)
          - Target point (SURGE hotspot)
          - Legends
        """
        vis = frame.copy()

        # draw cost map overlay (semi-transparent)
        H, W = frame.shape[:2]
        # convert normalized cost map to 0-255 and apply color map
        cost_overlay = cv2.normalize(cost_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        cost_colored = cv2.applyColorMap(cost_overlay, cv2.COLORMAP_HOT)
        # overlay cost map (30% on top)
        vis = cv2.addWeighted(vis, 0.7, cost_colored, 0.3, 0)

        # draw thick green line along path
        if len(path) > 1:
            for i in range(len(path) - 1):
                pt1 = path[i]
                pt2 = path[i + 1]
                cv2.line(vis, pt1, pt2, (0, 255, 0), 3, cv2.LINE_AA)

        # draw green circles along path every 3rd point to avoid clutter
        for pt in path[::3]:  # draw every 3rd point
            cv2.circle(vis, pt, 3, (0, 255, 0), -1)

        # mark start (entry point)
        if path:
            start = path[0]
            # white border
            cv2.circle(vis, start, 12, (255, 255, 255), 3)
            # green fill
            cv2.circle(vis, start, 8, (0, 255, 0), -1)
            cv2.putText(vis, "ENTRY", (start[0] - 30, start[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # draw target (surge hotspot)
        # red border
        cv2.circle(vis, target, 15, (0, 0, 255), 3)
        # red fill
        cv2.circle(vis, target, 10, (0, 0, 255), -1)
        cv2.putText(vis, "SURGE", (target[0] - 30, target[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # add legend
        cv2.putText(vis, "GREEN PATH = Safe Route", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis, "RED = High Density/Surge", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return vis
