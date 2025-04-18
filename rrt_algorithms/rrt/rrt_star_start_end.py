# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.
from operator import itemgetter
import time
from datetime import datetime

from rrt_algorithms.rrt.heuristics import cost_to_go
from rrt_algorithms.rrt.heuristics import segment_cost, path_cost
from rrt_algorithms.rrt.rrt import RRT


class RRTStar(RRT):
    def __init__(self, X, q, x_init, x_goal, max_samples, r, prc=0.01, rewire_count=None):
        super().__init__(X, q, x_init, x_goal, max_samples, r, prc)
        self.rewire_count = rewire_count if rewire_count is not None else 0

    def get_nearby_vertices(self, tree, x_init, x_new):
        X_near = self.nearby(tree, x_new, self.current_rewire_count(tree))
        L_near = [(path_cost(self.trees[tree].E, x_init, x_near) + segment_cost(x_near, x_new), x_near)
                  for x_near in X_near]
        L_near.sort(key=itemgetter(0))
        return L_near

    def rewire(self, tree, x_new, L_near):
        for _, x_near in L_near:
            curr_cost = path_cost(self.trees[tree].E, self.x_init, x_near)
            tent_cost = path_cost(
                self.trees[tree].E, self.x_init, x_new) + segment_cost(x_new, x_near)
            if tent_cost < curr_cost and self.X.collision_free(x_near, x_new, self.r):
                self.trees[tree].E[x_near] = x_new

    def connect_shortest_valid(self, tree, x_new, L_near):
        for _, x_near in L_near:
            if self.connect_to_point(tree, x_near, x_new):
                break

    def current_rewire_count(self, tree):
        if self.rewire_count is None:
            return self.trees[tree].V_count
        return min(self.trees[tree].V_count, self.rewire_count)

    def rrt_star(self):
        start_time_obj = datetime.now()
        start = time.time()
        print(f"⏱️ RRT* START TIME: {start_time_obj.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

        self.add_vertex(0, self.x_init)
        self.add_edge(0, self.x_init, None)

        while True:
            x_new, x_nearest = self.new_and_near(0, self.q)
            if x_new is None:
                continue

            L_near = self.get_nearby_vertices(0, self.x_init, x_new)
            self.connect_shortest_valid(0, x_new, L_near)

            if x_new in self.trees[0].E:
                self.rewire(0, x_new, L_near)

            solution = self.check_solution()
            if solution[0]:
                end_time_obj = datetime.now()
                end = time.time()
                print(f"⏱️ RRT* END TIME:   {end_time_obj.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
                print(f"⏱️ RRT* Duration:   {end - start:.6f} seconds")
                return solution[1]


# === MAIN BLOCK FOR TESTING ===
if __name__ == "__main__":
    from rrt_algorithms.search_space.search_space import SearchSpace
    import numpy as np

    bounds = np.array([[0, 100], [0, 100]])
    obstacles = None  # or provide your own list of obstacles

    X = SearchSpace(bounds, obstacles)

    Q = [1]
    x_init = (0, 0)
    x_goal = (90, 90)
    max_samples = 500
    r = 1.0

    rrt_star = RRTStar(X, Q, x_init, x_goal, max_samples, r)
    path = rrt_star.rrt_star()

    if path:
        print("🎯 Path found!")
    else:
        print("❌ No path found.")
