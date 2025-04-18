import numpy as np
import time
from statistics import mean

from rrt_algorithms.rrt.rrt_star import RRTStar
from rrt_algorithms.search_space.search_space import SearchSpace
from rrt_algorithms.utilities.plotting import Plot

from latency_logger import track_latency, write_latency_csv, reset_latency_stats, print_latency_summary, latency_stats

X_dimensions = np.array([(0, 100), (0, 100)])

case_1 = np.array([
    (40, 55, 60, 100),
    (40, 0, 60, 45)
])

Obstacles = case_1
x_init = (80, 0)
x_goal = (80, 80)
q = 8
r = 1
max_samples = 1024
rewire_count = 32
prc = 0.1
MACHINE_NAME = "M1"  # change for M2, M3, etc.

def run_rrt_star_once():
    X = SearchSpace(X_dimensions, Obstacles)
    rrt = RRTStar(X, q, x_init, x_goal, max_samples, r, prc, rewire_count)

    # Wrap key methods for latency tracking
    rrt.new_and_near = track_latency("new_and_near")(rrt.new_and_near)
    rrt.get_nearby_vertices = track_latency("get_nearby_vertices")(rrt.get_nearby_vertices)
    rrt.connect_shortest_valid = track_latency("connect_shortest_valid")(rrt.connect_shortest_valid)
    rrt.rewire = track_latency("rewire")(rrt.rewire)
    rrt.X.collision_free = track_latency("collision_free")(rrt.X.collision_free)

    start = time.time()
    path = rrt.rrt_star()
    end = time.time()

    total_execution_time = end - start
    path_len = None
    if path is not None:
        path_len = sum(np.linalg.norm(np.array(path[i]) - np.array(path[i + 1])) for i in range(len(path) - 1))
    return total_execution_time, path_len, path, rrt.trees

execution_times = []
path_lengths = []
successes = 0

# Aggregate all latency stats manually
all_latencies = {}

def accumulate_latency():
    for key, vals in latency_stats.items():
        if key not in all_latencies:
            all_latencies[key] = []
        all_latencies[key].extend(vals)

for i in range(100):
    print(f"Running trial {i + 1}/100", end='\r')
    reset_latency_stats()
    exec_time, path_len, path, trees = run_rrt_star_once()
    execution_times.append(exec_time)
    if path_len:
        path_lengths.append(path_len)
        successes += 1
    accumulate_latency()

# Print combined latency summary
print("\n=== Latency Breakdown (Total over 100 runs) ===")
for label, times in all_latencies.items():
    total = sum(times)
    count = len(times)
    avg = total / count if count else 0
    print(f"{label:25s}: {count:6d} calls | avg: {avg:.6f}s | total: {total:.2f}s")

# Write to final CSV
import csv, os
with open(f"latency_{MACHINE_NAME}_2D.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["function", "calls", "avg_time_sec", "total_time_sec"])
    for label, times in all_latencies.items():
        total = sum(times)
        count = len(times)
        avg = total / count if count else 0
        writer.writerow([label, count, avg, total])

# Print RRT* stats
print("\n== 2D rrt* results ==")
print(f"Success rate: {successes}%")
print(f"Mean exec time: {mean(execution_times):.4f}s")
print(f"Min time: {min(execution_times):.4f}s")
print(f"Max time: {max(execution_times):.4f}s")

if path_lengths:
    print(f"Avg Path Length: {mean(path_lengths):.2f}")
    print(f"Shortest Path : {min(path_lengths):.2f}")
    print(f"Longest Path : {max(path_lengths):.2f}")

X = SearchSpace(X_dimensions, Obstacles)
plot = Plot("rrt_star_2d_final")
plot.plot_tree(X, trees)
if path is not None:
    plot.plot_path(X, path)
plot.plot_obstacles(X, Obstacles)
plot.plot_start(X, x_init)
plot.plot_goal(X, x_goal)
plot.draw(auto_open=True)
