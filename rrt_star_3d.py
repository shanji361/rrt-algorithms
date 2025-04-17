import numpy as np
import time
from statistics import mean, median, stdev

from rrt_algorithms.rrt.rrt_star import RRTStar
from rrt_algorithms.search_space.search_space import SearchSpace
from rrt_algorithms.utilities.plotting import Plot

# Search Space dimensions
X_dimensions = np.array([(0, 100), (0, 100), (0, 100)])


case_1_3d = np.array([
    (0, 60, 0, 70, 100, 100),  
    (30, 0, 0, 100, 40, 100)    
])


case_2_3d = np.array([
    (5, 88, 0, 15, 98, 100),
    (20, 60, 0, 25, 95, 100),
    (75, 55, 0, 80, 95, 100),
    (30, 88, 0, 70, 92, 100),
    (20, 48, 0, 80, 52, 100),
    (30, 10, 0, 70, 14, 100),
    (55, 55, 0, 85, 85, 100),
    (47, 5, 0, 53, 95, 100),
    (80, 5, 0, 90, 15, 100),
    (90, 15, 0, 100, 35, 100)
])

case_3_3d = np.array([
    
    [10.0, 0.0, 0.0, 30.0, 20.0, 5.0],      
    [30.0, 0.0, 5.0, 50.0, 20.0, 10.0],
    [50.0, 0.0, 10.0, 70.0, 20.0, 15.0],
    [70.0, 0.0, 15.0, 90.0, 20.0, 20.0],
    [90.0, 0.0, 20.0, 110.0, 20.0, 25.0],  
    
    [90.0, 20.0, 25.0, 110.0, 40.0, 30.0],  
    [90.0, 40.0, 30.0, 110.0, 60.0, 35.0],
    [90.0, 60.0, 35.0, 110.0, 80.0, 40.0],
    [90.0, 80.0, 40.0, 110.0, 100.0, 45.0],
    [90.0, 100.0, 45.0, 110.0, 120.0, 50.0], 
    
    [70.0, 100.0, 50.0, 90.0, 120.0, 55.0], 
    [50.0, 100.0, 55.0, 70.0, 120.0, 60.0],
    [30.0, 100.0, 60.0, 50.0, 120.0, 65.0],
    [10.0, 100.0, 65.0, 30.0, 120.0, 70.0],
    [-10.0, 100.0, 70.0, 10.0, 120.0, 75.0], 
    
    [-10.0, 80.0, 75.0, 10.0, 100.0, 80.0], 
    [-10.0, 60.0, 80.0, 10.0, 80.0, 85.0],
    [-10.0, 40.0, 85.0, 10.0, 60.0, 90.0],  
    
    [10.0, 40.0, 75.0, 30.0, 60.0, 80.0],
    [30.0, 40.0, 70.0, 50.0, 60.0, 75.0],
    [50.0, 40.0, 65.0, 70.0, 60.0, 70.0],
    [50.0, 60.0, 60.0, 70.0, 80.0, 65.0],
    [30.0, 60.0, 55.0, 50.0, 80.0, 60.0],
    [30.0, 40.0, 50.0, 50.0, 60.0, 55.0],
])

case_4_3d = np.array([
   
    [0, 90, 0, 10, 100, 100],
    [20, 90, 0, 80, 95, 100],
    [0, 70, 0, 5, 90, 100],
    [20, 60, 0, 40, 80, 100],
    [45, 60, 0, 55, 100, 100],
    [60, 60, 0, 80, 80, 100],
    [85, 70, 0, 95, 80, 100],
    [0, 40, 0, 30, 45, 100],
    [45, 35, 0, 100, 45, 100],
    [35, 20, 0, 40, 35, 100],
    [0, 20, 0, 10, 40, 100],
    [20, 10, 0, 40, 15, 100],
    [55, 0, 0, 65, 20, 100],
    [70, 10, 0, 90, 15, 100],
    [90, 20, 0, 100, 30, 100],
])


Obstacles = case_2_3d  

# Start and goal positions
x_init = (0, 0, 0)
x_goal = (100, 100, 100)

# RRT* parameters
q = 8  # length of tree edges
r = 1  # length of smallest edge to check for intersection with obstacles
max_samples = 1024  # max number of samples to take before timing out
rewire_count = 32  # optional, number of nearby branches to rewire
prc = 0.1  # probability of checking for a connection to goal

# Function to run a single RRT* and return execution time and path length
def run_single_rrtstar():
    # Create Search Space
    X = SearchSpace(X_dimensions, Obstacles)
    
    # Create RRT* search
    rrt = RRTStar(X, q, x_init, x_goal, max_samples, r, prc, rewire_count)
    
    # Start timer
    start_time = time.time()
    
    # Run RRT*
    path = rrt.rrt_star()
    
    # End timer
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Calculate path length if path exists
    path_length = None
    if path is not None:
        path_length = 0
        for i in range(len(path) - 1):
            path_length += np.linalg.norm(np.array(path[i]) - np.array(path[i+1]))
    
    return execution_time, path_length, path, rrt.trees

# Run RRT* 100 times and collect stats
print(f"Running RRT* 100 times with selected obstacle configuration...")
execution_times = []
path_lengths = []
success_count = 0

for i in range(100):
    print(f"Run {i+1}/100", end="\r")
    exec_time, path_length, path, trees = run_single_rrtstar()
    execution_times.append(exec_time)
    
    if path_length is not None:
        path_lengths.append(path_length)
        success_count += 1

# Print statistics
print("\nResults:")
print(f"Success rate: {success_count}%")
print(f"Average execution time: {mean(execution_times):.4f} seconds")
print(f"Median execution time: {median(execution_times):.4f} seconds")
print(f"Standard deviation: {stdev(execution_times):.4f} seconds")
print(f"Min execution time: {min(execution_times):.4f} seconds")
print(f"Max execution time: {max(execution_times):.4f} seconds")

if path_lengths:
    print(f"Average path length: {mean(path_lengths):.4f}")
    print(f"Median path length: {median(path_lengths):.4f}")
    print(f"Min path length: {min(path_lengths):.4f}")
    print(f"Max path length: {max(path_lengths):.4f}")

# Plot the final run
X = SearchSpace(X_dimensions, Obstacles)
plot = Plot("rrt_star_3d")
plot.plot_tree(X, trees)
if path is not None:
    plot.plot_path(X, path)
plot.plot_obstacles(X, Obstacles)
plot.plot_start(X, x_init)
plot.plot_goal(X, x_goal)
plot.draw(auto_open=True)