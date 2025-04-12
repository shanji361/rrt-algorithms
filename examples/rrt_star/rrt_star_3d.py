import numpy as np
import time
from statistics import mean, median, stdev

from rrt_algorithms.rrt.rrt_star import RRTStar
from rrt_algorithms.search_space.search_space import SearchSpace
from rrt_algorithms.utilities.plotting import Plot

# Search Space dimensions
X_dimensions = np.array([(0, 100), (0, 100), (0, 100)])

# Different obstacle configurations - uncomment the one you want to use
# Configuration 1: Original obstacles
obstacles_config1 = np.array([
    (20, 20, 20, 40, 40, 40), 
    (20, 20, 60, 40, 40, 80), 
    (20, 60, 20, 40, 80, 40), 
    (60, 60, 20, 80, 80, 40),
    (60, 20, 20, 80, 40, 40), 
    (60, 20, 60, 80, 40, 80), 
    (20, 60, 60, 40, 80, 80), 
    (60, 60, 60, 80, 80, 80)
])



# Configuration 2: Spiral-like obstacle
obstacles_config2 = np.array([
    (30, 30, 30, 70, 40, 70),  # Bottom piece
    (30, 30, 30, 40, 70, 70),  # Left piece
    (30, 60, 30, 70, 70, 70),  # Top piece
    (60, 40, 30, 70, 60, 70),  # Right piece (with gap)
    (40, 40, 30, 60, 50, 70),  # Inner piece (with gap)
    
    # Additional obstacles
    (10, 10, 70, 20, 90, 90),
    (80, 20, 10, 90, 40, 30)
])



# Configuration 3: Tetris-like shapes
obstacles_config3 = np.array([
    # T-shape
    (20, 10, 20, 50, 20, 30),
    (30, 20, 20, 40, 40, 30),
    
    # Z-shape
    (60, 60, 40, 80, 70, 50),
    (50, 70, 40, 70, 80, 50),
    
    # L-shape
    (20, 60, 60, 30, 70, 80),
    (30, 60, 60, 40, 80, 70),
    
    # Square
    (60, 20, 70, 80, 40, 90),
    
    # Line
    (10, 45, 30, 90, 55, 40)
])



# Configuration 4: Few large obstacles
obstacles_config4 = np.array([
    (30, 30, 30, 70, 70, 70),
    (10, 10, 10, 25, 25, 90),
    (75, 75, 10, 90, 90, 90)
])

# Configuration 5: Many small obstacles
obstacles_config5 = np.array([
    (10, 10, 10, 20, 20, 20),
    (30, 30, 30, 40, 40, 40),
    (50, 50, 50, 60, 60, 60),
    (70, 70, 70, 80, 80, 80),
    (20, 50, 20, 30, 60, 30),
    (50, 20, 50, 60, 30, 60),
    (70, 30, 10, 80, 40, 20),
    (10, 70, 70, 20, 80, 80),
    (30, 10, 70, 40, 20, 80),
    (60, 70, 30, 70, 80, 40),
    (80, 50, 50, 90, 60, 60),
    (40, 80, 80, 50, 90, 90),
    (85, 85, 10, 95, 95, 20),
    (15, 50, 85, 25, 60, 95)
])

# Configuration 6: Pyramid and floating platforms
obstacles_config6 = np.array([
    # Pyramid (layers stacked with decreasing size)
    (20, 20, 10, 80, 80, 20),
    (30, 30, 20, 70, 70, 30),
    (40, 40, 30, 60, 60, 40),
    
    # Floating platforms at different heights
    (10, 60, 50, 30, 80, 55),
    (40, 10, 60, 60, 30, 65),
    (70, 50, 70, 90, 70, 75),
    (30, 70, 80, 50, 90, 85)
])

obstacles_config7 = np.array([
    # Bottom level maze
    (0, 0, 10, 90, 40, 20),
    (0, 60, 10, 40, 100, 20),
    (60, 0, 10, 100, 40, 20),
    
    # Middle level - vertical shaft access only
    (0, 0, 40, 90, 90, 50),
    (0, 0, 40, 40, 40, 50),
    (60, 60, 40, 100, 100, 50),
    
    # Top level maze
    (0, 0, 70, 40, 40, 80),
    (0, 60, 70, 90, 100, 80),
    (60, 0, 70, 100, 40, 80)
])
# Select which obstacle configuration to use
Obstacles = obstacles_config7  # Change this to obstacles_config2, obstacles_config3, etc.

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