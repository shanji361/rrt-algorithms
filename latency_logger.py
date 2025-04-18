import time
from collections import defaultdict
import csv
import os

latency_stats = defaultdict(list)
# so basically, it tracks how much time each key function takes to run, counts
#  how many times it’s called.
#  u just use track_latency("name") on any function you want to measure 
def track_latency(label):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            latency_stats[label].append(end - start)
            return result
        return wrapper
    return decorator

def write_latency_csv(machine_name, dimension_label, out_dir="."):
    filename = f"latency_{machine_name}_{dimension_label}.csv"
    filepath = os.path.join(out_dir, filename)
    with open(filepath, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["function", "calls", "avg_time_sec", "total_time_sec"])
        for func, times in latency_stats.items():
            total = sum(times)
            count = len(times)
            avg = total / count if count else 0
            writer.writerow([func, count, avg, total])
    print(f"[Latency CSV written to] {filepath}")

def reset_latency_stats():
    latency_stats.clear()

def print_latency_summary():
    print("\n=== Latency Breakdown (per function) ===")
    for label, times in latency_stats.items():
        total = sum(times)
        count = len(times)
        avg = total / count if count else 0
        print(f"{label:25s}: {count:6d} calls | avg: {avg:.6f}s | total: {total:.2f}s")
