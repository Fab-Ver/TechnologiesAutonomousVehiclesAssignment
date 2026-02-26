# Loads a driving network from OpenStreetMap (osmnx), computes edge weights as travel time
# (length / maxspeed), and runs the A* shortest-path algorithm (heapq) between 10 random
# node pairs per city. Tracks iterations per run, computes the average, and saves
# graph visualizations to the plots/ folder. Cities: Aosta, Turin.

import argparse
import json
import math
import os
import osmnx as ox
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
import heapq

NUM_RUNS = 10   # number of random pairs per city

PLACES = {
    "Aosta": "Aosta, Aosta, Italy",
    "Turin": "Turin, Piedmont, Italy",
}

def heuristic_manhattan(node, dest):
    """h(n) = |x1 - x2| + |y1 - y2|  (lat/lon treated as flat coordinates)"""
    x1, y1 = G.nodes[node]["x"], G.nodes[node]["y"]
    x2, y2 = G.nodes[dest]["x"], G.nodes[dest]["y"]
    return abs(x1 - x2) + abs(y1 - y2)

def heuristic_euclidean(node, dest):
    """h(n) = sqrt((x1-x2)^2 + (y1-y2)^2)  (flat-earth approximation)"""
    x1, y1 = G.nodes[node]["x"], G.nodes[node]["y"]
    x2, y2 = G.nodes[dest]["x"], G.nodes[dest]["y"]
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def heuristic_haversine(node, dest):
    """h(n) = great-circle distance via haversine formula (R = 6371 km)"""
    R = 6371  # Earth radius in km
    lat1, lon1 = math.radians(G.nodes[node]["y"]), math.radians(G.nodes[node]["x"])
    lat2, lon2 = math.radians(G.nodes[dest]["y"]), math.radians(G.nodes[dest]["x"])
    dphi = lat1 - lat2
    dlambda = lon1 - lon2
    a = math.sin(dphi/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

HEURISTICS = {
    "manhattan" : heuristic_manhattan,
    "euclidean" : heuristic_euclidean,
    "haversine" : heuristic_haversine,
}

def style_unvisited_edge(edge):        
    G.edges[edge]["color"] = "gray"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 0.2

def style_visited_edge(edge):
    G.edges[edge]["color"] = "green"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 1

def style_active_edge(edge):
    G.edges[edge]["color"] = "red"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 1

def style_path_edge(edge):
    G.edges[edge]["color"] = "white"
    G.edges[edge]["alpha"] = 1
    G.edges[edge]["linewidth"] = 5

def clean_maxspeed():
    for edge in G.edges:
        # Cleaning the "maxspeed" attribute, some values are lists, some are strings, some are None
        maxspeed = 40
        if "maxspeed" in G.edges[edge]:
            maxspeed = G.edges[edge]["maxspeed"]
            if isinstance(maxspeed, list):
                speeds = [int(s) if s != "walk" else 1 for s in maxspeed]
                maxspeed = min(speeds)
            elif isinstance(maxspeed, str):
                if maxspeed == "walk":
                    maxspeed = 1
                else:
                    # take the numeric part only (handles "50", "50 mph", "50 km/h", etc.)
                    maxspeed = int(maxspeed.split()[0])
        G.edges[edge]["maxspeed"] = maxspeed
        # Adding the "weight" attribute (time = distance / speed)
        G.edges[edge]["weight"] = G.edges[edge]["length"] / maxspeed

def plot_graph(filepath=None):
    os.makedirs(os.path.dirname(filepath), exist_ok=True) if filepath else None
    ox.plot_graph(
        G,
        node_size =  [ G.nodes[node]["size"] for node in G.nodes ],
        edge_color = [ G.edges[edge]["color"] for edge in G.edges ],
        edge_alpha = [ G.edges[edge]["alpha"] for edge in G.edges ],
        edge_linewidth = [ G.edges[edge]["linewidth"] for edge in G.edges ],
        node_color = "white",
        bgcolor = "black",
        show = filepath is None,
        save = filepath is not None,
        filepath = filepath,
    )
    plt.close("all")

def plot_heatmap(filepath):
    """Color edges by astar_uses frequency using a heatmap colormap."""
    uses = [G.edges[e]["astar_uses"] for e in G.edges]
    max_uses = max(uses) if max(uses) > 0 else 1
    norm = mcolors.Normalize(vmin=0, vmax=max_uses)
    cmap = plt.get_cmap("plasma")
    edge_colors = []
    edge_widths = []
    for e in G.edges:
        u = G.edges[e]["astar_uses"]
        edge_colors.append(mcolors.to_hex(cmap(norm(u))) if u > 0 else "#1a1a1a")
        edge_widths.append(0.5 + 4.5 * norm(u))  # 0.5 (unused) → 5.0 (max used)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    ox.plot_graph(
        G,
        node_size=0,
        edge_color=edge_colors,
        edge_linewidth=edge_widths,
        edge_alpha=1,
        node_color="white",
        bgcolor="black",
        show=False,
        save=True,
        filepath=filepath,
    )
    plt.close("all")

def astar(orig, dest, heuristic):
    # Reset all node and edge state — ensures no contamination between runs
    for node in G.nodes:
        G.nodes[node]["visited"] = False
        G.nodes[node]["distance"] = float("inf")
        G.nodes[node]["previous"] = None
        G.nodes[node]["size"] = 0
    for edge in G.edges:
        style_unvisited_edge(edge)
    G.nodes[orig]["distance"] = 0
    G.nodes[orig]["size"] = 50
    G.nodes[dest]["size"] = 50
    pq = [(heuristic(orig, dest), orig)]
    step = 0
    while pq:
        _, node = heapq.heappop(pq)
        if node == dest:
            return step
        if G.nodes[node]["visited"]: continue
        G.nodes[node]["visited"] = True
        for edge in G.out_edges(node):
            style_visited_edge((edge[0], edge[1], 0))
            neighbor = edge[1]
            weight = G.edges[(edge[0], edge[1], 0)]["weight"]
            if G.nodes[neighbor]["distance"] > G.nodes[node]["distance"] + weight:
                G.nodes[neighbor]["distance"] = G.nodes[node]["distance"] + weight
                G.nodes[neighbor]["previous"] = node
                priority = G.nodes[neighbor]["distance"] + heuristic(neighbor, dest)
                heapq.heappush(pq, (priority, neighbor))
                for edge2 in G.out_edges(neighbor):
                    style_active_edge((edge2[0], edge2[1], 0))
        step += 1
    return None # No path found

def reconstruct_path(orig, dest, plot=False, algorithm=None, filepath=None):
    for edge in G.edges:
        style_unvisited_edge(edge)
    dist = 0
    curr = dest
    while curr != orig:
        prev = G.nodes[curr]["previous"]
        dist += G.edges[(prev, curr, 0)]["length"]
        style_path_edge((prev, curr, 0))
        if algorithm:
            G.edges[(prev, curr, 0)][f"{algorithm}_uses"] = G.edges[(prev, curr, 0)].get(f"{algorithm}_uses", 0) + 1
        curr = prev
    dist /= 1000
    if plot:
        plot_graph(filepath=filepath)
    return dist

def main():
    global G
    parser = argparse.ArgumentParser(description="Run A* on OSM city graphs.")
    parser.add_argument(
        "-n", "--runs",
        type=int,
        default=NUM_RUNS,
        help=f"Number of random pairs per city (default: {NUM_RUNS})",
    )
    args = parser.parse_args()
    num_runs = args.runs

    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    results_fp = "results/results.json"
    if not os.path.exists(results_fp):
        print(f"ERROR: {results_fp} not found. Run Dijkstra.py first.")
        return
    with open(results_fp) as f:
        all_results = json.load(f)

    for city_name, place_query in PLACES.items():
        city_abbr = city_name.lower()

        # Load fixed pairs generated by Dijkstra
        pairs_fp = f"results/pairs_{city_abbr}.json"
        if not os.path.exists(pairs_fp):
            print(f"ERROR: {pairs_fp} not found. Run Dijkstra.py first.")
            continue
        with open(pairs_fp) as f:
            pairs = json.load(f)

        print(f"\n{'='*55}")
        print(f"  CITY  : {city_name}")
        print(f"  Query : {place_query}")
        print(f"  Loading graph...")
        G = ox.graph_from_place(place_query, network_type="drive")
        clean_maxspeed()
        print(f"  Graph : {len(G.nodes):,} nodes, {len(G.edges):,} edges")
        print(f"  Using {len(pairs)} fixed pairs from Dijkstra run")

        for h_name, h_func in HEURISTICS.items():
            print(f"\n  {'─'*49}")
            print(f"  Heuristic : A* [{h_name}]")
            print(f"  {'─'*49}")

            for edge in G.edges:
                G.edges[edge]["astar_uses"] = 0

            run_results = []
            for i, (start, end) in enumerate(pairs[:num_runs], 1):
                steps = astar(start, end, h_func)
                if steps is None:
                    print(f"    Run {i:2d}/{num_runs} | WARNING: no path found for this pair")
                    run_results.append({"run": i, "iterations": None})
                    continue
                run_results.append({"run": i, "iterations": steps})
                fp = f"plots/astar_{h_name}_run{i:02d}_{city_abbr}.png"
                reconstruct_path(start, end, plot=True, algorithm="astar", filepath=fp)
                print(f"    Run {i:2d}/{num_runs} | Iterations: {steps:6d} | Plot saved: {fp}")

            valid = [r["iterations"] for r in run_results if r["iterations"] is not None]
            avg = sum(valid) / len(valid) if valid else 0
            print(f"\n    Steps per run : {[r['iterations'] for r in run_results]}")
            print(f"    Average steps : {avg:.1f}")

            hm_fp = f"plots/astar_{h_name}_heatmap_{city_abbr}.png"
            plot_heatmap(hm_fp)
            print(f"    Heatmap saved : {hm_fp}")

            # Update results for this city/heuristic
            all_results[city_name]["results"][f"astar_{h_name}"] = {
                "runs": run_results,
                "average": round(avg, 2),
            }

        # Write updated results after each city
        with open(results_fp, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\n  Results saved : {results_fp}")


if __name__ == "__main__":
    main()

