#!/usr/bin/env python3
"""
Integrated TSP Graph Pruning + LKH Evaluation Pipeline.
- Computes metadata for all instances.
- Runs LKH only for instances within a specified node range.
- Produces per‑mode summary CSVs and detailed LKH result CSVs.
- No persistent heavy files (sparse.tsp etc.) are kept.
"""

import os
import sys
import json
import argparse
import math
import random
import time
import shutil
import tempfile
import subprocess
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import KDTree, Delaunay, QhullError

# ------------------- PyTorch / PyG -------------------
import torch
from torch_geometric.data import Data


# Remove the dummy class definition and import the real one
from newTrain import EdgeGNNWithFeatureSelection, parse_opt_tour, pairwise_distances, set_seed

# ------------------- User-defined model class (replace with your actual implementation) -------------------
# Make sure this class SIGNS EXACTLY with the one used to save the checkpoints.
# class EdgeGNNWithFeatureSelection(torch.nn.Module):
#     def __init__(self, in_node_feats, in_edge_feats, node_feat_indices, edge_feat_indices, hidden_dim, n_layers):
#         super().__init__()
#         self.node_feat_indices = node_feat_indices
#         self.edge_feat_indices = edge_feat_indices
#         inp_node = len(node_feat_indices) if node_feat_indices else in_node_feats
#         inp_edge = len(edge_feat_indices) if edge_feat_indices else in_edge_feats

#         # Minimal GNN – replace with your actual architecture
#         self.node_embed = torch.nn.Linear(inp_node, hidden_dim)
#         self.edge_embed = torch.nn.Linear(inp_edge, hidden_dim)
#         self.convs = torch.nn.ModuleList([
#             torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)
#         ])
#         self.out = torch.nn.Linear(hidden_dim, 1)

#     def forward(self, x, edge_index, edge_attr):
#         # Feature selection (matching the original logic)
#         node_features = x[:, self.node_feat_indices] if self.node_feat_indices else x
#         edge_features = edge_attr[:, self.edge_feat_indices] if self.edge_feat_indices else edge_attr

#         h_node = self.node_embed(node_features)
#         h_edge = self.edge_embed(edge_features)

#         # dummy message passing – replace with your real message‑passing layers
#         for conv in self.convs:
#             h_node = torch.relu(conv(h_node))

#         # Compute edge scores as sigmoid of (node_u + node_v + edge) -> linear
#         src, dst = edge_index
#         h_edge_out = h_node[src] + h_node[dst] + h_edge
#         return torch.sigmoid(self.out(h_edge_out)).squeeze(-1)

# ------------------------------------------------------------------
# 1. TSP parsing & distance matrix
# ------------------------------------------------------------------

# Cascade stage-1 fixed k value (matches training)
CASCADE_K1 = 10


def parse_tsp_fixed(filepath):
    """Parse TSPLIB file, return coordinates and distance matrix (Euclidean)."""
    name = os.path.basename(filepath)
    n = None
    coords = {}
    reading_coords = False

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            up = line.upper()
            if up.startswith('DIMENSION'):
                n = int(line.split(':')[1].strip())
            elif up.startswith('NODE_COORD_SECTION'):
                reading_coords = True
            elif reading_coords and line[0].isdigit():
                parts = line.split()
                if len(parts) >= 3:
                    idx = int(parts[0]) - 1
                    x = float(parts[1])
                    y = float(parts[2])
                    coords[idx] = (x, y)
            elif up.startswith('EOF'):
                break

    if not coords or n is None:
        return None

    coords_arr = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        coords_arr[i] = coords[i]

    D = pairwise_distances(coords_arr)
    return {'name': name, 'n': n, 'coords': coords_arr, 'D': D}

def pairwise_distances(coords):
    coords = np.asarray(coords, dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))

# ------------------------------------------------------------------
# 2. Candidate edge building
# ------------------------------------------------------------------
def build_candidate_edges(coords, full_threshold=300, knn_k=25):
    n = len(coords)
    D = pairwise_distances(coords)
    if n <= full_threshold:
        edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    else:
        tree = KDTree(coords)
        edges_set = set()
        for i in range(n):
            _, idxs = tree.query(coords[i], k=min(knn_k+1, n))
            for j in idxs:
                if i != j:
                    a, b = (i, j) if i < j else (j, i)
                    edges_set.add((a, b))
        edges = sorted(edges_set)
    return edges, D

# ------------------------------------------------------------------
# 3. Feature computation (GNN)
# ------------------------------------------------------------------
def compute_extended_features(coords, edges, D, knn_feat_k=10):
    n = coords.shape[0]
    coords = np.asarray(coords, dtype=np.float32)
    mean_d = D[np.triu_indices(n, 1)].mean()
    bbox_w = np.ptp(coords[:, 0]) + 1e-6
    bbox_h = np.ptp(coords[:, 1]) + 1e-6

    tree = KDTree(coords)
    avg_knn = np.zeros(n)
    for i in range(n):
        k = min(knn_feat_k + 1, n)
        dists, idxs = tree.query(coords[i], k=k)
        avg_knn[i] = np.mean([d for d, j in zip(dists, idxs) if j != i])

    node_feat = np.stack([
        coords[:, 0] / bbox_w,
        coords[:, 1] / bbox_h,
        avg_knn / (mean_d + 1e-12)
    ], axis=1)

    sorted_idx = np.argsort(D, axis=1)
    rank = np.zeros_like(sorted_idx)
    for i in range(n):
        rank[i, sorted_idx[i]] = np.arange(n)

    edge_index = [[], []]
    edge_attr = []
    for i, j in edges:
        edge_index[0].append(i)
        edge_index[1].append(j)
        edge_attr.append([
            D[i, j] / (mean_d + 1e-12),
            abs(coords[i, 0] - coords[j, 0]) / bbox_w,
            abs(coords[i, 1] - coords[j, 1]) / bbox_h,
            rank[i, j] / n,
            rank[j, i] / n
        ])
    return (node_feat,
            np.array(edge_index, dtype=np.int64),
            np.array(edge_attr, dtype=np.float32))

def build_pyg_data_for_pruning(tsp_path, knn_k=25, knn_feat_k=10, full_threshold=300):
    parsed = parse_tsp_fixed(tsp_path)
    if parsed is None:
        return None, None, None, None
    coords = parsed['coords']
    n = parsed['n']
    edges, D = build_candidate_edges(coords, full_threshold, knn_k)
    node_feat, edge_index_np, edge_attr = compute_extended_features(
        coords, edges, D, knn_feat_k
    )
    data = Data(
        x=torch.tensor(node_feat, dtype=torch.float),
        edge_index=torch.tensor(edge_index_np, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
    )
    edge_mapping = {i: (edges[i][0], edges[i][1]) for i in range(len(edges))}
    return data, coords, edge_mapping, D

# ------------------------------------------------------------------
# 4. Model loading
# ------------------------------------------------------------------
def load_model_from_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = checkpoint['model_config']
    model = EdgeGNNWithFeatureSelection(
        in_node_feats=3,
        in_edge_feats=5,
        node_feat_indices=model_cfg['node_feats'],
        edge_feat_indices=model_cfg['edge_feats'],
        hidden_dim=model_cfg['hidden_dim'],
        n_layers=model_cfg['n_layers']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

# ------------------------------------------------------------------
# 5. Pruning functions
# ------------------------------------------------------------------
def topk_prune_per_node(edge_index, scores, k, n_nodes):
    src, dst = edge_index
    kept_edges = set()
    for node in range(n_nodes):
        mask = (src == node) | (dst == node)
        idx = mask.nonzero(as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        node_scores = scores[idx]
        topk_count = min(k, idx.numel())
        topk_idx = torch.topk(node_scores, topk_count).indices
        for i in topk_idx:
            kept_edges.add(idx[i].item())
    return sorted(kept_edges)

def prune_nearest_k(D, candidate_edges, edge_to_idx, k):
    n = D.shape[0]
    adj = {i: [] for i in range(n)}
    for (u, v) in candidate_edges:
        adj[u].append((v, D[u, v]))
        adj[v].append((u, D[u, v]))
    kept = set()
    for i in range(n):
        neighbours = sorted(adj[i], key=lambda x: x[1])[:k]
        for (j, _) in neighbours:
            kept.add(edge_to_idx[(min(i, j), max(i, j))])
    return sorted(kept)

def prune_random_k(n_nodes, candidate_edges, edge_to_idx, k, seed=None):
    if seed is not None:
        random.seed(seed)
    adj = {i: [] for i in range(n_nodes)}
    for (u, v) in candidate_edges:
        adj[u].append(v)
        adj[v].append(u)
    kept = set()
    for i in range(n_nodes):
        if len(adj[i]) <= k:
            choices = adj[i]
        else:
            choices = random.sample(adj[i], k)
        for j in choices:
            kept.add(edge_to_idx[(min(i, j), max(i, j))])
    return sorted(kept)

def prune_delaunay_k(coords, D, candidate_edges, edge_to_idx, k):
    n = len(coords)
    try:
        tri = Delaunay(coords)
        delaunay_edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                u, v = simplex[i], simplex[(i+1)%3]
                delaunay_edges.add((min(u, v), max(u, v)))
        candidate_set = set(candidate_edges)
        adj = {i: [] for i in range(n)}
        for (u, v) in delaunay_edges:
            if (u, v) in candidate_set:
                adj[u].append((v, D[u, v]))
                adj[v].append((u, D[u, v]))
    except QhullError:
        # Fallback to nearest_k
        return prune_nearest_k(D, candidate_edges, edge_to_idx, k)

    kept = set()
    for i in range(n):
        neighbours = sorted(adj[i], key=lambda x: x[1])[:k]
        for (j, _) in neighbours:
            kept.add(edge_to_idx[(min(i, j), max(i, j))])
    return sorted(kept)

# ------------------------------------------------------------------
# 6. Metadata functions
# ------------------------------------------------------------------
def compute_degree_stats(kept_edges, edge_mapping, n_nodes):
    degree = np.zeros(n_nodes, dtype=int)
    for e in kept_edges:
        u, v = edge_mapping[e]
        degree[u] += 1
        degree[v] += 1
    return {
        "avg_degree": float(degree.mean()),
        "min_degree": int(degree.min()),
        "max_degree": int(degree.max())
    }

def compute_connectivity(kept_edges, edge_mapping, n_nodes):
    parent = list(range(n_nodes))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    for e in kept_edges:
        u, v = edge_mapping[e]
        union(u, v)
    roots = set(find(i) for i in range(n_nodes))
    return {"is_connected": len(roots) == 1, "num_connected_components": len(roots)}

def compute_knn_overlap(coords, kept_edges, edge_mapping, knn_k=10):
    n = len(coords)
    tree = KDTree(coords)
    knn_sets = []
    for i in range(n):
        _, idx = tree.query(coords[i], k=min(knn_k+1, n))
        knn_sets.append(set(j for j in idx if j != i))
    overlap = []
    for i in range(n):
        neighbors = set()
        for e in kept_edges:
            u, v = edge_mapping[e]
            if u == i: neighbors.add(v)
            elif v == i: neighbors.add(u)
        if knn_sets[i]:
            overlap.append(len(neighbors & knn_sets[i]) / len(knn_sets[i]))
    return {"avg_knn_overlap": float(np.mean(overlap)) if overlap else 0.0}

def compute_edge_length_stats(kept_edges, edge_mapping, D):
    full_lengths = D[np.triu_indices(D.shape[0], 1)]
    pruned_lengths = [D[u, v] for e in kept_edges for u, v in [edge_mapping[e]]]
    pruned_lengths = np.array(pruned_lengths)
    return {
        "mean_edge_length_full": float(full_lengths.mean()),
        "mean_edge_length_pruned": float(pruned_lengths.mean()) if len(pruned_lengths) else 0.0,
        "length_reduction_ratio": float(pruned_lengths.mean() / (full_lengths.mean() + 1e-12)) if len(pruned_lengths) else 0.0
    }

def compute_tour_metrics(kept_edges, edge_mapping, opt_tour, n_nodes):
    kept_set = set()
    for e in kept_edges:
        u, v = edge_mapping[e]
        kept_set.add((min(u, v), max(u, v)))
    opt_edge_set = set()
    for i in range(len(opt_tour)):
        u = opt_tour[i]
        v = opt_tour[(i + 1) % len(opt_tour)]
        opt_edge_set.add((min(u, v), max(u, v)))
    kept_opt = kept_set & opt_edge_set
    return {
        "tour_edge_recall": len(kept_opt) / len(opt_edge_set),
        "tour_edges_kept": len(kept_opt),
        "tour_edges_total": len(opt_edge_set),
        "tour_broken": len(kept_opt) < len(opt_edge_set)
    }

def parse_opt_tour(filepath):
    if not os.path.exists(filepath):
        return None
    tour = []
    with open(filepath, 'r') as f:
        began = False
        for line in f:
            line = line.strip()
            if line.upper().startswith('TOUR_SECTION'):
                began = True
                continue
            if not began:
                continue
            if line == '-1' or line == 'EOF':
                break
            for num in line.split():
                tour.append(int(num)-1)
    return tour if tour else None

# ------------------------------------------------------------------
# 7. Temporary file helpers & LKH interface
# ------------------------------------------------------------------
def tour_length(tour, D):
    if tour is None:
        return float('inf')
    return float(sum(D[tour[i], tour[(i+1)%len(tour)]] for i in range(len(tour))))

def solve_tsp_lkh(par_file, seed=None):
    """Run LKH, parse tour file. Returns (tour, runtime)."""
    # The .par file already contains OUTPUT_TOUR_FILE.
    try:
        with open(par_file, 'r') as f:
            lines = f.readlines()
    except Exception:
        return None, 0.0

    # Inject seed if given
    if seed is not None:
        tmp_dir = tempfile.mkdtemp(prefix="lkh_seed_")
        tmp_par = os.path.join(tmp_dir, "run.par")
        with open(tmp_par, 'w') as f:
            has_seed = False
            for line in lines:
                if line.strip().upper().startswith("SEED"):
                    f.write(f"SEED = {seed}\n")
                    has_seed = True
                else:
                    f.write(line)
            if not has_seed:
                f.write(f"SEED = {seed}\n")
        par_file = tmp_par
    else:
        tmp_dir = None

    # Read output tour path from the par file we are actually using
    sol_file = None
    with open(par_file, 'r') as f:
        for line in f:
            if line.strip().startswith("OUTPUT_TOUR_FILE"):
                sol_file = line.split("=")[1].strip()
    if not sol_file:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return None, 0.0

    sol_file = os.path.abspath(sol_file)
    t0 = time.perf_counter()
    try:
        subprocess.run([LKH_EXE, par_file], capture_output=True, text=True, timeout=TIME_LIMIT)
        runtime = time.perf_counter() - t0
    except subprocess.TimeoutExpired:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return None, time.perf_counter() - t0

    if tmp_dir:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not os.path.exists(sol_file):
        return None, runtime

    # Parse tour
    tour = []
    with open(sol_file, 'r') as f:
        reading = False
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.upper() == "TOUR_SECTION":
                reading = True
                continue
            if line in ("-1", "EOF"):
                break
            if reading:
                try:
                    tour.append(int(line) - 1)
                except:
                    pass
    if not tour:
        return None, runtime
    return tour, runtime

# ------------------------------------------------------------------
# 8. Main pipeline
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Integrated Prune + LKH pipeline")
    parser.add_argument('--tsp_dir', nargs='+', default=['graphss_chosen'],
                        help='Directories with .tsp files')
    parser.add_argument('--output_dir', default='pipeline_output',
                        help='Root output directory')
    parser.add_argument('--gnn_modes', nargs='+', default=[],
                        help='List of "ModeName:checkpoint.pt" pairs for GNN modes')
    parser.add_argument('--baseline_modes', nargs='+', default=['nearest_k', 'random_k', 'delaunay_k'],
                        help='Baseline pruning methods')
    parser.add_argument('--k_values_gnn', type=int, nargs='+',
                        default=[2,3,5,6,7,9,12,15,17,20,22,25],
                        help='k values for GNN modes')
    parser.add_argument('--k_values_baseline', type=int, nargs='+',
                        default=[2,3,5,6,7,9,12,15,17,20,22,25],
                        help='k values for baseline modes')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1],
                        help='Random seeds for LKH')
    parser.add_argument('--lkh_range', type=int, nargs=2, default=[200, 1100],
                        help='Node count range for LKH execution (min max)')
    parser.add_argument('--time_limit', type=int, default=3600,
                        help='LKH time limit per run (seconds)')
    parser.add_argument('--full_threshold', type=int, default=300,
                        help='Graphs with n <= full_threshold use full matrix, else KNN')
    parser.add_argument('--knn_k', type=int, default=25,
                        help='k for building candidate KNN graph')
    parser.add_argument('--knn_feat_k', type=int, default=10,
                        help='k for feature computation (GNN)')
    parser.add_argument('--opt_tour_dir', type=str, default='tsplib_data',
                        help='Directory containing .opt.tour files')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (per instance)')
    parser.add_argument('--lkh_exe', type=str, default=r'C:/LKH/LKH-3.exe',
                        help='Path to LKH executable')
    args = parser.parse_args()

    global LKH_EXE, TIME_LIMIT
    LKH_EXE = args.lkh_exe
    TIME_LIMIT = args.time_limit

    gnn_modes = []
    for item in args.gnn_modes:
        if ':' in item:
            parts = item.split(':', 1)
            name = parts[0]
            ckpt = parts[1]
            is_cascade = 'ModeD' in name          # adjust if other cascade modes exist
            stage2_ckpt = None
            if is_cascade:
                # First look next to the base checkpoint
                stage2_ckpt = os.path.join(os.path.dirname(ckpt), "cascade_stage2_model.pt")
                if not os.path.exists(stage2_ckpt):
                    # Fallback: look one level up (e.g., trainResults/)
                    stage2_ckpt = os.path.join(os.path.dirname(os.path.dirname(ckpt)),
                                           "cascade_stage2_model.pt")
                if not os.path.exists(stage2_ckpt):
                    print(f"❌ Stage2 model not found for cascade mode {name}, skipping cascade logic")
                    is_cascade = False
                    stage2_ckpt = None
                else:
                    print(f"   Cascade stage2: {stage2_ckpt}")
            gnn_modes.append((name, ckpt, is_cascade, stage2_ckpt))
        else:
            print(f"Invalid GNN mode format: {item} (use ModeName:checkpoint.pt)")

    # Baseline mode names
    baseline_modes = args.baseline_modes

    if not gnn_modes and not baseline_modes:
        print("No modes specified. Exiting.")
        return

    # Collect all unique TSP instances
    tsp_files = {}  # instance_name -> filepath
    for d in args.tsp_dir:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.endswith('.tsp'):
                name = os.path.splitext(f)[0]
                if name not in tsp_files:
                    tsp_files[name] = os.path.join(d, f)

    instances = list(tsp_files.keys())
    print(f"Found {len(instances)} unique instances.")

    # Setup output directories
    os.makedirs(args.output_dir, exist_ok=True)

    # Device for GNN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Cache for full graph results per instance and seed
    full_cache = {}  # inst_name -> seed -> (cost_full, runtime_full)

    # Process each mode
    for mode_source in [("gnn", gnn_modes), ("baseline", baseline_modes)]:
        mode_type, mode_list = mode_source
        for mode_info in mode_list:
            if mode_type == "gnn":
                mode_name, checkpoint_path, is_cascade, stage2_ckpt = mode_info
                k_values = args.k_values_gnn
                print(f"\nLoading GNN model {mode_name} from {checkpoint_path}")
                model = load_model_from_checkpoint(checkpoint_path, device)
                if is_cascade:
                    print(f"Loading cascade stage-2 model from {stage2_ckpt}")
                    stage2_model = load_model_from_checkpoint(stage2_ckpt, device)
                else:
                    stage2_model = None
            else:  # baseline
                mode_name = mode_info
                checkpoint_path = None
                k_values = args.k_values_baseline
                model = None

            mode_output_dir = os.path.join(args.output_dir, mode_name)
            os.makedirs(mode_output_dir, exist_ok=True)

            # Containers for CSVs
            mode_lkh_rows = []  # LKH detailed results
            # For each k, we collect metadata per instance
            k_metadata = {k: [] for k in k_values}  # list of metadata dicts

            print(f"Processing mode: {mode_name}")

            for inst_name in tqdm(instances, desc=mode_name):
                inst_path = tsp_files[inst_name]
                try:
                    tqdm.write(f"\n--- Processing instance: {inst_name} ---")
                    # 1. Parse and build candidate graph
                    data, coords, edge_mapping, D = build_pyg_data_for_pruning(
                        inst_path, knn_k=args.knn_k, knn_feat_k=args.knn_feat_k,
                        full_threshold=args.full_threshold
                    )
                    if data is None:
                        tqdm.write(f"  ⚠️ Could not parse {inst_name}, skipping")
                        continue
                    n_nodes = data.x.size(0)
                    candidate_edges = list(edge_mapping.values())
                    edge_to_idx = {(min(u,v), max(u,v)): i for i, (u,v) in enumerate(candidate_edges)}
                    original_edges = len(candidate_edges)

                    # 2. Load optimal tour (if available)
                    opt_tour = None
                    if args.opt_tour_dir:
                        opt_path = os.path.join(args.opt_tour_dir, f"{inst_name}.opt.tour")
                        opt_tour = parse_opt_tour(opt_path)

                    # 3. Determine if we should run LKH for this instance
                    is_synthetic = inst_name.lower().startswith('synthetic')
                    run_lkh = (args.lkh_range[0] <= n_nodes <= args.lkh_range[1]) and not is_synthetic

                    tqdm.write(f"  Nodes: {n_nodes} | Edges (candidate): {original_edges} | LKH: {'YES' if run_lkh else 'NO'}")
                    if run_lkh:
                        tqdm.write(f"     (k-values = {k_values}, seeds = {args.seeds})")

                    for k in k_values:
                        tqdm.write(f"  → k = {k}")
                        # 3a. Prune
                        if mode_type == "gnn":
                            data_device = data.to(device)
                            if is_cascade:
                                # ----- Stage 1: prune to fixed CASCADE_K1 -----
                                with torch.no_grad():
                                    scores1 = model(data_device.x, data_device.edge_index, data_device.edge_attr)
                                kept_stage1 = topk_prune_per_node(data_device.edge_index, scores1, CASCADE_K1, n_nodes)
                                keep_mask = torch.zeros(data_device.edge_index.size(1), dtype=torch.bool, device=device)
                                keep_mask[kept_stage1] = True

                                # Build stage-2 graph
                                stage2_edge_index = data_device.edge_index[:, keep_mask]
                                stage2_edge_attr  = data_device.edge_attr[keep_mask]

                                # ----- Stage 2: score and prune to final k -----
                                with torch.no_grad():
                                    scores2 = stage2_model(data_device.x, stage2_edge_index, stage2_edge_attr)
                                kept_stage2 = topk_prune_per_node(stage2_edge_index, scores2, k, n_nodes)

                                # Map stage-2 indices back to original edge indices
                                stage1_global_positions = torch.where(keep_mask)[0]
                                kept_edges = stage1_global_positions[kept_stage2].tolist()
                            else:
                                # Original single-stage pruning
                                with torch.no_grad():
                                    scores = model(data_device.x, data_device.edge_index, data_device.edge_attr)
                                kept_edges = topk_prune_per_node(data_device.edge_index, scores, k, n_nodes)
                        else:
                            if mode_name == 'nearest_k':
                                kept_edges = prune_nearest_k(D, candidate_edges, edge_to_idx, k)
                            elif mode_name == 'random_k':
                                kept_edges = prune_random_k(n_nodes, candidate_edges, edge_to_idx, k, seed=42)
                            elif mode_name == 'delaunay_k':
                                kept_edges = prune_delaunay_k(coords, D, candidate_edges, edge_to_idx, k)
                            else:
                                raise ValueError(f"Unknown baseline mode: {mode_name}")

                        # 3b. Compute metadata (always)
                        metadata = {
                            'instance_name': inst_name,
                            'n_nodes': n_nodes,
                            'original_edges': original_edges,
                            'kept_edges': len(kept_edges),
                            'k': k,
                            'sparsity': len(kept_edges) / original_edges * 100,
                            'timestamp': datetime.now().isoformat()
                        }
                        metadata.update(compute_degree_stats(kept_edges, edge_mapping, n_nodes))
                        metadata.update(compute_connectivity(kept_edges, edge_mapping, n_nodes))
                        metadata.update(compute_knn_overlap(coords, kept_edges, edge_mapping, knn_k=10))
                        metadata.update(compute_edge_length_stats(kept_edges, edge_mapping, D))
                        if opt_tour:
                            metadata.update(compute_tour_metrics(kept_edges, edge_mapping, opt_tour, n_nodes))
                        metadata['feasible'] = metadata.get('is_connected', False)
                        k_metadata[k].append(metadata)
                        tqdm.write(f"    pruned edges: {len(kept_edges)} | connected: {metadata['is_connected']} | avg deg: {metadata['avg_degree']:.1f}")

                        # 3c. Run LKH (if in range)
                        if run_lkh:
                            for seed in args.seeds:
                                tqdm.write(f"    [LKH] seed {seed}: ", end="")
                                if inst_name not in full_cache:
                                    full_cache[inst_name] = {}
                                if seed not in full_cache[inst_name]:
                                    # Solve full graph once per seed
                                    with tempfile.TemporaryDirectory() as tmpd:
                                        full_tsp = os.path.join(tmpd, f"{inst_name}_full.tsp")
                                        with open(full_tsp, 'w') as f:
                                            f.write(f"NAME : {inst_name}\n")
                                            f.write(f"TYPE : TSP\n")
                                            f.write(f"DIMENSION : {n_nodes}\n")
                                            f.write(f"EDGE_WEIGHT_TYPE: EXPLICIT\n")
                                            f.write(f"EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
                                            f.write(f"EDGE_WEIGHT_SECTION\n")
                                            for i in range(n_nodes):
                                                row = [str(int(round(D[i,j]))) for j in range(n_nodes)]
                                                f.write(" ".join(row) + "\n")
                                            f.write("EOF\n")
                                        full_par = os.path.join(tmpd, "run.par")
                                        full_tour = os.path.join(tmpd, "full.sol")
                                        with open(full_par, 'w') as f:
                                            f.write(f"PROBLEM_FILE = {full_tsp}\n")
                                            f.write(f"OUTPUT_TOUR_FILE = {full_tour}\n")
                                            f.write("RUNS = 1\n")
                                            f.write("MOVE_TYPE = 5\n")
                                            f.write(f"TIME_LIMIT = {TIME_LIMIT}\n")
                                        tour_full, runtime_full = solve_tsp_lkh(full_par, seed=seed)
                                        if tour_full is not None:
                                            cost_full = tour_length(tour_full, D)
                                            full_cache[inst_name][seed] = (cost_full, runtime_full)
                                            tqdm.write(f"full (cost={cost_full:.1f})", end="")
                                        else:
                                            full_cache[inst_name][seed] = (None, runtime_full)
                                            tqdm.write("full FAILED", end="")
                                else:
                                    cost_full, runtime_full = full_cache[inst_name][seed]
                                    if cost_full is not None:
                                        tqdm.write(f"full (cached) (cost={cost_full:.1f})", end="")
                                    else:
                                        tqdm.write("full FAILED (cached)", end="")

                                cost_full, runtime_full = full_cache[inst_name][seed]
                                if cost_full is None:
                                    tqdm.write(" ⇒ skipping pruned")
                                    continue

                                # Build temporary sparse .tsp and run LKH
                                with tempfile.TemporaryDirectory() as tmpd:
                                    sparse_tsp = os.path.join(tmpd, f"{inst_name}_k{k}.tsp")
                                    kept_set = set()
                                    for e in kept_edges:
                                        u, v = edge_mapping[e]
                                        kept_set.add((u, v))
                                        kept_set.add((v, u))
                                    with open(sparse_tsp, 'w') as f:
                                        f.write(f"NAME : {inst_name}_k{k}\n")
                                        f.write(f"TYPE : TSP\n")
                                        f.write(f"DIMENSION : {n_nodes}\n")
                                        f.write(f"EDGE_WEIGHT_TYPE: EXPLICIT\n")
                                        f.write(f"EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
                                        f.write(f"EDGE_WEIGHT_SECTION\n")
                                        for i in range(n_nodes):
                                            row = []
                                            for j in range(n_nodes):
                                                if i == j:
                                                    row.append("0")
                                                elif (i, j) in kept_set:
                                                    row.append(str(int(round(D[i, j]))))
                                                else:
                                                    row.append("9999999")
                                            f.write(" ".join(row) + "\n")
                                        f.write("EOF\n")

                                    par_file = os.path.join(tmpd, "run.par")
                                    pruned_tour_path = os.path.join(args.output_dir, mode_name,
                                                        f"{inst_name}_k{k}_seed{seed}.sol")
                                    with open(par_file, 'w') as f:
                                        f.write(f"PROBLEM_FILE = {sparse_tsp}\n")
                                        f.write(f"OUTPUT_TOUR_FILE = {pruned_tour_path}\n")
                                        f.write("RUNS = 1\n")
                                        f.write("MOVE_TYPE = 5\n")
                                        f.write(f"TIME_LIMIT = {TIME_LIMIT}\n")

                                    tour_pruned, runtime_pruned = solve_tsp_lkh(par_file, seed=seed)

                                    if tour_pruned is not None:
                                        cost_pruned = tour_length(tour_pruned, D)
                                        tour_found = True
                                        tqdm.write(f" → pruned cost={cost_pruned:.1f}")
                                    else:
                                        cost_pruned = float('inf')
                                        tour_found = False
                                        tqdm.write(" → pruned FAILED")

                                    gap_abs = cost_pruned - cost_full
                                    gap_rel = (gap_abs / cost_full) * 100 if cost_full > 0 else float('inf')
                                    opt_cost = tour_length(opt_tour, D) if opt_tour else None

                                    mode_lkh_rows.append({
                                        'instance': inst_name,
                                        'mode': mode_name,
                                        'seed': seed,
                                        'k': k,
                                        'n_nodes': n_nodes,
                                        'cost_full': cost_full,
                                        'cost_pruned': cost_pruned,
                                        'gap_abs': gap_abs,
                                        'gap_rel_percent': gap_rel,
                                        'runtime_full': runtime_full,
                                        'runtime_pruned': runtime_pruned,
                                        'tour_found': tour_found,
                                        'opt_cost': opt_cost
                                    })
                        # end if run_lkh
                    # end for k
                except Exception as e:
                    import traceback
                    tqdm.write(f"ERROR on {inst_name}: {e}")
                    traceback.print_exc()
                    continue

            # After processing all instances for this mode, write per‑k summaries and master summary
            for k in k_values:
                rows = k_metadata[k]
                if rows:
                    df_k = pd.DataFrame(rows)
                    # Add average row
                    numeric_cols = ['n_nodes', 'original_edges', 'kept_edges',
                                    'sparsity', 'avg_degree', 'min_degree', 'max_degree',
                                    'avg_knn_overlap', 'tour_edge_recall']
                    # Include is_connected as proportion
                    df_k['is_connected_num'] = df_k['is_connected'].astype(int)
                    numeric_cols.append('is_connected_num')
                    average = {'instance_name': '** AVERAGE **', 'k': k}
                    for col in numeric_cols:
                        if col in df_k.columns:
                            average[col] = df_k[col].mean()
                    average['is_connected'] = average.pop('is_connected_num')
                    df_k = pd.concat([pd.DataFrame([average]), df_k], ignore_index=True)
                    csv_path = os.path.join(mode_output_dir, f"summary_k{k}.csv")
                    df_k.to_csv(csv_path, index=False)
                    print(f"Saved {csv_path}")

            # Write master summary across k (just concatenation)
            master_rows = []
            for k in k_values:
                if k_metadata[k]:
                    for row in k_metadata[k]:
                        master_rows.append(row)
            if master_rows:
                master_df = pd.DataFrame(master_rows)
                master_csv = os.path.join(mode_output_dir, "k_sweep_master_summary.csv")
                master_df.to_csv(master_csv, index=False)
                print(f"Saved {master_csv}")

            # Write LKH detailed results for this mode
            if mode_lkh_rows:
                lkh_df = pd.DataFrame(mode_lkh_rows)
                lkh_csv = os.path.join(mode_output_dir, "lkh_results.csv")
                lkh_df.to_csv(lkh_csv, index=False)
                print(f"Saved {lkh_csv}")

    print("\nPipeline finished.")

if __name__ == "__main__":
    main()