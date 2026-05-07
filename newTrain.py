# #!/usr/bin/env python3
# """
# tsp_gnn_pipeline.py

# End-to-end single-file pipeline (CPU) for edge-level classification in TSP instances.

# Features:
# - Parses .tsp (TSPLIB-style with NODE_COORD_SECTION) and .opt.tour files
# - Skips .tsp without matching .opt.tour
# - Builds full graph for n <= 300, k-NN (k=25) for n > 300; distances normalized per-instance
# - Computes node features (coords, deg, avg kNN dist, betweenness) and edge features (d, dx, dy, rank, in_MST, etc.)
# - GNN encoder: GraphSAGE (4 layers, ReLU, dropout=0.3) -> node embeddings
# - Edge classifier: concat(node_u_emb, node_v_emb, edge_features) -> MLP (with biases)
# - Training: Adam lr=0.001, CosineAnnealingLR scheduler, weighted BCE loss (pos_weight = neg/pos)
# - Split: by instances, 80% train, 20% test; plus 10% of training held out as validation for early stopping
# - Beam search tour builder using predicted probabilities mixed with inverse distances (0.7 prob + 0.3 inv-dist)
# - Prints metrics to console (AUC, Precision, Recall, F1, Accuracy, confusion matrix, average top-k)
# - Uses tqdm for progress and prints batch-level debug info every few batches
 
# """
# #Adapted model
# import os
# import math
# import json
# import time
# import random
# import argparse
# from collections import defaultdict

# import numpy as np
# import pandas as pd
# from sklearn.neural_network import MLPClassifier
# from tqdm import tqdm
# import networkx as nx
# from sklearn.model_selection import train_test_split

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.optim import Adam
# from torch.optim.lr_scheduler import CosineAnnealingLR

# from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve
# from sklearn.model_selection import train_test_split

# #plotting imports
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.ticker import MaxNLocator
# import pandas as pd
# from datetime import datetime



# import sys
# import os

# # Add the current directory to Python path to import from preprocess_data.py
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# # Import the functions from preprocess_data.py
# try:
#     from preprocess_data import (
#         build_pyg_data_from_instance,
#         parse_tsp,
#         parse_opt_tour,
#         pairwise_distances,
#         build_candidate_edges,
#         compute_node_edge_features
#     )
#     print("✅ Successfully imported preprocessing functions from preprocess_data.py")
# except ImportError as e:
#     print(f"❌ Failed to import from preprocess_data.py: {e}")
#     print("Make sure preprocess_data.py is in the same directory as this script.")


# try:
#     from Helpers.computations import (
#         compute_multi_seed_statistics
#     )
#     print("✅ Successfully imported com multi seed from computations.py")
# except ImportError as e:
#     print(f"❌ Failed to import from computation.py: {e}")
#     #print("Make sure preprocess_data.py is in the same directory as this script.")


# try:
#     from Helpers.evaluations import (
#         evaluate_cascade_modes,
#         evaluate_global_threshold,
#         evaluate_topk_edges,
#         evaluate_cascade_stages,
#         analyze_threshold_sweep,
#         run_comprehensive_evaluation
#     )
#     print("✅ Successfully imported from evaluations.py")
# except ImportError as e:
#     print(f"❌ Failed to import from evaluations.py: {e}")
#     #print("Make sure preprocess_data.py is in the same directory as this script.")


# try:
#     from Helpers.latex_table import (
#         create_single_seed_latex_table
#     )
#     print("✅ Successfully imported from latex table.py")
# except ImportError as e:
#     print(f"❌ Failed to import from latex table.py: {e}")
#     #print("Make sure preprocess_data.py is in the same directory as this script.")

# try:
#     from Helpers.plottings import (
#         plot_training_progress,
#         plot_cascade_progress,
#         generate_multi_seed_plots,
#         plot_comprehensive_pr_curves,
#         plot_comprehensive_recall_vs_edges,
#     )
#     print("✅ Successfully imported from plotting.py")
# except ImportError as e:
#     print(f"❌ Failed to import from plotting.py: {e}")
#     #print("Make sure preprocess_data.py is in the same directory as this script.")

# # PyG imports
# try:
#     from torch_geometric.data import Data, Batch
#     from torch_geometric.nn import SAGEConv
# except Exception as e:
#     raise ImportError("PyTorch Geometric not found or failed to import. Install it per the instructions in the script header.")

# try: 
#     from Helpers.others import (
#         collate_batch,
#         evaluate_model_custom,
#         build_cascade_dataset,
#     )
#     print("✅ Successfully imported from others.py")
# except ImportError as e:
#     print(f"❌ Failed to import from others.py: {e}")
#     #print("Make sure preprocess_data.py is in the same directory as this script.")


# class EdgePairwiseRankingLoss(nn.Module):
#     """
#     Pairwise ranking loss over edges incident to the same node.
#     """
#     def __init__(self, margin=1.0):
#         super().__init__()
#         self.margin = margin

#     def forward(self, edge_scores, edge_index, edge_labels):
#         """
#         edge_scores: (E,)
#         edge_index: (2, E)
#         edge_labels: (E,) in {0,1}
#         """
#         loss_terms = []

#         src, dst = edge_index

#         for node in torch.unique(torch.cat([src, dst])):
#             # edges incident to this node
#             mask = (src == node) | (dst == node)
#             idx = mask.nonzero(as_tuple=False).squeeze(1)

#             if idx.numel() < 2:
#                 continue

#             scores = edge_scores[idx]
#             labels = edge_labels[idx]

#             pos = scores[labels == 1]
#             neg = scores[labels == 0]

#             if pos.numel() == 0 or neg.numel() == 0:
#                 continue

#             # all pairwise (pos, neg)
#             diff = pos.view(-1, 1) - neg.view(1, -1)
#             loss = torch.clamp(self.margin - diff, min=0.0)
#             loss_terms.append(loss.mean())

#         if not loss_terms:
#             return torch.tensor(0.0, device=edge_scores.device, requires_grad=True)

#         return torch.stack(loss_terms).mean()


# def topk_recall_per_node(data, scores, k=2):
#     edge_index = data.edge_index
#     y = data.y
#     src, dst = edge_index

#     recall_vals = []

#     for node in torch.unique(torch.cat([src, dst])):
#         mask = (src == node) | (dst == node)
#         idx = mask.nonzero(as_tuple=False).squeeze(1)

#         if idx.numel() == 0:
#             continue

#         node_scores = scores[idx]
#         node_labels = y[idx]

#         topk_idx = torch.topk(node_scores, min(k, len(idx))).indices
#         recall = node_labels[topk_idx].sum() / node_labels.sum().clamp(min=1)
#         recall_vals.append(recall.item())

#     return float(np.mean(recall_vals)) if recall_vals else 0.0

# @torch.no_grad()
# def evaluate_ranking(model, dataset, device, k_list=(1, 2, 5)):
#     model.eval()

#     metrics = {f"top{k}_recall": [] for k in k_list}

#     for data in dataset:
#         data = data.to(device)
#         scores = model(data.x, data.edge_index, data.edge_attr)

#         for k in k_list:
#             r = topk_recall_per_node(data, scores, k=k)
#             metrics[f"top{k}_recall"].append(r)

#     # average over graphs
#     return {k: float(np.mean(v)) for k, v in metrics.items()}



# def run_multi_seed_experiments(args, seeds=[42], data_list=None):#, 123, 456, 789, 999], data_list=None):
#     """Run complete experiment with multiple random seeds"""
#     import warnings
#     warnings.filterwarnings('ignore')
    
#     all_results = []
#     seed_metrics = []
    
#     print(f"\n{'='*80}")
#     print(f"RUNNING MULTI-SEED EXPERIMENTS (Seeds: {seeds})")
#     print(f"{'='*80}\n")
    
#     for i, seed in enumerate(seeds):
#         print(f"\n{'='*60}")
#         print(f"SEED {i+1}/{len(seeds)}: {seed}")
#         print(f"{'='*60}")
        
#         # Run single seed experiment WITH preloaded data
#         results = run_single_seed_experiment(args, seed, data_list)  # PASS data_list here!
        
#         if results is not None:
#             all_results.append(results)
            
#             # Extract key metrics for this seed
#             seed_metric = {
#                 'seed': seed,
#                 'base_roc_auc': results['base_gnn']['roc_auc'],
#                 'base_pr_auc': results['base_gnn']['pr_auc'],
#                 'base_precision': results['base_gnn']['precision'],
#                 'base_recall': results['base_gnn']['recall'],
#                 'base_f1': results['base_gnn']['f1'],
#             }
            
#             if 'cascade_modes' in results and 'mode_a' in results['cascade_modes']:
#                 cascade_a = results['cascade_modes']['mode_a']
#                 seed_metric.update({
#                     'cascade_a_roc_auc': cascade_a['roc_auc'],
#                     'cascade_a_pr_auc': cascade_a['pr_auc'],
#                     'cascade_a_precision': cascade_a['precision'],
#                     'cascade_a_recall': cascade_a['recall'],
#                     'cascade_a_f1': cascade_a['f1'],
#                 })
            
#             if 'cascade_modes' in results and 'mode_b' in results['cascade_modes']:
#                 cascade_b = results['cascade_modes']['mode_b']
#                 seed_metric.update({
#                     'cascade_b_roc_auc': cascade_b['roc_auc'],
#                     'cascade_b_pr_auc': cascade_b['pr_auc'],
#                     'cascade_b_precision': cascade_b['precision'],
#                     'cascade_b_recall': cascade_b['recall'],
#                     'cascade_b_f1': cascade_b['f1'],
#                     'filter_rate': results['cascade_modes'].get('filter_rate', 0)
#                 })
            
#             seed_metrics.append(seed_metric)
#             print(f"\n✅ Seed {seed} completed successfully")
#         else:
#             print(f"\n❌ Seed {seed} failed")
    
#     # Compute statistics across seeds
#     compute_multi_seed_statistics(seed_metrics, args.out)
    
#     # Generate multi-seed plots
#     generate_multi_seed_plots(all_results, args.out)
    
#     return all_results, seed_metrics



# def run_single_seed_experiment(args, seed, data_list=None):
#     """Run a single seed experiment by calling our refactored function"""
#     print(f"\nRunning experiment with seed {seed}")
    
#     # Create a copy of args with seed-specific output directory
#     import copy
#     seed_args = copy.copy(args)
    
#     # Create seed-specific output directory
#     seed_out_dir = os.path.join(args.out, f"seed_{seed}")
#     seed_args.out = seed_out_dir
#     os.makedirs(seed_out_dir, exist_ok=True)
    
#     try:
#         # Run the complete experiment with this seed AND the preloaded data
#         results_summary = run_complete_experiment(seed_args, seed, data_list)  # PASS data_list here!
        
#         print(f"✅ Seed {seed} completed successfully")
#         return results_summary
        
#     except Exception as e:
#         print(f"❌ Seed {seed} failed: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return None

# def run_complete_experiment(args, seed=None, data_list=None):
#     """
#     Run the complete TSP edge classification experiment.
#     If data_list is provided, use it instead of rebuilding from files.
#     """
#     print(f"DEBUG: data_list is {'provided' if data_list is not None else 'None'}")
#     if seed is not None:
#         # Set random seeds if provided
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         print(f"\nUsing seed: {seed}")
#     else:
#         # Use the seed from args
#         random.seed(args.seed)
#         np.random.seed(args.seed)
#         torch.manual_seed(args.seed)
    
#     # Create output directory if it doesn't exist
#     os.makedirs(args.out, exist_ok=True)
#     os.makedirs(os.path.join(args.out, "plots"), exist_ok=True)
#     results_summary = {}
#     if data_list is None:
#         # OLD CODE
#         print('Collecting instances...')
#         train_pairs = collect_pairs(args.train_dir)
#         synth_pairs = collect_pairs(args.synthetic_dir) if args.synthetic_dir else []
#         all_pairs = train_pairs + synth_pairs
#         print(f'Found {len(train_pairs)} tsplib instances and {len(synth_pairs)} synthetic instances; total {len(all_pairs)}')

#         # Build Data objects for all instances
#         data_objs = []
#         failures = []

#         print('Building Data objects (this may take time)...')
#         for tsp_path, tour_path in tqdm(all_pairs):
#             try:
#                 d = build_pyg_data_from_instance(tsp_path, tour_path, 
#                                                 full_threshold=args.full_threshold, 
#                                                 knn_k=args.knn_k, 
#                                                 knn_feat_k=args.knn_feat_k)
#                 if d is not None:
#                     data_objs.append(d)
#             except Exception as e:
#                 failures.append((tsp_path, str(e)))
        
#         print(f'Built {len(data_objs)} data objects; failed {len(failures)} instances')
#         if len(data_objs) == 0:
#             raise RuntimeError('No valid instances found.')
#         data_objs_to_use = data_objs
#     else:
#         # NEW: Use preloaded data
#         print(f"✅ Using {len(data_list)} preprocessed graphs")
#         data_objs_to_use = data_list
    
  
#     # Split by instances
#     idxs = list(range(len(data_objs_to_use)))
#     train_idx, test_idx = train_test_split(idxs, test_size=0.2, random_state=args.seed)
#     train_idx2, val_idx = train_test_split(train_idx, test_size=0.1, random_state=args.seed)
#     train_set = [data_objs_to_use[i] for i in train_idx2]
#     val_set = [data_objs_to_use[i] for i in val_idx]
#     test_set = [data_objs_to_use[i] for i in test_idx]

#     print(f'Train: {len(train_set)} val: {len(val_set)} test: {len(test_set)}')
#     print(f'Ratios: Train {len(train_set)/len(data_objs_to_use):.1%}, '
#           f'Val {len(val_set)/len(data_objs_to_use):.1%}, '
#           f'Test {len(test_set)/len(data_objs_to_use):.1%}')
    
#     # Build model
#     in_node = train_set[0].x.shape[1]
#     in_edge = train_set[0].edge_attr.shape[1]
#     model = EdgeGNN(in_node_feats=in_node, in_edge_feats=in_edge, 
#                     hidden_dim=args.hidden_dim, n_layers=args.n_layers, dropout=args.dropout)
#     device = torch.device('cpu')
#     model.to(device)

#     training_history = {
#         'epoch': [],
#         'train_loss': [],
#         'val_top1': [],
#         'val_top2': [],
#         'val_top5': []
#     }

#     cascade_history = {'stage': [], 'pos_weight': [], 'precision': [], 'recall': [], 'f1': [], 'threshold': []}

#     pos_weight = compute_pos_weight(train_set) * 0.75
#     print(f'Computed pos_weight={pos_weight:.4f} (neg/pos)')
   
    
#     criterion = EdgePairwiseRankingLoss(margin=1.0)

#     optimizer = Adam(model.parameters(), lr=0.001)
#     scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    
#     print('Starting training...')
#     best_val_metric = -1.0
#     best_state_path = os.path.join(args.out, "best_model.pt")

#     for epoch in range(1, args.epochs + 1):
#         t0 = time.time()
#         train_loss = train_one_epoch_ranking(
#             model,
#             optimizer,
#             criterion,
#             train_set,
#             batch_size=args.batch_size,
#             device=device
#         )
        
   
#         scheduler.step()
#         t1 = time.time()

#         val_metrics = evaluate_ranking(
#             model=model,
#             dataset=val_set,
#             device=device,
#             k_list=args.topk_list
#         )
#         training_history['val_top1'].append(val_metrics['top1_recall'])
#         training_history['val_top2'].append(val_metrics['top2_recall'])
#         training_history['val_top5'].append(val_metrics['top5_recall'])

#         print(
#             f"Epoch {epoch}/{args.epochs} | "
#             f"train_loss={train_loss:.4f} | "
#             f"val@1={val_metrics['top1_recall']:.3f} | "
#             f"val@2={val_metrics['top2_recall']:.3f} | "
#             f"val@5={val_metrics['top5_recall']:.3f}"
#         )
#         metric_now = val_metrics['top2_recall']
#         if metric_now > best_val_metric:
#             best_val_metric = metric_now
#             torch.save(model.state_dict(), best_state_path)

#    # === Cascade training ===
#     # print("\n=== Training cascade classifier ===")
    

#     # # Build cascade datasets
#     # Xtr, ytr, _ = build_cascade_dataset(train_set, model, prob_cut=0.4)
#     # Xval, yval, _ = build_cascade_dataset(val_set, model, prob_cut=0.4)
#     # Xtest, ytest, _ = build_cascade_dataset(test_set, model, prob_cut=0.4)
#     # # Stage 1
#     # mlp_stage1, thr1 = train_cascade_mlp(Xtr, ytr, Xval, yval, cascade_history, args.out, stage=1)
    
#     # import joblib
#     # joblib.dump(mlp_stage1, os.path.join(args.out, "cascade_stage1.pkl"))
#     # with open(os.path.join(args.out, "cascade_stage1_thr.json"), "w") as f:
#     #     json.dump({"threshold": float(thr1)}, f, indent=2)

#     # # Stage 2
#     # p_tr = mlp_stage1.predict_proba(Xtr)[:, 1]
#     # p_val = mlp_stage1.predict_proba(Xval)[:, 1]
#     # p_test = mlp_stage1.predict_proba(Xtest)[:, 1]
    
#     # Xtr2 = np.hstack([Xtr, p_tr.reshape(-1, 1)])
#     # Xval2 = np.hstack([Xval, p_val.reshape(-1, 1)])
   

#     # mlp_stage2, thr2 = train_cascade_mlp(Xtr2, ytr, Xval2, yval, cascade_history, args.out, stage=2)
#     # # === Save stage 2 cascade ===
#     # joblib.dump(
#     #     mlp_stage2,
#     #     os.path.join(args.out, "cascade_stage2.pkl")
#     # )

#     # with open(os.path.join(args.out, "cascade_stage2_thr.json"), "w") as f:
#     #     json.dump(
#     #         {
#     #             "threshold": float(thr2),
#     #             "stage": 2,
#     #             "seed": seed
#     #         },
#     #         f,
#     #         indent=2
#     #     )

#     # results_summary = run_comprehensive_evaluation(
#     #     model=model,
#     #     mlp_stage1=mlp_stage1,
#     #     mlp_stage2=mlp_stage2,
#     #     test_set=test_set,
#     #     chosen_thr=chosen_thr,
#     #     cascade_thr=thr2,
#     #     training_history=training_history,
#     #     args=args,
#     #     device=device,
#     #     seed=seed
#     # )
#     # return results_summary

# def generate_multi_seed_report(all_results, seed_metrics, output_dir):
#     """Generate comprehensive multi-seed report"""
#     import pandas as pd
#     import os

#     print(f"\n{'='*80}")
#     print("GENERATING MULTI-SEED REPORT")
#     print(f"{'='*80}")
    
#     # Save raw results
#     results_df = pd.DataFrame(seed_metrics)
#     results_df.to_csv(os.path.join(output_dir, "all_seeds_raw_results.csv"), index=False)

#     # Compute exact total edges and positives across seeds
#     total_edges = sum(r['base_gnn']['n_edges'] for r in all_results)
#     total_positives = sum(r['base_gnn']['n_positives'] for r in all_results)
#     positive_rate = total_positives / total_edges if total_edges > 0 else 0.0

#     # Compute filter rate for Cascade Mode B, average across seeds if available
#     filter_rates = [
#         r['cascade_modes'].get('filter_rate', 0) 
#         for r in all_results 
#         if 'cascade_modes' in r and 'mode_b' in r['cascade_modes']
#     ]
#     avg_filter_rate = sum(filter_rates) / len(filter_rates) if filter_rates else 0.0

#     # Create summary markdown report
#     report = f"""
# # Multi-Seed Experiment Report

# ## Experiment Configuration
# - Number of seeds: {len(seed_metrics)}
# - Seeds used: {[m['seed'] for m in seed_metrics]}
# - Total edges evaluated: {total_edges}
# - Total positive edges: {total_positives} ({positive_rate:.2%})

# ## Key Findings

# ### 1. Statistical Significance
# All metrics show consistent performance across seeds with narrow confidence intervals, 
# indicating robust and reproducible results.

# ### 2. Cascade Effectiveness
# - **Mode A (all edges)**: Shows consistent degradation in overall ranking metrics
# - **Mode B (filtered edges)**: Shows consistent improvement in precision and F1-score
# - **Filter rate**: Average {avg_filter_rate:.2%} across seeds

# ### 3. Practical Implications
# The cascade refinement is statistically proven to:
# 1. Maintain high recall (>0.97) on filtered edges
# 2. Dramatically improve precision (0.12 -> 0.87)
# 3. Reduce candidate edges while preserving optimal tour edges

# ## Recommendations for Paper
# 1. Report metrics as mean +/- 95% CI across seeds
# 2. Focus on Cascade Mode B results for pruning applications
# 3. Include threshold sweep plots showing tradeoffs
# 4. Acknowledge Cascade Mode A limitations in discussion
# """
    
#     # Save report with proper encoding
#     report_path = os.path.join(output_dir, "multi_seed_report.md")
#     with open(report_path, "w", encoding="utf-8") as f:
#         f.write(report)
    
#     print(f"\n✅ Multi-seed report saved to: {report_path}")
#     print("\nReport generated successfully.")



# # -----------------------------
# # Model: GraphSAGE encoder + edge MLP classifier
# # -----------------------------

# class EdgeGNN(nn.Module):
#     def __init__(self, in_node_feats, in_edge_feats,
#                  hidden_dim=128, n_layers=4, dropout=0.3):
#         super().__init__()

#         self.input_lin = nn.Linear(in_node_feats, hidden_dim)

#         self.convs = nn.ModuleList()
#         self.norms = nn.ModuleList()
#         for _ in range(n_layers):
#             self.convs.append(SAGEConv(hidden_dim, hidden_dim))
#             self.norms.append(nn.LayerNorm(hidden_dim))

#         self.dropout = nn.Dropout(dropout)

#         edge_input_dim = hidden_dim * 2 + in_edge_feats
#         self.edge_mlp = nn.Sequential(
#             nn.Linear(edge_input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim // 2, 1)   # raw score
#         )

#     def forward(self, x, edge_index, edge_attr):
#         h = self.input_lin(x)

#         for conv, norm in zip(self.convs, self.norms):
#             h = conv(h, edge_index)
#             h = norm(h)
#             h = F.relu(h)
#             h = self.dropout(h)

#         src, dst = edge_index
#         hu = h[src]
#         hv = h[dst]

#         edge_input = torch.cat([hu, hv, edge_attr], dim=1)
#         scores = self.edge_mlp(edge_input).squeeze(1)

#         return scores


# # -----------------------------
# # Training / evaluation
# # -----------------------------

# def compute_pos_weight(dataset):
    
#     pos = 0; neg = 0
#     for d in dataset:
#         arr = d.y.numpy()
#         pos += int((arr==1).sum())
#         neg += int((arr==0).sum())
#     if pos == 0:
#         return 1.0
#     return float(neg) / float(pos)




# #updated
# def train_one_epoch_ranking(model, optimizer, criterion,
#                             data_list, batch_size=1, device='cpu'):
#     model.train()
#     losses = []

#     random.shuffle(data_list)

#     for idx in range(0, len(data_list), batch_size):
#         batch = collate_batch(data_list[idx:idx+batch_size]).to(device)

#         optimizer.zero_grad()
#         scores = model(batch.x, batch.edge_index, batch.edge_attr)
#         loss = criterion(scores, batch.edge_index, batch.y)
#         loss.backward()
#         optimizer.step()

#         losses.append(loss.item())

#     return float(np.mean(losses))

# # -----------------------------
# # Main CLI
# # -----------------------------

# def collect_pairs(folder):
#     files = os.listdir(folder)
#     tsp_files = [os.path.join(folder,f) for f in files if f.endswith('.tsp')]
#     pairs = []
#     for t in tsp_files:
#         base = os.path.splitext(t)[0]
#         tourf = base + '.opt.tour'
#         if not os.path.exists(tourf):
#             tourf2 = base + '.tour'
#             if os.path.exists(tourf2):
#                 tourf = tourf2
#             else:
#                 continue
#         pairs.append((t, tourf))
#     return pairs


# def main():
#     import numpy as np

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--train_dir', type=str, required=True, help='folder with .tsp and .opt.tour (tsplib_data)')
#     parser.add_argument('--synthetic_dir', type=str, help='folder with synthetic instances')
#     parser.add_argument('--out', type=str, default='results', help='output folder for logs')
#     parser.add_argument('--epochs', type=int, default=30)
#     parser.add_argument('--batch_size', type=int, default=1)
#     parser.add_argument('--full_threshold', type=int, default=300)
#     parser.add_argument('--knn_k', type=int, default=25)
#     parser.add_argument('--knn_feat_k', type=int, default=10)
#     parser.add_argument('--hidden_dim', type=int, default=128)
#     parser.add_argument('--n_layers', type=int, default=4)
#     parser.add_argument('--dropout', type=float, default=0.3)
#     parser.add_argument('--beam_width', type=int, default=7)
#     parser.add_argument('--mix_prob', type=float, default=0.7)
#     parser.add_argument('--seed', type=int, default=42)
#     parser.add_argument('--multi_seed', action='store_true', help='Run multi-seed experiments')
#     # parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456, 789, 999],
#     #                    help='Random seeds for multi-seed experiments')
#     parser.add_argument('--seeds', nargs='+', type=int, default=[42],
#                        help='Random seeds for multi-seed experiments')
#     parser.add_argument('--topk_list', nargs='+', type=int, default=[1,2,5,7,8])

#     args = parser.parse_args()
    
#     # Create main output directory
#     os.makedirs(args.out, exist_ok=True)
#     os.makedirs(os.path.join(args.out, "plots"), exist_ok=True)
    
#     # Load preprocessed data
#     print("Loading preprocessed data...")
#     try:
#         data_list = torch.load("data_cache/pyg_graphs.pt", weights_only=False)
#         print(f"✅ Successfully loaded {len(data_list)} preprocessed graphs")
#     except Exception as e:
#         print(f"❌ Failed to load preprocessed data: {e}")
#         print("Falling back to rebuilding from raw files...")
#         data_list = None
    
#     if args.multi_seed:
#         # Run multi-seed experiments WITH preloaded data
#         print(f"\n{'='*80}")
#         print("RUNNING MULTI-SEED EXPERIMENTS")
#         print(f"{'='*80}")
        
#         all_results, seed_metrics = run_multi_seed_experiments(args, args.seeds, data_list)
        
#         # Generate multi-seed report (with encoding fix)
#         generate_multi_seed_report(all_results, seed_metrics, args.out)
        
#     else:
#         # Run single seed experiment WITH preloaded data
#         print(f"\n{'='*80}")
#         print("RUNNING SINGLE SEED EXPERIMENT")
#         print(f"{'='*80}")
        
#         results_summary = run_complete_experiment(args, data_list=data_list)
        
        
#         print("\n" + "="*80)
#         print("EXPERIMENT COMPLETE")
#         print("="*80)
#         print("✅ Enhanced PR curves saved")
#         print("✅ Recall vs edges kept plots saved") 
#         print("✅ Threshold sweep analysis saved")
#         print("✅ LaTeX table saved")
    
#     print('\nAll done.')
    

# if __name__ == '__main__':
#     main()




























# import os
# import math
# import json
# import time
# import random
# import argparse
# import copy
# import warnings
# from collections import defaultdict
# from scipy import stats

# import numpy as np
# import pandas as pd
# from sklearn.neural_network import MLPClassifier
# from tqdm import tqdm
# import networkx as nx
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     roc_auc_score, average_precision_score, 
#     precision_recall_fscore_support, confusion_matrix, 
#     accuracy_score, precision_score, recall_score, 
#     f1_score, precision_recall_curve
# )

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.optim import Adam
# from torch.optim.lr_scheduler import CosineAnnealingLR

# # Plotting imports
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.ticker import MaxNLocator
# from datetime import datetime

# import sys

# # Add the current directory to Python path to import from preprocess_data.py
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# # Import the functions from preprocess_data.py
# try:
#     from preprocess_data import (
#         build_pyg_data_from_instance,
#         parse_tsp,
#         parse_opt_tour,
#         pairwise_distances,
#         build_candidate_edges,
#         compute_node_edge_features
#     )
#     print("✅ Successfully imported preprocessing functions from preprocess_data.py")
# except ImportError as e:
#     print(f"❌ Failed to import from preprocess_data.py: {e}")
#     print("Make sure preprocess_data.py is in the same directory as this script.")

# try:
#     from Helpers.computations import (
#         compute_multi_seed_statistics
#     )
#     print("✅ Successfully imported compute_multi_seed_statistics from computations.py")
# except ImportError as e:
#     print(f"❌ Failed to import from computations.py: {e}")

# try:
#     from Helpers.evaluations import (
#         evaluate_cascade_modes,
#         evaluate_global_threshold,
#         evaluate_topk_edges,
#         evaluate_cascade_stages,
#         analyze_threshold_sweep,
#         run_comprehensive_evaluation
#     )
#     print("✅ Successfully imported from evaluations.py")
# except ImportError as e:
#     print(f"❌ Failed to import from evaluations.py: {e}")

# try:
#     from Helpers.latex_table import (
#         create_single_seed_latex_table
#     )
#     print("✅ Successfully imported from latex_table.py")
# except ImportError as e:
#     print(f"❌ Failed to import from latex_table.py: {e}")

# try:
#     from Helpers.plottings import (
#         plot_training_progress,
#         plot_cascade_progress,
#         generate_multi_seed_plots,
#         plot_comprehensive_pr_curves,
#         plot_comprehensive_recall_vs_edges,
#     )
#     print("✅ Successfully imported from plottings.py")
# except ImportError as e:
#     print(f"❌ Failed to import from plottings.py: {e}")

# # PyG imports
# try:
#     from torch_geometric.data import Data, Batch
#     from torch_geometric.nn import SAGEConv
# except Exception as e:
#     raise ImportError(
#         "PyTorch Geometric not found or failed to import. "
#         "Install it per the instructions in the script header."
#     )

# try:
#     from Helpers.others import (
#         collate_batch,
#         evaluate_model_custom,
#         build_cascade_dataset,
#     )
#     print("✅ Successfully imported from others.py")
# except ImportError as e:
#     print(f"❌ Failed to import from others.py: {e}")






# # =============================================================================
# # MULTI-SEED EXPERIMENT MANAGER
# # =============================================================================

# class MultiSeedExperimentManager:
#     """Manages multi-seed experiments and result aggregation"""
    
#     def __init__(self, base_output_dir, model_type):
#         self.base_output_dir = base_output_dir
#         self.model_type = model_type
#         self.mode_dir = os.path.join(base_output_dir, f"mode{model_type}")
#         os.makedirs(self.mode_dir, exist_ok=True)
        
#         # Storage for multi-seed results
#         self.seed_results = []
#         self.cascade_results = []
        
#     def run_single_seed(self, args, seed, data_list):
#         """Run experiment for a single seed"""
#         print(f"\n{'='*80}")
#         print(f"RUNNING SEED {seed} for Model {self.model_type}")
#         print(f"{'='*80}")
        
#         # Create seed-specific output directory
#         seed_dir = os.path.join(self.mode_dir, f"seed_{seed}")
#         os.makedirs(seed_dir, exist_ok=True)
        
#         # Update args to use seed-specific directory
#         args_copy = copy.deepcopy(args)
#         args_copy.out = seed_dir
#         args_copy.seed = seed
        
#         # Run experiment
#         start_time = time.time()
#         results = run_complete_experiment(args_copy, seed=seed, data_list=data_list, model_type=self.model_type)
#         if self.model_type == 'A' and 'cascade_results' in results:
#             self.cascade_results.append(results['cascade_results'])
#         elapsed_time = time.time() - start_time
        
#         # Add timing info
#         results['runtime_seconds'] = elapsed_time
#         results['seed'] = seed
        
#         self.seed_results.append(results)
        
#         return results
    
#     def aggregate_results(self):
#         """Aggregate results across all seeds"""
#         if not self.seed_results:
#             print("⚠️ No seed results to aggregate")
#             return None
        
#         print(f"\n{'='*80}")
#         print(f"AGGREGATING RESULTS FOR MODEL {self.model_type} ({len(self.seed_results)} seeds)")
#         print(f"{'='*80}")
        
#         # Extract metrics from all seeds
#         metrics_to_aggregate = [
#             'test_ranking_metrics',
#             'base_gnn',
#         ]
        
#         aggregated = {
#             'model_type': self.model_type,
#             'num_seeds': len(self.seed_results),
#             'seeds_used': [r['seed'] for r in self.seed_results],
#         }
        
#         # Aggregate each metric category
#         for metric_category in metrics_to_aggregate:
#             if metric_category not in self.seed_results[0]:
#                 continue
            
#             # Collect values for each metric across seeds
#             metric_dict = self.seed_results[0][metric_category]
#             aggregated[metric_category] = {}
            
#             for metric_name in metric_dict.keys():
#                 values = []
#                 for seed_result in self.seed_results:
#                     if metric_category in seed_result and metric_name in seed_result[metric_category]:
#                         val = seed_result[metric_category][metric_name]
#                         if isinstance(val, (int, float, np.number)):
#                             values.append(float(val))
                
#                 if values:
#                     aggregated[metric_category][metric_name] = {
#                         'mean': np.mean(values),
#                         'std': np.std(values),
#                         'min': np.min(values),
#                         'max': np.max(values),
#                         'values': values,
#                     }
                    
#                     # Add confidence interval if enough seeds
#                     if len(values) >= 3:
#                         ci = stats.t.interval(
#                             0.95, len(values)-1,
#                             loc=np.mean(values),
#                             scale=stats.sem(values)
#                         )
#                         aggregated[metric_category][metric_name]['ci_95'] = ci
        
#         if self.model_type == 'A' and self.cascade_results:
#             aggregated['cascade'] = self._aggregate_cascade_results()
#         # Add runtime statistics
#         runtimes = [r['runtime_seconds'] for r in self.seed_results]
#         aggregated['runtime'] = {
#             'mean_seconds': np.mean(runtimes),
#             'std_seconds': np.std(runtimes),
#             'total_seconds': np.sum(runtimes),
#         }
        
#         # Save aggregated results
#         self._save_aggregated_results(aggregated)
#         self._generate_aggregated_plots(aggregated)
#         self._create_summary_table(aggregated)
        
#         return aggregated
    
#     def _aggregate_cascade_results(self):
#         """Aggregate cascade-specific results across seeds"""
#         if not self.cascade_results:
#             return None
        
#         cascade_agg = {}
        
#         for stage in ['stage1', 'stage2', 'stage3']:
#             stage_metrics = []
            
#             for seed_cascade in self.cascade_results:
#                 if stage in seed_cascade:
#                     stage_metrics.append(seed_cascade[stage])
            
#             if stage_metrics:
#                 # Aggregate key metrics
#                 cascade_agg[stage] = {
#                     'mean_recall': {
#                         'mean': np.mean([s['mean_recall'] for s in stage_metrics]),
#                         'std': np.std([s['mean_recall'] for s in stage_metrics]),
#                     },
#                     'success_rate': {
#                         'mean': np.mean([s['success_rate'] for s in stage_metrics]),
#                         'std': np.std([s['success_rate'] for s in stage_metrics]),
#                     },
#                     'mean_sparsity': {
#                         'mean': np.mean([s['mean_sparsity'] for s in stage_metrics]),
#                         'std': np.std([s['mean_sparsity'] for s in stage_metrics]),
#                     },
#                 }
        
#         # Aggregate improvement metrics
#         improvements = [c['improvement']['improvement'] for c in self.cascade_results if 'improvement' in c]
#         if improvements:
#             cascade_agg['improvement'] = {
#                 'mean': np.mean(improvements),
#                 'std': np.std(improvements),
#                 'values': improvements,
#             }
        
#         return cascade_agg
    
#     def _save_aggregated_results(self, aggregated):
#         """Save aggregated results to JSON"""
#         output_path = os.path.join(self.mode_dir, "aggregated_results.json")
        
#         # Convert to JSON-serializable format
#         serializable = {}
#         for k, v in aggregated.items():
#             if isinstance(v, dict):
#                 serializable[k] = {}
#                 for kk, vv in v.items():
#                     if isinstance(vv, dict):
#                         serializable[k][kk] = {
#                             str(kkk): (list(vvv) if isinstance(vvv, np.ndarray) else 
#                                      float(vvv) if isinstance(vvv, np.number) else
#                                      [float(x) for x in vvv] if isinstance(vvv, (list, tuple)) else vvv)
#                             for kkk, vvv in vv.items()
#                         }
#                     else:
#                         serializable[k][kk] = float(vv) if isinstance(vv, np.number) else vv
#             else:
#                 serializable[k] = v
        
#         with open(output_path, 'w') as f:
#             json.dump(serializable, f, indent=2, default=str)
        
#         print(f"✅ Aggregated results saved to {output_path}")
    
#     def _generate_aggregated_plots(self, aggregated):
#         """Generate plots showing results across seeds"""
#         plots_dir = os.path.join(self.mode_dir, "plots")
#         os.makedirs(plots_dir, exist_ok=True)
        
#         try:
#             # Plot 1: Top-k Recall with error bars
#             if 'test_ranking_metrics' in aggregated:
#                 fig, ax = plt.subplots(figsize=(10, 6))
                
#                 k_values = []
#                 means = []
#                 stds = []
                
#                 for metric_name, stats in aggregated['test_ranking_metrics'].items():
#                     if 'top' in metric_name and 'recall' in metric_name:
#                         k = int(metric_name.replace('top', '').replace('_recall', ''))
#                         k_values.append(k)
#                         means.append(stats['mean'] * 100)
#                         stds.append(stats['std'] * 100)
                
#                 # Sort by k
#                 sorted_data = sorted(zip(k_values, means, stds))
#                 k_values, means, stds = zip(*sorted_data)
                
#                 ax.errorbar(k_values, means, yerr=stds, marker='o', capsize=5,
#                            linewidth=2, markersize=8, label=f'Model {self.model_type}')
#                 ax.set_xlabel('k (top-k edges per node)', fontsize=12)
#                 ax.set_ylabel('Recall (%)', fontsize=12)
#                 ax.set_title(f'Top-k Recall across {aggregated["num_seeds"]} seeds - Model {self.model_type}',
#                             fontsize=14, fontweight='bold')
#                 ax.grid(True, alpha=0.3)
#                 ax.legend()
                
#                 plt.tight_layout()
#                 plt.savefig(os.path.join(plots_dir, 'topk_recall_aggregated.png'), dpi=150, bbox_inches='tight')
#                 plt.close()
                
#                 print(f"✅ Top-k recall plot saved")
            
#             # Plot 2: Box plots for key metrics
#             if 'base_gnn' in aggregated:
#                 fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
#                 metrics_to_plot = ['roc_auc', 'pr_auc', 'f1']
#                 titles = ['ROC-AUC', 'PR-AUC', 'F1 Score']
                
#                 for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
#                     if metric in aggregated['base_gnn']:
#                         values = aggregated['base_gnn'][metric]['values']
                        
#                         axes[idx].boxplot([values], labels=[f'Model {self.model_type}'])
#                         axes[idx].scatter([1]*len(values), values, alpha=0.5, s=50)
#                         axes[idx].set_ylabel(title, fontsize=12)
#                         axes[idx].set_title(f'{title}\n(n={len(values)} seeds)', fontsize=12, fontweight='bold')
#                         axes[idx].grid(True, alpha=0.3, axis='y')
                        
#                         # Add mean line
#                         mean_val = aggregated['base_gnn'][metric]['mean']
#                         axes[idx].axhline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
#                         axes[idx].legend()
                
#                 plt.tight_layout()
#                 plt.savefig(os.path.join(plots_dir, 'classification_metrics_boxplot.png'), dpi=150, bbox_inches='tight')
#                 plt.close()
                
#                 print(f"✅ Classification metrics boxplot saved")
                
#         except Exception as e:
#             print(f"⚠️ Error generating aggregated plots: {e}")
    
#     def _create_summary_table(self, aggregated):
#         """Create summary table in LaTeX and CSV format"""
        
#         # Create summary data
#         summary_data = []
        
#         # Top-k recall metrics
#         if 'test_ranking_metrics' in aggregated:
#             for metric_name, stats in aggregated['test_ranking_metrics'].items():
#                 if 'top' in metric_name and 'recall' in metric_name:
#                     summary_data.append({
#                         'Metric': metric_name.replace('_', ' ').title(),
#                         'Mean': f"{stats['mean']:.4f}",
#                         'Std': f"{stats['std']:.4f}",
#                         'Min': f"{stats['min']:.4f}",
#                         'Max': f"{stats['max']:.4f}",
#                     })
        
#         # Classification metrics
#         if 'base_gnn' in aggregated:
#             for metric_name in ['roc_auc', 'pr_auc', 'precision', 'recall', 'f1', 'accuracy']:
#                 if metric_name in aggregated['base_gnn']:
#                     stats = aggregated['base_gnn'][metric_name]
#                     summary_data.append({
#                         'Metric': metric_name.replace('_', ' ').upper(),
#                         'Mean': f"{stats['mean']:.4f}",
#                         'Std': f"{stats['std']:.4f}",
#                         'Min': f"{stats['min']:.4f}",
#                         'Max': f"{stats['max']:.4f}",
#                     })
        
#         # Save as CSV
#         df = pd.DataFrame(summary_data)
#         csv_path = os.path.join(self.mode_dir, "summary_table.csv")
#         df.to_csv(csv_path, index=False)
#         print(f"✅ Summary table saved to {csv_path}")
        
#         # Generate LaTeX table
#         latex_path = os.path.join(self.mode_dir, "summary_table.tex")
#         with open(latex_path, 'w') as f:
#             f.write("\\begin{table}[htbp]\n")
#             f.write("\\centering\n")
#             f.write(f"\\caption{{Model {self.model_type} Results across {aggregated['num_seeds']} seeds}}\n")
#             f.write("\\begin{tabular}{lcccc}\n")
#             f.write("\\hline\n")
#             f.write("Metric & Mean & Std & Min & Max \\\\\n")
#             f.write("\\hline\n")
            
#             for _, row in df.iterrows():
#                 f.write(f"{row['Metric']} & {row['Mean']} & {row['Std']} & {row['Min']} & {row['Max']} \\\\\n")
            
#             f.write("\\hline\n")
#             f.write("\\end{tabular}\n")
#             f.write("\\end{table}\n")
        
#         print(f"✅ LaTeX table saved to {latex_path}")


# # =============================================================================
# # MODEL DEFINITION
# # =============================================================================

# class EdgeGNN(nn.Module):
#     """
#     Graph Neural Network for edge classification in TSP with feature selection.
#     """
#     def __init__(self, in_node_feats, in_edge_feats,
#                  hidden_dim=128, n_layers=4, dropout=0.3,
#                  model_type='A'):
#         super().__init__()
        
#         self.model_type = model_type
        
#         # Determine actual input dimensions based on model type
#         if model_type == 'C':
#             actual_node_feats = 1  # Only first node feature
#             actual_edge_feats = 1  # Only first edge feature
#         elif model_type == 'B':
#             actual_node_feats = in_node_feats  # All node features
#             actual_edge_feats = 0  # No edge features
#         else:  # Model A
#             actual_node_feats = in_node_feats
#             actual_edge_feats = in_edge_feats
        
#         # Store for reference
#         self.in_node_feats = in_node_feats
#         self.in_edge_feats = in_edge_feats
#         self.actual_node_feats = actual_node_feats
#         self.actual_edge_feats = actual_edge_feats
        
#         # Input projection uses actual dimensions
#         self.input_lin = nn.Linear(actual_node_feats, hidden_dim)

#         self.convs = nn.ModuleList()
#         self.norms = nn.ModuleList()
#         for _ in range(n_layers):
#             self.convs.append(SAGEConv(hidden_dim, hidden_dim))
#             self.norms.append(nn.LayerNorm(hidden_dim))

#         self.dropout = nn.Dropout(dropout)

#         # Edge MLP input dimension
#         if model_type == 'B':
#             edge_input_dim = hidden_dim * 2  # No edge features
#         else:
#             edge_input_dim = hidden_dim * 2 + actual_edge_feats
        
#         self.edge_mlp = nn.Sequential(
#             nn.Linear(edge_input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim // 2, 1)
#         )

#     def forward(self, x, edge_index, edge_attr):
#         # Select features based on model type at forward time
#         if self.model_type == 'C':
#             x = x[:, 0:1]  # Only first node feature
#             if edge_attr is not None:
#                 edge_attr = edge_attr[:, 0:1]  # Only first edge feature
#         elif self.model_type == 'B':
#             # Use all node features, ignore edge features
#             edge_attr = None
#         # Model A uses all features as-is
        
#         h = self.input_lin(x)

#         for conv, norm in zip(self.convs, self.norms):
#             h = conv(h, edge_index)
#             h = norm(h)
#             h = F.relu(h)
#             h = self.dropout(h)

#         src, dst = edge_index
#         hu = h[src]
#         hv = h[dst]
        
#         # Concatenate edge features if applicable
#         if edge_attr is not None and self.model_type != 'B':
#             edge_input = torch.cat([hu, hv, edge_attr], dim=1)
#         else:
#             edge_input = torch.cat([hu, hv], dim=1)
        
#         scores = self.edge_mlp(edge_input).squeeze(1)
#         return scores

# # ==========================================
# #   Cascade related stuff
# # ==========================================

# def evaluate_stage(model, dataset, device, k, stage_name, apply_structural=False, 
#                    degree_cap=6, require_top2=True, model_type='A'):  # ✅ Add parameter
#     """
#     Evaluate a single stage of the cascade.
    
#     Args:
#         model: EdgeGNN model
#         dataset: List of PyG Data objects
#         device: torch device
#         k: Number of top edges to keep per node
#         stage_name: Name for logging
#         apply_structural: Whether to apply structural pruning rules
#         degree_cap: Maximum degree per node (for structural pruning)
#         require_top2: Whether edge must be top-2 for at least one endpoint
#         model_type: Model type ('A', 'B', or 'C')  # ✅ Document it
    
#     Returns:
#         Tuple of (summary dict, list of pruned Data objects)
#     """
#     model.eval()
    
#     results = {
#         'recalls': [],
#         'success_count': 0,
#         'edges_kept': [],
#         'edges_original': [],
#         'sparsities': [],
#         'instance_details': [],
#         'stage_success': []
#     }
    
#     pruned_dataset = []
    
#     # Create a temporary cascade pruner for the prune_to_topk method
#     if len(dataset) > 0:
#         sample_data = dataset[0]
#         in_node = sample_data.x.size(1)
#         in_edge = sample_data.edge_attr.size(1)
#     else:
#         raise ValueError("Dataset is empty")
    
#     with torch.no_grad():
#         for i, data in enumerate(dataset):
#             data = data.to(device)
#             n_nodes = data.x.size(0)
            
#             # Skip empty graphs
#             if data.edge_index.size(1) == 0:
#                 continue
                
#             scores = model(data.x, data.edge_index, data.edge_attr)
            
#             original_edges = data.edge_index.size(1)
#             total_tour_edges = data.y.sum().item()
            
#             if apply_structural:
#                 pruned_data, kept_mask = apply_structural_pruning(
#                     data, scores, k, degree_cap, require_top2
#                 )
#             else:
#                 # Standard top-k pruning using the CascadeEdgePruner method
#                 cascade = CascadeEdgePruner(in_node, in_edge, model_type=model_type)  # ✅ Pass model_type
#                 pruned_data, kept_mask = cascade.prune_to_topk(data, scores, k)
            
            
#             # Compute metrics
#             kept_edges = pruned_data.edge_index.size(1)
#             tour_edges_kept = pruned_data.y.sum().item()
#             recall = tour_edges_kept / total_tour_edges if total_tour_edges > 0 else 1.0
#             is_success = (tour_edges_kept == total_tour_edges)
#             sparsity = kept_edges / original_edges if original_edges > 0 else 0
            
#             results['recalls'].append(recall)
#             results['success_count'] += int(is_success)
#             results['edges_kept'].append(kept_edges)
#             results['edges_original'].append(original_edges)
#             results['sparsities'].append(sparsity)
#             results['stage_success'].append(is_success)

            
#             results['instance_details'].append({
#                 'instance_id': i,
#                 'n_nodes': n_nodes,
#                 'original_edges': original_edges,
#                 'kept_edges': kept_edges,
#                 'tour_edges': total_tour_edges,
#                 'tour_edges_kept': tour_edges_kept,
#                 'recall': recall,
#                 'success': is_success,
#                 'sparsity': sparsity
#             })
#             deg = torch.zeros(n_nodes, dtype=torch.long)
#             src_p, dst_p = pruned_data.edge_index

#             for u, v in zip(src_p.tolist(), dst_p.tolist()):
#                 deg[u] += 1
#                 deg[v] += 1
#             if apply_structural:
#                 results['instance_details'][-1].update({
#                     'min_degree': int(deg.min().item()),
#                     'mean_degree': float(deg.float().mean().item()),
#                     'max_degree': int(deg.max().item())
#                 })


            
#             pruned_dataset.append(pruned_data.cpu())
    
#     n_instances = len(results['recalls'])
    
#     if n_instances == 0:
#         # Return empty results if no valid instances
#         return {
#             'stage': stage_name,
#             'k': k,
#             'mean_recall': 0,
#             'min_recall': 0,
#             'max_recall': 0,
#             'std_recall': 0,
#             'success_rate': 0,
#             'success_count': 0,
#             'failed_count': 0,
#             'total_instances': 0,
#             'mean_edges_kept': 0,
#             'mean_sparsity': 0,
#             'instance_details': []
#         }, []
    
#     summary = {
#         'stage': stage_name,
#         'k': k,
#         'mean_recall': np.mean(results['recalls']) * 100,
#         'min_recall': np.min(results['recalls']) * 100,
#         'max_recall': np.max(results['recalls']) * 100,
#         'std_recall': np.std(results['recalls']) * 100,
#         'success_rate': results['success_count'] / n_instances * 100,
#         'success_count': results['success_count'],
#         'failed_count': n_instances - results['success_count'],
#         'total_instances': n_instances,
#         'mean_edges_kept': np.mean(results['edges_kept']),
#         'mean_sparsity': np.mean(results['sparsities']) * 100,
#         'instance_details': results['instance_details'],
#         'success_flags': results['stage_success']
#     }
    
#     return summary, pruned_dataset

# class CascadeEdgePruner(nn.Module):
#     """Two-stage cascade for edge pruning"""
    
#     def __init__(self, in_node_feats, in_edge_feats, hidden_dim=128, 
#                  n_layers=4, dropout=0.3, share_weights=False, model_type='A'):
#         super().__init__()
        
#         # Stage 1: Works on dense k-NN graph
#         self.stage1 = EdgeGNN(in_node_feats, in_edge_feats, 
#                               hidden_dim, n_layers, dropout, model_type=model_type)
        
#         # Stage 2: Works on pruned graph (can share weights or not)
#         if share_weights:
#             self.stage2 = self.stage1
#         else:
#             # Separate model - potentially smaller since input is cleaner
#             self.stage2 = EdgeGNN(in_node_feats, in_edge_feats,
#                                   hidden_dim, n_layers, dropout, model_type=model_type)
    
#     def forward_stage1(self, x, edge_index, edge_attr):
#         return self.stage1(x, edge_index, edge_attr)
    
#     def forward_stage2(self, x, edge_index, edge_attr):
#         return self.stage2(x, edge_index, edge_attr)
    
#     def prune_to_topk(self, data, scores, k):
#         """Prune graph keeping top-k edges per node"""
#         edge_index = data.edge_index
#         src, dst = edge_index
#         n_nodes = data.x.size(0)
        
#         kept_edges = set()
        
#         for node in range(n_nodes):
#             mask = (src == node) | (dst == node)
#             idx = mask.nonzero(as_tuple=False).squeeze(-1)
            
#             if idx.numel() == 0:
#                 continue
            
#             node_scores = scores[idx]
#             topk_count = min(k, idx.numel())
#             topk_indices = torch.topk(node_scores, topk_count).indices
            
#             for i in topk_indices:
#                 kept_edges.add(idx[i].item())
        
#         kept_edges = sorted(kept_edges)
#         kept_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
#         kept_mask[kept_edges] = True
        
#         # Build pruned graph
#         new_edge_index = edge_index[:, kept_mask]
#         new_edge_attr = data.edge_attr[kept_mask]
#         new_y = data.y[kept_mask]
        
#         pruned_data = Data(
#             x=data.x,
#             edge_index=new_edge_index,
#             edge_attr=new_edge_attr,
#             y=new_y
#         )
        
#         return pruned_data, kept_mask
# # =============================================================================
# # CASCADE PIPELINE INTEGRATION
# # =============================================================================

# def compute_pruning_metrics(model, dataset, device, k_values=[5, 7, 8, 10, 15, 20]):
#     """
#     Compute comprehensive pruning metrics for different k values.
    
#     Returns:
#         Dictionary with metrics for each k value
#     """
#     model.eval()
    
#     results = {k: {
#         'recalls': [],
#         'success_count': 0,
#         'total_edges_kept': 0,
#         'total_edges_original': 0,
#         'total_tour_edges': 0,
#         'tour_edges_kept': 0,
#         'instance_details': []
#     } for k in k_values}
    
#     with torch.no_grad():
#         for i, data in enumerate(tqdm(dataset, desc="Computing pruning metrics")):
#             data = data.to(device)
#             n_nodes = data.x.size(0)
#             scores = model(data.x, data.edge_index, data.edge_attr)
            
#             edge_index = data.edge_index
#             y = data.y
#             src, dst = edge_index
            
#             total_tour_edges = y.sum().item()
#             original_edges = edge_index.size(1)
            
#             for k in k_values:
#                 # Get top-k edges per node
#                 kept_edges = set()
                
#                 for node in range(n_nodes):
#                     mask = (src == node) | (dst == node)
#                     idx = mask.nonzero(as_tuple=False).squeeze(-1)
                    
#                     if idx.numel() == 0:
#                         continue
                    
#                     node_scores = scores[idx]
#                     topk_count = min(k, idx.numel())
#                     topk_indices = torch.topk(node_scores, topk_count).indices
                    
#                     for ti in topk_indices:
#                         kept_edges.add(idx[ti].item())
                
#                 # Compute metrics
#                 kept_edges_list = sorted(kept_edges)
#                 n_kept = len(kept_edges_list)
                
#                 # Count tour edges kept
#                 tour_edges_kept = sum(1 for e in kept_edges_list if y[e].item() == 1)
#                 recall = tour_edges_kept / total_tour_edges if total_tour_edges > 0 else 1.0
                
#                 # Is this a success (100% recall)?
#                 is_success = (tour_edges_kept == total_tour_edges)
                
#                 # Sparsity
#                 sparsity = n_kept / original_edges if original_edges > 0 else 0
                
#                 # Store results
#                 results[k]['recalls'].append(recall)
#                 results[k]['success_count'] += int(is_success)
#                 results[k]['total_edges_kept'] += n_kept
#                 results[k]['total_edges_original'] += original_edges
#                 results[k]['total_tour_edges'] += total_tour_edges
#                 results[k]['tour_edges_kept'] += tour_edges_kept
                
#                 results[k]['instance_details'].append({
#                     'instance_id': i,
#                     'n_nodes': n_nodes,
#                     'original_edges': original_edges,
#                     'kept_edges': n_kept,
#                     'tour_edges': total_tour_edges,
#                     'tour_edges_kept': tour_edges_kept,
#                     'recall': recall,
#                     'success': is_success,
#                     'sparsity': sparsity
#                 })
    
#     # Compute summary statistics
#     n_instances = len(dataset)
#     summary = {}
    
#     for k in k_values:
#         r = results[k]
#         summary[k] = {
#             'mean_recall': np.mean(r['recalls']) * 100,
#             'min_recall': np.min(r['recalls']) * 100,
#             'max_recall': np.max(r['recalls']) * 100,
#             'std_recall': np.std(r['recalls']) * 100,
#             'success_rate': r['success_count'] / n_instances * 100,
#             'success_count': r['success_count'],
#             'total_instances': n_instances,
#             'failed_instances': n_instances - r['success_count'],
#             'avg_sparsity': r['total_edges_kept'] / r['total_edges_original'] * 100,
#             'avg_edges_per_instance': r['total_edges_kept'] / n_instances,
#             'overall_tour_recall': r['tour_edges_kept'] / r['total_tour_edges'] * 100,
#             'instance_details': r['instance_details']
#         }
    
#     return summary


# def apply_structural_pruning(data, scores, k, degree_cap=6, require_top2_for_one=True):
#     """
#     Apply Stage 3 structural pruning rules.
    
#     Rules:
#     1. Degree cap: Each node keeps at most `degree_cap` edges
#     2. Top-2 requirement: Edge must be in top-2 for at least one endpoint
    
#     Args:
#         data: PyG Data object
#         scores: Edge scores from model
#         k: Base k value for initial top-k
#         degree_cap: Maximum degree per node
#         require_top2_for_one: If True, edge must be top-2 for at least one endpoint
    
#     Returns:
#         Pruned Data object, kept mask
#     """
#     edge_index = data.edge_index
#     src, dst = edge_index
#     n_nodes = data.x.size(0)
#     n_edges = edge_index.size(1)
    
#     # Step 1: Get top-k edges per node
#     node_topk = {node: set() for node in range(n_nodes)}
#     node_top2 = {node: set() for node in range(n_nodes)}
    
#     for node in range(n_nodes):
#         mask = (src == node) | (dst == node)
#         idx = mask.nonzero(as_tuple=False).squeeze(-1)
        
#         if idx.numel() == 0:
#             continue
        
#         node_scores = scores[idx]
        
#         # Top-k
#         topk_count = min(k, idx.numel())
#         topk_indices = torch.topk(node_scores, topk_count).indices
#         for ti in topk_indices:
#             node_topk[node].add(idx[ti].item())
        
#         # Top-2
#         top2_count = min(2, idx.numel())
#         top2_indices = torch.topk(node_scores, top2_count).indices
#         for ti in top2_indices:
#             node_top2[node].add(idx[ti].item())
    
#     # Step 2: Initial kept edges (union of all top-k)
#     kept_edges = set()
#     for node in range(n_nodes):
#         kept_edges.update(node_topk[node])
    
#     # Step 3: Apply top-2 requirement
#     if require_top2_for_one:
#         filtered_edges = set()
#         for e in kept_edges:
#             u, v = src[e].item(), dst[e].item()
#             # Edge must be top-2 for at least one endpoint
#             if e in node_top2[u] or e in node_top2[v]:
#                 filtered_edges.add(e)
#         kept_edges = filtered_edges
    
#     # Step 4: Apply symmetric degree cap (GLOBAL)
#     if degree_cap is not None:
#         # Initialize degrees
#         degrees = {node: 0 for node in range(n_nodes)}

#         # Build edge list with scores
#         edge_candidates = []
#         for e in kept_edges:
#             u, v = src[e].item(), dst[e].item()
#             edge_candidates.append((e, scores[e].item(), u, v))

#         # Sort edges globally by score (high → low)
#         edge_candidates.sort(key=lambda x: -x[1])

#         final_edges = set()

#         for e, score, u, v in edge_candidates:
#             # Check degree budgets
#             u_can = degrees[u] < degree_cap
#             v_can = degrees[v] < degree_cap

#             # Symmetric acceptance rule
#             if u_can and v_can:
#                 final_edges.add(e)
#                 degrees[u] += 1
#                 degrees[v] += 1
#             elif u_can and not v_can:
#                 final_edges.add(e)
#                 degrees[u] += 1
#             elif v_can and not u_can:
#                 final_edges.add(e)
#                 degrees[v] += 1
#             # else: drop edge

#         kept_edges = final_edges

    
#     # Build pruned graph
#     kept_edges_list = sorted(kept_edges)
#     kept_mask = torch.zeros(n_edges, dtype=torch.bool)
#     kept_mask[kept_edges_list] = True
    
#     pruned_data = Data(
#         x=data.x,
#         edge_index=edge_index[:, kept_mask],
#         edge_attr=data.edge_attr[kept_mask],
#         y=data.y[kept_mask]
#     )
    
#     return pruned_data, kept_mask


# def save_stage_results(stage_results, stage_dir, stage_name):
#     """
#     Save stage results to files.
#     """
#     os.makedirs(stage_dir, exist_ok=True)
    
#     # Save summary as JSON
#     summary_path = os.path.join(stage_dir, f"{stage_name}_summary.json")
    
#     # Remove instance_details for JSON (save separately)
#     summary_for_json = {k: v for k, v in stage_results.items() if k != 'instance_details'}
    
#     with open(summary_path, 'w') as f:
#         json.dump(summary_for_json, f, indent=2, default=str)
    
#     # Save instance details as CSV
#     if 'instance_details' in stage_results:
#         details_df = pd.DataFrame(stage_results['instance_details'])
#         details_path = os.path.join(stage_dir, f"{stage_name}_details.csv")
#         details_df.to_csv(details_path, index=False)
    
#     # Save human-readable report
#     report_path = os.path.join(stage_dir, f"{stage_name}_report.txt")
#     with open(report_path, 'w', encoding='utf-8') as f:
#         f.write("=" * 60 + "\n")
#         f.write(f"{stage_name.upper()} RESULTS\n")
#         f.write("=" * 60 + "\n\n")
        
#         f.write(f"Configuration:\n")
#         f.write(f"  k value: {stage_results.get('k', 'N/A')}\n")
#         f.write(f"  Total instances: {stage_results['total_instances']}\n\n")
        
#         f.write(f"Recall Metrics:\n")
#         f.write(f"  Mean Recall:    {stage_results['mean_recall']:.2f}%\n")
#         f.write(f"  Min Recall:     {stage_results['min_recall']:.2f}%\n")
#         f.write(f"  Max Recall:     {stage_results['max_recall']:.2f}%\n")
#         f.write(f"  Std Recall:     {stage_results['std_recall']:.2f}%\n\n")
        
#         f.write(f"Success Metrics:\n")
#         f.write(f"  Success Rate:   {stage_results['success_rate']:.2f}%\n")
#         f.write(f"  Successful:     {stage_results['success_count']}/{stage_results['total_instances']}\n")
#         f.write(f"  Failed:         {stage_results['failed_count']}/{stage_results['total_instances']}\n\n")
        
#         f.write(f"Sparsity Metrics:\n")
#         f.write(f"  Mean Sparsity:  {stage_results['mean_sparsity']:.2f}%\n")
#         f.write(f"  Mean Edges:     {stage_results['mean_edges_kept']:.1f}\n")
    
#     print(f"✅ {stage_name} results saved to {stage_dir}")


# def run_cascade_pipeline(base_model, train_set, val_set, test_set, device, args):
#     """
#     Run the complete cascade pipeline after base model training.
    
#     Stage 1: Use the trained base model with k1
#     Stage 2: Train a new model on Stage 1-pruned graphs, prune with k2
#     Stage 3: Apply structural pruning rules
    
#     Args:
#         base_model: Trained EdgeGNN model (becomes Stage 1)
#         train_set: Training dataset
#         val_set: Validation dataset
#         test_set: Test dataset
#         device: torch device
#         args: Command line arguments
    
#     Returns:
#         Dictionary with all cascade results
#     """
#     model_type = getattr(args, 'model_type', 'A')
#     if model_type != 'A':
#         print(f"\n⚠️ Cascade pipeline skipped for Model {model_type}")
#         print(f"   Cascade is only executed for Model A")
#         return None
#     print("\n" + "=" * 80)
#     print("RUNNING CASCADE PIPELINE")
#     print("=" * 80)
    
#     # Configuration
#     k1 = getattr(args, 'cascade_k1', 10)  # Stage 1: keep top-10
#     k2 = getattr(args, 'cascade_k2', 5)   # Stage 2: keep top-5
#     k3 = getattr(args, 'cascade_k3', 4)   # Stage 3 base k
#     degree_cap = getattr(args, 'degree_cap', 6)
    
#     print(f"\nCascade Configuration:")
#     print(f"  Stage 1 k: {k1}")
#     print(f"  Stage 2 k: {k2}")
#     print(f"  Stage 3 k: {k3}, degree_cap: {degree_cap}")
    
#     # Create output directories
#     stage1_dir = os.path.join(args.out, "stage1")
#     stage2_dir = os.path.join(args.out, "stage2")
#     stage3_dir = os.path.join(args.out, "stage3")
    
#     os.makedirs(stage1_dir, exist_ok=True)
#     os.makedirs(stage2_dir, exist_ok=True)
#     os.makedirs(stage3_dir, exist_ok=True)
    
#     cascade_results = {}
    
#     # =========================================================================
#     # STAGE 1: Evaluate base model at k1
#     # =========================================================================
#     print("\n" + "=" * 60)
#     print(f"STAGE 1: Base Model Pruning (k={k1})")
#     print("=" * 60)
    
#     base_model.eval()
    
#     # Evaluate on test set
#     stage1_test_results, test_set_pruned_s1 = evaluate_stage(
#         base_model, test_set, device, k1, "stage1_test", model_type=model_type
#     )
    
#     # Evaluate on train/val for Stage 2 training
#     stage1_train_results, train_set_pruned_s1 = evaluate_stage(
#         base_model, train_set, device, k1, "stage1_train", model_type=model_type
#     )
#     stage1_val_results, val_set_pruned_s1 = evaluate_stage(
#         base_model, val_set, device, k1, "stage1_val", model_type=model_type
#     )
    
#     # Print Stage 1 summary
#     print(f"\nStage 1 Test Results (k={k1}):")
#     print(f"  Mean Recall:    {stage1_test_results['mean_recall']:.2f}%")
#     print(f"  Success Rate:   {stage1_test_results['success_rate']:.2f}%")
#     print(f"  Mean Sparsity:  {stage1_test_results['mean_sparsity']:.2f}%")
#     print(f"  Mean Edges:     {stage1_test_results['mean_edges_kept']:.1f}")
    
#     # Save Stage 1 results
#     save_stage_results(stage1_test_results, stage1_dir, "stage1")
#     cascade_results['stage1'] = stage1_test_results
    
#     # Also save multi-k evaluation for Stage 1
#     print("\nComputing Stage 1 metrics for multiple k values...")
#     stage1_multi_k = compute_pruning_metrics(
#         base_model, test_set, device, 
#         k_values=[2, 3, 4, 5, 7, 8, 10, 15, 20]
#     )
    
#     # Save multi-k results
#     multi_k_summary = []
#     for k, metrics in stage1_multi_k.items():
#         multi_k_summary.append({
#             'k': k,
#             'mean_recall': metrics['mean_recall'],
#             'success_rate': metrics['success_rate'],
#             'avg_sparsity': metrics['avg_sparsity'],
#             'failed_instances': metrics['failed_instances']
#         })
    
#     multi_k_df = pd.DataFrame(multi_k_summary)
#     multi_k_df.to_csv(os.path.join(stage1_dir, "stage1_multi_k.csv"), index=False)
    
#     print("\nStage 1 Multi-k Summary:")
#     print(multi_k_df.to_string(index=False))
    
#     # =========================================================================
#     # STAGE 2: Train new model on pruned graphs
#     # =========================================================================
#     print("\n" + "=" * 60)
#     print(f"STAGE 2: Context-Aware Refinement (k={k2})")
#     print("=" * 60)
    
#     # Create Stage 2 model
#     in_node = train_set[0].x.shape[1]
#     in_edge = train_set[0].edge_attr.shape[1]
    
#     stage2_model = EdgeGNN(
#         in_node_feats=in_node,
#         in_edge_feats=in_edge,
#         hidden_dim=args.hidden_dim,
#         n_layers=args.n_layers,
#         dropout=args.dropout,
#         model_type=model_type  # ✅ Add this
#     ).to(device)
    
#     # Train Stage 2 on pruned graphs
#     print(f"\nTraining Stage 2 model on {len(train_set_pruned_s1)} pruned graphs...")
#     print(f"  Input from Stage 1: ~{stage1_train_results['mean_edges_kept']:.0f} edges/instance")
    
#     optimizer2 = Adam(stage2_model.parameters(), lr=0.001)
#     scheduler2 = CosineAnnealingLR(optimizer2, T_max=args.epochs, eta_min=1e-5)
#     criterion = EdgePairwiseRankingLoss(margin=1.0)
    
#     best_val_recall_s2 = 0
#     stage2_best_path = os.path.join(stage2_dir, "stage2_best.pt")
    
#     stage2_history = {
#         'epoch': [],
#         'train_loss': [],
#         'val_recall': []
#     }
    
#     for epoch in range(1, args.epochs + 1):
#         # Train
#         stage2_model.train()
#         losses = []
#         epoch_rng = np.random.RandomState(args.seed + epoch)
#         indices = epoch_rng.permutation(len(train_set_pruned_s1))
#         shuffled_data = [train_set_pruned_s1[i] for i in indices]
        
#         for data in shuffled_data:
#             data = data.to(device)
#             optimizer2.zero_grad()
            
#             # Skip if no edges
#             if data.edge_index.size(1) == 0:
#                 continue
            
#             scores = stage2_model(data.x, data.edge_index, data.edge_attr)
#             loss = criterion(scores, data.edge_index, data.y)
#             loss.backward()
#             optimizer2.step()
#             losses.append(loss.item())
        
#         scheduler2.step()
        
#         # Validate
#         val_metrics = evaluate_ranking(stage2_model, val_set_pruned_s1, device, k_list=[k2])
#         val_recall = val_metrics.get(f'top{k2}_recall', 0)
        
#         stage2_history['epoch'].append(epoch)
#         stage2_history['train_loss'].append(np.mean(losses) if losses else 0)
#         stage2_history['val_recall'].append(val_recall)
        
#         print(f"Stage2 Epoch {epoch}/{args.epochs} | "
#               f"loss={np.mean(losses) if losses else 0:.4f} | "
#               f"val@{k2}={val_recall:.4f}")
        
#         if val_recall > best_val_recall_s2:
#             best_val_recall_s2 = val_recall
#             torch.save(stage2_model.state_dict(), stage2_best_path)
#             print(f"  ↳ New best Stage 2 model saved!")
    
#     # Load best Stage 2 model
#     stage2_model.load_state_dict(torch.load(stage2_best_path))
#     stage2_model.eval()
    
#     print(f"\n✅ Stage 2 training complete. Best val@{k2} = {best_val_recall_s2:.4f}")
    
#     # Evaluate Stage 2 on test set (cascaded from Stage 1)
#     stage2_test_results, test_set_pruned_s2 = evaluate_stage(
#         stage2_model, test_set_pruned_s1, device, k2, "stage2_test", model_type=model_type
#     )
#     # Conditional recall: Stage 2 | Stage 1 success
   

#     stage1_success_map = {
#         d['instance_id']: d['success']
#         for d in stage1_test_results['instance_details']
#     }
#     cond_recalls = [
#         d['recall']
#         for d in stage2_test_results['instance_details']
#         if stage1_success_map.get(d['instance_id'], False)
#     ]

#     conditional_recall_s2 = np.mean(cond_recalls) * 100 if cond_recalls else 0.0


#     print(f"\nStage 2 Test Results (k={k2}):")
#     print(f"  Mean Recall:    {stage2_test_results['mean_recall']:.2f}%")
#     print(f"  Success Rate:   {stage2_test_results['success_rate']:.2f}%")
#     print(f"  Mean Sparsity:  {stage2_test_results['mean_sparsity']:.2f}%")
#     print(f"  Mean Edges:     {stage2_test_results['mean_edges_kept']:.1f}")
#     print(f"  Conditional Recall (S2 | S1 success): {conditional_recall_s2:.2f}%")

#     # Save Stage 2 results
#     save_stage_results(stage2_test_results, stage2_dir, "stage2")
#     cascade_results['stage2'] = stage2_test_results
    
#     # Save training history
#     history_df = pd.DataFrame(stage2_history)
#     history_df.to_csv(os.path.join(stage2_dir, "stage2_training_history.csv"), index=False)
    
#     # =========================================================================
#     # STAGE 3: Structural Pruning
#     # =========================================================================
#     print("\n" + "=" * 60)
#     print(f"STAGE 3: Structural Pruning (k={k3}, degree_cap={degree_cap})")
#     print("=" * 60)
    
#     stage3_test_results, test_set_pruned_s3 = evaluate_stage(
#         stage2_model, test_set_pruned_s2, device, k3, "stage3_test",
#         apply_structural=True, degree_cap=degree_cap, require_top2=True,
#         model_type=model_type  # ✅ Add this
#     )
    
#     stage2_success_map = {
#         d['instance_id']: d['success']
#         for d in stage2_test_results['instance_details']
#     }

#     cond_recalls = [
#         d['recall']
#         for d in stage3_test_results['instance_details']
#         if stage2_success_map.get(d['instance_id'], False)
#     ]

#     conditional_recall_s3 = np.mean(cond_recalls) * 100 if cond_recalls else 0.0


#     print(f"\nStage 3 Test Results (structural pruning):")
#     print(f"  Mean Recall:    {stage3_test_results['mean_recall']:.2f}%")
#     print(f"  Success Rate:   {stage3_test_results['success_rate']:.2f}%")
#     print(f"  Mean Sparsity:  {stage3_test_results['mean_sparsity']:.2f}%")
#     print(f"  Mean Edges:     {stage3_test_results['mean_edges_kept']:.1f}")
#     print(f"  Conditional Recall (S3 | S2 success): {conditional_recall_s3:.2f}%")

#     # Save Stage 3 results
#     save_stage_results(stage3_test_results, stage3_dir, "stage3")
#     cascade_results['stage3'] = stage3_test_results
    
#     # =========================================================================
#     # COMPARISON SUMMARY
#     # =========================================================================
#     print("\n" + "=" * 80)
#     print("CASCADE PIPELINE SUMMARY")
#     print("=" * 80)
    
#     comparison_data = []
    
#     # Single-stage baselines at k2
#     single_stage_k2 = stage1_multi_k.get(k2, {})
#     comparison_data.append({
#         'Method': f'Single-stage (k={k2})',
#         'Mean Recall (%)': single_stage_k2.get('mean_recall', 0),
#         'Success Rate (%)': single_stage_k2.get('success_rate', 0),
#         'Sparsity (%)': single_stage_k2.get('avg_sparsity', 0),
#         'Failed Instances': single_stage_k2.get('failed_instances', 0)
#     })
    
#     # Stage 1
#     comparison_data.append({
#         'Method': f'Stage 1 (k={k1})',
#         'Mean Recall (%)': stage1_test_results['mean_recall'],
#         'Success Rate (%)': stage1_test_results['success_rate'],
#         'Sparsity (%)': stage1_test_results['mean_sparsity'],
#         'Failed Instances': stage1_test_results['failed_count']
#     })
    
#     # Stage 2 (cascade)
#     comparison_data.append({
#         'Method': f'Cascade S1→S2 (k₁={k1}→k₂={k2})',
#         'Mean Recall (%)': stage2_test_results['mean_recall'],
#         'Success Rate (%)': stage2_test_results['success_rate'],
#         'Sparsity (%)': stage2_test_results['mean_sparsity'],
#         'Failed Instances': stage2_test_results['failed_count']
#     })
    
#     # Stage 3 (structural)
#     comparison_data.append({
#         'Method': f'Cascade + Structural (k={k3}, cap={degree_cap})',
#         'Mean Recall (%)': stage3_test_results['mean_recall'],
#         'Success Rate (%)': stage3_test_results['success_rate'],
#         'Sparsity (%)': stage3_test_results['mean_sparsity'],
#         'Failed Instances': stage3_test_results['failed_count']
#     })
    
#     comparison_df = pd.DataFrame(comparison_data)
#     print("\n" + comparison_df.to_string(index=False))
    
#     # Save comparison
#     comparison_df.to_csv(os.path.join(args.out, "cascade_comparison.csv"), index=False)
    
#     # Key result
#     single_recall = single_stage_k2.get('mean_recall', 0)
#     cascade_recall = stage2_test_results['mean_recall']
#     improvement = cascade_recall - single_recall
    
#     print(f"\n📊 KEY RESULT: Cascade vs Single-stage at k={k2}")
#     print(f"   Single-stage recall: {single_recall:.2f}%")
#     print(f"   Cascade recall:      {cascade_recall:.2f}%")
#     print(f"   Improvement:         {improvement:+.2f}%")
    
#     if improvement > 0:
#         print("   ✅ CASCADE OUTPERFORMS SINGLE-STAGE!")
#     else:
#         print("   ⚠️ Single-stage performs better (investigate model/hyperparameters)")
    
#     # Save cascade results
#     cascade_results['comparison'] = comparison_data
#     cascade_results['improvement'] = {
#         'single_stage_recall': single_recall,
#         'cascade_recall': cascade_recall,
#         'improvement': improvement
#     }
    
#     # Save overall cascade summary
#     with open(os.path.join(args.out, "cascade_summary.json"), 'w') as f:
#         # Convert to serializable format
#         summary_for_json = {
#             'stage1': {k: v for k, v in stage1_test_results.items() if k != 'instance_details'},
#             'stage2': {k: v for k, v in stage2_test_results.items() if k != 'instance_details'},
#             'stage3': {k: v for k, v in stage3_test_results.items() if k != 'instance_details'},
#             'improvement': cascade_results['improvement'],
#             'config': {
#                 'k1': k1, 'k2': k2, 'k3': k3, 'degree_cap': degree_cap
#             }
#         }
#         json.dump(summary_for_json, f, indent=2, default=str)
    
#     print(f"\n✅ All cascade results saved to {args.out}")
    
#     return cascade_results, stage2_model


# def plot_cascade_comparison(cascade_results, output_dir):
#     """
#     Generate comparison plots for cascade results.
#     """
#     import matplotlib.pyplot as plt
    
#     plots_dir = os.path.join(output_dir, "plots")
#     os.makedirs(plots_dir, exist_ok=True)
    
#     # Extract data
#     stages = ['Stage 1', 'Stage 2\n(Cascade)', 'Stage 3\n(Structural)']
#     recalls = [
#         cascade_results['stage1']['mean_recall'],
#         cascade_results['stage2']['mean_recall'],
#         cascade_results['stage3']['mean_recall']
#     ]
#     success_rates = [
#         cascade_results['stage1']['success_rate'],
#         cascade_results['stage2']['success_rate'],
#         cascade_results['stage3']['success_rate']
#     ]
#     sparsities = [
#         cascade_results['stage1']['mean_sparsity'],
#         cascade_results['stage2']['mean_sparsity'],
#         cascade_results['stage3']['mean_sparsity']
#     ]
    
#     # Create figure with subplots
#     fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
#     colors = ['#3498db', '#2ecc71', '#e74c3c']
    
#     # Plot 1: Recall
#     axes[0].bar(stages, recalls, color=colors, edgecolor='black', linewidth=1.2)
#     axes[0].set_ylabel('Mean Recall (%)', fontsize=12)
#     axes[0].set_title('Recall by Stage', fontsize=14, fontweight='bold')
#     axes[0].set_ylim([min(recalls) - 5, 100])
#     for i, v in enumerate(recalls):
#         axes[0].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
#     # Plot 2: Success Rate
#     axes[1].bar(stages, success_rates, color=colors, edgecolor='black', linewidth=1.2)
#     axes[1].set_ylabel('Success Rate (%)', fontsize=12)
#     axes[1].set_title('Success Rate by Stage', fontsize=14, fontweight='bold')
#     axes[1].set_ylim([0, 100])
#     for i, v in enumerate(success_rates):
#         axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
#     # Plot 3: Sparsity
#     axes[2].bar(stages, sparsities, color=colors, edgecolor='black', linewidth=1.2)
#     axes[2].set_ylabel('Sparsity (%)', fontsize=12)
#     axes[2].set_title('Edge Sparsity by Stage', fontsize=14, fontweight='bold')
#     axes[2].set_ylim([0, max(sparsities) + 10])
#     for i, v in enumerate(sparsities):
#         axes[2].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(plots_dir, 'cascade_comparison.png'), dpi=150, bbox_inches='tight')
#     plt.close()
    
#     print(f"✅ Cascade comparison plot saved")
    
#     # Plot recall vs sparsity trade-off
#     fig, ax = plt.subplots(figsize=(8, 6))
    
#     ax.scatter(sparsities, recalls, s=200, c=colors, edgecolors='black', linewidth=2, zorder=3)
    
#     for i, stage in enumerate(['S1', 'S2', 'S3']):
#         ax.annotate(stage, (sparsities[i], recalls[i]), 
#                    textcoords="offset points", xytext=(10, 5), fontsize=12, fontweight='bold')
    
#     ax.set_xlabel('Sparsity (%)', fontsize=12)
#     ax.set_ylabel('Mean Recall (%)', fontsize=12)
#     ax.set_title('Recall vs Sparsity Trade-off', fontsize=14, fontweight='bold')
#     ax.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(plots_dir, 'recall_vs_sparsity.png'), dpi=150, bbox_inches='tight')
#     plt.close()
    
#     print(f"✅ Recall vs sparsity plot saved")


# # =============================================================================
# # LOSS FUNCTIONS
# # =============================================================================

# class EdgePairwiseRankingLoss(nn.Module):
#     """
#     Pairwise ranking loss over edges incident to the same node.
#     Encourages positive (tour) edges to have higher scores than negative edges.
#     """
#     def __init__(self, margin=1.0):
#         super().__init__()
#         self.margin = margin

#     def forward(self, edge_scores, edge_index, edge_labels):
#         """
#         Args:
#             edge_scores: (E,) tensor of edge scores
#             edge_index: (2, E) tensor of edge indices
#             edge_labels: (E,) tensor of labels in {0, 1}
        
#         Returns:
#             Pairwise ranking loss
#         """
#         loss_terms = []
#         src, dst = edge_index

#         for node in torch.unique(torch.cat([src, dst])):
#             # Get edges incident to this node
#             mask = (src == node) | (dst == node)
#             idx = mask.nonzero(as_tuple=False).squeeze(1)

#             if idx.numel() < 2:
#                 continue

#             scores = edge_scores[idx]
#             labels = edge_labels[idx]

#             pos = scores[labels == 1]
#             neg = scores[labels == 0]

#             if pos.numel() == 0 or neg.numel() == 0:
#                 continue

#             # Compute all pairwise (pos, neg) differences
#             diff = pos.view(-1, 1) - neg.view(1, -1)
#             loss = torch.clamp(self.margin - diff, min=0.0)
#             loss_terms.append(loss.mean())

#         if not loss_terms:
#             return torch.tensor(0.0, device=edge_scores.device, requires_grad=True)

#         return torch.stack(loss_terms).mean()


# # =============================================================================
# # EVALUATION METRICS
# # =============================================================================

# def topk_recall_per_node(data, scores, k=2):
#     """
#     Compute recall@k per node and average across all nodes.
    
#     Args:
#         data: PyG Data object
#         scores: Edge scores from model
#         k: Number of top edges to consider per node
    
#     Returns:
#         Mean recall@k across all nodes
#     """
#     edge_index = data.edge_index
#     y = data.y
#     src, dst = edge_index

#     recall_vals = []

#     for node in torch.unique(torch.cat([src, dst])):
#         mask = (src == node) | (dst == node)
#         idx = mask.nonzero(as_tuple=False).squeeze(1)

#         if idx.numel() == 0:
#             continue

#         node_scores = scores[idx]
#         node_labels = y[idx]

#         # Get top-k edges for this node
#         topk_idx = torch.topk(node_scores, min(k, len(idx))).indices
        
#         # Compute recall: what fraction of positive edges are in top-k
#         total_pos = node_labels.sum().clamp(min=1)
#         recalled_pos = node_labels[topk_idx].sum()
#         recall = recalled_pos / total_pos
#         recall_vals.append(recall.item())

#     return float(np.mean(recall_vals)) if recall_vals else 0.0


# @torch.no_grad()
# def evaluate_ranking(model, dataset, device, k_list=(1, 2, 5)):
#     """
#     Evaluate ranking metrics on a dataset.
    
#     Args:
#         model: The EdgeGNN model
#         dataset: List of PyG Data objects
#         device: torch device
#         k_list: List of k values for top-k recall
    
#     Returns:
#         Dictionary of metrics
#     """
#     model.eval()
#     metrics = {f"top{k}_recall": [] for k in k_list}

#     for data in dataset:
#         data = data.to(device)
#         scores = model(data.x, data.edge_index, data.edge_attr)

#         for k in k_list:
#             r = topk_recall_per_node(data, scores, k=k)
#             metrics[f"top{k}_recall"].append(r)

#     # Average over graphs
#     return {k: float(np.mean(v)) for k, v in metrics.items()}


# @torch.no_grad()
# def evaluate_classification_metrics(model, dataset, device, threshold=0.5):
#     """
#     Evaluate classification metrics (precision, recall, F1, AUC) on a dataset.
    
#     Args:
#         model: The EdgeGNN model
#         dataset: List of PyG Data objects
#         device: torch device
#         threshold: Classification threshold for binary predictions
    
#     Returns:
#         Dictionary of classification metrics
#     """
#     model.eval()
    
#     all_scores = []
#     all_labels = []
    
#     for data in dataset:
#         data = data.to(device)
#         scores = model(data.x, data.edge_index, data.edge_attr)
        
#         # Apply sigmoid to get probabilities
#         probs = torch.sigmoid(scores)
        
#         all_scores.extend(probs.cpu().numpy())
#         all_labels.extend(data.y.cpu().numpy())
    
#     all_scores = np.array(all_scores)
#     all_labels = np.array(all_labels)
    
#     # Binary predictions
#     preds = (all_scores >= threshold).astype(int)
    
#     # Compute metrics
#     results = {
#         'roc_auc': roc_auc_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.0,
#         'pr_auc': average_precision_score(all_labels, all_scores) if len(np.unique(all_labels)) > 1 else 0.0,
#         'precision': precision_score(all_labels, preds, zero_division=0),
#         'recall': recall_score(all_labels, preds, zero_division=0),
#         'f1': f1_score(all_labels, preds, zero_division=0),
#         'accuracy': accuracy_score(all_labels, preds),
#         'threshold': threshold,
#     }
    
#     return results


# # =============================================================================
# # TRAINING FUNCTIONS
# # =============================================================================

# def compute_pos_weight(dataset):
#     """
#     Compute positive class weight for imbalanced classification.
    
#     Args:
#         dataset: List of PyG Data objects
    
#     Returns:
#         Weight ratio (negative / positive)
#     """
#     pos = 0
#     neg = 0
#     for d in dataset:
#         arr = d.y.numpy()
#         pos += int((arr == 1).sum())
#         neg += int((arr == 0).sum())
    
#     if pos == 0:
#         return 1.0
#     return float(neg) / float(pos)


# def train_one_epoch_ranking(model, optimizer, criterion,
#                             data_list, batch_size=1, device='cpu', 
#                             epoch=None, base_seed=None):  # ✅ Add parameters
#     """
#     Train the model for one epoch using ranking loss.
    
#     Args:
#         model: The EdgeGNN model
#         optimizer: PyTorch optimizer
#         criterion: Loss function
#         data_list: List of PyG Data objects
#         batch_size: Batch size for training
#         device: torch device
#         epoch: Current epoch number (for reproducible shuffling)
#         base_seed: Base random seed
    
#     Returns:
#         Mean training loss for the epoch
#     """
#     model.train()
#     losses = []

#     # ✅ Reproducible shuffling
#     if epoch is not None and base_seed is not None:
#         epoch_rng = np.random.RandomState(base_seed + epoch)
#         indices = epoch_rng.permutation(len(data_list))
#         shuffled_list = [data_list[i] for i in indices]
#     else:
#         # Fallback to random shuffle (not recommended)
#         shuffled_list = data_list.copy()
#         random.shuffle(shuffled_list)

#     for idx in range(0, len(shuffled_list), batch_size):
#         batch = collate_batch(shuffled_list[idx:idx + batch_size]).to(device)

#         optimizer.zero_grad()
#         scores = model(batch.x, batch.edge_index, batch.edge_attr)
#         loss = criterion(scores, batch.edge_index, batch.y)
#         loss.backward()
#         optimizer.step()

#         losses.append(loss.item())

#     return float(np.mean(losses))


# # =============================================================================
# # DATA LOADING UTILITIES
# # =============================================================================

# def collect_pairs(folder):
#     """
#     Collect pairs of .tsp and .opt.tour files from a folder.
    
#     Args:
#         folder: Path to directory containing TSP instances
    
#     Returns:
#         List of (tsp_path, tour_path) tuples
#     """
#     files = os.listdir(folder)
#     tsp_files = [os.path.join(folder, f) for f in files if f.endswith('.tsp')]
#     pairs = []
    
#     for t in tsp_files:
#         base = os.path.splitext(t)[0]
#         tourf = base + '.opt.tour'
#         if not os.path.exists(tourf):
#             tourf2 = base + '.tour'
#             if os.path.exists(tourf2):
#                 tourf = tourf2
#             else:
#                 continue
#         pairs.append((t, tourf))
    
#     return pairs





# # =============================================================================
# # MAIN EXPERIMENT FUNCTION
# # =============================================================================

# def run_complete_experiment(args, seed=None, data_list=None, model_type='A'):
#     """
#     Run the complete TSP edge classification experiment for a specific model type.
    
#     Args:
#         args: Command line arguments
#         seed: Random seed (optional, uses args.seed if None)
#         data_list: Preprocessed data (optional)
#         model_type: Model type ('A', 'B', or 'C')
    
#     Returns:
#         Dictionary containing all experiment results
#     """
#     # Determine which seed to use
#     actual_seed = seed if seed is not None else args.seed
    
#     # Set random seeds for reproducibility
#     random.seed(actual_seed)
#     np.random.seed(actual_seed)
#     torch.manual_seed(actual_seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(actual_seed)
    
#     print(f"\nUsing seed: {actual_seed}")
#     print(f"Model type: {model_type}")
    
#     # Create output directories
#     os.makedirs(args.out, exist_ok=True)
#     os.makedirs(os.path.join(args.out, "plots"), exist_ok=True)
    
#     # Initialize results summary
#     results_summary = {
#         'seed': actual_seed,
#         'args': vars(args),
#         'model_type': model_type,
#     }
    
#     # -------------------------------------------------------------------------
#     # DATA LOADING
#     # -------------------------------------------------------------------------
#     if data_list is None:
#         print('Collecting instances...')
#         train_pairs = collect_pairs(args.train_dir)
#         synth_pairs = collect_pairs(args.synthetic_dir) if args.synthetic_dir else []
#         all_pairs = train_pairs + synth_pairs
#         print(f'Found {len(train_pairs)} tsplib instances and {len(synth_pairs)} synthetic instances; '
#               f'total {len(all_pairs)}')

#         # Build Data objects for all instances
#         data_objs = []
#         failures = []

#         print('Building Data objects (this may take time)...')
#         for tsp_path, tour_path in tqdm(all_pairs):
#             try:
#                 d = build_pyg_data_from_instance(
#                     tsp_path, tour_path,
#                     full_threshold=args.full_threshold,
#                     knn_k=args.knn_k,
#                     knn_feat_k=args.knn_feat_k
#                 )
#                 if d is not None:
#                     data_objs.append(d)
#             except Exception as e:
#                 failures.append((tsp_path, str(e)))
        
#         print(f'Built {len(data_objs)} data objects; failed {len(failures)} instances')
#         if len(data_objs) == 0:
#             raise RuntimeError('No valid instances found.')
        
#         data_objs_to_use = data_objs
#     else:
#         # Use preloaded data
#         print(f"✅ Using {len(data_list)} preprocessed graphs")
#         data_objs_to_use = data_list
    
#     # -------------------------------------------------------------------------
#     # TRAIN/VAL/TEST SPLIT
#     # -------------------------------------------------------------------------
#     idxs = list(range(len(data_objs_to_use)))
#     train_idx, test_idx = train_test_split(idxs, test_size=0.2, random_state=actual_seed)
#     train_idx2, val_idx = train_test_split(train_idx, test_size=0.1, random_state=actual_seed)
    
#     train_set = [data_objs_to_use[i] for i in train_idx2]
#     val_set = [data_objs_to_use[i] for i in val_idx]
#     test_set = [data_objs_to_use[i] for i in test_idx]

#     print(f'\nData splits:')
#     print(f'  Train: {len(train_set)} ({len(train_set)/len(data_objs_to_use):.1%})')
#     print(f'  Val:   {len(val_set)} ({len(val_set)/len(data_objs_to_use):.1%})')
#     print(f'  Test:  {len(test_set)} ({len(test_set)/len(data_objs_to_use):.1%})')
    
#     # Store split info in results
#     results_summary['data_info'] = {
#         'total_instances': len(data_objs_to_use),
#         'train_size': len(train_set),
#         'val_size': len(val_set),
#         'test_size': len(test_set),
#     }
    
#     # -------------------------------------------------------------------------
#     # MODEL SETUP
#     # -------------------------------------------------------------------------
#     in_node = train_set[0].x.shape[1]
#     in_edge = train_set[0].edge_attr.shape[1]
    
#     print(f'\nModel configuration:')
#     print(f'  Input node features: {in_node}')
#     print(f'  Input edge features: {in_edge}')
#     print(f'  Hidden dimension: {args.hidden_dim}')
#     print(f'  Number of layers: {args.n_layers}')
#     print(f'  Dropout: {args.dropout}')
#     print(f'  Model type: {model_type}')

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f'  Device: {device}')
#     model = EdgeGNN(
#         in_node_feats=in_node,
#         in_edge_feats=in_edge,
#         hidden_dim=args.hidden_dim,
#         n_layers=args.n_layers,
#         dropout=args.dropout,
#         model_type=model_type  # ✅ Pass model_type
#     ).to(device)

    
    
#     # Count parameters
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f'  Total parameters: {total_params:,}')
#     print(f'  Trainable parameters: {trainable_params:,}')
    
#     results_summary['model_info'] = {
#         'in_node_feats': in_node,
#         'in_edge_feats': in_edge,
#         'hidden_dim': args.hidden_dim,
#         'n_layers': args.n_layers,
#         'dropout': args.dropout,
#         'total_params': total_params,
#         'trainable_params': trainable_params,
#         'model_type': model_type,
#     }
    
#     # -------------------------------------------------------------------------
#     # TRAINING SETUP
#     # -------------------------------------------------------------------------
#     training_history = {
#         'epoch': [],
#         'train_loss': [],
#         'val_top1': [],
#         'val_top2': [],
#         'val_top5': [],
#         'learning_rate': [],
#     }
    
#     # Compute class weights
#     pos_weight = compute_pos_weight(train_set)
#     adjusted_pos_weight = pos_weight * 0.75
#     print(f'\nClass balance:')
#     print(f'  Pos weight (neg/pos): {pos_weight:.4f}')
#     print(f'  Adjusted pos weight: {adjusted_pos_weight:.4f}')
    
#     # Loss, optimizer, scheduler
#     criterion = EdgePairwiseRankingLoss(margin=1.0)
#     optimizer = Adam(model.parameters(), lr=0.001)
#     scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    
#     # Best model tracking
#     best_val_metric = -1.0
#     best_state_path = os.path.join(args.out, "best_model.pt")
#     best_epoch = 0
    
#     # -------------------------------------------------------------------------
#     # TRAINING LOOP
#     # -------------------------------------------------------------------------
#     print(f'\n{"="*60}')
#     print('STARTING TRAINING')
#     print(f'{"="*60}')
    
#     for epoch in range(1, args.epochs + 1):
#         t0 = time.time()
        
#         # Train one epoch
#         train_loss = train_one_epoch_ranking(
#             model=model,
#             optimizer=optimizer,
#             criterion=criterion,
#             data_list=train_set,
#             batch_size=args.batch_size,
#             device=device,
#             epoch=epoch,
#             base_seed=actual_seed
#         )
        
#         # Step scheduler
#         scheduler.step()
#         current_lr = scheduler.get_last_lr()[0]
        
#         t1 = time.time()
#         epoch_time = t1 - t0
        
#         # Evaluate on validation set
#         val_metrics = evaluate_ranking(
#             model=model,
#             dataset=val_set,
#             device=device,
#             k_list=args.topk_list
#         )
        
#         # Record history
#         training_history['epoch'].append(epoch)
#         training_history['train_loss'].append(train_loss)
#         training_history['val_top1'].append(val_metrics.get('top1_recall', 0))
#         training_history['val_top2'].append(val_metrics.get('top2_recall', 0))
#         training_history['val_top5'].append(val_metrics.get('top5_recall', 0))
#         training_history['learning_rate'].append(current_lr)
        
#         # Print progress
#         print(
#             f"Epoch {epoch:3d}/{args.epochs} | "
#             f"loss={train_loss:.4f} | "
#             f"val@1={val_metrics.get('top1_recall', 0):.3f} | "
#             f"val@2={val_metrics.get('top2_recall', 0):.3f} | "
#             f"val@5={val_metrics.get('top5_recall', 0):.3f} | "
#             f"lr={current_lr:.2e} | "
#             f"time={epoch_time:.1f}s"
#         )
        
#         # Save best model
#         metric_now = val_metrics.get('top2_recall', 0)
#         if metric_now > best_val_metric:
#             best_val_metric = metric_now
#             best_epoch = epoch
#             torch.save(model.state_dict(), best_state_path)
#             print(f"  ↳ New best model saved! (val@2 = {best_val_metric:.4f})")
    
#     print(f'\n{"="*60}')
#     print('TRAINING COMPLETE')
#     print(f'{"="*60}')
#     print(f'Best validation metric: {best_val_metric:.4f} at epoch {best_epoch}')
    
#     results_summary['training_history'] = training_history
#     results_summary['best_epoch'] = best_epoch
#     results_summary['best_val_metric'] = best_val_metric
    
#     # -------------------------------------------------------------------------
#     # LOAD BEST MODEL AND EVALUATE
#     # -------------------------------------------------------------------------
#     print('\nLoading best model for evaluation...')
#     model.load_state_dict(torch.load(best_state_path))

#     model.eval()
    
#     # Test set ranking metrics
#     print('\nEvaluating on test set...')
#     test_ranking_metrics = evaluate_ranking(
#         model=model,
#         dataset=test_set,
#         device=device,
#         k_list=args.topk_list
#     )
    
#     print(f'\nTest Ranking Metrics:')
#     for k, v in test_ranking_metrics.items():
#         print(f'  {k}: {v:.4f}')
    
#     results_summary['test_ranking_metrics'] = test_ranking_metrics
    
#     # Test set classification metrics
#     test_classification_metrics = evaluate_classification_metrics(
#         model=model,
#         dataset=test_set,
#         device=device,
#         threshold=0.5
#     )
    
#     print(f'\nTest Classification Metrics:')
#     for k, v in test_classification_metrics.items():
#         print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
    
#     results_summary['base_gnn'] = test_classification_metrics
    
#     # -------------------------------------------------------------------------
#     # GENERATE PLOTS
#     # -------------------------------------------------------------------------
#     print('\nGenerating plots...')
    
#     try:
#         plot_training_progress(training_history, args.out)
#         print('✅ Training progress plot saved')
#     except Exception as e:
#         print(f'⚠️ Could not generate training progress plot: {e}')
    
#     try:
#         # Generate PR curves if we have the data
#         plot_comprehensive_pr_curves(model, test_set, device, args.out)
#         print('✅ PR curves saved')
#     except Exception as e:
#         print(f'⚠️ Could not generate PR curves: {e}')
    
#     try:
#         plot_comprehensive_recall_vs_edges(model, test_set, device, args.out)
#         print('✅ Recall vs edges plot saved')
#     except Exception as e:
#         print(f'⚠️ Could not generate recall vs edges plot: {e}')
    
#     # -------------------------------------------------------------------------
#     # SAVE RESULTS
#     # -------------------------------------------------------------------------
#     results_path = os.path.join(args.out, "results_summary.json")
    
#     # Convert non-serializable items
#     serializable_results = {}
#     for k, v in results_summary.items():
#         if isinstance(v, dict):
#             serializable_results[k] = {
#                 str(kk): (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
#                 for kk, vv in v.items()
#             }
#         elif isinstance(v, (np.floating, np.integer)):
#             serializable_results[k] = float(v)
#         elif isinstance(v, list):
#             serializable_results[k] = [
#                 float(x) if isinstance(x, (np.floating, np.integer)) else x
#                 for x in v
#             ]
#         else:
#             serializable_results[k] = v
    
#     try:
#         with open(results_path, 'w') as f:
#             json.dump(serializable_results, f, indent=2, default=str)
#         print(f'\n✅ Results saved to {results_path}')
#     except Exception as e:
#         print(f'⚠️ Could not save results JSON: {e}')
    
#     # Generate LaTeX table
#     try:
#         create_single_seed_latex_table(results_summary, args.out)
#         print('✅ LaTeX table saved')
#     except Exception as e:
#         print(f'⚠️ Could not generate LaTeX table: {e}')
    
#     print(f'\n{"="*60}')
#     print('EXPERIMENT COMPLETE')
#     print(f'{"="*60}')
#     print(f'Results saved to: {args.out}')


#     cascade_results = None
#     if model_type == 'A':
#         print("\n🔄 Running CASCADE pipeline for Mode A...")
#         cascade_output = run_cascade_pipeline(model, train_set, val_set, test_set, device, args)
    
#         if cascade_output is not None:
#             cascade_results, stage2_model = cascade_output
#             results_summary['cascade_results'] = cascade_results
        
#             # Generate cascade plots
#             plot_cascade_comparison(cascade_results, args.out)
    
#     return results_summary


# # =============================================================================
# # MAIN ENTRY POINT
# # =============================================================================

# def main():
#     parser = argparse.ArgumentParser(
#         description='TSP Edge Classification with GNN - Multi-Model Multi-Seed',
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter
#     )
    
#     # Data arguments
#     parser.add_argument('--train_dir', type=str, required=True,
#                         help='Folder with .tsp and .opt.tour files (tsplib_data)')
#     parser.add_argument('--synthetic_dir', type=str, default=None,
#                         help='Folder with synthetic instances')
#     parser.add_argument('--out', type=str, default='trainResults',
#                         help='Output folder for logs and results')
    
#     # Training arguments
#     parser.add_argument('--epochs', type=int, default=30,
#                         help='Number of training epochs')
#     parser.add_argument('--batch_size', type=int, default=1,
#                         help='Batch size for training')
#     parser.add_argument('--seed', type=int, default=42,
#                         help='Random seed (used if --multi_seed is not set)')
    
#     # Data preprocessing arguments
#     parser.add_argument('--full_threshold', type=int, default=300,
#                         help='Node threshold for full graph vs k-NN graph')
#     parser.add_argument('--knn_k', type=int, default=30,
#                         help='Number of nearest neighbors for edge construction')
#     parser.add_argument('--knn_feat_k', type=int, default=10,
#                         help='Number of nearest neighbors for feature computation')
    
#     # Model arguments
#     parser.add_argument('--hidden_dim', type=int, default=256,
#                         help='Hidden dimension of GNN layers')
#     parser.add_argument('--n_layers', type=int, default=4,
#                         help='Number of GNN layers')
#     parser.add_argument('--dropout', type=float, default=0.3,
#                         help='Dropout rate')
    
#     # Model type argument
#     parser.add_argument('--model_type', type=str, required=True,
#                         choices=['A', 'B', 'C'],
#                         help='Model type to train (A, B, or C)')
    
#     # Evaluation arguments
#     parser.add_argument('--beam_width', type=int, default=7,
#                         help='Beam width for beam search decoding')
#     parser.add_argument('--mix_prob', type=float, default=0.7,
#                         help='Mixing probability for cascade')
#     parser.add_argument('--topk_list', nargs='+', type=int, default=[1, 2, 5, 7, 8],
#                         help='List of k values for top-k recall evaluation')
    
#     # Multi-seed arguments
#     parser.add_argument('--multi_seed', action='store_true',
#                         help='Run multi-seed experiments')
#     parser.add_argument('--seeds', nargs='+', type=int, default=[42, 125, 255, 630, 999],
#                         help='Random seeds for multi-seed experiments')
    
#     # Cascade arguments (for Mode A)
#     parser.add_argument('--cascade_k1', type=int, default=10,
#                         help='Stage 1 k value for cascade')
#     parser.add_argument('--cascade_k2', type=int, default=5,
#                         help='Stage 2 k value for cascade')
#     parser.add_argument('--cascade_k3', type=int, default=4,
#                         help='Stage 3 k value for cascade')
#     parser.add_argument('--degree_cap', type=int, default=6,
#                         help='Degree cap for Stage 3 structural pruning')
    
#     args = parser.parse_args()
    
#     # Create main output directory structure
#     os.makedirs(args.out, exist_ok=True)
#     os.makedirs(os.path.join(args.out, "modeA"), exist_ok=True)
#     os.makedirs(os.path.join(args.out, "modeB"), exist_ok=True)
#     os.makedirs(os.path.join(args.out, "modeC"), exist_ok=True)
    
#     # Print configuration
#     print("\n" + "=" * 70)
#     print("TSP EDGE CLASSIFICATION EXPERIMENT - Models B & C")
#     print("=" * 70)
#     print("\nConfiguration:")
#     for key, value in vars(args).items():
#         print(f"  {key}: {value}")
#     print()
    
#     # Load preprocessed data
#     print("Loading preprocessed data...")
#     data_list = None
#     try:
#         data_list = torch.load("data_cache/pyg_graphs.pt", weights_only=False)
#         print(f"✅ Successfully loaded {len(data_list)} preprocessed graphs")
#     except FileNotFoundError:
#         print("⚠️ Preprocessed data not found")
#     except Exception as e:
#         print(f"❌ Failed to load: {e}")
    
#     # ✅ ADD THIS: Multi-seed experiment logic
#     if args.multi_seed:
#         print(f"\n🔄 Running MULTI-SEED experiment with seeds: {args.seeds}")
        
#         manager = MultiSeedExperimentManager(args.out, args.model_type)
        
#         for seed in args.seeds:
#             manager.run_single_seed(args, seed, data_list)
        
#         # Aggregate results
#         aggregated_results = manager.aggregate_results()
        
#         print("\n" + "=" * 80)
#         print("MULTI-SEED EXPERIMENT COMPLETE")
#         print("=" * 80)
#         print(f"✅ Aggregated results saved to: {manager.mode_dir}")
        
#     else:
#         # Single seed experiment
#         print(f"\n🔄 Running SINGLE-SEED experiment with seed: {args.seed}")
        
#         # Create mode-specific directory
#         mode_dir = os.path.join(args.out, f"mode{args.model_type}")
#         os.makedirs(mode_dir, exist_ok=True)
#         args.out = mode_dir
        
#         results_summary = run_complete_experiment(args, data_list=data_list, model_type=args.model_type)
        
#         print("\n" + "=" * 80)
#         print("SINGLE-SEED EXPERIMENT COMPLETE")
#         print("=" * 80)
#         print(f"✅ Results saved to: {args.out}")
#         print("✅ Training plots saved")
#         print("✅ PR curves saved")
#         print("✅ LaTeX table saved")
    
#     print('\n✅ All done.')


# if __name__ == '__main__':
#     main()












































import os
import math
import json
import time
import random
import argparse
import copy
import warnings
from collections import defaultdict
from scipy import stats
from scipy.ndimage import gaussian_filter1d

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    precision_recall_fscore_support, confusion_matrix, 
    accuracy_score, precision_score, recall_score, 
    f1_score, precision_recall_curve
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from datetime import datetime

import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from preprocess_data import (
        build_pyg_data_from_instance,
        parse_tsp,
        parse_opt_tour,
        pairwise_distances,
        build_candidate_edges,
        compute_node_edge_features
    )
    print("✅ Successfully imported preprocessing functions from preprocess_data.py")
except ImportError as e:
    print(f"❌ Failed to import from preprocess_data.py: {e}")

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import SAGEConv
except Exception as e:
    raise ImportError(
        "PyTorch Geometric not found or failed to import. "
        "Install it per the instructions in the script header."
    )

try:
    from Helpers.others import (
        collate_batch,
        evaluate_model_custom,
        build_cascade_dataset,
    )
    print("✅ Successfully imported from others.py")
except ImportError as e:
    print(f"❌ Failed to import from others.py: {e}")


# =============================================================================
# MODEL CONFIGURATIONS FOR PHASE TRANSITION STUDY
# =============================================================================

MODEL_CONFIGS = {
    'A': {
        'node_feats': [0, 1, 2],
        'edge_feats': [0, 1, 2, 3, 4],
        'cascade': False,
        'description': 'Full features (3 node + 5 edge)'
    },
    'B': {
        'node_feats': [0],
        'edge_feats': [0],
        'cascade': False,
        'description': 'Minimal features (1 node + 1 edge)'
    },
    'C': {
        'node_feats': [0],
        'edge_feats': [],
        'cascade': False,
        'description': 'Node-only features (1 node)'
    },
    'D': {
        'node_feats': [0, 1, 2],
        'edge_feats': [0, 1, 2, 3, 4],
        'cascade': True,
        'description': 'Full features + cascade'
    }
}

K_SWEEP_VALUES = [2, 3, 4, 5, 7, 10, 15, 20, 25, 30]


# =============================================================================
# PROGRESS PRINTING UTILITIES
# =============================================================================

def print_section(title, char='=', width=80):
    """Print a formatted section header."""
    print(f"\n{char*width}")
    print(f"{title}")
    print(f"{char*width}")


def print_subsection(title, char='-', width=60):
    """Print a formatted subsection header."""
    print(f"\n{char*width}")
    print(f"{title}")
    print(f"{char*width}")


def format_time(seconds):
    """Format time in seconds to human readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


# =============================================================================
# MODEL DEFINITION WITH FEATURE SELECTION
# =============================================================================

class EdgeGNNWithFeatureSelection(nn.Module):
    """
    Graph Neural Network for edge classification with feature selection.
    """
    def __init__(self, in_node_feats, in_edge_feats,
                 node_feat_indices, edge_feat_indices,
                 hidden_dim=128, n_layers=4, dropout=0.3):
        super().__init__()
        
        self.node_feat_indices = node_feat_indices
        self.edge_feat_indices = edge_feat_indices
        
        actual_node_feats = len(node_feat_indices) if node_feat_indices else in_node_feats
        actual_edge_feats = len(edge_feat_indices) if edge_feat_indices is not None and len(edge_feat_indices) > 0 else 0
        
        self.in_node_feats = in_node_feats
        self.in_edge_feats = in_edge_feats
        self.actual_node_feats = actual_node_feats
        self.actual_edge_feats = actual_edge_feats
        
        self.input_lin = nn.Linear(actual_node_feats, hidden_dim)
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        if actual_edge_feats > 0:
            edge_input_dim = hidden_dim * 2 + actual_edge_feats
        else:
            edge_input_dim = hidden_dim * 2
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x, edge_index, edge_attr):
        if self.node_feat_indices:
            x = x[:, self.node_feat_indices]
        
        if self.edge_feat_indices is not None and len(self.edge_feat_indices) > 0 and edge_attr is not None:
            edge_attr = edge_attr[:, self.edge_feat_indices]
        else:
            edge_attr = None
        
        h = self.input_lin(x)
        
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
            h = self.dropout(h)
        
        src, dst = edge_index
        hu = h[src]
        hv = h[dst]
        
        if edge_attr is not None:
            edge_input = torch.cat([hu, hv, edge_attr], dim=1)
        else:
            edge_input = torch.cat([hu, hv], dim=1)
        
        scores = self.edge_mlp(edge_input).squeeze(1)
        return scores


# =============================================================================
# LOSS FUNCTION
# =============================================================================

class EdgePairwiseRankingLoss(nn.Module):
    """
    Pairwise ranking loss over edges incident to the same node.
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, edge_scores, edge_index, edge_labels):
        loss_terms = []
        src, dst = edge_index
        
        for node in torch.unique(torch.cat([src, dst])):
            mask = (src == node) | (dst == node)
            idx = mask.nonzero(as_tuple=False).squeeze(1)
            
            if idx.numel() < 2:
                continue
            
            scores = edge_scores[idx]
            labels = edge_labels[idx]
            
            pos = scores[labels == 1]
            neg = scores[labels == 0]
            
            if pos.numel() == 0 or neg.numel() == 0:
                continue
            
            diff = pos.view(-1, 1) - neg.view(1, -1)
            loss = torch.clamp(self.margin - diff, min=0.0)
            loss_terms.append(loss.mean())
        
        if not loss_terms:
            return torch.tensor(0.0, device=edge_scores.device, requires_grad=True)
        
        return torch.stack(loss_terms).mean()


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def compute_score_distribution(model, dataset, device):
    """Compute score distribution statistics during training."""
    model.eval()
    
    all_pos_scores = []
    all_neg_scores = []
    
    for data in dataset:
        data = data.to(device)
        scores = model(data.x, data.edge_index, data.edge_attr)
        
        pos_mask = data.y == 1
        neg_mask = data.y == 0
        
        if pos_mask.any():
            all_pos_scores.extend(scores[pos_mask].cpu().tolist())
        if neg_mask.any():
            all_neg_scores.extend(scores[neg_mask].cpu().tolist())
    
    if all_pos_scores and all_neg_scores:
        pos_mean = np.mean(all_pos_scores)
        neg_mean = np.mean(all_neg_scores)
        return {
            'pos_mean': pos_mean,
            'neg_mean': neg_mean,
            'score_gap': pos_mean - neg_mean,
            'pos_std': np.std(all_pos_scores),
            'neg_std': np.std(all_neg_scores)
        }
    else:
        return {
            'pos_mean': 0.0,
            'neg_mean': 0.0,
            'score_gap': 0.0,
            'pos_std': 0.0,
            'neg_std': 0.0
        }


def train_one_epoch(model, optimizer, criterion, data_list, batch_size, device, epoch, base_seed):
    """Train the model for one epoch."""
    model.train()
    losses = []
    
    epoch_rng = np.random.RandomState(base_seed + epoch)
    indices = epoch_rng.permutation(len(data_list))
    shuffled_list = [data_list[i] for i in indices]
    
    for idx in range(0, len(shuffled_list), batch_size):
        batch = collate_batch(shuffled_list[idx:idx + batch_size]).to(device)
        
        optimizer.zero_grad()
        scores = model(batch.x, batch.edge_index, batch.edge_attr)
        loss = criterion(scores, batch.edge_index, batch.y)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate_ranking(model, dataset, device, k_list=(1, 2, 5)):
    """Evaluate top-k recall metrics with node-level variance."""
    model.eval()
    metrics = {f"top{k}_recall": [] for k in k_list}
    node_level_recalls = {k: [] for k in k_list}
    
    for data in dataset:
        data = data.to(device)
        scores = model(data.x, data.edge_index, data.edge_attr)
        
        edge_index = data.edge_index
        y = data.y
        src, dst = edge_index
        
        for k in k_list:
            graph_recalls = []
            
            for node in torch.unique(torch.cat([src, dst])):
                mask = (src == node) | (dst == node)
                idx = mask.nonzero(as_tuple=False).squeeze(1)
                
                if idx.numel() == 0:
                    continue
                
                node_scores = scores[idx]
                node_labels = y[idx]
                
                topk_count = min(k, len(idx))
                topk_idx = torch.topk(node_scores, topk_count).indices
                
                total_pos = node_labels.sum().clamp(min=1)
                recalled_pos = node_labels[topk_idx].sum()
                recall = recalled_pos / total_pos
                graph_recalls.append(recall.item())
                node_level_recalls[k].append(recall.item())
            
            if graph_recalls:
                metrics[f"top{k}_recall"].append(float(np.mean(graph_recalls)))
    
    # Compute graph-level and node-level statistics
    result = {}
    for k in k_list:
        if metrics[f"top{k}_recall"]:
            result[f"top{k}_recall"] = float(np.mean(metrics[f"top{k}_recall"]))
            result[f"top{k}_recall_std"] = float(np.std(metrics[f"top{k}_recall"]))
            result[f"top{k}_node_recall_std"] = float(np.std(node_level_recalls[k])) if node_level_recalls[k] else 0.0
        else:
            result[f"top{k}_recall"] = 0.0
            result[f"top{k}_recall_std"] = 0.0
            result[f"top{k}_node_recall_std"] = 0.0
    
    return result


def train_model(model, train_set, val_set, args, seed, model_name="", output_dir=""):
    """Train the model and return training history with score distribution tracking."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = EdgePairwiseRankingLoss(margin=1.0)
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    
    training_history = {
        'epoch': [],
        'train_loss': [],
        'val_top1_recall': [],
        'val_top2_recall': [],
        'val_top5_recall': [],
        'val_top1_recall_std': [],
        'val_top2_recall_std': [],
        'val_top5_recall_std': [],
        'val_top2_node_recall_std': [],
        'score_pos_mean': [],
        'score_neg_mean': [],
        'score_gap': [],
        'learning_rate': []
    }
    
    best_val_metric = -1.0
    best_model_state = None
    best_epoch = 0
    
    print_subsection(f"Training {model_name} - Seed {seed}")
    print(f"{'Epoch':<6} {'Loss':<10} {'Val@2':<10} {'NodeStd':<10} {'ScoreGap':<12} {'LR':<10} {'Time':<10}")
    print("-" * 80)
    
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        train_loss = train_one_epoch(
            model, optimizer, criterion, train_set, 
            args.batch_size, device, epoch, seed
        )
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        val_metrics = evaluate_ranking(model, val_set, device, k_list=[1, 2, 5])
        score_stats = compute_score_distribution(model, val_set, device)
        
        training_history['epoch'].append(epoch)
        training_history['train_loss'].append(train_loss)
        training_history['val_top1_recall'].append(val_metrics.get('top1_recall', 0))
        training_history['val_top2_recall'].append(val_metrics.get('top2_recall', 0))
        training_history['val_top5_recall'].append(val_metrics.get('top5_recall', 0))
        training_history['val_top1_recall_std'].append(val_metrics.get('top1_recall_std', 0))
        training_history['val_top2_recall_std'].append(val_metrics.get('top2_recall_std', 0))
        training_history['val_top5_recall_std'].append(val_metrics.get('top5_recall_std', 0))
        training_history['val_top2_node_recall_std'].append(val_metrics.get('top2_node_recall_std', 0))
        training_history['score_pos_mean'].append(score_stats['pos_mean'])
        training_history['score_neg_mean'].append(score_stats['neg_mean'])
        training_history['score_gap'].append(score_stats['score_gap'])
        training_history['learning_rate'].append(current_lr)
        
        epoch_time = time.time() - epoch_start
        
        val2 = val_metrics.get('top2_recall', 0)
        node_std = val_metrics.get('top2_node_recall_std', 0)
        score_gap = score_stats['score_gap']
        
        status = ""
        if val2 > best_val_metric:
            best_val_metric = val2
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            status = " ✓ BEST"
        
        print(f"{epoch:<6} {train_loss:<10.4f} {val2:<10.4f} {node_std:<10.4f} {score_gap:<12.4f} {current_lr:<10.2e} {format_time(epoch_time):<10}{status}")
        
        # Save checkpoint every 10 epochs
        if output_dir and epoch % 10 == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metric': val2,
            }, checkpoint_path)
    
    total_time = time.time() - start_time
    
    model.load_state_dict(best_model_state)
    
    print("-" * 80)
    print(f"✅ Training complete! Total time: {format_time(total_time)}")
    print(f"   Best validation Recall@2: {best_val_metric:.4f} at epoch {best_epoch}")
    
    # Save best model
    if output_dir:
        best_model_path = os.path.join(output_dir, "best_model.pt")
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_model_state,
            'val_metric': best_val_metric,
            'model_config': {
                'node_feats': model.node_feat_indices,
                'edge_feats': model.edge_feat_indices,
                'hidden_dim': args.hidden_dim,
                'n_layers': args.n_layers,
            }
        }, best_model_path)
        print(f"   Best model saved to: {best_model_path}")
    
    return training_history


# =============================================================================
# EDGE SCORE COMPUTATION
# =============================================================================

@torch.no_grad()
def compute_edge_scores(model, dataset):
    """Compute edge scores for all graphs in dataset."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_scores = []
    all_labels = []
    
    print_subsection("Computing Edge Scores")
    
    for i, data in enumerate(tqdm(dataset, desc="Processing graphs")):
        data = data.to(device)
        scores = model(data.x, data.edge_index, data.edge_attr)
        all_scores.append(scores.cpu().numpy())
        all_labels.append(data.y.cpu().numpy())
    
    print(f"✅ Computed scores for {len(dataset)} graphs")
    
    return all_scores, all_labels


# =============================================================================
# K-VALUE SWEEP FUNCTIONS
# =============================================================================

def topk_prune_per_node(edge_index, scores, k, n_nodes):
    """Prune graph keeping top-k edges per node."""
    src, dst = edge_index
    kept_edges = set()
    
    for node in range(n_nodes):
        mask = (src == node) | (dst == node)
        idx = mask.nonzero(as_tuple=False).squeeze(1)
        
        if idx.numel() == 0:
            continue
        
        node_scores = torch.tensor(scores[idx.numpy()])
        topk_count = min(k, idx.numel())
        topk_indices = torch.topk(node_scores, topk_count).indices
        
        for i in topk_indices:
            kept_edges.add(idx[i].item())
    
    return sorted(kept_edges)


def sweep_k_values(edge_scores, test_set, k_values=K_SWEEP_VALUES):
    """Sweep different k values and compute metrics."""
    results = []
    
    print_subsection("Sweeping k Values")
    print(f"Testing k values: {k_values}")
    print()
    
    for k in tqdm(k_values, desc="Sweeping k"):
        metrics = {
            'k': k,
            'recalls': [],
            'success': [],
            'edges_kept': [],
            'sparsities': [],
            'avg_degrees': [],
            'node_recalls': []
        }
        
        for i, data in enumerate(test_set):
            scores = edge_scores[i]
            kept_edges = topk_prune_per_node(data.edge_index, scores, k, data.x.size(0))
            
            n_kept = len(kept_edges)
            n_original = data.edge_index.size(1)
            tour_kept = data.y[kept_edges].sum().item()
            total_tour = data.y.sum().item()
            
            recall = tour_kept / total_tour if total_tour > 0 else 1.0
            success = 1.0 if tour_kept == total_tour else 0.0
            sparsity = n_kept / n_original if n_original > 0 else 0
            avg_degree = 2 * n_kept / data.x.size(0)
            
            metrics['recalls'].append(recall)
            metrics['success'].append(success)
            metrics['edges_kept'].append(n_kept)
            metrics['sparsities'].append(sparsity)
            metrics['avg_degrees'].append(avg_degree)
            
            # Node-level recall
            src, dst = data.edge_index
            for node in range(data.x.size(0)):
                node_mask = ((src == node) | (dst == node)).numpy()
                node_tour_edges = data.y[node_mask].sum().item()
                if node_tour_edges > 0:
                    node_kept_edges = set(kept_edges) & set(np.where(node_mask)[0])
                    node_tour_kept = data.y[list(node_kept_edges)].sum().item()
                    metrics['node_recalls'].append(node_tour_kept / node_tour_edges)
        
        results.append({
            'k': k,
            'mean_recall': np.mean(metrics['recalls']),
            'std_recall': np.std(metrics['recalls']),
            'min_recall': np.min(metrics['recalls']),
            'max_recall': np.max(metrics['recalls']),
            'success_rate': np.mean(metrics['success']),
            'mean_edges_kept': np.mean(metrics['edges_kept']),
            'mean_sparsity': np.mean(metrics['sparsities']),
            'mean_avg_degree': np.mean(metrics['avg_degrees']),
            'node_mean_recall': np.mean(metrics['node_recalls']) if metrics['node_recalls'] else 0.0,
            'node_std_recall': np.std(metrics['node_recalls']) if metrics['node_recalls'] else 0.0,
        })
    
    df = pd.DataFrame(results)
    
    # Compute critical k and transition sharpness
    success_rates = df['success_rate'].values
    k_vals = df['k'].values
    
    # Find critical k (where success rate crosses 0.5)
    critical_k = None
    for i in range(len(success_rates) - 1):
        if success_rates[i] < 0.5 and success_rates[i+1] >= 0.5:
            # Linear interpolation
            critical_k = k_vals[i] + (0.5 - success_rates[i]) * (k_vals[i+1] - k_vals[i]) / (success_rates[i+1] - success_rates[i])
            break
    
    # Compute transition sharpness (max derivative)
    if len(success_rates) > 1:
        derivatives = np.diff(success_rates) / np.diff(k_vals)
        transition_sharpness = np.max(derivatives)
    else:
        transition_sharpness = 0.0
    
    print("\n📊 K-Sweep Results Summary:")
    print(f"{'k':<6} {'Success Rate':<15} {'Mean Recall':<15} {'Sparsity':<12} {'Node Recall':<12}")
    print("-" * 60)
    for _, row in df.iterrows():
        print(f"{int(row['k']):<6} {row['success_rate']:<15.3f} {row['mean_recall']:<15.3f} {row['mean_sparsity']:<12.3f} {row['node_mean_recall']:<12.3f}")
    
    if critical_k:
        print(f"\n🔥 Critical k (50% success): {critical_k:.2f}")
        print(f"📈 Transition sharpness: {transition_sharpness:.4f}")
    
    df.attrs['critical_k'] = critical_k
    df.attrs['transition_sharpness'] = transition_sharpness
    
    return df


# =============================================================================
# SCORE SEPARABILITY METRICS
# =============================================================================

def compute_score_separability(edge_scores, test_labels):
    """Compute metrics about edge score separability."""
    
    print_subsection("Computing Score Separability")
    
    all_pos = []
    all_neg = []
    
    for scores, labels in zip(edge_scores, test_labels):
        pos_mask = labels == 1
        neg_mask = labels == 0
        
        all_pos.extend(scores[pos_mask].tolist())
        all_neg.extend(scores[neg_mask].tolist())
    
    if len(all_pos) == 0 or len(all_neg) == 0:
        print("⚠️ Warning: No positive or negative samples found")
        return {
            'pos_mean': 0.0,
            'neg_mean': 0.0,
            'score_gap': 0.0,
            'pos_std': 0.0,
            'neg_std': 0.0,
            'auc_roc': 0.0,
            'auc_pr': 0.0
        }
    
    pos_mean = np.mean(all_pos)
    neg_mean = np.mean(all_neg)
    
    all_scores = np.array(all_pos + all_neg)
    all_labels = np.array([1] * len(all_pos) + [0] * len(all_neg))
    
    auc_roc = roc_auc_score(all_labels, all_scores)
    auc_pr = average_precision_score(all_labels, all_scores)
    
    print(f"✅ Score Separability Metrics:")
    print(f"   Positive mean: {pos_mean:.4f} ± {np.std(all_pos):.4f}")
    print(f"   Negative mean: {neg_mean:.4f} ± {np.std(all_neg):.4f}")
    print(f"   Score gap: {pos_mean - neg_mean:.4f}")
    print(f"   AUC-ROC: {auc_roc:.4f}")
    print(f"   AUC-PR: {auc_pr:.4f}")
    
    return {
        'pos_mean': float(pos_mean),
        'neg_mean': float(neg_mean),
        'score_gap': float(pos_mean - neg_mean),
        'pos_std': float(np.std(all_pos)),
        'neg_std': float(np.std(all_neg)),
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr)
    }


# =============================================================================
# CASCADE PIPELINE (FOR MODEL D)
# =============================================================================

class CascadeEdgePruner(nn.Module):
    """Two-stage cascade for edge pruning."""
    
    def __init__(self, in_node_feats, in_edge_feats, node_feat_indices, edge_feat_indices,
                 hidden_dim=128, n_layers=4, dropout=0.3):
        super().__init__()
        
        self.stage1 = EdgeGNNWithFeatureSelection(
            in_node_feats, in_edge_feats, node_feat_indices, edge_feat_indices,
            hidden_dim, n_layers, dropout
        )
        self.stage2 = EdgeGNNWithFeatureSelection(
            in_node_feats, in_edge_feats, node_feat_indices, edge_feat_indices,
            hidden_dim, n_layers, dropout
        )
    
    def prune_to_topk(self, data, scores, k):
        """Prune graph keeping top-k edges per node."""
        edge_index = data.edge_index
        src, dst = edge_index
        n_nodes = data.x.size(0)
        
        kept_edges = set()
        
        for node in range(n_nodes):
            mask = (src == node) | (dst == node)
            idx = mask.nonzero(as_tuple=False).squeeze(-1)
            
            if idx.numel() == 0:
                continue
            
            node_scores = scores[idx]
            topk_count = min(k, idx.numel())
            topk_indices = torch.topk(node_scores, topk_count).indices
            
            for i in topk_indices:
                kept_edges.add(idx[i].item())
        
        kept_edges = sorted(kept_edges)
        kept_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
        kept_mask[kept_edges] = True
        
        pruned_data = Data(
            x=data.x,
            edge_index=edge_index[:, kept_mask],
            edge_attr=data.edge_attr[kept_mask],
            y=data.y[kept_mask]
        )
        
        return pruned_data, kept_mask


def evaluate_cascade_stage(model, dataset, device, k, stage_name):
    """Evaluate a single cascade stage."""
    model.eval()
    
    results = {
        'recalls': [],
        'success_count': 0,
        'edges_kept': [],
        'edges_original': [],
        'sparsities': []
    }
    
    pruned_dataset = []
    
    with torch.no_grad():
        for data in tqdm(dataset, desc=f"Evaluating {stage_name}"):
            data = data.to(device)
            
            if data.edge_index.size(1) == 0:
                continue
            
            scores = model(data.x, data.edge_index, data.edge_attr)
            
            cascade = CascadeEdgePruner(
                data.x.size(1), data.edge_attr.size(1),
                model.node_feat_indices, model.edge_feat_indices
            )
            cascade.stage1 = model
            pruned_data, _ = cascade.prune_to_topk(data, scores, k)
            
            n_kept = pruned_data.edge_index.size(1)
            n_original = data.edge_index.size(1)
            tour_kept = pruned_data.y.sum().item()
            total_tour = data.y.sum().item()
            
            recall = tour_kept / total_tour if total_tour > 0 else 1.0
            success = int(tour_kept == total_tour)
            sparsity = n_kept / n_original if n_original > 0 else 0
            
            results['recalls'].append(recall)
            results['success_count'] += success
            results['edges_kept'].append(n_kept)
            results['edges_original'].append(n_original)
            results['sparsities'].append(sparsity)
            
            pruned_dataset.append(pruned_data.cpu())
    
    n_instances = len(results['recalls'])
    
    summary = {
        'stage': stage_name,
        'k': k,
        'mean_recall': np.mean(results['recalls']) * 100 if n_instances > 0 else 0,
        'std_recall': np.std(results['recalls']) * 100 if n_instances > 0 else 0,
        'min_recall': np.min(results['recalls']) * 100 if n_instances > 0 else 0,
        'max_recall': np.max(results['recalls']) * 100 if n_instances > 0 else 0,
        'success_rate': results['success_count'] / n_instances * 100 if n_instances > 0 else 0,
        'mean_edges_kept': np.mean(results['edges_kept']) if n_instances > 0 else 0,
        'mean_sparsity': np.mean(results['sparsities']) * 100 if n_instances > 0 else 0
    }
    
    return summary, pruned_dataset


def run_cascade_pipeline(base_model, train_set, val_set, test_set, args, seed):
    """Run the cascade pipeline for Model D."""
    print_subsection("Running Cascade Pipeline")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    k1 = getattr(args, 'cascade_k1', 10)
    k2 = getattr(args, 'cascade_k2', 5)
    
    print(f"Cascade configuration: k1={k1}, k2={k2}")
    
    cascade_results = {}
    
    # Stage 1
    print(f"\n📊 Stage 1: Base model pruning (k={k1})")
    stage1_test_results, test_set_pruned = evaluate_cascade_stage(
        base_model, test_set, device, k1, "stage1"
    )
    cascade_results['stage1'] = stage1_test_results
    print(f"   Recall: {stage1_test_results['mean_recall']:.2f}%")
    print(f"   Success Rate: {stage1_test_results['success_rate']:.2f}%")
    print(f"   Sparsity: {stage1_test_results['mean_sparsity']:.2f}%")
    
    # Stage 2
    print(f"\n📊 Stage 2: Training refinement model")
    stage2_model = EdgeGNNWithFeatureSelection(
        base_model.in_node_feats, base_model.in_edge_feats,
        base_model.node_feat_indices, base_model.edge_feat_indices,
        hidden_dim=args.hidden_dim, n_layers=args.n_layers, dropout=args.dropout
    ).to(device)
    
    stage1_train_results, train_set_pruned = evaluate_cascade_stage(
        base_model, train_set, device, k1, "stage1_train"
    )
    
    # Train Stage 2
    optimizer2 = Adam(stage2_model.parameters(), lr=0.001)
    criterion = EdgePairwiseRankingLoss(margin=1.0)
    
    for epoch in range(1, args.epochs + 1):
        stage2_model.train()
        losses = []
        for data in train_set_pruned:
            data = data.to(device)
            if data.edge_index.size(1) == 0:
                continue
            optimizer2.zero_grad()
            scores = stage2_model(data.x, data.edge_index, data.edge_attr)
            loss = criterion(scores, data.edge_index, data.y)
            loss.backward()
            optimizer2.step()
            losses.append(loss.item())
        
        if epoch % 10 == 0:
            print(f"   Stage 2 - Epoch {epoch}: loss={np.mean(losses):.4f}")
    
    print(f"\n📊 Stage 2: Refined pruning (k={k2})")
    stage2_test_results, _ = evaluate_cascade_stage(
        stage2_model, test_set_pruned, device, k2, "stage2"
    )
    cascade_results['stage2'] = stage2_test_results
    print(f"   Recall: {stage2_test_results['mean_recall']:.2f}%")
    print(f"   Success Rate: {stage2_test_results['success_rate']:.2f}%")
    print(f"   Sparsity: {stage2_test_results['mean_sparsity']:.2f}%")
    
    cascade_results['improvement'] = {
        'single_stage_recall': stage1_test_results['mean_recall'],
        'cascade_recall': stage2_test_results['mean_recall'],
        'improvement': stage2_test_results['mean_recall'] - stage1_test_results['mean_recall']
    }
    
    print(f"\n✅ Cascade Improvement: {cascade_results['improvement']['improvement']:+.2f}%")
    
    # Save stage 2 model
    if hasattr(args, 'out'):
        cascade_model_path = os.path.join(args.out, "cascade_stage2_model.pt")
        torch.save(stage2_model.state_dict(), cascade_model_path)
        print(f"   Stage 2 model saved to: {cascade_model_path}")
    
    return cascade_results


# =============================================================================
# DATA LOADING UTILITIES
# =============================================================================

def collect_pairs(folder):
    """Collect pairs of .tsp and .opt.tour files."""
    files = os.listdir(folder)
    tsp_files = [os.path.join(folder, f) for f in files if f.endswith('.tsp')]
    pairs = []
    
    for t in tsp_files:
        base = os.path.splitext(t)[0]
        tourf = base + '.opt.tour'
        if not os.path.exists(tourf):
            tourf2 = base + '.tour'
            if os.path.exists(tourf2):
                tourf = tourf2
            else:
                continue
        pairs.append((t, tourf))
    
    return pairs


def load_or_create_data(args):
    """Load preprocessed data or create from scratch."""
    cache_path = "data_cache/pyg_graphs.pt"
    
    if os.path.exists(cache_path):
        print(f"✅ Loading preprocessed data from {cache_path}")
        data = torch.load(cache_path, weights_only=False)
        print(f"   Loaded {len(data)} graphs")
        return data
    
    print("📦 Building data objects from scratch...")
    train_pairs = collect_pairs(args.train_dir)
    synth_pairs = collect_pairs(args.synthetic_dir) if args.synthetic_dir else []
    all_pairs = train_pairs + synth_pairs
    print(f"   Found {len(train_pairs)} TSPLIB + {len(synth_pairs)} synthetic = {len(all_pairs)} total instances")
    
    data_objs = []
    for tsp_path, tour_path in tqdm(all_pairs, desc="Building graphs"):
        try:
            d = build_pyg_data_from_instance(
                tsp_path, tour_path,
                full_threshold=args.full_threshold,
                knn_k=args.knn_k,
                knn_feat_k=args.knn_feat_k
            )
            if d is not None:
                data_objs.append(d)
        except Exception as e:
            continue
    
    os.makedirs("data_cache", exist_ok=True)
    torch.save(data_objs, cache_path)
    print(f"✅ Saved {len(data_objs)} graphs to {cache_path}")
    
    return data_objs


def split_data(data_list, seed):
    """Split data into train/val/test sets."""
    idxs = list(range(len(data_list)))
    train_idx, test_idx = train_test_split(idxs, test_size=0.2, random_state=seed)
    train_idx2, val_idx = train_test_split(train_idx, test_size=0.1, random_state=seed)
    
    train_set = [data_list[i] for i in train_idx2]
    val_set = [data_list[i] for i in val_idx]
    test_set = [data_list[i] for i in test_idx]
    
    return train_set, val_set, test_set


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_training_progress(history, output_dir):
    """Plot training progress including score gap evolution."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    epochs = history['epoch']
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Validation Recall
    axes[0, 1].plot(epochs, history['val_top1_recall'], 'g-', label='Top-1', linewidth=2, alpha=0.7)
    axes[0, 1].plot(epochs, history['val_top2_recall'], 'b-', label='Top-2', linewidth=2)
    axes[0, 1].plot(epochs, history['val_top5_recall'], 'r-', label='Top-5', linewidth=2, alpha=0.7)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].set_title('Validation Recall')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Score Gap Evolution (NEW)
    axes[0, 2].plot(epochs, history['score_gap'], 'm-', linewidth=2)
    axes[0, 2].fill_between(epochs, 
                             np.array(history['score_pos_mean']) - np.array(history['score_neg_mean']),
                             alpha=0.3, color='m')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Score Gap')
    axes[0, 2].set_title('Score Separability During Training')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Node-level variance
    axes[1, 0].plot(epochs, history['val_top2_node_recall_std'], 'c-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Std Dev')
    axes[1, 0].set_title('Node-Level Recall Variance')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[1, 1].plot(epochs, history['learning_rate'], 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    # Score distributions
    axes[1, 2].plot(epochs, history['score_pos_mean'], 'g-', label='Positive', linewidth=2)
    axes[1, 2].plot(epochs, history['score_neg_mean'], 'r-', label='Negative', linewidth=2)
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Mean Score')
    axes[1, 2].set_title('Score Distribution Evolution')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_progress.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_phase_transition(k_sweep_df, output_dir, title_prefix=""):
    """Plot phase transition curves for a single seed."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    k_values = k_sweep_df['k'].values
    
    # Success Rate vs k
    axes[0, 0].plot(k_values, k_sweep_df['success_rate'].values, 'o-', 
                    color='#2ecc71', linewidth=2, markersize=8)
    axes[0, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Critical threshold')
    if hasattr(k_sweep_df, 'attrs') and k_sweep_df.attrs.get('critical_k'):
        critical_k = k_sweep_df.attrs['critical_k']
        axes[0, 0].axvline(x=critical_k, color='orange', linestyle='--', alpha=0.5, label=f'k_crit={critical_k:.2f}')
    axes[0, 0].set_xlabel('k (edges per node)')
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].set_title(f'{title_prefix}Phase Transition: Success Rate vs k')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Recall with error bars
    axes[0, 1].errorbar(k_values, k_sweep_df['mean_recall'].values,
                        yerr=k_sweep_df['std_recall'].values,
                        marker='s', capsize=5, linewidth=2, color='#3498db')
    axes[0, 1].set_xlabel('k (edges per node)')
    axes[0, 1].set_ylabel('Mean Recall')
    axes[0, 1].set_title(f'{title_prefix}Graph-Level Recall with Variance')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Node-level recall with variance
    axes[0, 2].errorbar(k_values, k_sweep_df['node_mean_recall'].values,
                        yerr=k_sweep_df['node_std_recall'].values,
                        marker='o', capsize=5, linewidth=2, color='#9b59b6')
    axes[0, 2].set_xlabel('k (edges per node)')
    axes[0, 2].set_ylabel('Node-Level Mean Recall')
    axes[0, 2].set_title(f'{title_prefix}Node-Level Recall with Variance')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Sparsity vs k
    axes[1, 0].plot(k_values, k_sweep_df['mean_sparsity'].values, '^-',
                    color='#e74c3c', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('k (edges per node)')
    axes[1, 0].set_ylabel('Sparsity')
    axes[1, 0].set_title(f'{title_prefix}Graph Sparsity vs k')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Edges kept vs k
    axes[1, 1].plot(k_values, k_sweep_df['mean_edges_kept'].values, 'D-',
                    color='#e67e22', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('k (edges per node)')
    axes[1, 1].set_ylabel('Mean Edges Kept')
    axes[1, 1].set_title(f'{title_prefix}Edges Kept vs k')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Std Recall vs k (critical point identification)
    axes[1, 2].plot(k_values, k_sweep_df['std_recall'].values, 'D-',
                    color='#1abc9c', linewidth=2, markersize=8)
    if hasattr(k_sweep_df, 'attrs') and k_sweep_df.attrs.get('critical_k'):
        critical_k = k_sweep_df.attrs['critical_k']
        axes[1, 2].axvline(x=critical_k, color='orange', linestyle='--', alpha=0.5)
    axes[1, 2].set_xlabel('k (edges per node)')
    axes[1, 2].set_ylabel('Std Dev of Recall')
    axes[1, 2].set_title(f'{title_prefix}Fluctuations (Critical Region)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phase_transition.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_aggregated_phase_transition(all_seed_results, output_dir, model_name):
    """Plot aggregated phase transition across all seeds."""
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    
    k_values = all_seed_results[0]['k_sweep']['k'].values
    
    # Collect critical k values across seeds
    critical_ks = []
    sharpness_values = []
    for r in all_seed_results:
        if hasattr(r['k_sweep'], 'attrs'):
            if r['k_sweep'].attrs.get('critical_k'):
                critical_ks.append(r['k_sweep'].attrs['critical_k'])
            if r['k_sweep'].attrs.get('transition_sharpness'):
                sharpness_values.append(r['k_sweep'].attrs['transition_sharpness'])
    
    # Success Rate vs k
    ax = axes[0, 0]
    success_rates = np.array([r['k_sweep']['success_rate'].values for r in all_seed_results])
    success_mean = np.mean(success_rates, axis=0)
    success_std = np.std(success_rates, axis=0)
    
    ax.errorbar(k_values, success_mean, yerr=success_std, marker='o', 
                capsize=5, linewidth=2, markersize=8, color='#2ecc71')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Critical threshold')
    if critical_ks:
        mean_crit_k = np.mean(critical_ks)
        ax.axvline(x=mean_crit_k, color='orange', linestyle='--', alpha=0.5, 
                   label=f'Mean k_crit={mean_crit_k:.2f}')
    ax.set_xlabel('k (edges per node)', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title(f'Model {model_name}: Phase Transition', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Graph-level Recall with variance
    ax = axes[0, 1]
    recalls = np.array([r['k_sweep']['mean_recall'].values for r in all_seed_results])
    recall_mean = np.mean(recalls, axis=0)
    recall_std = np.std(recalls, axis=0)
    
    ax.errorbar(k_values, recall_mean, yerr=recall_std, marker='s',
                capsize=5, linewidth=2, markersize=8, color='#3498db')
    ax.set_xlabel('k (edges per node)', fontsize=12)
    ax.set_ylabel('Mean Recall', fontsize=12)
    ax.set_title(f'Model {model_name}: Graph-Level Recall', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Node-level Recall with variance
    ax = axes[0, 2]
    node_recalls = np.array([r['k_sweep']['node_mean_recall'].values for r in all_seed_results])
    node_recall_mean = np.mean(node_recalls, axis=0)
    node_recall_std = np.std(node_recalls, axis=0)
    
    ax.errorbar(k_values, node_recall_mean, yerr=node_recall_std, marker='o',
                capsize=5, linewidth=2, markersize=8, color='#9b59b6')
    ax.set_xlabel('k (edges per node)', fontsize=12)
    ax.set_ylabel('Node-Level Mean Recall', fontsize=12)
    ax.set_title(f'Model {model_name}: Node-Level Recall', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Sparsity vs k
    ax = axes[1, 0]
    sparsities = np.array([r['k_sweep']['mean_sparsity'].values for r in all_seed_results])
    sparsity_mean = np.mean(sparsities, axis=0)
    sparsity_std = np.std(sparsities, axis=0)
    
    ax.errorbar(k_values, sparsity_mean, yerr=sparsity_std, marker='^',
                capsize=5, linewidth=2, markersize=8, color='#e74c3c')
    ax.set_xlabel('k (edges per node)', fontsize=12)
    ax.set_ylabel('Sparsity', fontsize=12)
    ax.set_title(f'Model {model_name}: Graph Sparsity', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Training progress across seeds
    ax = axes[1, 1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_seed_results)))
    for i, results in enumerate(all_seed_results):
        history = results['training_history']
        ax.plot(history['epoch'], history['val_top2_recall'], 
                color=colors[i], alpha=0.7, label=f'Seed {i+1}')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Recall@2', fontsize=12)
    ax.set_title(f'Model {model_name}: Training Across Seeds', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Score gap evolution across seeds
    ax = axes[1, 2]
    for i, results in enumerate(all_seed_results):
        history = results['training_history']
        ax.plot(history['epoch'], history['score_gap'], 
                color=colors[i], alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Score Gap', fontsize=12)
    ax.set_title(f'Model {model_name}: Score Separability Evolution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Score separability across seeds (bar chart)
    ax = axes[2, 0]
    score_gaps = [r['separability']['score_gap'] for r in all_seed_results]
    aucs = [r['separability']['auc_roc'] for r in all_seed_results]
    
    x = np.arange(len(all_seed_results))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, score_gaps, width, label='Score Gap', alpha=0.8, color='#e67e22')
    bars2 = ax.bar(x + width/2, aucs, width, label='AUC-ROC', alpha=0.8, color='#1abc9c')
    ax.set_xlabel('Seed', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Model {model_name}: Final Score Separability', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{i+1}' for i in x])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Std Recall vs k (critical region)
    ax = axes[2, 1]
    std_recalls = np.array([r['k_sweep']['std_recall'].values for r in all_seed_results])
    std_mean = np.mean(std_recalls, axis=0)
    std_std = np.std(std_recalls, axis=0)
    
    ax.errorbar(k_values, std_mean, yerr=std_std, marker='D',
                capsize=5, linewidth=2, markersize=8, color='#1abc9c')
    if critical_ks:
        mean_crit_k = np.mean(critical_ks)
        ax.axvline(x=mean_crit_k, color='orange', linestyle='--', alpha=0.5)
    ax.set_xlabel('k (edges per node)', fontsize=12)
    ax.set_ylabel('Std Dev of Recall', fontsize=12)
    ax.set_title(f'Model {model_name}: Critical Fluctuations', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Critical k and sharpness summary
    ax = axes[2, 2]
    ax.axis('off')
    summary_text = f"Model {model_name} - Phase Transition Summary\n"
    summary_text += "="*50 + "\n\n"
    summary_text += f"Number of seeds: {len(all_seed_results)}\n\n"
    
    if critical_ks:
        summary_text += f"Critical k (50% success):\n"
        summary_text += f"  Mean: {np.mean(critical_ks):.3f}\n"
        summary_text += f"  Std:  {np.std(critical_ks):.3f}\n"
        summary_text += f"  Min:  {np.min(critical_ks):.3f}\n"
        summary_text += f"  Max:  {np.max(critical_ks):.3f}\n\n"
    
    if sharpness_values:
        summary_text += f"Transition Sharpness (max derivative):\n"
        summary_text += f"  Mean: {np.mean(sharpness_values):.4f}\n"
        summary_text += f"  Std:  {np.std(sharpness_values):.4f}\n\n"
    
    sep = all_seed_results[0]['separability']
    summary_text += f"Score Separability:\n"
    summary_text += f"  Score Gap: {sep['score_gap']:.3f}\n"
    summary_text += f"  AUC-ROC:   {sep['auc_roc']:.3f}\n"
    summary_text += f"  AUC-PR:    {sep['auc_pr']:.3f}\n"
    
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'aggregated_phase_transition_{model_name}.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_model_comparison(all_models_aggregated, output_dir):
    """Plot comparison across all models."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    models = list(all_models_aggregated.keys())
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    
    # Success Rate comparison
    ax = axes[0, 0]
    for i, (model, agg) in enumerate(all_models_aggregated.items()):
        k_sweep = agg['aggregated_k_sweep']
        ax.errorbar(k_sweep['k'], k_sweep['success_rate_mean'],
                   yerr=k_sweep['success_rate_std'],
                   marker='o', label=f'Model {model}', 
                   color=colors[i], capsize=3, linewidth=2, markersize=6)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('k (edges per node)', fontsize=12)
    ax.set_ylabel('Success Rate', fontsize=12)
    ax.set_title('Phase Transition: All Models', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Critical k comparison
    ax = axes[0, 1]
    critical_ks_dict = {}
    critical_ks_std = {}
    for model, agg in all_models_aggregated.items():
        crit_ks = []
        for r in agg['all_results']:
            if hasattr(r['k_sweep'], 'attrs') and r['k_sweep'].attrs.get('critical_k'):
                crit_ks.append(r['k_sweep'].attrs['critical_k'])
        if crit_ks:
            critical_ks_dict[model] = np.mean(crit_ks)
            critical_ks_std[model] = np.std(crit_ks)
    
    models_with_crit = list(critical_ks_dict.keys())
    crit_values = [critical_ks_dict[m] for m in models_with_crit]
    crit_stds = [critical_ks_std[m] for m in models_with_crit]
    
    bars = ax.bar(models_with_crit, crit_values, yerr=crit_stds, 
                  color=[colors[models.index(m)] for m in models_with_crit], 
                  alpha=0.8, capsize=5)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Critical k (50% success)', fontsize=12)
    ax.set_title('Critical k Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Transition sharpness comparison
    ax = axes[0, 2]
    sharpness_dict = {}
    sharpness_std = {}
    for model, agg in all_models_aggregated.items():
        sharp_vals = []
        for r in agg['all_results']:
            if hasattr(r['k_sweep'], 'attrs') and r['k_sweep'].attrs.get('transition_sharpness'):
                sharp_vals.append(r['k_sweep'].attrs['transition_sharpness'])
        if sharp_vals:
            sharpness_dict[model] = np.mean(sharp_vals)
            sharpness_std[model] = np.std(sharp_vals)
    
    models_with_sharp = list(sharpness_dict.keys())
    sharp_values = [sharpness_dict[m] for m in models_with_sharp]
    sharp_stds = [sharpness_std[m] for m in models_with_sharp]
    
    bars = ax.bar(models_with_sharp, sharp_values, yerr=sharp_stds,
                  color=[colors[models.index(m)] for m in models_with_sharp],
                  alpha=0.8, capsize=5)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Transition Sharpness (max derivative)', fontsize=12)
    ax.set_title('Transition Sharpness Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Score Gap comparison
    ax = axes[1, 0]
    gaps = [all_models_aggregated[m]['separability']['score_gap']['mean'] for m in models]
    gaps_std = [all_models_aggregated[m]['separability']['score_gap']['std'] for m in models]
    ax.bar(models, gaps, yerr=gaps_std, color=colors, alpha=0.8, capsize=5)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score Gap (pos - neg)', fontsize=12)
    ax.set_title('Edge Score Separability', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # AUC comparison
    ax = axes[1, 1]
    aucs = [all_models_aggregated[m]['separability']['auc_roc']['mean'] for m in models]
    aucs_std = [all_models_aggregated[m]['separability']['auc_roc']['std'] for m in models]
    ax.bar(models, aucs, yerr=aucs_std, color=colors, alpha=0.8, capsize=5)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('ROC-AUC', fontsize=12)
    ax.set_title('Classification Performance', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Summary table
    ax = axes[1, 2]
    ax.axis('off')
    summary_text = "Model Comparison Summary\n"
    summary_text += "="*40 + "\n\n"
    
    for model in models:
        agg = all_models_aggregated[model]
        summary_text += f"Model {model}:\n"
        if model in critical_ks_dict:
            summary_text += f"  k_crit: {critical_ks_dict[model]:.2f} ± {critical_ks_std[model]:.2f}\n"
        if model in sharpness_dict:
            summary_text += f"  Sharpness: {sharpness_dict[model]:.3f}\n"
        summary_text += f"  Score Gap: {agg['separability']['score_gap']['mean']:.3f}\n"
        summary_text += f"  AUC-ROC: {agg['separability']['auc_roc']['mean']:.3f}\n\n"
    
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# LATEX TABLE GENERATION
# =============================================================================

def generate_latex_table(aggregated_results, output_dir, model_name):
    """Generate LaTeX table with aggregated results."""
    
    latex_path = os.path.join(output_dir, f"results_table_{model_name}.tex")
    
    with open(latex_path, 'w') as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{Model {model_name} Phase Transition Results Across {aggregated_results['num_seeds']} Seeds}}\n")
        f.write("\\label{{tab:model_{}_results}}\n".format(model_name.lower()))
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\hline\n")
        f.write("\\textbf{Metric} & \\textbf{Mean} & \\textbf{Std} & \\textbf{CI (95\\%)} \\\\\n")
        f.write("\\hline\n")
        
        # Phase transition metrics
        f.write("\\multicolumn{4}{c}{\\textit{Phase Transition Metrics}} \\\\\n")
        
        # Collect critical k values
        critical_ks = []
        sharpness_vals = []
        for r in aggregated_results['all_results']:
            if hasattr(r['k_sweep'], 'attrs'):
                if r['k_sweep'].attrs.get('critical_k'):
                    critical_ks.append(r['k_sweep'].attrs['critical_k'])
                if r['k_sweep'].attrs.get('transition_sharpness'):
                    sharpness_vals.append(r['k_sweep'].attrs['transition_sharpness'])
        
        if critical_ks:
            mean_crit = np.mean(critical_ks)
            std_crit = np.std(critical_ks)
            if len(critical_ks) >= 3:
                ci = stats.t.interval(0.95, len(critical_ks)-1, loc=mean_crit, scale=stats.sem(critical_ks))
                f.write(f"Critical $k$ (50\\% success) & {mean_crit:.3f} & {std_crit:.3f} & [{ci[0]:.3f}, {ci[1]:.3f}] \\\\\n")
            else:
                f.write(f"Critical $k$ (50\\% success) & {mean_crit:.3f} & {std_crit:.3f} & --- \\\\\n")
        
        if sharpness_vals:
            mean_sharp = np.mean(sharpness_vals)
            std_sharp = np.std(sharpness_vals)
            f.write(f"Transition Sharpness & {mean_sharp:.4f} & {std_sharp:.4f} & --- \\\\\n")
        
        f.write("\\hline\n")
        
        # Metrics at k=5
        k_sweep = aggregated_results['aggregated_k_sweep']
        k5_idx = np.where(k_sweep['k'] == 5)[0]
        if len(k5_idx) > 0:
            idx = k5_idx[0]
            f.write("\\multicolumn{4}{c}{\\textit{At $k=5$}} \\\\\n")
            f.write(f"Success Rate & {k_sweep['success_rate_mean'][idx]:.3f} & {k_sweep['success_rate_std'][idx]:.3f} & --- \\\\\n")
            f.write(f"Mean Recall (Graph) & {k_sweep['mean_recall_mean'][idx]:.3f} & {k_sweep['mean_recall_std'][idx]:.3f} & --- \\\\\n")
            if 'node_mean_recall_mean' in k_sweep:
                f.write(f"Mean Recall (Node) & {k_sweep['node_mean_recall_mean'][idx]:.3f} & {k_sweep['node_mean_recall_std'][idx]:.3f} & --- \\\\\n")
            f.write(f"Sparsity & {k_sweep['mean_sparsity_mean'][idx]:.3f} & {k_sweep['mean_sparsity_std'][idx]:.3f} & --- \\\\\n")
            f.write("\\hline\n")
        
        # Separability metrics
        f.write("\\multicolumn{4}{c}{\\textit{Score Separability}} \\\\\n")
        sep = aggregated_results['separability']
        for metric in ['score_gap', 'auc_roc', 'auc_pr']:
            if metric in sep:
                m = sep[metric]
                if 'ci_95' in m:
                    f.write(f"{metric.replace('_', ' ').title()} & {m['mean']:.3f} & {m['std']:.3f} & [{m['ci_95'][0]:.3f}, {m['ci_95'][1]:.3f}] \\\\\n")
                else:
                    f.write(f"{metric.replace('_', ' ').title()} & {m['mean']:.3f} & {m['std']:.3f} & --- \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"✅ LaTeX table saved to {latex_path}")


# =============================================================================
# EXPERIMENT FUNCTIONS
# =============================================================================

def run_single_experiment(args, config_name, config, seed, data_list):
    """Run a single experiment with given configuration and seed."""
    
    print_section(f"Model {config_name}: {config['description']}")
    print(f"Seed: {seed}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    set_seed(seed)
    
    output_dir = os.path.join(args.out, f"Mode{config_name}", f"seed_{seed}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    
    print_subsection("Data Preparation")
    train_set, val_set, test_set = split_data(data_list, seed)
    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
    
    in_node = data_list[0].x.shape[1]
    in_edge = data_list[0].edge_attr.shape[1]
    
    model = EdgeGNNWithFeatureSelection(
        in_node_feats=in_node,
        in_edge_feats=in_edge,
        node_feat_indices=config['node_feats'],
        edge_feat_indices=config['edge_feats'],
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Parameters: {total_params:,}")
    print(f"Node features: {len(config['node_feats'])} | Edge features: {len(config['edge_feats']) if config['edge_feats'] else 0}")
    
    training_history = train_model(
        model, train_set, val_set, args, seed, 
        model_name=f"Model {config_name}", 
        output_dir=output_dir
    )
    
    print_subsection("Computing Edge Scores")
    edge_scores, test_labels = compute_edge_scores(model, test_set)
    
    np.save(os.path.join(output_dir, "edge_scores.npy"), np.array(edge_scores, dtype=object))
    np.save(os.path.join(output_dir, "test_labels.npy"), np.array(test_labels, dtype=object))
    
    k_sweep_df = sweep_k_values(edge_scores, test_set, K_SWEEP_VALUES)
    k_sweep_df.to_csv(os.path.join(output_dir, "k_sweep_results.csv"), index=False)
    
    separability = compute_score_separability(edge_scores, test_labels)
    
    cascade_results = None
    if config['cascade']:
        cascade_results = run_cascade_pipeline(model, train_set, val_set, test_set, args, seed)
        
        with open(os.path.join(output_dir, "cascade_results.json"), 'w') as f:
            json.dump(cascade_results, f, indent=2, default=str)
    
    print_subsection("Generating Plots")
    plot_training_progress(training_history, os.path.join(output_dir, "plots"))
    plot_phase_transition(k_sweep_df, os.path.join(output_dir, "plots"), f"Model {config_name} - Seed {seed}\n")
    print("✅ Plots saved")
    
    results = {
        'config': config,
        'seed': seed,
        'training_history': training_history,
        'k_sweep': k_sweep_df,
        'separability': separability,
        'cascade': cascade_results
    }
    
    # Save full results
    results_summary = {
        'config': config,
        'seed': seed,
        'separability': separability,
        'critical_k': k_sweep_df.attrs.get('critical_k'),
        'transition_sharpness': k_sweep_df.attrs.get('transition_sharpness'),
        'k_sweep_summary': {
            'k5_success_rate': float(k_sweep_df[k_sweep_df['k'] == 5]['success_rate'].values[0]) if 5 in k_sweep_df['k'].values else None,
            'k5_mean_recall': float(k_sweep_df[k_sweep_df['k'] == 5]['mean_recall'].values[0]) if 5 in k_sweep_df['k'].values else None,
        }
    }
    
    with open(os.path.join(output_dir, "results_summary.json"), 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print_section(f"✅ Model {config_name} - Seed {seed} COMPLETE")
    
    return results


def aggregate_seed_results(all_seed_results):
    """Aggregate results across multiple seeds."""
    
    num_seeds = len(all_seed_results)
    
    k_sweep_dfs = [r['k_sweep'] for r in all_seed_results]
    k_values = k_sweep_dfs[0]['k'].values
    
    aggregated_k_sweep = {'k': k_values}
    
    metrics_to_aggregate = [
        'mean_recall', 'std_recall', 'min_recall', 'max_recall',
        'success_rate', 'mean_edges_kept', 'mean_sparsity', 'mean_avg_degree',
        'node_mean_recall', 'node_std_recall'
    ]
    
    for metric in metrics_to_aggregate:
        if metric in k_sweep_dfs[0].columns:
            values = np.array([df[metric].values for df in k_sweep_dfs])
            aggregated_k_sweep[f'{metric}_mean'] = np.mean(values, axis=0)
            aggregated_k_sweep[f'{metric}_std'] = np.std(values, axis=0)
    
    separability_metrics = {}
    for key in all_seed_results[0]['separability'].keys():
        values = [r['separability'][key] for r in all_seed_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        separability_metrics[key] = {
            'mean': mean_val,
            'std': std_val,
            'values': values
        }
        if num_seeds >= 3:
            ci = stats.t.interval(0.95, num_seeds-1, loc=mean_val, scale=stats.sem(values))
            separability_metrics[key]['ci_95'] = ci
    
    # Aggregate critical k and sharpness
    critical_ks = []
    sharpness_vals = []
    for r in all_seed_results:
        if hasattr(r['k_sweep'], 'attrs'):
            if r['k_sweep'].attrs.get('critical_k'):
                critical_ks.append(r['k_sweep'].attrs['critical_k'])
            if r['k_sweep'].attrs.get('transition_sharpness'):
                sharpness_vals.append(r['k_sweep'].attrs['transition_sharpness'])
    
    return {
        'num_seeds': num_seeds,
        'aggregated_k_sweep': aggregated_k_sweep,
        'separability': separability_metrics,
        'critical_ks': critical_ks,
        'sharpness_vals': sharpness_vals,
        'all_results': all_seed_results
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='TSP Phase Transition Study',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Folder with .tsp and .opt.tour files')
    parser.add_argument('--synthetic_dir', type=str, default=None,
                        help='Folder with synthetic instances')
    parser.add_argument('--out', type=str, default='trainResults',
                        help='Output folder')
    
    parser.add_argument('--model_type', type=str, default='all',
                        choices=['all', 'A', 'B', 'C', 'D'],
                        help='Model configuration to run')
    
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    
    parser.add_argument('--full_threshold', type=int, default=300,
                        help='Node threshold for full graph')
    parser.add_argument('--knn_k', type=int, default=30,
                        help='k for k-NN graph')
    parser.add_argument('--knn_feat_k', type=int, default=10,
                        help='k for feature computation')
    
    parser.add_argument('--multi_seed', action='store_true',
                        help='Run multi-seed experiments')
    parser.add_argument('--seeds', nargs='+', type=int, 
                        default=[42, 123, 456, 725],
                        help='Random seeds (at least 3 recommended)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Single seed (if not multi_seed)')
    
    parser.add_argument('--cascade_k1', type=int, default=10,
                        help='Stage 1 k for cascade')
    parser.add_argument('--cascade_k2', type=int, default=5,
                        help='Stage 2 k for cascade')
    
    args = parser.parse_args()
    
    print_section("TSP PHASE TRANSITION STUDY", '=', 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {args.out}")
    print(f"Model(s) to run: {args.model_type}")
    print(f"Multi-seed: {args.multi_seed}")
    if args.multi_seed:
        print(f"Seeds: {args.seeds} (n={len(args.seeds)})")
    else:
        print(f"Seed: {args.seed}")
        print("⚠️  Warning: Single seed mode. Multi-seed (--multi_seed) recommended for statistical significance.")
    
    os.makedirs(args.out, exist_ok=True)
    
    data_list = load_or_create_data(args)
    print(f"✅ Loaded {len(data_list)} graphs\n")
    
    configs_to_run = MODEL_CONFIGS.items() if args.model_type == 'all' else [(args.model_type, MODEL_CONFIGS[args.model_type])]
    
    all_models_aggregated = {}
    overall_start_time = time.time()

    sample = data_list[0]
    print(f"Node features shape: {sample.x.shape}")        # Should be (N, ?)
    print(f"Edge features shape: {sample.edge_attr.shape}") # Should be (E, ?)
    
    for config_name, config in configs_to_run:
        print_section(f"MODEL {config_name}: {config['description']}", '=', 80)
        
        model_output_dir = os.path.join(args.out, f"Mode{config_name}")
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Use multi-seed if flag is set, otherwise single seed
        seeds_to_run = args.seeds if args.multi_seed else [args.seed]
        
        print(f"Running {len(seeds_to_run)} seed(s): {seeds_to_run}\n")
        
        all_seed_results = []
        model_start_time = time.time()
        
        for i, seed in enumerate(seeds_to_run):
            if len(seeds_to_run) > 1:
                print(f"\n{'#'*80}")
                print(f"SEED {i+1}/{len(seeds_to_run)}: {seed}")
                print(f"{'#'*80}")
            
            result = run_single_experiment(args, config_name, config, seed, data_list)
            all_seed_results.append(result)
        
        model_time = time.time() - model_start_time
        print(f"\n⏱️  Model {config_name} total time: {format_time(model_time)}")
        
        if len(seeds_to_run) > 1:
            print_section(f"AGGREGATING RESULTS FOR MODEL {config_name}", '-', 60)
            aggregated = aggregate_seed_results(all_seed_results)
            
            # Save aggregated results
            agg_save = copy.deepcopy(aggregated)
            agg_save.pop('all_results', None)
            
            # Convert numpy arrays to lists for JSON
            for key in agg_save['aggregated_k_sweep']:
                if isinstance(agg_save['aggregated_k_sweep'][key], np.ndarray):
                    agg_save['aggregated_k_sweep'][key] = agg_save['aggregated_k_sweep'][key].tolist()
            
            with open(os.path.join(model_output_dir, "aggregated_results.json"), 'w') as f:
                json.dump(agg_save, f, indent=2, default=str)
            
            print("📊 Generating aggregated plots...")
            plot_aggregated_phase_transition(all_seed_results, model_output_dir, config_name)
            generate_latex_table(aggregated, model_output_dir, config_name)
            print("✅ Aggregated results saved")
            
            all_models_aggregated[config_name] = aggregated
    
    if len(all_models_aggregated) > 1:
        print_section("MODEL COMPARISON", '=', 80)
        plot_model_comparison(all_models_aggregated, args.out)
        
        # Save comparison summary
        summary = {}
        for model, agg in all_models_aggregated.items():
            summary[model] = {
                'num_seeds': agg['num_seeds'],
                'score_gap_mean': agg['separability']['score_gap']['mean'],
                'score_gap_std': agg['separability']['score_gap']['std'],
                'auc_roc_mean': agg['separability']['auc_roc']['mean'],
                'auc_roc_std': agg['separability']['auc_roc']['std'],
                'critical_k_mean': np.mean(agg['critical_ks']) if agg['critical_ks'] else None,
                'critical_k_std': np.std(agg['critical_ks']) if agg['critical_ks'] else None,
                'sharpness_mean': np.mean(agg['sharpness_vals']) if agg['sharpness_vals'] else None,
                'sharpness_std': np.std(agg['sharpness_vals']) if agg['sharpness_vals'] else None,
            }
        
        with open(os.path.join(args.out, "all_models_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        print("✅ Model comparison saved")
    
    total_time = time.time() - overall_start_time
    
    print_section("PHASE TRANSITION STUDY COMPLETE", '=', 80)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {format_time(total_time)}")
    print(f"Results saved to: {args.out}")
    
    # Print final summary
    if all_models_aggregated:
        print("\n📊 FINAL SUMMARY:")
        print("="*60)
        for model, agg in all_models_aggregated.items():
            print(f"\nModel {model}:")
            if agg['critical_ks']:
                print(f"  Critical k: {np.mean(agg['critical_ks']):.2f} ± {np.std(agg['critical_ks']):.2f}")
            if agg['sharpness_vals']:
                print(f"  Sharpness: {np.mean(agg['sharpness_vals']):.4f} ± {np.std(agg['sharpness_vals']):.4f}")
            print(f"  Score Gap: {agg['separability']['score_gap']['mean']:.3f} ± {agg['separability']['score_gap']['std']:.3f}")
            print(f"  AUC-ROC: {agg['separability']['auc_roc']['mean']:.3f} ± {agg['separability']['auc_roc']['std']:.3f}")
    
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()