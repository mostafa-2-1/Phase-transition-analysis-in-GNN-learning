Overview This project provides a systematic empirical analysis of how graph sparsification affects heuristic solver reliability in large-scale Euclidean Traveling Salesman Problems (**TSP**).

Rather than proposing a new solver, this work studies how different pruning strategies reshape the structural landscape on which a fixed state-of-the-art heuristic (**LKH**-3.0.13) operates.

We compare three pruning strategies:
- Learned pruning via a pretrained **GNN**
- Geometric nearest-k pruning
- Random-k pruning
Across fourteen benchmark instances and a controlled sparsity sweep, we model solver success probability as a function of realized average graph degree and identify phase transitions in navigability.

### Core Research Question

Is the observed success of learned pruning a consequence of structural information preservation, or merely a byproduct of sparsity?

### Experimental Scope

14 large-scale **TSPLIB** and **VLSI** benchmark instances 15 sparsity levels per instance 3 pruning strategies 10 **LKH** seeds per reduced graph

Total:
- **630** reduced graphs
- **6,300** solver runs

### Methodological Pipeline

The experimental pipeline consists of four stages:

## Reduced Graph Construction

1. **GNN-based** top-k edge retention  
2. **Nearest-k** baseline  
3. **Random-k** baseline  

## Structural Measurement

1. Realized average degree  
2. Connectivity  
3. Tour recall  
4. Edge-length statistics  

## Solver Evaluation

1. **LKH-3.0.13**  
2. 10 independent seeds  
3. Success defined as relative gap ≤ 0.7%  

## Statistical Modeling

1. Logistic regression of P(success) vs average degree  
2. Bootstrap confidence intervals  
3. Connectivity decomposition  

Each reduced graph (instance × method × k) is treated as an independent structural observation.

### Setup Instructions

## Clone the repository

```bash
git clone https://github.com/mostafa-2-1/ML-Guided-Quantum-BnB.git
```

## Create a virtual environment

```bash
python -m venv venv
```

## Activate the environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing:

```bash
pipreqs . --force --encoding=utf-8
```

## External Dependency

This project requires:

- **LKH-3.0.13**

Download it from the official LKH repository and compile it locally.

⚠️ Solver paths must be configured inside the experiment scripts.

### Reproducing Results

## 1. Generate synthetic instances

`scripts/synthetic_tsp.py`

## 2. Process data

`scripts/preprocess_tsp.py`

## 3. Train the GNN model

`newTrain.py`

## 4. Prune graphs

1. **GNN** – `gnn_pruning.py`
2. **Baseline** – `baseline_pruning.py`

Results are generated in
- pruned_graphs_k (**GNN**)
- pruned_graphs_k_baseline (baselines)

## 5. Generate parameters for LKH

1. **GNN** - `parameters gnn_parameters.py`
2. **Baseline** - `parameters baseline_parameters.py`

Results are generated in
- parameters (**GNN**)
- baseline_parameters (baselines)

## 6. Solve reduced graphs using LKH

1. **GNN** - `solver endSolvers/exact_gnn.py` 
2. **Baseline** - `solver endSolvers/exact_baseline.py`

Results are generated in
- solver_results (**GNN**)
- baseline_solver_results (baselines)

## 7. Figures generation (success vs avg degree, gap vs k, success probability overlay)

`generate_comparison_plots.py`

Results are saved in
- structural_analysis_comparison
    - success_vs_avg_degree_overlay.png
    -  gap_vs_k_gnn_vs_knn.png
    - success_avg_degree_logistic_overlay.png

## 8. Logistic parameters table and critical k distribution 

`generate_combined_tables.py`

Results are saved in
- table_comparison
    - critical_k_distribution.png
    - logistic_parameters_table.png

## 9. Master logistic phase transition fit and logistic regression covariate table

`generate_master_phase_transition_plot.py`

Results are saved in
- structural_analysis_comparison
    - master_phase_transition_comparison.png
    - logistic_regression_covariate_table.png

## 10. Connectivity decomposition figure

`connectivity_decomposition_analysis.py`

Results are saved in
- structural_analysis_comparison_connectivity
  - connectivity_decomposition.png

Statistical and quantitative outputs are generated with each file to provide more details.
