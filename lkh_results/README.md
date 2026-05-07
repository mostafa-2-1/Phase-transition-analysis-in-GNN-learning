# LKH Experiment Results

## Directory Structure
```
results/
|-- full/
|-- pruned_ModeA/
|-- pruned_ModeB/
|-- pruned_ModeC/
|-- pruned_ModeD/
|-- pruned_baselines/
|   |-- nearest_k/
|   |-- random_k/
|   `-- delaunay_k/
`-- results_summary.csv
```

## Running Experiments
1. Run all: `parameters/run_all.bat`
2. Run individually via the batch files in `parameters/`

## Parameters
- Time limit per run: 3600 seconds
- LKH executable: {LKH_EXE_PATH}
