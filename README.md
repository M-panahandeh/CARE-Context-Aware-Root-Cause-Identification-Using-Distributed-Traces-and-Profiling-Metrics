# CARE-Context Aware Root Cause Identification Using Distributed Traces and Profiling Metrics
This repository contains scripts and resources for **CARE** Context-Aware Root Cause Identification Using Distributed Traces and Profiling Metrics), along with adjusted implementations of similar approaches for comparative analysis. The included methods are:
**CARE (Context-Aware Root Cause Identification)**: Our proposed approach, with a sub-directory `GNN_Parallelization` representing the GNN community detection component of CARE.
**Regular Spectrum Analysis**: Traditional root cause identification using spectrum analysis.
**MicroRank**: Adapted for our dataset and double root cause detection while retaining its original methodology. [MicroRank GitHub Repository](https://github.com/IntelligentDDS/MicroRank/tree/main)
**TraceRCA**: Customized to suit our experiments, based on its original approach. [TraceRCA GitHub Repository]([https://github.com/IntelligentDDS/MicroRank/tree/main](https://github.com/NetManAIOps/TraceRCA?tab=readme-ov-file))
**Second Dataset**: This folder includes adjusted scripts of each approach for the second dataset.

**Folder Structure**
Each approach is organized in a separate folder containing:
Scripts: Necessary scripts for feature selection, anomaly detection, and both single and multi-root-cause localization.
Artifacts: Generated artifacts and experimental results for CARE and baseline methods, provided for reproducibility and reference.

**Executing Each Approach**
To execute an approach, navigate to the respective folder. Each contains scripts for feature selection, anomaly detection, and root cause localization for both single and multi-root causes. To perform root cause localization, simply call the main function of the corresponding localization approach. Each script will generate a list of root cause candidates based on the provided dataset.

**Dataset**
The dataset used for these comparisons can be found at the following link.[Dataset Link](https://onedrive.live.com/?authkey=%21AAUszKmCUiodw94&id=BF4BCE76A3C5838D%21108&cid=BF4BCE76A3C5838D)
