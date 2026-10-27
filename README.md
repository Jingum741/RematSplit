# RematSplit: Recompute or Transmit Activation Management for Memory Constrained Split Fine Tuning of LLMs on Mobile Devices

Code, data summaries, and results for the RematSplit manuscript (under review). RematSplit plans, for every transformer unit on the device side of a split LoRA fine tuning pipeline, whether its activations are stored (S), recomputed in the backward pass (R), stored in int8 (Q), or transmitted to the server (T or TQ), so that the device stays inside a memory budget while total step time and energy are minimised over a measured 5G link.

## Contents

1. Overview
2. What is measured, derived, and assumed
3. Repository layout
4. Installation
5. Reproducing the results
6. Main results
7. Figures
8. Data sources and licences
9. Limitations
10. Citation and licence

## Overview

Mistral 7B (`mistralai/Mistral-7B-v0.1`) is split after layer 8, giving 16 device side units (attention and MLP per layer). For each sequence length L, device memory budget M ∈ {6, 8, 12, 16} GB, device class (Jetson Orin, smartphone), and link scale (1, 10, 50, 200 × the measured 5G trace), the planner selects a per unit action and is compared against Store (SplitLoRA), Checkpoint all, Offload all, and a Greedy planner, plus four ablations (without latency hiding, without transmit, without int8, Store/Recompute only). An online Hedge policy adapts the plan to the live downlink rate and is compared with EWMA, static, best fixed, and oracle policies.

## What is measured, derived, and assumed

| Category | Quantity |
|---|---|
| Measured | Per unit resident bytes, forward and backward time, and peak memory for each action (`unit_profile.csv`); server step time per L (`server_profile.csv`); predicted versus measured peak and resident memory (`memory_validation.csv`); gradient error and cosine similarity of every action against Store (`gradient_equivalence.csv`); training loss and perplexity before and after fine tuning |
| Real | LongAlpaca 12k sequence lengths (12,000 samples); uplink, downlink, and RTT from the public 5G production dataset (43,208 samples) |
| Derived | Step time, energy, and maximum feasible sequence length for every plan (`plan_sweep.csv`, `max_seq_len.csv`) |
| Assumed | Device class power and compute scaling for Jetson Orin and smartphone, link scale factors |

## Repository layout

```
RematSplit/
├── RematSplit.ipynb               # end to end Colab pipeline (add before publishing)
├── results/
│   ├── settings.json              # model, split layer, budgets, figure settings
│   ├── unit_profile.csv           # measured cost of S, R, Q, T per unit and L
│   ├── server_profile.csv         # measured server step time per L
│   ├── memory_validation.csv      # predicted vs measured peak and resident memory
│   ├── gradient_equivalence.csv   # gradient error of each action vs Store
│   ├── training_log.csv           # per step loss, bytes, and timing per method
│   ├── perplexity.csv             # perplexity before and after fine tuning
│   ├── plan_sweep.csv             # every method x device x link x budget x L
│   ├── max_seq_len.csv            # largest feasible L per method and budget
│   ├── headline_numbers.csv       # energy savings and length gains
│   ├── online_policy.csv          # Hedge, EWMA, static, oracle over 2000 steps
│   ├── dataset_length_stats.csv   # LongAlpaca 12k length statistics
│   └── trace_stats.csv            # 5G trace statistics
├── figures/                       # result figures (PNG)
├── requirements.txt
└── LICENSE
```

## Installation

```
git clone https://github.com/Jingum741/RematSplit.git
cd RematSplit
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The profiling steps need a CUDA GPU (the archived results were produced on a Colab A100) and a Hugging Face token with access to Mistral 7B.

## Reproducing the results

Open `RematSplit.ipynb` and run the cells from top to bottom. All CSV files in `results/` are the archived outputs of that run, so tables and figures can be regenerated from them without repeating the GPU profiling.

## Main results

Energy saving of RematSplit and maximum sequence length gain (from `results/headline_numbers.csv`).

| Device | Link scale | Saving vs Checkpoint all (%) | Saving vs Greedy (%) | Saving vs Offload all (%) | Max L gain vs Store | Max L gain vs Checkpoint all |
|---|---|---|---|---|---|---|
| Jetson Orin | 1 | 0.29 | 0.10 | 99.63 | 9.33× | 1.46× |
| Jetson Orin | 50 | 2.20 | 0.75 | 97.64 | 9.33× | 1.46× |
| Jetson Orin | 200 | 2.45 | 0.84 | 91.87 | 9.33× | 1.46× |
| Smartphone | 1 | 1.09 | 0.36 | 99.47 | 9.33× | 1.46× |
| Smartphone | 50 | 2.74 | 0.91 | 90.26 | 9.33× | 1.46× |
| Smartphone | 200 | 2.80 | 0.93 | 68.11 | 9.33× | 1.46× |

Perplexity (from `results/perplexity.csv`): before fine tuning 4.344; after fine tuning 4.147 (Store), 4.110 (Checkpoint all), 4.130 (RematSplit).

Gradient equivalence against Store (from `results/gradient_equivalence.csv`): cosine similarity 0.99995 for Recompute and Transmit, 0.99962 for int8 store and int8 transmit.

## Figures

| File | Content |
|---|---|
| `fig_memory_validation.png` | predicted versus measured device memory |
| `fig_max_seqlen.png` | maximum feasible sequence length per memory budget |
| `fig_plan_modes.png` | per unit action chosen by the planner |
| `fig_time_breakdown.png` | compute, recompute, link, and wait time |
| `fig_energy_vs_len.png` | energy per step versus sequence length |
| `fig_link_sensitivity.png` | sensitivity to link scale |
| `fig_online_regret.png` | online Hedge policy regret on the 5G trace |
| `fig_loss_curves.png` | training loss per method |
| `fig_ablation.png` | ablation study |

## Data sources and licences

1. Mistral 7B v0.1 (`mistralai/Mistral-7B-v0.1`), Apache 2.0.
2. LongAlpaca 12k (`Yukang/LongAlpaca-12k`), Chen et al., ICLR 2024. Only length statistics are included here; no text is redistributed.
3. 5G production dataset, University College Cork Mobile and Internet Systems Laboratory (uccmisl), Raca et al., ACM MMSys 2020. Only summary statistics are included.

## Limitations

Device time and energy for Jetson Orin and the smartphone class are scaled from profiling on a data centre GPU rather than measured on the handsets. The link is replayed from a recorded trace, not a live network. Results cover one model and one split point.

## Citation and licence

A citation will be added once the manuscript is published. The code is released under the MIT licence (see `LICENSE`). Data retain the licences of their original sources.
