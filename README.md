# ln-pathfinding-module

## Project Overview

`ln-pathfinding-module` is a research-focused module dedicated to analyzing and improving pathfinding strategies within the Bitcoin Lightning Network. It offers:

- Efficient algorithms implemented in **Go** (90.6% of codebase)
- Rich data analysis and visualization tools in **Python** (9.4%)
- Data insights and results based on realistic snapshots and comprehensive simulations

This module aims to aid researchers and developers in understanding payment routing behaviors and optimizing path selection in the Lightning Network.

---

## Data & Visualizations

- `LN_snapshot.csv` - Realistic snapshots of the Lightning Network graph
- `comprehensive_results.csv` - Detailed routing outcomes across scenarios  
- `summary_metrics_table.csv` - High-level performance summaries  
- `poster_fee.png`, `poster_fee_distribution.png`, `poster_success_rate.png` - Visual depictions of findings, including fee distribution, success rates, and more  

These artifacts support both exploratory analysis and presentation needs.

---

## Core Components

### Go Module 
`test.go` - Hosts the pathfinding logic and core algorithms—ideal for high-performance simulation and integration into routing systems.

### Visualization Script 
`visualization.py` - Builds visualizations from data sources like CSVs to reveal trends such as fee behavior and routing success distributions.

---

## Quickstart Guide

1. Clone the repo
   ```bash
   git clone https://github.com/rssalwekar/ln-pathfinding-module.git
   cd ln-pathfinding-module
   ```
2. Run the Go module
   ```bash
   go run test.go
   ```
3. Generate visualizations
   ```bash
   python visualization.py
   ```
4. Explore the outputs
   - View metrics in `summary_metrics_table.csv`

---

## Research Highlights

- Leverages a dual-language approach—**Go** for efficient algorithmic execution and **Python** for analytics and visuals
- Targets practical challenges like minimizing fees and maximizing transaction success rates in Lightning Network routing

---

## Contribution & Collaboration

Contributions are welcome! Whether it’s enhancing algorithms, adding new visualizations, or improving documentation, feel free to fork and open pull requests.

---

## Contact & Attribution

For questions or research collaboration, reach out to Rohan Salwekar (rssalwekar@gmail.com).
