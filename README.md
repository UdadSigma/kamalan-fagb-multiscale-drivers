# Multi-scale driving mechanisms of forest aboveground biomass spatial heterogeneity in the Greater Khingan Mountains

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23681464.svg)](https://doi.org/10.5281/zenodo.23681464)

Analysis code for the manuscript *"Multi-scale driving mechanisms of forest aboveground biomass spatial heterogeneity in the Greater Khingan Mountains: an integrated Geodetector and MGWR analysis"* (submitted to *Remote Sensing*).

The repository reproduces the complete analysis pipeline for the Kamalan River watershed (northern Greater Khingan Mountains, Northeast China, ~2,207 km²):

1. **Full-pixel spatial statistics** — global Moran's I computed wall-to-wall on ~21.8 million valid 10 m pixels with a GPU-convolved 3×3 Queen contiguity (PyTorch, float64), LISA cluster mapping, and the controlled **subsampling sensitivity experiment** (random sampling at 1–50% vs. systematic lattice thinning at 2×/3×/5×) that quantifies the support-scale artefact in Moran's I.
2. **Factor screening** — significance pre-screening (Geodetector q, p < 0.05), Pearson/Spearman correlation clustering, and iterative VIF elimination (VIF < 5) that reduce 26 candidate drivers to 12 independent factors.
3. **Geodetector analysis** — single-factor q and the complete 12×12 interaction matrix (66 pairs) with enhancement-type classification.
4. **MGWR** — back-fitting MGWR with per-coefficient adaptive bandwidth search (100–2,900 neighbours, GCV criterion), benchmarked against OLS and fixed-bandwidth GWR.
5. **Figure scripts** — all main-text figures.

The 10 m TAGB response map is the published SSEL product of Zhou et al. (2026) (*Remote Sens.* 2026, 18, 1627, https://doi.org/10.3390/rs18101627). It is openly distributed on Zenodo (https://doi.org/10.5281/zenodo.23681464, SHA-256: 72163e44a040e31207c6b98ec1c358f8284acd3ebd23984d760227677ec89d04) and is not duplicated in this repository.

## Repository structure

```
├── code/
│   ├── 01_spatial_statistics/   # wall-to-wall Moran's I benchmark (Phase 0),
│   │                            # subsampling sensitivity experiment (Phase 1–2),
│   │                            # GPU LISA clustering
│   ├── 02_factor_screening/     # correlation/VIF screening (P0-1, P0-2),
│   │                            # factor discretization, Geodetector interaction
│   │                            # matrix (P0-3), MGWR + OLS/GWR benchmark (P0-4)
│   └── 03_figures/              # main-text figure scripts
├── data/                        # 50,000-point driver sample tables (see data/README.md)
├── results/                     # key result tables reproduced in the manuscript
├── README.md
├── LICENSE
├── CITATION.cff
└── requirements.txt
```

## Environment

- Python 3.8, PyTorch (CUDA; developed on an NVIDIA RTX 4090, float64), rasterio, numpy, pandas, scipy, matplotlib, seaborn, statsmodels.
- Install: `pip install -r requirements.txt`

## How to run

Input rasters (TAGB map, SRTM DEM and derived indices, climate grids, soil products, permafrost map, disturbance layers) are described in Table 1 of the manuscript; their sources are all public, and the TAGB product is openly available on Zenodo at https://doi.org/10.5281/zenodo.23681464. Absolute paths inside the scripts must be adapted to the local layout.

1. `code/01_spatial_statistics/Phase0_基准复核与耗时.py` — verify the wall-to-wall benchmark Moran's I = 0.661154.
2. `code/01_spatial_statistics/Phase1_抽样敏感性主实验.py` — run (repeatedly, it is time-budgeted) until all 300 subsampling replicates complete; outputs `results/subsampling_moran_results.csv` and `results/systematic_sampling_results.csv`.
3. `code/01_spatial_statistics/Phase2_汇总统计与出图.py` — summary statistics and Fig. 4 / Fig. 5.
4. `code/02_factor_screening/P0_1 … P0_4` — screening, interaction matrix, MGWR (inputs in `data/`).
5. `code/03_figures/*.py` — manuscript figures.

## License and citation

Code is released under the MIT License (see `LICENSE`). If you use this code, please cite the manuscript (see `CITATION.cff`) and the SSEL product paper (Zhou et al., 2026).

## Contact

Wenqiang Zhou — corresponding author of the manuscript (opwen5782815@163.com).
