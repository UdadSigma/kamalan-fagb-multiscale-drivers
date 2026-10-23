# Data

This directory contains the two point-sample tables used by the factor-screening and
Geodetector/MGWR scripts:

- `Kamalan_Drivers_Continuous_50k_with_XY.csv` — 50,000 sample points with X/Y
  coordinates and the continuous values of the 26 candidate driving factors.
- `Kamalan_Drivers_Discretized_50k_Fixed.csv` — the same points after ecological
  discretization of each factor into strata (used by the Geodetector scripts;
  identical to the stratification of Table 4 in the manuscript).

The underlying rasters (SRTM DEM and derivatives, climate grids, soil products,
permafrost map, land-cover, road/settlement/fire vectors) are all publicly
available from the sources listed in Table 1 of the manuscript and are not
redistributed here. The 10 m FAGB response map is the published SSEL product
(Zhou et al., 2026, https://doi.org/10.3390/rs18101627) and is available from
the corresponding author upon reasonable request.
