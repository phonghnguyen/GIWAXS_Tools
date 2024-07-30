# GIWAXS Tools

**GIWAXS Tools** is a Python library designed to facilitate the analysis of Grazing Incidence Wide-Angle X-ray Scattering (GIWAXS) data. It provides methods for mapping X-ray scattering detector images into q-space and polar coordinate representations, as well as tools to assist in peak indexing. This project was inspired by WAXSTools, developed in Igor by the Toney Group (DOI 10.1021/acs.chemmater.7b00067). 

## Features

- **Image to Q-Space Mapping**: Convert detector images into q-space (qz-qxy plots) using established mathematical transformations based on the work of Stribeck and Nöchel (2009) (DOI 10.1107/S0021889809004713). Jacobian correction of intensity for the remapping process is applied by default to account for geometric distortions in the transformation. The equations are directly translatable from the reference and users are encouraged to assess the correctness and applicability of these methods. 
- **Polar Coordinate Transformation (Caking)**: Transform q-space data into polar coordinates (chi-q plots) for analysis of angular distributions (pole figure analysis).
- **Crystal Structure Analysis**: Rudimentary tools for analyzing and indexing various crystal structures and space groups.
- **sin(χ) Correction**: Corrections for variations in intensity as a function of chi, achieved by normalizing by sin(χ).

**Note**: Corrections for absorbance and polarization effects are not currently implemented.

## Prerequisites

Analysis of data requires knowing:
- The grazing incidence angle.
- The X-ray energy.
- Sample-to-detector distance and beam center, which can be calibrated using tools such as Nika's Beam Center and Geometry Correction modules.

The code utilizes `xarray` for data handling. Please see the example_usage notebook for intended usage.
