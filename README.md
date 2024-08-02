# GIWAXS Tools

**GIWAXS Tools** is a Python library designed to facilitate the analysis of Grazing Incidence Wide-Angle X-ray Scattering (GIWAXS) data. It provides methods for mapping X-ray scattering detector images into $q$-space and $χ$ - $q$ space. Similar to `PyHyperScattering`, this library uses `xarray` for convenient data handling. This project was inspired by WAXSTools, developed in Igor by the Toney Group (DOI 10.1021/acs.chemmater.7b00067). 

## Features

- **Image to $q$-Space Mapping**: Convert detector images into $q$-space based on the work of Stribeck and Nöchel (2009) (DOI 10.1107/S0021889809004713). Correction of intensity is applied by default to account for geometric distortions in the transformation. The equations are directly translatable from the reference and users are encouraged to assess the correctness and applicability of these methods. 
- **Polar Coordinate Transformation (Caking)**: Transform $q$-space data into polar coordinates ($χ$ - $q$ plots) for analysis of angular distributions (pole figure analysis).
- **Crystal Structure Analysis**: Rudimentary tools for analyzing and indexing various crystal structures and space groups.
- **|sin(χ)| Correction**: Intensity normalization for limited sampling of the $xy$ plane.
- **Calculation Methods**: Includes static methods for calculating penetration depth, refractive index, and critical angle based on given parameters.

**Note**: Corrections for absorbance and polarization effects are not currently implemented.

## Prerequisites

Analysis of data requires knowing:
- The grazing incidence angle.
- The X-ray energy.
- Sample-to-detector distance and beam center, which can be calibrated using tools such as Nika's Beam Center and Geometry Correction modules.

Please see the example_usage notebook for example usage.
