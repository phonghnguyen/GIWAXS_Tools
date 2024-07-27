# GIWAXS_Tools

This code is intended to facilitate the analysis of grazing incidence X-ray scattering detector images. It assumes that X-ray scattering detector images can be imported into Python as an image array. The user should know the grazing incidence angle and X-ray energy beforehand. Values for sample-to-detector distance and beam center can be calibrated using Nika's Beam center and Geometry cor. modules. The code uses xarray data arrays.  The mapping is based on Stribeck, N.; Nöchel, U. Direct Mapping of Fiber Diffraction Patterns into Reciprocal Space. J Appl Cryst 2009, 42 (2), 295–301. https://doi.org/10.1107/S0021889809004713. Jacobian correction of intensity for the remapping process is also done by default. Corrections for absorbance and polarization have not been implemented. 

Methods in GIWAXSDataProcessor can be used to map the detector image to q-space (generate qz-qxy plots), cake (generate chi-q plots), and apply sin(chi) correction for more quantitative analysis of angle-resolved populations. 

Methods in CrystalIndexingAnalyzer can be used for rudimentary analysis and indexing of different crystal structures and space groups. 

An example Jupyter notebook is included to demonstrate use of both GIWAXSDataProcessor and CrystalIndexingAnalyzer.