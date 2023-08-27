# GIWAXS_Tools

  A data processor for GIWAXS images that have been mapped into q-space using WAXStool and exported from Igor.
  
  The input .tif files should be reduced from the detector .tif using WAXSTools and exported from Igor using
  ImageSave or via Data > Save Waves > Save Image using a TIFF format with a 32-bit float/sample Sample Depth.
  The resulting qzqxy image will include the beam centering and sample-to-detector distance corrections done in Igor using Nika,
  as well as the missing-wedge correction done by WAXSTools, provided that the correct q_max used in WAXSTools
  is specified upon initializing GIWAXSDataProcessor.
