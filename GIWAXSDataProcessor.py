# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 12:19:40 2023

@author: Phong
"""

import numpy as np
import fabio
import xarray as xr
import dask.array as da
import skimage.transform
from scipy.ndimage import label
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.colors import LogNorm

class GIWAXSDataProcessor:
    def __init__(self, q_max):
        """
        A data processor for GIWAXS images that have been mapped into q-space using WAXStool and exported from Igor.
        
        The input .tif files should be reduced from the detector .tif using WAXSTools and exported from Igor using
        ImageSave or via Data > Save Waves > Save Image using a TIFF format with a 32-bit float/sample Sample Depth.
        The resulting qzqxy image will include the beam centering and sample-to-detector distance corrections done in Igor,
        as well as the missing-wedge correction done by WAXSTools, provided that the correct q_max used in WAXSTools
        is specified upon initializing this class.
        
        Parameters:
        q_max (float): The upper q-range value specified during the WAXStool image mapping.
        """
        self.q_max = q_max
    
    def automask(self, image, max_region_size=50, threshold_value=0.25):
        """
        This function generates an automatic mask for an input image. This mask is primarily used to hide regions 
        from the image that have a low-intensity value (less than or equal to a threshold value) and are larger 
        than a defined maximum size. The latter is particularly useful when you want to ignore non-contributing 
        regions in the image, such as the edges of a detector or structural components that are not X-ray sensitive.
    
        Default values for max_region_size and threshold_value are optimized for the Pilatus 1M and Pilatus 900k 
        detectors at NSLS-II SMI.
    
        Parameters:
        image (np.array): The input image provided as a NumPy array.
        max_region_size (int, optional): Defines the maximum size a region can be to remain unmasked. If a region 
                                         is larger than this value, it will be masked out. Default is 50.
        threshold_value (float, optional): A critical intensity value used to create the initial binary mask. Any 
                                           pixel with intensity less than or equal to this value will be marked for 
                                           potential masking. Default is 0.25.
    
        Returns:
        tuple: A tuple containing the masked image (np.array) and the binary mask (np.array). In the masked image, 
               the intensity of masked regions is replaced with NaN. In the binary mask, True values correspond to 
               the masked regions.
        """
        try:
            # Check if the image is a NumPy array
            if not isinstance(image, np.ndarray):
                raise ValueError("Input image must be a NumPy array.")
                        
            # Create a binary mask where each pixel is True if its intensity is less than or equal to the threshold_value
            binary_mask = (image <= threshold_value) | (image == 2)
        
            # Identify and label the connected regions in the binary mask
            labels, num_features = label(binary_mask)
        
            # Compute the region sizes for all labels
            region_sizes = np.bincount(labels.ravel())
            # Create a mask where True indicates a region size that is smaller than or equal to the max_region_size
            region_mask = region_sizes <= max_region_size
            # For each labeled region, mask it if its size is too large
            binary_mask = region_mask[labels]
        
            # Preserve high-intensity regions (intensity > threshold_value)
            binary_mask = binary_mask | (image > threshold_value)
        
            # Duplicate the original image to prevent unwanted modifications
            masked_image = np.copy(image)
        
            # Convert the image pixel values to float type to support NaN values
            masked_image = masked_image.astype(float)
            
            # In the masked_image, replace the intensity of all pixels that need to be masked (True in binary_mask) with NaN
            masked_image[~binary_mask] = np.nan
        
            # Return the masked image and the binary mask
            return masked_image, binary_mask
    
        except Exception as e:
            print(f"An error occurred: {e}")
            return None, None
    
    def sin_chi_correction(self, chiq, epsilon=1e-7):
        """
        This method performs a sin(chi) correction on a given DataArray. The correction is applied to each slice of data
        corresponding to a specific value of chi (the angle). The correction involves multiplying each slice by sin(chi).
    
        Parameters:
        chiq (xr.DataArray): The input DataArray containing chi and q coordinates.
        epsilon (float, optional): A small threshold value to check if chi is close to zero. Default is 1e-7.
    
        Returns:
        xr.DataArray: A new DataArray with the sin(chi) correction applied.
        """
        try:
            # Validate input type
            if not isinstance(chiq, xr.DataArray):
                raise ValueError("Input chiq must be an xarray DataArray.")
            
            # Create a copy of the original data
            corrected_chiq = chiq.copy(deep=True)
    
            for chi_val in chiq.chi:
                # Compute sin(chi) for current chi_val
                sin_chi = np.abs(np.sin(np.deg2rad(chi_val)))
    
                # Perform the division for the current chi slice if sin_chi is not zero
                if np.abs(chi_val) > epsilon:  # where epsilon is your small threshold value
                    corrected_chiq.loc[dict(chi=chi_val)] = chiq.sel(chi=chi_val) * sin_chi
                else:
                    corrected_chiq.loc[dict(chi=chi_val)] = np.nan  # Or any other value you want to assign when sin_chi is zero
    
            return corrected_chiq
    
        except Exception as e:
            print(f"An error occurred: {e}")
            return None    
    
    def process_giwaxs_file(self, file):
        """
        This method processes a GIWAXS file to perform various steps such as image masking,
        grid creation, polar transformation, and sin(chi) correction.
    
        Parameters:
        file (str): The path to the GIWAXS file to be processed.
    
        Returns:
        raw (xr.DataArray): The raw GIWAXS data.
        chiq (xr.DataArray): A processed DataArray without sin(chi) correction applied.
        corrected_chiq (xr.DataArray): A processed DataArray with sin(chi) correction applied.
        """
        try:
            # Validate input types
            if not isinstance(file, str):
                raise ValueError("Input file must be a string representing the file path.")
            if not isinstance(self.q_max, (int, float)):
                raise ValueError("q_max must be an integer or float.")
            
            # Load and process the image
            image = np.fliplr(np.flipud(np.rot90(fabio.open(file).data)))
            masked_image, mask = self.automask(image, threshold_value=2)
    
            # Create qxy and qz grids
            NumX, NumY = masked_image.shape
            qxy_new = np.linspace(-self.q_max, self.q_max, num=NumX)
            qz_new = np.linspace(0, self.q_max, num=NumY)
        
            # Stack the image data into a DataArray
            data = da.stack([masked_image],axis=2)
            raw = xr.DataArray(data, dims=("qxy", "qz", "energy"), coords={"qz":qz_new, "qxy":qxy_new}).rename(file[:-len('.tif')])
        
            # Calculate the center of the image
            center_x = float(xr.DataArray(np.linspace(0,len(raw.qxy)-1,len(raw.qxy)))
                        .assign_coords({'dim_0':raw.qxy.values})
                        .rename({'dim_0':'qxy'})
                        .interp(qxy=0)
                        .data)
            center_y = float(xr.DataArray(np.linspace(0,len(raw.qz)-1,len(raw.qz)))
                        .assign_coords({'dim_0':raw.qz.values})
                        .rename({'dim_0':'qz'})
                        .interp(qz=0)
                        .data)  
        
            # Apply the polar transformation
            TwoD = skimage.transform.warp_polar(raw.squeeze(), center=(center_x,center_y), radius = np.sqrt((raw.shape[0] - center_x)**2 + (raw.shape[1] - center_y)**2))
            TwoD = np.roll(TwoD, TwoD.shape[0]//2, axis=0)
        
            # Create the q and chi grids
            qxy = raw.qxy
            qz = raw.qz
            q = np.sqrt(qz**2+qxy**2)
            q = np.linspace(0,np.amax(q), TwoD.shape[1])
            chi = np.linspace(-179.5,179.5,360)
        
            # Create the chiq DataArray
            chiq = xr.DataArray(TwoD,dims=['chi','q'],coords={'q':q,'chi':chi},attrs=raw.attrs)
    
            # Apply the sin(chi) correction
            corrected_chiq = self.sin_chi_correction(chiq)
        
            # Return both the raw and corrected DataArrays along with the non-corrected chiq
            return raw, chiq, corrected_chiq
    
        except Exception as e:
            print(f"An error occurred: {e}")
            return None, None, None  # Return a tuple with three None values to indicate the failure
        
    def plot_qzqxy(self, qzqxy, qxy_limits=(-2, 2), qz_limits=(0, 2), cmap='viridis', dpi=150):
        """
        Plot the raw GIWAXS data.
    
        Parameters:
        qzqxy (xr.DataArray): The raw data to be plotted.
        qxy_limits (tuple): The lower and upper limits for the qxy axis.
        qz_limits (tuple): The lower and upper limits for the qz axis.
        cmap (str): The colormap to use for the plot.
        dpi (int): The DPI for the plot.
    
        Returns:
        fig (matplotlib.figure.Figure): The figure object.
        ax (matplotlib.axes._subplots.AxesSubplot): The axis object.
        """
        fig, ax = plt.subplots(dpi=dpi)
    
        cax = qzqxy.plot(x='qxy', y='qz', cmap=cmap, ax=ax, add_colorbar=False, norm=LogNorm(np.nanpercentile(qzqxy, 80), np.nanpercentile(qzqxy, 99)))
    
        # Add colorbar with custom label
        fig.colorbar(cax, ax=ax, label='Intensity (a.u.)', shrink=0.75)
    
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(AutoMinorLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(1))
    
        # Add axis labels
        ax.set_xlabel('$\it{q_{xy}}$ (Å$^{-1}$)')
        ax.set_ylabel('$\it{q_{z}}$ (Å$^{-1}$)')
        ax.xaxis.set_tick_params(which='both', size=5, width=2, direction='out')
        ax.yaxis.set_tick_params(which='both', size=5, width=2, direction='out')
    
        # Set aspect ratio to 1
        ax.set_aspect('equal')
    
        # Set axis limits
        ax.set_xlim(*qxy_limits)
        ax.set_ylim(*qz_limits)
    
        # Show the plot
        # plt.show()
        
        return fig, ax
    
    def plot_qzqxy_sq(self, qzqxy, qxy_limits=(0, 2), qz_limits=(0, 2), cmap='viridis', dpi=150):
        # Select only the negative qxy values and flip them to the positive side
        mirrored_qzqxy = qzqxy.sel(qxy=slice(None, 0))
        mirrored_qzqxy['qxy'] = -mirrored_qzqxy['qxy']
        
        # Truncate the original data from qxy=0 to qxy=2
        truncated_qzqxy = qzqxy.sel(qxy=slice(0, 2))
        
        # Sum the truncated and mirrored data
        combined_qzqxy = truncated_qzqxy + mirrored_qzqxy.reindex_like(truncated_qzqxy, method='nearest', tolerance=1e-5).fillna(0)
        
        fig, ax = plt.subplots(dpi=dpi)
    
        # Using LogNorm to handle NaN values
        cax = combined_qzqxy.plot(x='qxy', y='qz', cmap=cmap, ax=ax, add_colorbar=False, 
                                  norm=LogNorm(np.nanpercentile(qzqxy, 80), np.nanpercentile(qzqxy, 99)))
    
        # Add colorbar with custom label
        fig.colorbar(cax, ax=ax, label='Intensity (a.u.)', shrink=0.75)
    
        ax.set_xlabel('$\it{q_{xy}}$ (Å$^{-1}$)')
        ax.set_ylabel('$\it{q_{z}}$ (Å$^{-1}$)')
    
        ax.set_aspect('equal')
        ax.set_xlim(*qxy_limits)
        ax.set_ylim(*qz_limits)
    
        return fig, ax
        
    def plot_chiq(self, chiq, cmap='viridis', q_limits=None, dpi=200):
        """
        Plot the corrected chiq data.
    
        Parameters:
        chiq (xr.DataArray): The chiq data to be plotted.
        cmap (str): The colormap to use for the plot.
        q_limits (tuple): The x-values where vertical dashed lines should be plotted. If None, no lines are plotted.
        dpi (int): The DPI for the plot.
    
        Returns:
        fig (matplotlib.figure.Figure): The figure object.
        ax (matplotlib.axes._subplots.AxesSubplot): The axis object.
        """
        fig, ax = plt.subplots(dpi=dpi)
    
        cax = chiq.plot(x='q', y='chi', cmap=cmap, ax=ax, add_colorbar=False, 
                        norm=LogNorm(np.nanpercentile(chiq, 80), np.nanpercentile(chiq, 99)))
    
        # Add colorbar with custom label
        fig.colorbar(cax, ax=ax, label='Intensity (a.u.)', shrink=1)
    
        # Add axis labels
        ax.set_xlabel('$\it{q}$ (Å$^{-1}$)')
        ax.set_ylabel('Azimuth, $\it{\chi}$')
        ax.xaxis.set_tick_params(which='both', size=5, width=2, direction='in', top=True)
        ax.yaxis.set_tick_params(which='both', size=5, width=2, direction='in', right=True)
        ax.set_ylim([-90, 90])
    
        # Add vertical dashed lines if q_limits is specified
        if q_limits is not None:
            q_lower, q_upper = q_limits
            ax.vlines(x=q_lower, ymin=-90, ymax=90, colors='w', linestyles='dashed')
            ax.vlines(x=q_upper, ymin=-90, ymax=90, colors='w', linestyles='dashed')
    
        # Show the plot
        # plt.show()
        
        return fig, ax

# Example usage:
# processor = GIWAXSDataProcessor(q_max=4.0)
# corrected_data = processor.process_giwaxs_file("some_file.tif")
