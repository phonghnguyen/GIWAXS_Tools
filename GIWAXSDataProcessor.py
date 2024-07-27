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
from skimage.transform import warp_polar
from scipy.ndimage import label
from scipy.interpolate import RegularGridInterpolator
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.colors import LogNorm

class GIWAXSDataProcessor:
    def __init__(self):
        """
        A data processor for GIWAXS images. The images should be provided as arrays. 
        """
    
    def q_to_tif_mapping(self, q12, q3, wavelength, R, beta):
        """
        Map q-space coordinates to detector coordinates.
        
        Parameters:
        q12 (numpy array): Reciprocal space coordinate in xy-plane
        q3 (numpy array): Reciprocal space coordinate in z-plane
        wavelength (float): Wavelength of the beam in angstroms
        R (float): Distance to the detector in mm
        beta (float): Incidence angle in radians
    
        Returns:
        px (numpy array): Detector coordinate in x
        pz (numpy array): Detector coordinate in z
        """
        # Calculate s and s3 from q12 and q3
        s = np.sqrt(q12**2 + q3**2) / (2 * np.pi)
        s3 = q3 / (2 * np.pi)
        
        # Compute pz using equation (12)
        p3 = (2 * wavelength * R * s / (2 - wavelength**2 * s**2)) * (s3 / s - (wavelength * s / 2) * np.sin(beta)) / np.cos(beta)
        
        # Compute px using equation (13)
        term1 = 1 - (wavelength**2 * s**2) / 4
        term2 = ((s3 / s - (wavelength * s / 2) * np.sin(beta)) / np.cos(beta))**2
        
        # Check for negative values under the square root
        sqrt_term = term1 - term2
        valid_sqrt = sqrt_term >= 0  # Boolean mask where the square root argument is non-negative
        p12 = np.zeros_like(sqrt_term)  # Initialize px array
    
        # Compute px only where the square root term is valid
        p12[valid_sqrt] = (2 * wavelength * R * s[valid_sqrt] / (2 - wavelength**2 * s[valid_sqrt]**2)) * np.sqrt(sqrt_term[valid_sqrt])
        p12[~valid_sqrt] = np.nan  # Assign NaN where the term is negative
    
        return p12, p3
    
    def calculate_jacobian(self, q12, q3, wavelength, R):
        """
        Calculate the Jacobian for intensity correction.
    
        Parameters:
        q12 (numpy array): Reciprocal space coordinate in xy-plane
        q3 (numpy array): Reciprocal space coordinate in z-plane
        wavelength (float): Wavelength of the beam in angstroms
        R (float): Sample-to-detector-distance in mm
    
        Returns:
        numpy array: Jacobian values for intensity correction
        """
        s = np.sqrt(q12**2 + q3**2) / (2 * np.pi)  # s = q / (2 * pi)
        J_F = (wavelength**2 * R**2) / ((1 - (wavelength**2 * s**2) / 2)**3)
        return J_F
    
    def img_to_qzqxy(self, det_image, bc_x, bc_y, R, incidence, px_size_x, px_size_y, q_range, q_res, xray_en):
        """
        Convert a detector image to q-space without flattening the meshgrid.
    
        Parameters:
        det_image (2D array): Detector image
        bc_x, bc_y (floats): Beam center coordinates in pixels
        R (float): Sample-to-detector-distance in mm
        incidence (float): Grazing incidence angle
        px_size_x, px_size_y (floats): pixel sizes in mm in the x and y direction
        q_range, q_res (floats): Parameters defining the q-space grid resolution
        xray_en (float): Incident photon beam energy in keV
    
        Returns:
        2D array: Corrected image in q-space
        """
        wavelength = 12.398424437 / xray_en  # Wavelength of the beam in angstrom
        incidence = np.radians(incidence)  # Convert incidence angle to radians
    
        # Adjust beam center to mm
        bc_x *= px_size_x
        bc_y = (det_image.shape[1] - bc_y) * px_size_y
    
        # Define the detector coordinate grids
        x = np.linspace(-bc_x, (det_image.shape[0] * px_size_x - bc_x), det_image.shape[0])
        y = np.linspace(-bc_y, (det_image.shape[1] * px_size_y - bc_y), det_image.shape[1])
    
        # Define the q-space grid
        qxy_points = 2 * round(q_range / q_res)
        qz_points = round(q_range / q_res)
        qxy = np.linspace(-q_range, q_range, qxy_points)
        qz = np.linspace(0, q_range, qz_points)
        Qxy, Qz = np.meshgrid(qxy, qz)
    
        # Calculate detector coordinates from q-space coordinates
        px, pz = self.q_to_tif_mapping(Qxy, Qz, wavelength, R, incidence)
    
        # Create an interpolator and interpolate using the original meshgrid
        interpolator = RegularGridInterpolator((y, x), det_image, bounds_error=False, fill_value=0)
        detector_coords = np.stack([pz, px], axis=-1)
        q_image = interpolator(detector_coords)
    
        # Calculate and apply the Jacobian for intensity correction
        jacobian = self.calculate_jacobian(Qxy, Qz, wavelength, R)
        q_image *= jacobian
        
        qzqxy = xr.DataArray(q_image, dims=("qz", "qxy"), coords={"qxy":qxy, "qz":qz})
    
        return qzqxy
    
    def cake_and_corr(self, qzqxy):
        """
        This method processes qzqxy xarray to perform various steps such as image masking,
        grid creation, polar transformation, and sin(chi) correction.
    
        Parameters:
        raw (xr.DataArray): The raw GIWAXS data.
    
        Returns:
        chiq (xr.DataArray): A processed DataArray without sin(chi) correction applied.
        corrected_chiq (xr.DataArray): A processed DataArray with sin(chi) correction applied.
        """
        data = qzqxy.values
        qz = qzqxy.qz
        qxy = qzqxy.qxy
        
        # Calculate q from qz and qxy, finding maximum radius
        q = np.sqrt(qz**2 + qxy**2)
        
        # Create a meshgrid from qz and qxy
        Qz, Qxy = np.meshgrid(qz, qxy, indexing='ij')
        
        # Calculate q from the meshgrid of qz and qxy
        q = np.sqrt(Qz**2 + Qxy**2)
        
        # Determine the center from coordinates where qz and qxy are zeros
        center_x = float(xr.DataArray(np.linspace(0,len(qzqxy.qxy)-1,len(qzqxy.qxy)))
                    .assign_coords({'dim_0':qzqxy.qxy.values})
                    .rename({'dim_0':'qxy'})
                    .interp(qxy=0)
                    .data)
        center_y = float(xr.DataArray(np.linspace(0,len(qzqxy.qz)-1,len(qzqxy.qz)))
                    .assign_coords({'dim_0':qzqxy.qz.values})
                    .rename({'dim_0':'qz'})
                    .interp(qz=0)
                    .data)  
        center = (center_y, center_x)
        
        # Apply the polar transformation
        TwoD = warp_polar(data, output_shape=(360,1000), center=(center_y,center_x), radius = np.sqrt((data.shape[1] - center_x)**2 + (data.shape[0] - center_y)**2))
        TwoD = np.roll(TwoD, TwoD.shape[0]//4, axis=0)
        
        chi = np.linspace(-180,180,360)
        q = np.linspace(0,np.amax(q), 1000)
        
        # Create xarray with proper dimensions and coordinates
        chiq = xr.DataArray(TwoD, dims=("chi", "q"), coords={"chi": chi, "q": q})

        # Apply the sin(chi) correction
        corrected_chiq = self.sin_chi_correction(chiq)
    
        # Return both the raw and corrected DataArrays along with the non-corrected chiq
        return chiq, corrected_chiq

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
        cmap = mpl.cm.viridis
        cmap.set_bad((68/255, 1/255, 84/255), 1)
        
        fig, ax = plt.subplots(dpi=dpi)
    
        cax = qzqxy.plot(x='qxy', y='qz', cmap=cmap, ax=ax, add_colorbar=False, norm=LogNorm(np.nanpercentile(qzqxy, 80), np.nanpercentile(qzqxy, 99)))
    
        # Add colorbar with custom label
        fig.colorbar(cax, ax=ax, label='Intensity (a.u.)', shrink=0.75)
    
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(AutoMinorLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(1))
    
        # Add axis labels
        ax.set_xlabel(r'$\it{q}_{xy}$ (Å$^{-1}$)')
        ax.set_ylabel(r'$\it{q}_{z}$ (Å$^{-1}$)')
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
                                  norm=LogNorm(np.nanpercentile(qzqxy, 80), np.nanpercentile(qzqxy, 99.5)))
        
        cmap = mpl.cm.viridis
        cmap.set_bad((68/255, 1/255, 84/255), 1)
    
        # Add colorbar with custom label
        fig.colorbar(cax, ax=ax, label='Intensity (a.u.)', shrink=0.75)
    
        ax.set_xlabel(r'$\it{q}_{xy}$ (Å$^{-1}$)')
        ax.set_ylabel(r'$\it{q}_{z}$ (Å$^{-1}$)')

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
        cmap = mpl.cm.viridis
        cmap.set_bad((68/255, 1/255, 84/255), 1)
        
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
