import numpy as np
import fabio
import xarray as xr
import dask.array as da
import skimage.transform
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.colors import LogNorm
from skimage.transform import warp_polar
from scipy.ndimage import label
from scipy.interpolate import RegularGridInterpolator
from scipy.constants import physical_constants, c
from kkcalc import data
from kkcalc import kk

class Reduction:
    def __init__(self):
        """
        Initializes a data processor for Grazing-Incidence Wide-Angle X-ray Scattering (GIWAXS) images.
        """
    
    def img_to_qzqxy(self, det_image, bc_x, bc_y, R, incidence, px_size_x, px_size_y, q_range, q_res, xray_en):
        """
        Converts a detector image into q-space coordinates using given parameters. Currently only supports mapping of the first quadrant of the detector (upper right of beam center)

        Parameters:
        det_image (2D array): The detector image as a 2D array.
        bc_x, bc_y (floats): Beam center coordinates in the detector image, given in pixels.
        R (float): Sample-to-detector distance in millimeters.
        incidence (float): Grazing incidence angle in degrees.
        px_size_x, px_size_y (floats): Pixel sizes in the x and y directions in millimeters.
        q_range, q_res (floats): The range and resolution of the q-space grid, respectively.
        xray_en (float): Incident photon beam energy in keV.

        Returns:
        xr.DataArray: Corrected image in q-space, represented as an xarray DataArray.
        """
        
        wavelength = physical_constants['Planck constant in eV s'][0] * c * 1e7 / xray_en  # Wavelength of the beam in angstrom
        incidence = np.radians(incidence)  # Convert incidence angle to radians
    
        # Adjust beam center to mm
        bc_x *= px_size_x
        bc_y = (det_image.shape[1] - bc_y) * px_size_y

        x = np.linspace(-bc_x, (det_image.shape[1] * px_size_x - bc_x), det_image.shape[1])
        y = np.linspace(-bc_y, (det_image.shape[0] * px_size_y - bc_y), det_image.shape[0])
 
        # Define the q-space grid
        qxy_points = round(q_range / q_res)
        qz_points = round(q_range / q_res)
        qxy = np.linspace(0, q_range, qxy_points)
        qz = np.linspace(0, q_range, qz_points)
        Qxy, Qz = np.meshgrid(qxy, qz)
    
        # Calculate detector coordinates from q-space coordinates
        px, pz = self.q_to_image_mapping(Qxy, Qz, wavelength, R, incidence)
    
        # Create an interpolator and interpolate using the original meshgrid
        interpolator = RegularGridInterpolator((y, x), det_image, bounds_error=False, fill_value=np.nan)
            
        detector_coords = np.stack([pz, px], axis=-1)
        q_image = interpolator(detector_coords)
    
        # Calculate and apply the Jacobian for intensity correction
        jacobian = self.calculate_jacobian(Qxy, Qz, wavelength, R)
        q_image *= jacobian
        
        qzqxy = xr.DataArray(q_image, dims=("qz", "qxy"), coords={"qxy":qxy, "qz":qz})
    
        return qzqxy
    
    def q_to_image_mapping(self, q12, q3, wavelength, R, beta):
        """
        Maps q-space coordinates to detector coordinates using given parameters.
    
        Parameters:
        q12 (numpy array): Reciprocal space coordinates in the xy-plane.
        q3 (numpy array): Reciprocal space coordinates in the z-direction.
        wavelength (float): Wavelength of the X-ray beam in angstroms.
        R (float): Distance from the sample to the detector in millimeters.
        beta (float): Incidence angle in radians.
    
        Returns:
        tuple: A tuple containing arrays for the detector coordinates (px, pz).
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
        Calculates the Jacobian determinant for intensity correction in GIWAXS measurements.
    
        Parameters:
        q12 (numpy array): Reciprocal space coordinate in the xy-plane.
        q3 (numpy array): Reciprocal space coordinate in the z-direction.
        wavelength (float): Wavelength of the X-ray beam in angstroms.
        R (float): Sample-to-detector distance in millimeters.
    
        Returns:
        numpy array: Jacobian values for intensity correction across the q-space.
        """
        s = np.sqrt(q12**2 + q3**2) / (2 * np.pi)  # s = q / (2 * pi)
        J_F = (wavelength**2 * R**2) / ((1 - (wavelength**2 * s**2) / 2)**3)
        return J_F
    
    def cake_and_corr(self, qzqxy, tilt_offset = 0):
        """
        Applies polar transformation and sin(chi) correction to GIWAXS data represented in q-space.

        Parameters:
        qzqxy (xr.DataArray): GIWAXS data in q-space.
        tilt_offset (float): Angle offset due to sample not being flat in the plane of the detector. Clockwise is positive.

        Returns:
        tuple: A tuple of DataArrays containing the raw and corrected polar transformation data.
        """
        data = qzqxy.values
        qz = qzqxy.qz
        qxy = qzqxy.qxy
        
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
        
        chi = np.linspace(180 + tilt_offset,-180 + tilt_offset,360)
        q = np.linspace(0,np.amax(q), 1000)
        
        # Create xarray with proper dimensions and coordinates
        chiq = xr.DataArray(TwoD, dims=("chi", "q"), coords={"chi": chi, "q": q})

        # Apply the sin(chi) correction
        corrected_chiq = self.sin_chi_correction(chiq)
    
        # Return both the raw and corrected DataArrays along with the non-corrected chiq
        return chiq, corrected_chiq

    def automask(self, image, max_region_size=50, threshold_value=0.25):
        """
        Automatically generates a mask for the input image to hide low-intensity and large irrelevant regions.

        Parameters:
        image (np.array): The input image array.
        max_region_size (int): The maximum size of unmasked regions in pixels.
        threshold_value (float): Intensity threshold for determining low-intensity regions.

        Returns:
        tuple: A tuple containing the masked image array and the binary mask array.
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
        Applies a sin(chi) correction to GIWAXS data, useful for normalizing intensity variations across angles.

        Parameters:
        chiq (xr.DataArray): Input GIWAXS data in chi-q space.
        epsilon (float): Small threshold value used to avoid division by zero near chi=0.

        Returns:
        xr.DataArray: Corrected GIWAXS data after sin(chi) normalization.
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
        
    def plot_qzqxy(self, qzqxy, qxy_limits=(0, 2), qz_limits=(0, 2), cmap='viridis', dpi=100):
        """
        Plots GIWAXS data from qzqxy coordinates with specified visual parameters.

        Parameters:
        qzqxy (xr.DataArray): GIWAXS data to be plotted.
        qxy_limits (tuple): x-axis limits for plotting.
        qz_limits (tuple): y-axis limits for plotting.
        cmap (str): Colormap for the plot.
        dpi (int): Dots per inch resolution of the plot.

        Returns:
        tuple: A tuple containing the matplotlib figure and axis objects.
        """
        plt.rcParams['axes.linewidth'] = 2
        
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
        
        return fig, ax
    
    def plot_qzqxy_sq(self, qzqxy, qxy_limits=(0, 2), qz_limits=(0, 2), cmap='viridis', dpi=100):
        """
        Plots the symmetrically summed quarter of GIWAXS data, focusing on the positive qxy and qz quadrants.
        
        Parameters:
        qzqxy (xr.DataArray): GIWAXS data to be plotted.
        qxy_limits (tuple): Limits for the qxy axis.
        qz_limits (tuple): Limits for the qz axis.
        cmap (str): Colormap to use for the plot.
        dpi (int): Dots per inch resolution of the plot.
        
        Returns:
        tuple: A tuple containing the matplotlib figure and axis objects.
        """
        # Select only the negative qxy values and flip them to the positive side
        mirrored_qzqxy = qzqxy.sel(qxy=slice(None, 0))
        mirrored_qzqxy['qxy'] = -mirrored_qzqxy['qxy']
        
        # Truncate the original data from qxy=0 to qxy=2
        truncated_qzqxy = qzqxy.sel(qxy=slice(0, 2))
        
        # Sum the truncated and mirrored data
        combined_qzqxy = truncated_qzqxy + mirrored_qzqxy.reindex_like(truncated_qzqxy, method='nearest', tolerance=1e-5).fillna(0)
        
        plt.rcParams['axes.linewidth'] = 2

        fig, ax = plt.subplots(dpi=dpi)
    
        # Using LogNorm to handle NaN values
        cax = combined_qzqxy.plot(x='qxy', y='qz', cmap=cmap, ax=ax, add_colorbar=False, 
                                  norm=LogNorm(np.nanpercentile(qzqxy, 80), np.nanpercentile(qzqxy, 99.5)))
        
        # Add colorbar with custom label
        fig.colorbar(cax, ax=ax, label='Intensity (a.u.)', shrink=0.75)
    
        ax.set_xlabel(r'$\it{q}_{xy}$ (Å$^{-1}$)')
        ax.set_ylabel(r'$\it{q}_{z}$ (Å$^{-1}$)')

        ax.set_aspect('equal')
        ax.set_xlim(*qxy_limits)
        ax.set_ylim(*qz_limits)
    
        return fig, ax
        
    def plot_chiq(self, chiq, cmap='viridis', dpi=100):
        """
        Plot the corrected chiq data.
    
        Parameters:
        chiq (xr.DataArray): The chiq data to be plotted.
        cmap (str): The colormap to use for the plot.
        dpi (int): Dots per inch resolution of the plot.
    
        Returns:
        tuple: A tuple containing the matplotlib figure and axis objects.
        """
        plt.rcParams['axes.linewidth'] = 2

        fig, ax = plt.subplots(dpi=dpi)
    
        cax = chiq.plot(x='q', y='chi', cmap=cmap, ax=ax, add_colorbar=False, 
                        norm=LogNorm(np.nanpercentile(chiq, 80), np.nanpercentile(chiq, 99)))
    
        # Add colorbar with custom label
        fig.colorbar(cax, ax=ax, label='Intensity (a.u.)', shrink=1)
    
        # Add axis labels
        ax.set_xlabel('$q$ (Å$^{-1}$)')
        ax.set_ylabel('Azimuth, $\it{\chi}$ (°)')
        ax.xaxis.set_tick_params(which='both', size=5, width=2, direction='in', top=True)
        ax.yaxis.set_tick_params(which='both', size=5, width=2, direction='in', right=True)
        ax.set_ylim([0, 90])
    
        return fig, ax

class Calculation:
    @staticmethod
    def calc_penetration_depth(xray_en, alpha_i, alpha_c, beta):
        """
        Calculates the penetration depth of grazing incidence X-rays (< 1 degree) or non-grazing incidence (>= 1 degree).
        
        Parameters:
        xray_en (float): Incident photon beam energy in keV.
        alpha_i (float or array): Grazing-incidence angle in degrees.
        alpha_c (float): Critical angle in degrees.
        beta (float): The material's absorbance at the given xray_en.

        Returns:
        penetration depth (float): Depth where X-ray intensity is attenuated to 1/e (~37%) of incident intensity with units of angstrom.
        """
        wavelength = physical_constants['Planck constant in eV s'][0] * c * 1e7 / xray_en  # Wavelength of the beam in angstrom
        alpha_i_rad = np.radians(alpha_i)  # Convert from degrees to radians
        alpha_c_rad = np.radians(alpha_c)  # Convert from degrees to radians
        
        penetration_depth = np.zeros_like(alpha_i)
    
        conditions = (0 < alpha_i) & (alpha_i < 1)
        penetration_depth[conditions] = wavelength * np.sqrt(2 / (np.sqrt(((alpha_i_rad[conditions]**2 - alpha_c_rad**2))**2 + 4 * beta**2) - (alpha_i_rad[conditions]**2 - alpha_c_rad**2))) / (4 * np.pi)
    
        conditions = (1 <= alpha_i) & (alpha_i <= 90)
        penetration_depth[conditions] = 1 / ((4 * np.pi / wavelength) * beta) * np.cos(np.pi / 2 - alpha_i_rad[conditions])
    
        invalid_conditions = (np.degrees(alpha_i_rad) <= 0) | (np.degrees(alpha_i_rad) > 90)
        if np.any(invalid_conditions):
            raise ValueError('The incidence angle must be greater than 0 degrees and less than or equal to 90 degrees.')

        return penetration_depth
    
    @staticmethod
    def calc_refractive_index(xray_en, density, chemical_formula):
        """
        Calculates the real and imaginary portions of the complex refractive index based on the Henke Database of atomic scattering factors using kkcalc.
        
        Parameters:
        xray_en (float): Incident photon beam energy in keV.
        density (float): Mass density in grams per mole.
        chemical_formula (string): chemical formula representative of one mole, e.g., 'C10H14S' for a P3HT monomer

        Returns:
        tuple: a tuple containing the real and imaginary portions of the complex refractive indices, delta and beta, respectively.
        """
        xray_en *= 1000 # convert from keV to eV
        stoichiometry = kk.data.ParseChemicalFormula(chemical_formula)
        formula_mass = kk.data.calculate_FormulaMass(stoichiometry)

        ASF_E, ASF_Data = kk.data.calculate_asf(stoichiometry)
        imaginary = kk.data.coeffs_to_ASF(ASF_E, np.vstack((ASF_Data, ASF_Data[-1])))
        
        beta_cont = kk.data.convert_data(np.column_stack((ASF_E, imaginary)),'ASF','refractive_index', Density=density, Formula_Mass=formula_mass)
        
        merged = kk.data.merge_spectra(np.column_stack((ASF_E, imaginary)), ASF_E, ASF_Data, merge_points=(xray_en*0.9,xray_en*1.1), add_background=False, fix_distortions=False, plotting_extras=True)
        
        correction = kk.calc_relativistic_correction(stoichiometry)
        
        real = kk.KK_PP(merged[2][:,0], merged[0], merged[1], correction)
        delta_cont = kk.data.convert_data(np.column_stack((merged[2][:,0], real)),'ASF','refractive_index', Density=density, Formula_Mass=formula_mass)
        
        beta = np.interp(xray_en, beta_cont[:, 0], beta_cont[:, 1])
        delta = np.interp(xray_en, delta_cont[:, 0], delta_cont[:, 1])
        
        return delta, beta
    
    @staticmethod
    def calc_critical_angle(delta):
        """
        Calculates the critical angle (in degrees) using the real part of the refractive index.
        
        Parameters:
        delta (float): Real part of the refractive index.

        Returns:
        alpha_c (float): The critical angle in degrees.
        """
        alpha_c = np.degrees(np.sqrt(2 * delta))
        
        return alpha_c
