from itertools import product
from math import sqrt, pi, radians, sin, cos
import numpy as np
import pandas as pd

class Indexing:
    def __init__(self, space_group, a, b, c, alpha, beta, gamma):
        """
        Initializes an analyzer for indexing crystal structures using specified lattice parameters.
        
        Parameters:
        space_group (str): The crystallographic space group.
        a (float): Lattice constant a in Ångströms.
        b (float): Lattice constant b in Ångströms.
        c (float): Lattice constant c in Ångströms.
        alpha (float): Alpha lattice angle in degrees.
        beta (float): Beta lattice angle in degrees.
        gamma (float): Gamma lattice angle in degrees.
        """
        self.space_group = space_group.lower()
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def compute_spacing(self, custom_miller_indices, max_miller_index):
        """
        Computes the d-spacings and corresponding q-values for all possible Miller indices up to a specified maximum.

        Parameters:
        custom_miller_indices (list of tuples): Custom list of Miller indices to include.
        max_miller_index (int): The maximum Miller index to consider.

        Returns:
        DataFrame: A pandas DataFrame containing Miller indices with their respective real-space (d) and reciprocal-space (q) values.
        """
        if len(custom_miller_indices) == 0:
            indices = list(product(range(0, max_miller_index + 1), repeat=3))
            miller_indices_array = [index for index in indices if any(index)]
        else:
            miller_indices_array = custom_miller_indices
    
        # Generate negative counterparts for each set of Miller indices
        negative_miller_indices = [-np.array(index) for index in miller_indices_array if any(index)]
        miller_indices_array += negative_miller_indices
    
        miller_indices_array = np.array(miller_indices_array)
        h_array = miller_indices_array[:, 0]
        k_array = miller_indices_array[:, 1]
        l_array = miller_indices_array[:, 2]
        real_space_array = np.zeros(len(miller_indices_array))
        q_space_array = np.zeros(len(miller_indices_array))
    
        for i, miller_indices in enumerate(miller_indices_array):
            h, k, l = miller_indices
            d = self.compute_d_spacing(h, k, l)
            q = 2 * pi / d
            real_space_array[i] = d
            q_space_array[i] = q
    
        spacing_table = pd.DataFrame(
            {
                'h': h_array,
                'k': k_array,
                'l': l_array,
                'RealSpace_Angstroms': real_space_array,
                'QSpace_AngstromInverse': q_space_array
            }
        )
    
        return spacing_table.drop_duplicates()
    
    def compute_d_spacing(self, h, k, l):
        """
        Calculates the d-spacing based on Miller indices and crystal lattice type.

        Parameters:
        h (int): Miller index h.
        k (int): Miller index k.
        l (int): Miller index l.

        Returns:
        float: The calculated d-spacing in Ångströms.
        """
        space_group = self.space_group.lower()
    
        if space_group == 'cubic':
            d = self.a / sqrt(h**2 + k**2 + l**2)
        elif space_group == 'hexagonal':
            d = sqrt(1 / ((4/3 * (h**2 + h*k + k**2) / self.a**2) + (l**2 / self.c**2)))
        elif space_group == 'monoclinic':
            d = sqrt(1 / ((h**2 / self.a**2) + (k**2 * sin(radians(self.beta))**2 / self.b**2) + (l**2 / self.c**2) - (2 * h * l * cos(radians(self.beta)) / self.a * self.c)) / sin(radians(self.beta))**2)
        elif space_group == 'rhombohedral':
            d = sqrt(1 / (((h**2 + k**2 + l**2) * sin(radians(self.alpha))**2) + ((h*k + k*l + h*l) * 2 * cos(radians(self.alpha))**2) - cos(radians(self.alpha)) / self.a**2 * (1 - 3 * cos(radians(self.alpha))**2 + 2 * cos(radians(self.alpha))**3)))
        elif space_group == 'tetragonal':
            d = sqrt(1 / ((h**2 + k**2) / self.a**2 + l**2 / self.c**2))
        elif space_group == 'triclinic':
            V = self.a * self.b * self.c * sqrt(1 - cos(radians(self.alpha))**2 - cos(radians(self.beta))**2 - cos(radians(self.gamma))**2 + 2 * cos(radians(self.alpha)) * cos(radians(self.beta)) * cos(radians(self.gamma)))
            d = sqrt(1 / ((self.b**2 * self.c**2 * sin(radians(self.alpha))**2 * h**2) + (self.a**2 * self.c**2 * sin(radians(self.beta))**2 * k**2) + (self.a**2 * self.b**2 * sin(radians(self.gamma))**2 * l**2) + (2 * self.a * self.b * self.c**2 * (cos(radians(self.alpha)) * cos(radians(self.beta)) - cos(radians(self.gamma))) * h * k) + (2 * self.b * self.c * self.a**2 * (cos(radians(self.gamma)) * cos(radians(self.beta)) - cos(radians(self.alpha))) * l * k) + (2 * self.a * self.c * self.b**2 * (cos(radians(self.alpha)) * cos(radians(self.gamma)) - cos(radians(self.beta))) * h * l)) / V**2)
        elif space_group == 'orthorhombic':
            d = sqrt(1 / ((h**2 / self.a**2) + (k**2 / self.b**2) + (l**2 / self.c**2)))
        else:
            raise ValueError("Invalid space group. Accepted values: cubic, hexagonal, monoclinic, rhombohedral, tetragonal, triclinic, orthorhombic")
        return d
    
    def compute_interplanar_angle(self, h1, k1, l1, h2, k2, l2):
        """
        Calculates the angle between two planes defined by their Miller indices.
        
        Parameters:
        h1, k1, l1 (int): Miller indices of the first plane.
        h2, k2, l2 (int): Miller indices of the second plane.
        
        Returns:
        float: The angle in degrees between the two planes.
        """
        # Calculate d-spacing for each set of Miller indices
        d_hkl = self.compute_d_spacing(h1, k1, l1)
        d_hkl_prime = self.compute_d_spacing(h2, k2, l2)
    
        # Calculate theta using the general equation
        theta = np.arccos(d_hkl * d_hkl_prime * (h1*h2*self.a**-2 + k1*k2*self.b**-2 + l1*l2*self.c**-2 + 
                                                 (k1*l2 + l1*k2)*self.b**-1*self.c**-1*np.cos(np.radians(self.alpha)) +
                                                 (h1*l2 + l1*h2)*self.a**-1*self.c**-1*np.cos(np.radians(self.beta)) + 
                                                 (h1*k2 + k1*h2)*self.a**-1*self.b**-1*np.cos(np.radians(self.gamma))))
    
        # Convert the angle from radians to degrees
        theta = np.degrees(theta)
    
        # Apply logic to determine if the angle should be negative
        negative_indices = [h1 < 0, k1 < 0, l1 < 0, h2 < 0, k2 < 0, l2 < 0]
        if sum(negative_indices) % 2 != 0:  # If there's an odd number of negative indices
            theta *= -1
    
        return theta

    def calculate_interplanar_and_chi(self, spacing_table, h1, k1, l1, approx_chi=0):
        """
        Computes interplanar angles and adjusts chi values for given Miller indices, adding these to the provided DataFrame.

        Parameters:
        spacing_table (DataFrame): DataFrame containing d-spacing and q-values.
        h1, k1, l1 (int): Reference Miller indices for interplanar angle calculations.
        approx_chi (float): Initial chi value for adjustments.

        Returns:
        DataFrame: The updated DataFrame with 'Interplanar Angle' and 'Chi' columns.
        """
        # Compute the interplanar angles
        spacing_table['Interplanar Angle'] = spacing_table.apply(
            lambda row: self.compute_interplanar_angle(h1, k1, l1, row['h'], row['k'], row['l']),
            axis=1
        ).round(4)
        
        # Calculate 'Chi' values, adjusting them to be within the range -90 to 90 degrees
        def adjust_chi(angle):
            chi = approx_chi + angle
            return chi
        
        spacing_table['Chi'] = spacing_table['Interplanar Angle'].apply(adjust_chi).round(4)
        
        return spacing_table
        
    def compute_q_coordinates(self, spacing_table):
        """
        Computes q-space coordinates (qxy and qz) and adds these to the provided DataFrame based on the calculated chi values.

        Parameters:
        spacing_table (DataFrame): DataFrame containing Miller indices, chi, and d-spacing.

        Returns:
        DataFrame: The updated DataFrame with qxy and qz coordinates added.
        """
        # Compute qxy and qz values
        qxy_values = spacing_table['QSpace_AngstromInverse'] * np.sin(np.radians(spacing_table['Chi']))
        qz_values = spacing_table['QSpace_AngstromInverse'] * np.cos(np.radians(spacing_table['Chi']))
        
        # Add new columns to the DataFrame
        spacing_table['qxy'] = qxy_values
        spacing_table['qz'] = qz_values
    
        # Identify Miller indices with qxy = 0 and h, k, l > 0
        zero_qxy_indices = spacing_table[(spacing_table['qxy'] == 0) & (spacing_table['h'] >= 0) & (spacing_table['k'] >= 0) & (spacing_table['l'] >= 0)][['h', 'k', 'l']]
    
        # Remove negative counterparts of Miller indices with qxy = 0
        for _, row in zero_qxy_indices.iterrows():
            h, k, l = row['h'], row['k'], row['l']
            if h >= 0 and k >= 0 and l >= 0:
                negative_counterpart = spacing_table[(spacing_table['h'] == -h) & (spacing_table['k'] == -k) & (spacing_table['l'] == -l)]
                
                if not negative_counterpart.empty:
                    spacing_table.drop(negative_counterpart.index, inplace=True)
        
        return spacing_table.drop_duplicates().reset_index(drop=True)