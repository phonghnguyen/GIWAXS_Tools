# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 13:18:14 2023

@author: Phong
"""

from itertools import product
from math import sqrt, pi, radians, sin, cos
import numpy as np
import pandas as pd

class CrystalIndexingAnalyzer:
    def __init__(self, space_group, a, b, c, alpha, beta, gamma):
        self.space_group = space_group.lower()
        self.a = a
        self.b = b
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def compute_spacing(self, custom_miller_indices, max_miller_index):
        """
        Compute the d-spacings and corresponding q-values for all possible
        Miller indices up to a specified maximum, for a given crystal lattice type and
        lattice parameters.
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
        Compute the d-spacing for a set of Miller indices (h, k, l), given
        the type of crystal lattice and lattice parameters.
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
    
        return theta
    
    def calculate_interplanar_and_chi(self, spacing_table, h1, k1, l1, approx_chi=0):
        """
        Compute 'Interplanar Angle' and 'Chi' columns and add them to the spacing_table DataFrame.
        This method adjusts the 'Chi' values to be within the range -90 to 90 degrees by adding multiples
        of 360 degrees if needed.
        """
        # Compute the interplanar angles
        spacing_table['Interplanar Angle'] = spacing_table.apply(
            lambda row: self.compute_interplanar_angle(h1, k1, l1, row['h'], row['k'], row['l']),
            axis=1
        )
        
        # Calculate 'Chi' values, adjusting them to be within the range -90 to 90 degrees
        def adjust_chi(angle):
            chi = approx_chi + angle
            while chi > 90:
                chi -= 180
            while chi < -90:
                chi += 180
            return chi
        
        spacing_table['Chi'] = spacing_table['Interplanar Angle'].apply(adjust_chi)
        
        return spacing_table

    def filter_spacing_table(self, spacing_table):
        """
        Filter the spacing table based on the restrictions in the Pbca space group.
        """
        filtered_table = spacing_table[
            ((spacing_table['k'] + spacing_table['l']) % 2 == 0) &  # h + l = 2n condition
            (spacing_table['l'] % 2 == 0)  # k = 2n condition
        ].reset_index(drop=True)
    
        return filtered_table
        
    def compute_q_coordinates(self,spacing_table):
        """
        Compute the qxy and qz coordinates and add them to the spacing_table DataFrame.
        """
        # Compute qxy and qz values
        qxy_values = spacing_table['QSpace_AngstromInverse'] * np.sin(np.radians(spacing_table['Chi']))
        qz_values = spacing_table['QSpace_AngstromInverse'] * np.cos(np.radians(spacing_table['Chi']))
        
        # Add new columns to the DataFrame
        spacing_table['qxy'] = qxy_values
        spacing_table['qz'] = qz_values
    
        return spacing_table.drop_duplicates()

# Example usage:
# analyzer = CrystalIndexingAnalyzer(space_group='cubic', a=1.0, b=1.0, c=1.0, alpha=90, beta=90, gamma=90)
# spacing_table = analyzer.compute_spacing([], 4)
# ...
