# for Sellmeier coefficient, see : https://www.unitedcrystals.com/NLOCOverview.html
# for second-order suceptibility coefficients, see : https://www.sciencedirect.com/topics/chemistry/second-order-nonlinear-optical-susceptibility

# nb : The second and third indices j,k of d_ijk are then replaced by a single symbol l according to the piezoelectric contraction :
# jk:	11	22	33	23,32	31,13	12,21
# l:	1	2	3	4	    5	    6

# Lithium Niobate
LiNbO3 = {
    "name": 'LiNbO3',
    "x": [4.9048, 0.11768, 0.04750, 0.027169], # wavelength in 1e-6 m
    "y": [4.5820, 0.099169, 0.04443, 0.021950], # wavelength in 1e-6 m
    "d31": 7.11*1e-12, # type 1
    "d22": 3.07*1e-12, # type 0
    "d33": 29.1*1e-12, # type 0
}


# Potassium Titanyl Phosphate Single Crytal
KTP = {
    "name": 'KTP',
    "x": [3.0065, 0.03901, 0.04251, 0.01327], # wavelength in 1e-6 m
    "y": [3.0333, 0.04154, 0.04547, 0.01408], # wavelength in 1e-6 m
    "z": [3.3134, 0.05694, 0.05658, 0.01682], # wavelength in 1e-6 m
    "d24": 3.64*1e-12,  # type 2
    "d31": 2.54*1e-12,  # type 1
    "d32": 4.35*1e-12,  # type 1
    "d33": 16.9*1e-12,  # type 0
}