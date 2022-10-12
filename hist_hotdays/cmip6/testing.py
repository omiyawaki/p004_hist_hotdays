from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

# specify where file is
landfraction_file = Dataset('/project2/tas1/ockham/data9/tas/CMIP6_RAW/MPI-ESM1-2-LR/historical/atmos/fx/sftlf/r1i1p1f1/sftlf_fx_MPI-ESM1-2-LR_historical_r1i1p1f1_gn.nc', 'r')

# save data into variable
landfraction = landfraction_file.variables['sftlf'][:]

# take zonal average
landfraction_zonal = np.mean(landfraction,1)

# plot 
fig, ax = plt.subplots()
ax.plot(range(len(landfraction_zonal)), landfraction_zonal)
ax.set_xlabel('Latitude (number)')
ax.set_ylabel('Land fraction (\%)')
plt.savefig('landfraction.pdf', format='pdf', dpi=300)
