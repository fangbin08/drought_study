import os
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "sans-serif"
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import NullFormatter
import numpy as np
import glob
import h5py
import gdal
from osgeo import ogr
import fiona
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile
from rasterio.transform import from_origin, Affine
from rasterio.windows import Window
from rasterio.crs import CRS
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import datetime
import calendar

# Ignore runtime warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


########################################################################################################################
# (Function 1) Subset the coordinates table of desired area

def coordtable_subset(lat_input, lon_input, lat_extent_max, lat_extent_min, lon_extent_max, lon_extent_min):
    lat_output = lat_input[np.where((lat_input <= lat_extent_max) & (lat_input >= lat_extent_min))]
    row_output_ind = np.squeeze(np.array(np.where((lat_input <= lat_extent_max) & (lat_input >= lat_extent_min))))
    lon_output = lon_input[np.where((lon_input <= lon_extent_max) & (lon_input >= lon_extent_min))]
    col_output_ind = np.squeeze(np.array(np.where((lon_input <= lon_extent_max) & (lon_input >= lon_extent_min))))

    return lat_output, row_output_ind, lon_output, col_output_ind

def coordtable_subset_V2(lat_input, lon_input, lat_extent_max, lat_extent_min, lon_extent_max, lon_extent_min):
    row_ind_min = np.argmin(np.absolute(lat_input - lat_extent_max))
    row_ind_max = np.argmin(np.absolute(lat_input - lat_extent_min))
    row_output_ind = np.arange(row_ind_min, row_ind_max+1)
    lat_output = lat_input[row_ind_min:row_ind_max+1]
    col_ind_min = np.argmin(np.absolute(lon_input - lon_extent_min))
    col_ind_max = np.argmin(np.absolute(lon_input - lon_extent_max))
    col_output_ind = np.arange(col_ind_min, col_ind_max+1)
    lon_output = lon_input[col_ind_min:col_ind_max+1]

    return lat_output, row_output_ind, lon_output, col_output_ind

########################################################################################################################
# Function 2. Subset and reproject the Geotiff data to WGS84 projection

def sub_n_reproj(input_mat, kwargs_sub, sub_window, output_crs):
    # Get the georeference and bounding parameters of subset image
    kwargs_sub.update({
        'height': sub_window.height,
        'width': sub_window.width,
        'transform': rasterio.windows.transform(sub_window, kwargs_sub['transform'])})

    # Generate subset rasterio dataset
    input_ds_subset = MemoryFile().open(**kwargs_sub)
    input_mat = np.expand_dims(input_mat, axis=0)
    input_ds_subset.write(input_mat)

    # Reproject a dataset
    transform_reproj, width_reproj, height_reproj = \
        calculate_default_transform(input_ds_subset.crs, output_crs,
                                    input_ds_subset.width, input_ds_subset.height, *input_ds_subset.bounds)
    kwargs_reproj = input_ds_subset.meta.copy()
    kwargs_reproj.update({
            'crs': output_crs,
            'transform': transform_reproj,
            'width': width_reproj,
            'height': height_reproj
        })

    output_ds = MemoryFile().open(**kwargs_reproj)
    reproject(source=rasterio.band(input_ds_subset, 1), destination=rasterio.band(output_ds, 1),
              src_transform=input_ds_subset.transform, src_crs=input_ds_subset.crs,
              dst_transform=transform_reproj, dst_crs=output_crs, resampling=Resampling.nearest)

    return output_ds

#########################################################################################
# (Function 3) Convert latitude and longitude to the corresponding row and col in the
# EASE grid VERSION 2 used at CATDS since processor version 2.7, January 2014

def geo2easeGridV2(latitude, longitude, interdist, num_row, num_col):
    # Constant
    a = 6378137  # equatorial radius
    f = 1 / 298.257223563  # flattening
    b = 6356752.314  # polar radius b=a(1-f)
    e = 0.0818191908426  # eccentricity sqrt(2f-f^2)
    c = interdist  # interdistance pixel
    nl = num_row  # Number of lines
    nc = num_col  # Number of columns
    s0 = (nl - 1) / 2
    r0 = (nc - 1) / 2
    phi0 = 0
    lambda0 = 0  # map reference longitude
    phi1 = 30  # latitude true scale
    k0 = np.cos(np.deg2rad(phi1)) / np.sqrt(1 - (e ** 2 * np.sin(np.deg2rad(phi1)) ** 2))
    q = (1 - e ** 2) * ((np.sin(np.deg2rad(latitude)) / (1 - e ** 2 * np.sin(np.deg2rad(latitude)) ** 2)) -
                        (1 / (2 * e)) * np.log(
                (1 - e * np.sin(np.deg2rad(latitude))) / (1 + e * np.sin(np.deg2rad(latitude)))))
    x = a * k0 * (longitude - lambda0) * np.pi / 180
    y = a * q / (2 * k0)
    # as Brodzik et al
    column = np.round(r0 + (x / c)).astype(int)
    row = np.round(s0 - (y / c)).astype(int)

    del a, f, b, e, c, nl, nc, s0, r0, phi0, lambda0, phi1, k0, q, x, y

    return row, column

#########################################################################################
# (Function 4) Find and map the corresponding index numbers for the low spatial resolution
# row/col tables from the high spatial resolution row/col tables. The output is 1-dimensional
# nested list array containing index numbers. (Aggregate)

def find_easeind_lofrhi(lat_hires, lon_hires, interdist_lowres, num_row_lowres, num_col_lowres, row_lowres_ind, col_lowres_ind):

    lon_meshgrid, lat_meshgrid = np.meshgrid(lon_hires, lat_hires)

    # Select only the first row + first column to find the row/column indices
    lat_meshgrid_array = np.concatenate((lat_meshgrid[:, 0], lat_meshgrid[0, :]), axis=0)
    lon_meshgrid_array = np.concatenate((lon_meshgrid[:, 0], lon_meshgrid[0, :]), axis=0)

    [row_ind_toresp, col_ind_toresp] = \
        geo2easeGridV2(lat_meshgrid_array, lon_meshgrid_array, interdist_lowres,
                       num_row_lowres, num_col_lowres)

    row_ind_toresp = row_ind_toresp[:(len(lat_hires))]
    col_ind_toresp = col_ind_toresp[(len(lat_hires)):]

    # Assign the low resolution grids with corresponding high resolution grids index numbers
    row_ease_dest_init = []
    for x in range(len(row_lowres_ind)):
        row_ind = np.where(row_ind_toresp == row_lowres_ind[x])
        row_ind = np.array(row_ind).ravel()
        row_ease_dest_init.append(row_ind)

    row_ease_dest_ind = np.asarray(row_ease_dest_init)

    col_ease_dest_init = []
    for x in range(len(col_lowres_ind)):
        col_ind = np.where(col_ind_toresp == col_lowres_ind[x])
        col_ind = np.array(col_ind).ravel()
        col_ease_dest_init.append(col_ind)

    col_ease_dest_ind = np.asarray(col_ease_dest_init)

    # Assign the empty to-be-resampled grids with index numbers of corresponding nearest destination grids
    for x in range(len(row_ease_dest_ind)):
        if len(row_ease_dest_ind[x]) == 0 and x != 0 and x != len(row_ease_dest_ind)-1:
            # Exclude the first and last elements
            row_ease_dest_ind[x] = np.array([row_ease_dest_ind[x - 1], row_ease_dest_ind[x + 1]]).ravel()
        else:
            pass

    for x in range(len(col_ease_dest_ind)):
        if len(col_ease_dest_ind[x]) == 0 and x != 0 and x != len(col_ease_dest_ind)-1:
            # Exclude the first and last elements
            col_ease_dest_ind[x] = np.array([col_ease_dest_ind[x - 1], col_ease_dest_ind[x + 1]]).ravel()
        else:
            pass

    return row_ease_dest_ind, col_ease_dest_ind

########################################################################################################################

# 0. Input variables
# Specify file paths
# Path of current workspace
path_workspace = '/Users/binfang/Documents/SMAP_Project/smap_codes'
# Path of Land mask
path_lmask = '/Users/binfang/Downloads/Processing/processed_data'
# Path of dataset
path_datasets = '/Volumes/Elements/Datasets'
# Path of model data
path_model = '/Volumes/Elements/Datasets/model_data'
# # Path of EASE projection lat/lon tables
path_ease_coord_table = '/Volumes/Elements/Datasets/Lmask'
# Path of GIS data
path_gis_data = '/Users/binfang/Documents/SMAP_Project/data/gis_data'
# Path of results
path_results = '/Users/binfang/Documents/SMAP_Project/results/results_221102'
# Path of preview
path_model_evaluation = '/Users/binfang/Documents/SMAP_Project/results/results_221102/model_evaluation'
# Path of SMAP SM
path_smap = '/Volumes/Elements/Datasets/SMAP'
# Path of SMOS SM
path_smos = '/Volumes/Elements/Datasets/SMOS'
# Path of SMAP output
path_output = '/Users/binfang/Downloads/Processing/smap_output'

lst_folder = '/MYD11A1/'
ndvi_folder = '/MYD13A2/'
yearname = np.linspace(2010, 2025, 16, dtype='int')
monthnum = np.linspace(1, 12, 12, dtype='int')
monthname = np.arange(1, 13)
monthname = [str(i).zfill(2) for i in monthname]

# Load in geo-location parameters
os.chdir(path_workspace)
f = h5py.File("ds_parameters.hdf5", "r")
varname_list = ['lat_world_max', 'lat_world_min', 'lon_world_max', 'lon_world_min', 'cellsize_1km', 'cellsize_9km',
                'lat_world_ease_9km', 'lon_world_ease_9km', 'lat_world_ease_1km', 'lon_world_ease_1km',
                'lat_world_ease_25km', 'lon_world_ease_25km', 'row_world_ease_9km_from_1km_ind',
                'col_world_ease_9km_from_1km_ind', 'row_world_ease_25km_from_1km_ind',
                'col_world_ease_25km_from_1km_ind', 'interdist_ease_25km', 'size_world_ease_25km',
                'row_world_ease_25km_ind', 'col_world_ease_25km_ind', 'row_world_ease_1km_from_9km_ind',
                'col_world_ease_1km_from_9km_ind',  'row_world_ease_1km_from_25km_ind', 'col_world_ease_1km_from_25km_ind']
for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f.close()

# Generate sequence of string between start and end dates (Year + DOY)
start_date = '2010-01-01'
end_date = '2025-12-31'

start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
delta_date = end_date - start_date

date_seq = []
date_seq_ymd = []
for i in range(delta_date.days + 1):
    date_str = start_date + datetime.timedelta(days=i)
    date_seq.append(str(date_str.timetuple().tm_year) + str(date_str.timetuple().tm_yday).zfill(3))
    date_seq_ymd.append(date_str.strftime('%Y%m%d'))

# Count how many days for a specific year
yearname = np.linspace(2010, 2025, 16, dtype='int')
monthnum = np.linspace(1, 12, 12, dtype='int')
monthname = np.arange(1, 13)
monthname = [str(i).zfill(2) for i in monthname]

daysofyear = []
for idt in range(len(yearname)):
    # if idt == 0:
    #     f_date = datetime.date(yearname[idt], monthnum[3], 1)
    #     l_date = datetime.date(yearname[idt], monthnum[-1], 31)
    #     delta_1y = l_date - f_date
    #     daysofyear.append(delta_1y.days + 1)
    # else:
        f_date = datetime.date(yearname[idt], monthnum[0], 1)
        l_date = datetime.date(yearname[idt], monthnum[-1], 31)
        delta_1y = l_date - f_date
        daysofyear.append(delta_1y.days + 1)

daysofyear = np.asarray(daysofyear)

# Find the indices of each month in the list of days between 2015 - 2018
nlpyear = 1999 # non-leap year
lpyear = 2000 # leap year
daysofmonth_nlp = np.array([calendar.monthrange(nlpyear, x)[1] for x in range(1, len(monthnum)+1)])
ind_nlp = [np.arange(daysofmonth_nlp[0:x].sum(), daysofmonth_nlp[0:x+1].sum()) for x in range(0, len(monthnum))]
daysofmonth_lp = np.array([calendar.monthrange(lpyear, x)[1] for x in range(1, len(monthnum)+1)])
ind_lp = [np.arange(daysofmonth_lp[0:x].sum(), daysofmonth_lp[0:x+1].sum()) for x in range(0, len(monthnum))]
ind_iflpr = np.array([int(calendar.isleap(yearname[x])) for x in range(len(yearname))]) # Find out leap years
# Generate a sequence of the days of months for all years
daysofmonth_seq = np.array([np.tile(daysofmonth_nlp[x], len(yearname)) for x in range(0, len(monthnum))])
daysofmonth_seq[1, :] = daysofmonth_seq[1, :] + ind_iflpr # Add leap days to February
daysofmonth_seq_cumsum = np.cumsum(daysofmonth_seq, axis=0)
daysofmonth_seq_cumsum = np.concatenate((np.zeros((1, len(yearname)), dtype=int), daysofmonth_seq_cumsum), axis=0)

# Generate land/water mask provided by GLDAS/NASA
lmask_file = open(path_ease_coord_table + '/EASE2_M25km.LOCImask_land50_coast0km.1388x584.bin', 'r')
lmask_ease_25km = np.fromfile(lmask_file, dtype=np.dtype('uint8'))
lmask_ease_25km = np.reshape(lmask_ease_25km, [len(lat_world_ease_25km), len(lon_world_ease_25km)]).astype(float)
lmask_ease_25km[np.where(lmask_ease_25km != 0)] = np.nan
lmask_ease_25km[np.where(lmask_ease_25km == 0)] = 1
# lmask_ease_25km[np.where((lmask_ease_25km == 101) | (lmask_ease_25km == 255))] = 0
lmask_file.close()

# Find the indices of land pixels by the 25-km resolution land-ocean mask
[row_lmask_ease_25km_ind, col_lmask_ease_25km_ind] = np.where(~np.isnan(lmask_ease_25km))


########################################################################################################################
# 1. GLDAS Model Maps

matsize_1km_init = np.empty((len(lat_world_ease_1km), len(lon_world_ease_1km)), dtype='float32')
matsize_1km_init[:] = np.nan
matsize_9km_init = np.empty((len(lat_world_ease_9km), len(lon_world_ease_9km)), dtype='float32')
matsize_9km_init[:] = np.nan
matsize_25km_init = np.empty((len(lat_world_ease_25km), len(lon_world_ease_25km)), dtype='float32')
matsize_25km_init[:] = np.nan

# 1.1 Import data for plotting GLDAS model maps
hdf_file = path_model + '/gldas/ds_model_coef_nofill.hdf5'
f_read = h5py.File(hdf_file, "r")
varname_list = list(f_read.keys())

matsize_25km_monthly_init = np.repeat(matsize_25km_init[:, :, np.newaxis], len(monthname)*2, axis=2)
r2_mat_monthly = np.copy(matsize_25km_monthly_init)
rmse_mat_monthly = np.copy(matsize_25km_monthly_init)
slope_mat_monthly = np.copy(matsize_25km_monthly_init)

for imo in range(len(monthname)*2):
    r2_mat = f_read[varname_list[24+imo]][:, :, 0]
    rmse_mat = f_read[varname_list[24+imo]][:, :, 1]
    slope_mat = f_read[varname_list[imo]][:, :, 0]
    slope_mat_monthly[:, :, imo] = slope_mat
    r2_mat_monthly[:, :, imo] = r2_mat
    rmse_mat_monthly[:, :, imo] = rmse_mat
    del(r2_mat, rmse_mat, slope_mat)


# 1.2 Single maps
# os.chdir(path_results)

xx_wrd, yy_wrd = np.meshgrid(lon_world_ease_25km, lat_world_ease_25km) # Create the map matrix
shape_world = ShapelyFeature(Reader(path_gis_data + '/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
fig = plt.figure(num=None, figsize=(8, 5), dpi=100, facecolor='w', edgecolor='k')
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([-180, 180, -70, 90], ccrs.PlateCarree())
ax.add_feature(shape_world)
img = ax.pcolormesh(xx_wrd, yy_wrd, r2_mat_monthly[:, :, 6], transform=ccrs.PlateCarree(), vmin=0, vmax=0.6, cmap='viridis')
gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
gl.ylocator = mticker.FixedLocator([-90, -60, -30, 0, 30, 60, 90])
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
plt.suptitle('SMAP SM 9 km', y=0.96, fontsize=15)
plt.show()


# 1.3 Subplot maps
# 1.3.1 R^2 of AM
xx_wrd, yy_wrd = np.meshgrid(lon_world_ease_25km, lat_world_ease_25km) # Create the map matrix
shape_world = ShapelyFeature(Reader(path_gis_data + '/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
title_content = ['JFM', 'AMJ', 'JAS', 'OND']


fig = plt.figure(figsize=(12, 6), facecolor='w', edgecolor='k', dpi=150)
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, hspace=0.2, wspace=0.2)
for ipt in range(len(monthname)//3):
    ax = fig.add_subplot(2, 2, ipt+1, projection=ccrs.PlateCarree())
    ax.add_feature(shape_world, linewidth=0.5)
    ax.set_extent([-180, 180, -70, 90], ccrs.PlateCarree())
    img = ax.pcolormesh(xx_wrd, yy_wrd, np.nanmax(r2_mat_monthly[:, :, ipt*3:ipt*3+2], axis=2),
                        transform=ccrs.PlateCarree(), vmin=0, vmax=0.6, cmap='hot_r')
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = mticker.FixedLocator([90, 45, 0, -45, -90])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
    # ax.set_title(title_content[ipt], pad=20, fontsize=14, weight='bold')
    ax.text(-170, -45, title_content[ipt], fontsize=14, horizontalalignment='left',
            verticalalignment='top', weight='bold')
cbar_ax = fig.add_axes([0.2, 0.04, 0.6, 0.02])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both', pad=0.1, orientation='horizontal')
cbar.ax.tick_params(labelsize=10)
cbar_ax.locator_params(nbins=6)
plt.suptitle('$\mathregular{R^2}$ (a.m.)', fontsize=16, weight='bold')
plt.savefig(path_model_evaluation + '/r2_world_am.png')
plt.close()

# 1.3.2 RMSE of AM
fig = plt.figure(figsize=(12, 6), facecolor='w', edgecolor='k', dpi=150)
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, hspace=0.2, wspace=0.2)
for ipt in range(len(monthname)//3):
    ax = fig.add_subplot(2, 2, ipt+1, projection=ccrs.PlateCarree())
    ax.add_feature(shape_world, linewidth=0.5)
    ax.set_extent([-180, 180, -70, 90], ccrs.PlateCarree())
    img = ax.pcolormesh(xx_wrd, yy_wrd, np.nanmin(rmse_mat_monthly[:, :, ipt*3:ipt*3+2], axis=2),
                        transform=ccrs.PlateCarree(), vmin=0, vmax=0.08, cmap='Reds')
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = mticker.FixedLocator([90, 45, 0, -45, -90])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
    # ax.set_title(title_content[ipt], pad=20, fontsize=14, weight='bold')
    ax.text(-170, -45, title_content[ipt], fontsize=14, horizontalalignment='left',
            verticalalignment='top', weight='bold')
cbar_ax = fig.add_axes([0.2, 0.04, 0.6, 0.02])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both', pad=0.1, orientation='horizontal')
cbar.ax.tick_params(labelsize=10)
cbar_ax.locator_params(nbins=8)
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=10, x=1.05, y=0.05, labelpad=-15)
plt.suptitle('RMSE (a.m.)', fontsize=16, weight='bold')
plt.show()
plt.savefig(path_model_evaluation + '/rmse_world_am.png')
plt.close()


# 1.3.3 Slope of AM
fig = plt.figure(figsize=(12, 6), facecolor='w', edgecolor='k', dpi=150)
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, hspace=0.2, wspace=0.2)
for ipt in range(len(monthname)//3):
    ax = fig.add_subplot(2, 2, ipt+1, projection=ccrs.PlateCarree())
    ax.add_feature(shape_world, linewidth=0.5)
    ax.set_extent([-180, 180, -70, 90], ccrs.PlateCarree())
    img = ax.pcolormesh(xx_wrd, yy_wrd, np.nanmean(slope_mat_monthly[:, :, ipt*3:ipt*3+2], axis=2),
                        transform=ccrs.PlateCarree(), vmin=-0.01, vmax=0.01, cmap='bwr')
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = mticker.FixedLocator([90, 45, 0, -45, -90])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
    # ax.set_title(title_content[ipt], pad=20, fontsize=14, weight='bold')
    ax.text(-170, -45, title_content[ipt], fontsize=14, horizontalalignment='left',
            verticalalignment='top', weight='bold')
cbar_ax = fig.add_axes([0.2, 0.04, 0.6, 0.02])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both', pad=0.1, orientation='horizontal')
cbar.ax.tick_params(labelsize=10)
cbar_ax.locator_params(nbins=8)
plt.suptitle('Slope (a.m.)', fontsize=16, weight='bold')
plt.show()
plt.savefig(path_model_evaluation + '/slope_am.png')
plt.close()


# 1.3.4 Difference of metrics of AM
difference_data1 = r2_mat_monthly[:, :, 6] - r2_mat_monthly[:, :, 0]
difference_data2 = r2_mat_monthly[:, :, 18] - r2_mat_monthly[:, :, 12]
difference_data3 = rmse_mat_monthly[:, :, 6] - rmse_mat_monthly[:, :, 0]
difference_data4 = rmse_mat_monthly[:, :, 18] - rmse_mat_monthly[:, :, 12]
difference_data = np.stack((difference_data1, difference_data2, difference_data3, difference_data4))

# title_content = ['$\Delta \mathregular{R^2}$ (a.m.)', '$\Delta \mathregular{R^2}$ (p.m.)',
#                  '$\Delta$RMSE (a.m.)', '$\Delta$RMSE (p.m.)']
title_content = ['$\Delta \mathregular{R^2}$ (a.m.)', '$\Delta$RMSE (a.m.)']
fig = plt.figure(figsize=(8, 10), facecolor='w', edgecolor='k', dpi=150)
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.85, hspace=0.25, wspace=0.2)
for ipt in range(1):
    # Delta R^2
    ax1 = fig.add_subplot(2, 1, ipt+1, projection=ccrs.PlateCarree())
    ax1.add_feature(shape_world, linewidth=0.5)
    ax1.set_extent([-180, 180, -70, 90], ccrs.PlateCarree())
    img1 = ax1.pcolormesh(xx_wrd, yy_wrd, difference_data[ipt, :, :], transform=ccrs.PlateCarree(),
                        vmin=-0.4, vmax=0.4, cmap='bwr')
    gl = ax1.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = mticker.FixedLocator([90, 45, 0, -45, -90])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}
    cbar1 = plt.colorbar(img1, extend='both', orientation='horizontal', aspect=50, pad=0.1, shrink=0.75)
    cbar1.ax.tick_params(labelsize=12)
    cbar1.ax.locator_params(nbins=8)
    ax1.set_title(title_content[ipt], pad=20, fontsize=16, weight='bold')

    # Delta RMSE
    ax2 = fig.add_subplot(2, 1, ipt+2, projection=ccrs.PlateCarree())
    ax2.add_feature(shape_world, linewidth=0.5)
    ax2.set_extent([-180, 180, -70, 90], ccrs.PlateCarree())
    img2 = ax2.pcolormesh(xx_wrd, yy_wrd, difference_data[ipt+2, :, :], transform=ccrs.PlateCarree(),
                        vmin=-0.06, vmax=0.06, cmap='bwr')
    gl = ax2.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = mticker.FixedLocator([90, 45, 0, -45, -90])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}
    cbar2 = plt.colorbar(img2, extend='both', orientation='horizontal', aspect=50, pad=0.1, shrink=0.75)
    cbar2.set_label('$\mathregular{(m^3/m^3)}$', fontsize=12, x=0.95)
    cbar2.ax.tick_params(labelsize=12)
    cbar2.ax.locator_params(nbins=6)
    ax2.set_title(title_content[ipt+1], pad=20, fontsize=16, weight='bold')

plt.suptitle('Difference (July - January)', fontsize=20, weight='bold')
plt.show()
plt.savefig(path_model_evaluation + '/delta.png')
plt.close()


########################################################################################################################
# 2 SMAP SM maps (Worldwide)

# 2.1 Composite the data of the first 16 days of one specific month
# Load in SMAP data
year_plt = [2021]
month_plt = list([8])
days_begin = 1
days_end = 9
days_n = days_end - days_begin + 1

matsize_9km = [len(month_plt), len(lat_world_ease_9km), len(lon_world_ease_9km)]
smap_9km_mean_1_all = np.empty(matsize_9km, dtype='float32')
smap_9km_mean_1_all[:] = np.nan
smap_9km_mean_2_all = np.copy(smap_9km_mean_1_all)
smap_9km_mean_3_all = np.copy(smap_9km_mean_1_all)

# 9 km SMAP SM
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        hdf_file_smap_9km = path_smap + '/9km' + '/smap_sm_9km_' + str(iyr) + monthname[month_plt[imo]-1] + '.hdf5'
        f_read_smap_9km = h5py.File(hdf_file_smap_9km, "r")
        varname_list_smap_9km = list(f_read_smap_9km.keys())

        smap_9km_load = np.empty((len(lat_world_ease_9km), len(lon_world_ease_9km), days_n*2))
        smap_9km_load[:] = np.nan
        for idt in range(days_begin-1, days_end):
            smap_9km_load[:, :, 2*idt+0] = f_read_smap_9km[varname_list_smap_9km[0]][:, :, idt] # AM
            smap_9km_load[:, :, 2*idt+1] = f_read_smap_9km[varname_list_smap_9km[1]][:, :, idt] # PM
        f_read_smap_9km.close()

        smap_9km_mean_1 = np.nanmean(smap_9km_load[:, :, :days_n//3], axis=2)
        smap_9km_mean_2 = np.nanmean(smap_9km_load[:, :, days_n//3:days_n//3*2], axis=2)
        smap_9km_mean_3 = np.nanmean(smap_9km_load[:, :, days_n //3*2:], axis=2)
        del(smap_9km_load)

        smap_9km_mean_1_all[imo, :, :] = smap_9km_mean_1
        smap_9km_mean_2_all[imo, :, :] = smap_9km_mean_2
        smap_9km_mean_3_all[imo, :, :] = smap_9km_mean_3
        del(smap_9km_mean_1, smap_9km_mean_2, smap_9km_mean_3)
        print(imo)


# Load in SMAP 1 km SM (Gap-filled)
smap_1km_agg_stack = np.empty((len(lat_world_ease_9km), len(lon_world_ease_9km), days_n*2))
smap_1km_agg_stack[:] = np.nan
smap_1km_mean_1_all = np.empty(matsize_9km, dtype='float32')
smap_1km_mean_1_all[:] = np.nan
smap_1km_mean_2_all = np.copy(smap_1km_mean_1_all)
smap_1km_mean_3_all = np.copy(smap_1km_mean_1_all)

for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        for idt in range(days_n):
            str_date = str(iyr) + '-' + monthname[month_plt[imo]-1] + '-' + str(idt+1).zfill(2)
            str_doy = str(datetime.datetime.strptime(str_date, '%Y-%m-%d').date().timetuple().tm_yday)
            tif_file_smap_1km = path_smap + '/1km/' + str(iyr) + '/smap_sm_1km_ds_' + str(iyr) + str_doy.zfill(3) + '.tif'
            src_tf = gdal.Open(tif_file_smap_1km)
            src_tf_arr = src_tf.ReadAsArray().astype(np.float32)

            # Aggregate to 9 km
            for ilr in range(2):
                src_tf_arr_1layer = src_tf_arr[ilr, :, :]
                smap_sm_1km_agg = np.array\
                    ([np.nanmean(src_tf_arr_1layer[row_world_ease_9km_from_1km_ind[x], :], axis=0)
                        for x in range(len(lat_world_ease_9km))])
                smap_sm_1km_agg = np.array\
                    ([np.nanmean(smap_sm_1km_agg[:, col_world_ease_9km_from_1km_ind[y]], axis=1)
                        for y in range(len(lon_world_ease_9km))])
                smap_sm_1km_agg = np.fliplr(np.rot90(smap_sm_1km_agg, 3))
                smap_1km_agg_stack[:, :, 2*idt+ilr] = smap_sm_1km_agg
                del(smap_sm_1km_agg, src_tf_arr_1layer)

            print(str_date)
            del(src_tf_arr)

        smap_1km_mean_1 = np.nanmean(smap_1km_agg_stack[:, :, :days_n//3], axis=2)
        smap_1km_mean_2 = np.nanmean(smap_1km_agg_stack[:, :, days_n//3:days_n//3*2], axis=2)
        smap_1km_mean_3 = np.nanmean(smap_1km_agg_stack[:, :, days_n //3*2:], axis=2)

        smap_1km_mean_1_all[imo, :, :] = smap_1km_mean_1
        smap_1km_mean_2_all[imo, :, :] = smap_1km_mean_2
        smap_1km_mean_3_all[imo, :, :] = smap_1km_mean_3
        del(smap_1km_mean_1, smap_1km_mean_2, smap_1km_mean_3)


# Load in SMAP 1 km SM (Original)
smap_1km_agg_stack_ori = np.empty((len(lat_world_ease_9km), len(lon_world_ease_9km), days_n*2))
smap_1km_agg_stack_ori[:] = np.nan
smap_1km_mean_1_all_ori = np.empty(matsize_9km, dtype='float32')
smap_1km_mean_1_all_ori[:] = np.nan
smap_1km_mean_2_all_ori = np.copy(smap_1km_mean_1_all_ori)
smap_1km_mean_3_all_ori = np.copy(smap_1km_mean_1_all_ori)

for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        for idt in range(days_n):
            str_date = str(iyr) + '-' + monthname[month_plt[imo]-1] + '-' + str(idt+1).zfill(2)
            str_doy = str(datetime.datetime.strptime(str_date, '%Y-%m-%d').date().timetuple().tm_yday)
            tif_file_smap_1km = path_smap + '/1km/gldas_old_data/' + str(iyr) + '/smap_sm_1km_ds_' + str(iyr) + \
                                str_doy.zfill(3) + '.tif'
            src_tf = gdal.Open(tif_file_smap_1km)
            src_tf_arr = src_tf.ReadAsArray().astype(np.float32)

            # Aggregate to 9 km
            for ilr in range(2):
                src_tf_arr_1layer = src_tf_arr[ilr, :, :]
                smap_sm_1km_agg = np.array\
                    ([np.nanmean(src_tf_arr_1layer[row_world_ease_9km_from_1km_ind[x], :], axis=0)
                        for x in range(len(lat_world_ease_9km))])
                smap_sm_1km_agg = np.array\
                    ([np.nanmean(smap_sm_1km_agg[:, col_world_ease_9km_from_1km_ind[y]], axis=1)
                        for y in range(len(lon_world_ease_9km))])
                smap_sm_1km_agg = np.fliplr(np.rot90(smap_sm_1km_agg, 3))
                smap_1km_agg_stack_ori[:, :, 2*idt+ilr] = smap_sm_1km_agg
                del(smap_sm_1km_agg, src_tf_arr_1layer)

            print(str_date)
            del(src_tf_arr)

        smap_1km_mean_1_ori = np.nanmean(smap_1km_agg_stack_ori[:, :, :days_n//3], axis=2)
        smap_1km_mean_2_ori = np.nanmean(smap_1km_agg_stack_ori[:, :, days_n//3:days_n//3*2], axis=2)
        smap_1km_mean_3_ori = np.nanmean(smap_1km_agg_stack_ori[:, :, days_n //3*2:], axis=2)

        smap_1km_mean_1_all_ori[imo, :, :] = smap_1km_mean_1_ori
        smap_1km_mean_2_all_ori[imo, :, :] = smap_1km_mean_2_ori
        smap_1km_mean_3_all_ori[imo, :, :] = smap_1km_mean_3_ori
        del(smap_1km_mean_1_ori, smap_1km_mean_2_ori, smap_1km_mean_3_ori)

smap_data_stack = np.stack((smap_1km_mean_1_all, smap_1km_mean_1_all_ori, smap_1km_mean_2_all, smap_1km_mean_2_all_ori,
                            smap_1km_mean_3_all, smap_1km_mean_3_all_ori))

# # Save and load the data
# with h5py.File(path_model_evaluation + '/smap_data_stack.hdf5', 'w') as f:
#     f.create_dataset('smap_data_stack', data=smap_data_stack)
# f.close()



# 2.2 Maps of the world

# f_read = h5py.File(path_model_evaluation + '/smap_data_stack.hdf5', "r")
# smap_data_stack = f_read['smap_data_stack'][()]
# f_read.close()

xx_wrd, yy_wrd = np.meshgrid(lon_world_ease_9km, lat_world_ease_9km) # Create the map matrix
shape_world = ShapelyFeature(Reader(path_gis_data + '/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
shp_world_extent = [-180.0, 180.0, -70, 90]


# July
title_content = ['1 km(GF)\n(Jan 1-3 2020)', '1 km\n(Jan 1-3 2020)', '1 km(GF)\n(Jan 4-6 2020)', '1 km\n(Jan 4-6 2020)',
                 '1 km(GF)\n(Jan 7-9 2020)', '1 km\n(Jan 7-9 2020)']
# title_content = ['1 km(GF)\n(July 1-3 2020)', '1 km\n(July 1-3 2020)', '1 km(GF)\n(July 4-6 2020)', '1 km\n(July 4-6 2020)',
#                  '1 km(GF)\n(July 7-9 2020)', '1 km\n(July 7-9 2020)']
fig = plt.figure(figsize=(12, 9), facecolor='w', edgecolor='k', dpi=150)
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95, hspace=0.2, wspace=0.2)
for ipt in range(3):
    # 1 km
    ax = fig.add_subplot(3, 2, ipt*2+1, projection=ccrs.PlateCarree())
    ax.add_feature(shape_world, linewidth=0.5)
    img = ax.pcolormesh(xx_wrd, yy_wrd, smap_data_stack[ipt * 2, 0, :, :], vmin=0, vmax=0.5, cmap='Spectral')
    ax.set_extent([180, -180, -70, 90], ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = mticker.FixedLocator([90, 45, 0, -45, -90])
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # cbar = plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
    # cbar.ax.tick_params(labelsize=15)
    # cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=15, x=0.95)
    # ax.set_title(title_content[ipt], pad=12, fontsize=17, weight='bold')
    ax.text(-175, -40, title_content[ipt*2], fontsize=11, horizontalalignment='left',
            verticalalignment='top', weight='bold')
    # 1 km (original)
    ax = fig.add_subplot(3, 2, ipt*2+2, projection=ccrs.PlateCarree())
    ax.add_feature(shape_world, linewidth=0.5)
    img = ax.pcolormesh(xx_wrd, yy_wrd, smap_data_stack[ipt * 2 + 1, 0, :, :], vmin=0, vmax=0.5, cmap='Spectral')
    ax.set_extent([180, -180, -70, 90], ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = mticker.FixedLocator([90, 45, 0, -45, -90])
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # cbar = plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
    # cbar.ax.tick_params(labelsize=15)
    # cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=15, x=0.95)
    # ax.set_title(title_content[ipt], pad=12, fontsize=17, weight='bold')
    ax.text(-175, -40, title_content[ipt*2+1], fontsize=11, horizontalalignment='left',
            verticalalignment='top', weight='bold')
cbar_ax = fig.add_axes([0.2, 0.04, 0.5, 0.015])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both', pad=0.1, orientation='horizontal')
cbar.ax.tick_params(labelsize=10)
cbar_ax.locator_params(nbins=5)
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=10, x=1.05, y=0.05, labelpad=-15)
# plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.92, hspace=0.25, wspace=0.25)
plt.savefig(path_results + '/sm_comp_jan_ori.png')
plt.close()



########################################################################################################################
# 3. River Basin maps
# 3.1 Sacramento-San Joaquin RB

path_shp_ssj = path_gis_data + '/wrd_riverbasins/Aqueduct_river_basins_SACRAMENTO RIVER - SAN JOAQUIN RIVER'
os.chdir(path_shp_ssj)
shp_ssj_file = "Aqueduct_river_basins_SACRAMENTO RIVER - SAN JOAQUIN RIVER.shp"
shp_ssj_ds = ogr.GetDriverByName("ESRI Shapefile").Open(shp_ssj_file, 0)
shp_ssj_extent = list(shp_ssj_ds.GetLayer().GetExtent())

#Load and subset the region of Sacramento-San Joaquin RB (SMAP 9 km)
[lat_9km_ssj, row_ssj_9km_ind, lon_9km_ssj, col_ssj_9km_ind] = \
    coordtable_subset(lat_world_ease_9km, lon_world_ease_9km,
                      shp_ssj_extent[3], shp_ssj_extent[2], shp_ssj_extent[1], shp_ssj_extent[0])

# Load and subset SMAP 9 km SM of Sacramento-San Joaquin RB
# Fresno wildfire
year_plt = [2020]
month_plt = list([7])
days_begin = 11
days_end = 16
days_n = days_end - days_begin + 1

smap_9km_mean_1_allyear = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        hdf_file_smap_9km = path_smap + '/9km' + '/smap_sm_9km_' + str(iyr) + monthname[month_plt[imo]-1] + '.hdf5'
        f_read_smap_9km = h5py.File(hdf_file_smap_9km, "r")
        varname_list_smap_9km = list(f_read_smap_9km.keys())

        smap_9km_load_1 = f_read_smap_9km[varname_list_smap_9km[0]][
                                           row_ssj_9km_ind[0]:row_ssj_9km_ind[-1] + 1,
                                           col_ssj_9km_ind[0]:col_ssj_9km_ind[-1] + 1, days_begin-1:days_end]  # AM
        smap_9km_load_2 = f_read_smap_9km[varname_list_smap_9km[1]][
                                           row_ssj_9km_ind[0]:row_ssj_9km_ind[-1] + 1,
                                           col_ssj_9km_ind[0]:col_ssj_9km_ind[-1] + 1, days_begin-1:days_end]  # PM
        f_read_smap_9km.close()

        smap_9km_mean_1 = np.nanmean(np.stack((smap_9km_load_1, smap_9km_load_2), axis=3), axis=3)
        smap_9km_mean_1_allyear.append(smap_9km_mean_1)

        del(smap_9km_load_1, smap_9km_mean_1)
        print(monthname[month_plt[imo]-1])

smap_9km_data_stack_ssj = np.squeeze(np.array(smap_9km_mean_1_allyear))
del(smap_9km_mean_1_allyear)
smap_9km_data_stack_ssj = np.transpose(smap_9km_data_stack_ssj, (2, 0, 1))


#Load and subset the region of Sacramento-San Joaquin RB (SMAP 1 km)
[lat_1km_ssj, row_ssj_1km_ind, lon_1km_ssj, col_ssj_1km_ind] = \
    coordtable_subset(lat_world_ease_1km, lon_world_ease_1km,
                      shp_ssj_extent[3], shp_ssj_extent[2], shp_ssj_extent[1], shp_ssj_extent[0])

smap_1km_mean_1_allyear = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        smap_1km_load_1_stack = []
        for idt in range(days_begin, days_end+1):
            str_date = str(iyr) + '-' + monthname[month_plt[imo]-1] + '-' + str(idt).zfill(2)
            str_doy = str(datetime.datetime.strptime(str_date, '%Y-%m-%d').date().timetuple().tm_yday)
            tif_file_smap_1km = path_smap + '/1km/gldas/' + str(iyr) + '/smap_sm_1km_ds_' + str(iyr) + str_doy.zfill(3) + '.tif'
            src_tf = gdal.Open(tif_file_smap_1km)
            src_tf_arr = src_tf.ReadAsArray()[:, row_ssj_1km_ind[0]:row_ssj_1km_ind[-1]+1,
                         col_ssj_1km_ind[0]:col_ssj_1km_ind[-1]+1]
            smap_1km_load_1 = np.nanmean(src_tf_arr, axis=0)
            smap_1km_load_1_stack.append(smap_1km_load_1)

            print(str_date)
            del(src_tf_arr, smap_1km_load_1)

        smap_1km_load_1_stack = np.stack(smap_1km_load_1_stack)
        smap_1km_mean_1_allyear.append(smap_1km_load_1_stack)
        del(smap_1km_load_1_stack)

smap_1km_data_stack_ssj = np.squeeze(np.array(smap_1km_mean_1_allyear))

# 1 km (original)
smap_1km_mean_1_allyear_ori = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        smap_1km_load_1_stack = []
        for idt in range(days_begin, days_end+1):
            str_date = str(iyr) + '-' + monthname[month_plt[imo]-1] + '-' + str(idt).zfill(2)
            str_doy = str(datetime.datetime.strptime(str_date, '%Y-%m-%d').date().timetuple().tm_yday)
            tif_file_smap_1km = path_smap + '/1km/gldas_old_data/' + str(iyr) + '/smap_sm_1km_ds_' + str(iyr) + \
                                str_doy.zfill(3) + '.tif'
            src_tf = gdal.Open(tif_file_smap_1km)
            src_tf_arr = src_tf.ReadAsArray()[:, row_ssj_1km_ind[0]:row_ssj_1km_ind[-1]+1,
                         col_ssj_1km_ind[0]:col_ssj_1km_ind[-1]+1]
            smap_1km_load_1 = np.nanmean(src_tf_arr, axis=0)
            smap_1km_load_1_stack.append(smap_1km_load_1)

            print(str_date)
            del(src_tf_arr, smap_1km_load_1)

        smap_1km_load_1_stack = np.stack(smap_1km_load_1_stack)
        smap_1km_mean_1_allyear_ori.append(smap_1km_load_1_stack)
        del(smap_1km_load_1_stack)

smap_1km_data_stack_ssj_ori = np.squeeze(np.array(smap_1km_mean_1_allyear_ori))


with h5py.File(path_model_evaluation + '/smap_ssj_sm.hdf5', 'w') as f:
    f.create_dataset('smap_9km_data_stack_ssj', data=smap_9km_data_stack_ssj)
    f.create_dataset('smap_1km_data_stack_ssj', data=smap_1km_data_stack_ssj)
f.close()

# Read the map data
f_read = h5py.File(path_model_evaluation + "/smap_ssj_sm.hdf5", "r")
varname_read_list = ['smap_9km_data_stack_ssj', 'smap_1km_data_stack_ssj']
for x in range(len(varname_read_list)):
    var_obj = f_read[varname_read_list[x]][()]
    exec(varname_read_list[x] + '= var_obj')
    del(var_obj)
f_read.close()



# Subplot maps
output_crs = 'EPSG:4326'
shapefile_ssj = fiona.open(path_shp_ssj + '/' + shp_ssj_file, 'r')
crop_shape_ssj = [feature["geometry"] for feature in shapefile_ssj]

# Subset and reproject the SMAP SM data at watershed
# 1 km
masked_ds_ssj_1km_all = []
for n in range(smap_1km_data_stack_ssj.shape[0]):
    sub_window_ssj_1km = Window(col_ssj_1km_ind[0], row_ssj_1km_ind[0], len(col_ssj_1km_ind), len(row_ssj_1km_ind))
    kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                      'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
    smap_sm_ssj_1km_output = sub_n_reproj(smap_1km_data_stack_ssj[n, :, :], kwargs_1km_sub, sub_window_ssj_1km, output_crs)

    masked_ds_ssj_1km, mask_transform_ds_ssj_1km = mask(dataset=smap_sm_ssj_1km_output, shapes=crop_shape_ssj, crop=True)
    masked_ds_ssj_1km[np.where(masked_ds_ssj_1km == 0)] = np.nan
    masked_ds_ssj_1km = masked_ds_ssj_1km.squeeze()

    masked_ds_ssj_1km_all.append(masked_ds_ssj_1km)

masked_ds_ssj_1km_all = np.asarray(masked_ds_ssj_1km_all)

# 1 km (original)
masked_ds_ssj_1km_all_ori = []
for n in range(smap_1km_data_stack_ssj_ori.shape[0]):
    sub_window_ssj_1km = Window(col_ssj_1km_ind[0], row_ssj_1km_ind[0], len(col_ssj_1km_ind), len(row_ssj_1km_ind))
    kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                      'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
    smap_sm_ssj_1km_output = sub_n_reproj(smap_1km_data_stack_ssj_ori[n, :, :], kwargs_1km_sub, sub_window_ssj_1km, output_crs)

    masked_ds_ssj_1km, mask_transform_ds_ssj_1km = mask(dataset=smap_sm_ssj_1km_output, shapes=crop_shape_ssj, crop=True)
    masked_ds_ssj_1km[np.where(masked_ds_ssj_1km == 0)] = np.nan
    masked_ds_ssj_1km = masked_ds_ssj_1km.squeeze()

    masked_ds_ssj_1km_all_ori.append(masked_ds_ssj_1km)

masked_ds_ssj_1km_all_ori = np.asarray(masked_ds_ssj_1km_all_ori)


# 9 km
masked_ds_ssj_9km_all = []
for n in range(smap_9km_data_stack_ssj.shape[0]):
    sub_window_ssj_9km = Window(col_ssj_9km_ind[0], row_ssj_9km_ind[0], len(col_ssj_9km_ind), len(row_ssj_9km_ind))
    kwargs_9km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_9km),
                      'height': len(lat_world_ease_9km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(9008.05, 0.0, -17367530.44516138, 0.0, -9008.05, 7314540.79258289)}
    smap_sm_ssj_9km_output = sub_n_reproj(smap_9km_data_stack_ssj[n, :, :], kwargs_9km_sub, sub_window_ssj_9km, output_crs)

    masked_ds_ssj_9km, mask_transform_ds_ssj_9km = mask(dataset=smap_sm_ssj_9km_output, shapes=crop_shape_ssj, crop=True)
    masked_ds_ssj_9km[np.where(masked_ds_ssj_9km == 0)] = np.nan
    masked_ds_ssj_9km = masked_ds_ssj_9km.squeeze()

    masked_ds_ssj_9km_all.append(masked_ds_ssj_9km)

masked_ds_ssj_9km_all = np.asarray(masked_ds_ssj_9km_all)
masked_ds_ssj_9km_all[masked_ds_ssj_9km_all >= 0.5] = np.nan

# Make the subplot maps
title_content = ['1 km(GF)\n(July 11 2020)', '1 km\n(July 11 2020)', '1 km(GF)\n(July 12 2020)', '1 km\n(July 12 2020)',
                 '1 km(GF)\n(July 13 2020)', '1 km\n(July 13 2020)', '1 km(GF)\n(July 14 2020)', '1 km\n(July 14 2020)',
                 '1 km(GF)\n(July 15 2020)', '1 km\n(July 15 2020)', '1 km(GF)\n(July 16 2020)', '1 km\n(July 16 2020)']
feature_shp_ssj = ShapelyFeature(Reader(path_shp_ssj + '/' + shp_ssj_file).geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
extent_ssj = np.array(smap_sm_ssj_1km_output.bounds)
extent_ssj = extent_ssj[[0, 2, 1, 3]]

fig = plt.figure(figsize=(7, 12), facecolor='w', edgecolor='k', dpi=150)
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95, hspace=0.25, wspace=0.2)
for ipt in range(3):
    # 1 km
    ax = fig.add_subplot(4, 3, ipt+1, projection=ccrs.PlateCarree())
    ax.add_feature(feature_shp_ssj)
    img = ax.imshow(masked_ds_ssj_1km_all[ipt, :, :], origin='upper', vmin=0, vmax=0.5, cmap='Spectral',
               extent=extent_ssj)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=2)
    gl.ylocator = mticker.MultipleLocator(base=2)
    gl.xlabel_style = {'size': 6}
    gl.ylabel_style = {'size': 6}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.text(-123, 36.7, title_content[ipt*2], fontsize=8, horizontalalignment='left',
            verticalalignment='top', weight='bold')
    # 1 km (original)
    ax = fig.add_subplot(4, 3, ipt+4, projection=ccrs.PlateCarree())
    ax.add_feature(feature_shp_ssj)
    img = ax.imshow(masked_ds_ssj_1km_all_ori[ipt, :, :], origin='upper', vmin=0, vmax=0.5, cmap='Spectral',
               extent=extent_ssj)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=2)
    gl.ylocator = mticker.MultipleLocator(base=2)
    gl.xlabel_style = {'size': 6}
    gl.ylabel_style = {'size': 6}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.text(-123, 36.7, title_content[ipt*2+1], fontsize=8, horizontalalignment='left',
            verticalalignment='top', weight='bold')
    # 1 km (2)
    ax = fig.add_subplot(4, 3, ipt+7, projection=ccrs.PlateCarree())
    ax.add_feature(feature_shp_ssj)
    img = ax.imshow(masked_ds_ssj_1km_all[ipt+3, :, :], origin='upper', vmin=0, vmax=0.5, cmap='Spectral',
               extent=extent_ssj)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=2)
    gl.ylocator = mticker.MultipleLocator(base=2)
    gl.xlabel_style = {'size': 6}
    gl.ylabel_style = {'size': 6}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.text(-123, 36.7, title_content[ipt*2+6], fontsize=8, horizontalalignment='left',
            verticalalignment='top', weight='bold')
    # 1 km (2, original)
    ax = fig.add_subplot(4, 3, ipt+10, projection=ccrs.PlateCarree())
    ax.add_feature(feature_shp_ssj)
    img = ax.imshow(masked_ds_ssj_1km_all_ori[ipt+3, :, :], origin='upper', vmin=0, vmax=0.5, cmap='Spectral',
               extent=extent_ssj)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=2)
    gl.ylocator = mticker.MultipleLocator(base=2)
    gl.xlabel_style = {'size': 6}
    gl.ylabel_style = {'size': 6}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.text(-123, 36.7, title_content[ipt*2+1+6], fontsize=8, horizontalalignment='left',
            verticalalignment='top', weight='bold')

cbar_ax = fig.add_axes([0.2, 0.04, 0.5, 0.015])
cbar = plt.colorbar(img, cax=cbar_ax, extend='both', orientation='horizontal', pad=0.1)
cbar.ax.tick_params(labelsize=7)
cbar_ax.locator_params(nbins=5)
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=7, x=1.08, y=0.05, labelpad=-15)
plt.savefig(path_results + '/sm_comp_ssj_1_ori.png')
plt.close()


########################################################################################################################
# 3.2 Ganga-Brahmaputra RB
path_shp_gb = path_gis_data + '/wrd_riverbasins/Aqueduct_river_basins_GANGES - BRAHMAPUTRA'
os.chdir(path_shp_gb)
shp_gb_file = "Aqueduct_river_basins_GANGES - BRAHMAPUTRA.shp"
shp_gb_ds = ogr.GetDriverByName("ESRI Shapefile").Open(shp_gb_file, 0)
shp_gb_extent = list(shp_gb_ds.GetLayer().GetExtent())

#Load and subset the region of Ganga-Brahmaputra RB (SMAP 9 km)
[lat_9km_gb, row_gb_9km_ind, lon_9km_gb, col_gb_9km_ind] = \
    coordtable_subset(lat_world_ease_9km, lon_world_ease_9km, shp_gb_extent[3], shp_gb_extent[2], shp_gb_extent[1], shp_gb_extent[0])

# Load and subset SMAP 9 km SM of Ganga-Brahmaputra RB
year_plt = [2020]
month_plt = list([7])
days_begin = 10
days_end = 15
days_n = days_end - days_begin + 1

smap_9km_mean_1_allyear = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        hdf_file_smap_9km = path_smap + '/9km' + '/smap_sm_9km_' + str(iyr) + monthname[month_plt[imo]-1] + '.hdf5'
        f_read_smap_9km = h5py.File(hdf_file_smap_9km, "r")
        varname_list_smap_9km = list(f_read_smap_9km.keys())

        # smap_9km_load = smap_9km_init
        smap_9km_load_1 = f_read_smap_9km[varname_list_smap_9km[0]][
                                           row_gb_9km_ind[0]:row_gb_9km_ind[-1] + 1,
                                           col_gb_9km_ind[0]:col_gb_9km_ind[-1] + 1, days_begin-1:days_end]  # AM
        smap_9km_load_2 = f_read_smap_9km[varname_list_smap_9km[1]][
                                           row_gb_9km_ind[0]:row_gb_9km_ind[-1] + 1,
                                           col_gb_9km_ind[0]:col_gb_9km_ind[-1] + 1, days_begin-1:days_end]  # PM
        f_read_smap_9km.close()

        smap_9km_mean_1 = np.nanmean(np.stack((smap_9km_load_1, smap_9km_load_2), axis=3), axis=3)
        smap_9km_mean_1_allyear.append(smap_9km_mean_1)

        del(smap_9km_load_1, smap_9km_mean_1)
        print(monthname[month_plt[imo]-1])

smap_9km_data_stack_gb = np.squeeze(np.array(smap_9km_mean_1_allyear))
del(smap_9km_mean_1_allyear)
smap_9km_data_stack_gb = np.transpose(smap_9km_data_stack_gb, (2, 0, 1))


#Load and subset the region of Ganga-Brahmaputra RB (SMAP 1 km)
[lat_1km_gb, row_gb_1km_ind, lon_1km_gb, col_gb_1km_ind] = \
    coordtable_subset(lat_world_ease_1km, lon_world_ease_1km, shp_gb_extent[3], shp_gb_extent[2], shp_gb_extent[1], shp_gb_extent[0])

smap_1km_mean_1_allyear = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        smap_1km_load_1_stack = []
        for idt in range(days_begin, days_end+1):
            str_date = str(iyr) + '-' + monthname[month_plt[imo]-1] + '-' + str(idt).zfill(2)
            str_doy = str(datetime.datetime.strptime(str_date, '%Y-%m-%d').date().timetuple().tm_yday)
            tif_file_smap_1km = path_smap + '/1km/gldas/' + str(iyr) + '/smap_sm_1km_ds_' + str(iyr) + str_doy.zfill(3) + '.tif'
            src_tf = gdal.Open(tif_file_smap_1km)
            src_tf_arr = src_tf.ReadAsArray()[:, row_gb_1km_ind[0]:row_gb_1km_ind[-1]+1,
                         col_gb_1km_ind[0]:col_gb_1km_ind[-1]+1]
            smap_1km_load_1 = np.nanmean(src_tf_arr, axis=0)
            smap_1km_load_1_stack.append(smap_1km_load_1)

            print(str_date)
            del(src_tf_arr, smap_1km_load_1)

        smap_1km_load_1_stack = np.stack(smap_1km_load_1_stack)
        smap_1km_mean_1_allyear.append(smap_1km_load_1_stack)
        del(smap_1km_load_1_stack)

smap_1km_data_stack_gb = np.squeeze(np.array(smap_1km_mean_1_allyear))


# 1 km (original)
smap_1km_mean_1_allyear_ori = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        smap_1km_load_1_stack = []
        for idt in range(days_begin, days_end+1):
            str_date = str(iyr) + '-' + monthname[month_plt[imo]-1] + '-' + str(idt).zfill(2)
            str_doy = str(datetime.datetime.strptime(str_date, '%Y-%m-%d').date().timetuple().tm_yday)
            tif_file_smap_1km = path_smap + '/1km/gldas_old_data/' + str(iyr) + '/smap_sm_1km_ds_' + str(iyr) + \
                                str_doy.zfill(3) + '.tif'
            src_tf = gdal.Open(tif_file_smap_1km)
            src_tf_arr = src_tf.ReadAsArray()[:, row_gb_1km_ind[0]:row_gb_1km_ind[-1]+1,
                         col_gb_1km_ind[0]:col_gb_1km_ind[-1]+1]
            smap_1km_load_1 = np.nanmean(src_tf_arr, axis=0)
            smap_1km_load_1_stack.append(smap_1km_load_1)

            print(str_date)
            del(src_tf_arr, smap_1km_load_1)

        smap_1km_load_1_stack = np.stack(smap_1km_load_1_stack)
        smap_1km_mean_1_allyear_ori.append(smap_1km_load_1_stack)
        del(smap_1km_load_1_stack)

smap_1km_data_stack_gb_ori = np.squeeze(np.array(smap_1km_mean_1_allyear_ori))


with h5py.File(path_model_evaluation + '/smap_gb_sm.hdf5', 'w') as f:
    f.create_dataset('smap_9km_data_stack_gb', data=smap_9km_data_stack_gb)
    f.create_dataset('smap_1km_data_stack_gb', data=smap_1km_data_stack_gb)
f.close()


# Read the map data
f_read = h5py.File(path_model_evaluation + "/smap_gb_sm.hdf5", "r")
varname_read_list = ['smap_9km_data_stack_gb', 'smap_1km_data_stack_gb']
for x in range(len(varname_read_list)):
    var_obj = f_read[varname_read_list[x]][()]
    exec(varname_read_list[x] + '= var_obj')
    del(var_obj)
f_read.close()


# Subplot maps
shapefile_gb = fiona.open(path_shp_gb + '/' + shp_gb_file, 'r')
crop_shape_gb = [feature["geometry"] for feature in shapefile_gb]
# shp_gb_extent = list(shapefile_gb.bounds)
output_crs = 'EPSG:4326'

# Subset and reproject the SMAP SM data at watershed
# 1 km
masked_ds_gb_1km_all = []
for n in range(smap_1km_data_stack_gb.shape[0]):
    sub_window_gb_1km = Window(col_gb_1km_ind[0], row_gb_1km_ind[0], len(col_gb_1km_ind), len(row_gb_1km_ind))
    kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                      'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
    smap_sm_gb_1km_output = sub_n_reproj(smap_1km_data_stack_gb[n, :, :], kwargs_1km_sub, sub_window_gb_1km, output_crs)

    masked_ds_gb_1km, mask_transform_ds_gb_1km = mask(dataset=smap_sm_gb_1km_output, shapes=crop_shape_gb, crop=True)
    masked_ds_gb_1km[np.where(masked_ds_gb_1km == 0)] = np.nan
    masked_ds_gb_1km = masked_ds_gb_1km.squeeze()

    masked_ds_gb_1km_all.append(masked_ds_gb_1km)

masked_ds_gb_1km_all = np.asarray(masked_ds_gb_1km_all)

# 1 km (original)
masked_ds_gb_1km_all_ori = []
for n in range(smap_1km_data_stack_gb.shape[0]):
    sub_window_gb_1km = Window(col_gb_1km_ind[0], row_gb_1km_ind[0], len(col_gb_1km_ind), len(row_gb_1km_ind))
    kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                      'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
    smap_sm_gb_1km_output = sub_n_reproj(smap_1km_data_stack_gb_ori[n, :, :], kwargs_1km_sub, sub_window_gb_1km, output_crs)

    masked_ds_gb_1km, mask_transform_ds_gb_1km = mask(dataset=smap_sm_gb_1km_output, shapes=crop_shape_gb, crop=True)
    masked_ds_gb_1km[np.where(masked_ds_gb_1km == 0)] = np.nan
    masked_ds_gb_1km = masked_ds_gb_1km.squeeze()

    masked_ds_gb_1km_all_ori.append(masked_ds_gb_1km)

masked_ds_gb_1km_all_ori = np.asarray(masked_ds_gb_1km_all_ori)

# 9 km
masked_ds_gb_9km_all = []
for n in range(smap_9km_data_stack_gb.shape[0]):
    sub_window_gb_9km = Window(col_gb_9km_ind[0], row_gb_9km_ind[0], len(col_gb_9km_ind), len(row_gb_9km_ind))
    kwargs_9km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_9km),
                      'height': len(lat_world_ease_9km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(9008.05, 0.0, -17367530.44516138, 0.0, -9008.05, 7314540.79258289)}
    smap_sm_gb_9km_output = sub_n_reproj(smap_9km_data_stack_gb[n, :, :], kwargs_9km_sub, sub_window_gb_9km, output_crs)

    masked_ds_gb_9km, mask_transform_ds_gb_9km = mask(dataset=smap_sm_gb_9km_output, shapes=crop_shape_gb, crop=True)
    masked_ds_gb_9km[np.where(masked_ds_gb_9km == 0)] = np.nan
    masked_ds_gb_9km = masked_ds_gb_9km.squeeze()

    masked_ds_gb_9km_all.append(masked_ds_gb_9km)

masked_ds_gb_9km_all = np.asarray(masked_ds_gb_9km_all)
# masked_ds_gb_9km_all[masked_ds_gb_9km_all >= 0.5] = np.nan


# Make the subplot maps
title_content = ['1 km(GF)\n(July 10 2020)', '1 km\n(July 10 2020)', '1 km(GF)\n(July 11 2020)', '1 km\n(July 11 2020)',
                 '1 km(GF)\n(July 12 2020)', '1 km\n(July 12 2020)', '1 km(GF)\n(July 13 2020)', '1 km\n(July 13 2020)',
                 '1 km(GF)\n(July 14 2020)', '1 km\n(July 14 2020)', '1 km(GF)\n(July 15 2020)', '1 km\n(July 15 2020)', ]
feature_shp_gb = ShapelyFeature(Reader(path_shp_gb + '/' + shp_gb_file).geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
extent_gb = np.array(smap_sm_gb_1km_output.bounds)
extent_gb = extent_gb[[0, 2, 1, 3]]

fig = plt.figure(figsize=(7, 10), facecolor='w', edgecolor='k', dpi=200)
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95, hspace=0.25, wspace=0.2)
for ipt in range(6):
    # 1 km
    ax = fig.add_subplot(6, 2, ipt*2+1, projection=ccrs.PlateCarree())
    ax.add_feature(feature_shp_gb)
    # ax.set_title(title_content[ipt*2], pad=20, fontsize=13, fontweight='bold')
    img = ax.imshow(masked_ds_gb_1km_all[ipt, :, :], origin='upper', vmin=0, vmax=0.5, cmap='Spectral',
               extent=extent_gb)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=5)
    gl.ylocator = mticker.MultipleLocator(base=4)
    gl.xlabel_style = {'size': 6}
    gl.ylabel_style = {'size': 6}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.text(97.7, 24.4, title_content[ipt*2], fontsize=7, horizontalalignment='right',
            verticalalignment='top', weight='bold')
    # 1 km (original)
    ax = fig.add_subplot(6, 2, ipt*2+2, projection=ccrs.PlateCarree())
    ax.add_feature(feature_shp_gb)
    # ax.set_title(title_content[ipt*2+1], pad=20, fontsize=13, fontweight='bold')
    img = ax.imshow(masked_ds_gb_1km_all_ori[ipt, :, :], origin='upper', vmin=0, vmax=0.5, cmap='Spectral',
               extent=extent_gb)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=5)
    gl.ylocator = mticker.MultipleLocator(base=4)
    gl.xlabel_style = {'size': 6}
    gl.ylabel_style = {'size': 6}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.text(97.7, 24.4, title_content[ipt*2+1], fontsize=7, horizontalalignment='right',
            verticalalignment='top', weight='bold')

cbar_ax = fig.add_axes([0.2, 0.04, 0.5, 0.015])
cbar = plt.colorbar(img, cax=cbar_ax, extend='both', orientation='horizontal', pad=0.1)
cbar.ax.tick_params(labelsize=7)
cbar_ax.locator_params(nbins=5)
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=7, x=1.08, y=0.05, labelpad=-15)
plt.savefig(path_results + '/sm_comp_gb_1_ori.png')
plt.close()


########################################################################################################################
# 3.3 Murray-Darling RB
path_shp_md = path_gis_data + '/wrd_riverbasins/Aqueduct_river_basins_MURRAY - DARLING'
os.chdir(path_shp_md)
shp_md_file = "Aqueduct_river_basins_MURRAY - DARLING.shp"
shp_md_ds = ogr.GetDriverByName("ESRI Shapefile").Open(shp_md_file, 0)
shp_md_extent = list(shp_md_ds.GetLayer().GetExtent())


#Load and subset the region of Murray-Darling RB (SMAP 9 km)
[lat_9km_md, row_md_9km_ind, lon_9km_md, col_md_9km_ind] = \
    coordtable_subset(lat_world_ease_9km, lon_world_ease_9km, shp_md_extent[3], shp_md_extent[2], shp_md_extent[1], shp_md_extent[0])

# Load and subset SMAP 9 km SM of GB RB
year_plt = [2020]
month_plt = list([2])
days_begin = 6
days_end = 11
days_n = days_end - days_begin + 1

smap_9km_mean_1_allyear = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        hdf_file_smap_9km = path_smap + '/9km' + '/smap_sm_9km_' + str(iyr) + monthname[month_plt[imo]-1] + '.hdf5'
        f_read_smap_9km = h5py.File(hdf_file_smap_9km, "r")
        varname_list_smap_9km = list(f_read_smap_9km.keys())

        # smap_9km_load = smap_9km_init
        smap_9km_load_1 = f_read_smap_9km[varname_list_smap_9km[0]][
                                           row_md_9km_ind[0]:row_md_9km_ind[-1] + 1,
                                           col_md_9km_ind[0]:col_md_9km_ind[-1] + 1, days_begin-1:days_end]  # AM
        smap_9km_load_2 = f_read_smap_9km[varname_list_smap_9km[1]][
                                           row_md_9km_ind[0]:row_md_9km_ind[-1] + 1,
                                           col_md_9km_ind[0]:col_md_9km_ind[-1] + 1, days_begin-1:days_end]  # PM
        f_read_smap_9km.close()

        smap_9km_mean_1 = np.nanmean(np.stack((smap_9km_load_1, smap_9km_load_2), axis=3), axis=3)
        smap_9km_mean_1_allyear.append(smap_9km_mean_1)

        del(smap_9km_load_1, smap_9km_mean_1)
        print(monthname[month_plt[imo]-1])

smap_9km_data_stack_md = np.squeeze(np.array(smap_9km_mean_1_allyear))
del(smap_9km_mean_1_allyear)
smap_9km_data_stack_md = np.transpose(smap_9km_data_stack_md, (2, 0, 1))


#Load and subset the region of Murray-Darling RB (SMAP 1 km)
[lat_1km_md, row_md_1km_ind, lon_1km_md, col_md_1km_ind] = \
    coordtable_subset(lat_world_ease_1km, lon_world_ease_1km, shp_md_extent[3], shp_md_extent[2], shp_md_extent[1], shp_md_extent[0])

smap_1km_mean_1_allyear = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        smap_1km_load_1_stack = []
        for idt in range(days_begin, days_end+1):
            str_date = str(iyr) + '-' + monthname[month_plt[imo]-1] + '-' + str(idt).zfill(2)
            str_doy = str(datetime.datetime.strptime(str_date, '%Y-%m-%d').date().timetuple().tm_yday)
            tif_file_smap_1km = path_smap + '/1km/gldas/' + str(iyr) + '/smap_sm_1km_ds_' + str(iyr) + str_doy.zfill(3) + '.tif'
            src_tf = gdal.Open(tif_file_smap_1km)
            src_tf_arr = src_tf.ReadAsArray()[:, row_md_1km_ind[0]:row_md_1km_ind[-1]+1,
                         col_md_1km_ind[0]:col_md_1km_ind[-1]+1]
            smap_1km_load_1 = np.nanmean(src_tf_arr, axis=0)
            smap_1km_load_1_stack.append(smap_1km_load_1)

            print(str_date)
            del(src_tf_arr, smap_1km_load_1)

        smap_1km_load_1_stack = np.stack(smap_1km_load_1_stack)
        smap_1km_mean_1_allyear.append(smap_1km_load_1_stack)
        del (smap_1km_load_1_stack)

smap_1km_data_stack_md = np.squeeze(np.array(smap_1km_mean_1_allyear))

# 1 km (original)
smap_1km_mean_1_allyear_ori = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        smap_1km_load_1_stack = []
        for idt in range(days_begin, days_end+1):
            str_date = str(iyr) + '-' + monthname[month_plt[imo]-1] + '-' + str(idt).zfill(2)
            str_doy = str(datetime.datetime.strptime(str_date, '%Y-%m-%d').date().timetuple().tm_yday)
            tif_file_smap_1km = path_smap + '/1km/gldas_old_data/' + str(iyr) + '/smap_sm_1km_ds_' + str(iyr) + \
                                str_doy.zfill(3) + '.tif'
            src_tf = gdal.Open(tif_file_smap_1km)
            src_tf_arr = src_tf.ReadAsArray()[:, row_md_1km_ind[0]:row_md_1km_ind[-1]+1,
                         col_md_1km_ind[0]:col_md_1km_ind[-1]+1]
            smap_1km_load_1 = np.nanmean(src_tf_arr, axis=0)
            smap_1km_load_1_stack.append(smap_1km_load_1)

            print(str_date)
            del(src_tf_arr, smap_1km_load_1)

        smap_1km_load_1_stack = np.stack(smap_1km_load_1_stack)
        smap_1km_mean_1_allyear_ori.append(smap_1km_load_1_stack)
        del (smap_1km_load_1_stack)

smap_1km_data_stack_md_ori = np.squeeze(np.array(smap_1km_mean_1_allyear_ori))


with h5py.File(path_model_evaluation + '/smap_md_sm.hdf5', 'w') as f:
    f.create_dataset('smap_9km_data_stack_md', data=smap_9km_data_stack_md)
    f.create_dataset('smap_1km_data_stack_md', data=smap_1km_data_stack_md)
f.close()

# Read the map data
f_read = h5py.File(path_model_evaluation + "/smap_md_sm.hdf5", "r")
varname_read_list = ['smap_9km_data_stack_md', 'smap_1km_data_stack_md']
for x in range(len(varname_read_list)):
    var_obj = f_read[varname_read_list[x]][()]
    exec(varname_read_list[x] + '= var_obj')
    del(var_obj)
f_read.close()


# Subplot maps
# Load in watershed shapefile boundaries
shapefile_md = fiona.open(path_shp_md + '/' + shp_md_file, 'r')
crop_shape_md = [feature["geometry"] for feature in shapefile_md]
# shp_md_extent = list(shapefile_md.bounds)

output_crs = 'EPSG:4326'

# Subset and reproject the SMAP SM data at watershed
# 1 km
masked_ds_md_1km_all = []
for n in range(smap_1km_data_stack_md.shape[0]):
    sub_window_md_1km = Window(col_md_1km_ind[0], row_md_1km_ind[0], len(col_md_1km_ind), len(row_md_1km_ind))
    kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                      'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
    smap_sm_md_1km_output = sub_n_reproj(smap_1km_data_stack_md[n, :, :], kwargs_1km_sub, sub_window_md_1km, output_crs)

    masked_ds_md_1km, mask_transform_ds_md_1km = mask(dataset=smap_sm_md_1km_output, shapes=crop_shape_md, crop=True)
    masked_ds_md_1km[np.where(masked_ds_md_1km == 0)] = np.nan
    masked_ds_md_1km = masked_ds_md_1km.squeeze()

    masked_ds_md_1km_all.append(masked_ds_md_1km)

masked_ds_md_1km_all = np.asarray(masked_ds_md_1km_all)

# 1 km (original)
masked_ds_md_1km_all_ori = []
for n in range(smap_1km_data_stack_md_ori.shape[0]):
    sub_window_md_1km = Window(col_md_1km_ind[0], row_md_1km_ind[0], len(col_md_1km_ind), len(row_md_1km_ind))
    kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                      'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
    smap_sm_md_1km_output = sub_n_reproj(smap_1km_data_stack_md_ori[n, :, :], kwargs_1km_sub, sub_window_md_1km, output_crs)

    masked_ds_md_1km, mask_transform_ds_md_1km = mask(dataset=smap_sm_md_1km_output, shapes=crop_shape_md, crop=True)
    masked_ds_md_1km[np.where(masked_ds_md_1km == 0)] = np.nan
    masked_ds_md_1km = masked_ds_md_1km.squeeze()

    masked_ds_md_1km_all_ori.append(masked_ds_md_1km)

masked_ds_md_1km_all_ori = np.asarray(masked_ds_md_1km_all_ori)

# 9 km
masked_ds_md_9km_all = []
for n in range(smap_9km_data_stack_md.shape[0]):
    sub_window_md_9km = Window(col_md_9km_ind[0], row_md_9km_ind[0], len(col_md_9km_ind), len(row_md_9km_ind))
    kwargs_9km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_9km),
                      'height': len(lat_world_ease_9km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(9008.05, 0.0, -17367530.44516138, 0.0, -9008.05, 7314540.79258289)}
    smap_sm_md_9km_output = sub_n_reproj(smap_9km_data_stack_md[n, :, :], kwargs_9km_sub, sub_window_md_9km, output_crs)

    masked_ds_md_9km, mask_transform_ds_md_9km = mask(dataset=smap_sm_md_9km_output, shapes=crop_shape_md, crop=True)
    masked_ds_md_9km[np.where(masked_ds_md_9km == 0)] = np.nan
    masked_ds_md_9km = masked_ds_md_9km.squeeze()

    masked_ds_md_9km_all.append(masked_ds_md_9km)

masked_ds_md_9km_all = np.asarray(masked_ds_md_9km_all)
# masked_ds_md_9km_all[masked_ds_md_9km_all >= 0.5] = np.nan


# Make the subplot maps
title_content = ['1 km(GF)\n(Feb 6 2020)', '1 km\n(Feb 6 2020)', '1 km(GF)\n(Feb 7 2020)', '1 km\n(Feb 7 2020)',
                 '1 km(GF)\n(Feb 8 2020)', '1 km\n(Feb 8 2020)', '1 km(GF)\n(Feb 9 2020)', '1 km\n(Feb 9 2020)',
                 '1 km(GF)\n(Feb 10 2020)', '1 km\n(Feb 10 2020)', '1 km(GF)\n(Feb 11 2020)', '1 km\n(Feb 11 2020)']
feature_shp_md = ShapelyFeature(Reader(path_shp_md + '/' + shp_md_file).geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
extent_md = np.array(smap_sm_md_1km_output.bounds)
extent_md = extent_md[[0, 2, 1, 3]]

fig = plt.figure(figsize=(4.3, 10), facecolor='w', edgecolor='k', dpi=200)
plt.subplots_adjust(left=0.03, right=0.88, bottom=0.05, top=0.95, hspace=0.2, wspace=0.2)
for ipt in range(6):
    # 1 km
    ax = fig.add_subplot(6, 2, ipt*2+1, projection=ccrs.PlateCarree())
    ax.add_feature(feature_shp_md, linewidth=0.7)
    # ax.set_title(title_content[ipt*2], pad=20, fontsize=13, fontweight='bold')
    img = ax.imshow(masked_ds_md_1km_all[ipt, :, :], origin='upper', vmin=0, vmax=0.5, cmap='Spectral',
               extent=extent_md)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=5)
    gl.ylocator = mticker.MultipleLocator(base=4)
    gl.xlabel_style = {'size': 5}
    gl.ylabel_style = {'size': 5}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.text(139, -25, title_content[ipt*2], fontsize=6, horizontalalignment='left',
            verticalalignment='top', weight='bold')
    # 1 km (original)
    ax = fig.add_subplot(6, 2, ipt*2+2, projection=ccrs.PlateCarree())
    ax.add_feature(feature_shp_md, linewidth=0.7)
    # ax.set_title(title_content[ipt*2+1], pad=20, fontsize=13, fontweight='bold')
    img = ax.imshow(masked_ds_md_1km_all_ori[ipt, :, :], origin='upper', vmin=0, vmax=0.5, cmap='Spectral',
               extent=extent_md)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=5)
    gl.ylocator = mticker.MultipleLocator(base=4)
    gl.xlabel_style = {'size': 5}
    gl.ylabel_style = {'size': 5}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.text(139, -25, title_content[ipt*2+1], fontsize=6, horizontalalignment='left',
            verticalalignment='top', weight='bold')

cbar_ax = fig.add_axes([0.93, 0.2, 0.015, 0.6])
cbar = plt.colorbar(img, cax=cbar_ax, extend='both', orientation='vertical', pad=0.1)
cbar.ax.tick_params(labelsize=5)
cbar_ax.locator_params(nbins=5)
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=5, x=1.08, y=1.05, labelpad=-15, rotation=0)
plt.savefig(path_results + '/sm_comp_md_1_ori.png')
plt.close()


########################################################################################################################
# 3.4 Danube RB
# Read the map data
path_shp_dan = path_gis_data + '/wrd_riverbasins/Aqueduct_river_basins_DANUBE'
os.chdir(path_shp_dan)
shp_dan_file = "Aqueduct_river_basins_DANUBE.shp"
shp_dan_ds = ogr.GetDriverByName("ESRI Shapefile").Open(shp_dan_file, 0)
shp_dan_extent = list(shp_dan_ds.GetLayer().GetExtent())

#Load and subset the region of Danube RB (SMAP 9 km)
[lat_9km_dan, row_dan_9km_ind, lon_9km_dan, col_dan_9km_ind] = \
    coordtable_subset(lat_world_ease_9km, lon_world_ease_9km,
                      shp_dan_extent[3], shp_dan_extent[2], shp_dan_extent[1], shp_dan_extent[0])

# Load and subset SMAP 9 km SM of Danube RB
year_plt = [2020]
month_plt = list([9])
days_begin = 1
days_end = 6
days_n = days_end - days_begin + 1

smap_9km_mean_1_allyear = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        hdf_file_smap_9km = path_smap + '/9km' + '/smap_sm_9km_' + str(iyr) + monthname[month_plt[imo]-1] + '.hdf5'
        f_read_smap_9km = h5py.File(hdf_file_smap_9km, "r")
        varname_list_smap_9km = list(f_read_smap_9km.keys())

        # smap_9km_load = smap_9km_init
        smap_9km_load_1 = f_read_smap_9km[varname_list_smap_9km[0]][
                                           row_dan_9km_ind[0]:row_dan_9km_ind[-1] + 1,
                                           col_dan_9km_ind[0]:col_dan_9km_ind[-1] + 1, days_begin-1:days_end]  # AM
        smap_9km_load_2 = f_read_smap_9km[varname_list_smap_9km[1]][
                                           row_dan_9km_ind[0]:row_dan_9km_ind[-1] + 1,
                                           col_dan_9km_ind[0]:col_dan_9km_ind[-1] + 1, days_begin-1:days_end]  # PM
        f_read_smap_9km.close()

        smap_9km_mean_1 = np.nanmean(np.stack((smap_9km_load_1, smap_9km_load_2), axis=3), axis=3)
        smap_9km_mean_1_allyear.append(smap_9km_mean_1)

        del(smap_9km_load_1, smap_9km_mean_1)
        print(monthname[month_plt[imo]-1])

smap_9km_data_stack_dan = np.squeeze(np.array(smap_9km_mean_1_allyear))
del(smap_9km_mean_1_allyear)
smap_9km_data_stack_dan = np.transpose(smap_9km_data_stack_dan, (2, 0, 1))


#Load and subset the region of Danube RB (SMAP 1 km)
[lat_1km_dan, row_dan_1km_ind, lon_1km_dan, col_dan_1km_ind] = \
    coordtable_subset(lat_world_ease_1km, lon_world_ease_1km,
                      shp_dan_extent[3], shp_dan_extent[2], shp_dan_extent[1], shp_dan_extent[0])

smap_1km_mean_1_allyear = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        smap_1km_load_1_stack = []
        for idt in range(days_begin, days_end+1): # 30 days
            str_date = str(iyr) + '-' + monthname[month_plt[imo]-1] + '-' + str(idt).zfill(2)
            str_doy = str(datetime.datetime.strptime(str_date, '%Y-%m-%d').date().timetuple().tm_yday)
            tif_file_smap_1km = path_smap + '/1km/gldas/' + str(iyr) + '/smap_sm_1km_ds_' + str(iyr) + str_doy.zfill(3) + '.tif'
            src_tf = gdal.Open(tif_file_smap_1km)
            src_tf_arr = src_tf.ReadAsArray()[:, row_dan_1km_ind[0]:row_dan_1km_ind[-1]+1,
                         col_dan_1km_ind[0]:col_dan_1km_ind[-1]+1]
            smap_1km_load_1 = np.nanmean(src_tf_arr, axis=0)
            smap_1km_load_1_stack.append(smap_1km_load_1)

            print(str_date)
            del(src_tf_arr, smap_1km_load_1)

        smap_1km_load_1_stack = np.stack(smap_1km_load_1_stack)
        smap_1km_mean_1_allyear.append(smap_1km_load_1_stack)
        del (smap_1km_load_1_stack)

smap_1km_data_stack_dan = np.squeeze(np.array(smap_1km_mean_1_allyear))


smap_1km_mean_1_allyear_ori = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        smap_1km_load_1_stack = []
        for idt in range(days_begin, days_end+1): # 30 days
            str_date = str(iyr) + '-' + monthname[month_plt[imo]-1] + '-' + str(idt).zfill(2)
            str_doy = str(datetime.datetime.strptime(str_date, '%Y-%m-%d').date().timetuple().tm_yday)
            tif_file_smap_1km = path_smap + '/1km/gldas_old_data/' + str(iyr) + '/smap_sm_1km_ds_' + str(iyr) + \
                                str_doy.zfill(3) + '.tif'
            src_tf = gdal.Open(tif_file_smap_1km)
            src_tf_arr = src_tf.ReadAsArray()[:, row_dan_1km_ind[0]:row_dan_1km_ind[-1]+1,
                         col_dan_1km_ind[0]:col_dan_1km_ind[-1]+1]
            smap_1km_load_1 = np.nanmean(src_tf_arr, axis=0)
            smap_1km_load_1_stack.append(smap_1km_load_1)

            print(str_date)
            del(src_tf_arr, smap_1km_load_1)

        smap_1km_load_1_stack = np.stack(smap_1km_load_1_stack)
        smap_1km_mean_1_allyear_ori.append(smap_1km_load_1_stack)
        del (smap_1km_load_1_stack)

smap_1km_data_stack_dan_ori = np.squeeze(np.array(smap_1km_mean_1_allyear_ori))


with h5py.File(path_model_evaluation + '/smap_dan_sm.hdf5', 'w') as f:
    f.create_dataset('smap_9km_data_stack_dan', data=smap_9km_data_stack_dan)
    f.create_dataset('smap_1km_data_stack_dan', data=smap_1km_data_stack_dan)
f.close()

# Read the map data
f_read = h5py.File(path_model_evaluation + "/smap_dan_sm.hdf5", "r")
varname_read_list = ['smap_9km_data_stack_dan', 'smap_1km_data_stack_dan']
for x in range(len(varname_read_list)):
    var_obj = f_read[varname_read_list[x]][()]
    exec(varname_read_list[x] + '= var_obj')
    del(var_obj)
f_read.close()


# Subplot maps
# Load in watershed shapefile boundaries
shapefile_dan = fiona.open(path_shp_dan + '/' + shp_dan_file, 'r')
crop_shape_dan = [feature["geometry"] for feature in shapefile_dan]
# shp_dan_extent = list(shapefile_dan.bounds)

output_crs = 'EPSG:4326'

# Subset and reproject the SMAP SM data at watershed
# 1 km
masked_ds_dan_1km_all = []
for n in range(smap_1km_data_stack_dan.shape[0]):
    sub_window_dan_1km = Window(col_dan_1km_ind[0], row_dan_1km_ind[0], len(col_dan_1km_ind), len(row_dan_1km_ind))
    kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                      'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
    smap_sm_dan_1km_output = sub_n_reproj(smap_1km_data_stack_dan[n, :, :], kwargs_1km_sub, sub_window_dan_1km, output_crs)

    masked_ds_dan_1km, mask_transform_ds_dan_1km = mask(dataset=smap_sm_dan_1km_output, shapes=crop_shape_dan, crop=True)
    masked_ds_dan_1km[np.where(masked_ds_dan_1km == 0)] = np.nan
    masked_ds_dan_1km = masked_ds_dan_1km.squeeze()

    masked_ds_dan_1km_all.append(masked_ds_dan_1km)

masked_ds_dan_1km_all = np.asarray(masked_ds_dan_1km_all)

# 1 km (original)
masked_ds_dan_1km_all_ori = []
for n in range(smap_1km_data_stack_dan_ori.shape[0]):
    sub_window_dan_1km = Window(col_dan_1km_ind[0], row_dan_1km_ind[0], len(col_dan_1km_ind), len(row_dan_1km_ind))
    kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                      'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
    smap_sm_dan_1km_output = sub_n_reproj(smap_1km_data_stack_dan_ori[n, :, :], kwargs_1km_sub, sub_window_dan_1km, output_crs)

    masked_ds_dan_1km, mask_transform_ds_dan_1km = mask(dataset=smap_sm_dan_1km_output, shapes=crop_shape_dan, crop=True)
    masked_ds_dan_1km[np.where(masked_ds_dan_1km == 0)] = np.nan
    masked_ds_dan_1km = masked_ds_dan_1km.squeeze()

    masked_ds_dan_1km_all_ori.append(masked_ds_dan_1km)

masked_ds_dan_1km_all_ori = np.asarray(masked_ds_dan_1km_all_ori)


# 9 km
masked_ds_dan_9km_all = []
for n in range(smap_9km_data_stack_dan.shape[0]):
    sub_window_dan_9km = Window(col_dan_9km_ind[0], row_dan_9km_ind[0], len(col_dan_9km_ind), len(row_dan_9km_ind))
    kwargs_9km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_9km),
                      'height': len(lat_world_ease_9km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(9008.05, 0.0, -17367530.44516138, 0.0, -9008.05, 7314540.79258289)}
    smap_sm_dan_9km_output = sub_n_reproj(smap_9km_data_stack_dan[n, :, :], kwargs_9km_sub, sub_window_dan_9km, output_crs)

    masked_ds_dan_9km, mask_transform_ds_dan_9km = mask(dataset=smap_sm_dan_9km_output, shapes=crop_shape_dan, crop=True)
    masked_ds_dan_9km[np.where(masked_ds_dan_9km == 0)] = np.nan
    masked_ds_dan_9km = masked_ds_dan_9km.squeeze()

    masked_ds_dan_9km_all.append(masked_ds_dan_9km)

masked_ds_dan_9km_all = np.asarray(masked_ds_dan_9km_all)
# masked_ds_dan_9km_all[masked_ds_dan_9km_all >= 0.5] = np.nan


# Make the subplot maps
title_content = ['1 km(GF)\n(Sept 1 2020)', '1 km\n(Sept 1 2020)', '1 km(GF)\n(Sept 2 2020)', '1 km\n(Sept 2 2020)',
                 '1 km(GF)\n(Sept 3 2020)', '1 km\n(Sept 3 2020)', '1 km(GF)\n(Sept 4 2020)', '1 km\n(Sept 4 2020)',
                 '1 km(GF)\n(Sept 5 2020)', '1 km\n(Sept 5 2020)', '1 km(GF)\n(Sept 6 2020)', '1 km\n(Sept 6 2020)']
feature_shp_dan = ShapelyFeature(Reader(path_shp_dan + '/' + shp_dan_file).geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
extent_dan = np.array(smap_sm_dan_1km_output.bounds)
extent_dan = extent_dan[[0, 2, 1, 3]]

fig = plt.figure(figsize=(7, 10), facecolor='w', edgecolor='k', dpi=200)
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95, hspace=0.25, wspace=0.2)
for ipt in range(6):
    # 1 km
    ax = fig.add_subplot(6, 2, ipt*2+1, projection=ccrs.PlateCarree())
    ax.add_feature(feature_shp_dan)
    # ax.set_title(title_content[ipt*2], pad=20, fontsize=13, fontweight='bold')
    img = ax.imshow(masked_ds_dan_1km_all[ipt, :, :], origin='upper', vmin=0, vmax=0.5, cmap='Spectral',
               extent=extent_dan)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=5)
    gl.ylocator = mticker.MultipleLocator(base=4)
    gl.xlabel_style = {'size': 6}
    gl.ylabel_style = {'size': 6}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.text(8.5, 44, title_content[ipt*2], fontsize=7, horizontalalignment='left',
            verticalalignment='top', weight='bold')
    # 9 km
    ax = fig.add_subplot(6, 2, ipt*2+2, projection=ccrs.PlateCarree())
    ax.add_feature(feature_shp_dan)
    # ax.set_title(title_content[ipt*2+1], pad=20, fontsize=13, fontweight='bold')
    img = ax.imshow(masked_ds_dan_1km_all_ori[ipt, :, :], origin='upper', vmin=0, vmax=0.5, cmap='Spectral',
               extent=extent_dan)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=5)
    gl.ylocator = mticker.MultipleLocator(base=4)
    gl.xlabel_style = {'size': 6}
    gl.ylabel_style = {'size': 6}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.text(8.5, 44, title_content[ipt*2+1], fontsize=7, horizontalalignment='left',
            verticalalignment='top', weight='bold')

cbar_ax = fig.add_axes([0.2, 0.04, 0.5, 0.015])
cbar = plt.colorbar(img, cax=cbar_ax, extend='both', orientation='horizontal', pad=0.1)
cbar.ax.tick_params(labelsize=7)
cbar_ax.locator_params(nbins=5)
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=7, x=1.08, y=0.05, labelpad=-15)
plt.savefig(path_results + '/sm_comp_dan_1_ori.png')
plt.close()


########################################################################################################################
# 4. Scatter plots
# 4.1 Select geograhical locations by using index tables, and plot delta T - SM relationship lines through each NDVI class

# Lat/lon of the locations in the world:

lat_slc = [32.34, -35.42, 40.33, 34.94, 40.88948, 55.8776, 50.5149, 51.38164, 30.31]
lon_slc = [87.03, 146.2, -5.04, -97.65, 25.8522, 9.2683, 6.37559, -106.41583, -98.78]
name_slc = ['CTP', 'OZNET', 'REMEDHUS', 'SOILSCAPE', 'GROW(47emqp81)',
            'HOBE(3.08)', 'TERENO(Schoeneseiffen)', 'RISMA(SK4)', 'TxSON']
ndvi_class = np.linspace(0, 1, 11)
viridis_r = plt.cm.get_cmap('viridis_r', 10)

import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
base_cmap = cm.get_cmap('Spectral')  # This is a LinearSegmentedColormap
colors = base_cmap(np.linspace(0, 1, 7))  # Sample 10 evenly spaced colors
spectral_listed = ListedColormap(colors, name='SpectralListed')


row_25km_ind_sub = []
col_25km_ind_sub = []
for ico in range(len(lat_slc)):
    row_dist = np.absolute(lat_slc[ico] - lat_world_ease_25km)
    row_match = np.argmin(row_dist)
    col_dist = np.absolute(lon_slc[ico] - lon_world_ease_25km)
    col_match = np.argmin(col_dist)
    # ind = np.intersect1d(row_match, col_match)[0]
    row_25km_ind_sub.append(row_match)
    col_25km_ind_sub.append(col_match)
    del(row_dist, row_match, col_dist, col_match)

row_25km_ind_sub = np.asarray(row_25km_ind_sub)
col_25km_ind_sub = np.asarray(col_25km_ind_sub)


hdf_file = path_model + '/gldas/ds_model_coef_nofill.hdf5'
f_read = h5py.File(hdf_file, "r")
varname_list = list(f_read.keys())

r_sq_all = []
rmse_all = []
for x in range(len(row_25km_ind_sub)):
    r_sq = f_read[varname_list[30]][row_25km_ind_sub[x], col_25km_ind_sub[x], ::2]
    rmse = f_read[varname_list[30]][row_25km_ind_sub[x], col_25km_ind_sub[x], 1::2]
    r_sq_all.append(r_sq)
    rmse_all.append(rmse)
r_sq_all = np.asarray(r_sq_all)
rmse_all = np.asarray(rmse_all)
metric_all = [r_sq_all, rmse_all]
metric_all = np.asarray(metric_all)

np.savetxt(path_model_evaluation + '/regression_metric.csv', r_sq_all, delimiter=",", fmt='%f')
np.savetxt(path_model_evaluation + '/regression_metric_rmse.csv', rmse_all, delimiter=",", fmt='%f')

# Extract the indexes for the arrays to make scatter plots
coord_25km_ind = [np.intersect1d(np.where(col_lmask_ease_25km_ind == col_25km_ind_sub[x]),
                                np.where(row_lmask_ease_25km_ind == row_25km_ind_sub[x]))[0]
                  for x in np.arange(len(row_25km_ind_sub))]

lmask_ease_25km_1d = lmask_ease_25km.reshape(1, -1).ravel()
coord_25km_land_ind = np.where(lmask_ease_25km_1d == 1)[0]


# Load in data
os.chdir(path_model)
hdf_file = path_model + '/gldas/ds_model_01.hdf5'
f_read = h5py.File(hdf_file, "r")
varname_list_01 = list(f_read.keys())
hdf_file_2 = path_model + '/gldas/ds_model_07.hdf5'
f_read_2 = h5py.File(hdf_file_2, "r")
varname_list_07 = list(f_read_2.keys())

lst_am_delta = np.array([f_read[varname_list_01[0]][x, :] for x in coord_25km_ind])
lst_pm_delta = np.array([f_read[varname_list_01[1]][x, :] for x in coord_25km_ind])
ndvi = np.array([f_read[varname_list_01[2]][x, :] for x in coord_25km_ind])
sm_am = np.array([f_read[varname_list_01[3]][x, :] for x in coord_25km_ind])
sm_pm = np.array([f_read[varname_list_01[4]][x, :] for x in coord_25km_ind])

lst_am_delta_2 = np.array([f_read_2[varname_list_07[0]][x, :] for x in coord_25km_ind])
lst_pm_delta_2 = np.array([f_read_2[varname_list_07[1]][x, :] for x in coord_25km_ind])
ndvi_2 = np.array([f_read_2[varname_list_07[2]][x, :] for x in coord_25km_ind])
sm_am_2 = np.array([f_read_2[varname_list_07[3]][x, :] for x in coord_25km_ind])
sm_pm_2 = np.array([f_read_2[varname_list_07[4]][x, :] for x in coord_25km_ind])
#
# # Replace the OZNET data from July by January
# lst_am_delta[1, :] = lst_am_delta_2[1, :]
# lst_pm_delta[1, :] = lst_pm_delta_2[1, :]
# ndvi[1, :] = ndvi_2[1, :]
# sm_am[1, :] = sm_am_2[1, :]
# sm_pm[1, :] = sm_pm_2[1, :]

# Subplots of GLDAS SM vs. LST difference
# 4.2
fig = plt.figure(figsize=(11, 8), dpi=150)
plt.subplots_adjust(left=0.1, right=0.85, bottom=0.13, top=0.93, hspace=0.25, wspace=0.25)
for i in range(4):
    x = sm_am_2[i, :]
    y = lst_am_delta_2[i, :]
    c = ndvi_2[i, :]

    ax = fig.add_subplot(2, 2, i+1)
    sc = ax.scatter(x, y, c=c, marker='o', s=3, cmap='viridis_r')
    sc.set_clim(vmin=0, vmax=0.7)
    sc.set_label('NDVI')

    # Plot by NDVI classes
    for n in range(len(ndvi_class)-1):
        ndvi_ind = np.array([np.where((c>=ndvi_class[n]) & (c<ndvi_class[n+1]))[0]])
        if ndvi_ind.shape[1] > 10:
            coef, intr = np.polyfit(x[ndvi_ind].squeeze(), y[ndvi_ind].squeeze(), 1)
            ax.plot(x[ndvi_ind].squeeze(), intr + coef * x[ndvi_ind].squeeze(), '-', color=viridis_r.colors[n])
        else:
            pass

    plt.xlim(0, 0.4)
    ax.set_xticks(np.arange(0, 0.5, 0.1))
    plt.ylim(0, 40)
    ax.set_yticks(np.arange(0, 50, 10))
    ax.text(0.02, 35, name_slc[i],fontsize=12)
    plt.grid(linestyle='--')
    # cbar = plt.colorbar(sc, extend='both')
    # cbar.set_label('NDVI')
cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
cbar = fig.colorbar(sc, cax=cbar_ax, extend='both', pad=0.1, orientation='vertical')
cbar_ax.tick_params(labelsize=10)
cbar_ax.locator_params(nbins=8)
cbar.set_label('NDVI', fontsize=10, labelpad=10)
fig.text(0.5, 0.03, 'Soil Moisture ($\mathregular{m^3/m^3)}$', ha='center', fontsize=14, fontweight='bold')
fig.text(0.02, 0.5, 'Temperature Difference (K)', va='center', rotation='vertical', fontsize=14, fontweight='bold')
plt.savefig(path_model_evaluation + '/gldas_comp_1.png')
plt.close()



# Subplots of GLDAS SM vs. LST difference (short plot)
# name_slc = ['REMEDHUS', 'SOILSCAPE']
site_ind = [2, 3, 8]
# 4.2
fig = plt.figure(figsize=(8, 8), dpi=150)
plt.subplots_adjust(left=0.1, right=0.85, bottom=0.13, top=0.93, hspace=0.25, wspace=0.25)
for i in range(3):
    x = sm_am_2[site_ind[i], :]
    y = lst_am_delta_2[site_ind[i], :]
    c = ndvi_2[site_ind[i], :]

    ax = fig.add_subplot(3, 1, i+1)
    sc = ax.scatter(x, y, c=c, marker='o', s=3, cmap=spectral_listed)
    sc.set_clim(vmin=0, vmax=0.7)
    sc.set_label('NDVI')

    # Plot by NDVI classes
    for n in range(len(ndvi_class)-1):
        ndvi_ind = np.array([np.where((c>=ndvi_class[n]) & (c<ndvi_class[n+1]))[0]])
        if ndvi_ind.shape[1] > 10:
            coef, intr = np.polyfit(x[ndvi_ind].squeeze(), y[ndvi_ind].squeeze(), 1)
            ax.plot(x[ndvi_ind].squeeze(), intr + coef * x[ndvi_ind].squeeze(), '-', color=spectral_listed.colors[n])
        else:
            pass

    plt.xlim(0, 0.4)
    ax.set_xticks(np.arange(0, 0.5, 0.1))
    ax.tick_params(axis='x', labelsize=12)
    plt.ylim(0, 40)
    ax.set_yticks(np.arange(0, 50, 10))
    ax.tick_params(axis='y', labelsize=12)
    ax.text(0.02, 35, name_slc[site_ind[i]],fontsize=14)
    plt.grid(linestyle='--')
    # cbar = plt.colorbar(sc, extend='both')
    # cbar.set_label('NDVI')
cbar_ax = fig.add_axes([0.9, 0.2, 0.015, 0.6])
cbar = fig.colorbar(sc, cax=cbar_ax, extend='both', pad=0.1, orientation='vertical')
cbar_ax.tick_params(labelsize=12)
cbar_ax.locator_params(nbins=8)
cbar.set_label('NDVI', fontsize=12, labelpad=10)
fig.text(0.5, 0.03, 'Soil Moisture ($\mathregular{m^3/m^3)}$', ha='center', fontsize=14, fontweight='bold')
fig.text(0.02, 0.5, 'Temperature Difference (K)', va='center', rotation='vertical', fontsize=14, fontweight='bold')
plt.savefig(path_model_evaluation + '/gldas_comp_2.png')
plt.close()


# 4.3 Load in GLDAS and MODIS LST (2018)
lmask_ease_25km_1d = lmask_ease_25km.reshape(1, -1).ravel()
coord_25km_land_ind = np.where(lmask_ease_25km_1d == 1)[0]

# GLDAS
os.chdir(path_model)
hdf_file = path_model + '/gldas/ds_model_01.hdf5'
f_read = h5py.File(hdf_file, "r")
varname_list_01 = list(f_read.keys())
hdf_file_2 = path_model + '/gldas/ds_model_07.hdf5'
f_read_2 = h5py.File(hdf_file_2, "r")
varname_list_07 = list(f_read_2.keys())

lmask_init = np.empty((len(lat_world_ease_25km), len(lon_world_ease_25km)), dtype='float32')
lmask_init = lmask_init.reshape(1, -1)
lmask_init[:] = 0

lst_am_delta_01 = f_read[varname_list_01[0]][:, -31:]
lst_pm_delta_01 = f_read[varname_list_01[1]][:, -31:]
lst_delta_01 = (np.nanmean(lst_am_delta_01, axis=1)+np.nanmean(lst_pm_delta_01, axis=1))/2
lst_gldas_delta_01 = np.copy(lmask_init)
lst_gldas_delta_01[0, coord_25km_land_ind] = lst_delta_01
lst_gldas_delta_01 = lst_gldas_delta_01.reshape((len(lat_world_ease_25km), len(lon_world_ease_25km)))
lst_gldas_delta_01[lst_gldas_delta_01 == 0] = np.nan

lst_am_delta_07 = f_read_2[varname_list_07[0]][:, -31:]
lst_pm_delta_07 = f_read_2[varname_list_07[1]][:, -31:]
lst_delta_07 = (np.nanmean(lst_am_delta_07, axis=1)+np.nanmean(lst_pm_delta_07, axis=1))/2
lst_gldas_delta_07 = np.copy(lmask_init)
lst_gldas_delta_07[0, coord_25km_land_ind] = lst_delta_07
lst_gldas_delta_07 = lst_gldas_delta_07.reshape((len(lat_world_ease_25km), len(lon_world_ease_25km)))
lst_gldas_delta_07[lst_gldas_delta_07 == 0] = np.nan

# MODIS
[row_world_ease_25km_from_1km_ind, col_world_ease_25km_from_1km_ind] = find_easeind_lofrhi\
    (lat_world_ease_1km, lon_world_ease_1km, interdist_ease_25km, size_world_ease_25km[0], size_world_ease_25km[1],
     row_world_ease_25km_ind, col_world_ease_25km_ind)

# Load in variables
f = h5py.File(path_model + '/gap_filling/coord_world_1km_ind.hdf5', "r")
varname_list_0 = list(f.keys())
varname_list_0 = [varname_list_0[0]]
# varname_list_0 = [varname_list_0[0], varname_list_0[2]]
for x in range(len(varname_list_0)):
    var_obj = f[varname_list_0[x]][()]
    exec(varname_list_0[x] + '= var_obj')
    del(var_obj)
f.close()
del(f, varname_list_0)

# os.chdir(path_model)
hdf_file = path_datasets + '/MODIS/HDF_Data/modis_lst_201801.hdf5'
f_read = h5py.File(hdf_file, "r")
varname_list_01 = list(f_read.keys())
hdf_file_2 = path_datasets + '/MODIS/HDF_Data/modis_lst_201807.hdf5'
f_read_2 = h5py.File(hdf_file_2, "r")
varname_list_07 = list(f_read_2.keys())

lst_am_modis_01 = f_read[varname_list_01[0]][()]
lst_am_modis_01 = np.nanmean(lst_am_modis_01, axis=0)
lst_pm_modis_01 = f_read[varname_list_01[1]][()]
lst_pm_modis_01 = np.nanmean(lst_pm_modis_01, axis=0)
lst_delta_01 = lst_am_modis_01-lst_pm_modis_01

lst_am_modis_07 = f_read_2[varname_list_07[0]][()]
lst_am_modis_07 = np.nanmean(lst_am_modis_07, axis=0)
lst_pm_modis_07 = f_read_2[varname_list_07[1]][()]
lst_pm_modis_07 = np.nanmean(lst_pm_modis_07, axis=0)
lst_delta_07 = lst_am_modis_07-lst_pm_modis_07

# Save monthly MODIS LST land pixels to hdf file
data_name = ['lst_delta_01', 'lst_delta_07']
var_name = ['lst_delta_01', 'lst_delta_07']
with h5py.File(path_model + '/lst_delta.hdf5', 'w') as f:
    for idv in range(len(var_name)):
        f.create_dataset(var_name[idv], data=eval(data_name[idv]), compression='lzf')
f.close()

# Load in geo-location parameters
os.chdir(path_workspace)
f = h5py.File(path_model + '/lst_delta.hdf5', "r")
varname_list = ['lst_delta_01', 'lst_delta_07']
for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f.close()

lmask_init_1km = np.empty((len(lat_world_ease_1km), len(lon_world_ease_1km)), dtype='float32')
lmask_init_1km = lmask_init_1km.reshape(1, -1)
lmask_init_1km[:] = 0

lst_delta_01_2d = np.copy(lmask_init_1km)
lst_delta_01_2d[0, coord_world_1km_ind] = lst_delta_01
lst_delta_01_2d = lst_delta_01_2d.reshape((len(lat_world_ease_1km), len(lon_world_ease_1km)))
lst_delta_01_2d[lst_delta_01_2d == 0] = np.nan

lst_delta_07_2d = np.copy(lmask_init_1km)
lst_delta_07_2d[0, coord_world_1km_ind] = lst_delta_07
lst_delta_07_2d = lst_delta_07_2d.reshape((len(lat_world_ease_1km), len(lon_world_ease_1km)))
lst_delta_07_2d[lst_delta_07_2d == 0] = np.nan

lst_modis_delta_01 = np.array \
    ([np.nanmean(lst_delta_01_2d[row_world_ease_25km_from_1km_ind[x], :], axis=0)
      for x in range(len(lat_world_ease_25km))])
lst_modis_delta_01 = np.array \
    ([np.nanmean(lst_modis_delta_01[:, col_world_ease_25km_from_1km_ind[y]], axis=1)
      for y in range(len(lon_world_ease_25km))])
lst_modis_delta_01 = np.fliplr(np.rot90(lst_modis_delta_01, 3))

lst_modis_delta_07 = np.array \
    ([np.nanmean(lst_delta_07_2d[row_world_ease_25km_from_1km_ind[x], :], axis=0)
      for x in range(len(lat_world_ease_25km))])
lst_modis_delta_07 = np.array \
    ([np.nanmean(lst_modis_delta_07[:, col_world_ease_25km_from_1km_ind[y]], axis=1)
      for y in range(len(lon_world_ease_25km))])
lst_modis_delta_07 = np.fliplr(np.rot90(lst_modis_delta_07, 3))

lst_01 = lst_gldas_delta_01-lst_modis_delta_01
lst_07 = lst_gldas_delta_07-lst_modis_delta_07

# 1.3.4 Difference map
# title_content = ['$\Delta \mathregular{R^2}$ (a.m.)', '$\Delta \mathregular{R^2}$ (p.m.)',
#                  '$\Delta$RMSE (a.m.)', '$\Delta$RMSE (p.m.)']
xx_wrd, yy_wrd = np.meshgrid(lon_world_ease_25km, lat_world_ease_25km) # Create the map matrix
shape_world = ShapelyFeature(Reader(path_gis_data + '/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')

title_content = ['January', 'July']
fig = plt.figure(figsize=(8, 10), facecolor='w', edgecolor='k', dpi=150)
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.85, hspace=0.25, wspace=0.2)
for ipt in range(1):
    # Delta R^2
    ax1 = fig.add_subplot(2, 1, ipt+1, projection=ccrs.PlateCarree())
    ax1.add_feature(shape_world, linewidth=0.5)
    ax1.set_extent([-180, 180, -70, 90], ccrs.PlateCarree())
    img1 = ax1.pcolormesh(xx_wrd, yy_wrd, lst_01*0.4, transform=ccrs.PlateCarree(),
                        vmin=-10, vmax=10, cmap='bwr')
    gl = ax1.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = mticker.FixedLocator([90, 45, 0, -45, -90])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}
    cbar1 = plt.colorbar(img1, extend='both', orientation='horizontal', aspect=50, pad=0.1, shrink=0.75)
    cbar1.set_label('(k)', fontsize=12, x=0.95)
    cbar1.ax.tick_params(labelsize=12)
    cbar1.ax.locator_params(nbins=4)
    ax1.set_title(title_content[ipt], pad=20, fontsize=16, weight='bold')

    # Delta RMSE
    ax2 = fig.add_subplot(2, 1, ipt+2, projection=ccrs.PlateCarree())
    ax2.add_feature(shape_world, linewidth=0.5)
    ax2.set_extent([-180, 180, -70, 90], ccrs.PlateCarree())
    img2 = ax2.pcolormesh(xx_wrd, yy_wrd, lst_07*0.4, transform=ccrs.PlateCarree(),
                        vmin=-10, vmax=10, cmap='bwr')
    gl = ax2.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = mticker.FixedLocator([90, 45, 0, -45, -90])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}
    cbar2 = plt.colorbar(img2, extend='both', orientation='horizontal', aspect=50, pad=0.1, shrink=0.75)
    cbar2.set_label('(k)', fontsize=12, x=0.95)
    cbar2.ax.tick_params(labelsize=12)
    cbar2.ax.locator_params(nbins=4)
    ax2.set_title(title_content[ipt+1], pad=20, fontsize=16, weight='bold')

plt.suptitle('LST Difference (GLDAS - MODIS)', fontsize=20, weight='bold')
plt.show()
plt.savefig(path_model_evaluation + '/lst_delta.png')
plt.close()



########################################################################################################################
# 5 SMOS SM maps (Worldwide)

# 5.1 Composite the data of the first 16 days of one specific month
# Load in smos data
year_plt = [2021]
month_plt = list([8])
days_begin = 1
days_end = 9
days_n = days_end - days_begin + 1

matsize_25km = [len(month_plt), len(lat_world_ease_25km), len(lon_world_ease_25km)]
smos_25km_mean_1_all = np.empty(matsize_25km, dtype='float32')
smos_25km_mean_1_all[:] = np.nan
smos_25km_mean_2_all = np.copy(smos_25km_mean_1_all)
smos_25km_mean_3_all = np.copy(smos_25km_mean_1_all)

# 9 km smos SM
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        hdf_file_smos_25km = path_smos + '/25km' + '/smos_sm_25km_' + str(iyr) + monthname[month_plt[imo]-1] + '.hdf5'
        f_read_smos_25km = h5py.File(hdf_file_smos_25km, "r")
        varname_list_smos_25km = list(f_read_smos_25km.keys())

        smos_25km_load = np.empty((len(lat_world_ease_25km), len(lon_world_ease_25km), days_n*2))
        smos_25km_load[:] = np.nan
        for idt in range(days_begin-1, days_end):
            smos_25km_load[:, :, 2*idt+0] = f_read_smos_25km[varname_list_smos_25km[0]][:, :, idt] # AM
            smos_25km_load[:, :, 2*idt+1] = f_read_smos_25km[varname_list_smos_25km[1]][:, :, idt] # PM
        f_read_smos_25km.close()

        smos_25km_mean_1 = np.nanmean(smos_25km_load[:, :, :days_n//3], axis=2)
        smos_25km_mean_2 = np.nanmean(smos_25km_load[:, :, days_n//3:days_n//3*2], axis=2)
        smos_25km_mean_3 = np.nanmean(smos_25km_load[:, :, days_n //3*2:], axis=2)
        del(smos_25km_load)

        smos_25km_mean_1_all[imo, :, :] = smos_25km_mean_1
        smos_25km_mean_2_all[imo, :, :] = smos_25km_mean_2
        smos_25km_mean_3_all[imo, :, :] = smos_25km_mean_3
        del(smos_25km_mean_1, smos_25km_mean_2, smos_25km_mean_3)
        print(imo)


# Load in smos 1 km SM (Gap-filled)
smos_1km_agg_stack = np.empty((len(lat_world_ease_25km), len(lon_world_ease_25km), days_n*2))
smos_1km_agg_stack[:] = np.nan
smos_1km_mean_1_all = np.empty(matsize_25km, dtype='float32')
smos_1km_mean_1_all[:] = np.nan
smos_1km_mean_2_all = np.copy(smos_1km_mean_1_all)
smos_1km_mean_3_all = np.copy(smos_1km_mean_1_all)

for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        for idt in range(days_n):
            str_date = str(iyr) + '-' + monthname[month_plt[imo]-1] + '-' + str(idt+1).zfill(2)
            str_doy = str(datetime.datetime.strptime(str_date, '%Y-%m-%d').date().timetuple().tm_yday)
            tif_file_smos_1km = path_smos + '/1km/' + str(iyr) + '/smos_sm_1km_ds_' + str(iyr) + str_doy.zfill(3) + '.tif'
            src_tf = gdal.Open(tif_file_smos_1km)
            src_tf_arr = src_tf.ReadAsArray().astype(np.float32)

            # Aggregate to 9 km
            for ilr in range(2):
                src_tf_arr_1layer = src_tf_arr[ilr, :, :]
                smos_sm_1km_agg = np.array\
                    ([np.nanmean(src_tf_arr_1layer[row_world_ease_25km_from_1km_ind[x], :], axis=0)
                        for x in range(len(lat_world_ease_25km))])
                smos_sm_1km_agg = np.array\
                    ([np.nanmean(smos_sm_1km_agg[:, col_world_ease_25km_from_1km_ind[y]], axis=1)
                        for y in range(len(lon_world_ease_25km))])
                smos_sm_1km_agg = np.fliplr(np.rot90(smos_sm_1km_agg, 3))
                smos_1km_agg_stack[:, :, 2*idt+ilr] = smos_sm_1km_agg
                del(smos_sm_1km_agg, src_tf_arr_1layer)

            print(str_date)
            del(src_tf_arr)

        smos_1km_mean_1 = np.nanmean(smos_1km_agg_stack[:, :, :days_n//3], axis=2)
        smos_1km_mean_2 = np.nanmean(smos_1km_agg_stack[:, :, days_n//3:days_n//3*2], axis=2)
        smos_1km_mean_3 = np.nanmean(smos_1km_agg_stack[:, :, days_n //3*2:], axis=2)

        smos_1km_mean_1_all[imo, :, :] = smos_1km_mean_1
        smos_1km_mean_2_all[imo, :, :] = smos_1km_mean_2
        smos_1km_mean_3_all[imo, :, :] = smos_1km_mean_3
        del(smos_1km_mean_1, smos_1km_mean_2, smos_1km_mean_3)



smos_data_stack = np.stack((smos_1km_mean_1_all, smos_1km_mean_1_all_ori, smos_1km_mean_2_all, smos_1km_mean_2_all_ori,
                            smos_1km_mean_3_all, smos_1km_mean_3_all_ori))

# # Save and load the data
# with h5py.File(path_model_evaluation + '/smos_data_stack.hdf5', 'w') as f:
#     f.create_dataset('smos_data_stack', data=smos_data_stack)
# f.close()



# 5.2 Maps of the world

smos_data_stack = np.stack((smos_1km_mean_1_all, smos_25km_mean_1_all, smos_1km_mean_2_all, smos_25km_mean_2_all,
                            smos_1km_mean_3_all, smos_25km_mean_3_all))

# f_read = h5py.File(path_model_evaluation + '/smos_data_stack.hdf5', "r")
# smos_data_stack = f_read['smos_data_stack'][()]
# f_read.close()

xx_wrd, yy_wrd = np.meshgrid(lon_world_ease_25km, lat_world_ease_25km) # Create the map matrix
shape_world = ShapelyFeature(Reader(path_gis_data + '/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
shp_world_extent = [-180.0, 180.0, -70, 90]


# July
title_content = ['1 km\n(July 1-3 2020)', '25 km\n(July 1-3 2020)', '1 km\n(July 4-6 2020)', '25 km\n(July 4-6 2020)',
                 '1 km\n(July 7-9 2020)', '25 km\n(July 7-9 2020)']

fig = plt.figure(figsize=(12, 9), facecolor='w', edgecolor='k', dpi=150)
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95, hspace=0.2, wspace=0.2)
for ipt in range(3):
    # 1 km
    ax = fig.add_subplot(3, 2, ipt*2+1, projection=ccrs.PlateCarree())
    ax.add_feature(shape_world, linewidth=0.5)
    img = ax.pcolormesh(xx_wrd, yy_wrd, smos_data_stack[ipt * 2, 0, :, :], vmin=0, vmax=0.5, cmap='Spectral')
    ax.set_extent([180, -180, -70, 90], ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = mticker.FixedLocator([90, 45, 0, -45, -90])
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # cbar = plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
    # cbar.ax.tick_params(labelsize=15)
    # cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=15, x=0.95)
    # ax.set_title(title_content[ipt], pad=12, fontsize=17, weight='bold')
    ax.text(-175, -40, title_content[ipt*2], fontsize=11, horizontalalignment='left',
            verticalalignment='top', weight='bold')
    # 25 km
    ax = fig.add_subplot(3, 2, ipt*2+2, projection=ccrs.PlateCarree())
    ax.add_feature(shape_world, linewidth=0.5)
    img = ax.pcolormesh(xx_wrd, yy_wrd, smos_data_stack[ipt * 2 + 1, 0, :, :], vmin=0, vmax=0.5, cmap='Spectral')
    ax.set_extent([180, -180, -70, 90], ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = mticker.FixedLocator([90, 45, 0, -45, -90])
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # cbar = plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
    # cbar.ax.tick_params(labelsize=15)
    # cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=15, x=0.95)
    # ax.set_title(title_content[ipt], pad=12, fontsize=17, weight='bold')
    ax.text(-175, -40, title_content[ipt*2+1], fontsize=11, horizontalalignment='left',
            verticalalignment='top', weight='bold')
cbar_ax = fig.add_axes([0.2, 0.04, 0.5, 0.015])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both', pad=0.1, orientation='horizontal')
cbar.ax.tick_params(labelsize=10)
cbar_ax.locator_params(nbins=5)
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=10, x=1.05, y=0.05, labelpad=-15)
# plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.92, hspace=0.25, wspace=0.25)
plt.savefig(path_results + '/sm_comp_jan_ori.png')
plt.close()


########################################################################################################################
# 5.3 Danube RB
# Read the map data
path_shp_dan = path_gis_data + '/wrd_riverbasins/Aqueduct_river_basins_DANUBE'
os.chdir(path_shp_dan)
shp_dan_file = "Aqueduct_river_basins_DANUBE.shp"
shp_dan_ds = ogr.GetDriverByName("ESRI Shapefile").Open(shp_dan_file, 0)
shp_dan_extent = list(shp_dan_ds.GetLayer().GetExtent())

#Load and subset the region of Danube RB (smos 25 km)
[lat_25km_dan, row_dan_25km_ind, lon_25km_dan, col_dan_25km_ind] = \
    coordtable_subset(lat_world_ease_25km, lon_world_ease_25km,
                      shp_dan_extent[3], shp_dan_extent[2], shp_dan_extent[1], shp_dan_extent[0])

# Load and subset smos 9 km SM of Danube RB
year_plt = [2020]
month_plt = list([7])
days_begin = 1
days_end = 6
days_n = days_end - days_begin + 1

smos_25km_mean_1_allyear = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        hdf_file_smos_25km = path_smos + '/25km' + '/smos_sm_25km_' + str(iyr) + monthname[month_plt[imo]-1] + '.hdf5'
        f_read_smos_25km = h5py.File(hdf_file_smos_25km, "r")
        varname_list_smos_25km = list(f_read_smos_25km.keys())

        # smos_25km_load = smos_25km_init
        smos_25km_load_1 = f_read_smos_25km[varname_list_smos_25km[0]][
                                           row_dan_25km_ind[0]:row_dan_25km_ind[-1] + 1,
                                           col_dan_25km_ind[0]:col_dan_25km_ind[-1] + 1, days_begin-1:days_end]  # AM
        smos_25km_load_2 = f_read_smos_25km[varname_list_smos_25km[1]][
                                           row_dan_25km_ind[0]:row_dan_25km_ind[-1] + 1,
                                           col_dan_25km_ind[0]:col_dan_25km_ind[-1] + 1, days_begin-1:days_end]  # PM
        f_read_smos_25km.close()

        smos_25km_mean_1 = np.nanmean(np.stack((smos_25km_load_1, smos_25km_load_2), axis=3), axis=3)
        smos_25km_mean_1_allyear.append(smos_25km_mean_1)

        del(smos_25km_load_1, smos_25km_mean_1)
        print(monthname[month_plt[imo]-1])

smos_25km_data_stack_dan = np.squeeze(np.array(smos_25km_mean_1_allyear))
del(smos_25km_mean_1_allyear)
smos_25km_data_stack_dan = np.transpose(smos_25km_data_stack_dan, (2, 0, 1))


#Load and subset the region of Danube RB (smos 1 km)
[lat_1km_dan, row_dan_1km_ind, lon_1km_dan, col_dan_1km_ind] = \
    coordtable_subset(lat_world_ease_1km, lon_world_ease_1km,
                      shp_dan_extent[3], shp_dan_extent[2], shp_dan_extent[1], shp_dan_extent[0])

smos_1km_mean_1_allyear = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        smos_1km_load_1_stack = []
        for idt in range(days_begin, days_end+1): # 30 days
            str_date = str(iyr) + '-' + monthname[month_plt[imo]-1] + '-' + str(idt).zfill(2)
            str_doy = str(datetime.datetime.strptime(str_date, '%Y-%m-%d').date().timetuple().tm_yday)
            tif_file_smos_1km = path_smos + '/1km/' + str(iyr) + '/smos_sm_1km_ds_' + str(iyr) + str_doy.zfill(3) + '.tif'
            src_tf = gdal.Open(tif_file_smos_1km)
            src_tf_arr = src_tf.ReadAsArray()[:, row_dan_1km_ind[0]:row_dan_1km_ind[-1]+1,
                         col_dan_1km_ind[0]:col_dan_1km_ind[-1]+1]
            smos_1km_load_1 = np.nanmean(src_tf_arr, axis=0)
            smos_1km_load_1_stack.append(smos_1km_load_1)

            print(str_date)
            del(src_tf_arr, smos_1km_load_1)

        smos_1km_load_1_stack = np.stack(smos_1km_load_1_stack)
        smos_1km_mean_1_allyear.append(smos_1km_load_1_stack)
        del (smos_1km_load_1_stack)

smos_1km_data_stack_dan = np.squeeze(np.array(smos_1km_mean_1_allyear))


with h5py.File(path_model_evaluation + '/smos_dan_sm.hdf5', 'w') as f:
    f.create_dataset('smos_25km_data_stack_dan', data=smos_25km_data_stack_dan)
    f.create_dataset('smos_1km_data_stack_dan', data=smos_1km_data_stack_dan)
f.close()

# Read the map data
f_read = h5py.File(path_model_evaluation + "/smos_dan_sm.hdf5", "r")
varname_read_list = ['smos_25km_data_stack_dan', 'smos_1km_data_stack_dan']
for x in range(len(varname_read_list)):
    var_obj = f_read[varname_read_list[x]][()]
    exec(varname_read_list[x] + '= var_obj')
    del(var_obj)
f_read.close()


# Subplot maps
# Load in watershed shapefile boundaries
shapefile_dan = fiona.open(path_shp_dan + '/' + shp_dan_file, 'r')
crop_shape_dan = [feature["geometry"] for feature in shapefile_dan]
# shp_dan_extent = list(shapefile_dan.bounds)

output_crs = 'EPSG:4326'

# Subset and reproject the smos SM data at watershed
# 1 km
masked_ds_dan_1km_all = []
for n in range(smos_1km_data_stack_dan.shape[0]):
    sub_window_dan_1km = Window(col_dan_1km_ind[0], row_dan_1km_ind[0], len(col_dan_1km_ind), len(row_dan_1km_ind))
    kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                      'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
    smos_sm_dan_1km_output = sub_n_reproj(smos_1km_data_stack_dan[n, :, :], kwargs_1km_sub, sub_window_dan_1km, output_crs)

    masked_ds_dan_1km, mask_transform_ds_dan_1km = mask(dataset=smos_sm_dan_1km_output, shapes=crop_shape_dan, crop=True)
    masked_ds_dan_1km[np.where(masked_ds_dan_1km == 0)] = np.nan
    masked_ds_dan_1km = masked_ds_dan_1km.squeeze()

    masked_ds_dan_1km_all.append(masked_ds_dan_1km)

masked_ds_dan_1km_all = np.asarray(masked_ds_dan_1km_all)


# 25 km
masked_ds_dan_25km_all = []
for n in range(smos_25km_data_stack_dan.shape[0]):
    sub_window_dan_25km = Window(col_dan_25km_ind[0], row_dan_25km_ind[0], len(col_dan_25km_ind), len(row_dan_25km_ind))
    kwargs_25km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_25km),
                      'height': len(lat_world_ease_25km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(25067.525, 0.0, -17367530.44516138, 0.0, -25067.525, 7314540.79258289)}
    smos_sm_dan_25km_output = sub_n_reproj(smos_25km_data_stack_dan[n, :, :], kwargs_25km_sub, sub_window_dan_25km, output_crs)

    masked_ds_dan_25km, mask_transform_ds_dan_25km = mask(dataset=smos_sm_dan_25km_output, shapes=crop_shape_dan, crop=True)
    masked_ds_dan_25km[np.where(masked_ds_dan_25km == 0)] = np.nan
    masked_ds_dan_25km = masked_ds_dan_25km.squeeze()

    masked_ds_dan_25km_all.append(masked_ds_dan_25km)

masked_ds_dan_25km_all = np.asarray(masked_ds_dan_25km_all)
# masked_ds_dan_25km_all[masked_ds_dan_25km_all >= 0.5] = np.nan


# Make the subplot maps
title_content = ['1 km\n(Sept 1 2020)', '25 km\n(Sept 1 2020)', '1 km\n(Sept 2 2020)', '25 km\n(Sept 2 2020)',
                 '1 km\n(Sept 3 2020)', '25 km\n(Sept 3 2020)', '1 km\n(Sept 4 2020)', '25 km\n(Sept 4 2020)',
                 '1 km\n(Sept 5 2020)', '25 km\n(Sept 5 2020)', '1 km\n(Sept 6 2020)', '25 km\n(Sept 6 2020)']
feature_shp_dan = ShapelyFeature(Reader(path_shp_dan + '/' + shp_dan_file).geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
extent_dan = np.array(smos_sm_dan_1km_output.bounds)
extent_dan = extent_dan[[0, 2, 1, 3]]

fig = plt.figure(figsize=(7, 10), facecolor='w', edgecolor='k', dpi=200)
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95, hspace=0.25, wspace=0.2)
for ipt in range(6):
    # 1 km
    ax = fig.add_subplot(6, 2, ipt*2+1, projection=ccrs.PlateCarree())
    ax.add_feature(feature_shp_dan)
    # ax.set_title(title_content[ipt*2], pad=20, fontsize=13, fontweight='bold')
    img = ax.imshow(masked_ds_dan_1km_all[ipt, :, :], origin='upper', vmin=0, vmax=0.5, cmap='Spectral',
               extent=extent_dan)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=5)
    gl.ylocator = mticker.MultipleLocator(base=4)
    gl.xlabel_style = {'size': 6}
    gl.ylabel_style = {'size': 6}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.text(8.5, 44, title_content[ipt*2], fontsize=7, horizontalalignment='left',
            verticalalignment='top', weight='bold')
    # 25 km
    ax = fig.add_subplot(6, 2, ipt*2+2, projection=ccrs.PlateCarree())
    ax.add_feature(feature_shp_dan)
    # ax.set_title(title_content[ipt*2+1], pad=20, fontsize=13, fontweight='bold')
    img = ax.imshow(masked_ds_dan_25km_all[ipt, :, :], origin='upper', vmin=0, vmax=0.5, cmap='Spectral',
               extent=extent_dan)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=5)
    gl.ylocator = mticker.MultipleLocator(base=4)
    gl.xlabel_style = {'size': 6}
    gl.ylabel_style = {'size': 6}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.text(8.5, 44, title_content[ipt*2+1], fontsize=7, horizontalalignment='left',
            verticalalignment='top', weight='bold')

cbar_ax = fig.add_axes([0.2, 0.04, 0.5, 0.015])
cbar = plt.colorbar(img, cax=cbar_ax, extend='both', orientation='horizontal', pad=0.1)
cbar.ax.tick_params(labelsize=7)
cbar_ax.locator_params(nbins=5)
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=7, x=1.08, y=0.05, labelpad=-15)
plt.savefig(path_results + '/sm_comp_dan_1.png')
plt.close()



########################################################################################################################
# 6. Maps to compare 1km/9km SMAP/SMOS SM
# 6.1 Maps of the world

smap_data = np.stack((smap_1km_mean_1_all, smap_1km_mean_2_all, smap_1km_mean_3_all,
                      smap_9km_mean_1_all, smap_9km_mean_2_all, smap_9km_mean_3_all), axis=0).squeeze()
smos_data = np.stack((smos_1km_mean_1_all, smos_1km_mean_2_all, smos_1km_mean_3_all,
                      smos_25km_mean_1_all, smos_25km_mean_2_all, smos_25km_mean_3_all), axis=0).squeeze()
with h5py.File(path_model_evaluation + '/smap_smos_sm.hdf5', 'w') as f:
    f.create_dataset('smap_data', data=smap_data)
    f.create_dataset('smos_data', data=smos_data)
f.close()

f_read = h5py.File(path_model_evaluation + '/smap_smos_sm.hdf5', "r")
smap_data = f_read['smap_data'][()]
smos_data = f_read['smos_data'][()]
f_read.close()
smap_data = smap_data[[0, 1, 3, 4], :, :]
smos_data = smos_data[[0, 1, 3, 4], :, :]


xx_wrd_9km, yy_wrd_9km = np.meshgrid(lon_world_ease_9km, lat_world_ease_9km) # Create the map matrix (9 km)
xx_wrd_25km, yy_wrd_25km = np.meshgrid(lon_world_ease_25km, lat_world_ease_25km) # Create the map matrix (25 km)
shape_world = ShapelyFeature(Reader(path_gis_data + '/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
shp_world_extent = [-180.0, 180.0, -70, 90]

# August
fig = plt.figure(figsize=(12, 9), facecolor='w', edgecolor='k', dpi=150)
plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.9, hspace=0.0, wspace=0.1)
for ipt in range(4):
    # SMOS
    ax = fig.add_subplot(4, 2, ipt+1, projection=ccrs.PlateCarree())
    ax.add_feature(shape_world, linewidth=0.5)
    img = ax.pcolormesh(xx_wrd_25km, yy_wrd_25km, smos_data[ipt, :, :], vmin=0, vmax=0.5, cmap='turbo_r')
    ax.set_extent([180, -180, -70, 90], ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = mticker.FixedLocator([90, 45, 0, -45, -90])
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    if ipt == 0 or ipt == 2:
        gl.left_labels = True
    else:
        gl.left_labels = False
    gl.right_labels = False
    gl.top_labels = False
    gl.bottom_labels = False

    # SMAP
    ax = fig.add_subplot(4, 2, ipt+5, projection=ccrs.PlateCarree())
    ax.add_feature(shape_world, linewidth=0.5)
    img = ax.pcolormesh(xx_wrd_9km, yy_wrd_9km, smap_data[ipt, :, :], vmin=0, vmax=0.5, cmap='turbo_r')
    ax.set_extent([180, -180, -70, 90], ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = mticker.FixedLocator([90, 45, 0, -45, -90])
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    if ipt == 2:
        gl.left_labels = True
        gl.bottom_labels = True
    elif ipt == 0:
        gl.left_labels = True
        gl.bottom_labels = False
    elif ipt == 3:
        gl.left_labels = False
        gl.bottom_labels = True
    elif ipt == 1:
        gl.left_labels = False
        gl.bottom_labels = False
    gl.right_labels = False
    gl.top_labels = False

cbar_ax = fig.add_axes([0.25, 0.04, 0.5, 0.015])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both', pad=0.1, orientation='horizontal')
cbar.ax.tick_params(labelsize=10)
cbar_ax.locator_params(nbins=5)
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=10, x=1.05, y=0.05, labelpad=-15)
fig.text(0.22, 0.95, 'August 1-3, 2021', fontsize=12, fontweight='bold')
fig.text(0.67, 0.95, 'August 4-6, 2021', fontsize=12, fontweight='bold')
fig.text(0.02, 0.75, 'SMOS 1km', fontsize=12, fontweight='bold', rotation=90)
fig.text(0.02, 0.55, 'SMOS 25km', fontsize=12, fontweight='bold', rotation=90)
fig.text(0.02, 0.35, 'SMAP 1km', fontsize=12, fontweight='bold', rotation=90)
fig.text(0.02, 0.15, 'SMAP 9km', fontsize=12, fontweight='bold', rotation=90)
plt.savefig(path_results + '/sm_comp_smos_smap.png')
plt.close()


# 6.2 River Basin maps
# 6.2.1 Middle Colorado RB (TxSON)

path_shp_txs = path_gis_data + '/watershed_boundary/'
os.chdir(path_shp_txs)
shp_txs_file = "txson_watersheds.shp"
shp_txs_ds = ogr.GetDriverByName("ESRI Shapefile").Open(shp_txs_file, 0)
shp_txs_extent = list(shp_txs_ds.GetLayer().GetExtent())

# 6.2.1.1 SMAP
#Load and subset the region of Middle Colorado RB (SMAP 9 km)
[lat_9km_txs, row_txs_9km_ind, lon_9km_txs, col_txs_9km_ind] = \
    coordtable_subset(lat_world_ease_9km, lon_world_ease_9km,
                      shp_txs_extent[3], shp_txs_extent[2], shp_txs_extent[1], shp_txs_extent[0])
[lat_1km_txs, row_txs_1km_ind, lon_1km_txs, col_txs_1km_ind] = \
    coordtable_subset(lat_world_ease_1km, lon_world_ease_1km,
                      shp_txs_extent[3], shp_txs_extent[2], shp_txs_extent[1], shp_txs_extent[0])

row_txs_9km_ind_dis = row_world_ease_1km_from_9km_ind[row_txs_1km_ind]
row_txs_9km_ind_dis_unique = np.unique(row_txs_9km_ind_dis)
row_txs_9km_ind_dis_zero = row_txs_9km_ind_dis - row_txs_9km_ind_dis[0]
col_txs_9km_ind_dis = col_world_ease_1km_from_9km_ind[col_txs_1km_ind]
col_txs_9km_ind_dis_unique = np.unique(col_txs_9km_ind_dis)
col_txs_9km_ind_dis_zero = col_txs_9km_ind_dis - col_txs_9km_ind_dis[0]

col_meshgrid_9km, row_meshgrid_9km = np.meshgrid(col_txs_9km_ind_dis_zero, row_txs_9km_ind_dis_zero)
col_meshgrid_9km = col_meshgrid_9km.reshape(1, -1).squeeze()
row_meshgrid_9km = row_meshgrid_9km.reshape(1, -1).squeeze()

# Load and subset SMAP 9 km SM of Middle Colorado RB
year_plt = [2021]
month_plt = list([8])
days_begin = 1
days_end = 31
days_n = days_end - days_begin + 1

smap_9km_mean_1_allyear = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        hdf_file_smap_9km = path_smap + '/9km' + '/smap_sm_9km_' + str(iyr) + monthname[month_plt[imo]-1] + '.hdf5'
        f_read_smap_9km = h5py.File(hdf_file_smap_9km, "r")
        varname_list_smap_9km = list(f_read_smap_9km.keys())

        smap_9km_load_1 = f_read_smap_9km[varname_list_smap_9km[0]][
                                           row_txs_9km_ind_dis_unique[0]:row_txs_9km_ind_dis_unique[-1] + 1,
                                           col_txs_9km_ind_dis_unique[0]:col_txs_9km_ind_dis_unique[-1] + 1, days_begin-1:days_end]  # AM
        smap_9km_load_2 = f_read_smap_9km[varname_list_smap_9km[1]][
                                           row_txs_9km_ind_dis_unique[0]:row_txs_9km_ind_dis_unique[-1] + 1,
                                           col_txs_9km_ind_dis_unique[0]:col_txs_9km_ind_dis_unique[-1] + 1, days_begin-1:days_end]  # PM

        smap_9km_load_1_disagg = np.array([smap_9km_load_1[row_meshgrid_9km[x], col_meshgrid_9km[x], :]
                                      for x in range(len(col_meshgrid_9km))])
        smap_9km_load_1_disagg = smap_9km_load_1_disagg.reshape((len(row_txs_9km_ind_dis), len(col_txs_9km_ind_dis),
                                                                 smap_9km_load_1.shape[2]))
        smap_9km_load_2_disagg = np.array([smap_9km_load_2[row_meshgrid_9km[x], col_meshgrid_9km[x], :]
                                      for x in range(len(col_meshgrid_9km))])
        smap_9km_load_2_disagg = smap_9km_load_2_disagg.reshape((len(row_txs_9km_ind_dis), len(col_txs_9km_ind_dis),
                                                                 smap_9km_load_2.shape[2]))
        f_read_smap_9km.close()

        smap_9km_mean_1 = np.nanmean(np.stack((smap_9km_load_1_disagg, smap_9km_load_2_disagg), axis=3), axis=3)
        smap_9km_mean_1_allyear.append(smap_9km_mean_1)

        del(smap_9km_load_1, smap_9km_mean_1, smap_9km_load_1_disagg, smap_9km_load_2_disagg)
        print(monthname[month_plt[imo]-1])

smap_9km_data_stack_txs = np.squeeze(np.array(smap_9km_mean_1_allyear))
del(smap_9km_mean_1_allyear)
smap_9km_data_stack_txs = np.transpose(smap_9km_data_stack_txs, (2, 0, 1))


#Load and subset the region of Middle Colorado RB (SMAP 1 km)
smap_1km_mean_1_allyear = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        smap_1km_load_1_stack = []
        for idt in range(days_begin, days_end+1):
            str_date = str(iyr) + '-' + monthname[month_plt[imo]-1] + '-' + str(idt).zfill(2)
            str_doy = str(datetime.datetime.strptime(str_date, '%Y-%m-%d').date().timetuple().tm_yday)
            tif_file_smap_1km = path_smap + '/1km/' + str(iyr) + '/smap_sm_1km_ds_' + str(iyr) + str_doy.zfill(3) + '.tif'
            src_tf = gdal.Open(tif_file_smap_1km)
            src_tf_arr = src_tf.ReadAsArray()[:, row_txs_1km_ind[0]:row_txs_1km_ind[-1]+1,
                         col_txs_1km_ind[0]:col_txs_1km_ind[-1]+1]
            smap_1km_load_1 = np.nanmean(src_tf_arr, axis=0)
            smap_1km_load_1_stack.append(smap_1km_load_1)

            print(str_date)
            del(src_tf_arr, smap_1km_load_1)

        smap_1km_load_1_stack = np.stack(smap_1km_load_1_stack)
        smap_1km_mean_1_allyear.append(smap_1km_load_1_stack)
        del(smap_1km_load_1_stack)

smap_1km_data_stack_txs = np.squeeze(np.array(smap_1km_mean_1_allyear))
del(smap_1km_mean_1_allyear)

with h5py.File(path_model_evaluation + '/smap_txs_sm.hdf5', 'w') as f:
    f.create_dataset('smap_9km_data_stack_txs', data=smap_9km_data_stack_txs)
    f.create_dataset('smap_1km_data_stack_txs', data=smap_1km_data_stack_txs)
f.close()

# Read the map data
f_read = h5py.File(path_model_evaluation + "/smap_txs_sm.hdf5", "r")
varname_read_list = ['smap_9km_data_stack_txs', 'smap_1km_data_stack_txs']
for x in range(len(varname_read_list)):
    var_obj = f_read[varname_read_list[x]][()]
    exec(varname_read_list[x] + '= var_obj')
    del(var_obj)
f_read.close()

# 6.2.1.2 SMOS

row_txs_25km_ind_dis = row_world_ease_1km_from_25km_ind[row_txs_1km_ind]
row_txs_25km_ind_dis_unique = np.unique(row_txs_25km_ind_dis)
row_txs_25km_ind_dis_zero = row_txs_25km_ind_dis - row_txs_25km_ind_dis[0]
col_txs_25km_ind_dis = col_world_ease_1km_from_25km_ind[col_txs_1km_ind]
col_txs_25km_ind_dis_unique = np.unique(col_txs_25km_ind_dis)
col_txs_25km_ind_dis_zero = col_txs_25km_ind_dis - col_txs_25km_ind_dis[0]

col_meshgrid_25km, row_meshgrid_25km = np.meshgrid(col_txs_25km_ind_dis_zero, row_txs_25km_ind_dis_zero)
col_meshgrid_25km = col_meshgrid_25km.reshape(1, -1).squeeze()
row_meshgrid_25km = row_meshgrid_25km.reshape(1, -1).squeeze()

#Load and subset the region of Middle Colorado RB (smos 25 km)
smos_25km_mean_1_allyear = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        hdf_file_smos_25km = path_smos + '/25km' + '/smos_sm_25km_' + str(iyr) + monthname[month_plt[imo]-1] + '.hdf5'
        f_read_smos_25km = h5py.File(hdf_file_smos_25km, "r")
        varname_list_smos_25km = list(f_read_smos_25km.keys())

        smos_25km_load_1 = f_read_smos_25km[varname_list_smos_25km[0]][
                                           row_txs_25km_ind_dis_unique[0]:row_txs_25km_ind_dis_unique[-1] + 1,
                                           col_txs_25km_ind_dis_unique[0]:col_txs_25km_ind_dis_unique[-1] + 1, days_begin-1:days_end]  # AM
        smos_25km_load_2 = f_read_smos_25km[varname_list_smos_25km[1]][
                                           row_txs_25km_ind_dis_unique[0]:row_txs_25km_ind_dis_unique[-1] + 1,
                                           col_txs_25km_ind_dis_unique[0]:col_txs_25km_ind_dis_unique[-1] + 1, days_begin-1:days_end]  # PM

        smos_25km_load_1_disagg = np.array([smos_25km_load_1[row_meshgrid_25km[x], col_meshgrid_25km[x], :]
                                      for x in range(len(col_meshgrid_25km))])
        smos_25km_load_1_disagg = smos_25km_load_1_disagg.reshape((len(row_txs_25km_ind_dis), len(col_txs_25km_ind_dis),
                                                                 smos_25km_load_1.shape[2]))
        smos_25km_load_2_disagg = np.array([smos_25km_load_2[row_meshgrid_25km[x], col_meshgrid_25km[x], :]
                                      for x in range(len(col_meshgrid_25km))])
        smos_25km_load_2_disagg = smos_25km_load_2_disagg.reshape((len(row_txs_25km_ind_dis), len(col_txs_25km_ind_dis),
                                                                 smos_25km_load_2.shape[2]))
        f_read_smos_25km.close()

        smos_25km_mean_1 = np.nanmean(np.stack((smos_25km_load_1_disagg, smos_25km_load_2_disagg), axis=3), axis=3)
        smos_25km_mean_1_allyear.append(smos_25km_mean_1)

        del(smos_25km_load_1, smos_25km_mean_1, smos_25km_load_1_disagg, smos_25km_load_2_disagg)
        print(monthname[month_plt[imo]-1])

smos_25km_data_stack_txs = np.squeeze(np.array(smos_25km_mean_1_allyear))
del(smos_25km_mean_1_allyear)
smos_25km_data_stack_txs = np.transpose(smos_25km_data_stack_txs, (2, 0, 1))

#Load and subset the region of Middle Colorado RB (smos 1 km)
smos_1km_mean_1_allyear = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        smos_1km_load_1_stack = []
        for idt in range(days_begin, days_end+1):
            str_date = str(iyr) + '-' + monthname[month_plt[imo]-1] + '-' + str(idt).zfill(2)
            str_doy = str(datetime.datetime.strptime(str_date, '%Y-%m-%d').date().timetuple().tm_yday)
            tif_file_smos_1km = path_smos + '/1km/' + str(iyr) + '/smos_sm_1km_ds_' + str(iyr) + str_doy.zfill(3) + '.tif'
            src_tf = gdal.Open(tif_file_smos_1km)
            src_tf_arr = src_tf.ReadAsArray()[:, row_txs_1km_ind[0]:row_txs_1km_ind[-1]+1,
                         col_txs_1km_ind[0]:col_txs_1km_ind[-1]+1]
            smos_1km_load_1 = np.nanmean(src_tf_arr, axis=0)
            smos_1km_load_1_stack.append(smos_1km_load_1)

            print(str_date)
            del(src_tf_arr, smos_1km_load_1)

        smos_1km_load_1_stack = np.stack(smos_1km_load_1_stack)
        smos_1km_mean_1_allyear.append(smos_1km_load_1_stack)
        del(smos_1km_load_1_stack)

smos_1km_data_stack_txs = np.squeeze(np.array(smos_1km_mean_1_allyear))
del(smos_1km_mean_1_allyear)

with h5py.File(path_model_evaluation + '/smos_txs_sm.hdf5', 'w') as f:
    f.create_dataset('smos_25km_data_stack_txs', data=smos_25km_data_stack_txs)
    f.create_dataset('smos_1km_data_stack_txs', data=smos_1km_data_stack_txs)
f.close()


# Read the map data
f_read = h5py.File(path_model_evaluation + "/smos_txs_sm.hdf5", "r")
varname_read_list = ['smos_25km_data_stack_txs', 'smos_1km_data_stack_txs']
for x in range(len(varname_read_list)):
    var_obj = f_read[varname_read_list[x]][()]
    exec(varname_read_list[x] + '= var_obj')
    del(var_obj)
f_read.close()


# 6.2.2 Subplot maps
output_crs = 'EPSG:4326'
shapefile_txs = fiona.open(path_shp_txs + '/' + shp_txs_file, 'r')
crop_shape_txs = [feature["geometry"] for feature in shapefile_txs]

# 6.2.2.1 Subset and reproject the SMAP SM data at watershed
# 1 km
smap_masked_ds_txs_1km_all = []
for n in range(smap_1km_data_stack_txs.shape[0]):
    sub_window_txs_1km = Window(col_txs_1km_ind[0], row_txs_1km_ind[0], len(col_txs_1km_ind), len(row_txs_1km_ind))
    kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                      'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
    smap_sm_txs_1km_output = sub_n_reproj(smap_1km_data_stack_txs[n, :, :], kwargs_1km_sub, sub_window_txs_1km, output_crs)

    masked_ds_txs_1km, mask_transform_ds_txs_1km = mask(dataset=smap_sm_txs_1km_output, shapes=crop_shape_txs, crop=True)
    masked_ds_txs_1km[np.where(masked_ds_txs_1km == 0)] = np.nan
    masked_ds_txs_1km = masked_ds_txs_1km.squeeze()

    smap_masked_ds_txs_1km_all.append(masked_ds_txs_1km)

smap_masked_ds_txs_1km_all = np.asarray(smap_masked_ds_txs_1km_all)


# 9 km
smap_masked_ds_txs_9km_all = []
for n in range(smap_9km_data_stack_txs.shape[0]):
    sub_window_txs_9km = Window(col_txs_1km_ind[0], row_txs_1km_ind[0], len(col_txs_1km_ind), len(row_txs_1km_ind))
    kwargs_9km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                      'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
    smap_sm_txs_9km_output = sub_n_reproj(smap_9km_data_stack_txs[n, :, :], kwargs_9km_sub, sub_window_txs_9km, output_crs)

    masked_ds_txs_9km, mask_transform_ds_txs_9km = mask(dataset=smap_sm_txs_9km_output, shapes=crop_shape_txs, crop=True)
    masked_ds_txs_9km[np.where(masked_ds_txs_9km == 0)] = np.nan
    masked_ds_txs_9km = masked_ds_txs_9km.squeeze()

    smap_masked_ds_txs_9km_all.append(masked_ds_txs_9km)

smap_masked_ds_txs_9km_all = np.asarray(smap_masked_ds_txs_9km_all)
# masked_ds_txs_9km_all[masked_ds_txs_9km_all >= 0.5] = np.nan

# 6.2.2.2 Subset and reproject the SMOS SM data at watershed
# 1 km
smos_masked_ds_txs_1km_all = []
for n in range(smos_1km_data_stack_txs.shape[0]):
    sub_window_txs_1km = Window(col_txs_1km_ind[0], row_txs_1km_ind[0], len(col_txs_1km_ind), len(row_txs_1km_ind))
    kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                      'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
    smos_sm_txs_1km_output = sub_n_reproj(smos_1km_data_stack_txs[n, :, :], kwargs_1km_sub, sub_window_txs_1km, output_crs)

    masked_ds_txs_1km, mask_transform_ds_txs_1km = mask(dataset=smos_sm_txs_1km_output, shapes=crop_shape_txs, crop=True)
    masked_ds_txs_1km[np.where(masked_ds_txs_1km == 0)] = np.nan
    masked_ds_txs_1km = masked_ds_txs_1km.squeeze()

    smos_masked_ds_txs_1km_all.append(masked_ds_txs_1km)

smos_masked_ds_txs_1km_all = np.asarray(smos_masked_ds_txs_1km_all)


# 25 km
smos_masked_ds_txs_25km_all = []
for n in range(smos_25km_data_stack_txs.shape[0]):
    sub_window_txs_25km = Window(col_txs_1km_ind[0], row_txs_1km_ind[0], len(col_txs_1km_ind), len(row_txs_1km_ind))
    kwargs_25km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                      'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
    smos_sm_txs_25km_output = sub_n_reproj(smos_25km_data_stack_txs[n, :, :], kwargs_25km_sub, sub_window_txs_25km, output_crs)

    masked_ds_txs_25km, mask_transform_ds_txs_25km = mask(dataset=smos_sm_txs_25km_output, shapes=crop_shape_txs, crop=True)
    masked_ds_txs_25km[np.where(masked_ds_txs_25km == 0)] = np.nan
    masked_ds_txs_25km = masked_ds_txs_25km.squeeze()

    smos_masked_ds_txs_25km_all.append(masked_ds_txs_25km)

smos_masked_ds_txs_25km_all = np.asarray(smos_masked_ds_txs_25km_all)
# masked_ds_txs_25km_all[masked_ds_txs_25km_all >= 0.5] = np.nan

# Calculate the 3-day averaged maps
smap_1km_size = smap_masked_ds_txs_1km_all.shape
smap_masked_ds_txs_1km_avg = np.reshape(smap_masked_ds_txs_1km_all[:30, :, :], (3, 10, smap_1km_size[1], smap_1km_size[2]))
smap_masked_ds_txs_1km_avg = np.nanmean(smap_masked_ds_txs_1km_avg, axis=0)
smap_masked_ds_txs_9km_avg = np.reshape(smap_masked_ds_txs_9km_all[:30, :, :], (3, 10, smap_1km_size[1], smap_1km_size[2]))
smap_masked_ds_txs_9km_avg = np.nanmean(smap_masked_ds_txs_9km_avg, axis=0)
smos_masked_ds_txs_1km_avg = np.reshape(smos_masked_ds_txs_1km_all[:30, :, :], (3, 10, smap_1km_size[1], smap_1km_size[2]))
smos_masked_ds_txs_1km_avg = np.nanmean(smos_masked_ds_txs_1km_avg, axis=0)
smos_masked_ds_txs_25km_avg = np.reshape(smos_masked_ds_txs_25km_all[:30, :, :], (3, 10, smap_1km_size[1], smap_1km_size[2]))
smos_masked_ds_txs_25km_avg = np.nanmean(smos_masked_ds_txs_25km_avg, axis=0)
sm_masked_ds_txs_stack = list((smos_masked_ds_txs_1km_avg, smos_masked_ds_txs_25km_avg,
                               smap_masked_ds_txs_1km_avg, smap_masked_ds_txs_9km_avg))

# 6.2.2.3 Make the subplot maps
feature_shp_txs = ShapelyFeature(Reader(path_shp_txs + '/' + shp_txs_file).geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
extent_txs = np.array(shapefile_txs.bounds)
extent_txs = extent_txs[[0, 2, 1, 3]]

fig = plt.figure(figsize=(9, 10), facecolor='w', edgecolor='k', dpi=150)
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.95, hspace=0.0, wspace=0.0)
for irow in range(10):
    for icol in range(4):
        # 1 km
        ax = fig.add_subplot(10, 4, irow*4+icol+1, projection=ccrs.PlateCarree())
        ax.add_feature(feature_shp_txs)
        img = ax.imshow(sm_masked_ds_txs_stack[icol][irow, :, :], origin='upper', vmin=0, vmax=0.4, cmap='turbo_r',
                   extent=extent_txs)
        gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
        gl.xlocator = mticker.MultipleLocator(base=1)
        gl.ylocator = mticker.MultipleLocator(base=0.5)
        gl.xlabel_style = {'size': 6}
        gl.ylabel_style = {'size': 6}
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        if icol == 0:
            gl.left_labels = True
            gl.bottom_labels = False
        elif irow == 9:
            gl.left_labels = False
            gl.bottom_labels = True
        else:
            gl.left_labels = False
            gl.bottom_labels = False
        if icol == 0 and irow == 9:
            gl.left_labels = True
            gl.bottom_labels = True
        gl.right_labels = False
        gl.top_labels = False
cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both')
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=9, x=0.95)
cbar.ax.tick_params(labelsize=9)
cbar_ax.locator_params(nbins=6)
fig.text(0.15, 0.98, 'SMOS 1km', fontsize=10, fontweight='bold')
fig.text(0.35, 0.98, 'SMOS 25km', fontsize=10, fontweight='bold')
fig.text(0.55, 0.98, 'SMAP 1km', fontsize=10, fontweight='bold')
fig.text(0.75, 0.98, 'SMAP 9km', fontsize=10, fontweight='bold')
fig.text(0.04, 0.87, 'Aug 1-3', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.04, 0.78, 'Aug 4-6', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.04, 0.69, 'Aug 7-9', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.04, 0.60, 'Aug 10-12', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.04, 0.51, 'Aug 13-15', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.04, 0.42, 'Aug 16-18', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.04, 0.33, 'Aug 19-21', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.04, 0.24, 'Aug 22-24', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.04, 0.15, 'Aug 25-27', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.04, 0.06, 'Aug 28-31', fontsize=8, fontweight='bold', rotation=90)
# cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=7, x=1.08, y=0.05, labelpad=-15)
plt.savefig(path_results + '/sm_comp_txs_1.png')
plt.close()


# Shorter version
fig = plt.figure(figsize=(9, 4), facecolor='w', edgecolor='k', dpi=200)
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.95, hspace=0.0, wspace=0.0)
for irow in range(3):
    for icol in range(4):
        # 1 km
        ax = fig.add_subplot(3, 4, irow*4+icol+1, projection=ccrs.PlateCarree())
        ax.add_feature(feature_shp_txs)
        img = ax.imshow(sm_masked_ds_txs_stack[icol][irow+2, :, :], origin='upper', vmin=0, vmax=0.4, cmap='turbo_r',
                   extent=extent_txs)
        gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
        gl.xlocator = mticker.MultipleLocator(base=1)
        gl.ylocator = mticker.MultipleLocator(base=0.5)
        gl.xlabel_style = {'size': 6}
        gl.ylabel_style = {'size': 6}
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        if icol == 0:
            gl.left_labels = True
            gl.bottom_labels = False
        elif irow == 2:
            gl.left_labels = False
            gl.bottom_labels = True
        else:
            gl.left_labels = False
            gl.bottom_labels = False
        if icol == 0 and irow == 2:
            gl.left_labels = True
            gl.bottom_labels = True
        gl.right_labels = False
        gl.top_labels = False
cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both')
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=9, x=0.95)
cbar.ax.tick_params(labelsize=9)
cbar_ax.locator_params(nbins=6)
fig.text(0.15, 0.94, 'SMOS 1km', fontsize=10, fontweight='bold')
fig.text(0.35, 0.94, 'SMOS 25km', fontsize=10, fontweight='bold')
fig.text(0.55, 0.94, 'SMAP 1km', fontsize=10, fontweight='bold')
fig.text(0.75, 0.94, 'SMAP 9km', fontsize=10, fontweight='bold')
fig.text(0.03, 0.75, 'Aug 7-9', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.03, 0.45, 'Aug 10-12', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.03, 0.15, 'Aug 13-15', fontsize=8, fontweight='bold', rotation=90)
# cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=7, x=1.08, y=0.05, labelpad=-15)
plt.savefig(path_results + '/sm_comp_txs_1_new.png')
plt.close()




# 6.2.3 Nu-Salween RB
path_shp_nur = path_gis_data + '/watershed_boundary/'
os.chdir(path_shp_nur)
shp_nur_file = "nu_salween_upper_riverbasin.shp"
shp_nur_ds = ogr.GetDriverByName("ESRI Shapefile").Open(shp_nur_file, 0)
shp_nur_extent = list(shp_nur_ds.GetLayer().GetExtent())

# 6.2.3.1 SMAP
#Load and subset the region of Nu-Salween RB (SMAP 9 km)
[lat_9km_nur, row_nur_9km_ind, lon_9km_nur, col_nur_9km_ind] = \
    coordtable_subset(lat_world_ease_9km, lon_world_ease_9km,
                      shp_nur_extent[3], shp_nur_extent[2], shp_nur_extent[1], shp_nur_extent[0])
[lat_1km_nur, row_nur_1km_ind, lon_1km_nur, col_nur_1km_ind] = \
    coordtable_subset(lat_world_ease_1km, lon_world_ease_1km,
                      shp_nur_extent[3], shp_nur_extent[2], shp_nur_extent[1], shp_nur_extent[0])

row_nur_9km_ind_dis = row_world_ease_1km_from_9km_ind[row_nur_1km_ind]
row_nur_9km_ind_dis_unique = np.unique(row_nur_9km_ind_dis)
row_nur_9km_ind_dis_zero = row_nur_9km_ind_dis - row_nur_9km_ind_dis[0]
col_nur_9km_ind_dis = col_world_ease_1km_from_9km_ind[col_nur_1km_ind]
col_nur_9km_ind_dis_unique = np.unique(col_nur_9km_ind_dis)
col_nur_9km_ind_dis_zero = col_nur_9km_ind_dis - col_nur_9km_ind_dis[0]

col_meshgrid_9km, row_meshgrid_9km = np.meshgrid(col_nur_9km_ind_dis_zero, row_nur_9km_ind_dis_zero)
col_meshgrid_9km = col_meshgrid_9km.reshape(1, -1).squeeze()
row_meshgrid_9km = row_meshgrid_9km.reshape(1, -1).squeeze()

# Load and subset SMAP 9 km SM of Nu-Salween RB
year_plt = [2021]
month_plt = list([8])
days_begin = 1
days_end = 31
days_n = days_end - days_begin + 1

smap_9km_mean_1_allyear = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        hdf_file_smap_9km = path_smap + '/9km' + '/smap_sm_9km_' + str(iyr) + monthname[month_plt[imo]-1] + '.hdf5'
        f_read_smap_9km = h5py.File(hdf_file_smap_9km, "r")
        varname_list_smap_9km = list(f_read_smap_9km.keys())

        smap_9km_load_1 = f_read_smap_9km[varname_list_smap_9km[0]][
                                           row_nur_9km_ind_dis_unique[0]:row_nur_9km_ind_dis_unique[-1] + 1,
                                           col_nur_9km_ind_dis_unique[0]:col_nur_9km_ind_dis_unique[-1] + 1, days_begin-1:days_end]  # AM
        smap_9km_load_2 = f_read_smap_9km[varname_list_smap_9km[1]][
                                           row_nur_9km_ind_dis_unique[0]:row_nur_9km_ind_dis_unique[-1] + 1,
                                           col_nur_9km_ind_dis_unique[0]:col_nur_9km_ind_dis_unique[-1] + 1, days_begin-1:days_end]  # PM

        smap_9km_load_1_disagg = np.array([smap_9km_load_1[row_meshgrid_9km[x], col_meshgrid_9km[x], :]
                                      for x in range(len(col_meshgrid_9km))])
        smap_9km_load_1_disagg = smap_9km_load_1_disagg.reshape((len(row_nur_9km_ind_dis), len(col_nur_9km_ind_dis),
                                                                 smap_9km_load_1.shape[2]))
        smap_9km_load_2_disagg = np.array([smap_9km_load_2[row_meshgrid_9km[x], col_meshgrid_9km[x], :]
                                      for x in range(len(col_meshgrid_9km))])
        smap_9km_load_2_disagg = smap_9km_load_2_disagg.reshape((len(row_nur_9km_ind_dis), len(col_nur_9km_ind_dis),
                                                                 smap_9km_load_2.shape[2]))
        f_read_smap_9km.close()

        smap_9km_mean_1 = np.nanmean(np.stack((smap_9km_load_1_disagg, smap_9km_load_2_disagg), axis=3), axis=3)
        smap_9km_mean_1_allyear.append(smap_9km_mean_1)

        del(smap_9km_load_1, smap_9km_mean_1, smap_9km_load_1_disagg, smap_9km_load_2_disagg)
        print(monthname[month_plt[imo]-1])

smap_9km_data_stack_nur = np.squeeze(np.array(smap_9km_mean_1_allyear))
del(smap_9km_mean_1_allyear)
smap_9km_data_stack_nur = np.transpose(smap_9km_data_stack_nur, (2, 0, 1))


#Load and subset the region of Nu-Salween RB (SMAP 1 km)
smap_1km_mean_1_allyear = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        smap_1km_load_1_stack = []
        for idt in range(days_begin, days_end+1):
            str_date = str(iyr) + '-' + monthname[month_plt[imo]-1] + '-' + str(idt).zfill(2)
            str_doy = str(datetime.datetime.strptime(str_date, '%Y-%m-%d').date().timetuple().tm_yday)
            tif_file_smap_1km = path_smap + '/1km/' + str(iyr) + '/smap_sm_1km_ds_' + str(iyr) + str_doy.zfill(3) + '.tif'
            src_tf = gdal.Open(tif_file_smap_1km)
            src_tf_arr = src_tf.ReadAsArray()[:, row_nur_1km_ind[0]:row_nur_1km_ind[-1]+1,
                         col_nur_1km_ind[0]:col_nur_1km_ind[-1]+1]
            smap_1km_load_1 = np.nanmean(src_tf_arr, axis=0)
            smap_1km_load_1_stack.append(smap_1km_load_1)

            print(str_date)
            del(src_tf_arr, smap_1km_load_1)

        smap_1km_load_1_stack = np.stack(smap_1km_load_1_stack)
        smap_1km_mean_1_allyear.append(smap_1km_load_1_stack)
        del(smap_1km_load_1_stack)

smap_1km_data_stack_nur = np.squeeze(np.array(smap_1km_mean_1_allyear))
del(smap_1km_mean_1_allyear)

with h5py.File(path_model_evaluation + '/smap_nur_sm.hdf5', 'w') as f:
    f.create_dataset('smap_9km_data_stack_nur', data=smap_9km_data_stack_nur)
    f.create_dataset('smap_1km_data_stack_nur', data=smap_1km_data_stack_nur)
f.close()

# Read the map data
f_read = h5py.File(path_model_evaluation + "/smap_nur_sm.hdf5", "r")
varname_read_list = ['smap_9km_data_stack_nur', 'smap_1km_data_stack_nur']
for x in range(len(varname_read_list)):
    var_obj = f_read[varname_read_list[x]][()]
    exec(varname_read_list[x] + '= var_obj')
    del(var_obj)
f_read.close()

# 6.2.3.2 SMOS

row_nur_25km_ind_dis = row_world_ease_1km_from_25km_ind[row_nur_1km_ind]
row_nur_25km_ind_dis_unique = np.unique(row_nur_25km_ind_dis)
row_nur_25km_ind_dis_zero = row_nur_25km_ind_dis - row_nur_25km_ind_dis[0]
col_nur_25km_ind_dis = col_world_ease_1km_from_25km_ind[col_nur_1km_ind]
col_nur_25km_ind_dis_unique = np.unique(col_nur_25km_ind_dis)
col_nur_25km_ind_dis_zero = col_nur_25km_ind_dis - col_nur_25km_ind_dis[0]

col_meshgrid_25km, row_meshgrid_25km = np.meshgrid(col_nur_25km_ind_dis_zero, row_nur_25km_ind_dis_zero)
col_meshgrid_25km = col_meshgrid_25km.reshape(1, -1).squeeze()
row_meshgrid_25km = row_meshgrid_25km.reshape(1, -1).squeeze()

#Load and subset the region of Nu-Salween RB (smos 25 km)
smos_25km_mean_1_allyear = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        hdf_file_smos_25km = path_smos + '/25km' + '/smos_sm_25km_' + str(iyr) + monthname[month_plt[imo]-1] + '.hdf5'
        f_read_smos_25km = h5py.File(hdf_file_smos_25km, "r")
        varname_list_smos_25km = list(f_read_smos_25km.keys())

        smos_25km_load_1 = f_read_smos_25km[varname_list_smos_25km[0]][
                                           row_nur_25km_ind_dis_unique[0]:row_nur_25km_ind_dis_unique[-1] + 1,
                                           col_nur_25km_ind_dis_unique[0]:col_nur_25km_ind_dis_unique[-1] + 1, days_begin-1:days_end]  # AM
        smos_25km_load_2 = f_read_smos_25km[varname_list_smos_25km[1]][
                                           row_nur_25km_ind_dis_unique[0]:row_nur_25km_ind_dis_unique[-1] + 1,
                                           col_nur_25km_ind_dis_unique[0]:col_nur_25km_ind_dis_unique[-1] + 1, days_begin-1:days_end]  # PM

        smos_25km_load_1_disagg = np.array([smos_25km_load_1[row_meshgrid_25km[x], col_meshgrid_25km[x], :]
                                      for x in range(len(col_meshgrid_25km))])
        smos_25km_load_1_disagg = smos_25km_load_1_disagg.reshape((len(row_nur_25km_ind_dis), len(col_nur_25km_ind_dis),
                                                                 smos_25km_load_1.shape[2]))
        smos_25km_load_2_disagg = np.array([smos_25km_load_2[row_meshgrid_25km[x], col_meshgrid_25km[x], :]
                                      for x in range(len(col_meshgrid_25km))])
        smos_25km_load_2_disagg = smos_25km_load_2_disagg.reshape((len(row_nur_25km_ind_dis), len(col_nur_25km_ind_dis),
                                                                 smos_25km_load_2.shape[2]))
        f_read_smos_25km.close()

        smos_25km_mean_1 = np.nanmean(np.stack((smos_25km_load_1_disagg, smos_25km_load_2_disagg), axis=3), axis=3)
        smos_25km_mean_1_allyear.append(smos_25km_mean_1)

        del(smos_25km_load_1, smos_25km_mean_1, smos_25km_load_1_disagg, smos_25km_load_2_disagg)
        print(monthname[month_plt[imo]-1])

smos_25km_data_stack_nur = np.squeeze(np.array(smos_25km_mean_1_allyear))
del(smos_25km_mean_1_allyear)
smos_25km_data_stack_nur = np.transpose(smos_25km_data_stack_nur, (2, 0, 1))

#Load and subset the region of Nu-Salween RB (smos 1 km)
smos_1km_mean_1_allyear = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        smos_1km_load_1_stack = []
        for idt in range(days_begin, days_end+1):
            str_date = str(iyr) + '-' + monthname[month_plt[imo]-1] + '-' + str(idt).zfill(2)
            str_doy = str(datetime.datetime.strptime(str_date, '%Y-%m-%d').date().timetuple().tm_yday)
            tif_file_smos_1km = path_smos + '/1km/' + str(iyr) + '/smos_sm_1km_ds_' + str(iyr) + str_doy.zfill(3) + '.tif'
            src_tf = gdal.Open(tif_file_smos_1km)
            src_tf_arr = src_tf.ReadAsArray()[:, row_nur_1km_ind[0]:row_nur_1km_ind[-1]+1,
                         col_nur_1km_ind[0]:col_nur_1km_ind[-1]+1]
            smos_1km_load_1 = np.nanmean(src_tf_arr, axis=0)
            smos_1km_load_1_stack.append(smos_1km_load_1)

            print(str_date)
            del(src_tf_arr, smos_1km_load_1)

        smos_1km_load_1_stack = np.stack(smos_1km_load_1_stack)
        smos_1km_mean_1_allyear.append(smos_1km_load_1_stack)
        del(smos_1km_load_1_stack)

smos_1km_data_stack_nur = np.squeeze(np.array(smos_1km_mean_1_allyear))
del(smos_1km_mean_1_allyear)

with h5py.File(path_model_evaluation + '/smos_nur_sm.hdf5', 'w') as f:
    f.create_dataset('smos_25km_data_stack_nur', data=smos_25km_data_stack_nur)
    f.create_dataset('smos_1km_data_stack_nur', data=smos_1km_data_stack_nur)
f.close()

# Read the map data
f_read = h5py.File(path_model_evaluation + "/smos_nur_sm.hdf5", "r")
varname_read_list = ['smos_25km_data_stack_nur', 'smos_1km_data_stack_nur']
for x in range(len(varname_read_list)):
    var_obj = f_read[varname_read_list[x]][()]
    exec(varname_read_list[x] + '= var_obj')
    del(var_obj)
f_read.close()


# 6.2.3.3 Subplot maps
output_crs = 'EPSG:4326'
shapefile_nur = fiona.open(path_shp_nur + '/' + shp_nur_file, 'r')
crop_shape_nur = [feature["geometry"] for feature in shapefile_nur]

# 6.2.3.3.1 Subset and reproject the SMAP SM data at watershed
# 1 km
smap_masked_ds_nur_1km_all = []
for n in range(smap_1km_data_stack_nur.shape[0]):
    sub_window_nur_1km = Window(col_nur_1km_ind[0], row_nur_1km_ind[0], len(col_nur_1km_ind), len(row_nur_1km_ind))
    kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                      'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
    smap_sm_nur_1km_output = sub_n_reproj(smap_1km_data_stack_nur[n, :, :], kwargs_1km_sub, sub_window_nur_1km, output_crs)

    masked_ds_nur_1km, mask_transform_ds_nur_1km = mask(dataset=smap_sm_nur_1km_output, shapes=crop_shape_nur, crop=True)
    masked_ds_nur_1km[np.where(masked_ds_nur_1km == 0)] = np.nan
    masked_ds_nur_1km = masked_ds_nur_1km.squeeze()

    smap_masked_ds_nur_1km_all.append(masked_ds_nur_1km)

smap_masked_ds_nur_1km_all = np.asarray(smap_masked_ds_nur_1km_all)


# 9 km
smap_masked_ds_nur_9km_all = []
for n in range(smap_9km_data_stack_nur.shape[0]):
    sub_window_nur_9km = Window(col_nur_1km_ind[0], row_nur_1km_ind[0], len(col_nur_1km_ind), len(row_nur_1km_ind))
    kwargs_9km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                      'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
    smap_sm_nur_9km_output = sub_n_reproj(smap_9km_data_stack_nur[n, :, :], kwargs_9km_sub, sub_window_nur_9km, output_crs)

    masked_ds_nur_9km, mask_transform_ds_nur_9km = mask(dataset=smap_sm_nur_9km_output, shapes=crop_shape_nur, crop=True)
    masked_ds_nur_9km[np.where(masked_ds_nur_9km == 0)] = np.nan
    masked_ds_nur_9km = masked_ds_nur_9km.squeeze()

    smap_masked_ds_nur_9km_all.append(masked_ds_nur_9km)

smap_masked_ds_nur_9km_all = np.asarray(smap_masked_ds_nur_9km_all)
# masked_ds_nur_9km_all[masked_ds_nur_9km_all >= 0.5] = np.nan

# 6.2.3.3.2 Subset and reproject the SMOS SM data at watershed
# 1 km
smos_masked_ds_nur_1km_all = []
for n in range(smos_1km_data_stack_nur.shape[0]):
    sub_window_nur_1km = Window(col_nur_1km_ind[0], row_nur_1km_ind[0], len(col_nur_1km_ind), len(row_nur_1km_ind))
    kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                      'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
    smos_sm_nur_1km_output = sub_n_reproj(smos_1km_data_stack_nur[n, :, :], kwargs_1km_sub, sub_window_nur_1km, output_crs)

    masked_ds_nur_1km, mask_transform_ds_nur_1km = mask(dataset=smos_sm_nur_1km_output, shapes=crop_shape_nur, crop=True)
    masked_ds_nur_1km[np.where(masked_ds_nur_1km == 0)] = np.nan
    masked_ds_nur_1km = masked_ds_nur_1km.squeeze()

    smos_masked_ds_nur_1km_all.append(masked_ds_nur_1km)

smos_masked_ds_nur_1km_all = np.asarray(smos_masked_ds_nur_1km_all)


# 25 km
smos_masked_ds_nur_25km_all = []
for n in range(smos_25km_data_stack_nur.shape[0]):
    sub_window_nur_25km = Window(col_nur_1km_ind[0], row_nur_1km_ind[0], len(col_nur_1km_ind), len(row_nur_1km_ind))
    kwargs_25km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                      'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
    smos_sm_nur_25km_output = sub_n_reproj(smos_25km_data_stack_nur[n, :, :], kwargs_25km_sub, sub_window_nur_25km, output_crs)

    masked_ds_nur_25km, mask_transform_ds_nur_25km = mask(dataset=smos_sm_nur_25km_output, shapes=crop_shape_nur, crop=True)
    masked_ds_nur_25km[np.where(masked_ds_nur_25km == 0)] = np.nan
    masked_ds_nur_25km = masked_ds_nur_25km.squeeze()

    smos_masked_ds_nur_25km_all.append(masked_ds_nur_25km)

smos_masked_ds_nur_25km_all = np.asarray(smos_masked_ds_nur_25km_all)
# masked_ds_nur_25km_all[masked_ds_nur_25km_all >= 0.5] = np.nan

# Calculate the 3-day averaged maps
smap_1km_size = smap_masked_ds_nur_1km_all.shape
smap_masked_ds_nur_1km_avg = np.reshape(smap_masked_ds_nur_1km_all[:30, :, :], (3, 10, smap_1km_size[1], smap_1km_size[2]))
smap_masked_ds_nur_1km_avg = np.nanmean(smap_masked_ds_nur_1km_avg, axis=0)
smap_masked_ds_nur_9km_avg = np.reshape(smap_masked_ds_nur_9km_all[:30, :, :], (3, 10, smap_1km_size[1], smap_1km_size[2]))
smap_masked_ds_nur_9km_avg = np.nanmean(smap_masked_ds_nur_9km_avg, axis=0)
smos_masked_ds_nur_1km_avg = np.reshape(smos_masked_ds_nur_1km_all[:30, :, :], (3, 10, smap_1km_size[1], smap_1km_size[2]))
smos_masked_ds_nur_1km_avg = np.nanmean(smos_masked_ds_nur_1km_avg, axis=0)
smos_masked_ds_nur_25km_avg = np.reshape(smos_masked_ds_nur_25km_all[:30, :, :], (3, 10, smap_1km_size[1], smap_1km_size[2]))
smos_masked_ds_nur_25km_avg = np.nanmean(smos_masked_ds_nur_25km_avg, axis=0)
sm_masked_ds_nur_stack = list((smos_masked_ds_nur_1km_avg, smos_masked_ds_nur_25km_avg,
                               smap_masked_ds_nur_1km_avg, smap_masked_ds_nur_9km_avg))

# 6.2.3.4 Make the subplot maps
feature_shp_nur = ShapelyFeature(Reader(path_shp_nur + '/' + shp_nur_file).geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
extent_nur = np.array(shapefile_nur.bounds)
extent_nur = extent_nur[[0, 2, 1, 3]]

fig = plt.figure(figsize=(9, 10), facecolor='w', edgecolor='k', dpi=150)
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.95, hspace=0.0, wspace=0.0)
for irow in range(10):
    for icol in range(4):
        # 1 km
        ax = fig.add_subplot(10, 4, irow*4+icol+1, projection=ccrs.PlateCarree())
        ax.add_feature(feature_shp_nur)
        img = ax.imshow(sm_masked_ds_nur_stack[icol][irow, :, :], origin='upper', vmin=0, vmax=0.6, cmap='turbo_r',
                   extent=extent_nur)
        gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
        gl.xlocator = mticker.MultipleLocator(base=1)
        gl.ylocator = mticker.MultipleLocator(base=0.5)
        gl.xlabel_style = {'size': 6}
        gl.ylabel_style = {'size': 6}
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        if icol == 0:
            gl.left_labels = True
            gl.bottom_labels = False
        elif irow == 9:
            gl.left_labels = False
            gl.bottom_labels = True
        else:
            gl.left_labels = False
            gl.bottom_labels = False
        if icol == 0 and irow == 9:
            gl.left_labels = True
            gl.bottom_labels = True
        gl.right_labels = False
        gl.top_labels = False
cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both')
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=9, x=0.95)
cbar.ax.tick_params(labelsize=9)
cbar_ax.locator_params(nbins=6)
fig.text(0.15, 0.98, 'SMOS 1km', fontsize=10, fontweight='bold')
fig.text(0.35, 0.98, 'SMOS 25km', fontsize=10, fontweight='bold')
fig.text(0.55, 0.98, 'SMAP 1km', fontsize=10, fontweight='bold')
fig.text(0.75, 0.98, 'SMAP 9km', fontsize=10, fontweight='bold')
fig.text(0.04, 0.87, 'Aug 1-3', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.04, 0.78, 'Aug 4-6', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.04, 0.69, 'Aug 7-9', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.04, 0.60, 'Aug 10-12', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.04, 0.51, 'Aug 13-15', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.04, 0.42, 'Aug 16-18', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.04, 0.33, 'Aug 19-21', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.04, 0.24, 'Aug 22-24', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.04, 0.15, 'Aug 25-27', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.04, 0.06, 'Aug 28-31', fontsize=8, fontweight='bold', rotation=90)
# cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=7, x=1.08, y=0.05, labelpad=-15)
plt.savefig(path_results + '/sm_comp_nur_1_ori.png')
plt.close()


# Shorter version
fig = plt.figure(figsize=(9, 5), facecolor='w', edgecolor='k', dpi=200)
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.95, hspace=0.0, wspace=0.0)
for irow in range(3):
    for icol in range(4):
        # 1 km
        ax = fig.add_subplot(3, 4, irow*4+icol+1, projection=ccrs.PlateCarree())
        ax.add_feature(feature_shp_nur)
        img = ax.imshow(sm_masked_ds_nur_stack[icol][irow+5, :, :], origin='upper', vmin=0, vmax=0.6, cmap='turbo_r',
                   extent=extent_nur)
        gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
        gl.xlocator = mticker.MultipleLocator(base=1)
        gl.ylocator = mticker.MultipleLocator(base=0.5)
        gl.xlabel_style = {'size': 6}
        gl.ylabel_style = {'size': 6}
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        if icol == 0:
            gl.left_labels = True
            gl.bottom_labels = False
        elif irow == 2:
            gl.left_labels = False
            gl.bottom_labels = True
        else:
            gl.left_labels = False
            gl.bottom_labels = False
        if icol == 0 and irow == 2:
            gl.left_labels = True
            gl.bottom_labels = True
        gl.right_labels = False
        gl.top_labels = False
cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both')
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=9, x=0.95)
cbar.ax.tick_params(labelsize=9)
cbar_ax.locator_params(nbins=6)
fig.text(0.15, 0.94, 'SMOS 1km', fontsize=10, fontweight='bold')
fig.text(0.35, 0.94, 'SMOS 25km', fontsize=10, fontweight='bold')
fig.text(0.55, 0.94, 'SMAP 1km', fontsize=10, fontweight='bold')
fig.text(0.75, 0.94, 'SMAP 9km', fontsize=10, fontweight='bold')
fig.text(0.03, 0.73, 'Aug 16-18', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.03, 0.41, 'Aug 19-21', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.03, 0.12, 'Aug 22-24', fontsize=8, fontweight='bold', rotation=90)
# cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=7, x=1.08, y=0.05, labelpad=-15)
plt.savefig(path_results + '/sm_comp_nur_1_new.png')
plt.close()



# 6.2.4 Assiniboine-Red River basin

path_shp_arr = path_gis_data + '/watershed_boundary/'
os.chdir(path_shp_arr)
shp_arr_file = "assiniboine_red_riverbasin.shp"
shp_arr_ds = ogr.GetDriverByName("ESRI Shapefile").Open(shp_arr_file, 0)
shp_arr_extent = list(shp_arr_ds.GetLayer().GetExtent())

# # 6.2.1.1 SMAP
#Load and subset the region of Middle Colorado RB (SMAP 9 km)
[lat_9km_arr, row_arr_9km_ind, lon_9km_arr, col_arr_9km_ind] = \
    coordtable_subset(lat_world_ease_9km, lon_world_ease_9km,
                      shp_arr_extent[3], shp_arr_extent[2], shp_arr_extent[1], shp_arr_extent[0])
[lat_1km_arr, row_arr_1km_ind, lon_1km_arr, col_arr_1km_ind] = \
    coordtable_subset(lat_world_ease_1km, lon_world_ease_1km,
                      shp_arr_extent[3], shp_arr_extent[2], shp_arr_extent[1], shp_arr_extent[0])


# Read the map data
f_read = h5py.File(path_model_evaluation + "/smap_arr_sm.hdf5", "r")
varname_read_list = ['smap_9km_data_stack_arr', 'smap_1km_data_stack_arr']
for x in range(len(varname_read_list)):
    var_obj = f_read[varname_read_list[x]][()]
    exec(varname_read_list[x] + '= var_obj')
    del(var_obj)
f_read.close()

f_read = h5py.File(path_model_evaluation + "/smos_arr_sm.hdf5", "r")
varname_read_list = ['smos_25km_data_stack_arr', 'smos_1km_data_stack_arr']
for x in range(len(varname_read_list)):
    var_obj = f_read[varname_read_list[x]][()]
    exec(varname_read_list[x] + '= var_obj')
    del(var_obj)
f_read.close()

# 6.2.2 Subplot maps
output_crs = 'EPSG:4326'
shapefile_arr = fiona.open(path_shp_arr + '/' + shp_arr_file, 'r')
crop_shape_arr = [feature["geometry"] for feature in shapefile_arr]

# 6.2.2.1 Subset and reproject the SMAP SM data at watershed
# 1 km
smap_masked_ds_arr_1km_all = []
for n in range(smap_1km_data_stack_arr.shape[0]):
    sub_window_arr_1km = Window(col_arr_1km_ind[0], row_arr_1km_ind[0], len(col_arr_1km_ind), len(row_arr_1km_ind))
    kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                      'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
    smap_sm_arr_1km_output = sub_n_reproj(smap_1km_data_stack_arr[n, :, :], kwargs_1km_sub, sub_window_arr_1km, output_crs)

    masked_ds_arr_1km, mask_transform_ds_arr_1km = mask(dataset=smap_sm_arr_1km_output, shapes=crop_shape_arr, crop=True)
    masked_ds_arr_1km[np.where(masked_ds_arr_1km == 0)] = np.nan
    masked_ds_arr_1km = masked_ds_arr_1km.squeeze()

    smap_masked_ds_arr_1km_all.append(masked_ds_arr_1km)

smap_masked_ds_arr_1km_all = np.asarray(smap_masked_ds_arr_1km_all)


# 9 km
smap_masked_ds_arr_9km_all = []
for n in range(smap_9km_data_stack_arr.shape[0]):
    sub_window_arr_9km = Window(col_arr_1km_ind[0], row_arr_1km_ind[0], len(col_arr_1km_ind), len(row_arr_1km_ind))
    kwargs_9km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                      'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
    smap_sm_arr_9km_output = sub_n_reproj(smap_9km_data_stack_arr[n, :, :], kwargs_9km_sub, sub_window_arr_9km, output_crs)

    masked_ds_arr_9km, mask_transform_ds_arr_9km = mask(dataset=smap_sm_arr_9km_output, shapes=crop_shape_arr, crop=True)
    masked_ds_arr_9km[np.where(masked_ds_arr_9km == 0)] = np.nan
    masked_ds_arr_9km = masked_ds_arr_9km.squeeze()

    smap_masked_ds_arr_9km_all.append(masked_ds_arr_9km)

smap_masked_ds_arr_9km_all = np.asarray(smap_masked_ds_arr_9km_all)
# masked_ds_arr_9km_all[masked_ds_arr_9km_all >= 0.5] = np.nan

# 6.2.2.2 Subset and reproject the SMOS SM data at watershed
# 1 km
smos_masked_ds_arr_1km_all = []
for n in range(smos_1km_data_stack_arr.shape[0]):
    sub_window_arr_1km = Window(col_arr_1km_ind[0], row_arr_1km_ind[0], len(col_arr_1km_ind), len(row_arr_1km_ind))
    kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                      'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
    smos_sm_arr_1km_output = sub_n_reproj(smos_1km_data_stack_arr[n, :, :], kwargs_1km_sub, sub_window_arr_1km, output_crs)

    masked_ds_arr_1km, mask_transform_ds_arr_1km = mask(dataset=smos_sm_arr_1km_output, shapes=crop_shape_arr, crop=True)
    masked_ds_arr_1km[np.where(masked_ds_arr_1km == 0)] = np.nan
    masked_ds_arr_1km = masked_ds_arr_1km.squeeze()

    smos_masked_ds_arr_1km_all.append(masked_ds_arr_1km)

smos_masked_ds_arr_1km_all = np.asarray(smos_masked_ds_arr_1km_all)


# 25 km
smos_masked_ds_arr_25km_all = []
for n in range(smos_25km_data_stack_arr.shape[0]):
    sub_window_arr_25km = Window(col_arr_1km_ind[0], row_arr_1km_ind[0], len(col_arr_1km_ind), len(row_arr_1km_ind))
    kwargs_25km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                      'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
    smos_sm_arr_25km_output = sub_n_reproj(smos_25km_data_stack_arr[n, :, :], kwargs_25km_sub, sub_window_arr_25km, output_crs)

    masked_ds_arr_25km, mask_transform_ds_arr_25km = mask(dataset=smos_sm_arr_25km_output, shapes=crop_shape_arr, crop=True)
    masked_ds_arr_25km[np.where(masked_ds_arr_25km == 0)] = np.nan
    masked_ds_arr_25km = masked_ds_arr_25km.squeeze()

    smos_masked_ds_arr_25km_all.append(masked_ds_arr_25km)

smos_masked_ds_arr_25km_all = np.asarray(smos_masked_ds_arr_25km_all)
# masked_ds_arr_25km_all[masked_ds_arr_25km_all >= 0.5] = np.nan

# Calculate the 3-day averaged maps
smap_1km_size = smap_masked_ds_arr_1km_all.shape
smap_masked_ds_arr_1km_avg = np.reshape(smap_masked_ds_arr_1km_all[:30, :, :], (3, 10, smap_1km_size[1], smap_1km_size[2]))
smap_masked_ds_arr_1km_avg = np.nanmean(smap_masked_ds_arr_1km_avg, axis=0)
smap_masked_ds_arr_9km_avg = np.reshape(smap_masked_ds_arr_9km_all[:30, :, :], (3, 10, smap_1km_size[1], smap_1km_size[2]))
smap_masked_ds_arr_9km_avg = np.nanmean(smap_masked_ds_arr_9km_avg, axis=0)
smos_masked_ds_arr_1km_avg = np.reshape(smos_masked_ds_arr_1km_all[:30, :, :], (3, 10, smap_1km_size[1], smap_1km_size[2]))
smos_masked_ds_arr_1km_avg = np.nanmean(smos_masked_ds_arr_1km_avg, axis=0)
smos_masked_ds_arr_25km_avg = np.reshape(smos_masked_ds_arr_25km_all[:30, :, :], (3, 10, smap_1km_size[1], smap_1km_size[2]))
smos_masked_ds_arr_25km_avg = np.nanmean(smos_masked_ds_arr_25km_avg, axis=0)
sm_masked_ds_arr_stack = list((smos_masked_ds_arr_1km_avg, smos_masked_ds_arr_25km_avg,
                               smap_masked_ds_arr_1km_avg, smap_masked_ds_arr_9km_avg))

# 6.2.2.3 Make the subplot maps
feature_shp_arr = ShapelyFeature(Reader(path_shp_arr + '/' + shp_arr_file).geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
extent_arr = np.array(shapefile_arr.bounds)
extent_arr = extent_arr[[0, 2, 1, 3]]

fig = plt.figure(figsize=(9, 10), facecolor='w', edgecolor='k', dpi=150)
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.95, hspace=0.0, wspace=0.0)
for irow in range(10):
    for icol in range(4):
        # 1 km
        ax = fig.add_subplot(10, 4, irow*4+icol+1, projection=ccrs.PlateCarree())
        ax.add_feature(feature_shp_arr)
        img = ax.imshow(sm_masked_ds_arr_stack[icol][irow, :, :], origin='upper', vmin=0, vmax=0.4, cmap='turbo_r',
                   extent=extent_arr)
        gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
        gl.xlocator = mticker.MultipleLocator(base=1)
        gl.ylocator = mticker.MultipleLocator(base=0.5)
        gl.xlabel_style = {'size': 6}
        gl.ylabel_style = {'size': 6}
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        if icol == 0:
            gl.left_labels = True
            gl.bottom_labels = False
        elif irow == 9:
            gl.left_labels = False
            gl.bottom_labels = True
        else:
            gl.left_labels = False
            gl.bottom_labels = False
        if icol == 0 and irow == 9:
            gl.left_labels = True
            gl.bottom_labels = True
        gl.right_labels = False
        gl.top_labels = False
cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both')
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=9, x=0.95)
cbar.ax.tick_params(labelsize=9)
cbar_ax.locator_params(nbins=6)
fig.text(0.15, 0.98, 'SMOS 1km', fontsize=10, fontweight='bold')
fig.text(0.35, 0.98, 'SMOS 25km', fontsize=10, fontweight='bold')
fig.text(0.55, 0.98, 'SMAP 1km', fontsize=10, fontweight='bold')
fig.text(0.75, 0.98, 'SMAP 9km', fontsize=10, fontweight='bold')
fig.text(0.04, 0.87, 'Aug 1-3', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.04, 0.78, 'Aug 4-6', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.04, 0.69, 'Aug 7-9', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.04, 0.60, 'Aug 10-12', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.04, 0.51, 'Aug 13-15', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.04, 0.42, 'Aug 16-18', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.04, 0.33, 'Aug 19-21', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.04, 0.24, 'Aug 22-24', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.04, 0.15, 'Aug 25-27', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.04, 0.06, 'Aug 28-31', fontsize=8, fontweight='bold', rotation=90)
# cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=7, x=1.08, y=0.05, labelpad=-15)
plt.savefig(path_results + '/sm_comp_arr_1.png')
plt.close()


# Shorter version
fig = plt.figure(figsize=(9, 3), facecolor='w', edgecolor='k', dpi=200)
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.95, hspace=0.0, wspace=0.0)
for irow in range(3):
    for icol in range(4):
        # 1 km
        ax = fig.add_subplot(3, 4, irow*4+icol+1, projection=ccrs.PlateCarree())
        ax.add_feature(feature_shp_arr)
        img = ax.imshow(sm_masked_ds_arr_stack[icol][irow+2, :, :], origin='upper', vmin=0, vmax=0.4, cmap='turbo_r',
                   extent=extent_arr)
        gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
        gl.xlocator = mticker.MultipleLocator(base=1)
        gl.ylocator = mticker.MultipleLocator(base=0.5)
        gl.xlabel_style = {'size': 6}
        gl.ylabel_style = {'size': 6}
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        if icol == 0:
            gl.left_labels = True
            gl.bottom_labels = False
        elif irow == 2:
            gl.left_labels = False
            gl.bottom_labels = True
        else:
            gl.left_labels = False
            gl.bottom_labels = False
        if icol == 0 and irow == 2:
            gl.left_labels = True
            gl.bottom_labels = True
        gl.right_labels = False
        gl.top_labels = False
cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both')
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=9, x=0.95)
cbar.ax.tick_params(labelsize=9)
cbar_ax.locator_params(nbins=6)
fig.text(0.15, 0.94, 'SMOS 1km', fontsize=10, fontweight='bold')
fig.text(0.35, 0.94, 'SMOS 25km', fontsize=10, fontweight='bold')
fig.text(0.55, 0.94, 'SMAP 1km', fontsize=10, fontweight='bold')
fig.text(0.75, 0.94, 'SMAP 9km', fontsize=10, fontweight='bold')
fig.text(0.03, 0.73, 'Aug 7-9', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.03, 0.41, 'Aug 10-12', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.03, 0.12, 'Aug 13-15', fontsize=8, fontweight='bold', rotation=90)
# cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=7, x=1.08, y=0.05, labelpad=-15)
plt.savefig(path_results + '/sm_comp_arr_1_new.png')
plt.close()




# 6.2.5 Skjern River basin

path_shp_skj = path_gis_data + '/wrd_riverbasins/'
# os.chdir(path_shp_skj)
shp_skj_file = path_shp_skj + "/Aqueduct_river_basins_SKJERN A/Aqueduct_river_basins_SKJERN A.shp"
shp_skj_ds = ogr.GetDriverByName("ESRI Shapefile").Open(shp_skj_file, 0)
shp_skj_extent = list(shp_skj_ds.GetLayer().GetExtent())

# # 6.2.1.1 SMAP
#Load and subset the region of Middle Colorado RB (SMAP 9 km)
[lat_9km_skj, row_skj_9km_ind, lon_9km_skj, col_skj_9km_ind] = \
    coordtable_subset(lat_world_ease_9km, lon_world_ease_9km,
                      shp_skj_extent[3], shp_skj_extent[2], shp_skj_extent[1], shp_skj_extent[0])
[lat_1km_skj, row_skj_1km_ind, lon_1km_skj, col_skj_1km_ind] = \
    coordtable_subset(lat_world_ease_1km, lon_world_ease_1km,
                      shp_skj_extent[3], shp_skj_extent[2], shp_skj_extent[1], shp_skj_extent[0])


# Read the map data
f_read = h5py.File(path_model_evaluation + "/smap_skj_sm.hdf5", "r")
varname_read_list = ['smap_9km_data_stack_skj', 'smap_1km_data_stack_skj']
for x in range(len(varname_read_list)):
    var_obj = f_read[varname_read_list[x]][()]
    exec(varname_read_list[x] + '= var_obj')
    del(var_obj)
f_read.close()

f_read = h5py.File(path_model_evaluation + "/smos_skj_sm.hdf5", "r")
varname_read_list = ['smos_25km_data_stack_skj', 'smos_1km_data_stack_skj']
for x in range(len(varname_read_list)):
    var_obj = f_read[varname_read_list[x]][()]
    exec(varname_read_list[x] + '= var_obj')
    del(var_obj)
f_read.close()

# 6.2.2 Subplot maps
output_crs = 'EPSG:4326'
shapefile_skj = fiona.open(shp_skj_file, 'r')
crop_shape_skj = [feature["geometry"] for feature in shapefile_skj]

# 6.2.2.1 Subset and reproject the SMAP SM data at watershed
# 1 km
smap_masked_ds_skj_1km_all = []
for n in range(smap_1km_data_stack_skj.shape[0]):
    sub_window_skj_1km = Window(col_skj_1km_ind[0], row_skj_1km_ind[0], len(col_skj_1km_ind), len(row_skj_1km_ind))
    kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                      'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
    smap_sm_skj_1km_output = sub_n_reproj(smap_1km_data_stack_skj[n, :, :], kwargs_1km_sub, sub_window_skj_1km, output_crs)

    masked_ds_skj_1km, mask_transform_ds_skj_1km = mask(dataset=smap_sm_skj_1km_output, shapes=crop_shape_skj, crop=True)
    masked_ds_skj_1km[np.where(masked_ds_skj_1km == 0)] = np.nan
    masked_ds_skj_1km = masked_ds_skj_1km.squeeze()

    smap_masked_ds_skj_1km_all.append(masked_ds_skj_1km)

smap_masked_ds_skj_1km_all = np.asarray(smap_masked_ds_skj_1km_all)


# 9 km
smap_masked_ds_skj_9km_all = []
for n in range(smap_9km_data_stack_skj.shape[0]):
    sub_window_skj_9km = Window(col_skj_1km_ind[0], row_skj_1km_ind[0], len(col_skj_1km_ind), len(row_skj_1km_ind))
    kwargs_9km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                      'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
    smap_sm_skj_9km_output = sub_n_reproj(smap_9km_data_stack_skj[n, :, :], kwargs_9km_sub, sub_window_skj_9km, output_crs)

    masked_ds_skj_9km, mask_transform_ds_skj_9km = mask(dataset=smap_sm_skj_9km_output, shapes=crop_shape_skj, crop=True)
    masked_ds_skj_9km[np.where(masked_ds_skj_9km == 0)] = np.nan
    masked_ds_skj_9km = masked_ds_skj_9km.squeeze()

    smap_masked_ds_skj_9km_all.append(masked_ds_skj_9km)

smap_masked_ds_skj_9km_all = np.asarray(smap_masked_ds_skj_9km_all)
# masked_ds_skj_9km_all[masked_ds_skj_9km_all >= 0.5] = np.nan

# 6.2.2.2 Subset and reproject the SMOS SM data at watershed
# 1 km
smos_masked_ds_skj_1km_all = []
for n in range(smos_1km_data_stack_skj.shape[0]):
    sub_window_skj_1km = Window(col_skj_1km_ind[0], row_skj_1km_ind[0], len(col_skj_1km_ind), len(row_skj_1km_ind))
    kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                      'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
    smos_sm_skj_1km_output = sub_n_reproj(smos_1km_data_stack_skj[n, :, :], kwargs_1km_sub, sub_window_skj_1km, output_crs)

    masked_ds_skj_1km, mask_transform_ds_skj_1km = mask(dataset=smos_sm_skj_1km_output, shapes=crop_shape_skj, crop=True)
    masked_ds_skj_1km[np.where(masked_ds_skj_1km == 0)] = np.nan
    masked_ds_skj_1km = masked_ds_skj_1km.squeeze()

    smos_masked_ds_skj_1km_all.append(masked_ds_skj_1km)

smos_masked_ds_skj_1km_all = np.asarray(smos_masked_ds_skj_1km_all)


# 25 km
smos_masked_ds_skj_25km_all = []
for n in range(smos_25km_data_stack_skj.shape[0]):
    sub_window_skj_25km = Window(col_skj_1km_ind[0], row_skj_1km_ind[0], len(col_skj_1km_ind), len(row_skj_1km_ind))
    kwargs_25km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                      'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
    smos_sm_skj_25km_output = sub_n_reproj(smos_25km_data_stack_skj[n, :, :], kwargs_25km_sub, sub_window_skj_25km, output_crs)

    masked_ds_skj_25km, mask_transform_ds_skj_25km = mask(dataset=smos_sm_skj_25km_output, shapes=crop_shape_skj, crop=True)
    masked_ds_skj_25km[np.where(masked_ds_skj_25km == 0)] = np.nan
    masked_ds_skj_25km = masked_ds_skj_25km.squeeze()

    smos_masked_ds_skj_25km_all.append(masked_ds_skj_25km)

smos_masked_ds_skj_25km_all = np.asarray(smos_masked_ds_skj_25km_all)
# masked_ds_skj_25km_all[masked_ds_skj_25km_all >= 0.5] = np.nan

# Calculate the 3-day averaged maps
smap_1km_size = smap_masked_ds_skj_1km_all.shape
smap_masked_ds_skj_1km_avg = np.reshape(smap_masked_ds_skj_1km_all[:30, :, :], (3, 10, smap_1km_size[1], smap_1km_size[2]))
smap_masked_ds_skj_1km_avg = np.nanmean(smap_masked_ds_skj_1km_avg, axis=0)
smap_masked_ds_skj_9km_avg = np.reshape(smap_masked_ds_skj_9km_all[:30, :, :], (3, 10, smap_1km_size[1], smap_1km_size[2]))
smap_masked_ds_skj_9km_avg = np.nanmean(smap_masked_ds_skj_9km_avg, axis=0)
smos_masked_ds_skj_1km_avg = np.reshape(smos_masked_ds_skj_1km_all[:30, :, :], (3, 10, smap_1km_size[1], smap_1km_size[2]))
smos_masked_ds_skj_1km_avg = np.nanmean(smos_masked_ds_skj_1km_avg, axis=0)
smos_masked_ds_skj_25km_avg = np.reshape(smos_masked_ds_skj_25km_all[:30, :, :], (3, 10, smap_1km_size[1], smap_1km_size[2]))
smos_masked_ds_skj_25km_avg = np.nanmean(smos_masked_ds_skj_25km_avg, axis=0)
sm_masked_ds_skj_stack = list((smos_masked_ds_skj_1km_avg, smos_masked_ds_skj_25km_avg,
                               smap_masked_ds_skj_1km_avg, smap_masked_ds_skj_9km_avg))

# 6.2.2.3 Make the subplot maps
feature_shp_skj = ShapelyFeature(Reader( shp_skj_file).geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
extent_skj = np.array(shapefile_skj.bounds)
extent_skj = extent_skj[[0, 2, 1, 3]]

fig = plt.figure(figsize=(9, 10), facecolor='w', edgecolor='k', dpi=150)
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.95, hspace=0.0, wspace=0.0)
for irow in range(10):
    for icol in range(4):
        # 1 km
        ax = fig.add_subplot(10, 4, irow*4+icol+1, projection=ccrs.PlateCarree())
        ax.add_feature(feature_shp_skj)
        img = ax.imshow(sm_masked_ds_skj_stack[icol][irow, :, :], origin='upper', vmin=0, vmax=0.4, cmap='turbo_r',
                   extent=extent_skj)
        gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
        gl.xlocator = mticker.MultipleLocator(base=1)
        gl.ylocator = mticker.MultipleLocator(base=0.5)
        gl.xlabel_style = {'size': 6}
        gl.ylabel_style = {'size': 6}
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        if icol == 0:
            gl.left_labels = True
            gl.bottom_labels = False
        elif irow == 9:
            gl.left_labels = False
            gl.bottom_labels = True
        else:
            gl.left_labels = False
            gl.bottom_labels = False
        if icol == 0 and irow == 9:
            gl.left_labels = True
            gl.bottom_labels = True
        gl.right_labels = False
        gl.top_labels = False
cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both')
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=9, x=0.95)
cbar.ax.tick_params(labelsize=9)
cbar_ax.locator_params(nbins=6)
fig.text(0.15, 0.98, 'SMOS 1km', fontsize=10, fontweight='bold')
fig.text(0.35, 0.98, 'SMOS 25km', fontsize=10, fontweight='bold')
fig.text(0.55, 0.98, 'SMAP 1km', fontsize=10, fontweight='bold')
fig.text(0.75, 0.98, 'SMAP 9km', fontsize=10, fontweight='bold')
fig.text(0.04, 0.87, 'Aug 1-3', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.04, 0.78, 'Aug 4-6', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.04, 0.69, 'Aug 7-9', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.04, 0.60, 'Aug 10-12', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.04, 0.51, 'Aug 13-15', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.04, 0.42, 'Aug 16-18', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.04, 0.33, 'Aug 19-21', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.04, 0.24, 'Aug 22-24', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.04, 0.15, 'Aug 25-27', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.04, 0.06, 'Aug 28-31', fontsize=8, fontweight='bold', rotation=90)
# cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=7, x=1.08, y=0.05, labelpad=-15)
plt.savefig(path_results + '/sm_comp_skj_1.png')
plt.close()


# Shorter version
fig = plt.figure(figsize=(9, 3), facecolor='w', edgecolor='k', dpi=200)
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.95, hspace=0.0, wspace=0.0)
for irow in range(3):
    for icol in range(4):
        # 1 km
        ax = fig.add_subplot(3, 4, irow*4+icol+1, projection=ccrs.PlateCarree())
        ax.add_feature(feature_shp_skj)
        img = ax.imshow(sm_masked_ds_skj_stack[icol][irow+2, :, :], origin='upper', vmin=0, vmax=0.3, cmap='turbo_r',
                   extent=extent_skj)
        gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
        gl.xlocator = mticker.MultipleLocator(base=0.5)
        gl.ylocator = mticker.MultipleLocator(base=0.5)
        gl.xlabel_style = {'size': 6}
        gl.ylabel_style = {'size': 6}
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        if icol == 0:
            gl.left_labels = True
            gl.bottom_labels = False
        elif irow == 2:
            gl.left_labels = False
            gl.bottom_labels = True
        else:
            gl.left_labels = False
            gl.bottom_labels = False
        if icol == 0 and irow == 2:
            gl.left_labels = True
            gl.bottom_labels = True
        gl.right_labels = False
        gl.top_labels = False
cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both')
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=9, x=0.93)
cbar.ax.tick_params(labelsize=9)
cbar_ax.locator_params(nbins=5)
fig.text(0.15, 0.94, 'SMOS 1km', fontsize=10, fontweight='bold')
fig.text(0.35, 0.94, 'SMOS 25km', fontsize=10, fontweight='bold')
fig.text(0.55, 0.94, 'SMAP 1km', fontsize=10, fontweight='bold')
fig.text(0.75, 0.94, 'SMAP 9km', fontsize=10, fontweight='bold')
fig.text(0.03, 0.73, 'Aug 7-9', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.03, 0.41, 'Aug 10-12', fontsize=8, fontweight='bold', rotation=90)
fig.text(0.03, 0.12, 'Aug 13-15', fontsize=8, fontweight='bold', rotation=90)
# cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=7, x=1.08, y=0.05, labelpad=-15)
plt.savefig(path_results + '/sm_comp_skj_1_new.png')
plt.close()




# Eastern MD
# import cartopy.io.img_tiles as cimgt
# tiler = cimgt.OSM()
path_results = '/Users/binfang/Downloads'
path_shp = '/Users/binfang/Downloads/Processing/shapefiles'
smap_tif_files = sorted(glob.glob(path_output + '/california/smap_1km/2025/*.tif'))

smap_sm_all = []
for idt in range(len(smap_tif_files)):
    src_tf = rasterio.open(smap_tif_files[idt]).read()
    src_tf = np.nanmean(src_tf, axis=0)
    smap_sm_all.append(src_tf)
    del(src_tf)
# smap_sm_all = smap_sm_all[31:76]
smap_sm_all = smap_sm_all[0:45]

divide_ind = np.arange(0, len(smap_sm_all), 3)
smap_sm_all_divide = np.split(smap_sm_all, divide_ind[1:], axis=0)

smap_sm_avg_all = []
for idt in range(len(smap_sm_all_divide)):
    smap_sm_all_avg = np.nanmean(np.stack(smap_sm_all_divide[idt], axis=0), axis=0)
    smap_sm_avg_all.append(smap_sm_all_avg)
    del(smap_sm_all_avg)

start_date = datetime.datetime(2025, 1, 1)
# Generate titles for 15 subplots with a 3-day interval
titles = [f"{(start_date + datetime.timedelta(days=3*i)).strftime('%b %d')} - "
          f"{(start_date + datetime.timedelta(days=3*i+2)).strftime('%b %d')}"
          for i in range(15)]

# Make subplot maps (Palisades)
# latlon_extent = [-77, -75, 37.9, 39.7]
latlon_extent = [-118.88, -117.8, 33.82, 34.3] # Palisades, CA
shape_conus = ShapelyFeature(Reader(path_gis_data + '/cb_2015_us_state_500k/cb_2015_us_state_500k.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')

fig = plt.figure(figsize=(14, 2.7), facecolor='w', edgecolor='k', dpi=200)
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95, hspace=0.2, wspace=0.2)
for idt in range(len(smap_sm_avg_all)):

    # title = os.path.basename(smap_tif_files[idt]).split('_')[-1][:8]
    # SMAP SM
    ax = fig.add_subplot(3, 5, idt+1, projection=ccrs.PlateCarree())
    ax.add_feature(shape_conus, linewidth=0.5)
    # ax.add_image(tiler, 10, alpha=0.3)
    img = ax.imshow(smap_sm_avg_all[idt], origin='upper', vmin=0, vmax=0.3, cmap='Spectral', extent=latlon_extent)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=0.3)
    gl.ylocator = mticker.MultipleLocator(base=0.3)
    gl.xlabel_style = {'size': 5}
    gl.ylabel_style = {'size': 5}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    if idt % 5 == 0:  # First column (0, 5, 10, etc.)
        gl.left_labels = True
    else:
        gl.left_labels = False

    if idt >= 10:  # Last row (indices 10, 11, 12, 13, 14)
        gl.bottom_labels = True
    else:
        gl.bottom_labels = False
    # Disable right and top labels for all subplots
    gl.right_labels = False
    gl.top_labels = False
    # ax.text(-76.7, 39.95, titles[idt], fontsize=6, horizontalalignment='left', verticalalignment='top',
    #         color='black')
    ax.text(-118.6, 34.4, titles[idt], fontsize=6, horizontalalignment='left', verticalalignment='top',
            color='black')
    # ax.legend(['N/A', 'Irrigated', 'Non-irrigated'], loc='right', bbox_to_anchor=(1, 1))
    # fig.text(0.47, 0.95, title + ' - ' + str(int(title)+7), ha='center', fontsize=14, fontweight='bold')
    # fig.text(0.05, 0.73, 'SMAP SM', rotation='vertical', fontsize=14, fontweight='bold')
cbar_ax = fig.add_axes([0.35, 0.07, 0.3, 0.01])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both', pad=0, orientation='horizontal')
cbar.ax.tick_params(labelsize=5)
cbar_ax.locator_params(nbins=5)
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=5, x=1.05, y=0.05, labelpad=-15)
plt.savefig(path_results + '/sm_palisades.png')
plt.close()


# Make subplot maps (California)
output_crs = 'EPSG:4326'
shapefile = fiona.open(path_shp + '/california/California_State_Boundary/California_State_Boundary.shp', 'r')
crop_shape = [feature["geometry"] for feature in shapefile]
# 1 km
smap_masked_ds_1km_all = []
for n in range(len(smap_sm_avg_all)):
    sub_window_1km = Window(col_sub_ease_1km_ind[0], row_sub_ease_1km_ind[0], len(col_sub_ease_1km_ind), len(row_sub_ease_1km_ind))
    kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                      'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
    smap_sm_1km_output = sub_n_reproj(smap_sm_avg_all[n], kwargs_1km_sub, sub_window_1km, output_crs)

    masked_ds_1km, mask_transform_ds_1km = mask(dataset=smap_sm_1km_output, shapes=crop_shape, crop=True)
    masked_ds_1km[np.where(masked_ds_1km == 0)] = np.nan
    masked_ds_1km = masked_ds_1km.squeeze()

    smap_masked_ds_1km_all.append(masked_ds_1km)

smap_masked_ds_1km_all = np.asarray(smap_masked_ds_1km_all)

# latlon_extent = [-77, -75, 37.9, 39.7]
latlon_extent = [-124.5, -114.1, 32.5, 42]  # Palisades, CA
shape_conus = ShapelyFeature(Reader(path_gis_data + '/cb_2015_us_state_500k/cb_2015_us_state_500k.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')

fig = plt.figure(figsize=(12, 4), facecolor='w', edgecolor='k', dpi=200)
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95, hspace=0.2, wspace=0.2)
for idt in range(len(smap_sm_avg_all)):

    # title = os.path.basename(smap_tif_files[idt]).split('_')[-1][:8]
    # SMAP SM
    ax = fig.add_subplot(3, 5, idt+1, projection=ccrs.PlateCarree())
    ax.add_feature(shape_conus, linewidth=0.5)
    # ax.add_image(tiler, 10, alpha=0.3)
    img = ax.imshow(smap_masked_ds_1km_all[idt, :, :], origin='upper', vmin=0, vmax=0.5, cmap='Spectral', extent=latlon_extent)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=4)
    gl.ylocator = mticker.MultipleLocator(base=4)
    gl.xlabel_style = {'size': 5}
    gl.ylabel_style = {'size': 5}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    if idt % 5 == 0:  # First column (0, 5, 10, etc.)
        gl.left_labels = True
    else:
        gl.left_labels = False

    if idt >= 10:  # Last row (indices 10, 11, 12, 13, 14)
        gl.bottom_labels = True
    else:
        gl.bottom_labels = False
    # Disable right and top labels for all subplots
    gl.right_labels = False
    gl.top_labels = False
    # ax.text(-76.7, 39.95, titles[idt], fontsize=6, horizontalalignment='left', verticalalignment='top',
    #         color='black')
    ax.text(-122, 43, titles[idt], fontsize=6, horizontalalignment='left', verticalalignment='top',
            color='black')
    # ax.legend(['N/A', 'Irrigated', 'Non-irrigated'], loc='right', bbox_to_anchor=(1, 1))
    # fig.text(0.47, 0.95, title + ' - ' + str(int(title)+7), ha='center', fontsize=14, fontweight='bold')
    # fig.text(0.05, 0.73, 'SMAP SM', rotation='vertical', fontsize=14, fontweight='bold')
cbar_ax = fig.add_axes([0.35, 0.05, 0.3, 0.01])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both', pad=0, orientation='horizontal')
cbar.ax.tick_params(labelsize=5)
cbar_ax.locator_params(nbins=5)
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=5, x=1.05, y=0.05, labelpad=-15)
plt.savefig(path_results + '/sm_california.png')
plt.close()
