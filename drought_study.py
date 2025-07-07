import os
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import glob
import h5py
import gdal
import fiona
import rasterio
import calendar
import datetime
import osr
import itertools
import pandas as pd
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
from itertools import chain
from sklearn import preprocessing

# Ignore runtime warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


#########################################################################################
# (Function 1) Subset the coordinates table of desired area

def coordtable_subset(lat_input, lon_input, lat_extent_max, lat_extent_min, lon_extent_max, lon_extent_min):
    lat_output = lat_input[np.where((lat_input <= lat_extent_max) & (lat_input >= lat_extent_min))]
    row_output_ind = np.squeeze(np.array(np.where((lat_input <= lat_extent_max) & (lat_input >= lat_extent_min))))
    lon_output = lon_input[np.where((lon_input <= lon_extent_max) & (lon_input >= lon_extent_min))]
    col_output_ind = np.squeeze(np.array(np.where((lon_input <= lon_extent_max) & (lon_input >= lon_extent_min))))

    return lat_output, row_output_ind, lon_output, col_output_ind

#########################################################################################
# (Function 2) Generate lat/lon tables of Geographic projection in world/CONUS

def geo_coord_gen(lat_geo_extent_max, lat_geo_extent_min, lon_geo_extent_max, lon_geo_extent_min, cellsize):
    lat_geo_output = np.linspace(lat_geo_extent_max - cellsize / 2, lat_geo_extent_min + cellsize / 2,
                                    num=int((lat_geo_extent_max - lat_geo_extent_min) / cellsize))
    lat_geo_output = np.round(lat_geo_output, decimals=3)
    lon_geo_output = np.linspace(lon_geo_extent_min + cellsize / 2, lon_geo_extent_max - cellsize / 2,
                                    num=int((lon_geo_extent_max - lon_geo_extent_min) / cellsize))
    lon_geo_output = np.round(lon_geo_output, decimals=3)

    return lat_geo_output, lon_geo_output

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
# Function 5. Subset and reproject the Geotiff data to WGS84 projection

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

########################################################################################################################

# 0. Input variables
# Specify file paths
# Path of current workspace
path_workspace = '/Users/binfang/Documents/SMAP_Project/smap_codes'
# Path of Land mask
path_lmask = '/Volumes/MyPassport/SMAP_Project/Datasets/Lmask'
# Path of processed data
path_procdata = '/Volumes/MyPassport/SMAP_Project/Datasets/processed_data'
# Path of downscaled SM
path_smap_sm_ds = '/Volumes/MyPassport/SMAP_Project/Datasets/SMAP/1km/gldas'
# Path of GIS data
path_gis_data = '/Users/binfang/Documents/SMAP_Project/data/gis_data'
# Path of Australia soil data
path_soil = '/Users/binfang/Downloads/soilgrids'
# Path of results
path_results = '/Users/binfang/Documents/SMAP_Project/results/results_200605'
# Path of preview
path_preview = '/Users/binfang/Documents/SMAP_Project/results/results_191202/preview'
# Path of swdi data
path_swdi = '/Volumes/MyPassport/SMAP_Project/Datasets/Australia/swdi'
# Path of model data
path_model = '/Volumes/MyPassport/SMAP_Project/Datasets/model_data'
# Path of processed data
path_processed = '/Volumes/MyPassport/SMAP_Project/Datasets/processed_data'
# Path of downscaled SM
path_smap_sm_ds = '/Volumes/MyPassport/SMAP_Project/Datasets/SMAP/1km/gldas'
# Path of processed data 2
path_processed_2 = '/Users/binfang/Downloads/Processing/processed_data'
# Path of GPM
path_gpm = '/Volumes/MyPassport/SMAP_Project/Datasets/GPM'
# Path of ISMN
path_ismn = '/Volumes/MyPassport/SMAP_Project/Datasets/ISMN/Ver_1/processed_data'
# Path of GLDAS
path_gldas = '/Volumes/MyPassport/SMAP_Project/Datasets/GLDAS'

lst_folder = '/MYD11A1/'
ndvi_folder = '/MYD13A2/'
yearname = np.linspace(2015, 2019, 5, dtype='int')
monthnum = np.linspace(1, 12, 12, dtype='int')
monthname = np.arange(1, 13)
monthname = [str(i).zfill(2) for i in monthname]

# Load in geo-location parameters
os.chdir(path_workspace)
f = h5py.File("ds_parameters.hdf5", "r")
varname_list = ['lat_world_max', 'lat_world_min', 'lon_world_max', 'lon_world_min', 'cellsize_1km', 'cellsize_9km',
                'lat_world_ease_9km', 'lon_world_ease_9km', 'lat_world_ease_1km', 'lon_world_ease_1km',
                'lat_world_geo_10km', 'lon_world_geo_10km',
                'lat_world_ease_25km', 'lon_world_ease_25km', 'row_world_ease_9km_from_1km_ind', 'interdist_ease_1km',
                'col_world_ease_9km_from_1km_ind', 'size_world_ease_1km', 'row_conus_ease_1km_ind', 'col_conus_ease_1km_ind',
                'col_world_ease_1km_from_25km_ind', 'row_world_ease_1km_from_25km_ind']
for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f.close()

# Load in Australian variables
os.chdir(path_workspace)
f = h5py.File("aus_parameters.hdf5", "r")
varname_list = list(f.keys())

for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
f.close()

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


# # Generate land/water mask provided by GLDAS/NASA
# os.chdir(path_lmask)
# lmask_file = open('EASE2_M25km.LOCImask_land50_coast0km.1388x584.bin', 'r')
# lmask_ease_25km = np.fromfile(lmask_file, dtype=np.dtype('uint8'))
# lmask_ease_25km = np.reshape(lmask_ease_25km, [len(lat_world_ease_25km), len(lon_world_ease_25km)]).astype(float)
# lmask_ease_25km[np.where(lmask_ease_25km != 0)] = np.nan
# lmask_ease_25km[np.where(lmask_ease_25km == 0)] = 1
# # lmask_ease_25km[np.where((lmask_ease_25km == 101) | (lmask_ease_25km == 255))] = 0
# lmask_file.close()

# # Find the indices of land pixels by the 25-km resolution land-ocean mask
# [row_lmask_ease_25km_ind, col_lmask_ease_25km_ind] = np.where(~np.isnan(lmask_ease_25km))

# Convert the 1 km from 25 km match table files to 1-d linear
col_meshgrid_from_25km, row_meshgrid_from_25km = np.meshgrid(col_world_ease_1km_from_25km_ind, row_world_ease_1km_from_25km_ind)
col_meshgrid_from_25km = col_meshgrid_from_25km.reshape(1, -1)
row_meshgrid_from_25km = row_meshgrid_from_25km.reshape(1, -1)


########################################################################################################################
# 1. Extract the geographic information of soil data
path_toread = path_soil + '/clay_0-5cm_mean/'
list_subdir = [d for d in os.listdir(path_toread) if os.path.isdir(os.path.join(path_toread, d))]
list_subdir.sort()

tif_files_all = []
for idt in range(len(list_subdir)):
    tif_files = sorted(glob.glob(path_toread + list_subdir[idt] + '/*.tif'))
    tif_files_all.append(tif_files)
    del(tif_files)
tif_files_all = list(itertools.chain(*tif_files_all))

src_tf_all = []
for idt in range(1000):#range(len(tif_files_all)):
    src_tf = rasterio.open(tif_files_all[idt]).read()
    src_tf_all.append(src_tf)
    del(src_tf)




src_tf = gdal.Open(path_aus_soil + '/australia_soil_data/CLY_000_005_EV_N_P_AU_NAT_C_20140801.tif')
src_tf_arr = src_tf.ReadAsArray().astype(np.float32)

cellsize_aussoil = src_tf.GetGeoTransform()[1]
size_aus = src_tf_arr.shape
lat_aus_max = src_tf.GetGeoTransform()[3] - cellsize_aussoil/2
lon_aus_min = src_tf.GetGeoTransform()[0] - cellsize_aussoil/2
lat_aus_min = lat_aus_max - cellsize_aussoil*(size_aus[0]-1)
lon_aus_max = lon_aus_min + cellsize_aussoil*(size_aus[1]-1)

lat_aus_90m = np.linspace(lat_aus_max, lat_aus_min, size_aus[0])
lon_aus_90m = np.linspace(lon_aus_min, lon_aus_max, size_aus[1])

# Subset the Australian region
[lat_aus_ease_1km, row_aus_ease_1km_ind, lon_aus_ease_1km, col_aus_ease_1km_ind] = coordtable_subset\
    (lat_world_ease_1km, lon_world_ease_1km, lat_aus_max, lat_aus_min, lon_aus_max, lon_aus_min)
[lat_aus_ease_25km, row_aus_ease_25km_ind, lon_aus_ease_25km, col_aus_ease_25km_ind] = coordtable_subset\
    (lat_world_ease_25km, lon_world_ease_25km, lat_aus_max, lat_aus_min, lon_aus_max, lon_aus_min)
[lat_aus_ease_9km, row_aus_ease_9km_ind, lon_aus_ease_9km, col_aus_ease_9km_ind] = coordtable_subset\
    (lat_world_ease_9km, lon_world_ease_9km, lat_aus_max, lat_aus_min, lon_aus_max, lon_aus_min)

# Save variables
os.chdir(path_workspace)
var_name = ['col_aus_ease_1km_ind', 'row_aus_ease_1km_ind', 'col_aus_ease_9km_ind', 'row_aus_ease_9km_ind',
            'lat_aus_90m', 'lon_aus_90m', 'lat_aus_ease_1km', 'lon_aus_ease_1km', 'lat_aus_ease_9km', 'lon_aus_ease_9km',
            'lat_aus_ease_25km', 'row_aus_ease_25km_ind', 'lon_aus_ease_25km', 'col_aus_ease_25km_ind']

with h5py.File('aus_parameters.hdf5', 'w') as f:
    for x in var_name:
        f.create_dataset(x, data=eval(x))
f.close()


# Aggregate the Australian soil data from 90 m to 1 km
# Generate the aggregate table for 1 km from 90 m
[row_aus_ease_1km_from_90m_ind, col_aus_ease_1km_from_90m_ind] = \
    find_easeind_lofrhi(lat_aus_90m, lon_aus_90m, interdist_ease_1km,
                        size_world_ease_1km[0], size_world_ease_1km[1], row_aus_ease_1km_ind, col_aus_ease_1km_ind)

os.chdir(path_aus_soil + '/australia_soil_data')
aussoil_files = sorted(glob.glob('*.tif'))

aussoil_1day_init = np.empty([len(lat_aus_ease_1km), len(lon_aus_ease_1km)], dtype='float32')
aussoil_1day_init[:] = np.nan
aussoil_data = np.empty([len(lat_aus_ease_1km), len(lon_aus_ease_1km), len(aussoil_files)], dtype='float32')
aussoil_data[:] = np.nan

for idt in range(len(aussoil_files)):
    aussoil_1day = np.copy(aussoil_1day_init)
    src_tf = gdal.Open(aussoil_files[idt])
    src_tf_arr = src_tf.ReadAsArray().astype(np.float32)
    aussoil_1day = np.array\
        ([np.nanmean(src_tf_arr[row_aus_ease_1km_from_90m_ind[x], :], axis=0)
          for x in range(len(lat_aus_ease_1km))])
    aussoil_1day = np.array \
        ([np.nanmean(aussoil_1day[:, col_aus_ease_1km_from_90m_ind[y]], axis=1)
          for y in range(len(lon_aus_ease_1km))])
    aussoil_1day = np.fliplr(np.rot90(aussoil_1day, 3))
    aussoil_1day[np.where(aussoil_1day <= 0)] = np.nan
    aussoil_data[:, :, idt] = aussoil_1day
    del(aussoil_1day, src_tf, src_tf_arr)
    print(idt)

# Create a raster of EASE grid projection at 1 km resolution
out_ds_tiff = gdal.GetDriverByName('GTiff').Create(path_aus_soil + '/aussoil_data.tif',
     len(lon_aus_ease_1km), len(lat_aus_ease_1km), len(aussoil_files),  # Number of bands
     gdal.GDT_Float32, ['COMPRESS=LZW', 'TILED=YES'])
# out_ds_tiff.SetGeoTransform(dst_tran)
# out_ds_tiff.SetProjection(src_tf.GetProjection())

# Loop write each band to Geotiff file
for idl in range(len(aussoil_files)):
    out_ds_tiff.GetRasterBand(idl + 1).WriteArray(aussoil_data[:, :, idl])
    out_ds_tiff.GetRasterBand(idl + 1).SetNoDataValue(0)
    out_ds_tiff.GetRasterBand(idl + 1).SetDescription(aussoil_files[idl].split('_')[0])
out_ds_tiff = None  # close dataset to write to disc

del(aussoil_data, src_tf_arr)


########################################################################################################################
# 2. Calculate the Soil water deficit index (SWDI) required input parameters

# Load in Australian variables
os.chdir(path_workspace)
f = h5py.File("aus_parameters.hdf5", "r")
varname_list = list(f.keys())

for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
f.close()


aussoil_tf = gdal.Open(path_aus_soil + '/aussoil_data.tif')
aussoil_arr = aussoil_tf.ReadAsArray().astype(np.float32)

cly = aussoil_arr[0, :, :]/100  # Clay
snd = aussoil_arr[1, :, :]/100  # Sand
soc = aussoil_arr[2, :, :]/100  # Soil organic carbon

om = soc/0.58  # Convert from Soil organic carbon to soil organic matter


# 2.1 Calculate the parameters for calculating SWDI
theta_wp_fs = -0.024*snd + 0.487*cly + 0.006*om + 0.005*(snd*om) - 0.013*(cly*om) + 0.068*(snd*cly) + 0.031
theta_wp = theta_wp_fs + (0.14 * theta_wp_fs - 0.02)
theta_fc_fs = -0.251*snd + 0.195*cly + 0.011*om + 0.006*(snd*om) - 0.027*(cly*om) + 0.452*(snd*cly) + 0.299
theta_fc = theta_fc_fs + [1.283*(theta_fc_fs**2) - 0.374 * theta_fc_fs - 0.015]
theta_fc = np.squeeze(theta_fc)
theta_awc = theta_fc - theta_wp

del(aussoil_tf, aussoil_arr)

# Get geographic referencing information of Ausralia
smap_sm_aus_1km_data = gdal.Open('/Volumes/MyPassport/SMAP_Project/Datasets/SMAP/1km/gldas/2019/smap_sm_1km_ds_2019001.tif')
smap_sm_aus_1km_data = smap_sm_aus_1km_data.ReadAsArray().astype(np.float32)[0, :, :]
output_crs = 'EPSG:6933'
sub_window_aus_1km = Window(col_aus_ease_1km_ind[0], row_aus_ease_1km_ind[0], len(col_aus_ease_1km_ind), len(row_aus_ease_1km_ind))
kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                  'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                  'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956,
                                      7314540.79258289)}
smap_sm_aus_1km_output = sub_n_reproj(smap_sm_aus_1km_data, kwargs_1km_sub, sub_window_aus_1km, output_crs)
kwargs = smap_sm_aus_1km_output.meta.copy()


# Load in SMAP 1 km SM
for iyr in range(len(yearname)):
    os.chdir(path_smap_sm_ds + '/' + str(yearname[iyr]))
    tif_files = sorted(glob.glob('*.tif'))

    for idt in range(len(tif_files)):
        sm_tf = gdal.Open(tif_files[idt])
        sm_arr = sm_tf.ReadAsArray().astype(np.float32)
        sm_arr = sm_arr[:, row_aus_ease_1km_ind[0]:row_aus_ease_1km_ind[-1]+1, col_aus_ease_1km_ind[0]:col_aus_ease_1km_ind[-1]+1]
        sm_arr = np.nanmean(sm_arr, axis=0)
        swdi = (sm_arr - theta_fc) / theta_awc * 10
        swdi = np.expand_dims(swdi, axis=0)

        name = 'aus_swdi_' + os.path.splitext(tif_files[idt])[0].split('_')[-1]
        with rasterio.open(path_swdi + '/' + str(yearname[iyr]) + '/' + name + '.tif', 'w', **kwargs) as dst_file:
            dst_file.write(swdi)
        print(name)

        del(sm_tf, sm_arr, name, swdi)



# 2.2 Generate the seasonal SWDI maps

# Extract the days for each season
seasons_div_norm = np.array([0, 90, 181, 273, 365])
seasons_div_leap = np.array([0, 91, 182, 274, 366])

ind_season_all_years = []
for iyr in range(len(yearname)):
    # os.chdir(path_smap_sm_ds + '/' + str(yearname[iyr]))
    # tif_files = sorted(glob.glob('*.tif'))
    # names_daily = np.array([int(os.path.splitext(tif_files[idt])[0].split('_')[-1][-3:]) for idt in range(len(tif_files))])

    # Divide by seasons
    os.chdir(path_swdi + '/' + str(yearname[iyr]))
    tif_files = sorted(glob.glob('*.tif'))
    names_daily = np.array([int(os.path.splitext(tif_files[idt])[0].split('_')[-1][-3:]) for idt in range(len(tif_files))])
    ind_season_all = []

    if iyr != 1:
        for i in range(len(seasons_div_norm)-1):
            ind_season = np.where((names_daily > seasons_div_norm[i]) & (names_daily <= seasons_div_norm[i+1]))[0]
            ind_season_all.append(ind_season)
            del (ind_season)
    else:
        for i in range(len(seasons_div_leap)-1):
            ind_season = np.where((names_daily > seasons_div_leap[i]) & (names_daily <= seasons_div_leap[i+1]))[0]
            ind_season_all.append(ind_season)
            del(ind_season)

    ind_season_all_years.append(ind_season_all)


# Average the SWDI data by seasons and map
swdi_arr_avg_all = np.empty([len(lat_aus_ease_1km), len(lon_aus_ease_1km), 4], dtype='float32')
swdi_arr_avg_all[:] = np.nan
for iyr in [4]:#range(len(yearname)):
    os.chdir(path_swdi + '/' + str(yearname[iyr]))
    tif_files = sorted(glob.glob('*.tif'))

    swdi_arr_avg_all = []
    for ise in range(len(ind_season_all_years[iyr])):
        season_list = ind_season_all_years[iyr][ise]

        swdi_arr_all = []
        for idt in range(len(season_list)):
            swdi_tf = gdal.Open(tif_files[season_list[idt]])
            swdi_arr = swdi_tf.ReadAsArray().astype(np.float32)
            swdi_arr_all.append(swdi_arr)
            print(tif_files[season_list[idt]])

        swdi_arr_all = np.array(swdi_arr_all)
        swdi_arr_avg = np.nanmean(swdi_arr_all, axis=0)

        swdi_arr_avg_all.append(swdi_arr_avg)
        del(swdi_arr_all, swdi_arr_avg, season_list)

swdi_arr_avg_all = np.array(swdi_arr_avg_all)


# 2.3 Make the seasonally averaged SWDI maps in Australia (2019)
shape_world = ShapelyFeature(Reader(path_gis_data + '/gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')

xx_wrd, yy_wrd = np.meshgrid(lon_aus_ease_1km, lat_aus_ease_1km) # Create the map matrix
title_content = ['JFM', 'AMJ', 'JAS', 'OND']
columns = 2
rows = 2
fig = plt.figure(figsize=(10, 8), facecolor='w', edgecolor='k')
for ipt in range(len(monthname)//3):
    ax = fig.add_subplot(rows, columns, ipt+1, projection=ccrs.PlateCarree(),
                         extent=[lon_aus_ease_1km[0], lon_aus_ease_1km[-1], lat_aus_ease_1km[-1], lat_aus_ease_1km[0]])
    ax.add_feature(shape_world, linewidth=0.5)
    img = ax.pcolormesh(xx_wrd, yy_wrd, swdi_arr_allyear[ipt, :, :], vmin=-30, vmax=30, cmap='coolwarm_r')
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=10)
    gl.ylocator = mticker.MultipleLocator(base=10)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # cbar = plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
    # cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=10, x=0.95)
    # ax.set_title(title_content[ipt], pad=25, fontsize=16, weight='bold')
    ax.text(117, -39, title_content[ipt], fontsize=18, horizontalalignment='left',
            verticalalignment='top', weight='bold')
plt.subplots_adjust(left=0.04, right=0.88, bottom=0.05, top=0.92, hspace=0.25, wspace=0.25)
cbar_ax = fig.add_axes([0.93, 0.2, 0.02, 0.6])
fig.colorbar(img, cax=cbar_ax, extend='both')
plt.show()
plt.savefig(path_results + '/swdi_aus.png')


# 2.4 Make the seasonally averaged SWDI maps in Murray-Darling River basin (2019)
# Load in watershed shapefile boundaries
path_shp_md = path_gis_data + '/wrd_riverbasins/Aqueduct_river_basins_MURRAY - DARLING'
shp_md_file = "Aqueduct_river_basins_MURRAY - DARLING.shp"
shapefile_md = fiona.open(path_shp_md + '/' + shp_md_file, 'r')
crop_shape_md = [feature["geometry"] for feature in shapefile_md]
shp_md_extent = list(shapefile_md.bounds)
output_crs = 'EPSG:4326'

#Subset the region of Murray-Darling
[lat_1km_md, row_md_1km_ind, lon_1km_md, col_md_1km_ind] = \
    coordtable_subset(lat_aus_ease_1km, lon_aus_ease_1km, shp_md_extent[3], shp_md_extent[1], shp_md_extent[2], shp_md_extent[0])

[lat_1km_md, row_md_1km_world_ind, lon_1km_md, col_md_1km_world_ind] = \
    coordtable_subset(lat_world_ease_1km, lon_world_ease_1km, shp_md_extent[3], shp_md_extent[1], shp_md_extent[2], shp_md_extent[0]) # world

[lat_10km_md, row_md_10km_world_ind, lon_10km_md, col_md_10km_world_ind] = \
    coordtable_subset(lat_world_geo_10km, lon_world_geo_10km, shp_md_extent[3], shp_md_extent[1], shp_md_extent[2], shp_md_extent[0]) # GPM


swdi_arr_avg_all_md = swdi_arr_avg_all[:, row_md_1km_ind[0]:row_md_1km_ind[-1]+1, col_md_1km_ind[0]:col_md_1km_ind[-1]+1]

