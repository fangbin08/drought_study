import os
import matplotlib.pyplot as plt
import pandas as pd
# from ds_algorithm_txson import path_smap_sm_ds
plt.rcParams["font.family"] = "sans-serif"
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import NullFormatter
import numpy as np
import glob
import calendar
import h5py
from osgeo import gdal
import scipy.ndimage
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
from pyproj import Transformer
import datetime

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
path_lmask = '/Volumes/MyPassport/SMAP_Project/Datasets/Lmask'
# Path of model data
path_model = '/Volumes/Elements/Datasets/model_data/gldas/'
# Path of GIS data
path_gis_data = '/Users/binfang/Documents/SMAP_Project/data/gis_data'
# Path of results
path_results = '/Users/binfang/Documents/SMAP_Project/results/results_241116'
path_model_evaluation = '/Users/binfang/Documents/SMAP_Project/results/results_241116/model_evaluation'
# Path of model output SM
path_model_sm = '/Volumes/Elements2/VIIRS/SM_model/'
# Path of downscaled SM
path_smap_sm_ds_era = '/Volumes/Elements2/SMAP/SM_downscaled_era/'
path_smap_sm_ds_gldas = '/Volumes/Elements2/SMAP/SM_downscaled_gldas/'
# Path of 9 km SMAP SM
path_smap = '/Volumes/Elements/Datasets/SMAP'
path_smap_400m = '/Volumes/Elements2/SMAP/SM_downscaled_gldas/'
path_smap_400m_era = '/Volumes/UVA_data/Dataset/SM_downscaled_era/'
# Path of VIIRS data
path_viirs_lst = '/Volumes/UVA_data/Dataset/VIIRS/LST/'
# Path of VIIRS data regridded output
path_viirs_output = '/Users/binfang/Downloads/Processing/VIIRS/'
path_lst_geo = '/Volumes/UVA_data/Dataset/VIIRS/LST_geo/'
path_lst_ease = '/Volumes/UVA_data/Dataset/VIIRS/LST_ease/'
path_viirs_lai = '/Volumes/UVA_data/Dataset/VIIRS/LAI/'
path_lai_geo = '/Volumes/UVA_data/Dataset/VIIRS/LAI_geo/'
path_lai_ease = '/Volumes/UVA_data/Dataset/VIIRS/LAI_ease/'
smap_sm_9km_name = ['smap_sm_9km_am', 'smap_sm_9km_pm']

# Load in geo-location parameters
os.chdir(path_workspace)
f = h5py.File("ds_parameters.hdf5", "r")
varname_list = ['lat_world_max', 'lat_world_min', 'lon_world_max', 'lon_world_min', 'lat_world_ease_1km',
                'lon_world_ease_1km', 'row_world_ease_1km_from_9km_ind', 'col_world_ease_1km_from_9km_ind',
                'lat_world_geo_400m', 'lon_world_geo_400m', 'lat_world_ease_400m', 'lon_world_ease_400m',
                'lat_world_ease_9km', 'lon_world_ease_9km', 'lat_world_ease_12_5km', 'lon_world_ease_12_5km',
                'row_world_ease_400m_ind', 'col_world_ease_400m_ind', 'row_world_ease_9km_ind', 'col_world_ease_9km_ind']

for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f.close()

row_world_geo_400m_ind = np.arange(len(lat_world_geo_400m))
col_world_geo_400m_ind = np.arange(len(lon_world_geo_400m))

# Generate sequence of string between start and end dates (Year + DOY)
start_date = '2010-01-01'
end_date = '2023-12-31'

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
yearname = np.linspace(2010, 2023, 14, dtype='int')
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

date_seq_array = np.array([int(date_seq[x]) for x in range(len(date_seq))])
daysofyear_cumsum = np.cumsum(daysofyear)
date_seq_array_cumsum = np.hsplit(date_seq_array, daysofyear_cumsum)[:-1] # split by each month
date_seq_ymd_group = np.split(date_seq_ymd, daysofyear_cumsum)[:-1]


########################################################################################################################
# 1. Process viirs LST data

# For 400 m viirs data: lat: -60~75, lon: -180~180
# Geographic projection extent:
# Geographic projection dimensions: 33750, 90000 (full size dimensions: 45000, 90000)
# Geographic grid size: 3750*3750
# Row:9, col: 24

# Ease-grid projection dimensions: 36540, 86760
# Ease-grid grid size: 4060*3615
# Row:9, col: 24
lat_extent_max = 75
lat_extent_min = -60
lon_extent_max = 180
lon_extent_min = -180
interdist_ease_400m = 400.358009339824

# Create tables for tile numbers
row_world_ease_400m_ind_tile = np.repeat(np.arange(9), 4060)
col_world_ease_400m_ind_tile = np.repeat(np.arange(24), 3615)
row_world_ease_400m_ind_tile_local = np.tile(np.arange(4060), 9)
col_world_ease_400m_ind_tile_local = np.tile(np.arange(3615), 24)
df_row_world_ease_400m_ind = pd.DataFrame({'row_ind_world': row_world_ease_400m_ind,
                                           'row_ind_local': row_world_ease_400m_ind_tile_local,
                                           'row_ind_tile': row_world_ease_400m_ind_tile})
df_col_world_ease_400m_ind = pd.DataFrame({'col_ind_world': col_world_ease_400m_ind,
                                           'col_ind_local': col_world_ease_400m_ind_tile_local,
                                           'col_ind_tile': col_world_ease_400m_ind_tile})

viirs_mat_fill = np.empty([3750, 3750], dtype='float32')
viirs_mat_fill[:] = np.nan


########################################################################################################################
# 2. SMAP SM maps (Worldwide)

# 2.1 Composite the data of the first 16 days of one specific month
# Load in SMAP data
year_plt = [2022]
month_plt = list([7])
days_begin = 1
days_end = 7
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


# Load in SMAP 400 m SM
# Check number of files in each tile folder
tile_num_all = []
for iyr in [10]:
    tile_name = sorted(glob.glob(path_smap_sm_ds_gldas + str(yearname[iyr]) + '/*', recursive=True))
    for ite in range(len(tile_name)):
        smap_file_list = sorted(glob.glob(tile_name[ite] + '/*'))
        tile_num_all.append(len(smap_file_list))
tile_num_all = np.array(tile_num_all)


# April
for iyr in [13]:

    tile_name = sorted(glob.glob(path_smap_sm_ds_gldas + str(yearname[iyr]) + '/*', recursive=True))
    tile_name_base = [int(tile_name[x].split('/')[-1][1:]) for x in range(len(tile_name))]
    tile_name_ind = np.array(tile_name_base) - 1

    smap_tile_read_all_april = []
    for ite in range(len(tile_name)):

        smap_file_list = sorted(glob.glob(tile_name[ite] + '/*'))
        smap_file_list = smap_file_list[90:120]

        smap_mat_read_all = []
        for idt in range(len(smap_file_list)):
            src_tf = rasterio.open(smap_file_list[idt]).read()
            src_tf = np.nanmean(np.stack((src_tf), axis=0), axis=0)
            smap_mat_read_all.append(src_tf)
            del(src_tf)
        smap_mat_read_all = np.nanmean(np.stack((smap_mat_read_all), axis=0), axis=0)
        smap_mat_read_all = scipy.ndimage.zoom(smap_mat_read_all, zoom=0.2, order=1)
        smap_tile_read_all_april.append(smap_mat_read_all)
        print(tile_name[ite])
        del(smap_mat_read_all)

# July
for iyr in [13]:

    tile_name = sorted(glob.glob(path_smap_sm_ds_gldas + str(yearname[iyr]) + '/*', recursive=True))
    tile_name_base = [int(tile_name[x].split('/')[-1][1:]) for x in range(len(tile_name))]
    tile_name_ind = np.array(tile_name_base) - 1

    smap_tile_read_all_july = []
    for ite in range(len(tile_name)):

        smap_file_list = sorted(glob.glob(tile_name[ite] + '/*'))
        smap_file_list = smap_file_list[181:212]

        smap_mat_read_all = []
        for idt in range(len(smap_file_list)):
            src_tf = rasterio.open(smap_file_list[idt]).read()
            src_tf = np.nanmean(np.stack((src_tf), axis=0), axis=0)
            smap_mat_read_all.append(src_tf)
            del(src_tf)
        smap_mat_read_all = np.nanmean(np.stack((smap_mat_read_all), axis=0), axis=0)
        smap_mat_read_all = scipy.ndimage.zoom(smap_mat_read_all, zoom=0.2, order=1)
        smap_tile_read_all_july.append(smap_mat_read_all)
        print(tile_name[ite])
        del(smap_mat_read_all)

# Assemble the tiles of SMAP SM data into a world map
smap_sm_mat_fill = np.empty([812, 723], dtype='float32')
smap_sm_mat_fill[:] = np.nan

tile_data_list_april = [smap_sm_mat_fill for _ in range(216)]
for ite in range(len(tile_name_ind)):
    tile_data_list_april[tile_name_ind[ite]] = smap_tile_read_all_april[ite]

tile_data_list_div_april = [tile_data_list_april[i:i + 24] for i in range(0, len(tile_data_list_april), 24)]
tile_data_list_div_by_row_april = [np.hstack(tile_data_list_div_april[x]) for x in range(len(tile_data_list_div_april))]
smap_400m_world_april = np.vstack(tile_data_list_div_by_row_april)


# Assemble the tiles of SMAP SM data into a world map
smap_sm_mat_fill = np.empty([812, 723], dtype='float32')
smap_sm_mat_fill[:] = np.nan

tile_data_list_july = [smap_sm_mat_fill for _ in range(216)]
for ite in range(len(tile_name_ind)):
    tile_data_list_july[tile_name_ind[ite]] = smap_tile_read_all_july[ite]

tile_data_list_div_july = [tile_data_list_july[i:i + 24] for i in range(0, len(tile_data_list_july), 24)]
tile_data_list_div_by_row_july = [np.hstack(tile_data_list_div_july[x]) for x in range(len(tile_data_list_div_july))]
smap_400m_world_july = np.vstack(tile_data_list_div_by_row_july)


with h5py.File(path_model_evaluation + '/smap_400m_world_2018.hdf5', 'w') as f:
    f.create_dataset('smap_400m_world_2018_april', data=smap_400m_world_april)
    f.create_dataset('smap_400m_world_2018_july', data=smap_400m_world_july)
f.close()

del(smap_400m_world_april, smap_400m_world_july)


# All 12 months (LST)
for iyr in [10]:

    tile_name = sorted(glob.glob(path_smap_400m + str(yearname[iyr]) + '/*', recursive=True))
    tile_name_base = [int(tile_name[x].split('/')[-1][1:]) for x in range(len(tile_name))]
    tile_name_ind = np.array(tile_name_base) - 1

    smap_tile_read_all_tiles = []
    for ite in [0, 1]:#range(len(tile_name)):

        smap_tile_read_all_monthly = []
        for imo in range(len(monthnum)):

            smap_file_list = sorted(glob.glob(tile_name[ite] + '/*'))
            smap_file_list = smap_file_list[daysofmonth_seq_cumsum[:, iyr][imo]:daysofmonth_seq_cumsum[:, iyr][imo+1]]

            smap_mat_read_all = []
            for idt in range(len(smap_file_list)):
                src_tf = rasterio.open(smap_file_list[idt]).read()
                src_tf = np.nanmean(np.stack((src_tf), axis=0), axis=0)
                # src_tf = h5py.File(smap_file_list[idt], 'r')
                # src_tf = src_tf['viirs_lst_delta_am'][()]
                smap_mat_read_all.append(src_tf)
                del(src_tf)
            smap_mat_read_all = np.nanmean(np.stack((smap_mat_read_all), axis=0), axis=0)
            smap_mat_read_all = scipy.ndimage.zoom(smap_mat_read_all, zoom=0.2, order=1)
            smap_tile_read_all_monthly.append(smap_mat_read_all)
            print(tile_name[ite] + '_' + str(imo+1))
            del(smap_mat_read_all)

        smap_tile_read_all_tiles.append(smap_tile_read_all_monthly)
        del(smap_tile_read_all_monthly)


# Assemble the tiles of SMAP SM data into a world map
smap_sm_mat_fill = np.empty([812, 723], dtype='float32')
smap_sm_mat_fill[:] = np.nan

tile_data_list_allmonth = []
for imo in range(len(monthnum)):
    tile_data_list = [smap_sm_mat_fill for _ in range(216)]
    for ite in range(len(smap_tile_read_all_tiles)):
        tile_data_list[tile_name_ind[ite]] = smap_tile_read_all_tiles[ite][imo]
    tile_data_list_allmonth.append(tile_data_list)
    print(imo)
    del(tile_data_list)

smap_400m_world_allmonth = []
for imo in range(len(monthnum)):
    tile_data_list_div = [tile_data_list_allmonth[imo][i:i + 24] for i in range(0, len(tile_data_list_allmonth[imo]), 24)]
    tile_data_list_div_by_row = [np.hstack(tile_data_list_div[x]) for x in range(len(tile_data_list_div))]
    smap_400m_world = np.vstack(tile_data_list_div_by_row)
    smap_400m_world_allmonth.append(smap_400m_world)
    print(imo)
    del(smap_400m_world)


with h5py.File(path_model_evaluation + '/smap_400m_world_' + str(yearname[iyr]) + '.hdf5', 'w') as f:
    for imo in range(len(monthnum)):
        f.create_dataset('smap_400m_world_' + str(imo+1).zfill(2), data=smap_400m_world_allmonth[imo])
f.close()



# f_read = h5py.File('/Users/binfang/Documents/SMAP_Project/results/results_240615/model_evaluation/smap_400m_world_2021_july.hdf5', "r")
# smap_400m_world_2021_july = f_read['smap_400m_world_july'][()]
# f_read.close()


# 2.2 Maps of the world

# Read the map data
f_read = h5py.File(path_model_evaluation + "/smap_400m_world_2021.hdf5", "r")
varname_read_list = list(f_read.keys())
# for x in range(len(varname_read_list)):
#     var_obj = f_read[varname_read_list[x]][()]
#     exec(varname_read_list[x] + '= var_obj')
#     del(var_obj)
# f_read.close()
smap_400m_world_april = f_read['smap_400m_world_04'][()]
smap_400m_world_july = f_read['smap_400m_world_07'][()]

smap_400m_world_april[smap_400m_world_april <= 0] = np.nan
smap_400m_world_july[smap_400m_world_july <= 0] = np.nan

# smap_400m_world_2020_april_res = scipy.ndimage.zoom(smap_400m_world_2020_april, zoom=0.2, order=1)
# smap_400m_world_2020_july_res = scipy.ndimage.zoom(smap_400m_world_2020_july, zoom=0.2, order=1)


lat_world_ease_400m_res = scipy.ndimage.zoom(lat_world_ease_400m, zoom=0.2, order=1)
lon_world_ease_400m_res = scipy.ndimage.zoom(lon_world_ease_400m, zoom=0.2, order=1)
xx_wrd, yy_wrd = np.meshgrid(lon_world_ease_400m_res, lat_world_ease_400m_res) # Create the map matrix
shape_world = ShapelyFeature(Reader(path_gis_data + '/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
shp_world_extent = [-180.0, 180.0, -45, 65]

smap_400m_world_data = list((smap_400m_world_april, smap_400m_world_july))
# smap_400m_world_data = list((smap_400m_world_all[3], smap_400m_world_all[6]))
title_content = ['April 2021', 'July 2021']

fig = plt.figure(figsize=(12, 7), facecolor='w', edgecolor='k', dpi=200)
for ipt in range(2):
    plt.subplots_adjust(left=0.08, right=0.92, bottom=0.1, top=0.95, hspace=0.1, wspace=0.1)
    ax = fig.add_subplot(2, 1, ipt+1, projection=ccrs.PlateCarree())
    ax.add_feature(shape_world, linewidth=0.5)
    img = ax.pcolormesh(xx_wrd, yy_wrd, smap_400m_world_data[ipt], vmin=0, vmax=0.6, cmap='turbo_r')
    ax.set_extent([-180.0, 180.0, -45, 65], ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = mticker.FixedLocator([65, 30, 0, -30, -45])
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # cbar = plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
    # cbar.ax.tick_params(labelsize=15)
    # cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=15, x=0.95)
    ax.set_title(title_content[ipt], pad=12, fontsize=13, weight='bold')
    # ax.text(-175, -40, title_content, fontsize=11, horizontalalignment='left',
    #         verticalalignment='top', weight='bold')
cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.015])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both', pad=0.1, orientation='horizontal')
cbar.ax.tick_params(labelsize=10)
cbar_ax.locator_params(nbins=5)
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=10, x=1.05, y=0.05, labelpad=-5)
# plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.92, hspace=0.25, wspace=0.25)
plt.savefig(path_results + '/smap_400m_world_2021_new.png')
plt.close()



# delta_lst
lat_world_ease_400m_res = scipy.ndimage.zoom(lat_world_ease_400m, zoom=0.2, order=1)
lon_world_ease_400m_res = scipy.ndimage.zoom(lon_world_ease_400m, zoom=0.2, order=1)
xx_wrd, yy_wrd = np.meshgrid(lon_world_ease_400m_res, lat_world_ease_400m_res) # Create the map matrix
shape_world = ShapelyFeature(Reader(path_gis_data + '/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
shp_world_extent = [-180.0, 180.0, -45, 65]

smap_400m_world_data = list((smap_400m_world_april, smap_400m_world_july))
title_content = ['April 2020', 'July 2020']

fig = plt.figure(figsize=(12, 7), facecolor='w', edgecolor='k', dpi=200)
for ipt in range(2):
    plt.subplots_adjust(left=0.08, right=0.92, bottom=0.1, top=0.95, hspace=0.1, wspace=0.1)
    ax = fig.add_subplot(2, 1, ipt+1, projection=ccrs.PlateCarree())
    ax.add_feature(shape_world, linewidth=0.5)
    img = ax.pcolormesh(xx_wrd, yy_wrd, smap_400m_world_data[ipt], vmin=0, vmax=80, cmap='turbo')
    ax.set_extent([-180.0, 180.0, -45, 65], ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = mticker.FixedLocator([65, 30, 0, -30, -45])
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # cbar = plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
    # cbar.ax.tick_params(labelsize=15)
    # cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=15, x=0.95)
    ax.set_title(title_content[ipt], pad=12, fontsize=13, weight='bold')
    # ax.text(-175, -40, title_content, fontsize=11, horizontalalignment='left',
    #         verticalalignment='top', weight='bold')
cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.015])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both', pad=0.1, orientation='horizontal')
cbar.ax.tick_params(labelsize=10)
cbar_ax.locator_params(nbins=5)
cbar.set_label('(k)', fontsize=10, x=1.05, y=0.05, labelpad=-5)
# plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.92, hspace=0.25, wspace=0.25)
plt.savefig(path_results + '/smap_400m_world_2020_lst.png')
plt.close()


# 2.3 Maps of the world (All 12 months)

# Read the map data
f_read = h5py.File("/Users/binfang/Downloads/smap_400m_world_2024.hdf5", "r")
varname_read_list = list(f_read.keys())
smap_400m_world_all = []
for x in range(len(varname_read_list)):
    var_obj = f_read[varname_read_list[x]][()]
    var_obj[var_obj <= 0] = np.nan
    smap_400m_world_all.append(var_obj)
    del(var_obj)
f_read.close()


lat_world_ease_400m_res = scipy.ndimage.zoom(lat_world_ease_400m, zoom=0.2, order=1)
lon_world_ease_400m_res = scipy.ndimage.zoom(lon_world_ease_400m, zoom=0.2, order=1)
xx_wrd, yy_wrd = np.meshgrid(lon_world_ease_400m_res, lat_world_ease_400m_res) # Create the map matrix
shape_world = ShapelyFeature(Reader(path_gis_data + '/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
shp_world_extent = [-180.0, 180.0, -45, 65]

title_content = '2024'
for imo in range(len(monthnum)):
    fig = plt.figure(figsize=(12, 5), facecolor='w', edgecolor='k')
    # plt.subplots_adjust(left=0.08, right=0.92, bottom=0.1, top=0.95, hspace=0.1, wspace=0.1)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.add_feature(shape_world, linewidth=0.5)
    img = ax.pcolormesh(xx_wrd, yy_wrd, smap_400m_world_all[imo], vmin=0, vmax=0.6, cmap='turbo_r')
    ax.set_extent([-180.0, 180.0, -45, 65], ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = mticker.FixedLocator([65, 30, 0, -30, -45])
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.set_title(title_content + '/' + str(imo+1).zfill(2), pad=12, fontsize=13, weight='bold')
    cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.015])
    cbar = fig.colorbar(img, cax=cbar_ax, extend='both', pad=0.1, orientation='horizontal')
    cbar.ax.tick_params(labelsize=10)
    cbar_ax.locator_params(nbins=5)
    cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=10, x=1.05, y=0.05, labelpad=-5)
    plt.savefig(path_results + '/smap_400m_world_' + title_content + '_' + str(imo+1).zfill(2) + '.png', dpi=300, bbox_inches='tight')
    print(imo)
    plt.close()



# delta_lst
lat_world_ease_400m_res = scipy.ndimage.zoom(lat_world_ease_400m, zoom=0.2, order=1)
lon_world_ease_400m_res = scipy.ndimage.zoom(lon_world_ease_400m, zoom=0.2, order=1)
xx_wrd, yy_wrd = np.meshgrid(lon_world_ease_400m_res, lat_world_ease_400m_res) # Create the map matrix
shape_world = ShapelyFeature(Reader(path_gis_data + '/gshhg-shp-2.3.7/GSHHS_shp/c/GSHHS_c_L1.shp').geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
shp_world_extent = [-180.0, 180.0, -45, 65]

smap_400m_world_data = list((smap_400m_world_april, smap_400m_world_july))
title_content = ['April 2020', 'July 2020']

fig = plt.figure(figsize=(12, 7), facecolor='w', edgecolor='k', dpi=200)
for ipt in range(2):
    plt.subplots_adjust(left=0.08, right=0.92, bottom=0.1, top=0.95, hspace=0.1, wspace=0.1)
    ax = fig.add_subplot(2, 1, ipt+1, projection=ccrs.PlateCarree())
    ax.add_feature(shape_world, linewidth=0.5)
    img = ax.pcolormesh(xx_wrd, yy_wrd, smap_400m_world_data[ipt], vmin=0, vmax=80, cmap='turbo')
    ax.set_extent([-180.0, 180.0, -45, 65], ccrs.PlateCarree())
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.FixedLocator([-180, -90, 0, 90, 180])
    gl.ylocator = mticker.FixedLocator([65, 30, 0, -30, -45])
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # cbar = plt.colorbar(img, extend='both', orientation='horizontal', aspect=50, pad=0.1)
    # cbar.ax.tick_params(labelsize=15)
    # cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=15, x=0.95)
    ax.set_title(title_content[ipt], pad=12, fontsize=13, weight='bold')
    # ax.text(-175, -40, title_content, fontsize=11, horizontalalignment='left',
    #         verticalalignment='top', weight='bold')
cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.015])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both', pad=0.1, orientation='horizontal')
cbar.ax.tick_params(labelsize=10)
cbar_ax.locator_params(nbins=5)
cbar.set_label('(k)', fontsize=10, x=1.05, y=0.05, labelpad=-5)
# plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.92, hspace=0.25, wspace=0.25)
plt.savefig(path_results + '/smap_400m_world_2020_lst.png')
plt.close()



# ########################################################################################################################
# 6.2 River Basin maps
# 6.2.1 Middle Colorado RB (TxSON)

path_shp_txs = path_gis_data + '/watershed_boundary/'
os.chdir(path_shp_txs)
shp_txs_file = "txson.shp"
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
[lat_400m_txs, row_txs_400m_ind, lon_400m_txs, col_txs_400m_ind] = \
    coordtable_subset(lat_world_ease_400m, lon_world_ease_400m,
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
year_plt = [2022]
month_plt = list([7])
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



#Load and subset the region of Middle Colorado RB (SMAP 400 m)

df_row_txs_400m_ind = df_row_world_ease_400m_ind.iloc[row_txs_400m_ind, :]
df_col_txs_400m_ind = df_col_world_ease_400m_ind.iloc[col_txs_400m_ind, :]

tiles_num_row = pd.unique(df_row_txs_400m_ind['row_ind_tile'])
tiles_num_col = pd.unique(df_col_txs_400m_ind['col_ind_tile'])
tiles_num = tiles_num_row[0] * 24 + tiles_num_col[0] + 1

row_txs_400m_ind_local = df_row_txs_400m_ind['row_ind_local'].tolist()
col_txs_400m_ind_local = df_col_txs_400m_ind['col_ind_local'].tolist()


# 400m (GLDAS)
smap_400m_mean_1_allyear = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        smap_400m_load_1_stack = []
        for idt in range(days_begin, days_end+1):
            str_date = str(iyr) + '-' + monthname[month_plt[imo]-1] + '-' + str(idt).zfill(2)
            str_doy = str(datetime.datetime.strptime(str_date, '%Y-%m-%d').date().timetuple().tm_yday)
            tif_file_smap_400m = (path_smap_400m + str(iyr) + '/T' + str(tiles_num).zfill(3) + '/smap_sm_400m_' + str(iyr)
                                  + str_doy.zfill(3) + '_T' + str(tiles_num).zfill(3) + '.tif')
            src_tf = gdal.Open(tif_file_smap_400m)
            src_tf_arr = src_tf.ReadAsArray()[:, row_txs_400m_ind_local[0]:row_txs_400m_ind_local[-1]+1,
                         col_txs_400m_ind_local[0]:col_txs_400m_ind_local[-1]+1]
            smap_400m_load_1 = np.nanmean(src_tf_arr, axis=0)
            smap_400m_load_1_stack.append(smap_400m_load_1)

            print(str_date)
            del(src_tf_arr, smap_400m_load_1)

        smap_400m_load_1_stack = np.stack(smap_400m_load_1_stack)
        smap_400m_mean_1_allyear.append(smap_400m_load_1_stack)
        del(smap_400m_load_1_stack)

smap_400m_data_stack_txs = np.squeeze(np.array(smap_400m_mean_1_allyear))
del(smap_400m_mean_1_allyear)



# 400m (ERA5)
smap_400m_mean_1_allyear = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        smap_400m_load_1_stack = []
        for idt in range(days_begin, days_end+1):
            str_date = str(iyr) + '-' + monthname[month_plt[imo]-1] + '-' + str(idt).zfill(2)
            str_doy = str(datetime.datetime.strptime(str_date, '%Y-%m-%d').date().timetuple().tm_yday)
            tif_file_smap_400m = (path_smap_400m_era + str(iyr) + '/T' + str(tiles_num).zfill(3) + '/smap_sm_400m_' + str(iyr)
                                  + str_doy.zfill(3) + '_T' + str(tiles_num).zfill(3) + '.tif')
            src_tf = gdal.Open(tif_file_smap_400m)
            src_tf_arr = src_tf.ReadAsArray()[:, row_txs_400m_ind_local[0]:row_txs_400m_ind_local[-1]+1,
                         col_txs_400m_ind_local[0]:col_txs_400m_ind_local[-1]+1]
            smap_400m_load_1 = np.nanmean(src_tf_arr, axis=0)
            smap_400m_load_1_stack.append(smap_400m_load_1)

            print(str_date)
            del(src_tf_arr, smap_400m_load_1)

        smap_400m_load_1_stack = np.stack(smap_400m_load_1_stack)
        smap_400m_mean_1_allyear.append(smap_400m_load_1_stack)
        del(smap_400m_load_1_stack)

smap_400m_data_stack_txs_era = np.squeeze(np.array(smap_400m_mean_1_allyear))
del(smap_400m_mean_1_allyear)

with h5py.File(path_model_evaluation + '/smap_txs_sm_1.hdf5', 'w') as f:
    f.create_dataset('smap_9km_data_stack_txs', data=smap_9km_data_stack_txs)
    f.create_dataset('smap_1km_data_stack_txs', data=smap_1km_data_stack_txs)
    f.create_dataset('smap_400m_data_stack_txs', data=smap_400m_data_stack_txs)
    # f.create_dataset('smap_400m_data_stack_txs_era', data=smap_400m_data_stack_txs_era)
f.close()

# Read the map data
f_read = h5py.File(path_model_evaluation + "/smap_txs_sm_1.hdf5", "r")
varname_read_list = ['smap_9km_data_stack_txs', 'smap_1km_data_stack_txs', 'smap_400m_data_stack_txs']
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


# 400 m (GLDAS)
smap_masked_ds_txs_400m_all = []
for n in range(smap_400m_data_stack_txs.shape[0]):
    sub_window_txs_400m = Window(col_txs_400m_ind[0], row_txs_400m_ind[0], len(col_txs_400m_ind), len(row_txs_400m_ind))
    kwargs_400m_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_400m),
                      'height': len(lat_world_ease_400m), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(400.358009339824, 0.0, -17367530.44516138, 0.0, -400.358009339824, 7314540.79258289)}
    smap_sm_txs_400m_output = sub_n_reproj(smap_400m_data_stack_txs[n, :, :], kwargs_400m_sub, sub_window_txs_400m, output_crs)

    masked_ds_txs_400m, mask_transform_ds_txs_400m = mask(dataset=smap_sm_txs_400m_output, shapes=crop_shape_txs, crop=True)
    masked_ds_txs_400m[np.where(masked_ds_txs_400m == 0)] = np.nan
    masked_ds_txs_400m = masked_ds_txs_400m.squeeze()

    smap_masked_ds_txs_400m_all.append(masked_ds_txs_400m)

smap_masked_ds_txs_400m_all = np.asarray(smap_masked_ds_txs_400m_all)

# 400 m (ERA5)
smap_masked_ds_txs_400m_all_era = []
for n in range(smap_400m_data_stack_txs_era.shape[0]):
    sub_window_txs_400m = Window(col_txs_400m_ind[0], row_txs_400m_ind[0], len(col_txs_400m_ind), len(row_txs_400m_ind))
    kwargs_400m_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_400m),
                      'height': len(lat_world_ease_400m), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(400.358009339824, 0.0, -17367530.44516138, 0.0, -400.358009339824, 7314540.79258289)}
    smap_sm_txs_400m_output = sub_n_reproj(smap_400m_data_stack_txs_era[n, :, :], kwargs_400m_sub, sub_window_txs_400m, output_crs)

    masked_ds_txs_400m, mask_transform_ds_txs_400m = mask(dataset=smap_sm_txs_400m_output, shapes=crop_shape_txs, crop=True)
    masked_ds_txs_400m[np.where(masked_ds_txs_400m == 0)] = np.nan
    masked_ds_txs_400m = masked_ds_txs_400m.squeeze()

    smap_masked_ds_txs_400m_all_era.append(masked_ds_txs_400m)

smap_masked_ds_txs_400m_all_era = np.asarray(smap_masked_ds_txs_400m_all_era)



# Calculate the 7-day averaged maps
smap_1km_size = smap_masked_ds_txs_1km_all.shape
smap_400m_size = smap_masked_ds_txs_400m_all.shape
smap_masked_ds_txs_1km_avg = np.reshape(smap_masked_ds_txs_1km_all[:28, :, :], (7, 4, smap_1km_size[1], smap_1km_size[2]))
smap_masked_ds_txs_1km_avg = np.nanmean(smap_masked_ds_txs_1km_avg, axis=0)
smap_masked_ds_txs_9km_avg = np.reshape(smap_masked_ds_txs_9km_all[:28, :, :], (7, 4, smap_1km_size[1], smap_1km_size[2]))
smap_masked_ds_txs_9km_avg = np.nanmean(smap_masked_ds_txs_9km_avg, axis=0)
smap_masked_ds_txs_400m_avg = np.reshape(smap_masked_ds_txs_400m_all[:28, :, :], (7, 4, smap_400m_size[1], smap_400m_size[2]))
smap_masked_ds_txs_400m_avg = np.nanmean(smap_masked_ds_txs_400m_avg, axis=0)
smap_masked_ds_txs_400m_avg_era = np.reshape(smap_masked_ds_txs_400m_all_era[:28, :, :], (7, 4, smap_400m_size[1], smap_400m_size[2]))
smap_masked_ds_txs_400m_avg_era = np.nanmean(smap_masked_ds_txs_400m_avg_era, axis=0)

sm_masked_ds_txs_stack = list((smap_masked_ds_txs_400m_avg_era, smap_masked_ds_txs_400m_avg, smap_masked_ds_txs_1km_avg, smap_masked_ds_txs_9km_avg))


# 6.2.2.3 Make the subplot maps
feature_shp_txs = ShapelyFeature(Reader(path_shp_txs + '/' + shp_txs_file).geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
extent_txs = np.array(shapefile_txs.bounds)
extent_txs = extent_txs[[0, 2, 1, 3]]


# Shorter version
fig = plt.figure(figsize=(6, 4), facecolor='w', edgecolor='k', dpi=200)
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.9, hspace=0.1, wspace=0.1)
plt.rcParams['lines.linewidth'] = 0.5
for irow in range(4):
    for icol in range(4):
        # 1 km
        ax = fig.add_subplot(4, 4, irow*4+icol+1, projection=ccrs.PlateCarree())
        ax.add_feature(feature_shp_txs)
        img = ax.imshow(sm_masked_ds_txs_stack[icol][irow, :, :], origin='upper', vmin=0, vmax=0.4, cmap='turbo_r',
                   extent=extent_txs)
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
        elif irow == 3:
            gl.left_labels = False
            gl.bottom_labels = True
        else:
            gl.left_labels = False
            gl.bottom_labels = False
        if icol == 0 and irow == 3:
            gl.left_labels = True
            gl.bottom_labels = True
        gl.right_labels = False
        gl.top_labels = False
cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both')
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=7, x=0.95, labelpad=1)
cbar.ax.tick_params(labelsize=7)
cbar_ax.locator_params(nbins=6)
fig.text(0.12, 0.94, '400m (ERA5)', fontsize=9, fontweight='bold')
fig.text(0.32, 0.94, '400m (GLDAS)', fontsize=9, fontweight='bold')
fig.text(0.53, 0.94, '1km (GLDAS)', fontsize=9, fontweight='bold')
fig.text(0.78, 0.94, '9km', fontsize=9, fontweight='bold')
fig.text(0.02, 0.75, 'July 1-7', fontsize=7, fontweight='bold', rotation=90)
fig.text(0.02, 0.5, 'July 8-14', fontsize=7, fontweight='bold', rotation=90)
fig.text(0.02, 0.28, 'July 15-21', fontsize=7, fontweight='bold', rotation=90)
fig.text(0.02, 0.08, 'July 22-28', fontsize=7, fontweight='bold', rotation=90)
# cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=7, x=1.08, y=0.05, labelpad=-15)
plt.savefig(path_results + '/sm_comp_txs_1_new2.png')
plt.close()


# 6.2.1 Duero RB (Remedhus)

path_shp_rem = path_gis_data + '/watershed_boundary/'
os.chdir(path_shp_rem)
shp_rem_file = "remedhus.shp"
shp_rem_ds = ogr.GetDriverByName("ESRI Shapefile").Open(shp_rem_file, 0)
shp_rem_extent = list(shp_rem_ds.GetLayer().GetExtent())

# 6.2.1.1 SMAP
#Load and subset the region of Middle Colorado RB (SMAP 9 km)
[lat_9km_rem, row_rem_9km_ind, lon_9km_rem, col_rem_9km_ind] = \
    coordtable_subset(lat_world_ease_9km, lon_world_ease_9km,
                      shp_rem_extent[3], shp_rem_extent[2], shp_rem_extent[1], shp_rem_extent[0])
[lat_1km_rem, row_rem_1km_ind, lon_1km_rem, col_rem_1km_ind] = \
    coordtable_subset(lat_world_ease_1km, lon_world_ease_1km,
                      shp_rem_extent[3], shp_rem_extent[2], shp_rem_extent[1], shp_rem_extent[0])
[lat_400m_rem, row_rem_400m_ind, lon_400m_rem, col_rem_400m_ind] = \
    coordtable_subset(lat_world_ease_400m, lon_world_ease_400m,
                      shp_rem_extent[3], shp_rem_extent[2], shp_rem_extent[1], shp_rem_extent[0])

row_rem_9km_ind_dis = row_world_ease_1km_from_9km_ind[row_rem_1km_ind]
row_rem_9km_ind_dis_unique = np.unique(row_rem_9km_ind_dis)
row_rem_9km_ind_dis_zero = row_rem_9km_ind_dis - row_rem_9km_ind_dis[0]
col_rem_9km_ind_dis = col_world_ease_1km_from_9km_ind[col_rem_1km_ind]
col_rem_9km_ind_dis_unique = np.unique(col_rem_9km_ind_dis)
col_rem_9km_ind_dis_zero = col_rem_9km_ind_dis - col_rem_9km_ind_dis[0]

col_meshgrid_9km, row_meshgrid_9km = np.meshgrid(col_rem_9km_ind_dis_zero, row_rem_9km_ind_dis_zero)
col_meshgrid_9km = col_meshgrid_9km.reshape(1, -1).squeeze()
row_meshgrid_9km = row_meshgrid_9km.reshape(1, -1).squeeze()

# Load and subset SMAP 9 km SM of Duero RB (Remedhus)
year_plt = [2022]
month_plt = list([6])
days_begin = 1
days_end = 30
days_n = days_end - days_begin + 1

smap_9km_mean_1_allyear = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        hdf_file_smap_9km = path_smap + '/9km' + '/smap_sm_9km_' + str(iyr) + monthname[month_plt[imo]-1] + '.hdf5'
        f_read_smap_9km = h5py.File(hdf_file_smap_9km, "r")
        varname_list_smap_9km = list(f_read_smap_9km.keys())

        smap_9km_load_1 = f_read_smap_9km[varname_list_smap_9km[0]][
                                           row_rem_9km_ind_dis_unique[0]:row_rem_9km_ind_dis_unique[-1] + 1,
                                           col_rem_9km_ind_dis_unique[0]:col_rem_9km_ind_dis_unique[-1] + 1, days_begin-1:days_end]  # AM
        smap_9km_load_2 = f_read_smap_9km[varname_list_smap_9km[1]][
                                           row_rem_9km_ind_dis_unique[0]:row_rem_9km_ind_dis_unique[-1] + 1,
                                           col_rem_9km_ind_dis_unique[0]:col_rem_9km_ind_dis_unique[-1] + 1, days_begin-1:days_end]  # PM

        smap_9km_load_1_disagg = np.array([smap_9km_load_1[row_meshgrid_9km[x], col_meshgrid_9km[x], :]
                                      for x in range(len(col_meshgrid_9km))])
        smap_9km_load_1_disagg = smap_9km_load_1_disagg.reshape((len(row_rem_9km_ind_dis), len(col_rem_9km_ind_dis),
                                                                 smap_9km_load_1.shape[2]))
        smap_9km_load_2_disagg = np.array([smap_9km_load_2[row_meshgrid_9km[x], col_meshgrid_9km[x], :]
                                      for x in range(len(col_meshgrid_9km))])
        smap_9km_load_2_disagg = smap_9km_load_2_disagg.reshape((len(row_rem_9km_ind_dis), len(col_rem_9km_ind_dis),
                                                                 smap_9km_load_2.shape[2]))
        f_read_smap_9km.close()

        smap_9km_mean_1 = np.nanmean(np.stack((smap_9km_load_1_disagg, smap_9km_load_2_disagg), axis=3), axis=3)
        smap_9km_mean_1_allyear.append(smap_9km_mean_1)

        del(smap_9km_load_1, smap_9km_mean_1, smap_9km_load_1_disagg, smap_9km_load_2_disagg)
        print(monthname[month_plt[imo]-1])

smap_9km_data_stack_rem = np.squeeze(np.array(smap_9km_mean_1_allyear))
del(smap_9km_mean_1_allyear)
smap_9km_data_stack_rem = np.transpose(smap_9km_data_stack_rem, (2, 0, 1))


#Load and subset the region of Duero RB (Remedhus) (SMAP 1 km)
smap_1km_mean_1_allyear = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        smap_1km_load_1_stack = []
        for idt in range(days_begin, days_end+1):
            str_date = str(iyr) + '-' + monthname[month_plt[imo]-1] + '-' + str(idt).zfill(2)
            str_doy = str(datetime.datetime.strptime(str_date, '%Y-%m-%d').date().timetuple().tm_yday)
            tif_file_smap_1km = path_smap + '/1km/' + str(iyr) + '/smap_sm_1km_ds_' + str(iyr) + str_doy.zfill(3) + '.tif'
            src_tf = gdal.Open(tif_file_smap_1km)
            src_tf_arr = src_tf.ReadAsArray()[:, row_rem_1km_ind[0]:row_rem_1km_ind[-1]+1,
                         col_rem_1km_ind[0]:col_rem_1km_ind[-1]+1]
            smap_1km_load_1 = np.nanmean(src_tf_arr, axis=0)
            smap_1km_load_1_stack.append(smap_1km_load_1)

            print(str_date)
            del(src_tf_arr, smap_1km_load_1)

        smap_1km_load_1_stack = np.stack(smap_1km_load_1_stack)
        smap_1km_mean_1_allyear.append(smap_1km_load_1_stack)
        del(smap_1km_load_1_stack)

smap_1km_data_stack_rem = np.squeeze(np.array(smap_1km_mean_1_allyear))
del(smap_1km_mean_1_allyear)



#Load and subset the region of Duero RB (Remedhus) (SMAP 400 m)

df_row_rem_400m_ind = df_row_world_ease_400m_ind.iloc[row_rem_400m_ind, :]
df_col_rem_400m_ind = df_col_world_ease_400m_ind.iloc[col_rem_400m_ind, :]

tiles_num_row = pd.unique(df_row_rem_400m_ind['row_ind_tile'])
tiles_num_col = pd.unique(df_col_rem_400m_ind['col_ind_tile'])
tiles_num = tiles_num_row[0] * 24 + tiles_num_col[0] + 1

row_rem_400m_ind_local = df_row_rem_400m_ind['row_ind_local'].tolist()
col_rem_400m_ind_local = df_col_rem_400m_ind['col_ind_local'].tolist()

# GLDAS
smap_400m_mean_1_allyear = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        smap_400m_load_1_stack = []
        for idt in range(days_begin, days_end+1):
            str_date = str(iyr) + '-' + monthname[month_plt[imo]-1] + '-' + str(idt).zfill(2)
            str_doy = str(datetime.datetime.strptime(str_date, '%Y-%m-%d').date().timetuple().tm_yday)
            tif_file_smap_400m = (path_smap_400m + str(iyr) + '/T' + str(tiles_num).zfill(3) + '/smap_sm_400m_' + str(iyr)
                                  + str_doy.zfill(3) + '_T' + str(tiles_num).zfill(3) + '.tif')
            src_tf = gdal.Open(tif_file_smap_400m)
            src_tf_arr = src_tf.ReadAsArray()[:, row_rem_400m_ind_local[0]:row_rem_400m_ind_local[-1]+1,
                         col_rem_400m_ind_local[0]:col_rem_400m_ind_local[-1]+1]
            smap_400m_load_1 = np.nanmean(src_tf_arr, axis=0)
            smap_400m_load_1_stack.append(smap_400m_load_1)

            print(str_date)
            del(src_tf_arr, smap_400m_load_1)

        smap_400m_load_1_stack = np.stack(smap_400m_load_1_stack)
        smap_400m_mean_1_allyear.append(smap_400m_load_1_stack)
        del(smap_400m_load_1_stack)

smap_400m_data_stack_rem = np.squeeze(np.array(smap_400m_mean_1_allyear))
del(smap_400m_mean_1_allyear)


# ERA5
smap_400m_mean_1_allyear = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        smap_400m_load_1_stack = []
        for idt in range(days_begin, days_end+1):
            str_date = str(iyr) + '-' + monthname[month_plt[imo]-1] + '-' + str(idt).zfill(2)
            str_doy = str(datetime.datetime.strptime(str_date, '%Y-%m-%d').date().timetuple().tm_yday)
            tif_file_smap_400m = (path_smap_400m_era + str(iyr) + '/T' + str(tiles_num).zfill(3) + '/smap_sm_400m_' + str(iyr)
                                  + str_doy.zfill(3) + '_T' + str(tiles_num).zfill(3) + '.tif')
            src_tf = gdal.Open(tif_file_smap_400m)
            src_tf_arr = src_tf.ReadAsArray()[:, row_rem_400m_ind_local[0]:row_rem_400m_ind_local[-1]+1,
                         col_rem_400m_ind_local[0]:col_rem_400m_ind_local[-1]+1]
            smap_400m_load_1 = np.nanmean(src_tf_arr, axis=0)
            smap_400m_load_1_stack.append(smap_400m_load_1)

            print(str_date)
            del(src_tf_arr, smap_400m_load_1)

        smap_400m_load_1_stack = np.stack(smap_400m_load_1_stack)
        smap_400m_mean_1_allyear.append(smap_400m_load_1_stack)
        del(smap_400m_load_1_stack)

smap_400m_data_stack_rem_era = np.squeeze(np.array(smap_400m_mean_1_allyear))
del(smap_400m_mean_1_allyear)


with h5py.File(path_model_evaluation + '/smap_rem_sm.hdf5', 'w') as f:
    f.create_dataset('smap_9km_data_stack_rem', data=smap_9km_data_stack_rem)
    f.create_dataset('smap_1km_data_stack_rem', data=smap_1km_data_stack_rem)
    f.create_dataset('smap_400m_data_stack_rem', data=smap_400m_data_stack_rem)
    f.create_dataset('smap_400m_data_stack_rem_era', data=smap_400m_data_stack_rem_era)
f.close()

# Read the map data
f_read = h5py.File(path_model_evaluation + "/smap_rem_sm.hdf5", "r")
varname_read_list = ['smap_9km_data_stack_rem', 'smap_1km_data_stack_rem', 'smap_400m_data_stack_rem', 'smap_400m_data_stack_rem_era']
for x in range(len(varname_read_list)):
    var_obj = f_read[varname_read_list[x]][()]
    exec(varname_read_list[x] + '= var_obj')
    del(var_obj)
f_read.close()




# 6.2.2 Subplot maps for Duero RB (Remedhus)
output_crs = 'EPSG:4326'
shapefile_rem = fiona.open(path_shp_rem + '/' + shp_rem_file, 'r')
crop_shape_rem = [feature["geometry"] for feature in shapefile_rem]

# 6.2.2.1 Subset and reproject the SMAP SM data at watershed
# 1 km
smap_masked_ds_rem_1km_all = []
for n in range(smap_1km_data_stack_rem.shape[0]):
    sub_window_rem_1km = Window(col_rem_1km_ind[0], row_rem_1km_ind[0], len(col_rem_1km_ind), len(row_rem_1km_ind))
    kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                      'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
    smap_sm_rem_1km_output = sub_n_reproj(smap_1km_data_stack_rem[n, :, :], kwargs_1km_sub, sub_window_rem_1km, output_crs)

    masked_ds_rem_1km, mask_transform_ds_rem_1km = mask(dataset=smap_sm_rem_1km_output, shapes=crop_shape_rem, crop=True)
    masked_ds_rem_1km[np.where(masked_ds_rem_1km == 0)] = np.nan
    masked_ds_rem_1km = masked_ds_rem_1km.squeeze()

    smap_masked_ds_rem_1km_all.append(masked_ds_rem_1km)

smap_masked_ds_rem_1km_all = np.asarray(smap_masked_ds_rem_1km_all)


# 9 km
smap_masked_ds_rem_9km_all = []
for n in range(smap_9km_data_stack_rem.shape[0]):
    sub_window_rem_9km = Window(col_rem_1km_ind[0], row_rem_1km_ind[0], len(col_rem_1km_ind), len(row_rem_1km_ind))
    kwargs_9km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                      'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
    smap_sm_rem_9km_output = sub_n_reproj(smap_9km_data_stack_rem[n, :, :], kwargs_9km_sub, sub_window_rem_9km, output_crs)

    masked_ds_rem_9km, mask_transform_ds_rem_9km = mask(dataset=smap_sm_rem_9km_output, shapes=crop_shape_rem, crop=True)
    masked_ds_rem_9km[np.where(masked_ds_rem_9km == 0)] = np.nan
    masked_ds_rem_9km = masked_ds_rem_9km.squeeze()

    smap_masked_ds_rem_9km_all.append(masked_ds_rem_9km)

smap_masked_ds_rem_9km_all = np.asarray(smap_masked_ds_rem_9km_all)
# masked_ds_rem_9km_all[masked_ds_rem_9km_all >= 0.5] = np.nan

# 400 m (GLDAS)
smap_masked_ds_rem_400m_all = []
for n in range(smap_400m_data_stack_rem.shape[0]):
    sub_window_rem_400m = Window(col_rem_400m_ind[0], row_rem_400m_ind[0], len(col_rem_400m_ind), len(row_rem_400m_ind))
    kwargs_400m_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_400m),
                      'height': len(lat_world_ease_400m), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(400.358009339824, 0.0, -17367530.44516138, 0.0, -400.358009339824, 7314540.79258289)}
    smap_sm_rem_400m_output = sub_n_reproj(smap_400m_data_stack_rem[n, :, :], kwargs_400m_sub, sub_window_rem_400m, output_crs)

    masked_ds_rem_400m, mask_transform_ds_rem_400m = mask(dataset=smap_sm_rem_400m_output, shapes=crop_shape_rem, crop=True)
    masked_ds_rem_400m[np.where(masked_ds_rem_400m == 0)] = np.nan
    masked_ds_rem_400m = masked_ds_rem_400m.squeeze()

    smap_masked_ds_rem_400m_all.append(masked_ds_rem_400m)

smap_masked_ds_rem_400m_all = np.asarray(smap_masked_ds_rem_400m_all)

# 400 m (ERA5)
smap_masked_ds_rem_400m_all_era = []
for n in range(smap_400m_data_stack_rem_era.shape[0]):
    sub_window_rem_400m = Window(col_rem_400m_ind[0], row_rem_400m_ind[0], len(col_rem_400m_ind), len(row_rem_400m_ind))
    kwargs_400m_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_400m),
                      'height': len(lat_world_ease_400m), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(400.358009339824, 0.0, -17367530.44516138, 0.0, -400.358009339824, 7314540.79258289)}
    smap_sm_rem_400m_output = sub_n_reproj(smap_400m_data_stack_rem_era[n, :, :], kwargs_400m_sub, sub_window_rem_400m, output_crs)

    masked_ds_rem_400m, mask_transform_ds_rem_400m = mask(dataset=smap_sm_rem_400m_output, shapes=crop_shape_rem, crop=True)
    masked_ds_rem_400m[np.where(masked_ds_rem_400m == 0)] = np.nan
    masked_ds_rem_400m = masked_ds_rem_400m.squeeze()

    smap_masked_ds_rem_400m_all_era.append(masked_ds_rem_400m)

smap_masked_ds_rem_400m_all_era = np.asarray(smap_masked_ds_rem_400m_all_era)


# Calculate the 7-day averaged maps
smap_1km_size = smap_masked_ds_rem_1km_all.shape
smap_400m_size = smap_masked_ds_rem_400m_all.shape
smap_masked_ds_rem_1km_avg = np.reshape(smap_masked_ds_rem_1km_all[:28, :, :], (7, 4, smap_1km_size[1], smap_1km_size[2]))
smap_masked_ds_rem_1km_avg = np.nanmean(smap_masked_ds_rem_1km_avg, axis=0)
smap_masked_ds_rem_9km_avg = np.reshape(smap_masked_ds_rem_9km_all[:28, :, :], (7, 4, smap_1km_size[1], smap_1km_size[2]))
smap_masked_ds_rem_9km_avg = np.nanmean(smap_masked_ds_rem_9km_avg, axis=0)
smap_masked_ds_rem_400m_avg = np.reshape(smap_masked_ds_rem_400m_all[:28, :, :], (7, 4, smap_400m_size[1], smap_400m_size[2]))
smap_masked_ds_rem_400m_avg = np.nanmean(smap_masked_ds_rem_400m_avg, axis=0)
smap_masked_ds_rem_400m_avg_era = np.reshape(smap_masked_ds_rem_400m_all_era[:28, :, :], (7, 4, smap_400m_size[1], smap_400m_size[2]))
smap_masked_ds_rem_400m_avg_era = np.nanmean(smap_masked_ds_rem_400m_avg_era, axis=0)

sm_masked_ds_rem_stack = list((smap_masked_ds_rem_400m_avg_era, smap_masked_ds_rem_400m_avg, smap_masked_ds_rem_1km_avg, smap_masked_ds_rem_9km_avg))


# 6.2.2.3 Make the subplot maps
feature_shp_rem = ShapelyFeature(Reader(path_shp_rem + '/' + shp_rem_file).geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
extent_rem = np.array(shapefile_rem.bounds)
extent_rem = extent_rem[[0, 2, 1, 3]]

# Shorter version
fig = plt.figure(figsize=(5, 4), facecolor='w', edgecolor='k', dpi=200)
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.9, hspace=0.1, wspace=0.1)
plt.rcParams['lines.linewidth'] = 0.5
for irow in range(4):
    for icol in range(4):
        # 1 km
        ax = fig.add_subplot(4, 4, irow*4+icol+1, projection=ccrs.PlateCarree())
        ax.add_feature(feature_shp_rem)
        img = ax.imshow(sm_masked_ds_rem_stack[icol][irow, :, :], origin='upper', vmin=0, vmax=0.15, cmap='turbo_r',
                   extent=extent_rem)
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
        elif irow == 3:
            gl.left_labels = False
            gl.bottom_labels = True
        else:
            gl.left_labels = False
            gl.bottom_labels = False
        if icol == 0 and irow == 3:
            gl.left_labels = True
            gl.bottom_labels = True
        gl.right_labels = False
        gl.top_labels = False
cbar_ax = fig.add_axes([0.9, 0.2, 0.015, 0.6])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both')
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=7, x=0.95, labelpad=-6)
cbar.ax.tick_params(labelsize=7)
cbar_ax.locator_params(nbins=4)
fig.text(0.12, 0.94, '400m (ERA5)', fontsize=8, fontweight='bold')
fig.text(0.31, 0.94, '400m (GLDAS)', fontsize=8, fontweight='bold')
fig.text(0.53, 0.94, '1km (GLDAS)', fontsize=8, fontweight='bold')
fig.text(0.79, 0.94, '9km', fontsize=8, fontweight='bold')
fig.text(0.02, 0.75, 'Aug 1-7', fontsize=7, fontweight='bold', rotation=90)
fig.text(0.02, 0.5, 'Aug 8-14', fontsize=7, fontweight='bold', rotation=90)
fig.text(0.02, 0.28, 'Aug 15-21', fontsize=7, fontweight='bold', rotation=90)
fig.text(0.02, 0.08, 'Aug 22-28', fontsize=7, fontweight='bold', rotation=90)
# cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=7, x=1.08, y=0.05, labelpad=-15)
plt.savefig(path_results + '/sm_comp_rem_1_new.png')
plt.close()




################################################################################################################################
# Subset the region of interest (SMAP 400 m)

# shp_txs_extent = [-123.8, -118.7, 36.7, 45.8]
shp_txs_extent = [-117, -109.9, 43.4, 49]

[lat_400m_txs, row_txs_400m_ind, lon_400m_txs, col_txs_400m_ind] = \
    coordtable_subset(lat_world_ease_400m, lon_world_ease_400m,
                      shp_txs_extent[3], shp_txs_extent[2], shp_txs_extent[1], shp_txs_extent[0])

df_row_txs_400m_ind = df_row_world_ease_400m_ind.iloc[row_txs_400m_ind, :]
df_col_txs_400m_ind = df_col_world_ease_400m_ind.iloc[col_txs_400m_ind, :]

df_row_txs_400m_ind_split = df_row_txs_400m_ind.groupby(by=['row_ind_tile'])
df_row_txs_400m_ind_group = [df_row_txs_400m_ind_split.get_group(x) for x in df_row_txs_400m_ind_split.groups]
df_col_txs_400m_ind_split = df_col_txs_400m_ind.groupby(by=['col_ind_tile'])
df_col_txs_400m_ind_group = [df_col_txs_400m_ind_split.get_group(x) for x in df_col_txs_400m_ind_split.groups]

tiles_num_row = pd.unique(df_row_txs_400m_ind['row_ind_tile'])
tiles_num_col = pd.unique(df_col_txs_400m_ind['col_ind_tile'])

tile_num = []
row_extent_all = []
col_extent_all = []
for irow in range(len(tiles_num_row)):
    for icol in range(len(tiles_num_col)):
        tile_num_single = tiles_num_row[irow] * 24 + tiles_num_col[icol] + 1
        row_extent = np.array(df_row_txs_400m_ind_group[irow]['row_ind_local'].tolist())[[0, -1]]
        col_extent = np.array(df_col_txs_400m_ind_group[icol]['col_ind_local'].tolist())[[0, -1]]

        tile_num.append(tile_num_single)
        row_extent_all.append(row_extent)
        col_extent_all.append(col_extent)
        del(tile_num_single, row_extent, col_extent)

# tiles_num = tiles_num_row[0] * 24 + tiles_num_col[0] + 1
# Convert from Lat/Lon coordinates to EASE grid projection meter units
transformer = Transformer.from_crs("epsg:4326", "epsg:6933", always_xy=True)
[lon_400m_txs_min, lat_400m_txs_max] = transformer.transform(lon_400m_txs[0], lat_400m_txs[0])

profile = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_400m_txs),
           'height': len(lat_400m_txs), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
           'transform': Affine(interdist_ease_400m, 0.0, lon_400m_txs_min-interdist_ease_400m/2,
                               0.0, -interdist_ease_400m, lat_400m_txs_max+interdist_ease_400m/2)}

for iyr in range(9, len(yearname)):
    os.makedirs(path_output + '/nw_usa/site_2/' + str(yearname[iyr]))


for iyr in range(9, len(yearname)):  # range(yearname):
    for idt in range(daysofyear[iyr]):
        smap_400m_stack = []
        for ite in range(len(tile_num)):
            tif_file_smap_400m_name = ('smap_sm_400m_' + str(yearname[iyr]) + str(idt+1).zfill(3) + '_T' +
                                       str(tile_num[ite]).zfill(3) + '.tif')
            tif_file_smap_400m = (path_smap_400m + str(yearname[iyr]) + '/T' + str(tile_num[ite]).zfill(3) + '/'
                                  + tif_file_smap_400m_name)
            src_tf = gdal.Open(tif_file_smap_400m)
            src_tf_arr = src_tf.ReadAsArray()[:, row_extent_all[ite][0]:row_extent_all[ite][-1]+1,
                         col_extent_all[ite][0]:col_extent_all[ite][-1]+1]
            src_tf_arr_avg = np.nanmean(src_tf_arr, axis=0)
            smap_400m_stack.append(src_tf_arr_avg)
            del(src_tf_arr_avg, src_tf, src_tf_arr)

        smap_400m_stack_div = [smap_400m_stack[i:i + len(tiles_num_col)] for i in range(len(tiles_num_row))]
        smap_400m_stack_div_by_row = [np.hstack(smap_400m_stack_div[x]) for x in range(len(smap_400m_stack_div))]
        smap_400m_stack_complete = np.vstack(smap_400m_stack_div_by_row)
        smap_400m_stack_complete = np.expand_dims(smap_400m_stack_complete, axis=0)

        tif_file_smap_400m_name_output = tif_file_smap_400m_name.split('.')[0][:-5]
        dst_writer = rasterio.open(path_output + '/nw_usa/site_2/' + str(yearname[iyr]) + '/' +
                                   tif_file_smap_400m_name_output + '.tif', 'w', **profile)
        dst_writer.write(smap_400m_stack_complete)
        dst_writer = None

        print(tif_file_smap_400m_name_output)
        del(smap_400m_stack, smap_400m_stack_div, smap_400m_stack_div_by_row, smap_400m_stack_complete,
            tif_file_smap_400m, tif_file_smap_400m_name, tif_file_smap_400m_name_output)




################################################################################################################################
# 6.3.1  (SoilSCAPE)
path_shp_sca = path_gis_data + '/watershed_boundary/'
os.chdir(path_shp_sca)
shp_sca_file = "soilscape.shp"
shp_sca_ds = ogr.GetDriverByName("ESRI Shapefile").Open(shp_sca_file, 0)
shp_sca_extent = list(shp_sca_ds.GetLayer().GetExtent())


# 6.3.1.1 SMAP
#Load and subset the region of Middle Colorado RB (SMAP 9 km)
[lat_9km_sca, row_sca_9km_ind, lon_9km_sca, col_sca_9km_ind] = \
    coordtable_subset(lat_world_ease_9km, lon_world_ease_9km,
                      shp_sca_extent[3], shp_sca_extent[2], shp_sca_extent[1], shp_sca_extent[0])
[lat_1km_sca, row_sca_1km_ind, lon_1km_sca, col_sca_1km_ind] = \
    coordtable_subset(lat_world_ease_1km, lon_world_ease_1km,
                      shp_sca_extent[3], shp_sca_extent[2], shp_sca_extent[1], shp_sca_extent[0])
[lat_400m_sca, row_sca_400m_ind, lon_400m_sca, col_sca_400m_ind] = \
    coordtable_subset(lat_world_ease_400m, lon_world_ease_400m,
                      shp_sca_extent[3], shp_sca_extent[2], shp_sca_extent[1], shp_sca_extent[0])

row_sca_9km_ind_dis = row_world_ease_1km_from_9km_ind[row_sca_1km_ind]
row_sca_9km_ind_dis_unique = np.unique(row_sca_9km_ind_dis)
row_sca_9km_ind_dis_zero = row_sca_9km_ind_dis - row_sca_9km_ind_dis[0]
col_sca_9km_ind_dis = col_world_ease_1km_from_9km_ind[col_sca_1km_ind]
col_sca_9km_ind_dis_unique = np.unique(col_sca_9km_ind_dis)
col_sca_9km_ind_dis_zero = col_sca_9km_ind_dis - col_sca_9km_ind_dis[0]

col_meshgrid_9km, row_meshgrid_9km = np.meshgrid(col_sca_9km_ind_dis_zero, row_sca_9km_ind_dis_zero)
col_meshgrid_9km = col_meshgrid_9km.reshape(1, -1).squeeze()
row_meshgrid_9km = row_meshgrid_9km.reshape(1, -1).squeeze()

# Load and subset SMAP 9 km SM
year_plt = [2022]
month_plt = list([7])
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
                                           row_sca_9km_ind_dis_unique[0]:row_sca_9km_ind_dis_unique[-1] + 1,
                                           col_sca_9km_ind_dis_unique[0]:col_sca_9km_ind_dis_unique[-1] + 1, days_begin-1:days_end]  # AM
        smap_9km_load_2 = f_read_smap_9km[varname_list_smap_9km[1]][
                                           row_sca_9km_ind_dis_unique[0]:row_sca_9km_ind_dis_unique[-1] + 1,
                                           col_sca_9km_ind_dis_unique[0]:col_sca_9km_ind_dis_unique[-1] + 1, days_begin-1:days_end]  # PM

        smap_9km_load_1_disagg = np.array([smap_9km_load_1[row_meshgrid_9km[x], col_meshgrid_9km[x], :]
                                      for x in range(len(col_meshgrid_9km))])
        smap_9km_load_1_disagg = smap_9km_load_1_disagg.reshape((len(row_sca_9km_ind_dis), len(col_sca_9km_ind_dis),
                                                                 smap_9km_load_1.shape[2]))
        smap_9km_load_2_disagg = np.array([smap_9km_load_2[row_meshgrid_9km[x], col_meshgrid_9km[x], :]
                                      for x in range(len(col_meshgrid_9km))])
        smap_9km_load_2_disagg = smap_9km_load_2_disagg.reshape((len(row_sca_9km_ind_dis), len(col_sca_9km_ind_dis),
                                                                 smap_9km_load_2.shape[2]))
        f_read_smap_9km.close()

        smap_9km_mean_1 = np.nanmean(np.stack((smap_9km_load_1_disagg, smap_9km_load_2_disagg), axis=3), axis=3)
        smap_9km_mean_1_allyear.append(smap_9km_mean_1)

        del(smap_9km_load_1, smap_9km_mean_1, smap_9km_load_1_disagg, smap_9km_load_2_disagg)
        print(monthname[month_plt[imo]-1])

smap_9km_data_stack_sca = np.squeeze(np.array(smap_9km_mean_1_allyear))
del(smap_9km_mean_1_allyear)
smap_9km_data_stack_sca = np.transpose(smap_9km_data_stack_sca, (2, 0, 1))


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
            src_tf_arr = src_tf.ReadAsArray()[:, row_sca_1km_ind[0]:row_sca_1km_ind[-1]+1,
                         col_sca_1km_ind[0]:col_sca_1km_ind[-1]+1]
            smap_1km_load_1 = np.nanmean(src_tf_arr, axis=0)
            smap_1km_load_1_stack.append(smap_1km_load_1)

            print(str_date)
            del(src_tf_arr, smap_1km_load_1)

        smap_1km_load_1_stack = np.stack(smap_1km_load_1_stack)
        smap_1km_mean_1_allyear.append(smap_1km_load_1_stack)
        del(smap_1km_load_1_stack)

smap_1km_data_stack_sca = np.squeeze(np.array(smap_1km_mean_1_allyear))
del(smap_1km_mean_1_allyear)



#Load and subset the region of Middle Colorado RB (SMAP 400 m)

df_row_sca_400m_ind = df_row_world_ease_400m_ind.iloc[row_sca_400m_ind, :]
df_col_sca_400m_ind = df_col_world_ease_400m_ind.iloc[col_sca_400m_ind, :]

tiles_num_row = pd.unique(df_row_sca_400m_ind['row_ind_tile'])
tiles_num_col = pd.unique(df_col_sca_400m_ind['col_ind_tile'])
tiles_num = tiles_num_row * 24 + tiles_num_col + 1

row_sca_400m_ind_local = df_row_sca_400m_ind['row_ind_local'].tolist()
col_sca_400m_ind_local = df_col_sca_400m_ind['col_ind_local'].tolist()
divider = np.where(np.array(col_sca_400m_ind_local) == 0)[0]
col_sca_400m_ind_local_div = np.split(col_sca_400m_ind_local, divider)

smap_400m_mean_1_allyear = []
for iyr in year_plt:  # range(yearname):
    for imo in range(len(month_plt)):  # range(len(monthname)):
        smap_400m_load_1_stack = []
        for idt in range(days_begin, days_end+1):
            smap_tiles_all = []
            for ite in range(len(tiles_num)):
                str_date = str(iyr) + '-' + monthname[month_plt[imo]-1] + '-' + str(idt).zfill(2)
                str_doy = str(datetime.datetime.strptime(str_date, '%Y-%m-%d').date().timetuple().tm_yday)
                tif_file_smap_400m = (path_smap_400m + str(iyr) + '/T' + str(tiles_num[ite]).zfill(3) + '/smap_sm_400m_' + str(iyr)
                                      + str_doy.zfill(3) + '_T' + str(tiles_num[ite]).zfill(3) + '.tif')
                src_tf = gdal.Open(tif_file_smap_400m)
                src_tf_arr = src_tf.ReadAsArray()[:, row_sca_400m_ind_local[0]:row_sca_400m_ind_local[-1]+1,
                             col_sca_400m_ind_local_div[ite][0]:col_sca_400m_ind_local_div[ite][-1]+1]
                smap_400m_load_1 = np.nanmean(src_tf_arr, axis=0)
                smap_tiles_all.append(smap_400m_load_1)
                del(smap_400m_load_1, src_tf_arr)

            smap_tiles_combine = np.concatenate(smap_tiles_all, axis=1)
            smap_400m_load_1_stack.append(smap_tiles_combine)

            print(str_date)

        smap_400m_load_1_stack = np.stack(smap_400m_load_1_stack)
        smap_400m_mean_1_allyear.append(smap_400m_load_1_stack)
        del(smap_400m_load_1_stack)

smap_400m_data_stack_sca = np.squeeze(np.array(smap_400m_mean_1_allyear))
del(smap_400m_mean_1_allyear)


with h5py.File(path_model_evaluation + '/smap_sca_sm.hdf5', 'w') as f:
    f.create_dataset('smap_9km_data_stack_sca', data=smap_9km_data_stack_sca)
    f.create_dataset('smap_1km_data_stack_sca', data=smap_1km_data_stack_sca)
    f.create_dataset('smap_400m_data_stack_sca', data=smap_400m_data_stack_sca)
f.close()

# Read the map data
f_read = h5py.File(path_model_evaluation + "/smap_sca_sm.hdf5", "r")
varname_read_list = ['smap_9km_data_stack_sca', 'smap_1km_data_stack_sca', 'smap_400m_data_stack_sca']
for x in range(len(varname_read_list)):
    var_obj = f_read[varname_read_list[x]][()]
    exec(varname_read_list[x] + '= var_obj')
    del(var_obj)
f_read.close()



# 6.3.2 Subplot maps
output_crs = 'EPSG:4326'
shapefile_sca = fiona.open(path_shp_sca + '/' + shp_sca_file, 'r')
crop_shape_sca = [feature["geometry"] for feature in shapefile_sca]

# 6.3.2.1 Subset and reproject the SMAP SM data at watershed
# 1 km
smap_masked_ds_sca_1km_all = []
for n in range(smap_1km_data_stack_sca.shape[0]):
    sub_window_sca_1km = Window(col_sca_1km_ind[0], row_sca_1km_ind[0], len(col_sca_1km_ind), len(row_sca_1km_ind))
    kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                      'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
    smap_sm_sca_1km_output = sub_n_reproj(smap_1km_data_stack_sca[n, :, :], kwargs_1km_sub, sub_window_sca_1km, output_crs)

    masked_ds_sca_1km, mask_transform_ds_sca_1km = mask(dataset=smap_sm_sca_1km_output, shapes=crop_shape_sca, crop=True)
    masked_ds_sca_1km[np.where(masked_ds_sca_1km == 0)] = np.nan
    masked_ds_sca_1km = masked_ds_sca_1km.squeeze()

    smap_masked_ds_sca_1km_all.append(masked_ds_sca_1km)

smap_masked_ds_sca_1km_all = np.asarray(smap_masked_ds_sca_1km_all)


# 9 km
smap_masked_ds_sca_9km_all = []
for n in range(smap_9km_data_stack_sca.shape[0]):
    sub_window_sca_9km = Window(col_sca_1km_ind[0], row_sca_1km_ind[0], len(col_sca_1km_ind), len(row_sca_1km_ind))
    kwargs_9km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                      'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956, 7314540.79258289)}
    smap_sm_sca_9km_output = sub_n_reproj(smap_9km_data_stack_sca[n, :, :], kwargs_9km_sub, sub_window_sca_9km, output_crs)

    masked_ds_sca_9km, mask_transform_ds_sca_9km = mask(dataset=smap_sm_sca_9km_output, shapes=crop_shape_sca, crop=True)
    masked_ds_sca_9km[np.where(masked_ds_sca_9km == 0)] = np.nan
    masked_ds_sca_9km = masked_ds_sca_9km.squeeze()

    smap_masked_ds_sca_9km_all.append(masked_ds_sca_9km)

smap_masked_ds_sca_9km_all = np.asarray(smap_masked_ds_sca_9km_all)
# masked_ds_sca_9km_all[masked_ds_sca_9km_all >= 0.5] = np.nan

# 400 m
smap_masked_ds_sca_400m_all = []
for n in range(smap_400m_data_stack_sca.shape[0]):
    sub_window_sca_400m = Window(col_sca_400m_ind[0], row_sca_400m_ind[0], len(col_sca_400m_ind), len(row_sca_400m_ind))
    kwargs_400m_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_400m),
                      'height': len(lat_world_ease_400m), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                      'transform': Affine(400.358009339824, 0.0, -17367530.44516138, 0.0, -400.358009339824, 7314540.79258289)}
    smap_sm_sca_400m_output = sub_n_reproj(smap_400m_data_stack_sca[n, :, :], kwargs_400m_sub, sub_window_sca_400m, output_crs)

    masked_ds_sca_400m, mask_transform_ds_sca_400m = mask(dataset=smap_sm_sca_400m_output, shapes=crop_shape_sca, crop=True)
    masked_ds_sca_400m[np.where(masked_ds_sca_400m == 0)] = np.nan
    masked_ds_sca_400m = masked_ds_sca_400m.squeeze()

    smap_masked_ds_sca_400m_all.append(masked_ds_sca_400m)

smap_masked_ds_sca_400m_all = np.asarray(smap_masked_ds_sca_400m_all)


# Calculate the 7-day averaged maps
smap_1km_size = smap_masked_ds_sca_1km_all.shape
smap_400m_size = smap_masked_ds_sca_400m_all.shape
smap_masked_ds_sca_1km_avg = np.reshape(smap_masked_ds_sca_1km_all[:28, :, :], (7, 4, smap_1km_size[1], smap_1km_size[2]))
smap_masked_ds_sca_1km_avg = np.nanmean(smap_masked_ds_sca_1km_avg, axis=0)
smap_masked_ds_sca_9km_avg = np.reshape(smap_masked_ds_sca_9km_all[:28, :, :], (7, 4, smap_1km_size[1], smap_1km_size[2]))
smap_masked_ds_sca_9km_avg = np.nanmean(smap_masked_ds_sca_9km_avg, axis=0)
smap_masked_ds_sca_400m_avg = np.reshape(smap_masked_ds_sca_400m_all[:28, :, :], (7, 4, smap_400m_size[1], smap_400m_size[2]))
smap_masked_ds_sca_400m_avg = np.nanmean(smap_masked_ds_sca_400m_avg, axis=0)

sm_masked_ds_sca_stack = list((smap_masked_ds_sca_400m_avg, smap_masked_ds_sca_1km_avg, smap_masked_ds_sca_9km_avg))


# 6.3.2.3 Make the subplot maps
feature_shp_sca = ShapelyFeature(Reader(path_shp_sca + '/' + shp_sca_file).geometries(),
                                ccrs.PlateCarree(), edgecolor='black', facecolor='none')
extent_sca = np.array(shapefile_sca.bounds)
extent_sca = extent_sca[[0, 2, 1, 3]]


# Shorter version
fig = plt.figure(figsize=(6, 4), facecolor='w', edgecolor='k', dpi=200)
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.9, hspace=0.1, wspace=0.1)
plt.rcParams['lines.linewidth'] = 0.5
for irow in range(4):
    for icol in range(3):
        # 1 km
        ax = fig.add_subplot(4, 3, irow*3+icol+1, projection=ccrs.PlateCarree())
        ax.add_feature(feature_shp_sca)
        img = ax.imshow(sm_masked_ds_sca_stack[icol][irow, :, :], origin='upper', vmin=0, vmax=0.3, cmap='turbo_r',
                   extent=extent_sca)
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
        elif irow == 3:
            gl.left_labels = False
            gl.bottom_labels = True
        else:
            gl.left_labels = False
            gl.bottom_labels = False
        if icol == 0 and irow == 3:
            gl.left_labels = True
            gl.bottom_labels = True
        gl.right_labels = False
        gl.top_labels = False
cbar_ax = fig.add_axes([0.9, 0.2, 0.015, 0.6])
cbar = fig.colorbar(img, cax=cbar_ax, extend='both')
cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=7, x=0.95, labelpad=1)
cbar.ax.tick_params(labelsize=7)
cbar_ax.locator_params(nbins=6)
fig.text(0.17, 0.94, 'SMAP 400m', fontsize=9, fontweight='bold')
fig.text(0.45, 0.94, 'SMAP 1km', fontsize=9, fontweight='bold')
fig.text(0.7, 0.94, 'SMAP 9km', fontsize=9, fontweight='bold')
fig.text(0.02, 0.75, 'July 1-7', fontsize=7, fontweight='bold', rotation=90)
fig.text(0.02, 0.5, 'July 8-14', fontsize=7, fontweight='bold', rotation=90)
fig.text(0.02, 0.28, 'July 15-21', fontsize=7, fontweight='bold', rotation=90)
fig.text(0.02, 0.08, 'July 22-28', fontsize=7, fontweight='bold', rotation=90)
# cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=7, x=1.08, y=0.05, labelpad=-15)
plt.savefig(path_results + '/sm_comp_sca_1.png')
plt.close()

################################################################################################################################
# 7. Feature importance
feature_imp_read_all = sorted(glob.glob(path_results + '/feature_importance/*.csv'))
feature_imp_read = [feature_imp_read_all[2], feature_imp_read_all[1], feature_imp_read_all[3], feature_imp_read_all[0]]
feature_imp_first = pd.read_csv(feature_imp_read[0], index_col=0)
ind_feature_imp_first = feature_imp_first.index.tolist()
columns = ['R^2', 'ubRMSE', 'MAE']

feature_imp_all = []
for ife in range(len(feature_imp_read)):
    feature_imp_1file = pd.read_csv(feature_imp_read[ife], index_col=0)
    feature_imp_1file = feature_imp_1file.reindex(ind_feature_imp_first)
    feature_imp_all.append(feature_imp_1file)
    del (feature_imp_1file)

feature_imp_all = pd.concat(feature_imp_all, axis=1)
# feature_imp_all.columns = columns

# Plot components together
feature_imp_all_sub = feature_imp_all.iloc[:, [1, 5, 9]]
feature_imp_all_sub.columns = ['400m', '1km', '9km']
feature_imp_all_sub = feature_imp_all_sub.drop(['BIEBRZA', 'Ru_CFR', 'TERENO'])
ax = feature_imp_all_sub.plot.barh(figsize=(20, 10), color={"400m": "orange", "1km": "blue", "9km": "green"})
ax.legend(loc='lower right')
ax.grid('on', linestyle='--')
ax.tick_params(axis='both', labelsize=12)
ax.set_title(r'$\mathrm{R}^2$', fontdict={'fontsize': 20})
plt.subplots_adjust(left=0.24, right=0.9, top=0.9, bottom=0.1)
plt.gca().invert_yaxis()
plt.rcParams["figure.dpi"] = 200
plt.savefig(path_results + '/validation/feature_importance_r2_new.png')
plt.close()


feature_imp_all_sub = feature_imp_all.iloc[:, [2, 6, 10]]
feature_imp_all_sub.columns = ['400m', '1km', '9km']
feature_imp_all_sub = feature_imp_all_sub.drop(['BIEBRZA', 'Ru_CFR', 'TERENO'])
ax = feature_imp_all_sub.plot.barh(figsize=(20, 10), color={"400m": "orange", "1km": "blue", "9km": "green"})
ax.legend(loc='lower right')
ax.grid('on', linestyle='--')
ax.tick_params(axis='both', labelsize=12)
ax.set_title('$\mathregular{ubRMSE(m^3/m^3)}$', fontdict={'fontsize': 20})
plt.subplots_adjust(left=0.24, right=0.9, top=0.95, bottom=0.05)
plt.gca().invert_yaxis()
plt.rcParams["figure.dpi"] = 200
plt.savefig(path_results + '/validation/feature_importance_rmse_new.png')
plt.close()


feature_imp_all_sub = feature_imp_all.iloc[:, [3, 7, 11]]
feature_imp_all_sub.columns = ['400m', '1km', '9km']
feature_imp_all_sub = feature_imp_all_sub.drop(['BIEBRZA', 'Ru_CFR', 'TERENO'])
ax = feature_imp_all_sub.plot.barh(figsize=(20, 10), color={"400m": "orange", "1km": "blue", "9km": "green"})
ax.legend(loc='lower right')
ax.grid('on', linestyle='--')
ax.tick_params(axis='both', labelsize=12)
ax.set_title('$\mathregular{MAE(m^3/m^3)}$', fontdict={'fontsize': 20})
plt.subplots_adjust(left=0.24, right=0.9, top=0.9, bottom=0.1)
plt.gca().invert_yaxis()
plt.rcParams["figure.dpi"] = 200
plt.savefig(path_results + '/validation/feature_importance_mae_new.png')
plt.close()

feature_imp_all_sub = feature_imp_all.iloc[:, [12, 13, 14, 15]]
feature_imp_all_sub.columns = ['In situ', '400m', '1km', '9km']
feature_imp_all_sub = feature_imp_all_sub.drop(['BIEBRZA', 'Ru_CFR', 'TERENO'])
ax = feature_imp_all_sub.plot.barh(figsize=(20, 10), width=0.8,
                                   color={"In situ": "black", "400m": "orange", "1km": "blue", "9km": "green"})
ax.legend(loc='lower right')
ax.grid('on', linestyle='--')
ax.tick_params(axis='both', labelsize=12)
ax.set_title('$\mathregular{SSD(m^3/m^3)}$', fontdict={'fontsize': 20})
plt.subplots_adjust(left=0.24, right=0.9, top=0.9, bottom=0.1)
plt.gca().invert_yaxis()
plt.rcParams["figure.dpi"] = 200
plt.savefig(path_results + '/validation/feature_importance_ssd_new.png')
plt.close()

