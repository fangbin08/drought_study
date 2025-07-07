import os
import numpy as np
import matplotlib.ticker as mticker
import fiona
# import rasterio.mask
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import h5py
import netCDF4 as nc
import calendar
import datetime
import glob
import pandas as pd
import matplotlib.pyplot as plt
# import xlrd
import gdal
import itertools
import fiona
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile
from rasterio.transform import from_origin, Affine
from rasterio.windows import Window
from rasterio.crs import CRS
import netCDF4
from pyproj import Transformer
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scipy import stats
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# Ignore runtime warning
import warnings
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
# (Function 1.1) Subset the coordinates table of desired area (revised)

def coordtable_subset_rev(lat_input, lon_input, lat_extent_max, lat_extent_min, lon_extent_max, lon_extent_min):
    lat_output_init = lat_input[np.where((lat_input <= lat_extent_max) & (lat_input >= lat_extent_min))]
    if len(lat_output_init) <= 1:
        row_output_ind = np.array([np.argmin(np.absolute(lat_input - lat_extent_max))-1,
                                   np.argmin(np.absolute(lat_input - lat_extent_max)),
                                   np.argmin(np.absolute(lat_input - lat_extent_max))+1])
        lat_output = lat_input[row_output_ind]
    else:
        lat_output = lat_output_init
        row_output_ind = np.squeeze(np.array(np.where((lat_input <= lat_extent_max) & (lat_input >= lat_extent_min))))

    lon_output_init = lon_input[np.where((lon_input <= lon_extent_max) & (lon_input >= lon_extent_min))]
    if len(lon_output_init) <= 1:
        col_output_ind = np.array([np.argmin(np.absolute(lon_input - lon_extent_max))-1,
                                   np.argmin(np.absolute(lon_input - lon_extent_max)),
                                   np.argmin(np.absolute(lon_input - lon_extent_max))+1])
        lon_output = lon_input[col_output_ind]
    else:
        lon_output = lon_output_init
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

####################################################################################################################################
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

########################################################################################################################

# 0. Input variables
# Specify file paths
# Path of current workspace
path_workspace = '/Users/binfang/Documents/SMAP_Project/smap_codes'
# Path of source LTDR NDVI data
path_ltdr = '/Volumes/MyPassport/SMAP_Project/Datasets/LTDR/Ver5'
# Path of Land mask
path_lmask = '/Volumes/Elements/Datasets/Lmask'
# Path of model data
path_model = '/Users/binfang/Downloads/Processing/model_data'
# Path of source MODIS data
path_modis = '/Volumes/MyPassport/SMAP_Project/NewData/MODIS/HDF'
# Path of source output MODIS data
path_modis_op = '/Volumes/MyPassport/SMAP_Project/NewData/MODIS/Output'
# Path of MODIS data for SM downscaling model input
path_modis_model_ip = '/Volumes/MyPassport/SMAP_Project/NewData/MODIS/Model_Input'
# Path of SM model output
path_model_op = '/Volumes/MyPassport/SMAP_Project/Datasets/SMAP_ds/Model_Output'
# Path of downscaled SM
path_smap_sm_ds = '/Volumes/MyPassport/SMAP_Project/Datasets/MODIS/Downscale'
path_smap_org = '/Volumes/Seagate_6TB/SMAP'
path_smap_anc = '/Users/binfang/Downloads/Processing/smap_output/smap_ancillary_data'
path_smap_anc_1km = '/Users/binfang/Downloads/Processing/smap_output/smap_ancillary_data_1km'

# Path of 1 km/9 km SMAP SM
path_smap = '/Volumes/Elements/Datasets/SMAP'
path_smos = '/Volumes/Elements/Datasets/SMOS'
path_smap_9km = '/Volumes/MyPassport/SMAP_Project/Datasets/SMAP'
path_smap_400m = '/Volumes/Elements2/SMAP/SM_downscaled_gldas/'
# Path of GPM
path_gpm = '/Volumes/Elements/Datasets/GPM'
# Path of processed data
path_processed = '/Volumes/MyPassport/SMAP_Project/Datasets/processed_data'
# Path of SMAP Mat-files
path_matfile = '/Volumes/MyPassport/SMAP_Project/Datasets/CONUS'
# Path of shapefile
path_shp = '/Users/binfang/Downloads/Processing/shapefiles'
# Path of SMAP output
path_output = '/Users/binfang/Downloads/Processing/smap_output'
# Path of GIS
path_gis = '/Users/binfang/Documents/SMAP_Project/data/gis_data'
path_results = '/Users/binfang/Documents/SMAP_Project/results/results_220107'

lst_folder = '/MYD11A1/'
ndvi_folder = '/MYD13A2/'
smap_sm_9km_name = ['smap_sm_9km_am', 'smap_sm_9km_pm']

# Generate a sequence of string between start and end dates (Year + DOY)
start_date = '2010-01-01'
end_date = '2025-12-31'
year = 2025 - 2010 + 1

start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d').date()
end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d').date()
delta_date = end_date - start_date

date_seq = []
date_seq_doy = []
for i in range(delta_date.days + 1):
    date_str = start_date + datetime.timedelta(days=i)
    date_seq.append(date_str.strftime('%Y%m%d'))
    date_seq_doy.append(str(date_str.timetuple().tm_year) + str(date_str.timetuple().tm_yday).zfill(3))

# Count how many days for a specific year
yearname = np.linspace(2010, 2025, 16, dtype='int')
monthnum = np.linspace(1, 12, 12, dtype='int')
monthname = np.arange(1, 13)
monthname = [str(i).zfill(2) for i in monthname]

daysofyear = []
for idt in range(len(yearname)):
    if idt == 0:
        # f_date = datetime.date(yearname[idt], monthnum[3], 1)
        f_date = datetime.date(yearname[idt], monthnum[0], 1)
        l_date = datetime.date(yearname[idt], monthnum[-1], 31)
        delta_1y = l_date - f_date
        daysofyear.append(delta_1y.days + 1)
    else:
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


# Load in geo-location parameters
os.chdir(path_workspace)
f = h5py.File("ds_parameters.hdf5", "r")
varname_list = ['lat_world_max', 'lat_world_min', 'lon_world_max', 'lon_world_min', 'interdist_ease_9km',
                'lat_conus_ease_9km', 'lon_conus_ease_9km', 'lat_conus_ease_1km', 'lon_conus_ease_1km',
                'lat_world_ease_9km', 'lon_world_ease_9km', 'lat_world_ease_1km', 'lon_world_ease_1km',
                'lat_world_ease_25km', 'lon_world_ease_25km', 'lat_world_ease_400m', 'lon_world_ease_400m',
                'row_world_ease_400m_ind', 'col_world_ease_400m_ind', 'interdist_ease_400m',
                'interdist_ease_25km', 'lon_conus_ease_25km', 'lat_conus_ease_25km', 'row_conus_ease_1km_ind',
                'col_conus_ease_1km_ind', 'lat_world_geo_10km', 'lon_world_geo_10km']

for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)

varname_list_2 = ['lat_conus_max', 'lat_conus_min', 'lon_conus_max', 'lon_conus_min',
                'lat_conus_ease_1km', 'lon_conus_ease_1km', 'row_conus_ease_9km_ind', 'col_conus_ease_9km_ind',
                'row_conus_ease_1km_ind', 'col_conus_ease_1km_ind',
                'row_conus_ease_1km_from_9km_ind', 'col_conus_ease_1km_from_9km_ind',
                'row_conus_ease_9km_from_1km_ext33km_ind', 'col_conus_ease_9km_from_1km_ext33km_ind',
                'lat_conus_ease_9km', 'lon_conus_ease_9km', 'lat_conus_ease_12_5km', 'lon_conus_ease_12_5km',
                'row_conus_ease_1km_from_12_5km_ind', 'col_conus_ease_1km_from_12_5km_ind',
                'row_world_ease_1km_from_9km_ind', 'col_world_ease_1km_from_9km_ind']

for x in range(len(varname_list_2)):
    var_obj = f[varname_list_2[x]][()]
    exec(varname_list_2[x] + '= var_obj')
    del(var_obj)
f.close()

# Convert the 1 km from 9 km/25 km match table files to 1-d linear
col_meshgrid_from_9km, row_meshgrid_from_9km = np.meshgrid(col_world_ease_1km_from_9km_ind, row_world_ease_1km_from_9km_ind)
col_meshgrid_from_9km = col_meshgrid_from_9km.reshape(1, -1)
row_meshgrid_from_9km = row_meshgrid_from_9km.reshape(1, -1)

# Load in mask data
lmask_src = gdal.Open(path_lmask + '/modis_MOD44W_landmask.tif')
lmask = np.squeeze(lmask_src.ReadAsArray())
lmask[lmask == 4] = np.nan
lmask[lmask == 0] = np.nan
lmask[~np.isnan(lmask)] = 1
# lmask[12750:, 13000:] = np.nan
lmask = lmask.reshape(1, -1)
smap_sm_1km_1file_ind = np.where(~np.isnan(lmask))[1]

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

viirs_mat_fill = np.empty([2, 4060, 3615], dtype='float32')
viirs_mat_fill[:] = np.nan


########################################################################################################################
# 1. Subset SMAP 1 km/9 km data
# shapefile = fiona.open(path_shp + '/Pakistan_Study Area_Bounding_Box/Pakistan_Study Area_Bounding_Box.shp', 'r')
# shapefile = fiona.open(path_shp + '/sd6_box/sd6_1mile_bounding_box.shp', 'r')
# shapefile = fiona.open(path_shp + '/bounding_box_7_regions/bounding box 7 regions.shp', 'r')
shapefile = fiona.open(path_shp + '/BERAMBADI WATERSHED/Berambadi_Watershed_reprj.shp', 'r')
crop_shape = [feature["geometry"] for feature in shapefile]
shp_extent = list(shapefile.bounds)

# Extent: [W, S, E, N]
# shp_extent = [0, 46.2, 12, 52.2]
# shp_extent = [26.9, -30.7, 29.5, -28.5]
# shp_extent = [92, 5.5, 109.7, 28.5]
# shp_extent = [-89, 24, -74, 40]
# shp_extent = [-74, -40, -69, -31]
# shp_extent = [-125, 24, -66, 50]
shp_extent = [-107.16232, -105.24800, 36.97396, 38.41805] # Colorado
shp_extent = [-119.1, -117.94, 33.98, 34.5] # Los Angeles, CA
shp_extent = [66.5, 71.22, 23.59, 28.58] # Pakistan
shp_extent = [-77, -75, 37.9, 39.7] # MD eastern shore
shp_extent = [-118.88, -117.8, 33.82, 34.3] # Palisades, CA
shp_extent = [-124.5, -114.1, 32.5, 42] # CA
shp_extent = [shp_extent[0], shp_extent[2], shp_extent[1], shp_extent[3]]


lat_sub_max = shp_extent[3]
lat_sub_min = shp_extent[1]
lon_sub_max = shp_extent[2]
lon_sub_min = shp_extent[0]

output_crs = 'EPSG:4326'
[lat_sub_ease_1km, row_sub_ease_1km_ind, lon_sub_ease_1km, col_sub_ease_1km_ind] = coordtable_subset_V2\
    (lat_world_ease_1km, lon_world_ease_1km, lat_sub_max, lat_sub_min, lon_sub_max, lon_sub_min)
[lat_sub_ease_9km, row_sub_ease_9km_ind, lon_sub_ease_9km, col_sub_ease_9km_ind] = coordtable_subset_V2\
    (lat_world_ease_9km, lon_world_ease_9km, lat_sub_max, lat_sub_min, lon_sub_max, lon_sub_min)
[lat_sub_geo_10km, row_sub_geo_10km_ind, lon_sub_geo_10km, col_sub_geo_10km_ind] = coordtable_subset_V2\
    (lat_world_geo_10km, lon_world_geo_10km, lat_sub_max, lat_sub_min, lon_sub_max, lon_sub_min)
[lat_sub_ease_25km, row_sub_ease_25km_ind, lon_sub_ease_25km, col_sub_ease_25km_ind] = coordtable_subset_V2\
    (lat_world_ease_25km, lon_world_ease_25km, lat_sub_max, lat_sub_min, lon_sub_max, lon_sub_min)

for iyr in range(5, len(yearname)):
    os.makedirs(path_output + '/colorado/smap_1km_avg/' + str(yearname[iyr]))

# 1.1 1 km (daily, SMAP)
for iyr in range(5, len(yearname)):
    smap_file_path = path_smap + '/1km/' + str(yearname[iyr])
    smap_file_list = sorted(glob.glob(smap_file_path + '/*'))

    for idt in range(len(smap_file_list)):
        src_tf = rasterio.open(smap_file_list[idt])

        # Subset the SM data by bounding box
        sub_window_1km = Window(col_sub_ease_1km_ind[0], row_sub_ease_1km_ind[0],
                                len(col_sub_ease_1km_ind), len(row_sub_ease_1km_ind))
        kwargs_1km_sub = src_tf.meta.copy()
        kwargs_1km_sub.update({
            'height': sub_window_1km.height,
            'width': sub_window_1km.width,
            'transform': rasterio.windows.transform(sub_window_1km, src_tf.transform)})

        # Write to Geotiff file
        with rasterio.open(path_output + '/colorado/smap_1km/' + str(yearname[iyr]) + '/' +\
                           os.path.basename(smap_file_list[idt]).split('.')[0] + '.tif', 'w', **kwargs_1km_sub) \
                as output_ds:
            output_ds.write(src_tf.read(window=sub_window_1km))

        print(os.path.basename(smap_file_list[idt]))


# 1.1.1 1 km (daily-averaged, SMAP)
for iyr in range(5, len(yearname)):
    smap_file_path = path_output + '/colorado/smap_1km/' + str(yearname[iyr])
    smap_file_list = sorted(glob.glob(smap_file_path + '/*'))

    for idt in range(len(smap_file_list)):
        src_tf = rasterio.open(smap_file_list[idt])
        src_tf_mat = src_tf.read()
        src_tf_mat = np.nanmean(src_tf_mat, axis=0)
        src_tf_mat = np.expand_dims(src_tf_mat, 0)

        kwargs_1km_sub = src_tf.meta.copy()
        kwargs_1km_sub.update({
            'count': 1,
        })

        # Write to Geotiff file
        with rasterio.open(path_output + '/colorado/smap_1km_avg/' + str(yearname[iyr]) + '/' +\
                           os.path.basename(smap_file_list[idt]).split('.')[0] + '.tif', 'w', **kwargs_1km_sub) \
                as output_ds:
            output_ds.write(src_tf_mat)

        print(os.path.basename(smap_file_list[idt]))


# 1.1.2 1 km (daily, SMAP monthly)

for idt in range(len(smap_file_list_avg_all)):
    src_tf = smap_file_list_avg_all[idt]
    src_tf = np.expand_dims(src_tf, 0)
    # Subset the SM data by bounding box
    # sub_window_1km = Window(col_sub_ease_1km_ind[0], row_sub_ease_1km_ind[0],
    #                         len(col_sub_ease_1km_ind), len(row_sub_ease_1km_ind))
    # kwargs_1km_sub = src_tf.meta.copy()
    kwargs_1km_sub.update({
        'count': 1,
        'crs': CRS.from_dict(init='epsg:6933')})

    # Write to Geotiff file
    with rasterio.open(path_output + '/california/smap_1km_monthly/' + str(yearname[iyr]) + '/' +\
                       os.path.basename(smap_file_list[idt]).split('.')[0] + '.tif', 'w', **kwargs_1km_sub) \
            as output_ds:
        output_ds.write(src_tf)

    print(os.path.basename(smap_file_list[idt]))


# 1.2 Subset the 9 km SMAP SM
col_coor_sub = -17367530.44516138 + col_sub_ease_9km_ind[0].item() * interdist_ease_9km
row_coor_sub = 7314540.79258289 - row_sub_ease_9km_ind[0].item() * interdist_ease_9km

profile = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_sub_ease_9km),
           'height': len(lat_sub_ease_9km), 'count': 2, 'crs': CRS.from_dict(init='epsg:6933'),
           'transform': Affine(interdist_ease_9km, 0.0, col_coor_sub, 0.0, -interdist_ease_9km, row_coor_sub)}

for iyr in [12]:#range(len(yearname)):
    for imo in range(len(monthname)):
        hdf_file_smap_9km = path_smap + '/9km/' + 'smap_sm_9km_' + str(yearname[iyr]) + monthname[imo] + '.hdf5'
        if os.path.exists(hdf_file_smap_9km) == True:
            f_read_smap_9km = h5py.File(hdf_file_smap_9km, "r")
            varname_list_smap_9km = list(f_read_smap_9km.keys())
            smap_9km_load_am = f_read_smap_9km[varname_list_smap_9km[0]][row_sub_ease_9km_ind[0]:row_sub_ease_9km_ind[-1]+1,
                    col_sub_ease_9km_ind[0]:col_sub_ease_9km_ind[-1]+1, :]
            smap_9km_load_pm = f_read_smap_9km[varname_list_smap_9km[1]][row_sub_ease_9km_ind[0]:row_sub_ease_9km_ind[-1]+1,
                    col_sub_ease_9km_ind[0]:col_sub_ease_9km_ind[-1]+1, :]
            f_read_smap_9km.close()

            for idt in range(smap_9km_load_am.shape[2]):
                dst = np.stack([smap_9km_load_am[:, :, idt], smap_9km_load_pm[:, :, idt]], axis=0)
                day_str = str(yearname[iyr]) + '-' + monthname[imo] + '-' + str(idt+1).zfill(2)
                dst_writer = rasterio.open(path_output + '/SE/9km/' + str(yearname[iyr]) + '/smap_sm_9km_' +  str(yearname[iyr])
                                           + str(datetime.datetime.strptime(day_str, '%Y-%m-%d').timetuple().tm_yday).zfill(3) + '.tif',
                                   'w', **profile)
                dst_writer.write(dst)
                dst_writer = None

                print(day_str)
                del(day_str, dst_writer, dst)

            del (smap_9km_load_am, smap_9km_load_pm, f_read_smap_9km)

        else:
            pass


for iyr in range(5, len(yearname)):
    os.makedirs(path_output + '/pakistan/smos_1km/' + str(yearname[iyr]))

# 1.3 1 km (daily, SMOS)
for iyr in [12]:#range(0, len(yearname)):
    smap_file_path = path_smos + '/1km/' + str(yearname[iyr])
    smap_file_list = sorted(glob.glob(smap_file_path + '/*'))

    for idt in range(len(smap_file_list)):
        src_tf = rasterio.open(smap_file_list[idt])

        # Subset the SM data by bounding box
        sub_window_1km = Window(col_sub_ease_1km_ind[0], row_sub_ease_1km_ind[0],
                                len(col_sub_ease_1km_ind), len(row_sub_ease_1km_ind))
        kwargs_1km_sub = src_tf.meta.copy()
        kwargs_1km_sub.update({
            'height': sub_window_1km.height,
            'width': sub_window_1km.width,
            'crs': CRS.from_dict(init='epsg:6933'),
            'transform': rasterio.windows.transform(sub_window_1km, src_tf.transform)})

        # Write to Geotiff file
        with rasterio.open(path_output + '/pakistan/smos_1km/' + str(yearname[iyr]) + '/' +\
                           os.path.basename(smap_file_list[idt]).split('.')[0] + '.tif', 'w', **kwargs_1km_sub) \
                as output_ds:
            output_ds.write(src_tf.read(window=sub_window_1km))

        print(os.path.basename(smap_file_list[idt]))


# 1.4 Subset the 25 km SMOS SM
for iyr in range(0, len(yearname)):
    os.makedirs(path_output + '/chile/smos_25km/' + str(yearname[iyr]))

col_coor_sub = -17367530.44516138 + col_sub_ease_25km_ind[0].item() * interdist_ease_25km
row_coor_sub = 7314540.79258289 - row_sub_ease_25km_ind[0].item() * interdist_ease_25km

profile = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_sub_ease_25km),
           'height': len(lat_sub_ease_25km), 'count': 2, 'crs': CRS.from_dict(init='epsg:6933'),
           'transform': Affine(interdist_ease_25km, 0.0, col_coor_sub, 0.0, -interdist_ease_25km, row_coor_sub)}

for iyr in range(len(yearname)):
    for imo in range(len(monthname)):
        hdf_file_smos_25km = path_smos + '/25km/' + 'smos_sm_25km_' + str(yearname[iyr]) + monthname[imo] + '.hdf5'
        if os.path.exists(hdf_file_smos_25km) == True:
            f_read_smos_25km = h5py.File(hdf_file_smos_25km, "r")
            varname_list_smos_25km = list(f_read_smos_25km.keys())
            smos_25km_load_am = f_read_smos_25km[varname_list_smos_25km[0]][row_sub_ease_25km_ind[0]:row_sub_ease_25km_ind[-1]+1,
                    col_sub_ease_25km_ind[0]:col_sub_ease_25km_ind[-1]+1, :]
            smos_25km_load_pm = f_read_smos_25km[varname_list_smos_25km[1]][row_sub_ease_25km_ind[0]:row_sub_ease_25km_ind[-1]+1,
                    col_sub_ease_25km_ind[0]:col_sub_ease_25km_ind[-1]+1, :]
            f_read_smos_25km.close()

            for idt in range(smos_25km_load_am.shape[2]):
                dst = np.stack([smos_25km_load_am[:, :, idt], smos_25km_load_pm[:, :, idt]], axis=0)
                day_str = str(yearname[iyr]) + '-' + monthname[imo] + '-' + str(idt+1).zfill(2)
                dst_writer = rasterio.open(path_output + '/chile/smos_25km/' + str(yearname[iyr]) + '/smos_sm_25km_' +  str(yearname[iyr])
                                           + str(datetime.datetime.strptime(day_str, '%Y-%m-%d').timetuple().tm_yday).zfill(3) + '.tif',
                                   'w', **profile)
                dst_writer.write(dst)
                dst_writer = None

                print(day_str)
                del(day_str, dst_writer, dst)

            del (smos_25km_load_am, smos_25km_load_pm, f_read_smos_25km)

        else:
            pass

########################################################################################################################
# 2.1 Subset SMAP 1 km/9 km data (watershed boundaries)

# shapefile = fiona.open(path_shp + '/Chesapeake_Bay_Watershed_and_Basins/Chesapeake_Bay_Watershed_and_Basins.shp', 'r')
output_crs = 'EPSG:4326'
# shapefile_folder = sorted(glob.glob(path_gis + '/wrd_riverbasins/*'))
shapefile_folder = sorted(glob.glob(path_shp + '/us_NM_CA/*.shp'))
# shapefile_folder = list([shapefile_folder[1]])
shapefile_name = [shapefile_folder[x].split('/')[-1][:5] for x in range(len(shapefile_folder))]
# shapefile_name = [shapefile_folder[x].split('/')[-1].split('.')[-2] for x in range(len(shapefile_folder))]


row_sub_ease_1km_ind_all = []
col_sub_ease_1km_ind_all = []
crop_shape_all = []
row_sub_ease_9km_ind_all = []
col_sub_ease_9km_ind_all = []
lat_sub_ease_9km_all = []
lon_sub_ease_9km_all = []
for ife in range(len(shapefile_folder)):
    shapefile = fiona.open(shapefile_folder[ife], 'r')
    crop_shape = [feature["geometry"] for feature in shapefile]
    shp_extent = list(shapefile.bounds)

    lat_sub_max = shp_extent[3]
    lat_sub_min = shp_extent[1]
    lon_sub_max = shp_extent[2]
    lon_sub_min = shp_extent[0]

    [lat_sub_ease_1km, row_sub_ease_1km_ind, lon_sub_ease_1km, col_sub_ease_1km_ind] = coordtable_subset_rev\
        (lat_world_ease_1km, lon_world_ease_1km, lat_sub_max, lat_sub_min, lon_sub_max, lon_sub_min)
    [lat_sub_ease_9km, row_sub_ease_9km_ind, lon_sub_ease_9km, col_sub_ease_9km_ind] = coordtable_subset_rev\
        (lat_world_ease_9km, lon_world_ease_9km, lat_sub_max, lat_sub_min, lon_sub_max, lon_sub_min)

    row_sub_ease_1km_ind_all.append(row_sub_ease_1km_ind)
    col_sub_ease_1km_ind_all.append(col_sub_ease_1km_ind)
    crop_shape_all.append(crop_shape)
    row_sub_ease_9km_ind_all.append(row_sub_ease_9km_ind)
    col_sub_ease_9km_ind_all.append(col_sub_ease_9km_ind)
    lat_sub_ease_9km_all.append(lat_sub_ease_9km)
    lon_sub_ease_9km_all.append(lon_sub_ease_9km)

# src_tf = rasterio.open('/Users/binfang/Downloads/Processing/smap_output/1km/smap_sm_1km_ds_2015092.tif')

# 1 km
for ife in range(len(shapefile_name)):
    os.makedirs(path_output + '/us_NM_CA/' + shapefile_name[ife])
output_folder = sorted(glob.glob(path_output + '/us_NM_CA/*/'))

for iyr in range(5, len(yearname)):
    for ife in range(len(shapefile_name)):
        os.makedirs(output_folder[ife] + str(yearname[iyr]))

for iyr in range(5, len(yearname)):
    smap_file_path = path_smap + '/1km/' + str(yearname[iyr])
    smap_file_list = sorted(glob.glob(smap_file_path + '/*'))

    for idt in range(len(smap_file_list)):
        src_tf = rasterio.open(smap_file_list[idt])

        for ife in range(len(output_folder)):
            # Subset the SM data by bounding box
            sub_window_1km = Window(col_sub_ease_1km_ind_all[ife][0], row_sub_ease_1km_ind_all[ife][0],
                                    len(col_sub_ease_1km_ind_all[ife]), len(row_sub_ease_1km_ind_all[ife]))
            kwargs_1km_sub = src_tf.meta.copy()
            kwargs_1km_sub.update({
                'height': sub_window_1km.height,
                'width': sub_window_1km.width,
                'transform': rasterio.windows.transform(sub_window_1km, src_tf.transform)})

            src_tf_subset = src_tf.read(window=sub_window_1km)
            src_tf_subset = np.nanmean(src_tf_subset, axis=0)

            sub_window_1km = Window(col_sub_ease_1km_ind_all[ife][0], row_sub_ease_1km_ind_all[ife][0],
                                    len(col_sub_ease_1km_ind_all[ife]), len(row_sub_ease_1km_ind_all[ife]))
            kwargs_1km_sub = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_1km),
                              'height': len(lat_world_ease_1km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
                              'transform': Affine(1000.89502334956, 0.0, -17367530.44516138, 0.0, -1000.89502334956,
                                                  7314540.79258289)}
            smap_sm_1km_output = sub_n_reproj(src_tf_subset, kwargs_1km_sub, sub_window_1km, output_crs)

            masked_ds_1km, mask_transform_ds_1km = mask(dataset=smap_sm_1km_output, shapes=crop_shape_all[ife],
                                                                crop=True)
            masked_ds_1km[np.where(masked_ds_1km == 0)] = np.nan
            # masked_ds_1km = masked_ds_1km.squeeze()

            # Write to Geotiff file
            with rasterio.open(output_folder[ife] + str(yearname[iyr]) + '/' +\
                               os.path.basename(smap_file_list[idt]), 'w', **kwargs_1km_sub) as output_ds:
                output_ds.write(masked_ds_1km)

        print(os.path.basename(smap_file_list[idt]))
        del(src_tf, src_tf_subset, masked_ds_1km, mask_transform_ds_1km)



# 9 km
for ife in range(len(shapefile_name)):
    os.makedirs(path_output + '/us_ca/' + shapefile_name[ife])
output_folder = sorted(glob.glob(path_output + '/us_ca/*/'))

for iyr in range(5, len(yearname)):
    for ife in range(len(shapefile_name)):
        os.makedirs(output_folder[ife] + str(yearname[iyr]))

profile_all = []
for ife in range(len(output_folder)):
    col_coor_sub = -17367530.44516138 + col_sub_ease_9km_ind_all[ife][0].item() * interdist_ease_9km
    row_coor_sub = 7314540.79258289 - row_sub_ease_9km_ind_all[ife][0].item() * interdist_ease_9km
    profile = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_sub_ease_9km_all[ife]),
               'height': len(lat_sub_ease_9km_all[ife]), 'count': 2, 'crs': CRS.from_dict(init='epsg:6933'),
               'transform': Affine(interdist_ease_9km, 0.0, col_coor_sub, 0.0, -interdist_ease_9km, row_coor_sub)}
    profile_all.append(profile)
    del(profile)

for iyr in [14]:#range(5, len(yearname)):
    for imo in range(len(monthname)):
        hdf_file_smap_9km = path_smap + '/9km' + '/smap_sm_9km_' + str(yearname[iyr]) + monthname[imo] + '.hdf5'
        if os.path.exists(hdf_file_smap_9km) == True:
            f_read_smap_9km = h5py.File(hdf_file_smap_9km, "r")
            varname_list_smap_9km = list(f_read_smap_9km.keys())

            smap_9km_load_am_all = []
            smap_9km_load_pm_all = []
            for ife in range(len(output_folder)):
                smap_9km_load_am = \
                    f_read_smap_9km[varname_list_smap_9km[0]][row_sub_ease_9km_ind_all[ife][0]:row_sub_ease_9km_ind_all[ife][-1]+1,
                        col_sub_ease_9km_ind_all[ife][0]:col_sub_ease_9km_ind_all[ife][-1]+1, :]
                smap_9km_load_pm = \
                    f_read_smap_9km[varname_list_smap_9km[1]][row_sub_ease_9km_ind_all[ife][0]:row_sub_ease_9km_ind_all[ife][-1]+1,
                        col_sub_ease_9km_ind_all[ife][0]:col_sub_ease_9km_ind_all[ife][-1]+1, :]

                smap_9km_load_am_all.append(smap_9km_load_am)
                smap_9km_load_pm_all.append(smap_9km_load_pm)
                del(smap_9km_load_am, smap_9km_load_pm)
            f_read_smap_9km.close()

            for ife in range(len(output_folder)):
                for idt in range(smap_9km_load_am_all[0].shape[2]):
                    dst = np.stack([smap_9km_load_am_all[ife][:, :, idt], smap_9km_load_pm_all[ife][:, :, idt]], axis=0)
                    day_str = str(yearname[iyr]) + '-' + monthname[imo] + '-' + str(idt+1).zfill(2)
                    dst_writer = rasterio.open(output_folder[ife] + str(yearname[iyr]) + '/' + 'smap_sm_9km_' + str(yearname[iyr]) + \
                                               str(datetime.datetime.strptime(day_str, '%Y-%m-%d').timetuple().tm_yday).zfill(3) + \
                                               '.tif', 'w', **profile_all[ife])
                    dst_writer.write(dst)
                    dst_writer = None

                    print(day_str)
                    del(day_str, dst_writer, dst)

            del (smap_9km_load_am_all, smap_9km_load_pm_all, f_read_smap_9km)

        else:
            pass


########################################################################################################################
# 3. Average the SMAP 1 km / 9 km SM to monthly composite
monthly_seq_cumsum = np.stack([np.cumsum(daysofmonth_seq[:, x]) for x in range(daysofmonth_seq.shape[1])], axis=1)
path_smap_1km = '/Users/binfang/Downloads/Processing/smap_output/california/smap_1km'
path_smap_9km = '/Users/binfang/Downloads/Processing/smap_output/gmd4/smap_9km'

# 1 km SMAP
for iyr in range(12, len(yearname)):
    smap_file_list_all = sorted(glob.glob(path_smap_1km + '/' + str(yearname[iyr]) + '/*.tif'))
    monthly_seq_cumsum_1year = monthly_seq_cumsum[:, iyr]
    if iyr == 5:
        monthly_seq_cumsum_1year = monthly_seq_cumsum_1year - 90
        monthly_seq_cumsum_1year = monthly_seq_cumsum_1year[3:]
    elif iyr == 13:
        monthly_seq_cumsum_1year = monthly_seq_cumsum_1year[:273]
    else:
        pass

    monthly_seq_cumsum_1year = np.concatenate(([0], monthly_seq_cumsum_1year))

    for imo in range(len(monthly_seq_cumsum_1year)-1):
        smap_file_list_avg = smap_file_list_all[monthly_seq_cumsum_1year[imo]:monthly_seq_cumsum_1year[imo + 1]]
        src_profile = rasterio.open(smap_file_list_avg[0]).profile

        src_tf_all = []
        for ife in range(len(smap_file_list_avg)):
            src_tf = rasterio.open(smap_file_list_avg[ife]).read()
            src_tf_all.append(src_tf)
            del(src_tf)

        src_tf_all = np.stack(src_tf_all, axis=0)
        src_tf_all_avg = np.nanmean(src_tf_all, axis=0)

        # Write to file
        filename = 'smap_sm_1km_ds_' + str(yearname[iyr]) + '_' + str(monthname[imo])
        if iyr == 5:
            filename = 'smap_sm_1km_ds_' + str(yearname[iyr]) + '_' + str(monthname[imo+3])
        else:
            pass

        dst_writer = rasterio.open(path_output + '/la_plata/la_plata_monthly/' + filename + '.tif', 'w', **src_profile)
        dst_writer.write(src_tf_all_avg)
        dst_writer = None
        print(filename)
        # smap_file_list_avg = np.nanmean(np.nanmean(src_tf_all, axis=0), axis=0)
        # smap_file_list_avg_all.append(smap_file_list_avg)
        # print(idl)
        del(smap_file_list_avg, src_tf_all, src_tf_all_avg)

    del(smap_file_list_all, monthly_seq_cumsum_1year)


# 1 km SMAP (yearly averaged)
for iyr in range(12, len(yearname)):
    smap_file_list_all = sorted(glob.glob(path_smap_1km + '/' + str(yearname[iyr]) + '/*.tif'))
    if iyr == 5:
        smap_file_list_all = smap_file_list_all[0:183]
    elif iyr == 2 or 6 or 10:
        smap_file_list_all = smap_file_list_all[91:274]
    elif iyr == 13:
        smap_file_list_all = smap_file_list_all[90:242]
    else:
        smap_file_list_all = smap_file_list_all[90:273]

    # monthly_seq_cumsum_1year = np.concatenate(([0], monthly_seq_cumsum_1year))

    src_profile = rasterio.open(smap_file_list_all[0]).profile
    src_profile.update({'count': 1})
    src_tf_all = []
    for idt in range(len(smap_file_list_all)):
        # smap_file_list_avg = smap_file_list_all[monthly_seq_cumsum_1year[imo]:monthly_seq_cumsum_1year[imo + 1]]
        src_tf = rasterio.open(smap_file_list_all[idt]).read()
        src_tf_all.append(src_tf)
        del(src_tf)

    src_tf_all = np.stack(src_tf_all, axis=0)
    src_tf_all_avg = np.nanmean(np.nanmean(src_tf_all, axis=0), axis=0)

    # Write to file
    filename = 'smap_sm_1km_ds_' + str(yearname[iyr])
    src_tf_all_avg = np.expand_dims(src_tf_all_avg, axis=0)
    dst_writer = rasterio.open(path_output + '/la_plata/la_plata_monthly/1km/' + filename + '.tif', 'w', **src_profile)
    dst_writer.write(src_tf_all_avg)
    dst_writer = None
    print(filename)
    del(src_tf_all, src_tf_all_avg, smap_file_list_all)




# 9 km SMAP
for iyr in [12]:#range(5, len(yearname)-1):
    smap_file_list_all = sorted(glob.glob(path_smap_9km + '/' + str(yearname[iyr]) + '/*.tif'))
    monthly_seq_cumsum_1year = monthly_seq_cumsum[:, iyr]
    if iyr == 5:
        monthly_seq_cumsum_1year = monthly_seq_cumsum_1year - 90
        monthly_seq_cumsum_1year = monthly_seq_cumsum_1year[3:]
    else:
        pass

    monthly_seq_cumsum_1year = np.concatenate(([0], monthly_seq_cumsum_1year))

    for imo in range(len(monthly_seq_cumsum_1year)-1):
        smap_file_list_avg = smap_file_list_all[monthly_seq_cumsum_1year[imo]:monthly_seq_cumsum_1year[imo + 1]]
        src_profile = rasterio.open(smap_file_list_avg[0]).profile

        src_tf_all = []
        for ife in range(len(smap_file_list_avg)):
            src_tf = rasterio.open(smap_file_list_avg[ife]).read()
            src_tf_all.append(src_tf)
            del(src_tf)

        src_tf_all = np.stack(src_tf_all, axis=0)
        src_tf_all_avg = np.nanmean(src_tf_all, axis=0)

        # Write to file
        filename = 'smap_sm_9km_ds_' + str(yearname[iyr]) + '_' + str(monthname[imo])
        if iyr == 5:
            filename = 'smap_sm_9km_ds_' + str(yearname[iyr]) + '_' + str(monthname[imo+3])
        else:
            pass

        dst_writer = rasterio.open(path_output + '/la_plata/la_plata_monthly/9km/' + filename + '.tif', 'w', **src_profile)
        dst_writer.write(src_tf_all_avg)
        dst_writer = None
        print(filename)
        # smap_file_list_avg = np.nanmean(np.nanmean(src_tf_all, axis=0), axis=0)
        # smap_file_list_avg_all.append(smap_file_list_avg)
        # print(idl)
        del(smap_file_list_avg, src_tf_all, src_tf_all_avg)

    del(smap_file_list_all, monthly_seq_cumsum_1year)


path_smap_1km = '/Users/binfang/Downloads/smap_400m/T030'
path_output = '/Users/binfang/Downloads/smap_400m/T030_weekly'

# 1 km SMAP (yearly averaged)
smap_file_list_all = sorted(glob.glob(path_smap_1km + '/*.tif'))
file_divide_ind = np.arange(0, len(smap_file_list_all), 7)
file_name_split = np.hsplit(np.arange(len(smap_file_list_all)), file_divide_ind)[1:]

src_profile = rasterio.open(smap_file_list_all[0]).profile
src_profile.update({'count': 1})
src_tf_all = []
for idt in range(len(smap_file_list_all)):
    # smap_file_list_avg = smap_file_list_all[monthly_seq_cumsum_1year[imo]:monthly_seq_cumsum_1year[imo + 1]]
    src_tf = rasterio.open(smap_file_list_all[idt]).read()
    src_tf_all.append(src_tf)
    print(idt)
    del(src_tf)

src_tf_all = np.stack(src_tf_all, axis=0)
src_tf_all_avg = np.nanmean(src_tf_all, axis=1)

for idw in range(len(file_name_split)):
    smap_file_list_weekly = src_tf_all_avg[file_name_split[idw], :, :]
    smap_file_list_weekly = np.nanmean(smap_file_list_weekly, axis=0)
    # Write to file
    filename = 'weekly_' + os.path.basename(smap_file_list_all[file_name_split[idw][0]])
    smap_file_list_weekly = np.expand_dims(smap_file_list_weekly, axis=0)
    dst_writer = rasterio.open(path_output + '/' + filename, 'w', **src_profile)
    dst_writer.write(smap_file_list_weekly)
    dst_writer = None
    print(filename)
    del(smap_file_list_weekly)



# 10 km GPM
src_profile = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': 234, 'height': 216, 'count': 1,
               'crs': CRS.from_epsg(4326), 'transform': Affine(0.1, 0.0, -67, 0.0, -0.1, -14.1),
               'tiled': False, 'interleave': 'pixel'}

for iyr in range(5, len(yearname)-1):
    monthly_seq_cumsum_1year = monthly_seq_cumsum[:, iyr]
    monthly_seq_cumsum_1year = np.concatenate(([0], monthly_seq_cumsum_1year))

    f_gpm = h5py.File(path_gpm + '/gpm_precip_' + str(yearname[iyr]) + '.hdf5', 'r')
    varname_list_gpm = list(f_gpm.keys())
    gpm_precip = f_gpm[varname_list_gpm[0]][()]
    gpm_precip_ext = gpm_precip[row_sub_geo_10km_ind[0]:row_sub_geo_10km_ind[-1]+1,
                     col_sub_geo_10km_ind[0]:col_sub_geo_10km_ind[-1]+1, :]
    f_gpm.close()

    for imo in range(0, len(monthly_seq_cumsum_1year)-1):
        gpm_precip_ext_1month = gpm_precip_ext[:, :, monthly_seq_cumsum_1year[imo]:monthly_seq_cumsum_1year[imo + 1]]
        gpm_precip_ext_1month = np.nansum(gpm_precip_ext_1month, axis=2)
        gpm_precip_ext_1month = np.expand_dims(gpm_precip_ext_1month, axis=0)

        # Write to file
        filename = 'gpm_prec_10km_' + str(yearname[iyr]) + '_' + str(monthname[imo])
        dst_writer = rasterio.open(path_output + '/la_plata/la_plata_monthly/10km/' + filename + '.tif', 'w', **src_profile)
        dst_writer.write(gpm_precip_ext_1month)
        dst_writer = None
        print(filename)

        del(gpm_precip_ext_1month)

    del(gpm_precip, gpm_precip_ext, monthly_seq_cumsum_1year)


########################################################################################################################
# 4. Process SMAP enhanced L2 radiometer half-orbit 9 km data (Ancillary data)
# path_smap = '/Volumes/Elements2/Datasets/SMAP'
# path_smap_flag = '/Volumes/Elements2/Datasets/SMAP_flag'

# Variable name and layer number
# albedo: #2; clay_fraction: #6; roughness_coefficient: #19; soil_moisture: #22; surface_temperature: #29
# tbv_corrected: #42; vegetation_opacity: #44; vegetation_water_content: #48
# ind_var = [2, 6, 19, 22, 29, 42, 44, 48]
# list_var = ['albedo', 'clay_fraction', 'roughness_coefficient', 'soil_moisture', 'surface_temperature',
#             'tbv_corrected', 'vegetation_opacity', 'vegetation_water_content']
# var_num = 8
ind_var = [48]
list_var = ['vegetation_water_content']
var_num = 1

matsize_smap_1day = [len(lat_world_ease_9km), len(lon_world_ease_9km), var_num]
smap_mat_init_1day = np.empty(matsize_smap_1day, dtype='float32')
smap_mat_init_1day[:] = np.nan

profile = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_9km),
           'height': len(lat_world_ease_9km), 'count': var_num, 'crs': CRS.from_dict(init='epsg:6933'),
           'transform': Affine(interdist_ease_9km, 0.0, -17367530.44516138, 0.0, -interdist_ease_9km, 7314540.79258289)}

for iyr in [12]:#range(5, len(daysofyear)):

    os.chdir(path_smap_org + '/' + str(yearname[iyr]))
    smap_files_year = sorted(glob.glob('*.h5'))

    # Group SMAP data by month
    for imo in range(0, len(monthnum)):

        os.chdir(path_smap_org + '/' + str(yearname[iyr]))
        smap_files_group_1month = [smap_files_year.index(i) for i in smap_files_year if str(yearname[iyr]) + monthname[imo] in i]

        # Process each month
        if len(smap_files_group_1month) != 0:
            smap_files_month = [smap_files_year[smap_files_group_1month[i]] for i in range(len(smap_files_group_1month))]

            # Extract SMAP data layers and rebind to daily
            for idt in range(daysofmonth_seq[imo, iyr]):
                smap_files_group_1day = [smap_files_month.index(i) for i in smap_files_month if
                                         str(yearname[iyr]) + monthname[imo] + str(idt+1).zfill(2) in i]
                smap_files_1day = [smap_files_month[smap_files_group_1day[i]] for i in
                                    range(len(smap_files_group_1day))]
                smap_files_group_1day_am = [smap_files_1day.index(i) for i in smap_files_1day if
                                         'D_' + str(yearname[iyr]) + monthname[imo] + str(idt+1).zfill(2) in i]
                smap_files_group_1day_pm = [smap_files_1day.index(i) for i in smap_files_1day if
                                         'A_' + str(yearname[iyr]) + monthname[imo] + str(idt+1).zfill(2) in i]
                smap_mat_group_1day = \
                    np.empty([matsize_smap_1day[0], matsize_smap_1day[1], var_num, len(smap_files_group_1day)], dtype='float32')
                smap_mat_group_1day[:] = np.nan

                # Read swath files within a day and stack
                for ife in range(len(smap_files_1day)):
                    smap_mat_1file = np.copy(smap_mat_init_1day)
                    fe_smap = h5py.File(smap_files_1day[ife], "r")
                    group_list_smap = list(fe_smap.keys())
                    smap_data_group = fe_smap[group_list_smap[1]]
                    varname_list_smap = list(smap_data_group.keys())
                    # Extract variables
                    col_ind = smap_data_group[varname_list_smap[0]][()]
                    row_ind = smap_data_group[varname_list_smap[1]][()]
                    for i in range(var_num):
                        smap_var = smap_data_group[varname_list_smap[ind_var[i]]][()]
                        smap_mat_1file[row_ind, col_ind, i] = smap_var
                        del(smap_var)
                    smap_mat_1file[smap_mat_1file < 0] = np.nan
                    smap_mat_group_1day[:, :, :, ife] = smap_mat_1file
                    print(smap_files_1day[ife])
                    fe_smap.close()

                    del(smap_mat_1file, fe_smap, group_list_smap, smap_data_group, varname_list_smap, col_ind, row_ind)

                # Average values by am/pm indices separately
                smap_mat_1day_am = np.nanmean(smap_mat_group_1day[:, :, :, smap_files_group_1day_am], axis=3)
                smap_mat_1day_pm = np.nanmean(smap_mat_group_1day[:, :, :, smap_files_group_1day_pm], axis=3)
                smap_mat_1day_am = np.transpose(smap_mat_1day_am, (2, 0, 1))
                smap_mat_1day_pm = np.transpose(smap_mat_1day_pm, (2, 0, 1))
                smap_mat_1day_data = [smap_mat_1day_am, smap_mat_1day_pm]

                day_str = str(yearname[iyr]) + '-' + monthname[imo] + '-' + str(idt + 1).zfill(2)
                day_str_doy = str(datetime.datetime.strptime(day_str, '%Y-%m-%d').timetuple().tm_yday).zfill(3)
                filename = 'smap_anc_9km_' + str(yearname[iyr]) + day_str_doy
                overpass = ['_des.tif', '_asc.tif']
                for ilr in range(2):
                    dst_writer = rasterio.open(path_output + '/SMAP/smap_ancillary_data/' + str(yearname[iyr]) + '/' +
                                                  filename + overpass[ilr], 'w', **profile)
                    dst_writer.write(smap_mat_1day_data[ilr])
                    for j in range(var_num):
                        dst_writer.set_band_description(j+1, list_var[j])
                    dst_writer = None
                print(filename)

                del(smap_mat_group_1day, smap_files_group_1day, smap_files_1day, smap_mat_1day_am, smap_mat_1day_pm,
                    smap_mat_1day_data, day_str, day_str_doy, filename)

        else:
            pass



########################################################################################################################
# 5. Downscale the 9 km SMAP ancillary data to 1 km

# Create initial EASE grid projection matrices
# smap_sm_1km_disagg_init = np.empty([len(lat_world_ease_1km), len(lon_world_ease_1km)], dtype='float32')
# smap_sm_1km_disagg_init = smap_sm_1km_disagg_init.reshape(1, -1)
# smap_sm_1km_disagg_init[:] = np.nan

var_name = ['albedo', 'clay_fraction', 'roughness_coefficient', 'soil_moisture', 'surface_temperature',
            'tbv_corrected', 'vegetation_opacity', 'vegetation_water_content']

for iyr in range(5, len(yearname)):
    smap_file_list = sorted(glob.glob(path_smap_anc + '/' + str(yearname[iyr]) + '/*.tif'))

    for idt in range(len(smap_file_list)):
        filename = os.path.basename(smap_file_list[idt])
        ds_smap_sm_1km = gdal.Open(smap_file_list[idt])
        smap_sm_1km = ds_smap_sm_1km.ReadAsArray()
        smap_sm_1km = np.transpose(smap_sm_1km, (1, 2, 0))
        # smap_sm_1km = smap_sm_1km[:, :, [1, 2, 4, 5]]

        smap_sm_1km_anc_all = []
        for idf in range(smap_sm_1km.shape[2]):
            smap_sm_1km_anc = smap_sm_1km[:, :, idf]
            smap_sm_1km_smap_sm_1km_flag_disagg = \
                np.array([smap_sm_1km_anc[row_meshgrid_from_9km[0, smap_sm_1km_1file_ind[x]],
                                          col_meshgrid_from_9km[0, smap_sm_1km_1file_ind[x]]]
                          for x in range(len(smap_sm_1km_1file_ind))])
            # smap_sm_1km_disagg = np.copy(smap_sm_1km_disagg_init)
            # smap_sm_1km_disagg[0, smap_sm_1km_1file_ind] = smap_sm_1km_smap_sm_1km_flag_disagg
            # smap_sm_1km_disagg = smap_sm_1km_disagg.reshape(len(lat_world_ease_1km), len(lon_world_ease_1km))
            smap_sm_1km_anc_all.append(smap_sm_1km_smap_sm_1km_flag_disagg)
            del(smap_sm_1km_smap_sm_1km_flag_disagg, smap_sm_1km_anc)

        # Save file
        os.chdir(path_smap_anc_1km + '/' + str(yearname[iyr]))
        filename_out = 'smap_anc_1km_' + filename[13:24]

        with h5py.File(filename_out + '.hdf5', 'w') as f:
            for idv in range(len(var_name)):
                f.create_dataset(var_name[idv], data=smap_sm_1km_anc_all[idv])
        f.close()

        print(filename_out)
        del(smap_sm_1km_anc_all, ds_smap_sm_1km, smap_sm_1km, filename, filename_out)


        # # Save the daily 1 km SM model output to Geotiff files
        # # Build output path
        # os.chdir(path_smap_anc_1km + '/' + str(yearname[iyr]))
        #
        # # Create a raster of EASE grid projection at 1 km resolution
        # out_ds_tiff = gdal.GetDriverByName('GTiff').Create\
        #     ('smap_anc_1km_' + filename[13:], len(lon_world_ease_1km), len(lat_world_ease_1km), 4, # Number of bands
        #     gdal.GDT_Float32, ['COMPRESS=LZW', 'TILED=YES'])
        # out_ds_tiff.SetGeoTransform(lmask_src.GetGeoTransform())
        # out_ds_tiff.SetProjection(lmask_src.GetProjection())
        #
        # # Loop write each band to Geotiff file
        # for idf in range(len(smap_sm_1km_anc_all)):
        #     out_ds_tiff.GetRasterBand(idf + 1).WriteArray(smap_sm_1km_anc_all[idf])
        #     out_ds_tiff.GetRasterBand(idf + 1).SetNoDataValue(0)
        # out_ds_tiff = None  # close dataset to write to disc

        # print('smap_anc_1km_' + filename[13:])
        # del(smap_sm_1km_anc_all, ds_smap_sm_1km, smap_sm_1km, out_ds_tiff, filename)

    del(smap_file_list)



########################################################################################################################
# 6. Process SMAP enhanced L2 radiometer half-orbit 9 km data (TB data)
# path_smap = '/Volumes/Elements2/Datasets/SMAP'
# path_smap_flag = '/Volumes/Elements2/Datasets/SMAP_flag'

# variable_list
# tbv_corrected: #42
ind_var = [42]
list_var = ['tbv_corrected']
matsize_smap_1day = [len(lat_world_ease_9km), len(lon_world_ease_9km), 1]
smap_mat_init_1day = np.empty(matsize_smap_1day, dtype='float32')
smap_mat_init_1day[:] = np.nan

profile = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_world_ease_9km),
           'height': len(lat_world_ease_9km), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
           'transform': Affine(interdist_ease_9km, 0.0, -17367530.44516138, 0.0, -interdist_ease_9km, 7314540.79258289)}

for iyr in range(len(yearname)):
    os.mkdir(path_output + '/smap_ancillary_data/' + str(yearname[iyr]))

for iyr in [7, 8, 9, 10]:#range(5, len(daysofyear)):

    os.chdir(path_smap_org + '/' + str(yearname[iyr]))
    smap_files_year = sorted(glob.glob('*.h5'))

    # Group SMAP data by month
    for imo in range(0, len(monthnum)):

        os.chdir(path_smap_org + '/' + str(yearname[iyr]))
        smap_files_group_1month = [smap_files_year.index(i) for i in smap_files_year if str(yearname[iyr]) + monthname[imo] in i]

        # Process each month
        if len(smap_files_group_1month) != 0:
            smap_files_month = [smap_files_year[smap_files_group_1month[i]] for i in range(len(smap_files_group_1month))]

            # Extract SMAP data layers and rebind to daily
            for idt in range(daysofmonth_seq[imo, iyr]):
                smap_files_group_1day = [smap_files_month.index(i) for i in smap_files_month if
                                         str(yearname[iyr]) + monthname[imo] + str(idt+1).zfill(2) in i]
                smap_files_1day = [smap_files_month[smap_files_group_1day[i]] for i in
                                    range(len(smap_files_group_1day))]
                smap_files_group_1day_am = [smap_files_1day.index(i) for i in smap_files_1day if
                                         'D_' + str(yearname[iyr]) + monthname[imo] + str(idt+1).zfill(2) in i]
                smap_files_group_1day_pm = [smap_files_1day.index(i) for i in smap_files_1day if
                                         'A_' + str(yearname[iyr]) + monthname[imo] + str(idt+1).zfill(2) in i]
                smap_mat_group_1day = \
                    np.empty([matsize_smap_1day[0], matsize_smap_1day[1], 1, len(smap_files_group_1day)], dtype='float32')
                smap_mat_group_1day[:] = np.nan

                # Read swath files within a day and stack
                for ife in range(len(smap_files_1day)):
                    smap_mat_1file = np.copy(smap_mat_init_1day)
                    fe_smap = h5py.File(smap_files_1day[ife], "r")
                    group_list_smap = list(fe_smap.keys())
                    smap_data_group = fe_smap[group_list_smap[1]]
                    varname_list_smap = list(smap_data_group.keys())
                    # Extract variables
                    col_ind = smap_data_group[varname_list_smap[0]][()]
                    row_ind = smap_data_group[varname_list_smap[1]][()]
                    for i in range(1):
                        smap_var = smap_data_group[varname_list_smap[ind_var[i]]][()]
                        smap_mat_1file[row_ind, col_ind, i] = smap_var
                        del(smap_var)
                    smap_mat_1file[smap_mat_1file < 0] = np.nan
                    smap_mat_group_1day[:, :, :, ife] = smap_mat_1file
                    print(smap_files_1day[ife])
                    fe_smap.close()

                    del(smap_mat_1file, fe_smap, group_list_smap, smap_data_group, varname_list_smap, col_ind, row_ind)

                # Average values by am/pm indices separately
                smap_mat_1day_am = np.nanmean(smap_mat_group_1day[:, :, :, smap_files_group_1day_am], axis=3)
                smap_mat_1day_pm = np.nanmean(smap_mat_group_1day[:, :, :, smap_files_group_1day_pm], axis=3)
                smap_mat_1day_am = np.transpose(smap_mat_1day_am, (2, 0, 1))
                smap_mat_1day_pm = np.transpose(smap_mat_1day_pm, (2, 0, 1))
                smap_mat_1day_data = np.nanmean(np.stack((smap_mat_1day_am, smap_mat_1day_pm), axis=0), axis=0)
                smap_mat_1day_data = [smap_mat_1day_data]
                # smap_mat_1day_data = [smap_mat_1day_am, smap_mat_1day_pm]

                day_str = str(yearname[iyr]) + '-' + monthname[imo] + '-' + str(idt + 1).zfill(2)
                day_str_doy = str(datetime.datetime.strptime(day_str, '%Y-%m-%d').timetuple().tm_yday).zfill(3)
                filename = 'smap_anc_9km_' + str(yearname[iyr]) + day_str_doy
                overpass = ['_tbv.tif']
                # overpass = ['_des.tif', '_asc.tif']
                for ilr in range(1):
                    dst_writer = rasterio.open(path_output + '/smap_ancillary_data/' + str(yearname[iyr]) + '/' +
                                                  filename + overpass[ilr], 'w', **profile)
                    dst_writer.write(smap_mat_1day_data[ilr])
                    for j in range(1):
                        dst_writer.set_band_description(j+1, list_var[j])
                    dst_writer = None
                print(filename)

                del(smap_mat_group_1day, smap_files_group_1day, smap_files_1day, smap_mat_1day_am, smap_mat_1day_pm,
                    smap_mat_1day_data, day_str, day_str_doy, filename)

        else:
            pass



########################################################################################################################
# 7. Subset SMAP 1 km/9 km data (watershed boundaries)

# shapefile = fiona.open(path_gis + '/wrd_riverbasins/Aqueduct_river_basins_COLORADO RIVER (PACIFIC OCEAN)/'
#                                   'Aqueduct_river_basins_COLORADO RIVER (PACIFIC OCEAN).shp', 'r')
output_crs = 'EPSG:4326'
# shapefile_folder = sorted(glob.glob(path_gis + '/wrd_riverbasins/*'))
shapefile_folder = sorted(glob.glob(path_shp + '/eventExtentShapefiles/*.shp'))
# shapefile_folder = list([shapefile_folder[1]])
shapefile_name = [shapefile_folder[x].split('/')[-1] for x in range(len(shapefile_folder))]
# shapefile_name = [shapefile_folder[x].split('/')[-1].split('.')[-2] for x in range(len(shapefile_folder))]
# for ife in range(len(shapefile_name)):
#     os.mkdir(path_output + '/tb_watersheds/' + shapefile_name[ife])

row_sub_ease_1km_ind_all = []
col_sub_ease_1km_ind_all = []
row_sub_ease_9km_ind_all = []
col_sub_ease_9km_ind_all = []
lat_sub_ease_9km_all = []
lon_sub_ease_9km_all = []
crop_shape_all = []
for ife in range(len(shapefile_folder)):
    shapefile = fiona.open(shapefile_folder[ife], 'r')
    crop_shape = [feature["geometry"] for feature in shapefile]
    shp_extent = list(shapefile.bounds)

    lat_sub_max = shp_extent[3]
    lat_sub_min = shp_extent[1]
    lon_sub_max = shp_extent[2]
    lon_sub_min = shp_extent[0]

    [lat_sub_ease_1km, row_sub_ease_1km_ind, lon_sub_ease_1km, col_sub_ease_1km_ind] = coordtable_subset\
        (lat_world_ease_1km, lon_world_ease_1km, lat_sub_max, lat_sub_min, lon_sub_max, lon_sub_min)
    [lat_sub_ease_9km, row_sub_ease_9km_ind, lon_sub_ease_9km, col_sub_ease_9km_ind] = coordtable_subset\
        (lat_world_ease_9km, lon_world_ease_9km, lat_sub_max, lat_sub_min, lon_sub_max, lon_sub_min)

    row_sub_ease_1km_ind_all.append(row_sub_ease_1km_ind)
    col_sub_ease_1km_ind_all.append(col_sub_ease_1km_ind)
    row_sub_ease_9km_ind_all.append(row_sub_ease_9km_ind)
    col_sub_ease_9km_ind_all.append(col_sub_ease_9km_ind)
    lat_sub_ease_9km_all.append(lat_sub_ease_9km)
    lon_sub_ease_9km_all.append(lon_sub_ease_9km)
    crop_shape_all.append(crop_shape)

for ife in range(len(shapefile_name)):
    os.mkdir(path_output + '/tb_watersheds/' + shapefile_name[ife].split('.')[0])
    for iyr in [7, 8, 9, 10]:#range(len(yearname)):
        os.mkdir(path_output + '/tb_watersheds/' + shapefile_name[ife].split('.')[0] + '/' + str(yearname[iyr]))
output_folder = sorted(glob.glob(path_output + '/tb_watersheds/*/'))

# 9 km
for iyr in [7, 8, 9, 10]: #range(0, len(yearname)):
    smap_file_path = path_output + '/smap_ancillary_data/' + str(yearname[iyr])
    smap_file_list = sorted(glob.glob(smap_file_path + '/*'))

    for idt in range(len(smap_file_list)):
        src_tf = rasterio.open(smap_file_list[idt])

        for ife in range(len(output_folder)):
            # Subset the SM data by bounding box
            sub_window_9km = Window(col_sub_ease_9km_ind_all[ife][0], row_sub_ease_9km_ind_all[ife][0],
                                    len(col_sub_ease_9km_ind_all[ife]), len(row_sub_ease_9km_ind_all[ife]))
            kwargs_9km_sub = src_tf.meta.copy()
            kwargs_9km_sub.update({
                'height': sub_window_9km.height,
                'width': sub_window_9km.width,
                'transform': rasterio.windows.transform(sub_window_9km, src_tf.transform)})

            # Write to Geotiff file
            path_subset_output = output_folder[ife] + str(yearname[iyr])
            with rasterio.open(path_subset_output + '/' + os.path.basename(smap_file_list[idt]), 'w', **kwargs_9km_sub) \
                    as output_ds:
                output_ds.write(src_tf.read(window=sub_window_9km))

        print(os.path.basename(smap_file_list[idt]))

# profile_all = []
# for ife in range(len(output_folder)):
#     col_coor_sub = -17367530.44516138 + col_sub_ease_9km_ind_all[ife][0].item() * interdist_ease_9km
#     row_coor_sub = 7314540.79258289 - row_sub_ease_9km_ind_all[ife][0].item() * interdist_ease_9km
#     profile = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_sub_ease_9km_all[ife]),
#                'height': len(lat_sub_ease_9km_all[ife]), 'count': 2, 'crs': CRS.from_dict(init='epsg:6933'),
#                'transform': Affine(interdist_ease_9km, 0.0, col_coor_sub, 0.0, -interdist_ease_9km, row_coor_sub)}
#     profile_all.append(profile)
#     del(profile)
#
# for iyr in range(len(yearname)):
#     for imo in range(len(monthname)):
#         hdf_file_smap_9km = path_smap_9km + '/9km' + '/smap_sm_9km_' + str(yearname[iyr]) + monthname[imo] + '.hdf5'
#         if os.path.exists(hdf_file_smap_9km) == True:
#             f_read_smap_9km = h5py.File(hdf_file_smap_9km, "r")
#             varname_list_smap_9km = list(f_read_smap_9km.keys())
#
#             smap_9km_load_am_all = []
#             smap_9km_load_pm_all = []
#             for ife in range(len(output_folder)):
#                 smap_9km_load_am = \
#                     f_read_smap_9km[varname_list_smap_9km[0]][row_sub_ease_9km_ind_all[ife][0]:row_sub_ease_9km_ind_all[ife][-1]+1,
#                         col_sub_ease_9km_ind_all[ife][0]:col_sub_ease_9km_ind_all[ife][-1]+1, :]
#                 smap_9km_load_pm = \
#                     f_read_smap_9km[varname_list_smap_9km[1]][row_sub_ease_9km_ind_all[ife][0]:row_sub_ease_9km_ind_all[ife][-1]+1,
#                         col_sub_ease_9km_ind_all[ife][0]:col_sub_ease_9km_ind_all[ife][-1]+1, :]
#
#                 smap_9km_load_am_all.append(smap_9km_load_am)
#                 smap_9km_load_pm_all.append(smap_9km_load_pm)
#                 del(smap_9km_load_am, smap_9km_load_pm)
#             f_read_smap_9km.close()
#
#             for ife in range(len(output_folder)):
#                 for idt in range(smap_9km_load_am_all[0].shape[2]):
#                     dst = np.stack([smap_9km_load_am_all[ife][:, :, idt], smap_9km_load_pm_all[ife][:, :, idt]], axis=0)
#                     day_str = str(yearname[iyr]) + '-' + monthname[imo] + '-' + str(idt+1).zfill(2)
#                     dst_writer = rasterio.open(output_folder[ife] + 'smap_sm_9km_' + str(yearname[iyr]) + \
#                                                str(datetime.datetime.strptime(day_str, '%Y-%m-%d').timetuple().tm_yday).zfill(3) + \
#                                                '.tif', 'w', **profile_all[ife])
#                     dst_writer.write(dst)
#                     dst_writer = None
#
#                     print(day_str)
#                     del(day_str, dst_writer, dst)
#
#             del (smap_9km_load_am_all, smap_9km_load_pm_all, f_read_smap_9km)
#
#         else:
#             pass


# 8. Subset 1 km MODIS NDVI data
for iyr in range(len(yearname)):
    os.makedirs(path_output + '/la_grande/ndvi_1km/' + str(yearname[iyr]))

# 8.1 1 km (daily, modis)
for iyr in [13]:#range(5, len(yearname)):
    modis_file_path = path_modis + '/' + str(yearname[iyr])
    modis_file_list = sorted(glob.glob(modis_file_path + '/*'))

    for idt in range(len(modis_file_list)):
        src_tf = rasterio.open(modis_file_list[idt])

        # Subset the SM data by bounding box
        sub_window_1km = Window(col_sub_ease_1km_ind[0], row_sub_ease_1km_ind[0],
                                len(col_sub_ease_1km_ind), len(row_sub_ease_1km_ind))
        kwargs_1km_sub = src_tf.meta.copy()
        kwargs_1km_sub.update({
            'height': sub_window_1km.height,
            'width': sub_window_1km.width,
            'transform': rasterio.windows.transform(sub_window_1km, src_tf.transform)})

        # Write to Geotiff file
        with rasterio.open(path_output + '/la_grande/ndvi_1km/' + str(yearname[iyr]) + '/' +\
                           os.path.basename(modis_file_list[idt]).split('.')[0] + '.tif', 'w', **kwargs_1km_sub) \
                as output_ds:
            output_ds.write(src_tf.read(window=sub_window_1km))

        print(os.path.basename(modis_file_list[idt]))

########################################################################################################################
# 9. Make netcdf data from geotiff files

# 1 km
for iyr in range(5, len(yearname)):
    os.makedirs(path_output + '/chile/smap_1km_nc/' + str(yearname[iyr]))

wkt = ('PROJCS["WGS 84 / NSIDC EASE-Grid 2.0 Global",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,'
       'AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],'
       'UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]],PROJECTION["Cylindrical_Equal_Area"],'
       'PARAMETER["standard_parallel_1",30],PARAMETER["central_meridian",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],'
       'UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","6933"]]')

for iyr in range(5, len(yearname)):
    smap_file_path = path_output + '/chile/smap_1km/' + str(yearname[iyr])
    smap_file_list = sorted(glob.glob(smap_file_path + '/*'))

    for idt in range(len(smap_file_list)):
        src = rasterio.open(smap_file_list[idt])
        smap_sm = src.read()
        smap_sm = smap_sm[:, ::-1, :]
        transform = src.transform  # Affine transformation matrix
        width = src.width
        height = src.height
        bounds = src.bounds  # Get raster extent in meters
        crs = src.crs

        # smap_sm = np.nanmean(smap_sm, axis=0)
        filename = smap_file_list[idt].split('/')[-1].split('.')[0]

        nc_file = netCDF4.Dataset(path_output + '/chile/smap_1km_nc/' + str(yearname[iyr]) + '/' + filename + '.nc',
                                  mode='w', format='NETCDF4')
        nc_file.description = '1km downscaled SMAP SM'

        x_coords = np.linspace(bounds.left, bounds.right, width)
        y_coords = np.linspace(bounds.top, bounds.bottom, height)
        y_coords = y_coords[::-1]

        latitudes = y_coords
        longitudes = x_coords
        # sm_data = smap_sm
        time = 1

        # Create dimensions in the NetCDF file
        nc_file.createDimension('lat', len(latitudes))
        nc_file.createDimension('lon', len(longitudes))
        # nc_file.createDimension('time', time)

        # Create variables for latitude, longitude, and soil moisture
        latitude = nc_file.createVariable('lat', 'f4', ('lat',))
        longitude = nc_file.createVariable('lon', 'f4', ('lon',))
        sm_am = nc_file.createVariable('soil_moisture_am', 'f4', ('lat', 'lon'))
        sm_pm = nc_file.createVariable('soil_moisture_pm', 'f4', ('lat', 'lon'))

        # **Define the Projection as EPSG:6933**
        projection = nc_file.createVariable('projection', 'i4')
        projection.grid_mapping_name = "lambert_azimuthal_equal_area"
        projection.false_easting = 0.0
        projection.false_northing = 0.0
        projection.longitude_of_projection_origin = 0.0
        projection.latitude_of_projection_origin = 0.0
        projection.semi_major_axis = 6378137.0
        projection.inverse_flattening = 298.257223563
        projection.spatial_ref = wkt  # Use raster CRS in WKT format
        projection.EPSG_code = 6933

        # Link projection variable to soil moisture variables
        sm_am.grid_mapping = "projection"
        sm_pm.grid_mapping = "projection"

        # Assign data to latitude and longitude variables
        latitude[:] = latitudes
        longitude[:] = longitudes
        sm_am[:, :] = smap_sm[0, :, :]
        sm_pm[:, :] = smap_sm[1, :, :]

        # Add units attribute to variables
        latitude.units = 'degrees_north'
        longitude.units = 'degrees_east'
        sm_am.units = 'm^3/m^3'
        sm_pm.units = 'm^3/m^3'

        # Add global attributes
        nc_file.description = '1km downscaled SMAP SM'
        nc_file.history = 'Created'  # Add creation timestamp or history as needed
        nc_file.projection = "EPSG:6933"
        del(sm_am, sm_pm)
        # Close the NetCDF file
        nc_file.close()
        print(filename)


# 400 m
for iyr in range(5, len(yearname)-1):
    os.makedirs(path_output + '/chile/smap_400m_nc/' + str(yearname[iyr]))

for iyr in range(5, len(yearname)-1):
    smap_file_path = path_output + '/chile/smap_400m/' + str(yearname[iyr])
    smap_file_list = sorted(glob.glob(smap_file_path + '/*'))

    for idt in range(len(smap_file_list)):
        src = rasterio.open(smap_file_list[idt])
        smap_sm = src.read()
        smap_sm = smap_sm[:, ::-1, :]
        transform = src.transform  # Affine transformation matrix
        width = src.width
        height = src.height
        bounds = src.bounds  # Get raster extent in meters
        crs = src.crs

        filename = smap_file_list[idt].split('/')[-1].split('.')[0]

        nc_file = netCDF4.Dataset(path_output + '/chile/smap_400m_nc/' + str(yearname[iyr]) + '/' + filename + '.nc',
                                  mode='w', format='NETCDF4')
        nc_file.description = '400m downscaled SMAP SM'

        x_coords = np.linspace(bounds.left, bounds.right, width)
        y_coords = np.linspace(bounds.top, bounds.bottom, height)
        y_coords = y_coords[::-1]

        latitudes = y_coords
        longitudes = x_coords
        # sm_data = smap_sm
        time = 1

        # Create dimensions in the NetCDF file
        nc_file.createDimension('lat', len(latitudes))
        nc_file.createDimension('lon', len(longitudes))
        # nc_file.createDimension('time', time)

        # Create variables for latitude, longitude, and soil moisture
        latitude = nc_file.createVariable('lat', 'f4', ('lat',))
        longitude = nc_file.createVariable('lon', 'f4', ('lon',))
        sm_am = nc_file.createVariable('soil_moisture_am', 'f4', ('lat', 'lon'))
        sm_pm = nc_file.createVariable('soil_moisture_pm', 'f4', ('lat', 'lon'))

        # Assign data to latitude and longitude variables
        latitude[:] = latitudes
        longitude[:] = longitudes
        sm_am[:, :] = smap_sm[0, :, :]
        sm_pm[:, :] = smap_sm[1, :, :]

        # **Define the Projection as EPSG:6933**
        projection = nc_file.createVariable('projection', 'i4')
        projection.grid_mapping_name = "lambert_azimuthal_equal_area"
        projection.false_easting = 0.0
        projection.false_northing = 0.0
        projection.longitude_of_projection_origin = 0.0
        projection.latitude_of_projection_origin = 0.0
        projection.semi_major_axis = 6378137.0
        projection.inverse_flattening = 298.257223563
        projection.spatial_ref = wkt  # Use raster CRS in WKT format
        projection.EPSG_code = 6933

        # Link projection variable to soil moisture variables
        sm_am.grid_mapping = "projection"
        sm_pm.grid_mapping = "projection"

        # Add units attribute to variables
        latitude.units = 'degrees_north'
        longitude.units = 'degrees_east'
        sm_am.units = 'm^3/m^3'
        sm_pm.units = 'm^3/m^3'

        # Add global attributes
        nc_file.description = '400m downscaled SMAP SM'
        nc_file.history = 'Created'  # Add creation timestamp or history as needed
        nc_file.projection = "EPSG:6933"
        del(sm_am, sm_pm)
        # Close the NetCDF file
        nc_file.close()
        print(filename)


import xarray as xr
import matplotlib.pyplot as plt
ds = xr.open_dataset('/Users/binfang/Downloads/modis_data/modis_lst_25km.nc')
precip = ds['GPM_3IMERGM_06_precipitation'].values
lat = ds['lat'].values
lon = ds['lon'].values
plt.imshow(precip)

f_read_smap_9km = h5py.File('/Users/binfang/Downloads/model_sm_400m_2020183_T038.hdf5', "r")
varname_list_smap_9km = list(f_read_smap_9km.keys())
smap_9km_load_am = f_read_smap_9km[varname_list_smap_9km[0]][()]
smap_9km_load_pm = f_read_smap_9km[varname_list_smap_9km[1]][()]
plt.imshow(np.nanmean(np.stack((smap_9km_load_am, smap_9km_load_pm), axis=2), axis=2))



# 9.2 Make netcdf data from geotiff files (25 km MODIS data)
modis_lst_data = rasterio.open('/Users/binfang/Downloads/modis_data/LST/2001_01_01_LST_Day.tif')
modis_lst_data_transform = modis_lst_data.transform
height, width = modis_lst_data.shape
xsize = modis_lst_data_transform[0]
ysize = modis_lst_data_transform[4]
xmin = modis_lst_data_transform[2]
ymax = modis_lst_data_transform[5]
xmax = xmin + width * xsize
ymin = ymax + height * ysize
lon_25km = np.linspace(xmin, xmax, width + 1)
lat_25km = np.linspace(ymax, ymin, height + 1)
lat_25km = lat_25km + (ysize / 2)
lat_25km = lat_25km[0:-1]
lon_25km = lon_25km + (xsize / 2)
lon_25km = lon_25km[0:-1]


smap_file_path = '/Users/binfang/Downloads/modis_data/ET'
smap_file_list = sorted(glob.glob(smap_file_path + '/*.tif'))
smap_file_list_name = [os.path.basename(smap_file_list[x]).split('.')[0] for x in range(len(smap_file_list))]

smap_sm_all = []
for idt in range(len(smap_file_list)):
    smap_sm = rasterio.open(smap_file_list[idt]).read().squeeze()
    smap_sm_all.append(smap_sm)
    print(idt)
    del(smap_sm)


nc_file = netCDF4.Dataset('/Users/binfang/Downloads/modis_data/modis_et_25km.nc',
                          mode='w', format='NETCDF4')
# nc_file.description = 'MOD21C3.061 Terra Land Surface Temperature and 3-Band Emissivity Monthly L3'
nc_file.description = 'MOD16A2.061: Terra Net Evapotranspiration 8-Day'
# time = 1

# Add projection metadata
proj_var = nc_file.createVariable('crs', 'c', ())
proj_var.long_name = 'CRS definition'
proj_var.grid_mapping_name = 'latitude_longitude'
proj_var.epsg_code = 'EPSG:4326'
proj_var.semi_major_axis = 6378137.0
proj_var.inverse_flattening = 298.257223563

# Create dimensions in the NetCDF file
nc_file.createDimension('lat', len(lat_25km))
nc_file.createDimension('lon', len(lon_25km))
# nc_file.createDimension('time', time)

# Create variables for latitude, longitude, and soil moisture
latitude = nc_file.createVariable('lat', 'f4', ('lat',))
longitude = nc_file.createVariable('lon', 'f4', ('lon',))
# Assign data to latitude and longitude variables
latitude[:] = lat_25km
longitude[:] = lon_25km
# Add units attribute to variables
latitude.units = 'degrees_north'
longitude.units = 'degrees_east'

for idt in range(len(smap_file_list)):
    sm = nc_file.createVariable(smap_file_list_name[idt], 'f4', ('lat', 'lon'))
    sm[:, :] = smap_sm_all[idt]
    sm.units = 'kg/m^2/month'
    sm.grid_mapping = 'crs'
    del(sm)

# Close the NetCDF file
nc_file.close()







################################################################################################################################
# 10. Subset the region of interest (SMAP VIIRS 400 m)

# shapefile = fiona.open(path_shp + '/bounding_box_7_regions', 'r')
shapefile = fiona.open(path_shp + '/Jammu_Watershed/Watershed_boundary.shp', 'r')
crop_shape = [feature["geometry"] for feature in shapefile]
shp_extent = list(shapefile.bounds)
shp_extent = [shp_extent[0], shp_extent[2], shp_extent[1], shp_extent[3]]
# shp_extent = [-123.8, -118.7, 36.7, 45.8]
shp_extent = [-109.060253, -102.041524, 36.992426, 41.003444]
shp_extent = [-107.16232, -105.24800, 36.97396, 38.41805] # Colorado
# shp_extent = [123.9, 132.3, 32.7, 43.5]


[lat_400m, row_400m_ind, lon_400m, col_400m_ind] = \
    coordtable_subset(lat_world_ease_400m, lon_world_ease_400m,
                      shp_extent[3], shp_extent[2], shp_extent[1], shp_extent[0])

df_row_400m_ind = df_row_world_ease_400m_ind.iloc[row_400m_ind, :]
df_col_400m_ind = df_col_world_ease_400m_ind.iloc[col_400m_ind, :]

df_row_400m_ind_split = df_row_400m_ind.groupby(by=['row_ind_tile'])
df_row_400m_ind_group = [df_row_400m_ind_split.get_group(x) for x in df_row_400m_ind_split.groups]
df_col_400m_ind_split = df_col_400m_ind.groupby(by=['col_ind_tile'])
df_col_400m_ind_group = [df_col_400m_ind_split.get_group(x) for x in df_col_400m_ind_split.groups]

tiles_num_row = pd.unique(df_row_400m_ind['row_ind_tile'])
tiles_num_col = pd.unique(df_col_400m_ind['col_ind_tile'])

tile_num = []
row_extent_all = []
col_extent_all = []
for irow in range(len(tiles_num_row)):
    for icol in range(len(tiles_num_col)):
        tile_num_single = tiles_num_row[irow] * 24 + tiles_num_col[icol] + 1
        row_extent = np.array(df_row_400m_ind_group[irow]['row_ind_local'].tolist())[[0, -1]]
        col_extent = np.array(df_col_400m_ind_group[icol]['col_ind_local'].tolist())[[0, -1]]

        tile_num.append(tile_num_single)
        row_extent_all.append(row_extent)
        col_extent_all.append(col_extent)
        del(tile_num_single, row_extent, col_extent)

# tiles_num = tiles_num_row[0] * 24 + tiles_num_col[0] + 1
# Convert from Lat/Lon coordinates to EASE grid projection meter units
transformer = Transformer.from_crs("epsg:4326", "epsg:6933", always_xy=True)
[lon_400m_min, lat_400m_max] = transformer.transform(lon_400m[0], lat_400m[0])

profile = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_400m),
           'height': len(lat_400m), 'count': 1, 'crs': CRS.from_dict(init='epsg:6933'),
           'transform': Affine(interdist_ease_400m, 0.0, lon_400m_min-interdist_ease_400m/2,
                               0.0, -interdist_ease_400m, lat_400m_max+interdist_ease_400m/2)}

# daily averaged
for iyr in range(5, len(yearname)):

    path_output_1year = path_output + '/colorado/smap_400m_avg/' + str(yearname[iyr])
    if os.path.exists(path_output_1year) == False:
        os.makedirs(path_output_1year)
    else:
        pass

    for idt in range(0, daysofyear[iyr]):
        smap_400m_stack = []
        for ite in range(len(tile_num)):
            tif_file_smap_400m_name = ('smap_sm_400m_' + str(yearname[iyr]) + str(idt+1).zfill(3) + '_T' +
                                       str(tile_num[ite]).zfill(3) + '.tif')
            tif_file_smap_400m_path = (path_smap_400m + str(yearname[iyr]) + '/T' + str(tile_num[ite]).zfill(3) + '/'
                                  + tif_file_smap_400m_name)
            if os.path.exists(tif_file_smap_400m_path):
                src_tf = gdal.Open(tif_file_smap_400m_path).ReadAsArray()
            else:
                src_tf = viirs_mat_fill
            src_tf_arr = src_tf[:, row_extent_all[ite][0]:row_extent_all[ite][-1]+1,
                         col_extent_all[ite][0]:col_extent_all[ite][-1]+1]
            src_tf_arr_avg = np.nanmean(src_tf_arr, axis=0)
            smap_400m_stack.append(src_tf_arr_avg)
            del(src_tf_arr_avg, src_tf, src_tf_arr)

        smap_400m_stack_div = [smap_400m_stack[i * len(tiles_num_row) : (i+1) * len(tiles_num_row)]
                               for i in range(len(tiles_num_col))]
        smap_400m_stack_div_by_row = [np.hstack(smap_400m_stack_div[x]) for x in range(len(smap_400m_stack_div))]
        smap_400m_stack_complete = np.hstack(smap_400m_stack_div_by_row)
        smap_400m_stack_complete = np.expand_dims(smap_400m_stack_complete, axis=0)

        tif_file_smap_400m_name_output = tif_file_smap_400m_name.split('.')[0][:-5]
        dst_writer = rasterio.open(path_output_1year + '/' + tif_file_smap_400m_name_output + '.tif', 'w', **profile)
        dst_writer.write(smap_400m_stack_complete)
        dst_writer = None

        print(tif_file_smap_400m_name_output)
        del(smap_400m_stack, smap_400m_stack_div, smap_400m_stack_div_by_row, smap_400m_stack_complete,
            tif_file_smap_400m_path, tif_file_smap_400m_name, tif_file_smap_400m_name_output)


profile = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': 0.0, 'width': len(lon_400m),
           'height': len(lat_400m), 'count': 2, 'crs': CRS.from_dict(init='epsg:6933'),
           'transform': Affine(interdist_ease_400m, 0.0, lon_400m_min-interdist_ease_400m/2,
                               0.0, -interdist_ease_400m, lat_400m_max+interdist_ease_400m/2)}
# AM/PM overpass
for iyr in [13, 14]:#range(5, len(yearname)):

    path_output_1year = path_output + '/colorado/smap_400m/' + str(yearname[iyr])
    if os.path.exists(path_output_1year) == False:
        os.makedirs(path_output_1year)
    else:
        pass

    for idt in range(0, daysofyear[iyr]):
        smap_400m_stack = []
        for ite in range(len(tile_num)):
            tif_file_smap_400m_name = ('smap_sm_400m_' + str(yearname[iyr]) + str(idt+1).zfill(3) + '_T' +
                                       str(tile_num[ite]).zfill(3) + '.tif')
            tif_file_smap_400m_path = (path_smap_400m + str(yearname[iyr]) + '/T' + str(tile_num[ite]).zfill(3) + '/'
                                  + tif_file_smap_400m_name)
            if os.path.exists(tif_file_smap_400m_path):
                src_tf = gdal.Open(tif_file_smap_400m_path).ReadAsArray()
            else:
                src_tf = viirs_mat_fill
            src_tf_arr = src_tf[:, row_extent_all[ite][0]:row_extent_all[ite][-1]+1,
                         col_extent_all[ite][0]:col_extent_all[ite][-1]+1]
            src_tf_arr_avg = src_tf_arr
            # src_tf_arr_avg = np.nanmean(src_tf_arr, axis=0)
            smap_400m_stack.append(src_tf_arr_avg)
            del(src_tf_arr_avg, src_tf, src_tf_arr)

        smap_400m_stack_div = [smap_400m_stack[i * len(tiles_num_row) : (i+1) * len(tiles_num_row)]
                               for i in range(len(tiles_num_col))]
        smap_400m_stack_div_by_row = [np.hstack(smap_400m_stack_div[x]) for x in range(len(smap_400m_stack_div))]
        smap_400m_stack_complete = np.vstack(smap_400m_stack_div_by_row)
        # smap_400m_stack_complete = np.expand_dims(smap_400m_stack_complete, axis=0)

        tif_file_smap_400m_name_output = tif_file_smap_400m_name.split('.')[0][:-5]
        dst_writer = rasterio.open(path_output_1year + '/' + tif_file_smap_400m_name_output + '.tif', 'w', **profile)
        dst_writer.write(smap_400m_stack_complete)
        dst_writer = None

        print(tif_file_smap_400m_name_output)
        del(smap_400m_stack, smap_400m_stack_div, smap_400m_stack_div_by_row, smap_400m_stack_complete,
            tif_file_smap_400m_path, tif_file_smap_400m_name, tif_file_smap_400m_name_output)




# 11.  Repair data of 1 km (daily, SMAP)

ds_smap_sm_1km = gdal.Open(path_smap + '/1km/' + str(yearname[iyr]) + '/smap_sm_1km_ds_2023272.tif')
prj = ds_smap_sm_1km.GetProjection()


for iyr in [14]:#range(5, len(yearname)):
    smap_file_path = path_smap + '/1km/' + str(yearname[iyr])
    smap_file_list = sorted(glob.glob(smap_file_path + '/*'))

    for idt in range(0, len(smap_file_list)):
        filename = os.path.basename(smap_file_list[idt])
        ds_smap_sm_1km = gdal.Open(smap_file_list[idt])
        smap_sm_1km_ds_output = ds_smap_sm_1km.ReadAsArray()
        smap_sm_1km_ds_output = np.transpose(smap_sm_1km_ds_output, (1, 2, 0))

        # Save the daily 1 km SM model output to Geotiff files
        # Build output path
        # Create a raster of EASE grid projection at 1 km resolution
        out_ds_tiff = gdal.GetDriverByName('GTiff').Create \
            (path_smap + '/1km_rivanna/' + str(yearname[iyr]) + '/' + filename,
             len(lon_world_ease_1km), len(lat_world_ease_1km), 2,  # Number of bands
             gdal.GDT_Float32, ['COMPRESS=LZW', 'TILED=YES'])
        out_ds_tiff.SetGeoTransform(ds_smap_sm_1km.GetGeoTransform())
        out_ds_tiff.SetProjection(prj)

        # Loop write each band to Geotiff file
        for idf in range(2):
            out_ds_tiff.GetRasterBand(idf + 1).WriteArray(smap_sm_1km_ds_output[:, :, idf])
            out_ds_tiff.GetRasterBand(idf + 1).SetNoDataValue(0)
        out_ds_tiff = None  # close dataset to write to disc

        print(filename)
        del (filename, smap_sm_1km_ds_output, ds_smap_sm_1km, out_ds_tiff)

