import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import calendar
import datetime
import glob
import rasterio
from osgeo import gdal
from rasterio.windows import Window
from pyproj import Transformer
import pandas as pd
from scipy import stats
import skill_metrics as sm
# import mat73
import itertools
plt.rcParams["font.family"] = "serif"
# Ignore runtime warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


#########################################################################################################
# (Function 1) Define a function for reading and extracting useful information from each ISMN in-situ data file

def insitu_extraction(filepath):
    # Read each .stm file by line
    with open(filepath, "r") as ins:
        data_list = []
        for line in ins:
            data_list.append(line)
    ins.close()

    # Extract lat/lon, network and station name information from the first line of the current file
    net_name = data_list[0].split()[4]
    stn_name = data_list[0].split()[6]
    stn_lat = float(data_list[0].split()[7])
    stn_lon = float(data_list[0].split()[8])
    # Find the correct standard UTC time zone by lat/lon and convert to local time
    sign = stn_lon//abs(stn_lon)
    timezone_offset = (abs(stn_lon) + 7.5) // 15 * sign
    smap_overpass_correct = smap_overpass + (timezone_offset * -1.0)
    smap_overpass_correct = np.array(smap_overpass_correct, dtype=int)

    # Determine if the UTC time of the local area is one day before/after the current
    if smap_overpass_correct[0] < 0:
        smap_overpass_correct[0] = 24 + smap_overpass_correct[0]
        am_offset = 1
    else:
        am_offset = 0
    if smap_overpass_correct[1] >= 24:
        smap_overpass_correct[1] = smap_overpass_correct[1] - 24
        pm_offset = -1
    else:
        pm_offset = 0
    timezone_offset = [am_offset, pm_offset]

    smap_overpass_correct = [str(smap_overpass_correct[0]).zfill(2) + ':00',
                             str(smap_overpass_correct[1]).zfill(2) + ':00']

    # Extract 6 AM/PM SM from current file
    # sm_array = np.empty((2, len(date_seq)), dtype='float32')  # 2-dim for storing AM/PM overpass SM
    # sm_array[:] = np.nan
    # sm_array = np.copy(sm_array_init)

    sm_array_all = []
    for itm in range(len(smap_overpass_correct)):
        sm_array = np.empty((len(date_seq)), dtype='float32')
        sm_array[:] = np.nan
        datatime_match_ind = [data_list.index(i) for i in data_list if smap_overpass_correct[itm] in i]
        datatime_match = [data_list[datatime_match_ind[i]] for i in range(len(datatime_match_ind))]
        datatime_match_date = [datatime_match[i].split()[0].replace('/', '') for i in range(len(datatime_match))]

        datatime_match_date_ind = [datatime_match_date.index(item) for item in datatime_match_date if item in date_seq]
        datatime_match_date_seq_ind = [date_seq.index(item) for item in datatime_match_date if item in date_seq]
        datatime_match_date_seq_ind = np.array(datatime_match_date_seq_ind)
        datatime_match_date_seq_ind = datatime_match_date_seq_ind + timezone_offset[itm] #adjust by timezone offset of am/pm
        datatime_match_date_seq_ind = \
            datatime_match_date_seq_ind[(datatime_match_date_seq_ind >= 0) | (datatime_match_date_seq_ind < len(date_seq))]

        if len(datatime_match_date_ind) != 0:
            # Find the data values from the in situ data file
            sm_array_ext = [float(datatime_match[datatime_match_date_ind[i]].split()[12])
                            for i in range(len(datatime_match_date_ind))]
            # sm_array[itm, datatime_match_date_seq_ind] = sm_array_ext
            # Fill the data values to the corresponding place of date_seq
            sm_array[datatime_match_date_seq_ind] = sm_array_ext

        else:
            sm_array_ext = []
            pass

        sm_array_all.append(sm_array)

        del(datatime_match_ind, datatime_match, datatime_match_date, datatime_match_date_ind, sm_array_ext, sm_array)

    sm_array_all = np.stack(sm_array_all, axis=0)
    sm_array_all[sm_array_all < 0] = np.nan

    return net_name, stn_name, stn_lat, stn_lon, sm_array_all


####################################################################################################################################

# 0. Input variables
# Specify file paths
# Path of current workspace
path_workspace = '/Users/binfang/Documents/SMAP_Project/smap_codes'
# Path of model data
path_model = '/Volumes/Elements/Datasets/model_data'
# Path of results
path_results = '/Users/binfang/Documents/SMAP_Project/results/results_241116'
# Path of SMAP SM
path_smap = '/Volumes/Elements/Datasets/SMAP'
# Path of SMOS SM
# path_smap = '/Volumes/Elements/Datasets/SMOS'
# Path of downscaled SM by viirs
path_smap_viirs = '/Volumes/Elements2/SMAP/SM_downscaled_gldas/'
path_smap_viirs_era = '/Volumes/Elements2/SMAP/SM_downscaled_era/'
# Path of validation data
path_validation = '/Volumes/Elements/Datasets/processed_data'
# Path of ISMN data
path_ismn = '/Volumes/Elements/Datasets/ISMN'
# Path of GPM
path_gpm = '/Volumes/Elements/Datasets/GPM'
# Path of mask
path_lmask = '/Volumes/Elements/Datasets/Lmask'

lst_folder = '/MYD11A1/'
ndvi_folder = '/MYD13A2/'
smap_sm_9km_name = ['smap_sm_9km_am_slice', 'smap_sm_9km_pm_slice']
# region_name = ['Africa', 'Asia', 'Europe', 'NorthAmerica', 'Oceania', 'SouthAmerica']
# smap_overpass = ['06:00', '18:00']
smap_overpass = np.array([6, 18], dtype='int')
yearname = np.linspace(2010, 2023, 14, dtype='int')
monthnum = np.linspace(1, 12, 12, dtype='int')
monthname = np.arange(1, 13)
monthname = [str(i).zfill(2) for i in monthname]

# Generate a sequence of string between start and end dates (Year + DOY)
start_date = '2010-01-01'
end_date = '2023-12-31'
year = 2023 - 2010 + 1

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
yearname = np.linspace(2010, 2023, 14, dtype='int')
monthnum = np.linspace(1, 12, 12, dtype='int')
monthname = np.arange(1, 13)
monthname = [str(i).zfill(2) for i in monthname]

daysofyear = []
for idt in range(len(yearname)):
    if idt == 5:
        f_date = datetime.date(yearname[idt], monthnum[3], 1)
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

daysofmonth_seq_cumsum = np.cumsum(daysofmonth_seq, axis=0)
daysofmonth_seq_cumsum = np.concatenate((np.zeros((1, 14), dtype=int), daysofmonth_seq_cumsum), axis=0)
# ind_init = daysofmonth_seq_cumsum[2, :]
# ind_end = daysofmonth_seq_cumsum[8, :]
# ind_gpm = np.stack((ind_init, ind_end), axis=1)

# Load in geo-location parameters
os.chdir(path_workspace)
f = h5py.File("ds_parameters.hdf5", "r")
varname_list = ['lat_world_max', 'lat_world_min', 'lon_world_max', 'lon_world_min', 'lat_world_ease_1km',
                'lon_world_ease_1km', 'row_world_ease_1km_from_9km_ind', 'col_world_ease_1km_from_9km_ind',
                'lat_world_geo_400m', 'lon_world_geo_400m', 'lat_world_ease_400m', 'lon_world_ease_400m',
                'lat_world_ease_9km', 'lon_world_ease_9km', 'lat_world_ease_12_5km', 'lon_world_ease_12_5km',
                'lat_world_geo_10km', 'lon_world_geo_10km',
                'row_world_ease_400m_ind', 'col_world_ease_400m_ind', 'row_world_ease_9km_ind', 'col_world_ease_9km_ind']

for x in range(len(varname_list)):
    var_obj = f[varname_list[x]][()]
    exec(varname_list[x] + '= var_obj')
    del(var_obj)
f.close()


####################################################################################################################################
# 1.1 Extract SM in situ data from ISMN .stm files
# path_ismn = '/Users/binfang/Downloads/ISMN'
# region_name = 'Africa'

columns = ['lat', 'lon'] + date_seq_doy
# folder_region = os.listdir(path_ismn + '/Ver_2/')
# folder_region = sorted(folder_region)
folder_region = sorted(glob.glob(path_ismn + '/original_data/*'))
region_name = [folder_region[x].split('/')[-1] for x in range(len(folder_region))]

for ire in range(len(folder_region)): # Region (Continent) folders
    folder_network = sorted([name for name in os.listdir(folder_region[ire])
                             if os.path.isdir(os.path.join(folder_region[ire], name))])

    for inw in range(len(folder_network)): # Network folders
        folder_site = sorted([name for name in os.listdir(folder_region[ire] + '/' + folder_network[inw])
                       if os.path.isdir(os.path.join(folder_region[ire]+ '/' + folder_network[inw], name))])

        stn_name_all = []
        stn_lat_all = []
        stn_lon_all = []
        sm_array_am_all = []
        sm_array_pm_all = []
        for ist in range(len(folder_site)): # Site folders
            sm_file_path = folder_region[ire] + '/' + folder_network[inw] + '/' + folder_site[ist]
            sm_file_list = sorted(glob.glob(sm_file_path + '/*_sm_*'))
            if len(sm_file_list) != 0:
                sm_file = sm_file_list[0]
                net_name, stn_name, stn_lat, stn_lon, sm_array = insitu_extraction(sm_file)
                sm_array_am = sm_array[0, :]
                sm_array_pm = sm_array[1, :]

                stn_name_all.append(stn_name)
                stn_lat_all.append(stn_lat)
                stn_lon_all.append(stn_lon)
                sm_array_am_all.append(sm_array_am)
                sm_array_pm_all.append(sm_array_pm)
                print(sm_file)
            else:
                pass

        sm_mat_am = np.concatenate(
            (np.expand_dims(np.array(stn_lat_all), axis=1),
             np.expand_dims(np.array(stn_lon_all), axis=1),
             np.stack(sm_array_am_all)),
            axis=1)
        sm_mat_pm = np.concatenate(
            (np.expand_dims(np.array(stn_lat_all), axis=1),
             np.expand_dims(np.array(stn_lon_all), axis=1),
             np.stack(sm_array_pm_all)),
            axis=1)

        df_sm_am = pd.DataFrame(sm_mat_am, columns=columns, index=stn_name_all)
        df_sm_pm = pd.DataFrame(sm_mat_pm, columns=columns, index=stn_name_all)
        writer = pd.ExcelWriter(path_ismn + '/processed_data/' + region_name[ire] + '_' + folder_network[inw] + '_' +
                                'ismn_sm.xlsx')
        df_sm_am.to_excel(writer, sheet_name='AM')
        df_sm_pm.to_excel(writer, sheet_name='PM')
        writer.save()

        del(sm_mat_am, sm_mat_pm, df_sm_am, df_sm_pm, writer)



# 1.2 Extract Land cover types and main soil types
# folder_region = sorted(glob.glob('/Volumes/MyPassport/SMAP_Project/Datasets/ISMN/*/'))
folder_region = sorted(glob.glob(path_ismn + '/original_data/*'))
region_name = [folder_region[x].split('/')[-1] for x in range(len(folder_region))]

for ire in range(len(folder_region)): # Region folders
    folder_network = sorted([name for name in os.listdir(folder_region[ire])
                             if os.path.isdir(os.path.join(folder_region[ire], name))])

    for inw in range(len(folder_network)): # Network folders
        folder_site = sorted([name for name in os.listdir(folder_region[ire] + '/' + folder_network[inw])
                       if os.path.isdir(os.path.join(folder_region[ire] + '/' + folder_network[inw], name))])

        stn_name_all = []
        landcover_all = []
        soiltype_all = []
        climate_class_all = []
        for ist in range(len(folder_site)): # Site folders
            csv_file_path = folder_region[ire] + '/' + folder_network[inw] + '/' + folder_site[ist]
            csv_file_list = glob.glob(csv_file_path + '/*.csv')
            # stn_name = folder_site[ist]
            sm_file_list = glob.glob(csv_file_path + '/*_sm_*')

            if len(csv_file_list) != 0 and len(sm_file_list) != 0:
                csv_file = csv_file_list[0]
                df_file = pd.read_csv(csv_file, index_col=0, sep=';')
                if len(df_file.index) >= 14:
                    sm_file = sm_file_list[0]
                    net_name, stn_name, stn_lat, stn_lon, sm_array = insitu_extraction(sm_file)
                    landcover = df_file.loc['land cover classification']['description'][0]
                    soiltype_array = df_file.loc[['clay fraction', 'sand fraction', 'silt fraction']]['value'][[0, 2, 4]]
                    soiltype_ratio = soiltype_array
                    # soiltype_array = [float(soiltype_array[x]) for x in range(len(soiltype_array))]
                    # soiltype_ratio = np.array([soiltype_array[x*2]+soiltype_array[x*2+1] for x in range(2)])
                    soiltype = ['clay fraction', 'sand fraction', 'silt fraction']\
                        [np.where(soiltype_ratio == np.max(soiltype_ratio))[0][0].item()]
                    climate_class = df_file.loc['climate classification']['description'][0]

                    landcover_all.append(landcover)
                    soiltype_all.append(soiltype)
                    climate_class_all.append(climate_class)
                    # stn_name_all.append(stn_name)
                    print(csv_file_list[0])

                else:
                    landcover_all.append('')
                    soiltype_all.append('')
                    climate_class_all.append('')

                stn_name_all.append(stn_name)

            else:
                pass

        df_landcover = pd.DataFrame({'land cover': landcover_all, 'soiltype': soiltype_all, 'climate': climate_class_all},
                                    index=stn_name_all)
        writer = pd.ExcelWriter(path_ismn + '/landcover/' + region_name[ire] + '_' + folder_network[inw] + '_' + 'landcover.xlsx')
        df_landcover.to_excel(writer)
        writer.save()

        del(df_landcover, writer, landcover_all, soiltype_all, climate_class_all)


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

####################################################################################################################################
# 1.1 Load the site lat/lon from Excel files and Locate the 1/9 km SM positions by lat/lon of ISMN in-situ data

ismn_list = sorted(glob.glob(path_ismn + '/processed_data/[A-Z]*.xlsx'))

coords_all = []
df_table_am_all = []
df_table_pm_all = []
for ife in range(len(ismn_list)):
    df_table_am = pd.read_excel(ismn_list[ife], index_col=0, sheet_name='AM')
    df_table_pm = pd.read_excel(ismn_list[ife], index_col=0, sheet_name='PM')

    contname = os.path.basename(ismn_list[ife]).split('_')[0]
    contname = [contname] * df_table_am.shape[0]
    netname = os.path.basename(ismn_list[ife]).split('_')[1]
    netname = [netname] * df_table_am.shape[0]
    coords = df_table_am[['lat', 'lon']]
    coords_all.append(coords)

    df_table_am_value = df_table_am.iloc[:, 2:]
    df_table_am_value.insert(0, 'continent', contname)
    df_table_am_value.insert(1, 'network', netname)
    df_table_pm_value = df_table_pm.iloc[:, 2:]
    df_table_pm_value.insert(0, 'continent', contname)
    df_table_pm_value.insert(1, 'network', netname)
    df_table_am_all.append(df_table_am_value)
    df_table_pm_all.append(df_table_pm_value)
    del(df_table_am, df_table_pm, df_table_am_value, df_table_pm_value, coords, netname, contname)
    print(ife)

df_coords = pd.concat(coords_all)
df_table_am_all = pd.concat(df_table_am_all)
df_table_pm_all = pd.concat(df_table_pm_all)

new_index = [df_coords.index[x] for x in range(len(df_coords.index))] # Capitalize each word
# new_index = [df_coords.index[x].title() for x in range(len(df_coords.index))]
df_coords.index = new_index
df_table_am_all.index = new_index
df_table_pm_all.index = new_index
df_coords = pd.concat([df_table_am_all['continent'], df_table_am_all['network'], df_coords], axis=1)

df_table_am_insitu = df_table_am_all.drop(df_table_am_all.columns[2:1918], axis=1)
df_table_pm_insitu = df_table_pm_all.drop(df_table_pm_all.columns[2:1918], axis=1)

# writer = pd.ExcelWriter(path_results + '/validation/df_coords.xlsx')
# df_coords.to_excel(writer)
# writer.save()

# 1.2 Locate the SMAP 1/9 km and 400 m SM positions by lat/lon of in-situ data
stn_lat_all = np.array(df_coords['lat'])
stn_lon_all = np.array(df_coords['lon'])

stn_row_400m_ind_all = []
stn_col_400m_ind_all = []
stn_row_1km_ind_all = []
stn_col_1km_ind_all = []
stn_row_9km_ind_all = []
stn_col_9km_ind_all = []
for idt in range(len(stn_lat_all)):
    stn_row_400m_ind = np.argmin(np.absolute(stn_lat_all[idt] - lat_world_ease_400m)).item()
    stn_col_400m_ind = np.argmin(np.absolute(stn_lon_all[idt] - lon_world_ease_400m)).item()
    stn_row_400m_ind_all.append(stn_row_400m_ind)
    stn_col_400m_ind_all.append(stn_col_400m_ind)
    stn_row_1km_ind = np.argmin(np.absolute(stn_lat_all[idt] - lat_world_ease_1km)).item()
    stn_col_1km_ind = np.argmin(np.absolute(stn_lon_all[idt] - lon_world_ease_1km)).item()
    stn_row_1km_ind_all.append(stn_row_1km_ind)
    stn_col_1km_ind_all.append(stn_col_1km_ind)
    stn_row_9km_ind = np.argmin(np.absolute(stn_lat_all[idt] - lat_world_ease_9km)).item()
    stn_col_9km_ind = np.argmin(np.absolute(stn_lon_all[idt] - lon_world_ease_9km)).item()
    stn_row_9km_ind_all.append(stn_row_9km_ind)
    stn_col_9km_ind_all.append(stn_col_9km_ind)
    del(stn_row_400m_ind, stn_col_400m_ind, stn_row_1km_ind, stn_col_1km_ind, stn_row_9km_ind, stn_col_9km_ind)


# Convert from Lat/Lon coordinates to EASE grid projection meter units
transformer = Transformer.from_crs("epsg:4326", "epsg:6933", always_xy=True)
[stn_lon_all_ease, stn_lat_all_ease] = transformer.transform(stn_lon_all, stn_lat_all)
coords_zip = list(map(list, zip(stn_lon_all_ease, stn_lat_all_ease)))

########################################################################################################################
# 2. Extract the SMAP 1/9 km SM by the indexing files

# 2.1 Extract 1km SMAP SM
smap_1km_sta_all = []
tif_files_1km_name_ind_all = []
for iyr in range(5, len(yearname)):

    os.chdir(path_smap +'/1km/' + str(yearname[iyr]))
    tif_files = sorted(glob.glob('*.tif'))

    # Extract the file name
    tif_files_name = [os.path.splitext(tif_files[x])[0].split('_')[-1] for x in range(len(tif_files))]
    tif_files_name_1year_ind = [date_seq_doy.index(item) for item in tif_files_name if item in date_seq_doy]
    date_seq_doy_1year_ind = [tif_files_name.index(item) for item in tif_files_name if item in date_seq_doy]

    tif_files_1km_name_ind_all.append(tif_files_name_1year_ind)
    del(tif_files_name, tif_files_name_1year_ind)

    smap_1km_sta_1year = []
    for idt in range(len(date_seq_doy_1year_ind)):
        # src_tf = rasterio.open(tif_files[date_seq_doy_1year_ind[idt]]).read()
        # smap_1km_sta_1day = src_tf[:, stn_row_1km_ind_all, stn_col_1km_ind_all]
        src_tf = rasterio.open(tif_files[date_seq_doy_1year_ind[idt]])
        smap_1km_sta_1day_am = np.array([sample[0] for sample in src_tf.sample(coords_zip, indexes=1)])
        smap_1km_sta_1day_pm = np.array([sample[0] for sample in src_tf.sample(coords_zip, indexes=2)])
        smap_1km_sta_1day = np.stack((smap_1km_sta_1day_am, smap_1km_sta_1day_pm), axis=0)

        smap_1km_sta_1year.append(smap_1km_sta_1day)
        del(src_tf, smap_1km_sta_1day, smap_1km_sta_1day_am, smap_1km_sta_1day_pm)
        print(tif_files[date_seq_doy_1year_ind[idt]])

    smap_1km_sta_all.append(smap_1km_sta_1year)
    del(smap_1km_sta_1year, date_seq_doy_1year_ind)


tif_files_1km_name_ind_all = np.concatenate(tif_files_1km_name_ind_all)
# tif_files_1km_name_ind_all = tif_files_1km_name_ind_all - 4383
smap_1km_sta_all = np.concatenate(smap_1km_sta_all)

# Fill the extracted SMAP SM into the proper position of days
smap_1km_sta_am = np.empty((df_table_am_all.shape[0], df_table_am_all.shape[1]-2), dtype='float32')
smap_1km_sta_am[:] = np.nan
for idt in range(len(tif_files_1km_name_ind_all)):
    smap_1km_sta_am[:, tif_files_1km_name_ind_all[idt]] = smap_1km_sta_all[idt, 0, :]

smap_1km_sta_pm = np.empty((df_table_pm_all.shape[0], df_table_pm_all.shape[1]-2), dtype='float32')
smap_1km_sta_pm[:] = np.nan
for idt in range(len(tif_files_1km_name_ind_all)):
    smap_1km_sta_pm[:, tif_files_1km_name_ind_all[idt]] = smap_1km_sta_all[idt, 1, :]

index_validation = df_table_am_all.index
columns_validation = df_table_am_all.columns
continent_validation = df_table_am_all[['continent']]
network_validation = df_table_am_all[['network']]

smap_1km_sta_am = pd.DataFrame(smap_1km_sta_am, columns=date_seq_doy, index=index_validation)
smap_1km_sta_am = smap_1km_sta_am.iloc[:, 1916:]
df_smap_1km_sta_am = pd.concat([continent_validation, network_validation, smap_1km_sta_am], axis=1, sort=False)
smap_1km_sta_pm = pd.DataFrame(smap_1km_sta_pm, columns=date_seq_doy, index=index_validation)
smap_1km_sta_pm = smap_1km_sta_pm.iloc[:, 1916:]
df_smap_1km_sta_pm = pd.concat([continent_validation, network_validation, smap_1km_sta_pm], axis=1, sort=False)

# # Drop the days after March 31st, 2015
# columns_drop = columns_validation[1918:]
# df_smap_1km_sta_am = df_smap_1km_sta_am.drop(columns=columns_drop)
# df_smap_1km_sta_pm = df_smap_1km_sta_pm.drop(columns=columns_drop)
# df_insitu_table_am_all = df_table_am_all.drop(columns=columns_drop)
# df_insitu_table_pm_all = df_table_pm_all.drop(columns=columns_drop)


# 2.2 Extract 9km SMAP SM
smap_9km_sta_am = []
smap_9km_sta_pm = []
for iyr in range(5, len(yearname)):

    smap_9km_sta_am_1year = []
    smap_9km_sta_pm_1year = []
    for imo in range(len(monthname)):

        smap_9km_sta_am_1month = []
        smap_9km_sta_pm_1month = []
        # Load in SMOS 25km SM data
        smap_file_path = path_smap + '/9km/' + 'smap_sm_9km_' + str(yearname[iyr]) + monthname[imo] + '.hdf5'

        # Find the if the file exists in the directory
        if os.path.exists(smap_file_path) == True:

            f_smap_9km = h5py.File(smap_file_path, "r")
            varname_list_smap = list(f_smap_9km.keys())
            smap_9km_sta_am_1month = f_smap_9km[varname_list_smap[0]][()]
            smap_9km_sta_am_1month = smap_9km_sta_am_1month[stn_row_9km_ind_all, stn_col_9km_ind_all, :]
            smap_9km_sta_pm_1month = f_smap_9km[varname_list_smap[1]][()]
            smap_9km_sta_pm_1month = smap_9km_sta_pm_1month[stn_row_9km_ind_all, stn_col_9km_ind_all, :]

            print(smap_file_path)
            f_smap_9km.close()

        else:
            pass

        smap_9km_sta_am_1year.append(smap_9km_sta_am_1month)
        smap_9km_sta_pm_1year.append(smap_9km_sta_pm_1month)
        del(smap_9km_sta_am_1month, smap_9km_sta_pm_1month)

    smap_9km_sta_am.append(smap_9km_sta_am_1year)
    smap_9km_sta_pm.append(smap_9km_sta_pm_1year)
    del(smap_9km_sta_am_1year, smap_9km_sta_pm_1year)

# Remove the empty lists
# smap_9km_sta_am[5] = smap_9km_sta_am[5][0:3]
# smap_9km_sta_pm[5] = smap_9km_sta_pm[5][0:3]

smap_9km_sta_am = list(itertools.chain(*smap_9km_sta_am))
smap_9km_sta_am = smap_9km_sta_am[3:]
smap_9km_sta_am = np.concatenate(smap_9km_sta_am, axis=1)
smap_9km_sta_pm = list(itertools.chain(*smap_9km_sta_pm))
smap_9km_sta_pm = smap_9km_sta_pm[3:]
smap_9km_sta_pm = np.concatenate(smap_9km_sta_pm, axis=1)

smap_9km_sta_am = pd.DataFrame(smap_9km_sta_am, columns=date_seq_doy[1916:], index=index_validation)
df_smap_9km_sta_am = pd.concat([continent_validation, network_validation, smap_9km_sta_am], axis=1, sort=False)
smap_9km_sta_pm = pd.DataFrame(smap_9km_sta_pm, columns=date_seq_doy[1916:], index=index_validation)
df_smap_9km_sta_pm = pd.concat([continent_validation, network_validation, smap_9km_sta_pm], axis=1, sort=False)


# 2.3 Extract 400 m SMAP SM
# stn_row_400m_ind_all = []
# stn_col_400m_ind_all = []

stn_row_400m_ind_all_local = df_row_world_ease_400m_ind.iloc[stn_row_400m_ind_all, :]
stn_col_400m_ind_all_local = df_col_world_ease_400m_ind.iloc[stn_col_400m_ind_all, :]
stn_row_400m_ind_all_local.reset_index(drop=True, inplace=True)
stn_col_400m_ind_all_local.reset_index(drop=True, inplace=True)

stn_coord_400m_ind_all_local = pd.concat([stn_row_400m_ind_all_local, stn_col_400m_ind_all_local], axis=1)
tiles_num = stn_coord_400m_ind_all_local['row_ind_tile'] * 24 + stn_coord_400m_ind_all_local['col_ind_tile'] + 1
stn_coord_400m_ind_all_local['tiles_num'] = tiles_num
stn_coord_400m_ind_all_local_group = stn_coord_400m_ind_all_local.groupby(by='tiles_num').groups
stn_coord_400m_ind_all_local_group_ind = [value[1].tolist() for value in stn_coord_400m_ind_all_local_group.items()]
stn_coord_400m_ind_all_local_group_keys = [value for value in stn_coord_400m_ind_all_local_group.keys()]


smap_400m_sta_allyears = []
for iyr in range(5, len(yearname)):

    smap_400m_sta_all = []
    # tif_files_400m_name_ind_all = []

    for ite in range(len(stn_coord_400m_ind_all_local_group_keys)):
        path_tile_read = (path_smap_viirs + str(yearname[iyr]) + '/T' +
                          str(stn_coord_400m_ind_all_local_group_keys[ite]).zfill(3))

        if os.path.exists(path_tile_read) == True:
            os.chdir(path_tile_read)
            tif_files = sorted(glob.glob('*.tif'))

            # Extract the file name
            tif_files_name = [os.path.splitext(tif_files[x])[0].split('_')[-2] for x in range(len(tif_files))]
            tif_files_name_1year_ind = [date_seq_doy.index(item) for item in tif_files_name if item in date_seq_doy]
            date_seq_doy_1year_ind = [tif_files_name.index(item) for item in tif_files_name if item in date_seq_doy]

            # tif_files_400m_name_ind_all.append(tif_files_name_1year_ind)
            del(tif_files_name, tif_files_name_1year_ind)

            smap_400m_sta_1year = []
            for idt in range(len(date_seq_doy_1year_ind)):
                stn_row_400m_ind_all_tile_local = \
                stn_coord_400m_ind_all_local.iloc[stn_coord_400m_ind_all_local_group_ind[ite]]['row_ind_local'].tolist()
                stn_col_400m_ind_all_tile_local = \
                stn_coord_400m_ind_all_local.iloc[stn_coord_400m_ind_all_local_group_ind[ite]]['col_ind_local'].tolist()
                src_tf = rasterio.open(tif_files[date_seq_doy_1year_ind[idt]]).read()
                smap_400m_sta_1day = src_tf[:, stn_row_400m_ind_all_tile_local, stn_col_400m_ind_all_tile_local]
                # src_tf = rasterio.open(tif_files[date_seq_doy_1year_ind[idt]])
                # smap_400m_sta_1day_am = np.array([sample[0] for sample in src_tf.sample(coords_zip, indexes=1)])
                # smap_400m_sta_1day_pm = np.array([sample[0] for sample in src_tf.sample(coords_zip, indexes=2)])
                # smap_400m_sta_1day = np.stack((smap_400m_sta_1day_am, smap_400m_sta_1day_pm), axis=0)

                smap_400m_sta_1year.append(smap_400m_sta_1day)
                del(src_tf, smap_400m_sta_1day)
                print(tif_files[date_seq_doy_1year_ind[idt]])

            smap_400m_sta_1year = np.stack(smap_400m_sta_1year, axis=2)
            smap_400m_sta_all.append(smap_400m_sta_1year)
            del(smap_400m_sta_1year, date_seq_doy_1year_ind)

        else:
            smap_400m_sta_1year = np.empty((2, len(stn_coord_400m_ind_all_local_group_ind[ite]), daysofyear[iyr]))
            smap_400m_sta_1year[:] = np.nan
            smap_400m_sta_all.append(smap_400m_sta_1year)

    smap_400m_sta_allyears.append(smap_400m_sta_all)


# tif_files_400m_name_ind_all = np.concatenate(tif_files_400m_name_ind_all)
# tif_files_400m_name_ind_all = tif_files_400m_name_ind_all[0:183]
# tif_files_400m_name_ind_all = list(np.arange(92, 275))

smap_400m_sta_all_combine = [np.concatenate(smap_400m_sta_allyears[x], axis=1) for x in range(len(smap_400m_sta_allyears))]
smap_400m_sta_all_combine = np.concatenate(smap_400m_sta_all_combine, axis=2)

# # Fill the extracted SMOS SM into the proper position of days
# smap_400m_sta_am = np.empty((df_table_am_all.shape[0], df_table_am_all.shape[1]-2), dtype='float32')
# smap_400m_sta_am[:] = np.nan
# for idt in range(len(tif_files_400m_name_ind_all)):
#     smap_400m_sta_am[:, tif_files_400m_name_ind_all[idt]] = smap_400m_sta_all_comb[0, :, idt]
#
# smap_400m_sta_pm = np.empty((df_table_pm_all.shape[0], df_table_pm_all.shape[1]-2), dtype='float32')
# smap_400m_sta_pm[:] = np.nan
# for idt in range(len(tif_files_400m_name_ind_all)):
#     smap_400m_sta_pm[:, tif_files_400m_name_ind_all[idt]] = smap_400m_sta_all_comb[1, :, idt]

# Reposit the stations to the correct positions
stn_coord_400m_ind_all_local_group_ind_flat = list(itertools.chain(*stn_coord_400m_ind_all_local_group_ind))

smap_400m_sta_am_rep = np.empty((smap_400m_sta_all_combine.shape[1], smap_400m_sta_all_combine.shape[2]), dtype='float32')
for idt in range(smap_400m_sta_all_combine.shape[1]):
    smap_400m_sta_am_rep[stn_coord_400m_ind_all_local_group_ind_flat[idt], :] = smap_400m_sta_all_combine[0, idt, :]

smap_400m_sta_pm_rep = np.empty((smap_400m_sta_all_combine.shape[1], smap_400m_sta_all_combine.shape[2]), dtype='float32')
for idt in range(smap_400m_sta_all_combine.shape[1]):
    smap_400m_sta_pm_rep[stn_coord_400m_ind_all_local_group_ind_flat[idt], :] = smap_400m_sta_all_combine[1, idt, :]

smap_400m_sta_rep = np.nanmean(np.stack((smap_400m_sta_am_rep, smap_400m_sta_pm_rep), axis=1), axis=1)

index_validation = df_table_am_all.index
# columns_validation = df_table_am_all.columns
continent_validation = df_table_am_all[['continent']]
network_validation = df_table_am_all[['network']]

smap_400m_sta_am = pd.DataFrame(smap_400m_sta_am_rep, columns=date_seq_doy[3287:], index=index_validation)
df_smap_400m_sta_am = pd.concat([continent_validation, network_validation, smap_400m_sta_am], axis=1, sort=False)
smap_400m_sta_pm = pd.DataFrame(smap_400m_sta_pm_rep, columns=date_seq_doy[3287:], index=index_validation)
df_smap_400m_sta_pm = pd.concat([continent_validation, network_validation, smap_400m_sta_pm], axis=1, sort=False)


# Save variables
writer_insitu = pd.ExcelWriter(path_ismn + '/extraction/smap_validation_insitu.xlsx')
writer_400m = pd.ExcelWriter(path_ismn + '/extraction/smap_validation_400m_era.xlsx')
writer_1km = pd.ExcelWriter(path_ismn + '/extraction/smap_validation_1km.xlsx')
writer_9km = pd.ExcelWriter(path_ismn + '/extraction/smap_validation_9km.xlsx')
df_table_am_insitu.to_excel(writer_insitu, sheet_name='AM')
df_table_pm_insitu.to_excel(writer_insitu, sheet_name='PM')
df_smap_400m_sta_am.to_excel(writer_400m, sheet_name='AM')
df_smap_400m_sta_pm.to_excel(writer_400m, sheet_name='PM')
df_smap_1km_sta_am.to_excel(writer_1km, sheet_name='AM')
df_smap_1km_sta_pm.to_excel(writer_1km, sheet_name='PM')
df_smap_9km_sta_am.to_excel(writer_9km, sheet_name='AM')
df_smap_9km_sta_pm.to_excel(writer_9km, sheet_name='PM')
writer_insitu.save()
writer_400m.save()
writer_1km.save()
writer_9km.save()


########################################################################################################################
# 3. Plot validation results between 1 km, 9 km and in-situ data

df_smap_insitu_sta_am = pd.read_excel(path_ismn + '/extraction/smap_validation_insitu.xlsx', index_col=0, sheet_name='AM')
df_smap_400m_sta_am = pd.read_excel(path_ismn + '/extraction/smap_validation_400m.xlsx', index_col=0, sheet_name='AM')
df_smap_400m_era_sta_am = pd.read_excel(path_ismn + '/extraction/smap_validation_400m_era.xlsx', index_col=0, sheet_name='AM')
df_smap_1km_sta_am = pd.read_excel(path_ismn + '/extraction/smap_validation_1km.xlsx', index_col=0, sheet_name='AM')
df_smap_9km_sta_am = pd.read_excel(path_ismn + '/extraction/smap_validation_9km.xlsx', index_col=0, sheet_name='AM')
# df_smap_gpm = pd.read_excel(path_ismn + '/extraction_2022/smap_validation_gpm.xlsx', index_col=0)

size_validation = np.shape(df_smap_insitu_sta_am)
stn_name = df_smap_insitu_sta_am.index
network_name = df_smap_insitu_sta_am['network'].tolist()
network_unique = df_smap_insitu_sta_am['network'].unique()
network_all_group = [np.where(df_smap_1km_sta_am['network'] == network_unique[x]) for x in range(len(network_unique))]


# df_smap_stat_slc = \
#     pd.read_excel('/Users/binfang/Documents/SMAP_Project/results/results_210705/validation/stat_slc_081721.xlsx', index_col=0)
# df_smap_stat_slc = df_smap_stat_slc.iloc[1:401, :]
# index_drop = [df_smap_stat_slc.index[x] for x in range(265, 361)]
# index_drop = [df_smap_stat_slc.index[x] for x in range(292, 314)]
# df_smap_stat_slc = df_smap_stat_slc.drop(index_drop)
# index_drop = [df_smap_stat_slc.index[x] for x in range(292, 314)]
# df_smap_stat_slc = df_smap_stat_slc.drop(index_drop)
df_stat_1km_slc = pd.read_excel('/Volumes/Elements/Datasets/ISMN/extraction/smap_validation_1km.xlsx', index_col=0)

stn_name_slc = df_stat_1km_slc.index
stn_name_slc_ind = [np.where(df_smap_1km_sta_am.index == stn_name_slc[x])[0][0] for x in range(len(stn_name_slc))]

# stn_name_slc_ind = np.arange(len(df_smap_insitu_sta_am))

df_smap_insitu_sta_am_slc = df_smap_insitu_sta_am.iloc[stn_name_slc_ind, :]
df_smap_400m_sta_am_slc = df_smap_400m_sta_am.iloc[stn_name_slc_ind, :]
df_smap_400m_era_sta_am_slc = df_smap_400m_era_sta_am.iloc[stn_name_slc_ind, :]
df_smap_1km_sta_am_slc = df_smap_1km_sta_am.iloc[stn_name_slc_ind, :]
df_smap_9km_sta_am_slc = df_smap_9km_sta_am.iloc[stn_name_slc_ind, :]
network_slc = df_smap_9km_sta_am_slc['network'].tolist()
network_unique_slc = df_smap_1km_sta_am_slc['network'].unique()
network_all_group_slc = [np.where(df_smap_1km_sta_am_slc['network'] == network_unique_slc[x])
                         for x in range(len(network_unique_slc))]

# Select data from 2015-2021
# columns_drop = [df_smap_insitu_sta_am_slc.columns[x] for x in range(2, 1918)]
# df_smap_insitu_sta_am_slc = df_smap_insitu_sta_am_slc.drop(columns=columns_drop)
# df_smap_1km_sta_am_slc = df_smap_1km_sta_am_slc.drop(columns=columns_drop)
# df_smap_9km_sta_am_slc = df_smap_9km_sta_am_slc.drop(columns=columns_drop)

# # Create folders for each network
# for ife in range(len(network_unique)):
#     os.mkdir(path_results + '/validation/single_plots/' + network_unique[ife])

# df_smap_insitu_sta_am_slc = df_smap_insitu_sta_am
# df_smap_1km_sta_am_slc = df_smap_1km_sta_am
# df_smap_9km_sta_am_slc = df_smap_9km_sta_am
stn_name_slc = df_smap_insitu_sta_am.index.tolist()
# network_slc = df_smap_9km_sta_am['network'].tolist()

# 3.1 single plots
stat_array_obs = []
stat_array_400m_gldas = []
stat_array_1km = []
stat_array_9km = []
stat_array_400m_era = []
ind_slc_all = []
# for ist in range(size_validation[0]):
for ist in range(len(stn_name_slc)):
    x = np.array(df_smap_insitu_sta_am_slc.iloc[ist, 2+1371:], dtype=np.float)
    y0 = np.array(df_smap_400m_sta_am_slc.iloc[ist, 2+1371:], dtype=np.float)
    y1 = np.array(df_smap_1km_sta_am_slc.iloc[ist, 2+1371:], dtype=np.float)
    y2 = np.array(df_smap_9km_sta_am_slc.iloc[ist, 2+1371:], dtype=np.float)
    y3 = np.array(df_smap_400m_era_sta_am_slc.iloc[ist, 2:], dtype=np.float)
    # x[x == 0] = np.nan
    # y1[y1 == 0] = np.nan
    # y2[y2 == 0] = np.nan

    ind_nonnan = np.where(~np.isnan(x) & ~np.isnan(y0) & ~np.isnan(y1) & ~np.isnan(y2) & ~np.isnan(y3))[0]
    if len(ind_nonnan) > 5:
        x = x[ind_nonnan]
        y0 = y0[ind_nonnan]
        y1 = y1[ind_nonnan]
        y2 = y2[ind_nonnan]
        y3 = y3[ind_nonnan]

        stdev_x = np.std(x)
        stat_array_x = [stdev_x]

        slope_0, intercept_0, r_value_0, p_value_0, std_err_0 = stats.linregress(x, y0)
        y0_estimated = intercept_0 + slope_0 * x
        number_0 = len(y0)
        r_sq_0 = r_value_0 ** 2
        ubrmse_0 = np.sqrt(np.mean((x - y0_estimated) ** 2))
        bias_0 = np.mean(x - y0)
        conf_int_0 = std_err_0 * 1.96  # From the Z-value
        mae_0 = np.mean(np.abs(x - y0_estimated))
        stdev_0 = np.std(y0)
        stat_array_0 = [number_0, r_sq_0, ubrmse_0, stdev_0, mae_0, bias_0, p_value_0, conf_int_0]

        slope_1, intercept_1, r_value_1, p_value_1, std_err_1 = stats.linregress(x, y1)
        y1_estimated = intercept_1 + slope_1 * x
        number_1 = len(y1)
        r_sq_1 = r_value_1 ** 2
        ubrmse_1 = np.sqrt(np.mean((x - y1_estimated) ** 2))
        bias_1 = np.mean(x - y1)
        conf_int_1 = std_err_1 * 1.96  # From the Z-value
        mae_1 = np.mean(np.abs(x - y1_estimated))
        stdev_1 = np.std(y1)
        stat_array_1 = [number_1, r_sq_1, ubrmse_1, stdev_1, mae_1, bias_1, p_value_1, conf_int_1]

        slope_2, intercept_2, r_value_2, p_value_2, std_err_2 = stats.linregress(x, y2)
        y2_estimated = intercept_2 + slope_2 * x
        number_2 = len(y2)
        r_sq_2 = r_value_2 ** 2
        ubrmse_2 = np.sqrt(np.mean((x - y2_estimated) ** 2))
        bias_2 = np.mean(x - y2)
        conf_int_2 = std_err_2 * 1.96  # From the Z-value
        mae_2 = np.mean(np.abs(x - y2_estimated))
        stdev_2 = np.std(y2)
        stat_array_2 = [number_2, r_sq_2, ubrmse_2, stdev_2, mae_2, bias_2, p_value_2, conf_int_2]

        slope_3, intercept_3, r_value_3, p_value_3, std_err_3 = stats.linregress(x, y3)
        y3_estimated = intercept_3 + slope_3 * x
        number_3 = len(y3)
        r_sq_3 = r_value_3 ** 2
        ubrmse_3 = np.sqrt(np.mean((x - y3_estimated) ** 2))
        bias_3 = np.mean(x - y3)
        conf_int_3 = std_err_3 * 1.96  # From the Z-value
        mae_3 = np.mean(np.abs(x - y3_estimated))
        stdev_3 = np.std(y3)
        stat_array_3 = [number_3, r_sq_3, ubrmse_3, stdev_3, mae_3, bias_3, p_value_3, conf_int_3]

        fig = plt.figure(figsize=(8, 5), dpi=200)
        fig.subplots_adjust(hspace=0.2, wspace=0.2)
        ax = fig.add_subplot(111)

        ax.scatter(x, y0, s=5, c='r', marker='o', label='400 m(GLDAS)')
        ax.scatter(x, y1, s=5, c='k', marker='^', label='1 km')
        ax.scatter(x, y2, s=5, c='g', marker='x', label='9 km')
        ax.scatter(x, y3, s=5, c='b', marker='+', label='400 m(ERA5)')

        ax.plot(x, intercept_0 + slope_0 * x, '-', color='r')
        ax.plot(x, intercept_1 + slope_1 * x, '-', color='k')
        ax.plot(x, intercept_2 + slope_2 * x, '-', color='g')
        ax.plot(x, intercept_3 + slope_3 * x, '-', color='b')

        # coef1, intr1 = np.polyfit(x.squeeze(), y1.squeeze(), 1)
        # ax.plot(x.squeeze(), intr1 + coef1 * x.squeeze(), '-', color='r')
        #
        # coef2, intr2 = np.polyfit(x.squeeze(), y2.squeeze(), 1)
        # ax.plot(x.squeeze(), intr2 + coef2 * x.squeeze(), '-', color='k')

        plt.xlim(0, 0.4)
        ax.set_xticks(np.arange(0, 0.5, 0.1))
        plt.ylim(0, 0.4)
        ax.set_yticks(np.arange(0, 0.5, 0.1))
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
        plt.grid(linestyle='--')
        plt.legend(loc='upper left')
        plt.suptitle(network_slc[ist] +', ' + str(stn_name_slc[ist]), fontsize=18, y=0.99, fontweight='bold')
        # plt.show()
        plt.savefig(path_results +'/validation/single_plots/' + network_slc[ist] + '_' + str(stn_name_slc[ist]) + '.png')
        plt.close(fig)

        stat_array_obs.append(stat_array_x)
        stat_array_400m_gldas.append(stat_array_0)
        stat_array_1km.append(stat_array_1)
        stat_array_9km.append(stat_array_2)
        stat_array_400m_era.append(stat_array_3)
        ind_slc_all.append(ist)
        print(ist)
        del(stat_array_x, stat_array_0, stat_array_1, stat_array_2, stat_array_3)
    else:
        pass

stat_array_obs = np.array(stat_array_obs)
stat_array_400m_gldas = np.array(stat_array_400m_gldas)
stat_array_1km = np.array(stat_array_1km)
stat_array_9km = np.array(stat_array_9km)
stat_array_400m_era = np.array(stat_array_400m_era)

columns_validation = ['number', 'r_sq', 'ubrmse', 'stdev', 'mae', 'bias', 'p_value', 'conf_int']
index_validation = df_smap_1km_sta_am_slc.index[ind_slc_all]
network_validation = df_smap_1km_sta_am_slc['network'].iloc[ind_slc_all]

df_stat_400m_gldas = pd.DataFrame(stat_array_400m_gldas, columns=columns_validation, index=index_validation)
df_stat_1km = pd.DataFrame(stat_array_1km, columns=columns_validation, index=index_validation)
df_stat_9km = pd.DataFrame(stat_array_9km, columns=columns_validation, index=index_validation)
df_stat_400m_era = pd.DataFrame(stat_array_400m_era, columns=columns_validation, index=index_validation)
df_stat_400m_gldas = pd.concat([network_validation, df_stat_400m_gldas], axis=1, sort=False)
df_stat_1km = pd.concat([network_validation, df_stat_1km], axis=1, sort=False)
df_stat_9km = pd.concat([network_validation, df_stat_9km], axis=1, sort=False)
df_stat_400m_era = pd.concat([network_validation, df_stat_400m_era], axis=1, sort=False)
writer_400m_gldas = pd.ExcelWriter(path_results + '/validation/stat_400m_gldas_smap_compare.xlsx')
writer_1km = pd.ExcelWriter(path_results + '/validation/stat_1km_smap_compare.xlsx')
writer_9km = pd.ExcelWriter(path_results + '/validation/stat_9km_smap_compare.xlsx')
writer_400m_era = pd.ExcelWriter(path_results + '/validation/stat_400m_era_smap_compare.xlsx')
df_stat_400m_gldas.to_excel(writer_400m_gldas)
df_stat_1km.to_excel(writer_1km)
df_stat_9km.to_excel(writer_9km)
df_stat_400m_era.to_excel(writer_400m_era)
writer_400m_gldas.save()
writer_1km.save()
writer_9km.save()
writer_400m_era.save()

columns_validation_obs = ['stdev']
df_stat_insitu = pd.DataFrame(stat_array_obs, columns=columns_validation_obs, index=index_validation)
df_stat_insitu = pd.concat([network_validation, df_stat_insitu], axis=1, sort=False)
writer_insitu = pd.ExcelWriter(path_results + '/validation/stat_insitu_smap_compare.xlsx')
df_stat_insitu.to_excel(writer_insitu)
writer_insitu.save()



# Read validation data
df_stat_insitu = pd.read_excel(path_results + '/validation/stat_insitu_smap_compare.xlsx', index_col=0)
df_stat_400m = pd.read_excel(path_results + '/validation/stat_400m_smap_compare.xlsx', index_col=0)
df_stat_1km = pd.read_excel(path_results + '/validation/stat_1km_smap_compare.xlsx', index_col=0)
df_stat_9km = pd.read_excel(path_results + '/validation/stat_9km_smap_compare.xlsx', index_col=0)
# df_stat_400m_era = pd.read_excel(path_results + '/validation/stat_400m_era_smap_compare.xlsx', index_col=0)

# HOBE: 454, 455, 465, 466, 477, 478 (460, 461)
# OZNET: 1446, 1451, 1452, 1456, 1467, 1471, 1472, 1477 (1448, 1458, 1468, 1449)
# PTSMN: 1485, 1486, 1487, 1488, 1491, 1492
# RISMA: 732, 736, 740, 741, 742, 745, 747, 750
# stat_range = [2, 3, 4, 144, 147, 148, 149, 150, 157, 158, 275, 277, 279, 280, 281, 285, 290]
network_list = ['AMMA-CATCH', 'ARM', 'BIEBRZA', 'COSMOS', 'CTP', 'FLUXNET-AMERIFLUX', 'GROW', 'HOAL', 'HOBE', 'IMA',
                'KIHS', 'MAQU', 'NAQU', 'NGARI', 'OZNET', 'PBO', 'REMEDHUS', 'RISMA', 'RSMN', 'Ru', 'SCAN', 'SMN-SDR',
                'SMOSMANIA', 'SNOTEL', 'SOILSCAPE', 'SONTE-China', 'TAHMO', 'TERENO', 'TWENTE', 'TxSON', 'USCRN', 'WEGENERNET',
                'WIT-Network', 'XMS-CAT', 'iRON']
stat_range = np.array([431, 437, 455, 462, 463, 630, 659, 669, 673, 688, 735, 745, 765, 779, 892, 898, 903, 906, 911, 1043, 1062,
                       1063, 1064, 1075, 1154, 1155, 1176, 1186, 1191, 1193, 1194, 1195, 1488, 1495, 1496, 1522, 1525,
                       1535, 1536, 1571, 1574, 1575, 1578, 1580, 1588, 1596, 1597, 1623, 1625, 1629]) - 2
index_drop_stat = [df_stat_1km.index[x] for x in stat_range]

df_stat_1km_slc = df_stat_1km.drop(index_drop_stat)
index_stat = df_stat_1km_slc.index
# Find matched rows by original stat table
df_stat_400m_slc = df_stat_400m.loc[df_stat_400m.index.isin(df_stat_400m.index)]
df_stat_1km_slc = df_stat_1km.loc[df_stat_1km.index.isin(df_stat_1km.index)]
df_stat_9km_slc = df_stat_9km.loc[df_stat_9km.index.isin(df_stat_9km.index)]

df_stat_400m_slc = df_stat_400m.drop(index_drop_stat)
df_stat_1km_slc = df_stat_1km.drop(index_drop_stat)
df_stat_9km_slc = df_stat_9km.drop(index_drop_stat)

# df_stat_400m_slc = df_stat_400m
# df_stat_1km_slc = df_stat_1km
# df_stat_9km_slc = df_stat_9km


# network_unique = np.unique(df_stat_400m_slc['network'], return_index=True)[0]

df_stat_400m_mean = df_stat_400m_slc.groupby('network').mean()
df_stat_1km_mean = df_stat_1km_slc.groupby('network').mean()
df_stat_9km_mean = df_stat_9km_slc.groupby('network').mean()
df_stat_400m_sum = df_stat_400m_slc.groupby('network')['number'].sum()
df_stat_1km_sum = df_stat_1km_slc.groupby('network')['number'].sum()
df_stat_9km_sum = df_stat_9km_slc.groupby('network')['number'].sum()
df_stat_count = df_stat_1km.groupby('network')['network'].count()
df_stat_400m_mean['number'] = df_stat_400m_sum
df_stat_1km_mean['number'] = df_stat_1km_sum
df_stat_9km_mean['number'] = df_stat_9km_sum
df_stat_400m_mean.insert(loc=0, column='n_station', value=df_stat_count)
df_stat_1km_mean.insert(loc=0, column='n_station', value=df_stat_count)
df_stat_9km_mean.insert(loc=0, column='n_station', value=df_stat_count)
df_stat_400m_mean.round(3)
df_stat_1km_mean.round(3)
df_stat_9km_mean.round(3)
# df_stat_1km_mean_slc = df_stat_1km_mean.loc[network_list]
# df_stat_9km_mean_slc = df_stat_9km_mean.loc[network_list]
# network_list_slc = df_stat_1km_mean.loc[df_stat_1km_mean.index.isin(network_unique)].index.tolist()

df_stat_400m_mean_slc = df_stat_400m_mean.loc[network_list]
df_stat_1km_mean_slc = df_stat_1km_mean.loc[network_list]
df_stat_9km_mean_slc = df_stat_9km_mean.loc[network_list]

writer_400m = pd.ExcelWriter(path_results + '/validation/stat_400m_mean_smap_compare.xlsx')
writer_1km = pd.ExcelWriter(path_results + '/validation/stat_1km_mean_smap_compare.xlsx')
writer_9km = pd.ExcelWriter(path_results + '/validation/stat_9km_mean_smap_compare.xlsx')
df_stat_400m_mean.to_excel(writer_400m)
df_stat_1km_mean.to_excel(writer_1km)
df_stat_9km_mean.to_excel(writer_9km)
writer_400m.save()
writer_1km.save()
writer_9km.save()

writer_400m_slc = pd.ExcelWriter(path_results + '/validation/stat_400m_mean_smap_compare_slc.xlsx')
writer_1km_slc = pd.ExcelWriter(path_results + '/validation/stat_1km_mean_smap_compare_slc.xlsx')
writer_9km_slc = pd.ExcelWriter(path_results + '/validation/stat_9km_mean_smap_compare_slc.xlsx')
df_stat_400m_mean_slc.to_excel(writer_400m_slc)
df_stat_1km_mean_slc.to_excel(writer_1km_slc)
df_stat_9km_mean_slc.to_excel(writer_9km_slc)
writer_400m_slc.save()
writer_1km_slc.save()
writer_9km_slc.save()

# 3.2 subplots
df_stat_1km = pd.read_excel(path_results + '/validation/stat_1km_2015.xlsx', index_col=0)
df_stat_9km = pd.read_excel(path_results + '/validation/stat_9km_2015.xlsx', index_col=0)

# REMEDHUS: Carretoro, Casa_Periles, El_Tomillar, Granja_g, La_Atalaya, Las_Bodegas, Las_Vacas, Zamarron
# REMEDHUS ID:382, 383, 386, 387, 389, 392, 396, 400
# SOILSCAPE: node403, 404, 405, 406, 408, 415, 416, 417
# SOILSCAPE ID: 1425, 1426, 1427, 1428, 1429, 1436, 1437, 1438
# CTP: L18, L19, L21, L27, L33, L34, L36, L37
# CTP ID: 38, 39, 41, 46, 51, 52, 54, 55
# OZNET: Alabama, Banandra, Bundure, Samarra, Uri Park, Wollumbi, Yamma Road, Yammacoona
# OZNET ID: 1619, 1620, 1621, 1629, 1632, 1634, 1636, 1637

# network_name = ['REMEDHUS', 'SOILSCAPE', 'CTP', 'OZNET']
# site_ind = [[382, 383, 386, 387, 389, 392, 396, 400], [1425, 1426, 1427, 1428, 1429, 1436, 1437, 1438],
#             [38, 39, 41, 46, 51, 52, 54, 55], [1619, 1620, 1621, 1629, 1632, 1634, 1636, 1637]]
# network_name = ['TxSON']
# site_ind = [[8, 11, 12, 24, 25, 30]]



# TxSON: cr200_7, cr200_13, cr1000_3, lcra_3
# SoilSCAPE: node_405, node_406, node_408, node_416
# REMEDHUS: Carretoro, Casa periles, Las bodegas, Zamarron
network_name = ['TxSON', 'SoilSCAPE', 'REMEDHUS']
site_ind = list(np.array([[1297, 1305, 1325, 1330], [1205, 1206, 1207, 1215], [2265, 2266, 2276, 2284]]) - 2)

# network_name = list(stn_slc_all_unique)
# site_ind = stn_slc_all_group

for inw in range(len(site_ind)):
    fig = plt.figure(figsize=(11, 11))
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.08, top=0.89, hspace=0.25, wspace=0.25)
    for ist in range(len(site_ind[inw])):

        x = np.array(df_smap_insitu_sta_am.iloc[site_ind[inw][ist], 2+1371:], dtype=np.float64)
        x[x == 0] = np.nan
        y0 = np.array(df_smap_400m_sta_am.iloc[site_ind[inw][ist], 2+1371:], dtype=np.float64)
        y3 = np.array(df_smap_400m_era_sta_am.iloc[site_ind[inw][ist], 2:], dtype=np.float64)
        y1 = np.array(df_smap_1km_sta_am.iloc[site_ind[inw][ist], 2+1371:], dtype=np.float64)
        y2 = np.array(df_smap_9km_sta_am.iloc[site_ind[inw][ist], 2+1371:], dtype=np.float64)
        ind_nonnan = np.where(~np.isnan(x) & ~np.isnan(y0) & ~np.isnan(y1) & ~np.isnan(y2))[0]

        x = x[ind_nonnan]
        y0 = y0[ind_nonnan]
        y3 = y3[ind_nonnan]
        y1 = y1[ind_nonnan]
        y2 = y2[ind_nonnan]

        slope_0, intercept_0, r_value_0, p_value_0, std_err_0 = stats.linregress(x, y0)
        y0_estimated = intercept_0 + slope_0 * x
        number_0 = len(y0)
        r_sq_0 = r_value_0 ** 2
        ubrmse_0 = np.sqrt(np.mean((x - y0_estimated) ** 2))
        bias_0 = np.mean(x - y0)
        conf_int_0 = std_err_0 * 1.96  # From the Z-value
        mae_0 = np.mean(np.abs(x - y0_estimated))
        # stdev_0 = np.std(y0)
        stat_array_0 = [number_0, r_sq_0, ubrmse_0, mae_0, bias_0, p_value_0, conf_int_0]

        slope_3, intercept_3, r_value_3, p_value_3, std_err_3 = stats.linregress(x, y3)
        y3_estimated = intercept_3 + slope_3 * x
        number_3 = len(y3)
        r_sq_3 = r_value_3 ** 2
        ubrmse_3 = np.sqrt(np.mean((x - y3_estimated) ** 2))
        bias_3 = np.mean(x - y3)
        conf_int_3 = std_err_3 * 1.96  # From the Z-value
        mae_3 = np.mean(np.abs(x - y3_estimated))
        # stdev_0 = np.std(y0)
        stat_array_3 = [number_3, r_sq_3, ubrmse_3, mae_3, bias_3, p_value_3, conf_int_3]

        slope_1, intercept_1, r_value_1, p_value_1, std_err_1 = stats.linregress(x, y1)
        y1_estimated = intercept_1 + slope_1 * x
        number_1 = len(y1)
        r_sq_1 = r_value_1 ** 2
        ubrmse_1 = np.sqrt(np.mean((x - y1_estimated) ** 2))
        bias_1 = np.mean(x - y1)
        conf_int_1 = std_err_1 * 1.96  # From the Z-value
        stat_array_1 = [number_1, r_sq_1, ubrmse_1, bias_1, p_value_1, conf_int_1]

        slope_2, intercept_2, r_value_2, p_value_2, std_err_2 = stats.linregress(x, y2)
        y2_estimated = intercept_2 + slope_2 * x
        number_2 = len(y2)
        r_sq_2 = r_value_2 ** 2
        ubrmse_2 = np.sqrt(np.mean((x - y2_estimated) ** 2))
        bias_2 = np.mean(x - y2)
        conf_int_2 = std_err_2 * 1.96  # From the Z-value
        stat_array_2 = [number_2, r_sq_2, ubrmse_2, bias_2, p_value_2, conf_int_2]

        ax = fig.add_subplot(len(site_ind[inw])//2, 2, ist+1)
        sc0 = ax.scatter(x, y0, s=20, c='r', marker='o', label='400 m(GLDAS)')
        sc3 = ax.scatter(x, y3, s=20, c='b', marker='+', label='400 m(ERA5)')
        sc1 = ax.scatter(x, y1, s=20, c='k', marker='^', label='1 km')
        sc2 = ax.scatter(x, y2, s=20, c='g', marker='x', label='9 km')
        ax.plot(x, intercept_0 + slope_0 * x, '-', color='r')
        ax.plot(x, intercept_3 + slope_3 * x, '-', color='b')
        ax.plot(x, intercept_1 + slope_1 * x, '-', color='k')
        ax.plot(x, intercept_2 + slope_2 * x, '-', color='g')

        plt.xlim(0, 0.5)
        ax.set_xticks(np.arange(0, 0.6, 0.1))
        plt.ylim(0, 0.5)
        ax.set_yticks(np.arange(0, 0.6, 0.1))
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
        plt.grid(linestyle='--')
        ax.text(0.01, 0.45, df_smap_1km_sta_am.index[site_ind[inw][ist]].replace('_', '_'), fontsize=20)

    # add all legends together
    handles = [sc0] + [sc3] + [sc1] + [sc2]
    labels = [l.get_label() for l in handles]
    # leg = plt.legend([sc1, sc2, sc3], labels, loc=(-0.6, 3.55), mode="expand", borderaxespad=0, ncol=3, prop={"size": 13})
    leg = plt.legend([sc0, sc3, sc1, sc2], labels, loc=(-0.9, 2.3), mode="expand", borderaxespad=-4, ncol=4,
                     prop={"size": 15}, bbox_to_anchor=(0.2, 2.2, 1.5, 0.05))

    fig.text(0.52, 0.01, 'In Situ SM ($\mathregular{m^3/m^3)}$', ha='center', fontsize=18, fontweight='bold')
    fig.text(0.02, 0.4, 'SMAP SM ($\mathregular{m^3/m^3)}$', rotation='vertical', fontsize=18, fontweight='bold')
    plt.suptitle(network_name[inw], fontsize=25, y=0.99, fontweight='bold')
    plt.show()

    plt.savefig(path_results + '/validation/network_plots/' + network_name[inw] + '.png')
    plt.close(fig)


# 3.2.2 Make Taylor Diagram
# Index:
# txson 313, 314, 317, 319
# remedhus 13, 17, 19, 20
# Read validation data
df_stat_insitu = pd.read_excel(path_results + '/validation/stat_insitu_smap_compare.xlsx', index_col=0)
df_stat_400m = pd.read_excel(path_results + '/validation/stat_400m_smap_compare.xlsx', index_col=0)
df_stat_1km = pd.read_excel(path_results + '/validation/stat_1km_smap_compare.xlsx', index_col=0)
df_stat_9km = pd.read_excel(path_results + '/validation/stat_9km_smap_compare.xlsx', index_col=0)

ind_txs = list(np.array([889, 897, 917, 922]) - 2)
ind_rem = list(np.array([1481, 1482, 1491, 1499]) - 2)
ind_sca = list(np.array([836, 837, 838, 844]) - 2)
df_stat_insitu_txs = df_stat_insitu.iloc[ind_txs, :]
df_stat_insitu_rem = df_stat_insitu.iloc[ind_rem, :]
df_stat_insitu_sca = df_stat_insitu.iloc[ind_sca, :]
df_stat_400m_txs = df_stat_400m.iloc[ind_txs, :]
df_stat_400m_rem = df_stat_400m.iloc[ind_rem, :]
df_stat_400m_sca = df_stat_400m.iloc[ind_sca, :]
df_stat_1km_txs = df_stat_1km.iloc[ind_txs, :]
df_stat_1km_rem = df_stat_1km.iloc[ind_rem, :]
df_stat_1km_sca = df_stat_1km.iloc[ind_sca , :]
df_stat_9km_txs = df_stat_9km.iloc[ind_txs, :]
df_stat_9km_rem = df_stat_9km.iloc[ind_rem, :]
df_stat_9km_sca = df_stat_9km.iloc[ind_sca, :]

# df_stat_9km_rem_arr = np.array(df_stat_9km_rem.iloc[:, 2:5])
# df_stat_9km_rem_arr[:, 1] = np.sqrt(df_stat_9km_rem_arr[:, 1])
# df_stat_9km_rem_arr = np.concatenate((np.expand_dims(df_stat_9km_rem_arr[0, :], axis=0), df_stat_9km_rem_arr), axis=0)

# REMEDHUS
df_plot_rem_1 = np.stack((np.array([1, 0, df_stat_insitu_rem.iloc[0, 1]]),
                          np.array([1, 0, df_stat_insitu_rem.iloc[0, 1]]),
                          np.array(df_stat_400m_rem.iloc[0, 2:5]),
                          np.array(df_stat_1km_rem.iloc[0, 2:5]),
                          np.array(df_stat_9km_rem.iloc[0, 2:5])))
df_plot_rem_1 = np.array(df_plot_rem_1, dtype='float32')

df_plot_rem_2 = np.stack((np.array([1, 0, df_stat_insitu_rem.iloc[1, 1]]),
                          np.array([1, 0, df_stat_insitu_rem.iloc[1, 1]]),
                          np.array(df_stat_400m_rem.iloc[1, 2:5]),
                          np.array(df_stat_1km_rem.iloc[1, 2:5]),
                          np.array(df_stat_9km_rem.iloc[1, 2:5])))
df_plot_rem_2 = np.array(df_plot_rem_2, dtype='float32')

df_plot_rem_3 = np.stack((np.array([1, 0, df_stat_insitu_rem.iloc[2, 1]]),
                          np.array([1, 0, df_stat_insitu_rem.iloc[2, 1]]),
                          np.array(df_stat_400m_rem.iloc[2, 2:5]),
                          np.array(df_stat_1km_rem.iloc[2, 2:5]),
                          np.array(df_stat_9km_rem.iloc[2, 2:5])))
df_plot_rem_3 = np.array(df_plot_rem_3, dtype='float32')

df_plot_rem_4 = np.stack((np.array([1, 0, df_stat_insitu_rem.iloc[3, 1]]),
                          np.array([1, 0, df_stat_insitu_rem.iloc[3, 1]]),
                          np.array(df_stat_400m_rem.iloc[3, 2:5]),
                          np.array(df_stat_1km_rem.iloc[3, 2:5]),
                          np.array(df_stat_9km_rem.iloc[3, 2:5])))
df_plot_rem_4 = np.array(df_plot_rem_4, dtype='float32')

df_plot_rem = list((df_plot_rem_1, df_plot_rem_2, df_plot_rem_3, df_plot_rem_4))

# Plot together
label = ['Obs', 'Obs', '9km', '1km', '400m']
fig = plt.figure(figsize=(12, 8), dpi=150, facecolor='w', edgecolor='k')
plt.subplots_adjust(left=0.1, right=0.85, bottom=0.1, top=0.9, hspace=0.2, wspace=0.5)
for ipt in range(4):
    ax = fig.add_subplot(2, 2, ipt+1)
    sm.taylor_diagram(df_plot_rem[ipt][:, 2], df_plot_rem[ipt][:, 1], df_plot_rem[ipt][:, 0], markerLabel=label,
                      markerColor='r', markerSize=10, alpha=0.0, markerLegend='off', markerobs='^',
                      tickRMS=np.arange(0, 0.15, 0.03), colRMS='tab:green', styleRMS=':', widthRMS=1.0, titleRMS='on',
                      titleRMSDangle=40.0, showlabelsRMS='on', tickSTD=np.arange(0, 0.12, 0.03), axismax=0.12,
                      colSTD='black', styleSTD='-.', widthSTD=1.0, titleSTD='on',
                      colCOR='tab:blue', styleCOR='--', widthCOR=1.0, titleCOR='on')
    plt.xticks(np.arange(0, 0.13, 0.05))
    plt.text(0.09, 0.12, df_stat_400m_rem.index[ipt], fontsize=12, fontweight='bold')
plt.show()
plt.savefig(path_results + '/validation/' + 'td_400m_rem.png')


# TxSON
df_plot_txs_1 = np.stack((np.array([1, 0, df_stat_insitu_txs.iloc[0, 1]]),
                          np.array([1, 0, df_stat_insitu_txs.iloc[0, 1]]),
                          np.array(df_stat_400m_txs.iloc[0, 2:5]),
                          np.array(df_stat_1km_txs.iloc[0, 2:5]),
                          np.array(df_stat_9km_txs.iloc[0, 2:5])))
df_plot_txs_1 = np.array(df_plot_txs_1, dtype='float32')

df_plot_txs_2 = np.stack((np.array([1, 0, df_stat_insitu_txs.iloc[1, 1]]),
                          np.array([1, 0, df_stat_insitu_txs.iloc[1, 1]]),
                          np.array(df_stat_400m_txs.iloc[1, 2:5]),
                          np.array(df_stat_1km_txs.iloc[1, 2:5]),
                          np.array(df_stat_9km_txs.iloc[1, 2:5])))
df_plot_txs_2 = np.array(df_plot_txs_2, dtype='float32')

df_plot_txs_3 = np.stack((np.array([1, 0, df_stat_insitu_txs.iloc[2, 1]]),
                          np.array([1, 0, df_stat_insitu_txs.iloc[2, 1]]),
                          np.array(df_stat_400m_txs.iloc[2, 2:5]),
                          np.array(df_stat_1km_txs.iloc[2, 2:5]),
                          np.array(df_stat_9km_txs.iloc[2, 2:5])))
df_plot_txs_3 = np.array(df_plot_txs_3, dtype='float32')

df_plot_txs_4 = np.stack((np.array([1, 0, df_stat_insitu_txs.iloc[3, 1]]),
                          np.array([1, 0, df_stat_insitu_txs.iloc[3, 1]]),
                          np.array(df_stat_400m_txs.iloc[3, 2:5]),
                          np.array(df_stat_1km_txs.iloc[3, 2:5]),
                          np.array(df_stat_9km_txs.iloc[3, 2:5])))
df_plot_txs_4 = np.array(df_plot_txs_4, dtype='float32')

df_plot_txs = list((df_plot_txs_1, df_plot_txs_2, df_plot_txs_3, df_plot_txs_4))


# STDs = df_plot_txs[0][:, 2]
# CORs = df_plot_txs[0][:, 1]
# RMSs = np.sqrt(STDs**2 + STDs[0]**2 - 2*STDs*STDs[0]*CORs)

std_ref = STDs[0]
STDs = df_plot_txs[0][:, 2]
CORs = np.sqrt(df_plot_txs[0][:, 0])
RMSs = np.sqrt(std_ref**2 + STDs**2 - 2 * std_ref * STDs * CORs)

sm.taylor_diagram(STDs, RMSs, CORs, markerLabel=label,
                  markerColor='r', markerSize=10, alpha=0.0, markerLegend='off', markerobs='^',
                  tickRMS=np.arange(0, 0.15, 0.03), colRMS='tab:green', styleRMS=':', widthRMS=1.0, titleRMS='on',
                  titleRMSDangle=40.0, showlabelsRMS='on', tickSTD=np.arange(0, 0.12, 0.03), axismax=0.12,
                  colSTD='black', styleSTD='-.', widthSTD=1.0, titleSTD='on',
                  colCOR='tab:blue', styleCOR='--', widthCOR=1.0, titleCOR='on')


# Plot together
label = ['Obs', 'Obs', '400m', '1km', '9km']
fig = plt.figure(figsize=(12, 8), dpi=150, facecolor='w', edgecolor='k')
plt.subplots_adjust(left=0.1, right=0.85, bottom=0.1, top=0.9, hspace=0.2, wspace=0.5)
for ipt in range(4):
    ax = fig.add_subplot(2, 2, ipt+1)
    sm.taylor_diagram(df_plot_txs[ipt][:, 2], df_plot_txs[ipt][:, 1], df_plot_txs[ipt][:, 0], markerLabel=label,
                      markerColor='r', markerSize=10, alpha=0.0, markerLegend='off', markerobs='^',
                      tickRMS=np.arange(0, 0.15, 0.03), colRMS='tab:green', styleRMS=':', widthRMS=1.0, titleRMS='on',
                      titleRMSDangle=40.0, showlabelsRMS='on', tickSTD=np.arange(0, 0.12, 0.03), axismax=0.12,
                      colSTD='black', styleSTD='-.', widthSTD=1.0, titleSTD='on',
                      colCOR='tab:blue', styleCOR='--', widthCOR=1.0, titleCOR='on')
    plt.xticks(np.arange(0, 0.13, 0.05))
    plt.text(0.09, 0.12, df_stat_400m_txs.index[ipt], fontsize=12, fontweight='bold')
plt.show()
plt.savefig(path_results + '/validation/' + 'td_400m_txs.png')


# SoilSCAPE
df_plot_sca_1 = np.stack((np.array([1, 0, df_stat_insitu_sca.iloc[0, 1]]),
                          np.array([1, 0, df_stat_insitu_sca.iloc[0, 1]]),
                          np.array(df_stat_400m_sca.iloc[0, 2:5]),
                          np.array(df_stat_1km_sca.iloc[0, 2:5]),
                          np.array(df_stat_9km_sca.iloc[0, 2:5])))
df_plot_sca_1 = np.array(df_plot_sca_1, dtype='float32')

df_plot_sca_2 = np.stack((np.array([1, 0, df_stat_insitu_sca.iloc[1, 1]]),
                          np.array([1, 0, df_stat_insitu_sca.iloc[1, 1]]),
                          np.array(df_stat_400m_sca.iloc[1, 2:5]),
                          np.array(df_stat_1km_sca.iloc[1, 2:5]),
                          np.array(df_stat_9km_sca.iloc[1, 2:5])))
df_plot_sca_2 = np.array(df_plot_sca_2, dtype='float32')

df_plot_sca_3 = np.stack((np.array([1, 0, df_stat_insitu_sca.iloc[2, 1]]),
                          np.array([1, 0, df_stat_insitu_sca.iloc[2, 1]]),
                          np.array(df_stat_400m_sca.iloc[2, 2:5]),
                          np.array(df_stat_1km_sca.iloc[2, 2:5]),
                          np.array(df_stat_9km_sca.iloc[2, 2:5])))
df_plot_sca_3 = np.array(df_plot_sca_3, dtype='float32')

df_plot_sca_4 = np.stack((np.array([1, 0, df_stat_insitu_sca.iloc[3, 1]]),
                          np.array([1, 0, df_stat_insitu_sca.iloc[3, 1]]),
                          np.array(df_stat_400m_sca.iloc[3, 2:5]),
                          np.array(df_stat_1km_sca.iloc[3, 2:5]),
                          np.array(df_stat_9km_sca.iloc[3, 2:5])))
df_plot_sca_4 = np.array(df_plot_sca_4, dtype='float32')

df_plot_sca = list((df_plot_sca_1, df_plot_sca_2, df_plot_sca_3, df_plot_sca_4))

# Plot together
label = ['Obs', 'Obs', '9km', '1km', '400m']
fig = plt.figure(figsize=(12, 8), dpi=150, facecolor='w', edgecolor='k')
plt.subplots_adjust(left=0.1, right=0.85, bottom=0.1, top=0.9, hspace=0.2, wspace=0.5)
for ipt in range(4):
    ax = fig.add_subplot(2, 2, ipt+1)
    sm.taylor_diagram(df_plot_sca[ipt][:, 2], df_plot_sca[ipt][:, 1], df_plot_sca[ipt][:, 0], markerLabel=label,
                      markerColor='r', markerSize=10, alpha=0.0, markerLegend='off', markerobs='^',
                      tickRMS=np.arange(0, 0.15, 0.03), colRMS='tab:green', styleRMS=':', widthRMS=1.0, titleRMS='on',
                      titleRMSDangle=40.0, showlabelsRMS='on', tickSTD=np.arange(0, 0.12, 0.03), axismax=0.12,
                      colSTD='black', styleSTD='-.', widthSTD=1.0, titleSTD='on',
                      colCOR='tab:blue', styleCOR='--', widthCOR=1.0, titleCOR='on')
    plt.xticks(np.arange(0, 0.13, 0.05))
    plt.text(0.09, 0.12, df_stat_400m_sca.index[ipt], fontsize=12, fontweight='bold')
plt.show()
plt.savefig(path_results + '/validation/' + 'td_400m_sca.png')






# 3.3 Calculate SD for 1/9 km SM and in-situ
spstd_x_all = []
spstd_y0_all = []
spstd_y1_all = []
spstd_y2_all = []
for inw in range(len(network_all_group_slc)):
    x = np.array(df_smap_insitu_sta_am_slc.iloc[network_all_group_slc[inw][0].tolist(), 2:], dtype=np.float64)
    y0 = np.array(df_smap_400m_sta_am_slc.iloc[network_all_group_slc[inw][0].tolist(), 2:], dtype=np.float64)
    y1 = np.array(df_smap_1km_sta_am_slc.iloc[network_all_group_slc[inw][0].tolist(), 2:], dtype=np.float64)
    y2 = np.array(df_smap_9km_sta_am_slc.iloc[network_all_group_slc[inw][0].tolist(), 2:], dtype=np.float64)

    x_len = np.array([len(x[:, n][~np.isnan(x[:, n])]) for n in range(x.shape[1])])
    x_len_ind = np.where(x_len >= 3)[0]
    y0_len = np.array([len(y0[:, n][~np.isnan(y0[:, n])]) for n in range(y0.shape[1])])
    y0_len_ind = np.where(y0_len >= 3)[0]
    y1_len = np.array([len(y1[:, n][~np.isnan(y1[:, n])]) for n in range(y1.shape[1])])
    y1_len_ind = np.where(y1_len >= 3)[0]
    y2_len = np.array([len(y2[:, n][~np.isnan(y2[:, n])]) for n in range(y2.shape[1])])
    y2_len_ind = np.where(y2_len >= 3)[0]

    x_y1_ind = np.intersect1d(x_len_ind, y0_len_ind)
    spstd_ind = np.intersect1d(x_y1_ind, y1_len_ind)
    spstd_ind = np.intersect1d(spstd_ind, y2_len_ind)

    spstd_x = np.nanstd(x, axis=0)
    spstd_y0 = np.nanstd(y0, axis=0)
    spstd_y1 = np.nanstd(y1, axis=0)
    spstd_y2 = np.nanstd(y2, axis=0)

    spstd_x = np.nanmean(spstd_x[spstd_ind])
    spstd_y0 = np.nanmean(spstd_y0[spstd_ind])
    spstd_y1 = np.nanmean(spstd_y1[spstd_ind])
    spstd_y2 = np.nanmean(spstd_y2[spstd_ind])

    spstd_x_all.append(spstd_x)
    spstd_y0_all.append(spstd_y0)
    spstd_y1_all.append(spstd_y1)
    spstd_y2_all.append(spstd_y2)

    del(x, y0, y1, y2, x_len, x_len_ind, y0_len, y0_len_ind, y1_len, y1_len_ind, y2_len, y2_len_ind, x_y1_ind, spstd_ind,
        spstd_x, spstd_y0, spstd_y1, spstd_y2)
    print(inw)

spstd_x_all = np.array(spstd_x_all)
spstd_y0_all = np.array(spstd_y0_all)
spstd_y1_all = np.array(spstd_y1_all)
spstd_y2_all = np.array(spstd_y2_all)
spstd_all = np.stack([spstd_x_all, spstd_y0_all, spstd_y1_all, spstd_y2_all], axis=1)

# Save variables
df_table_spstd = pd.DataFrame(spstd_all, columns=['in-situ', '400m', '1km', '9km'], index=network_unique_slc)
df_table_spstd_slc = df_table_spstd.loc[network_list]
df_table_spstd_slc = df_table_spstd_slc.round(3)
writer_spstd = pd.ExcelWriter(path_results + '/validation/spstd_smap_new.xlsx')
df_table_spstd_slc.to_excel(writer_spstd)
writer_spstd.save()



# 3.4 Make the time-series plots

# Extract the GPM data by indices
df_coords = pd.read_excel(path_ismn + '/extraction/smap_coords.xlsx', index_col=0)
stn_lat_all = np.array(df_coords['lat'])
stn_lon_all = np.array(df_coords['lon'])

# Locate the corresponding GPM 10 km data located by lat/lon of in-situ data
stn_row_10km_ind_all = []
stn_col_10km_ind_all = []
for idt in range(len(stn_lat_all)):
    stn_row_10km_ind = np.argmin(np.absolute(stn_lat_all[idt] - lat_world_geo_10km)).item()
    stn_col_10km_ind = np.argmin(np.absolute(stn_lon_all[idt] - lon_world_geo_10km)).item()
    stn_row_10km_ind_all.append(stn_row_10km_ind)
    stn_col_10km_ind_all.append(stn_col_10km_ind)
    del(stn_row_10km_ind, stn_col_10km_ind)

gpm_precip_ext_all = []
for iyr in range(5, len(yearname)):

    f_gpm = h5py.File(path_gpm + '/gpm_precip_' + str(yearname[iyr]) + '.hdf5', 'r')
    varname_list_gpm = list(f_gpm.keys())

    for x in range(len(varname_list_gpm)):
        var_obj = f_gpm[varname_list_gpm[x]][()]
        exec(varname_list_gpm[x] + '= var_obj')
        del(var_obj)
    f_gpm.close()

    exec('gpm_precip = gpm_precip_10km_' + str(yearname[iyr]))
    gpm_precip_ext = gpm_precip[stn_row_10km_ind_all, stn_col_10km_ind_all, :]
    gpm_precip_ext_all.append(gpm_precip_ext)
    print(iyr)
    del(gpm_precip, gpm_precip_ext)

gpm_precip_ext_array = np.concatenate(gpm_precip_ext_all, axis=1)
gpm_precip_ext_array = gpm_precip_ext_array[:, 90:]
nonnan_fill = np.empty((2553, 122))
nonnan_fill[:] = np.nan
gpm_precip_ext_array = np.concatenate((gpm_precip_ext_array, nonnan_fill), axis=1)

# Save variables
df_table_gpm = pd.DataFrame(gpm_precip_ext_array, columns=date_seq_doy[1916:], index=index_validation)
df_table_gpm = pd.concat([continent_validation, network_validation, df_table_gpm], axis=1, sort=False)
writer_gpm = pd.ExcelWriter(path_ismn + '/extraction/smap_validation_gpm.xlsx')
df_table_gpm.to_excel(writer_gpm)
writer_gpm.save()


df_table_gpm = pd.read_excel(path_ismn + '/extraction/smap_validation_gpm.xlsx', index_col=0)


# Generate index tables for calculating monthly averages
monthly_seq = np.reshape(daysofmonth_seq, (1, -1), order='F')
monthly_seq = monthly_seq[:, 63:] # Remove the first 3 months in 2015
monthly_seq_cumsum = np.cumsum(monthly_seq)
array_allnan = np.empty([size_validation[0], 3], dtype='float32')
array_allnan[:] = np.nan

smap_insitu_am_split = \
    np.hsplit(np.array(df_smap_insitu_sta_am.iloc[:, 2:], dtype='float32'), monthly_seq_cumsum) # split by each month
smap_insitu_am_monthly = [np.nanmean(smap_insitu_am_split[x], axis=1) for x in range(len(smap_insitu_am_split))]
smap_insitu_am_monthly = np.stack(smap_insitu_am_monthly, axis=0)
smap_insitu_am_monthly = np.transpose(smap_insitu_am_monthly, (1, 0))
smap_insitu_am_monthly = smap_insitu_am_monthly[:, :-1]
smap_insitu_am_monthly = np.concatenate([array_allnan, smap_insitu_am_monthly], axis=1)

smap_1km_am_split = \
    np.hsplit(np.array(df_smap_1km_sta_am.iloc[:, 2:], dtype='float32'), monthly_seq_cumsum) # split by each month
smap_1km_am_monthly = [np.nanmean(smap_1km_am_split[x], axis=1) for x in range(len(smap_1km_am_split))]
smap_1km_am_monthly = np.stack(smap_1km_am_monthly, axis=0)
smap_1km_am_monthly = np.transpose(smap_1km_am_monthly, (1, 0))
smap_1km_am_monthly = smap_1km_am_monthly[:, :-1]
smap_1km_am_monthly = np.concatenate([array_allnan, smap_1km_am_monthly], axis=1)

smap_400m_am_split = \
    np.hsplit(np.array(df_smap_400m_sta_am.iloc[:, 2:], dtype='float32'), monthly_seq_cumsum) # split by each month
smap_400m_am_monthly = [np.nanmean(smap_400m_am_split[x], axis=1) for x in range(len(smap_400m_am_split))]
smap_400m_am_monthly = np.stack(smap_400m_am_monthly, axis=0)
smap_400m_am_monthly = np.transpose(smap_400m_am_monthly, (1, 0))
smap_400m_am_monthly = smap_400m_am_monthly[:, :-1]
smap_400m_am_monthly = np.concatenate([array_allnan, smap_400m_am_monthly], axis=1)

smap_9km_am_split = \
    np.hsplit(np.array(df_smap_9km_sta_am.iloc[:, 2:], dtype='float32'), monthly_seq_cumsum) # split by each month
smap_9km_am_monthly = [np.nanmean(smap_9km_am_split[x], axis=1) for x in range(len(smap_9km_am_split))]
smap_9km_am_monthly = np.stack(smap_9km_am_monthly, axis=0)
smap_9km_am_monthly = np.transpose(smap_9km_am_monthly, (1, 0))
smap_9km_am_monthly = smap_9km_am_monthly[:, :-1]
smap_9km_am_monthly = np.concatenate([array_allnan, smap_9km_am_monthly], axis=1)

smap_gpm_split = \
    np.hsplit(np.array(df_table_gpm.iloc[:, 2:], dtype='float32'), monthly_seq_cumsum) # split by each month
smap_gpm_monthly = [np.nansum(smap_gpm_split[x], axis=1) for x in range(len(smap_gpm_split))]
smap_gpm_monthly = np.stack(smap_gpm_monthly, axis=0)
smap_gpm_monthly = np.transpose(smap_gpm_monthly, (1, 0))
smap_gpm_monthly = smap_gpm_monthly[:, :-1]
smap_gpm_monthly = np.concatenate([array_allnan, smap_gpm_monthly], axis=1)

# Make the time-series plots
network_name = ['TxSON', 'SoilSCAPE', 'REMEDHUS']
site_ind = list(np.array([[1297, 1305, 1325, 1330], [1205, 1206, 1207, 1215], [2265, 2266, 2276, 2284]]) - 2)
# network_name = ['TxSON']
# site_ind = [[8, 12, 24, 30]]

# Network 1
for inw in range(len(site_ind)):

    fig = plt.figure(figsize=(10, 8), dpi=200)
    plt.subplots_adjust(left=0.12, right=0.88, bottom=0.08, top=0.9, hspace=0.3, wspace=0.25)
    for ist in range(len(site_ind[inw])):

        x = smap_insitu_am_monthly[site_ind[inw][ist], :]
        y0 = smap_400m_am_monthly[site_ind[inw][ist], :]
        y1 = smap_1km_am_monthly[site_ind[inw][ist], :]
        y2 = smap_9km_am_monthly[site_ind[inw][ist], :]
        z = smap_gpm_monthly[site_ind[inw][ist], :]
        ind_nonnan = np.where(~np.isnan(x) & ~np.isnan(y0) & ~np.isnan(y1) & ~np.isnan(y2))[0]
        # x = x[ind_nonnan]
        # y1 = y1[ind_nonnan]
        # y2 = y2[ind_nonnan]
        # z = z[ind_nonnan]

        ax = fig.add_subplot(4, 1, ist+1)
        lns1 = ax.plot(x, c='b', marker='s', label='In-situ', markersize=3, linestyle='--', linewidth=1)
        lns0 = ax.plot(y0, c='r', marker='o', label='400 m', markersize=3, linestyle='--', linewidth=1)
        lns2 = ax.plot(y1, c='k', marker='^', label='1 km', markersize=3, linestyle='--', linewidth=1)
        lns3 = ax.plot(y2, c='g', marker='x', label='9 km', markersize=3, linestyle='--', linewidth=1)
        ax.text(0, 0.45, df_smap_1km_sta_am.index[site_ind[inw][ist]].replace('_', '_'), fontsize=11, fontweight='bold')

        plt.xlim(0, len(x))
        ax.set_xticks(np.arange(0, len(x)+12, len(x)//9))
        ax.set_xticklabels([])
        labels = ['2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']
        mticks = ax.get_xticks()
        ax.set_xticks((mticks[:-1] + mticks[1:]) / 2, minor=True)
        ax.tick_params(axis='x', which='minor', length=0)
        ax.set_xticklabels(labels, minor=True)

        plt.ylim(0, 0.5)
        ax.set_yticks(np.arange(0, 0.6, 0.1))
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(linestyle='--')

        ax2 = ax.twinx()
        ax2.set_ylim(0, 150)
        ax2.invert_yaxis()
        ax2.set_yticks(np.arange(0, 180, 30))
        lns4 = ax2.bar(np.arange(len(x)), z, width=0.8, color='cornflowerblue', label='Precipitation', alpha=0.5)
        ax2.tick_params(axis='y', labelsize=10)

    # add all legends together
    handles = lns1+lns0+lns2+lns3+[lns4]
    # handles = lns1 + lns2 + [lns4]
    labels = [l.get_label() for l in handles]

    plt.legend(handles, labels, loc=(0, 4.95), mode="expand", borderaxespad=0, ncol=5, prop={"size": 10})
    fig.text(0.5, 0.02, 'Years', ha='center', fontsize=14, fontweight='bold')
    fig.text(0.02, 0.4, 'SM ($\mathregular{m^3/m^3)}$', rotation='vertical', fontsize=14, fontweight='bold')
    fig.text(0.96, 0.4, 'GPM Precipitation (mm)', rotation='vertical', fontsize=14, fontweight='bold')
    plt.suptitle(network_name[inw], fontsize=16, y=0.98, fontweight='bold')
    plt.savefig(path_results + '/validation/network_plots/' + network_name[inw] + '_tseries' + '.png')
    plt.close(fig)



# All Networks together
site_ind = [112, 758, 1259, 1921]
network_slc[site_ind[0]] = 'CTP-SMTMN'
fig = plt.figure(figsize=(10, 8), dpi=200)
plt.subplots_adjust(left=0.12, right=0.88, bottom=0.08, top=0.9, hspace=0.3, wspace=0.25)
for inw in range(len(site_ind)):

    x = smap_insitu_am_monthly[site_ind[inw], :]
    y1 = smap_1km_am_monthly[site_ind[inw], :]
    y2 = smap_9km_am_monthly[site_ind[inw], :]
    z = smap_gpm_monthly[site_ind[inw], :]
    ind_nonnan = np.where(~np.isnan(x) & ~np.isnan(y1) & ~np.isnan(y2))[0]
        # x = x[ind_nonnan]
        # y1 = y1[ind_nonnan]
        # y2 = y2[ind_nonnan]
        # z = z[ind_nonnan]

    ax = fig.add_subplot(4, 1, inw+1)
    lns1 = ax.plot(x, c='k', marker='s', label='In-situ', markersize=3, linestyle='--', linewidth=1)
    lns2 = ax.plot(y1, c='b', marker='o', label='1 km', markersize=3, linestyle='--', linewidth=1)
    lns3 = ax.plot(y2, c='g', marker='^', label='25 km', markersize=3, linestyle='--', linewidth=1)
    ax.text(0, 0.45, network_slc[site_ind[inw]], fontsize=11, fontweight='bold')
    ax.text(0, 0.4, df_smap_1km_sta_am.index[site_ind[inw]].replace('_', '_'), fontsize=11, fontweight='bold')

    plt.xlim(0, len(x))
    ax.set_xticks(np.arange(0, len(x)+12, len(x)//12))
    ax.set_xticklabels([])
    labels = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
    mticks = ax.get_xticks()
    ax.set_xticks((mticks[:-1] + mticks[1:]) / 2, minor=True)
    ax.tick_params(axis='x', which='minor', length=0)
    ax.set_xticklabels(labels, minor=True)

    plt.ylim(0, 0.5)
    ax.set_yticks(np.arange(0, 0.6, 0.1))
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(linestyle='--')

    ax2 = ax.twinx()
    ax2.set_ylim(0, 150)
    ax2.invert_yaxis()
    ax2.set_yticks(np.arange(0, 180, 30))
    lns4 = ax2.bar(np.arange(len(x)), z, width=0.8, color='cornflowerblue', label='Precipitation', alpha=0.5)
    ax2.tick_params(axis='y', labelsize=10)

# add all legends together
handles = lns1+lns2+lns3+[lns4]
# handles = lns1 + lns2 + [lns4]
labels = [l.get_label() for l in handles]

plt.legend(handles, labels, loc=(0, 4.95), mode="expand", borderaxespad=0, ncol=4, prop={"size": 10})
fig.text(0.5, 0.02, 'Years', ha='center', fontsize=14, fontweight='bold')
fig.text(0.02, 0.4, 'SM ($\mathregular{m^3/m^3)}$', rotation='vertical', fontsize=14, fontweight='bold')
fig.text(0.96, 0.4, 'GPM Precipitation (mm)', rotation='vertical', fontsize=14, fontweight='bold')
# plt.suptitle(network_name[inw], fontsize=16, y=0.98, fontweight='bold')
plt.savefig(path_results + '/validation/network_plots/all_tseries.png')
plt.close(fig)

########################################################################################################################
# 4. Plot validation results by all sites in one network

df_smap_1km_sta_am = pd.read_excel(path_ismn + '/extraction/smap_validation_1km.xlsx', index_col=0, sheet_name='AM')
df_smap_9km_sta_am = pd.read_excel(path_ismn + '/extraction/smap_validation_9km.xlsx', index_col=0, sheet_name='AM')
df_smap_insitu_sta_am = pd.read_excel(path_ismn + '/extraction/smap_validation_insitu.xlsx', index_col=0, sheet_name='AM')

df_smap_1km_sta_am_new = df_smap_1km_sta_am.reset_index()
df_smap_1km_sta_am_sp = df_smap_1km_sta_am_new.loc[df_smap_1km_sta_am_new['network'].isin(['CTP', 'HOBE', 'TxSON', 'RISMA'])]
df_smap_1km_sta_am_sp_ind = df_smap_1km_sta_am_new.index[df_smap_1km_sta_am_new['network'].isin(['CTP', 'HOBE', 'TxSON', 'RISMA'])].tolist()

df_smap_9km_sta_am_sp = df_smap_9km_sta_am.iloc[df_smap_1km_sta_am_sp_ind]
df_smap_insitu_sta_am_sp = df_smap_insitu_sta_am.iloc[df_smap_1km_sta_am_sp_ind]

stn_pos_1km_ind_all = np.ravel_multi_index([stn_row_1km_ind_all, stn_col_1km_ind_all],
                                           (len(lat_world_ease_1km), len(lon_world_ease_1km)))[df_smap_1km_sta_am_sp_ind]
stn_pos_9km_ind_all = np.ravel_multi_index([stn_row_9km_ind_all, stn_col_9km_ind_all],
                                           (len(lat_world_ease_9km), len(lon_world_ease_9km)))[df_smap_1km_sta_am_sp_ind]

df_stn_pos_9km_ind_all = pd.DataFrame({'index': stn_pos_9km_ind_all})
stn_pos_9km_ind_unique = df_stn_pos_9km_ind_all.groupby(by='index').groups
stn_pos_9km_ind_unique = pd.DataFrame(stn_pos_9km_ind_unique.items())

stn_pos_9km_name = [df_smap_1km_sta_am_sp['network'].iloc[stn_pos_9km_ind_unique.iloc[x, 1].to_list()[0]]
                     for x in range(stn_pos_9km_ind_unique.shape[0])]

# Number of points > 8
stn_slc = [1, 2, 14, 25, 26, 28]
stn_pos_9km_ind_unique_slc = stn_pos_9km_ind_unique.iloc[stn_slc]
stn_pos_9km_name_slc = [stn_pos_9km_name[stn_slc[x]] for x in range(len(stn_slc))]

df_smap_1km_sta_am_slc = [df_smap_1km_sta_am_sp.iloc[stn_pos_9km_ind_unique_slc.iloc[x, 1].to_list(), 1:]
                          for x in range(len(stn_slc))]
df_smap_9km_sta_am_slc = [df_smap_9km_sta_am_sp.iloc[stn_pos_9km_ind_unique_slc.iloc[x, 1].to_list()]
                          for x in range(len(stn_slc))]
df_smap_insitu_sta_am_slc = [df_smap_insitu_sta_am_sp.iloc[stn_pos_9km_ind_unique_slc.iloc[x, 1].to_list()]
                          for x in range(len(stn_slc))]

writer_df_smap_1km = pd.ExcelWriter(path_results + '/df_smap_1km_sta_am_slc.xlsx')
for x in range(len(df_smap_1km_sta_am_slc)):
    df_smap_1km_sta_am_slc[x].to_excel(writer_df_smap_1km, sheet_name=str(x))
writer_df_smap_1km.save()

writer_df_smap_9km = pd.ExcelWriter(path_results + '/df_smap_9km_sta_am_slc.xlsx')
for x in range(len(df_smap_9km_sta_am_slc)):
    df_smap_9km_sta_am_slc[x].to_excel(writer_df_smap_9km, sheet_name=str(x))
writer_df_smap_9km.save()

writer_df_smap_insitu = pd.ExcelWriter(path_results + '/df_smap_insitu_sta_am_slc.xlsx')
for x in range(len(df_smap_insitu_sta_am_slc)):
    df_smap_insitu_sta_am_slc[x].to_excel(writer_df_smap_insitu, sheet_name=str(x))
writer_df_smap_insitu.save()


# 4.1 plots by SMOS pixel

df_smap_1km_sta_am_slc = []
for x in range(6):
    sheet_data = pd.read_excel(path_results + '/df_smap_1km_sta_am_slc.xlsx', sheet_name=str(x), index_col=0)
    df_smap_1km_sta_am_slc.append(sheet_data)
    del(sheet_data)

df_smap_9km_sta_am_slc = []
for x in range(6):
    sheet_data = pd.read_excel(path_results + '/df_smap_9km_sta_am_slc.xlsx', sheet_name=str(x), index_col=0)
    df_smap_9km_sta_am_slc.append(sheet_data)
    del(sheet_data)

df_smap_insitu_sta_am_slc = []
for x in range(6):
    sheet_data = pd.read_excel(path_results + '/df_smap_insitu_sta_am_slc.xlsx', sheet_name=str(x), index_col=0)
    df_smap_insitu_sta_am_slc.append(sheet_data)
    del(sheet_data)

network_validation = [df_smap_1km_sta_am_slc[x]['network'].iloc[0] for x in range(len(df_smap_1km_sta_am_slc))]
network_validation = ['HOBE1', 'HOBE2', 'CTP', 'TxSON1', 'TxSON2', 'TxSON3']

# for ist in range(size_validation[0]):
df_stat_1km_all = []
df_stat_9km_all = []
for ipx in range(len(df_smap_1km_sta_am_slc)):

    stat_array_1km = []
    stat_array_9km = []
    ind_slc_all = []
    for ist in range(2, df_smap_1km_sta_am_slc[ipx].shape[1]):
        x = np.array(df_smap_insitu_sta_am_slc[ipx].iloc[:, ist], dtype=np.float)
        y1 = np.array(df_smap_1km_sta_am_slc[ipx].iloc[:, ist], dtype=np.float)
        y2 = np.array(df_smap_9km_sta_am_slc[ipx].iloc[:, ist], dtype=np.float)
        # x[x == 0] = np.nan
        # y1[y1 == 0] = np.nan
        # y2[y2 == 0] = np.nan

        ind_nonnan = np.where(~np.isnan(x) & ~np.isnan(y1) & ~np.isnan(y2))[0]
        if len(ind_nonnan) > 5:
            x = x[ind_nonnan]
            y1 = y1[ind_nonnan]
            y2 = y2[ind_nonnan]

            slope_1, intercept_1, r_value_1, p_value_1, std_err_1 = stats.linregress(x, y1)
            y1_estimated = intercept_1 + slope_1 * x
            number_1 = len(y1)
            r_sq_1 = r_value_1 ** 2
            ubrmse_1 = np.sqrt(np.mean((x - y1_estimated) ** 2))
            bias_1 = np.mean(x - y1)
            conf_int_1 = std_err_1 * 1.96  # From the Z-value
            mae_1 = np.mean(np.abs(x - y1_estimated))
            # stdev_1 = np.std(y1)
            stat_array_1 = [number_1, r_sq_1, ubrmse_1, mae_1, bias_1, p_value_1, conf_int_1]

            slope_2, intercept_2, r_value_2, p_value_2, std_err_2 = stats.linregress(x, y2)
            y2_estimated = intercept_2 + slope_2 * x
            number_2 = len(y2)
            r_sq_2 = r_value_2 ** 2
            ubrmse_2 = np.sqrt(np.mean((x - y2_estimated) ** 2))
            bias_2 = np.mean(x - y2)
            conf_int_2 = std_err_2 * 1.96  # From the Z-value
            mae_2 = np.mean(np.abs(x - y2_estimated))
            # stdev_2 = np.std(y2)
            stat_array_2 = [number_2, r_sq_2, ubrmse_2, mae_2, bias_2, p_value_2, conf_int_2]

            fig = plt.figure(figsize=(8, 5), dpi=200)
            fig.subplots_adjust(hspace=0.2, wspace=0.2)
            ax = fig.add_subplot(111)

            ax.scatter(x, y1, s=5, c='r', marker='o', label='1 km')
            ax.scatter(x, y2, s=5, c='k', marker='^', label='9 km')

            ax.plot(x, intercept_1 + slope_1 * x, '-', color='r')
            ax.plot(x, intercept_2 + slope_2 * x, '-', color='k')

            # coef1, intr1 = np.polyfit(x.squeeze(), y1.squeeze(), 1)
            # ax.plot(x.squeeze(), intr1 + coef1 * x.squeeze(), '-', color='r')
            #
            # coef2, intr2 = np.polyfit(x.squeeze(), y2.squeeze(), 1)
            # ax.plot(x.squeeze(), intr2 + coef2 * x.squeeze(), '-', color='k')

            plt.xlim(0, 0.4)
            ax.set_xticks(np.arange(0, 0.5, 0.1))
            plt.ylim(0, 0.4)
            ax.set_yticks(np.arange(0, 0.5, 0.1))
            ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
            plt.grid(linestyle='--')
            plt.legend(loc='upper left')
            plt.suptitle(date_seq[ist] +' - ' + str(ipx), fontsize=18, y=0.99, fontweight='bold')
            # plt.show()
            # plt.savefig(path_results +'/validation/single_plots/' + date_seq[ist] +' - ' + str(ipx) + '.png')
            plt.close(fig)
            stat_array_1km.append(stat_array_1)
            stat_array_9km.append(stat_array_2)
            ind_slc_all.append(ist)
            print(ist)
            del(stat_array_1, stat_array_2)
        else:
            pass

    stat_array_1km = np.array(stat_array_1km)
    stat_array_9km = np.array(stat_array_9km)

    columns_validation = ['number', 'r_sq', 'ubrmse', 'mae', 'bias', 'p_value', 'conf_int']
    index_validation = df_smap_1km_sta_am_slc[0].columns[ind_slc_all]
    ind_slc_all_array = np.array(ind_slc_all) - 2
    index_validation_2 = [date_seq[ind_slc_all_array[x]] for x in range(len(ind_slc_all_array))]
    index_validation_2 = pd.Series(index_validation_2, name='date')
    index_validation_2.index = index_validation
    df_stat_1km = pd.DataFrame(stat_array_1km, columns=columns_validation, index=index_validation)
    df_stat_9km = pd.DataFrame(stat_array_9km, columns=columns_validation, index=index_validation)
    df_stat_1km = pd.concat([index_validation_2, df_stat_1km], axis=1, sort=False)
    df_stat_9km = pd.concat([index_validation_2, df_stat_9km], axis=1, sort=False)

    df_stat_1km_all.append(df_stat_1km)
    df_stat_9km_all.append(df_stat_9km)
    del(df_stat_1km, df_stat_9km)

writer_df_stat_1km = pd.ExcelWriter(path_results + '/df_stat_1km.xlsx')
for x in range(len(df_stat_1km_all)):
    df_stat_1km_all[x].to_excel(writer_df_stat_1km, sheet_name=network_validation[x])
writer_df_stat_1km.save()

writer_df_stat_9km = pd.ExcelWriter(path_results + '/df_stat_9km.xlsx')
for x in range(len(df_stat_9km_all)):
    df_stat_9km_all[x].to_excel(writer_df_stat_9km, sheet_name=network_validation[x])
writer_df_stat_9km.save()

# writer_1km = pd.ExcelWriter(path_results + '/validation/stat_1km_smap_compare.xlsx')
# writer_9km = pd.ExcelWriter(path_results + '/validation/stat_9km_smap_compare.xlsx')
# df_stat_1km.to_excel(writer_1km)
# df_stat_9km.to_excel(writer_9km)
# writer_1km.save()
# writer_9km.save()


# Extract example stations
# txson  917, 897, 889, 922
# REMEDHUS 1481, 1482, 1491, 1499
# SoilSCAPE 836, 837, 838, 844

site_ind = list(np.array([[917, 897, 889, 922], [1481, 1482, 1491, 1499], [836, 837, 838, 844]]) - 2)
site_ind = list(itertools.chain(*site_ind))
df_stat_400m_slc = df_stat_400m.iloc[site_ind, :]
df_stat_1km_slc = df_stat_1km.iloc[site_ind, :]
df_stat_9km_slc = df_stat_9km.iloc[site_ind, :]

writer_400m = pd.ExcelWriter(path_results + '/validation_example/stat_400m_smap_slc.xlsx')
writer_1km = pd.ExcelWriter(path_results + '/validation_example/stat_1km_smap_slc.xlsx')
writer_9km = pd.ExcelWriter(path_results + '/validation_example/stat_9km_smap_slc.xlsx')
df_stat_400m_slc.to_excel(writer_400m)
df_stat_1km_slc.to_excel(writer_1km)
df_stat_9km_slc.to_excel(writer_9km)
writer_400m.save()
writer_1km.save()
writer_9km.save()
