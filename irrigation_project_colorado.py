import os
import numpy as np
import matplotlib.ticker as mticker
import geopandas as gpd
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import calendar
import datetime
import glob
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.io import MemoryFile
from rasterio.transform import from_origin, Affine
from rasterio.windows import Window
from rasterio.windows import from_bounds
from rasterio.crs import CRS
from rasterio.features import geometry_mask
from pyproj import Transformer
from scipy.ndimage import zoom
import re
from calendar import month_name
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
# (Function 2) Transforms the shapefile to match the CRS of the raster.
def prepare_shapefile_for_raster(shapefile_path, raster_path):
    """
    Transforms the shapefile to match the CRS of the raster.

    Parameters:
    shapefile_path (str): Path to the input shapefile.
    raster_path (str): Path to the raster file.

    Returns:
    gpd.GeoDataFrame: Transformed GeoDataFrame with CRS matching the raster.
    """
    # Step 1: Read the shapefile
    gdf = gpd.read_file(shapefile_path)

    # Step 2: Get the raster CRS
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs

    # Step 3: Transform shapefile CRS to match raster CRS
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)

    return gdf

#########################################################################################
# (Function 3)  Calculate weighted average of raster values for polygons in a shapefile.
def calculate_weighted_average_for_raster(transformed_gdf, raster_path):
    """
    Calculates weighted averages for a single raster using a pre-transformed shapefile.

    Parameters:
    transformed_gdf (gpd.GeoDataFrame): Shapefile transformed to match raster CRS.
    raster_path (str): Path to the raster file.

    Returns:
    pd.DataFrame: A DataFrame containing results with polygon IDs and weighted means.
    """
    results = []

    try:
        with rasterio.open(raster_path) as src:
            # Loop through polygons
            for _, row in transformed_gdf.iterrows():
                polygon = row.geometry

                # Skip invalid geometries
                if not polygon.is_valid:
                    continue

                # Mask raster with polygon
                try:
                    out_image, out_transform = rasterio.mask.mask(src, [polygon], crop=True, all_touched=True)
                    out_image = out_image[0]  # Extract the first band
                except ValueError:
                    # Skip polygons that do not intersect with the raster
                    continue

                # Exclude nodata values
                nodata = src.nodata
                if nodata is not None:
                    mask = (out_image != nodata) & ~np.isnan(out_image)
                else:
                    mask = ~np.isnan(out_image)

                # Create pixel area weights
                pixel_area = abs(out_transform[0] * out_transform[4])  # Width * Height
                intersection_areas = np.full_like(out_image, pixel_area, dtype=np.float32) * mask

                # Weighted average calculation
                pixel_values = out_image[mask]
                pixel_weights = intersection_areas[mask]

                if len(pixel_values) > 0:
                    weighted_mean = np.average(pixel_values, weights=pixel_weights)
                else:
                    weighted_mean = np.nan

                # Store results
                results.append({
                    "id": row["PARCEL_ID"],  # Assuming an 'id' column exists
                    "weighted_mean": weighted_mean
                })

    except rasterio.errors.RasterioIOError as e:
        print(f"Error processing raster {raster_path}: {e}")
        return None  # Return None if the raster could not be processed

    # Convert results to DataFrame
    return pd.DataFrame(results)


#########################################################################################
# 0. Input variables
# Specify file paths
# Path of current workspace

path_irrigation = '/Volumes/Elements2/rio_grande'
path_results = '/Volumes/Elements2/rio_grande/results_0428'


# Generate a sequence of string between start and end dates (Year + DOY)
start_date = '2010-01-01'
end_date = '2024-12-31'
year = 2024 - 2010 + 1

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
yearname = np.linspace(2010, 2024, 15, dtype='int')
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


########################################################################################################################
# 1. Subset the binary classified irrigation maps in Colorado

# 1.1 Calculate and write 8-day averaged SMAP SM to Geotiff file
tif_files_all = []
for iyr in range(5, len(yearname)-1):
    tif_files = sorted(glob.glob(path_irrigation + '/smap_400m/' + str(yearname[iyr]) + '/*.tif'))
    tif_files = tif_files[0:216] # Feb 26 -
    tif_files_all.append(tif_files)
    del(tif_files)

tif_files_all = list(itertools.chain(*tif_files_all))
tif_files_names_all = [[os.path.basename(tif_files_all[x]).split('_')[-1][0:7]] for x in range(len(tif_files_all))]
kwargs_output = rasterio.open(tif_files_all[0]).meta

src_tf_irr_all = []
for idt in range(len(tif_files_all)):
    src_tf = rasterio.open(tif_files_all[idt]).read().squeeze()
    src_tf_irr_all.append(src_tf)
    print(idt)
    del(src_tf)

src_tf_irr_all = np.array(src_tf_irr_all)
divide_ind = np.arange(0, len(src_tf_irr_all), 8)
src_tf_irr_all_divide = np.split(src_tf_irr_all, divide_ind[1:], axis=0)
# print_date = [date_seq[4748:5113][divide_ind[x]] for x in range(len(divide_ind))] # For the name of days in 2023
print_date = [datetime.datetime.strptime(tif_files_names_all[x][0], "%Y%j").strftime("%Y%m%d")
              for x in range(len(tif_files_names_all))]
print_date = print_date[0::8]

for idt in range(len(src_tf_irr_all_divide)):
    src_tf_irr_all_avg = np.nanmean(src_tf_irr_all_divide[idt], axis=0)
    src_tf_irr_all_avg = np.expand_dims(src_tf_irr_all_avg, axis=0)
    with rasterio.open(path_irrigation + '/smap_400m_8day/smap_400m_' + print_date[idt] + '.tif', 'w',
                       **kwargs_output) as output_ds:
        output_ds.write(src_tf_irr_all_avg)
    print(print_date[idt])
    del(src_tf_irr_all_avg)


# 1.2 Calculate and write 8-day averaged SMAP SM (1km) to Geotiff file
tif_files_all = []
for iyr in range(5, len(yearname)-1):
    tif_files = sorted(glob.glob(path_irrigation + '/smap_1km/' + str(yearname[iyr]) + '/*.tif'))
    if iyr == 5:
        tif_files = tif_files[275:] + tif_files[:275]
    else:
        pass
    tif_files = tif_files[0:216] # Feb 26 -
    tif_files_all.append(tif_files)
    del(tif_files)

tif_files_all = list(itertools.chain(*tif_files_all))
tif_files_names_all = [[os.path.basename(tif_files_all[x]).split('_')[-1][0:7]] for x in range(len(tif_files_all))]
kwargs_output = rasterio.open(tif_files_all[0]).meta

src_tf_irr_all = []
for idt in range(len(tif_files_all)):
    src_tf = rasterio.open(tif_files_all[idt]).read().squeeze()
    src_tf_irr_all.append(src_tf)
    print(idt)
    del(src_tf)

src_tf_irr_all = np.array(src_tf_irr_all)
divide_ind = np.arange(0, len(src_tf_irr_all), 8)
src_tf_irr_all_divide = np.split(src_tf_irr_all, divide_ind[1:], axis=0)
# print_date = [date_seq[4748:5113][divide_ind[x]] for x in range(len(divide_ind))] # For the name of days in 2023
print_date = [datetime.datetime.strptime(tif_files_names_all[x][0], "%Y%j").strftime("%Y%m%d")
              for x in range(len(tif_files_names_all))]
print_date = print_date[0::8]

for idt in range(len(src_tf_irr_all_divide)):
    src_tf_irr_all_avg = np.nanmean(src_tf_irr_all_divide[idt], axis=0)
    src_tf_irr_all_avg = np.expand_dims(src_tf_irr_all_avg, axis=0)
    with rasterio.open(path_irrigation + '/smap_1km_8day/smap_1km_' + print_date[idt] + '.tif', 'w',
                       **kwargs_output) as output_ds:
        output_ds.write(src_tf_irr_all_avg)
    print(print_date[idt])
    del(src_tf_irr_all_avg)



# 2.1 Calculate and write monthly averaged SMAP SM to Geotiff file
divider = [4, 4, 4, 4, 4, 4, 3]
divider = np.tile(divider, 8)
divider = np.cumsum(divider)
yearname_sub = yearname[6:-1]
monthname_sub = monthname[2:9]
print_date = [f"{year}_{month}" for year in yearname_sub for month in monthname_sub]

tif_files_all = sorted(glob.glob(path_irrigation + '/smap_400m_8day/*.tif'))
tif_files_all = tif_files_all[27:]
kwargs_output = rasterio.open(tif_files_all[0]).meta

src_tf_irr_all = []
for idt in range(len(tif_files_all)):
    src_tf = rasterio.open(tif_files_all[idt]).read().squeeze()
    src_tf_irr_all.append(src_tf)
    print(idt)
    del(src_tf)
src_tf_irr_all = np.array(src_tf_irr_all)
src_tf_irr_all_divide = np.split(src_tf_irr_all, divider, axis=0)[:-1]

for idt in range(len(src_tf_irr_all_divide)):
    src_tf_irr_all_avg = np.nanmean(src_tf_irr_all_divide[idt], axis=0)
    src_tf_irr_all_avg = np.expand_dims(src_tf_irr_all_avg, axis=0)
    with rasterio.open(path_irrigation + '/smap_400m_monthly/smap_400m_' + print_date[idt] + '.tif', 'w',
                       **kwargs_output) as output_ds:
        output_ds.write(src_tf_irr_all_avg)
    print(print_date[idt])
    del(src_tf_irr_all_avg)

del(tif_files_all, kwargs_output, src_tf_irr_all, src_tf_irr_all_divide)

# 2.2 CHRIPS
tif_files_all = sorted(glob.glob(path_irrigation + '/CHIRPS_SLV_TIFs/*.tif'))
tif_files_all = tif_files_all[27:]
kwargs_output = rasterio.open(tif_files_all[0]).meta

src_tf_irr_all = []
for idt in range(len(tif_files_all)):
    src_tf = rasterio.open(tif_files_all[idt]).read().squeeze()
    src_tf_irr_all.append(src_tf)
    print(idt)
    del(src_tf)
src_tf_irr_all = np.array(src_tf_irr_all)
src_tf_irr_all_divide = np.split(src_tf_irr_all, divider, axis=0)[:-1]

for idt in range(len(src_tf_irr_all_divide)):
    src_tf_irr_all_avg = np.nansum(src_tf_irr_all_divide[idt], axis=0)
    src_tf_irr_all_avg = np.expand_dims(src_tf_irr_all_avg, axis=0)
    with rasterio.open(path_irrigation + '/chirps_monthly/chirps_' + print_date[idt] + '.tif', 'w',
                       **kwargs_output) as output_ds:
        output_ds.write(src_tf_irr_all_avg)
    print(print_date[idt])
    del(src_tf_irr_all_avg)

del(tif_files_all, kwargs_output, src_tf_irr_all, src_tf_irr_all_divide)

# 2.3 MODIS ET
tif_files_all = sorted(glob.glob(path_irrigation + '/MODIS_SLV_ET/*.tif'))
tif_files_all = tif_files_all[27:]
kwargs_output = rasterio.open(tif_files_all[0]).meta

src_tf_irr_all = []
for idt in range(len(tif_files_all)):
    src_tf = rasterio.open(tif_files_all[idt]).read().squeeze()
    src_tf_irr_all.append(src_tf)
    print(idt)
    del(src_tf)
src_tf_irr_all = np.array(src_tf_irr_all)
src_tf_irr_all_divide = np.split(src_tf_irr_all, divider, axis=0)[:-1]

for idt in range(len(src_tf_irr_all_divide)):
    src_tf_irr_all_avg = np.nansum(src_tf_irr_all_divide[idt], axis=0)
    src_tf_irr_all_avg = np.expand_dims(src_tf_irr_all_avg, axis=0).astype('int16')
    with rasterio.open(path_irrigation + '/modis_monthly/modis_' + print_date[idt] + '.tif', 'w',
                       **kwargs_output) as output_ds:
        output_ds.write(src_tf_irr_all_avg)
    print(print_date[idt])
    del(src_tf_irr_all_avg)

del(tif_files_all, kwargs_output, src_tf_irr_all, src_tf_irr_all_divide)


# 2.4 Calculate and write 8-day averaged OpenET (1km) to Geotiff file
tif_files = sorted(glob.glob(path_irrigation + '/OpenET/*.tif'))
# Create a mapping from month name to number
month_to_num = {month: index for index, month in enumerate(month_name) if month}

# Sort by (year, month number)
sorted_list = sorted(
    tif_files,
    key=lambda path: (
        int(re.search(r'_(\d{4})\.tif', path).group(1)),  # year
        month_to_num[re.search(r'SLV_(\w+)_\d{4}', path).group(1)]  # month number
    )
)

sorted_list_all = []
for idt in range(len(sorted_list)):
    src_tf = rasterio.open(sorted_list[idt]).read().squeeze()
    sorted_list_all.append(src_tf)
    print(idt)
    del(src_tf)

# smap_tif_files = sorted(glob.glob(path_irrigation + '/smap_400m_8day/*.tif'))


########################################################################################################################
# 2. Make maps comparing SMAP/CHRIPS/MODIS
# Extents: 37.5559°N, 106.3328°W : 37.9286°N, 105.7552°W
# latlon_bounds = (-106.33, 37.56, -105.76, 37.93)
latlon_bounds = (-107.16232, 36.97396, -105.24800, 38.41805)

# SMAP SM
smap_tif_files = sorted(glob.glob(path_irrigation + '/smap_400m_8day/*.tif'))

smap_sm_all = []
for idt in range(len(smap_tif_files)):
    src_tf = rasterio.open(smap_tif_files[idt])
    transformer = Transformer.from_crs("EPSG:4326", src_tf.crs, always_xy=True)
    minx, miny = transformer.transform(latlon_bounds[0], latlon_bounds[1])
    maxx, maxy = transformer.transform(latlon_bounds[2], latlon_bounds[3])
    window = from_bounds(minx, miny, maxx, maxy, transform=src_tf.transform)
    src_tf_subset = src_tf.read(window=window).squeeze()
    smap_sm_all.append(src_tf_subset)
    del(src_tf, src_tf_subset, transformer, window)

# CHRIPS precipitation
chrips_tif_files = sorted(glob.glob(path_irrigation + '/IrrMapper2016_2023/*.tif'))

chrips_pcpn_all = []
for idt in range(len(chrips_tif_files)):
    src_tf = rasterio.open(chrips_tif_files[idt])
    transformer = Transformer.from_crs("EPSG:4326", src_tf.crs, always_xy=True)
    minx, miny = transformer.transform(latlon_bounds[0], latlon_bounds[1])
    maxx, maxy = transformer.transform(latlon_bounds[2], latlon_bounds[3])
    window = from_bounds(minx, miny, maxx, maxy, transform=src_tf.transform)
    src_tf_subset = src_tf.read(window=window).squeeze()
    src_tf_subset[src_tf_subset < 0] = np.nan
    chrips_pcpn_all.append(src_tf_subset)
    del(src_tf, src_tf_subset, transformer, window)

# MODIS ET
modis_tif_files = sorted(glob.glob(path_irrigation + '/MODIS_SLV_ET/*.tif'))

modis_et_all = []
for idt in range(len(modis_tif_files)):
    src_tf = rasterio.open(modis_tif_files[idt])
    transformer = Transformer.from_crs("EPSG:4326", src_tf.crs, always_xy=True)
    minx, miny = transformer.transform(latlon_bounds[0], latlon_bounds[1])
    maxx, maxy = transformer.transform(latlon_bounds[2], latlon_bounds[3])
    window = from_bounds(minx, miny, maxx, maxy, transform=src_tf.transform)
    src_tf_subset = src_tf.read(window=window).squeeze()
    src_tf_subset = src_tf_subset*0.1
    src_tf_subset[src_tf_subset <= 0] = np.nan
    modis_et_all.append(src_tf_subset)
    del(src_tf, src_tf_subset, transformer, window)

# Resample CHIRPS and MODIS data to the same dimensions as SMAP
shape_smap = smap_sm_all[0].shape
shape_chrips = chrips_pcpn_all[0].shape
shape_modis = modis_et_all[0].shape

chrips_pcpn_res_all = []
for idt in range(len(chrips_pcpn_all)):
    chrips_pcpn_res = zoom(chrips_pcpn_all[idt],
                           (shape_smap[0]/shape_chrips[0], shape_smap[1]/shape_chrips[1]), order=0)
    chrips_pcpn_res_all.append(chrips_pcpn_res)
    del(chrips_pcpn_res)

modis_et_res_all = []
for idt in range(len(modis_et_all)):
    modis_et_res = zoom(modis_et_all[idt],
                           (shape_smap[0]/shape_modis[0], shape_smap[1]/shape_modis[1]), order=0)
    modis_et_res_all.append(modis_et_res)
    del(modis_et_res)



# Make subplot maps
latlon_extent = [latlon_bounds[0], latlon_bounds[2], latlon_bounds[1], latlon_bounds[3]]

for idt in range(len(chrips_pcpn_res_all)):
    fig = plt.figure(figsize=(6, 8), facecolor='w', edgecolor='k', dpi=150)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, hspace=0.25, wspace=0.2)
    title = os.path.basename(smap_tif_files[idt]).split('_')[-1][:8]
    # SMAP SM
    ax = fig.add_subplot(3, 1, 1, projection=ccrs.PlateCarree())
    img = ax.imshow(smap_sm_all[idt], origin='upper', vmin=np.nanmin(smap_sm_all[idt]),
                    vmax=np.nanpercentile(smap_sm_all[idt], 95), cmap='Spectral', extent=latlon_extent)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=0.5)
    gl.ylocator = mticker.MultipleLocator(base=0.5)
    gl.xlabel_style = {'size': 6}
    gl.ylabel_style = {'size': 6}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.text(-100.78, 39.32, 'SMAP SM', fontsize=8, horizontalalignment='left', verticalalignment='top', weight='bold',
            color='white')
    # ax.legend(['N/A', 'Irrigated', 'Non-irrigated'], loc='right', bbox_to_anchor=(1, 1))
    cbar = plt.colorbar(img, extend='both', orientation='vertical', pad=0.1)
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.locator_params(nbins=6)
    cbar.set_label('$\mathregular{(m^3/m^3)}$', fontsize=7)
    # CHRIPS Precip
    ax = fig.add_subplot(3, 1, 2, projection=ccrs.PlateCarree())
    img = ax.imshow(chrips_pcpn_res_all[idt], origin='upper', vmin=np.nanmin(chrips_pcpn_res_all[idt]),
                    vmax=np.nanpercentile(chrips_pcpn_res_all[idt], 95), cmap='Blues', extent=latlon_extent)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=0.5)
    gl.ylocator = mticker.MultipleLocator(base=0.5)
    gl.xlabel_style = {'size': 6}
    gl.ylabel_style = {'size': 6}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.text(-100.78, 39.32, 'CHIRPS PCPN', fontsize=8, horizontalalignment='left', verticalalignment='top', weight='bold')
    cbar = plt.colorbar(img, extend='both', orientation='vertical', pad=0.1)
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.locator_params(nbins=6)
    cbar.set_label('(mm)', fontsize=7)
    # MODIS ET
    ax = fig.add_subplot(3, 1, 3, projection=ccrs.PlateCarree())
    img = ax.imshow(modis_et_res_all[idt], origin='upper', vmin=np.nanmin(modis_et_res_all[idt]),
                    vmax=np.nanpercentile(modis_et_res_all[idt], 95), cmap='RdYlGn', extent=latlon_extent)
    gl = ax.gridlines(draw_labels=True, linestyle='--', linewidth=0.5, alpha=0.5, color='black')
    gl.xlocator = mticker.MultipleLocator(base=0.5)
    gl.ylocator = mticker.MultipleLocator(base=0.5)
    gl.xlabel_style = {'size': 6}
    gl.ylabel_style = {'size': 6}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.text(-100.78, 39.32, 'MODIS ET', fontsize=8, horizontalalignment='left', verticalalignment='top', weight='bold')
    cbar = plt.colorbar(img, extend='both', orientation='vertical', pad=0.1)
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.locator_params(nbins=6)
    cbar.set_label('$\mathregular{(kg/m^2)}$', fontsize=7)
    fig.text(0.47, 0.95, title + ' - ' + str(int(title)+7), ha='center', fontsize=14, fontweight='bold')
    fig.text(0.05, 0.73, 'SMAP SM', rotation='vertical', fontsize=14, fontweight='bold')
    fig.text(0.05, 0.4, 'CHIRPS PCPN', rotation='vertical', fontsize=14, fontweight='bold')
    fig.text(0.05, 0.1, 'MODIS ET', rotation='vertical', fontsize=14, fontweight='bold')
    plt.savefig(path_results + '/maps/sm_precip_et_' + title + '.png')
    plt.close()
    print(idt)



########################################################################################################################
# 3. Extract weighted average value corresponding to each crop field

# 8-day
shapefile_path = sorted(glob.glob(path_irrigation + '/Div3_Irrig_2015_2021/*.shp'))
raster_path_smap = sorted(glob.glob(path_irrigation + '/smap_400m_8day/*.tif'))
raster_path_smap = [raster_path_smap[i*27:(i+1)*27] for i in range(9)] # split into 8 years
raster_path_smap_1km = sorted(glob.glob(path_irrigation + '/smap_1km_8day/*.tif'))
raster_path_smap_1km = [raster_path_smap_1km[i*27:(i+1)*27] for i in range(9)] # split into 8 years
raster_path_chirps = sorted(glob.glob(path_irrigation + '/CHIRPS_SLV_TIFs/*.tif'))
raster_path_chirps = [raster_path_chirps[i*27:(i+1)*27] for i in range(9)]
raster_path_modis_et = sorted(glob.glob(path_irrigation + '/MODIS_SLV_ET/*.tif'))
raster_path_modis_et = [raster_path_modis_et[i*27:(i+1)*27] for i in range(9)]
raster_path_irrigation = sorted(glob.glob(path_irrigation + '/IrrMapper2016_2023/*.tif'))
raster_path_irrigation = [[raster_path_irrigation[i]] for i in range(9)]
# raster_folders = [raster_path_smap[8:35], raster_path_modis_et[189:], raster_path_chirps[190:]]
raster_folders = [raster_path_smap, raster_path_smap_1km, raster_path_chirps, raster_path_modis_et, raster_path_irrigation]

#------------------------------------------------------------------------------------------------------------------
# monthly
divider = [4, 4, 4, 4, 4, 4, 3]
divider = np.tile(divider, 8)
divider = np.cumsum(divider)
yearname_sub = yearname[6:-1]
monthname_sub = monthname[2:9]
print_date = [f"{year}_{month}" for year in yearname_sub for month in monthname_sub]

shapefile_path = sorted(glob.glob(path_irrigation + '/Div3_Irrig_2015_2021/*.shp'))
shapefile_path = shapefile_path[1:]

raster_path_open_et = sorted(glob.glob(path_irrigation + '/OpenET/*.tif'))
month_to_num = {month: index for index, month in enumerate(month_name) if month}
# Sort by (year, month number)
raster_path_open_et = sorted(
    raster_path_open_et,
    key=lambda path: (
        int(re.search(r'_(\d{4})\.tif', path).group(1)),  # year
        month_to_num[re.search(r'SLV_(\w+)_\d{4}', path).group(1)]  # month number
    )
)
raster_path_open_et = [raster_path_open_et[i*7:(i+1)*7] for i in range(8)]

raster_path_smap = sorted(glob.glob(path_irrigation + '/smap_400m_monthly/*.tif'))
raster_path_smap = [raster_path_smap[i*7:(i+1)*7] for i in range(8)] # split into 8 years
raster_path_chirps = sorted(glob.glob(path_irrigation + '/chirps_monthly/*.tif'))
raster_path_chirps = [raster_path_chirps[i*7:(i+1)*7] for i in range(8)]
raster_path_modis_et = sorted(glob.glob(path_irrigation + '/modis_monthly/*.tif'))
raster_path_modis_et = [raster_path_modis_et[i*7:(i+1)*7] for i in range(8)]
raster_path_irrigation = sorted(glob.glob(path_irrigation + '/IrrMapper2016_2023/*.tif'))
raster_path_irrigation = raster_path_irrigation[1:]
raster_path_irrigation = [[raster_path_irrigation[i]] for i in range(8)]
# raster_folders = [raster_path_smap[8:35], raster_path_modis_et[189:], raster_path_chirps[190:]]
raster_folders = [raster_path_smap, raster_path_chirps, raster_path_modis_et, raster_path_open_et, raster_path_irrigation]

#------------------------------------------------------------------------------------------------------------------
# Extract shapefile attributes
gpd_all = []
for iyr in range(len(shapefile_path)):
    df_field = gpd.read_file(shapefile_path[iyr])
    df_field = df_field.set_index('PARCEL_ID')
    gpd_all.append(df_field)
    print(iyr)

result_df_all = []
for ifd in range(len(raster_folders)): # N raster layer folders

    result_df_1folder = []
    for iyr in range(len(shapefile_path)): # year

        transformed_gdf = prepare_shapefile_for_raster(shapefile_path[iyr], raster_folders[ifd][iyr][0])

        result_df_1year = []
        for ife in range(len(raster_folders[ifd][iyr])): # monthly data in a year
            result_df = calculate_weighted_average_for_raster(transformed_gdf, raster_folders[ifd][iyr][ife])
            result_df_1year.append(result_df)
            print(raster_folders[ifd][iyr][ife])
            del(result_df)

        result_df_1folder.append(result_df_1year)
        del(result_df_1year)

    result_df_all.append(result_df_1folder)
    del(result_df_1folder)


# Generate dates for each year
col_name_all = []
for iyr in range(len(shapefile_path)):
    col_name = [os.path.basename(raster_path_smap[iyr][x]).split('_')[2][0:8] for x in range(len(raster_path_smap[iyr]))]
    col_name_all.append(col_name)
    del(col_name)


col_name_all = print_date # For monthly data processing
col_name_all = [col_name_all[i * len(col_name_all) // 8 : (i + 1) * len(col_name_all) // 8] for i in range(8)]
# result_df_all[2][3][6] = result_df_all[2][3][5] # Fill the missing data

result_df_value_all = []
for ifd in range(len(result_df_all)):
    result_df_value_1year = []
    for iyr in range(len(shapefile_path)):  # year
        result_df_all_ind = result_df_all[ifd][iyr][0]['id']
        result_df_value = [result_df_all[ifd][iyr][x]['weighted_mean'] for x in range(len(result_df_all[ifd][iyr]))]
        result_df_value = pd.concat(result_df_value, axis=1)
        if ifd != 4: # Shapefile data folder, has only 8 files
            result_df_value.columns = col_name_all[iyr]
        else:
            result_df_value.columns = [str(yearname[iyr+5])] # Start from 2015

        result_df_value.index = result_df_all_ind
        result_df_value_1year.append(result_df_value)
        del(result_df_value, result_df_all_ind)

    result_df_value_all.append(result_df_value_1year)
    del(result_df_value_1year)


# Write extracted data to file
for iyr in range(len(raster_path_smap)):
    common_ind = [result_df_value_all[ifd][iyr].index for ifd in range(4)]
    common_ind = list(set(common_ind[0]) & set(common_ind[1]) & set(common_ind[2]) & set(common_ind[3]))

    shapefile_attributes = gpd_all[iyr].loc[common_ind][['DISTRICT', 'MASTER_ID', 'CROP_TYPE',
           'CROP_SRC', 'IRRIG_TYPE']]
    # irrigation = result_df_value_all[3][iyr]
    irrigation = result_df_value_all[4][iyr]
    irrigation.columns = ['IRRIG_STATUS']
    shapefile_attributes = shapefile_attributes.join(irrigation)
    field_values_smap = shapefile_attributes.join(result_df_value_all[0][iyr])
    field_values_smap_1km = shapefile_attributes.join(result_df_value_all[1][iyr])
    field_values_chirps = shapefile_attributes.join(result_df_value_all[2][iyr])
    field_values_modis_et = shapefile_attributes.join(result_df_value_all[3][iyr])
    # field_values_open_et = shapefile_attributes.join(result_df_value_all[4][iyr])

    writer_field = pd.ExcelWriter(path_irrigation + '/field_values/' + 'field_values_' + str(yearname[iyr+5]) + '.xlsx')
    field_values_smap.to_excel(writer_field, sheet_name='smap')
    field_values_smap_1km.to_excel(writer_field, sheet_name='smap_1km')
    field_values_chirps.to_excel(writer_field, sheet_name='chirps')
    field_values_modis_et.to_excel(writer_field, sheet_name='modis_et')
    # field_values_open_et.to_excel(writer_field, sheet_name='open_et')
    writer_field.save()

    print(str(yearname[iyr+5]))
    del(common_ind, shapefile_attributes, irrigation, field_values_smap, field_values_chirps, field_values_modis_et,
        field_values_smap_1km, writer_field)


########################################################################################################################
# 4. Summarize by crop type
# shapefile_path = path_shp + '/Div3_Irrig_2021.shp/Div3_Irrig_2021.shp'
# df_field = gpd.read_file(shapefile_path)
# df_field = df_field.set_index('PARCEL_ID')

shapefile_path = sorted(glob.glob(path_irrigation + '/field_values/*.xlsx'))
sheet_names = ['smap', 'chirps', 'modis_et']
field_values_all = []
for ife in range(len(shapefile_path)):
    field_values = [pd.read_excel(shapefile_path[ife], sheet_name=sheet_names[x], index_col=0) for x in range(3)]
    field_values_all.append(field_values)
    print(ife)
    del(field_values)

for iyr in range(len(field_values_all)):
    field_values_mean = [(field_values_all[iyr][y].drop(columns=['DISTRICT', 'MASTER_ID', 'CROP_SRC', 'IRRIG_TYPE'])
                          .groupby('CROP_TYPE').agg(lambda x: np.nanmean(x)).reset_index()) for y in range(3)]

    writer_field_mean = pd.ExcelWriter(path_results + '/field_values_crop_' + str(yearname[iyr+5]) + '.xlsx')
    field_values_mean[0].to_excel(writer_field_mean, sheet_name='smap')
    field_values_mean[1].to_excel(writer_field_mean, sheet_name='chirps')
    field_values_mean[2].to_excel(writer_field_mean, sheet_name='modis_et')
    writer_field_mean.save()

    print(str(yearname[iyr+5]))
    del(field_values_mean)



# Make time-series maps

for j in range(4):
    fig = plt.figure(figsize=(15, 6), dpi=200)
    plt.subplots_adjust(left=0.12, right=0.88, bottom=0.12, top=0.92, hspace=0.25, wspace=0.25)
    for i in range(3):
        x = field_values_smap_mean.loc[j*3+i][1:]
        y = field_values_chirps_mean.loc[j*3+i][1:]
        ax = fig.add_subplot(3, 1, i+1)
        lns1 = ax.plot(x, c='g', marker='o', label='Soil Moisture', markersize=3, linestyle='--', linewidth=1)
        plt.show()
        plt.xlim(0, len(x))
        plt.xticks(rotation=45)
        # ax.set_xticks(np.arange(0, 367, 30.5))
        ax.set_xticklabels(df_columns)
        if i!=2:
            ax.set_xticklabels([])
        # x_labels = df_columns
        # mticks = ax.get_xticks()
        # ax.set_xticks((mticks[:-1] + mticks[1:]) / 2, minor=True)
        # ax.tick_params(axis='x', which='minor', length=0)

        plt.ylim(0, 0.3)
        ax.set_yticks(np.arange(0, 0.36, 0.06))
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(linestyle='--')
        ax.text(0.35, 0.26, crop_type[j*3+i], fontsize=10)

        ax2 = ax.twinx()
        ax2.set_ylim(0, 25)
        ax2.invert_yaxis()
        ax2.set_yticks(np.arange(0, 30, 5))
        lns2 = ax2.bar(np.arange(len(x)), y, width=0.8, color='cornflowerblue', label='Precipitation', alpha=0.5)
        ax2.tick_params(axis='y', labelsize=10)
        handles = lns1 + [lns2]
        labels = [l.get_label() for l in handles]

        plt.legend(handles, labels, loc=(0, 3.52), mode="expand", borderaxespad=0, ncol=4, prop={"size": 10})
        fig.text(0.5, 0.01, 'Date', ha='center', fontsize=14, fontweight='bold')
        fig.text(0.02, 0.3, 'SMAP SM ($\mathregular{m^3/m^3)}$', rotation='vertical', fontsize=14, fontweight='bold')
        fig.text(0.96, 0.3, 'GPM Precipitation (mm)', rotation='vertical', fontsize=14, fontweight='bold')
        # plt.suptitle(str(yearname[iyr]) + '(#' + str(j*3+1) + '-' + str(j*3+3) + ')',
        #              fontsize=16, y=0.98, fontweight='bold')
    plt.savefig(path_results + '/time_series/smap_gpm_' + '_' + str(j*3+1) + '-' + str(j*3+3) + '.png')
    plt.close(fig)



