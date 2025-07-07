########################################################################################################################
# 1.0. Fill the missing data for each VIIRS tile (root-directory)

# 1.0.1 Yearly folder
for iyr in [12]:#range(len(yearname)):
    tile_name = sorted(glob.glob(path_viirs_missing + 'Yearly/' + str(yearname[iyr]) + '/*', recursive=True))
    tile_name_base = [tile_name[x].split('/')[-1].split('.')[0].split('_')[0][1:] for x in range(len(tile_name))]

    for ite in range(0, len(tile_name)):
        # Open the output tar file in write mode
        with tarfile.open(path_viirs_lst_filled + str(yearname[iyr]) + '/' + 'FINAL_LST_T' +
                          str(tile_name_base[ite]).zfill(3) + '_' + str(yearname[iyr]) + '.tar.gz', 'w') as out_tar:
            # Open the destination tar file in read mode
            with tarfile.open(path_viirs_lst + str(yearname[iyr]) + '/' + 'FINAL_LST_T' +
                          str(tile_name_base[ite]).zfill(3) + '_' + str(yearname[iyr]) + '.tar.gz', 'r') as dest_tar:
                # Copy the contents of the destination tar file to the output tar file
                for member in dest_tar.getmembers():
                    extracted_file = dest_tar.extractfile(member)
                    if extracted_file:
                        out_tar.addfile(member, extracted_file)

            # Get a list of existing file names in the output tar (initially from destination tar)
            existing_files = {os.path.basename(member.name) for member in out_tar.getmembers()}

            # Open the source tar file in read mode
            with tarfile.open(path_viirs_missing + 'Yearly/' + str(yearname[iyr]) + '/T' +
                              str(tile_name_base[ite]).zfill(3) + '_' + str(yearname[iyr]) + '.tar', 'r') as src_tar:
                for member in src_tar.getmembers():
                    # Extract only the file name (basename) from the full path
                    file_name = os.path.basename(member.name)

                    # # Check if the file already exists in the output tar
                    # if file_name in existing_files:
                    #     print(f"Skipping {file_name}, already exists.")
                    #     continue

                    # Modify the member object to use the basename instead of the full path
                    member.name = file_name

                    # Extract the file object from the source tar file
                    extracted_file = src_tar.extractfile(member)
                    if extracted_file:
                        # Add the file object to the output tar file
                        out_tar.addfile(member, extracted_file)

        print('/' + str(yearname[iyr]) + '/' + 'FINAL_LST_T' + str(tile_name_base[ite]).zfill(3) + '_' +
              str(yearname[iyr]) + '.tar.gz')



# 1.0.2 Missing folder
tile_name = sorted(glob.glob(path_viirs_missing + 'missing/*', recursive=True))
tile_name_base = [int(tile_name[x].split('/')[-1].split('.')[0][1:]) for x in range(len(tile_name))]

for iyr in [11]:#range(len(yearname)):
    for ite in range(0, len(tile_name)):
        missing_folder = path_viirs_missing + 'MISSING' + '/T' + str(tile_name_base[ite]).zfill(3) + '/' + str(yearname[iyr])

        if not any(os.scandir(missing_folder)):
            pass

        else:
            # Check if destination_tar exists
            if not os.path.exists(path_viirs_lst + str(yearname[iyr]) + '/' + 'FINAL_LST_T' +
                                  str(tile_name_base[ite]).zfill(3) + '_' + str(yearname[iyr]) + '.tar.gz'):
                # Create a new tar file and add files from the folder
                with tarfile.open(path_viirs_lst_filled + str(yearname[iyr]) + '/' + 'FINAL_LST_T' +
                                  str(tile_name_base[ite]).zfill(3) + '_' + str(yearname[iyr]) + '.tar.gz', 'w') as out_tar:
                    # Add files from the folder to the output tar file
                    for root, dirs, files in os.walk(missing_folder):
                        for file_name in files:
                            file_path = os.path.join(root, file_name)

                            # Add the file to the tar with its basename (no directory structure)
                            tar_info = out_tar.gettarinfo(file_path, arcname=file_name)
                            with open(file_path, 'rb') as file:
                                out_tar.addfile(tar_info, file)
            else:
                with tarfile.open(path_viirs_lst_filled + str(yearname[iyr]) + '/' + 'FINAL_LST_T' +
                                  str(tile_name_base[ite]).zfill(3) + '_' + str(yearname[iyr]) + '.tar.gz', 'w') as out_tar:
                    # Open the destination tar file in read mode
                    with tarfile.open(path_viirs_lst + str(yearname[iyr]) + '/' + 'FINAL_LST_T' +
                                  str(tile_name_base[ite]).zfill(3) + '_' + str(yearname[iyr]) + '.tar.gz', 'r') as dest_tar:
                        # Copy the contents of the destination tar file to the output tar file
                        for member in dest_tar.getmembers():
                            extracted_file = dest_tar.extractfile(member)
                            if extracted_file:
                                out_tar.addfile(member, extracted_file)

                    # Get a list of existing file names in the output tar (initially from destination tar)
                    existing_files = {os.path.basename(member.name) for member in out_tar.getmembers()}

                    for root, dirs, files in os.walk(missing_folder):
                        for file_name in files:
                            file_path = os.path.join(root, file_name)
                            # Check if the file already exists in the output tar
                            if file_name in existing_files:
                                pass
                                # print(f"Skipping {file_name}, already exists.")
                            else:
                                # Add the file to the tar with its basename (no directory structure)
                                tar_info = out_tar.gettarinfo(missing_folder, arcname=file_name)
                                with open(file_path, 'rb') as file:
                                    out_tar.addfile(tar_info, file)

            print('/' + str(yearname[iyr]) + '/' + 'FINAL_LST_T' + str(tile_name_base[ite]).zfill(3) + '_' +
                  str(yearname[iyr]) + '.tar')




# Test
with tarfile.open('/Volumes/UVA_data/Dataset/VIIRS/LST/Backup/FINAL_LST_T087_2015.tar', 'r') as test_tar:
    # Copy the contents of the destination tar file to the output tar file
    for member in test_tar.getmembers():
         extracted_file_test = {member.name for member in test_tar.getmembers()}
extracted_file_test = sorted(extracted_file_test)

with tarfile.open( '/Volumes/UVA_data/Dataset/VIIRS/LST/2015/FINAL_LST_T087_2015.tar', 'r') as test_org_tar:
    # Copy the contents of the destination tar file to the output tar file
    for member in test_org_tar.getmembers():
        extracted_file_test_org = {member.name for member in test_org_tar.getmembers()}
extracted_file_test_org = sorted(extracted_file_test_org)

extracted_file_test_missing = sorted(glob.glob('/Volumes/Elements2/VIIRS/Missing/MISSING/T087/2015/*', recursive=True))




########################################################################################################################
# 1.0.0 Bring the combined tar files back (deep subdirectory)

def has_root_level_files(input_tar):

    with tarfile.open(input_tar, 'r') as tar:
        for member in tar.getmembers():
            # Check if the member (file or directory) is at the root level (i.e., no '/' in the path)
            if '/' not in member.name:
                return True
    return False

def clean_tar_file(input_tar, output_tar):
    # Open the tar file to clean it
    with tarfile.open(input_tar, 'r') as tar:
        # Create a new tar file for writing the cleaned contents
        with tarfile.open(output_tar, 'w') as new_tar:
            for member in tar.getmembers():
                if member.isfile():
                    # Only add files that are not in the root directory (i.e., contain '/')
                    if '/' in member.name:
                        file_content = tar.extractfile(member)
                        if file_content:
                            # Add the file to the new tar file
                            new_tar.addfile(member, file_content)

    # print(f"Cleaned tar file saved as {output_tar}")


# Example usage:
# input_tar = 'example.tar'  # Replace with your input tar file path
# output_dir = 'cleaned_tar_files'  # Define the new directory to save the cleaned tar file

for iyr in [11]:#range(len(yearname)):
    tile_name = sorted(glob.glob(path_viirs_lst + '/' + str(yearname[iyr]) + '/*', recursive=True))
    tile_name_base = [int(tile_name[x].split('/')[-1].split('_')[2][1:]) for x in range(len(tile_name))]

    for ite in range(0, len(tile_name)):
        input_tar = (path_viirs_lst + str(yearname[iyr]) + '/' + 'FINAL_LST_T' +
                     str(tile_name_base[ite]).zfill(3) + '_' + str(yearname[iyr]) + '.tar.gz')
        output_tar = (path_viirs_lst_filled + str(yearname[iyr]) + '/' + 'FINAL_LST_T' +
                     str(tile_name_base[ite]).zfill(3) + '_' + str(yearname[iyr]) + '.tar.gz')
        if has_root_level_files(input_tar):
            clean_tar_file(input_tar, output_tar)
        else:
            pass

        print(output_tar)




# 1.0.2 Missing folder

def get_existing_filenames(input_tar):

    filenames = set()
    if os.path.exists(input_tar):
        with tarfile.open(input_tar, 'r') as tar:
            for member in tar.getmembers():
                if member.isfile():
                    # Add the filename (without path) to the set
                    filenames.add(os.path.basename(member.name))
    return filenames


def combine_gz_files_with_tar(input_tar, gz_files_dir, output_tar):

    # Check if gz_files_dir is empty
    gz_files = [f for f in os.listdir(gz_files_dir) if f.endswith('.gz')]
    if not gz_files:
        print(f"The directory {gz_files_dir} is empty or contains no .gz files.")
        return

    print(f"Found {len(gz_files)} .gz files in {gz_files_dir}. Proceeding...")

    # If input_tar does not exist, create a new tar with only the .gz files
    if not os.path.exists(input_tar):
        print(f"{input_tar} not found. Creating a new tar file {output_tar} with only .gz files.")
        with tarfile.open(output_tar, 'w') as new_tar:
            for gz_file in gz_files:
                gz_file_path = os.path.join(gz_files_dir, gz_file)
                # Add .gz file directly under root in the new tar
                with open(gz_file_path, 'rb') as f:
                    gz_member = tarfile.TarInfo(name=gz_file)
                    gz_member.size = os.path.getsize(gz_file_path)
                    new_tar.addfile(gz_member, f)
        print(f"New tar file created at {output_tar} with .gz files from {gz_files_dir}.")
        return

    # If input_tar exists, combine .gz files into the tar while skipping duplicates
    first_file_dir = get_first_file_directory(input_tar)
    if not first_file_dir:
        raise ValueError("The original tar file does not contain any files.")

    # Get the set of existing filenames in the tar
    existing_filenames = get_existing_filenames(input_tar)

    with tarfile.open(input_tar, 'r') as tar:
        with tarfile.open(output_tar, 'w') as new_tar:
            # Copy all members from the original tar to the new tar
            for member in tar.getmembers():
                file_content = tar.extractfile(member)
                if file_content:
                    new_tar.addfile(member, file_content)

            # Add .gz files from the local directory into the new tar under the extracted directory path
            for gz_file in gz_files:
                # Skip the file if it already exists in the tar
                if gz_file in existing_filenames:
                    print(f"Skipping existing file: {gz_file}")
                    continue

                gz_file_path = os.path.join(gz_files_dir, gz_file)
                # Construct the new path under the extracted directory structure
                gz_member_name = os.path.join(first_file_dir, gz_file)
                with open(gz_file_path, 'rb') as f:
                    gz_member = tarfile.TarInfo(name=gz_member_name)
                    gz_member.size = os.path.getsize(gz_file_path)
                    new_tar.addfile(gz_member, f)

    print(f"Combined tar file saved as {output_tar}")


def get_first_file_directory(input_tar):

    with tarfile.open(input_tar, 'r') as tar:
        for member in tar.getmembers():
            if member.isfile():
                # Get the directory path of the first file
                return os.path.dirname(member.name)
    return None


# # Example usage:
# input_tar = 'example.tar'  # Path to the original tar file (if exists)
# gz_files_dir = 'gz_files'  # Directory containing the .gz files to add
# output_tar = 'combined_output.tar'  # Path to the new combined tar file

# Combine the original tar with .gz files, skipping existing ones or creating a new tar if input_tar does not exist
tile_name = sorted(glob.glob(path_viirs_missing + 'missing/*', recursive=True))
tile_name_base = [int(tile_name[x].split('/')[-1].split('.')[0][1:]) for x in range(len(tile_name))]

for iyr in [11]:#range(len(yearname)):
    for ite in range(0, len(tile_name)):
        gz_files_dir = path_viirs_missing + 'MISSING' + '/T' + str(tile_name_base[ite]).zfill(3) + '/' + str(yearname[iyr])
        input_tar = (path_viirs_lst + str(yearname[iyr]) + '/' + 'FINAL_LST_T' +
                     str(tile_name_base[ite]).zfill(3) + '_' + str(yearname[iyr]) + '.tar.gz')
        output_tar = (path_viirs_lst_filled + str(yearname[iyr]) + '/' + 'FINAL_LST_T' +
                     str(tile_name_base[ite]).zfill(3) + '_' + str(yearname[iyr]) + '.tar.gz')
        combine_gz_files_with_tar(input_tar, gz_files_dir, output_tar)


# 1.0.1 Yearly folder

def get_existing_filenames(input_tar):

    filenames = set()
    if os.path.exists(input_tar):
        with tarfile.open(input_tar, 'r') as tar:
            for member in tar.getmembers():
                if member.isfile():
                    # Add the filename (without path) to the set
                    filenames.add(os.path.basename(member.name))
    return filenames


def combine_gz_files_with_tar(input_tar, gz_files_tar, output_tar):

    # Check if gz_files_tar contains .gz files
    with tarfile.open(gz_files_tar, 'r') as gz_tar:
        gz_files = [member for member in gz_tar.getmembers() if member.isfile() and member.name.endswith('.gz')]

    if not gz_files:
        print(f"The tar file {gz_files_tar} contains no .gz files.")
        return

    print(f"Found {len(gz_files)} .gz files in {gz_files_tar}. Proceeding...")

    # If input_tar does not exist, create a new tar with .gz files under the specified path
    if not os.path.exists(input_tar):
        print(
            f"{input_tar} not found. Creating a new tar file {output_tar} with .gz files under 'raid1/vmishra/VIIRS_LST/FINAL_VIIRS_TILES/JPSS'.")
        with tarfile.open(output_tar, 'w') as new_tar:
            for gz_file in gz_files:
                gz_member_name = os.path.join('raid1/vmishra/VIIRS_LST/FINAL_VIIRS_TILES/JPSS',
                                              os.path.basename(gz_file.name))
                gz_file_content = gz_tar.extractfile(gz_file)
                gz_member = tarfile.TarInfo(name=gz_member_name)
                gz_member.size = gz_file.size
                new_tar.addfile(gz_member, gz_file_content)
        print(f"New tar file created at {output_tar} with .gz files from {gz_files_tar}.")
        return

    # If input_tar exists, combine .gz files into the tar while skipping duplicates
    first_file_dir = get_first_file_directory(input_tar)
    if not first_file_dir:
        raise ValueError("The original tar file does not contain any files.")

    # Get the set of existing filenames in the tar
    existing_filenames = get_existing_filenames(input_tar)

    with tarfile.open(input_tar, 'r') as tar:
        with tarfile.open(output_tar, 'w') as new_tar:
            # Copy all members from the original tar to the new tar
            for member in tar.getmembers():
                file_content = tar.extractfile(member)
                if file_content:
                    new_tar.addfile(member, file_content)

            # Add .gz files from the tar file into the new tar under the extracted directory path
            with tarfile.open(gz_files_tar, 'r') as gz_tar:
                for gz_file in gz_files:
                    gz_basename = os.path.basename(gz_file.name)
                    # Skip the file if it already exists in the tar
                    if gz_basename in existing_filenames:
                        print(f"Skipping existing file: {gz_basename}")
                        continue

                    gz_file_content = gz_tar.extractfile(gz_file)
                    gz_member_name = os.path.join(first_file_dir, gz_basename)
                    gz_member = tarfile.TarInfo(name=gz_member_name)
                    gz_member.size = gz_file.size
                    new_tar.addfile(gz_member, gz_file_content)

    print(f"Combined tar file saved as {output_tar}")


def get_first_file_directory(input_tar):

    with tarfile.open(input_tar, 'r') as tar:
        for member in tar.getmembers():
            if member.isfile():
                # Get the directory path of the first file
                return os.path.dirname(member.name)
    return None


# # Example usage:
# input_tar = 'example_input.tar'  # Path to the original tar file (if exists)
# gz_files_tar = 'gz_files.tar'  # Tar file containing the .gz files to be added
# output_tar = 'combined_output.tar'  # Path to the new combined tar file
#
# # Combine the original tar with .gz files from another tar, skipping existing ones or creating a new tar if input_tar does not exist
# combine_gz_files_with_tar(input_tar, gz_files_tar, output_tar)

# Combine the original tar with .gz files, skipping existing ones or creating a new tar if input_tar does not exist


for iyr in [11]:#range(len(yearname)):
    tile_name = sorted(glob.glob(path_viirs_missing + 'Yearly/' + str(yearname[iyr]) + '/*', recursive=True))
    tile_name_base = [tile_name[x].split('/')[-1].split('.')[0].split('_')[0][1:] for x in range(len(tile_name))]
    for ite in [6]:#range(0, len(tile_name)):
        gz_files_dir = tile_name[ite]
        input_tar = (path_viirs_lst + str(yearname[iyr]) + '/' + 'FINAL_LST_T' +
                     str(tile_name_base[ite]).zfill(3) + '_' + str(yearname[iyr]) + '.tar.gz')
        output_tar = (path_viirs_lst_filled + str(yearname[iyr]) + '/' + 'FINAL_LST_T' +
                     str(tile_name_base[ite]).zfill(3) + '_' + str(yearname[iyr]) + '.tar.gz')
        combine_gz_files_with_tar(input_tar, gz_files_dir, output_tar)



hdf_files = sorted(glob.glob('/Volumes/Elements2/Dataset_UVA_data/VIIRS/LST_geo/2022/T078/*', recursive=True))

viirs_lst_all = []
for ife in range(364, 424):
    f = h5py.File(hdf_files[ife], "r")
    viirs_lst = f['viirs_lst'][()]
    viirs_lst_all.append(viirs_lst)
    del(viirs_lst)
    f.close()
    print(ife)

viirs_lst_all = np.array(viirs_lst_all)
viirs_lst_all = np.nanmean(viirs_lst_all, axis=0)


####################################################################################################################################

# [row_world_ease_9km_from_geo_400m_ind, col_world_ease_9km_from_geo_400m_ind] = find_easeind_lofrhi\
#     (lat_world_geo_400m, lon_world_geo_400m, interdist_ease_9km, size_world_ease_9km[0], size_world_ease_9km[1],
#      row_world_ease_9km_ind, col_world_ease_9km_ind)
# var_name_vlen = ['row_world_ease_9km_from_geo_400m_ind', 'col_world_ease_9km_from_geo_400m_ind']
#
# # Store variable-length type variables to the parameter file
# dt = h5py.special_dtype(vlen=np.int64)
# with h5py.File(path_viirs_lai + 'ds_parameters_9km.hdf5', 'w') as f:
#     for x in var_name_vlen:
#         f.create_dataset(x, data=eval(x), dtype=dt)
# f.close()
