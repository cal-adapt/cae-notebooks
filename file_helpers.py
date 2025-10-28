# Import libraries and functions
import zarr
from bs4 import BeautifulSoup
import requests
import boto3
import zipfile
import io
import pandas as pd
import os
import shutil

def delete_items(folders, csv_files, zip_files=None, gdb_folders=None, png_files=None):
    """
    Deletes the specified folders, CSV files, ZIP files, GDB folders, and PNG files if they exist.

    Parameters:
        folders (list): List of folder paths to delete.
        csv_files (list): List of CSV file paths to delete.
        zip_files (list, optional): List of ZIP file paths to delete. Default is None.
        gdb_folders (list, optional): List of GDB folder paths to delete. Default is None.
        png_files (list, optional): List of PNG file paths to delete. Default is None.
    """
    # Delete folders
    for folder in folders:
        if os.path.exists(folder):
            if os.path.isdir(folder):
                shutil.rmtree(folder)
                print(f"Deleted folder: {folder}")
            else:
                print(f"Path is not a folder: {folder}")
        else:
            print(f"Folder does not exist: {folder}")
    
    # Delete CSV files
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            if os.path.isfile(csv_file):
                os.remove(csv_file)
                print(f"Deleted file: {csv_file}")
            else:
                print(f"Path is not a file: {csv_file}")
        else:
            print(f"File does not exist: {csv_file}")
    
    # Delete ZIP files
    if zip_files:
        for zip_file in zip_files:
            if os.path.exists(zip_file):
                if os.path.isfile(zip_file) and zip_file.endswith('.zip'):
                    os.remove(zip_file)
                    print(f"Deleted ZIP file: {zip_file}")
                else:
                    print(f"Path is not a valid ZIP file: {zip_file}")
            else:
                print(f"ZIP file does not exist: {zip_file}")
    
    # Delete GDB folders
    if gdb_folders:
        for gdb_folder in gdb_folders:
            if os.path.exists(gdb_folder):
                if os.path.isdir(gdb_folder) and gdb_folder.endswith('.gdb'):
                    shutil.rmtree(gdb_folder)
                    print(f"Deleted GDB folder: {gdb_folder}")
                else:
                    print(f"Path is not a valid GDB folder: {gdb_folder}")
            else:
                print(f"GDB folder does not exist: {gdb_folder}")
    
    # Delete PNG files
    if png_files:
        for png_file in png_files:
            if os.path.exists(png_file):
                if os.path.isfile(png_file) and png_file.endswith('.png'):
                    os.remove(png_file)
                    print(f"Deleted PNG file: {png_file}")
                else:
                    print(f"Path is not a valid PNG file: {png_file}")
            else:
                print(f"PNG file does not exist: {png_file}")
                
def to_zarr(ds, top_dir, domain, indicator, data_source, save_name):
    """Converts netcdf to zarr and sends to s3 bucket"""
    # first check that folder is not already available
    aws_path = '{0}/{1}/{2}/{3}/'.format(
        top_dir, domain, indicator, data_source
    )
    aws_path = "s3://ca-climate-index/"+aws_path
    filepath_zarr = aws_path+save_name+".zarr"
    # let xarray optimize chunks
    ds = ds.chunk(chunks="auto")
    ds.to_zarr(store=filepath_zarr, mode="w")

def list_webdir(url, ext=''):
    """Lists objects on a webpage"""
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    return [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]

def pull_csv_from_directory(bucket_name, directory, output_folder, search_zipped=True, print_name=True):
    """
    Pulls CSV files from a specified directory in an S3 bucket and saves them to a designated output folder.
    
    Parameters:
    - bucket_name (str): The name of the S3 bucket.
    - directory (str): The directory within the bucket to search for CSV files.
    - output_folder (str): The folder where the CSV files will be saved.
    - search_zipped (bool): If True, search for CSV files within zip files. If False, search for CSV files directly.
    - print_name (bool): If True, will print all filenames. If False, nothing is printed. 
    """
    # Create an S3 client
    s3 = boto3.client('s3')

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List objects in the specified directory
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=directory)

    # Check if objects were found
    if 'Contents' in response:
        # Iterate through each object found
        for obj in response['Contents']:
            # Get the key (filename) of the object
            key = obj['Key']
            
            # Check if the object is a .zip file
            if search_zipped and key.endswith('.zip'):
                # Download the zip file into memory
                zip_object = s3.get_object(Bucket=bucket_name, Key=key)
                zip_data = io.BytesIO(zip_object['Body'].read())
                
                # Open the zip file
                with zipfile.ZipFile(zip_data, 'r') as zip_ref:
                    # Iterate through each file in the zip
                    for file_name in zip_ref.namelist():
                        # Check if the file is a .csv file
                        if file_name.endswith('.csv'):
                            # Read the .csv file
                            with zip_ref.open(file_name) as csv_file:
                                # Convert the csv content to pandas DataFrame
                                df = pd.read_csv(csv_file)
                                # Save the DataFrame to the output folder with a similar name as the .csv file
                                df_name = os.path.join(output_folder, file_name[:-4])  # Remove .csv extension and save in folder
                                df.to_csv(f"{df_name}.csv", index=False)
                                if print_name:
                                    print(f"Saved DataFrame as '{df_name}.csv'")
            elif not search_zipped and key.endswith('.csv'):
                # Directly download the CSV file
                csv_object = s3.get_object(Bucket=bucket_name, Key=key)
                csv_data = io.BytesIO(csv_object['Body'].read())
                # Convert the csv content to pandas DataFrame
                df = pd.read_csv(csv_data)
                # Save the DataFrame to the output folder with a similar name as the .csv file
                df_name = os.path.join(output_folder, key.split('/')[-1][:-4])  # Extract filename from key and save in folder
                df.to_csv(f"{df_name}.csv", index=False)
                if print_name:
                    print(f"Saved DataFrame as '{df_name}.csv'")
            
        if print_name == False:
            print(f"Metric data retrieved from {directory}.")

    else:
        print("No objects found in the specified directory.")

def pull_gpkg_from_directory(bucket_name, directory):
    """
    Pulls GeoPackage files from a specified directory in an S3 bucket.
    
    Parameters:
    - bucket_name (str): The name of the S3 bucket.
    - directory (str): The directory within the bucket to search for GeoPackage files.
    """
    # Create an S3 client
    s3 = boto3.client('s3')

    # List objects in the specified directory
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=directory)

    # Check if objects were found
    if 'Contents' in response:
        # Iterate through each object found
        for obj in response['Contents']:
            # Get the key (filename) of the object
            key = obj['Key']
            
            # Check if the object is a .gpkg file
            if key.endswith('.gpkg'):
                # Download the GeoPackage file into memory
                gpkg_object = s3.get_object(Bucket=bucket_name, Key=key)
                gpkg_data = io.BytesIO(gpkg_object['Body'].read())
                
                # Save the GeoPackage file locally
                gpkg_filename = os.path.basename(key)
                with open(gpkg_filename, 'wb') as gpkg_file:
                    gpkg_file.write(gpkg_data.getvalue())
                
                print(f"Saved GeoPackage as '{gpkg_filename}' locally")
                # You can now use the saved file for further processing
    else:
        print("No objects found in the specified directory.")

def pull_nc_from_directory(file_to_grab, filename):
    """Downloads a nc file from specified directory in s3 bucket"""

    s3 = boto3.client('s3')
    s3.download_file("ca-climate-index", file_to_grab, filename)
    print(f'{file_to_grab} downloaded!')

def upload_csv_aws(file_names, bucket_name, directory):
    """
    Uploads CSV files to a specified directory in an S3 bucket.
    Important: When running a single file, place the file_names entry as a bracket, this
    is because the code by default, looks for a list of files to upload
    
    Parameters:
    - bucket_name (str): The name of the S3 bucket.
    - directory (str): The directory within the bucket to search for CSV files.
    - file_names (str): .csv file to be uploaded to aws
    """  
    # Create an S3 client
    s3 = boto3.client('s3')
    # Iterate over each file name in the list
    for file_name in file_names:
        # Save the file to AWS S3 using the client
        with open(file_name, 'rb') as data:
            s3.upload_fileobj(data, bucket_name, f"{directory}/{file_name}")
            print(f"{file_name} uploaded to AWS")

def filter_counties(df, county_column, county_list=None):
    '''
    Filter a df's county column to a list of established CA counties
    Parameters
    ----------
    df: dataframe
        name of the dataframe to be filtered
    
    column: string
        name of the county column within your dataframe
    
    county_list: list
        list of counties to be filtered for, if left blank the default list is CA counties shown below
    '''

    # Default county list if not provided
    if county_list is None:
        county_list = [
                'alameda', 'alpine', 'amador', 'butte', 'calaveras', 'colusa', 'contra costa', 'del norte',
                'el dorado', 'fresno', 'glenn', 'humboldt', 'imperial', 'inyo', 'kern', 'kings', 'lake', 'lassen',
                'los angeles', 'madera', 'marin', 'mariposa', 'mendocino', 'merced', 'modoc', 'mono', 'monterey',
                'napa', 'nevada', 'orange', 'placer', 'plumas', 'riverside', 'sacramento', 'san benito',
                'san bernardino', 'san diego', 'san francisco', 'san joaquin', 'san luis obispo', 'san mateo',
                'santa barbara', 'santa clara', 'santa cruz', 'shasta', 'sierra', 'siskiyou', 'solano', 'sonoma',
                'stanislaus', 'sutter', 'tehama', 'trinity', 'tulare', 'tuolumne', 'ventura', 'yolo', 'yuba'
            ]
    
    # Convert county_list to lowercase for case-insensitive comparison
    county_list_lower = [county.lower() for county in county_list]
    
    # Filter rows where the value in the specified column matches any of the counties in the list
    filtered_df = df[df[county_column].str.lower().isin(county_list_lower)]
    
    # Omitted rows
    omitted_df = df[~df[county_column].str.lower().isin(county_list_lower)]
    
    return filtered_df, omitted_df

def data_stats_check(df, col):
    print('Calculating stats on {}...'.format(col))
    print('Data min: ', df[col].min())
    print('Data max: ', df[col].max())
    print('Data mean: ', df[col].mean())
    print('\n')

def county_count(df, county_col, county, counter):
    county_isolate = df[df[county_col]==county]
    county_isolate_drop_duplicates= county_isolate.drop_duplicates(subset=[county_col, counter])
    print(f'Length of df for {county} county without dropping duplicates:  {len(county_isolate)}')
    print(f'Length of df for {county} county after dropping duplicates: {len(county_isolate_drop_duplicates)}')
