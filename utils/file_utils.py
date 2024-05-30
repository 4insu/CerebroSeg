import os
import zipfile
import shutil

def unzip_file(zip_file_path, extract_to_directory):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_directory)

    print("Extraction completed successfully.")

#-----------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------#

def move_files(source_folder, destination_folder):
    try:
        shutil.move(source_folder, destination_folder)
        print("Files moved successfully.")
    except Exception as e:
        print(f"Error: {e}")

#-----------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------#

def delete_directory(directory):
    try:
        shutil.rmtree(directory)
        print(f"Directory '{directory}' and its contents deleted successfully.")
    except FileNotFoundError:
        print(f"Directory '{directory}' not found.")
    except Exception as e:
        print(f"Error: {e}")

#-----------------------------------------------------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------#

def rename_folder(old_name, new_name):
    try:
        os.rename(old_name, new_name)
        print(f"Folder '{old_name}' renamed to '{new_name}' successfully.")
    except FileNotFoundError:
        print(f"Folder '{old_name}' not found.")
    except Exception as e:
        print(f"Error: {e}")