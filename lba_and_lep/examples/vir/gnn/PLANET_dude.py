import os
import gzip
import shutil

# Define the main directory where all the PDBBind directories are located
main_directory = './all'

# Loop through all the directories and subdirectories
for root, dirs, files in os.walk(main_directory):
    # If the file is already unzipped, skip it
    if 'actives_final.mol2' in files and 'decoys_final.mol2' in files:
        continue
    for file in files:
        if file in ['actives_final.mol2.gz', 'decoys_final.mol2.gz']:
            file_path = os.path.join(root, file)
            output_file_path = file_path[:-3] 
            with gzip.open(file_path, 'rb') as f_in:
                with open(output_file_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f'Unzipped: {file_path} to {output_file_path}')
