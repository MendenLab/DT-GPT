

## General usage

Please note, you will need to adjust all paths to work with your local system.

The order of the data processing is as follows:

```
# Generate the CSVs using the "MIMIC-IV data pipeline" pipeline
./1_preprocessing/2024_01_31_script_version.py

# Make new folder
mkdir ./2_data_setup/1_raw_events

# Copy over the raw files
cp <PATH TO YOUR PIPELINE>/data/output/* ./2_data_setup/1_raw_events

# Make final data folders
mkdir ./0_final_data
mkdir ./0_final_data/events

# Run further preprocessing
./1_preprocessing/2024_03_15_runner.py

# Run post-processing
./2_data_setup/2024_03_14_set_constant_splits.py
./2_data_setup/2024_03_15_post_process_for_meta_data.py
```




