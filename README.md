# flight_data_tabular_converter
This repo contains a script which is used to convert Sample flight data from https://c3.ndc.nasa.gov/dashlink/projects/85/resources/ to tabular .parquet files.

# Instructions
The main.py contains a command line tool which will read the .mat files, interpolate each variables to the maximum sample rate found in all variables, allowing the dataset to become tabular, outputting each file to .parquet, which can be read using pandas or polars.

1. Copy the main.py to your working directory.
2. In command line navigate to your working directory and enter the command 'python main.py *mat_data_path* *output_data_folder*'.
3. *mat_data_path* is the path to a folder in the working directory containing the .mat files, this can be a directory structure as the functionality is recursive.
4. *output_data_folder* is a folder in your working directory where the .parquet files will be written too, this must be created before running the tool.
