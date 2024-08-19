import zipfile
import pandas as pd
import os


def read_multiple_csv_from_zipped_csv(zip_file_path, output_df_path=None):
    """
    Reads multiple compressed CSV files from a single ZIP archive
    and combines them into a single DataFrame.

    Args:
        zip_file_path (str): Path to the ZIP file containing CSV files.
        output_df_path (str, optional): Path to save the combined DataFrame as a CSV file. Defaults to None.

    Returns:
        pandas.DataFrame: The combined DataFrame containing data from all CSV files,
            or None if no CSV files were found or errors occurred.
    """

    try:
        # Open the ZIP file with context manager
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            all_data = []

            # Iterate through zip members
            for zip_info in zip_ref.infolist():
                # Check if member is a CSV file
                if zip_info.filename.endswith('.csv'):
                    # Extract CSV content
                    csv_data = zip_ref.read(zip_info.filename)

                    try:
                        # Read CSV data using pandas
                        df = pd.read_csv(csv_data)
                        all_data.append(df)
                    except pd.errors.EmptyDataError:
                        print(f"Skipping empty CSV file: {zip_info.filename}")

            # Concatenate DataFrames and handle empty CSV cases
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
            else:
                # Informative message when no CSV files are found
                print("No CSV files found in the ZIP archive.")
                return None

            # Optionally save the combined DataFrame as a CSV file
            if output_df_path:
                combined_df.to_csv(output_df_path, index=False)

            return combined_df

    except (FileNotFoundError, IOError, zipfile.BadZipFile) as e:
        print(f"Error reading CSV files: {e}")
        return None


def read_multiple_csv_from_zipped_folders(folder_path, output_df_path=None):
    """
    Reads CSV files from multiple ZIP files within a folder without recursion.

    Args:
        folder_path (str): Path to the folder containing the ZIP files.
        output_df_path (str, optional): Path to save the combined DataFrame as a CSV file. Defaults to None.

    Returns:
        pandas.DataFrame: The combined DataFrame containing data from all CSV files,
            or None if errors occurred.
    """

    try:
        # Create a temporary ZIP file
        temp_zip_path = os.path.join(folder_path, "temp_archive.zip")
        with zipfile.ZipFile(temp_zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zip_file:
            for filename in os.listdir(folder_path):
                if filename.endswith('.zip'):
                    zip_file_path = os.path.join(folder_path, filename)
                    with zipfile.ZipFile(zip_file_path, 'r') as inner_zip:
                        for inner_filename in inner_zip.namelist():
                            if inner_filename.endswith('.csv'):
                                inner_data = inner_zip.read(inner_filename)
                                zip_file.writestr(inner_filename, inner_data)

        # Read CSV files from the temporary ZIP file using the corrected function
        combined_df = read_multiple_csv_from_zipped_csv(temp_zip_path, output_df_path)

        # Delete the temporary ZIP file
        os.remove(temp_zip_path)

        return combined_df

    except (FileNotFoundError, IOError, zipfile.BadZipFile) as e:
        print(f"Error reading CSV files: {e}")
        return None
