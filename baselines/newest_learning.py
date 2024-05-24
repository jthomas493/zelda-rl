import os
import glob
import zipfile


def newest_learning(directory):
    """
    Traverses the current directory and its subdirectories to find the most recent valid ZIP file.

    Args:
        directory: The starting directory for the search.

    Returns:
        The full path to the most recently saved and validated ZIP file, or None if no ZIPs found.
    """
    # Initialize variables
    newest_zip = None
    newest_zip_time = None

    # Walk through the directory tree
    for root, dirs, files in os.walk(directory):
        # Construct the glob pattern for the current directory
        zip_pattern = os.path.join(root, "*.zip")

        # Get all ZIP file paths using glob
        zip_files = glob.glob(zip_pattern)

        for zip_file in zip_files:
            # Get the modification time of the current ZIP file
            current_time = os.path.getmtime(zip_file)

            # Update the newest ZIP if the current one is more recent
            if not newest_zip or current_time > newest_zip_time:
                try:
                    # Validate the ZIP file before considering it the newest
                    with zipfile.ZipFile(zip_file, "r") as zip_ref:
                        pass  # Just try opening the ZIP file
                    newest_zip = zip_file
                    newest_zip_time = current_time
                except zipfile.BadZipFile:
                    print(f"Found a file named '{zip_file}' but it's not a valid ZIP.")

    # Return the validated newest ZIP path if found
    return newest_zip
