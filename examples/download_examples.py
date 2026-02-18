import os
import requests
import zipfile
import io
import shutil

# URL for the examples (direct download link)
DROPBOX_URL = "https://www.dropbox.com/scl/fo/a3tzd2m52jgb1jgz8u36d/AFsKKz4JZd2_Nr12N_sLPmk?rlkey=xw47pluzbl5i1kfhejw420jc0&st=ljj57avw&dl=1"

def download_and_extract_examples(url, target_dir):
    """
    Downloads a zip file from the given URL and extracts it to the target directory,
    stripping the top-level folder if it exists.
    """
    print(f"Downloading examples from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()

        print("Download complete. Extracting files...")
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            all_members = z.namelist()
            if not all_members:
                print("Zip file is empty.")
                return
            
            # Filter out the root directory '/' if present and any system files like __MACOSX
            valid_members = [m for m in all_members if m != '/' and not m.startswith('__MACOSX')]

            if not valid_members:
                 print("No valid files found in zip.")
                 return

            # Determine if there's a common prefix (top-level directory)
            common_prefix = os.path.commonprefix(valid_members)
            
            # If the prefix ends with '/', it's a directory
            if common_prefix and not common_prefix.endswith('/'):
                 common_prefix = os.path.dirname(common_prefix)
                 if common_prefix:
                     common_prefix += '/'

            # Calculate prefix length once
            prefix_len = len(common_prefix)

            # Extract files, stripping the common prefix
            for member in valid_members:
                if member.endswith('/'): # Skip directories
                    continue
                
                # new_name is the member name without the common prefix
                if common_prefix and member.startswith(common_prefix):
                    new_name = member[prefix_len:]
                else:
                    new_name = member
                
                # Secure target path
                target_path = os.path.normpath(os.path.join(target_dir, new_name))
                
                # Ensure target directory exists
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                
                # Write file
                with z.open(member) as source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)
            
        print(f"Examples extracted to {target_dir}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading examples: {e}")
    except zipfile.BadZipFile:
        print("Error: The downloaded file is not a valid zip file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    download_and_extract_examples(DROPBOX_URL, script_dir)
