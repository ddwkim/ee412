#!/bin/bash

# Define the tarball name
tarball_name="20170082_hw2.tar.gz"

# Ensure the tarball exists
if [ ! -f "$tarball_name" ]; then
    echo "Error: Tarball '$tarball_name' does not exist."
    exit 1
fi

# Extract the list of files from the tarball without extracting the files themselves
# Here IFS (Internal Field Separator) is set to newline to handle filenames with spaces
OLD_IFS=$IFS
IFS=$'\n'
tarball_contents=($(tar -tzf "$tarball_name"))
IFS=$OLD_IFS

# Flag to track the status
all_match=true

# Loop through each file listed in the tarball
for file in "${tarball_contents[@]}"; do
    # Check if the file exists in the current directory
    if [[ -f "$file" ]]; then
        # Compare the file from the tarball to the file in the directory
        if ! diff <(tar -Oxzf "$tarball_name" "$file") "$file" >/dev/null; then
            echo "File '$file' is different."
            all_match=false
        else
            echo "File '$file' matches."
        fi
    else
        echo "File '$file' does not exist in the current directory."
        all_match=false
    fi
done

# Final output based on the comparison results
if $all_match; then
    echo "All files in the tarball match the current files."
else
    echo "Some files in the tarball are different or missing in the current directory."
fi
