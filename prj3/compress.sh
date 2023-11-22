#!/bin/bash

# Define an array with the file names to be included in the tarball
files=(
    "hw3_1.py"
    "hw3_2_p3.py"
    "hw3_3_p3.py"
    "hw3.pdf"
    "Ethics Oath.pdf"
)

# Define the tarball name
tarball_name="20170082_hw3.tar.gz"

if [ -f "$tarball_name" ]; then
    rm "$tarball_name"
fi

# Create the tarball with gzip compression
tar -czvf "$tarball_name" "${files[@]}"

echo "Tarball $tarball_name has been created successfully."