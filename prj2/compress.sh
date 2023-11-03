#!/bin/bash

# Define an array with the file names to be included in the tarball
files=(
    "hw2_1.py"
    "hw2_3b.py"
    "hw2.pdf"
    "hw2_3c.py"
    "output.txt"
    "Ethics Oath.pdf"
)

# Define the tarball name
tarball_name="20170082_hw2.tar.gz"

# Create the tarball with gzip compression
tar -czvf "$tarball_name" "${files[@]}"

echo "Tarball $tarball_name has been created successfully."
