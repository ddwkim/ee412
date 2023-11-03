#!/bin/bash

# Define the tarball name
tarball_name="20170082_hw2.tar.gz"

# Ensure the tarball exists
if [ ! -f "$tarball_name" ]; then
    echo "Error: Tarball '$tarball_name' does not exist."
    exit 1
fi

# Create a temporary directory for extraction
temp_dir=$(mktemp -d)

# Extract the tarball to the temporary directory
tar -xzf "$tarball_name" -C "$temp_dir"

# Verify the necessary files are present
required_files=("hw2_1.py" "hw2_3b.py" "hw2_3c.py" "kmeans.txt" "ratings.txt" "ratings_test.txt")
for file in "${required_files[@]}"; do
    if [ ! -f "$temp_dir/$file" ]; then
        echo "Error: Required file '$file' is missing in the tarball."
        exit 1
    fi
done

# Change into the temporary directory
cd "$temp_dir"

# Assuming 'spark-submit' is in the PATH and python environment is set up
# Execute the Python scripts with the required arguments
echo "Running 'spark-submit hw2_1.py kmeans.txt 1'"
spark-submit hw2_1.py kmeans.txt 1
if [ $? -ne 0 ]; then
    echo "Runtime error in 'spark-submit hw2_1.py kmeans.txt 1'"
    exit 1
fi

echo "Running 'python hw2_3b.py ratings.txt'"
python hw2_3b.py ratings.txt
if [ $? -ne 0 ]; then
    echo "Runtime error in 'python hw2_3b.py ratings.txt'"
    exit 1
fi

echo "Running 'python hw2_3c.py ratings.txt ratings_test.txt'"
python hw2_3c.py ratings.txt ratings_test.txt
if [ $? -ne 0 ]; then
    echo "Runtime error in 'python hw2_3c.py ratings.txt ratings_test.txt'"
    exit 1
fi

# Clean up: remove the temporary directory
rm -rf "$temp_dir"

echo "All scripts executed successfully, no runtime errors encountered."
