#!/bin/bash

# Define the tarball name
tarball_name="20170082_hw3.tar.gz"

# Ensure the tarball exists
if [ ! -f "$tarball_name" ]; then
    echo "Error: Tarball '$tarball_name' does not exist."
    exit 1
fi

# Create a temporary directory for extraction
temp_dir=$(mktemp -d)

cp "./graph.txt" "./training.csv" "./testing.csv" "./hw3_3_test.py" "$temp_dir/"

# Extract the tarball to the temporary directory
tar -xzf "$tarball_name" -C "$temp_dir"

# Verify the necessary files are present
required_files=("hw3_1.py" "hw3_2_p3.py" "hw3_3_p3.py")
for file in "${required_files[@]}"; do
    if [ ! -f "$temp_dir/$file" ]; then
        echo "Error: Required file '$file' is missing in the tarball."
        exit 1
    fi
done

# if test files exist, copy them to the temp directory and check diff of outputs
if [ -d "./tests" ]; then
    cp -r "./tests" "$temp_dir/"    
fi

# Change into the temporary directory
cd "$temp_dir"

# Function to run a command and check its output
run_and_check() {
    local cmd=$1
    local input_file=$2
    local output_file=$3
    local test_output_file="./tests/${output_file}"

    echo "Running '${cmd} ${input_file}'"
    ${cmd} ${input_file} > ${output_file}
    if [ $? -ne 0 ]; then
        echo "Runtime error in '${cmd} ${input_file}'"
        exit 1
    fi

    if [ -f "${test_output_file}" ]; then
        diff -q "${test_output_file}" "${output_file}"
        if [ $? -ne 0 ]; then
            echo "Output mismatch in '${cmd} ${input_file}'"
        fi
    fi
}

# Commands and parameters
run_and_check "spark-submit hw3_1.py" "graph.txt" "hw3_1.out"
run_and_check "python hw3_2_p3.py" "training.csv testing.csv" "hw3_2.out"
run_and_check "python hw3_3_p3.py" "graph.txt" "hw3_3.out"
run_and_check "python hw3_3_test.py" "hw3_3_test.out"

# Clean up: remove the temporary directory
rm -rf "$temp_dir"

echo "All scripts executed successfully, no runtime errors encountered."