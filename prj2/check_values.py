import math


# Function to calculate RMSE
def calculate_rmse(output_data, test_data):
    # Parse the data into dictionaries where keys are tuples of the first and second columns
    output_dict = {
        (int(row.split(",")[0]), int(row.split(",")[1])): float(row.split(",")[2])
        for row in output_data
    }
    test_dict = {
        (int(row.split(",")[0]), int(row.split(",")[1])): float(row.split(",")[2])
        for row in test_data
    }

    # Calculate squared differences
    squared_diffs = []
    for key in output_dict:
        if key in test_dict:
            squared_diffs.append((output_dict[key] - test_dict[key]) ** 2)

    # Calculate mean of squared differences
    mean_squared_diff = sum(squared_diffs) / len(squared_diffs)

    # Calculate the square root of the mean squared difference (RMSE)
    rmse = math.sqrt(mean_squared_diff)
    return rmse


def main():
    # Read the output and test datasets
    with open("output.txt", "r") as f:
        output = f.read()
    with open("ratingstest_local.txt", "r") as f:
        test_dataset = f.read()

    # Split the string data into lists of rows
    output_rows = output.strip().split("\n")
    test_rows = test_dataset.strip().split("\n")

    # Calculate RMSE
    rmse = calculate_rmse(output_rows, test_rows)
    print(f"The RMSE of the third column is: {rmse}")


if __name__ == "__main__":
    main()
