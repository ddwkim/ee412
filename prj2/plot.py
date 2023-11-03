import matplotlib.pyplot as plt

file_path = "avgdist.txt"

# Read the average distances from the file
with open(file_path, "r") as file:
    lines = file.readlines()
    # Convert each line to a float and store it in a list
    distances = [float(line.strip()) for line in lines]

k_values = range(1, 1 + len(distances))

# Plotting
plt.figure(figsize=(10, 6))  # Set the figure size as desired
plt.plot(k_values, distances, marker="o")  # Plot with a circle marker at each point
plt.title("Average Distance of Clusters vs Number of Clusters (k)")  # Add a title
plt.xlabel("Number of Clusters (k)")  # Label the x-axis
plt.ylabel("Average Distance")  # Label the y-axis
plt.xticks(k_values)  # Ensure that only integer k values are used as x-ticks
plt.grid(True)  # Show grid for better readability
# Save the plot as a PNG file
plt.savefig("cluster_plot.png")
