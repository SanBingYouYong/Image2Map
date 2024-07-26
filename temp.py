import numpy as np
from scipy.ndimage import convolve

def smooth_tile_types(matrix, threshold):
    # Get the shape of the input matrix
    rows, cols = matrix.shape
    
    # Create an empty matrix to store the smoothed output
    smoothed_matrix = np.copy(matrix)
    
    # Iterate through each element in the matrix
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Get the current 3x3 region
            region = matrix[i-1:i+2, j-1:j+2]
            
            # Get the unique values and their counts in the region
            unique, counts = np.unique(region, return_counts=True)
            
            # Create a dictionary of value counts
            value_counts = dict(zip(unique, counts))
            
            # Exclude the center value itself from the count
            center_value = matrix[i, j]
            value_counts[center_value] -= 1
            
            # Get the maximum count of the surrounding neighbors
            max_count = max(value_counts.values())
            
            # If the maximum count exceeds the threshold, update the center value
            if max_count >= threshold:
                new_value = max(value_counts, key=value_counts.get)
                smoothed_matrix[i, j] = new_value
    
    return smoothed_matrix

# Example usage
matrix = np.array([[0, 0, 0, 0],
                   [0, 2, 2, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]])

threshold = 3
smoothed_matrix = smooth_tile_types(matrix, threshold)
print(smoothed_matrix)
