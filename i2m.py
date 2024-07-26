import numpy as np
from skimage.color import rgb2lab, lab2rgb, deltaE_cie76
from sklearn.cluster import KMeans
from PIL import Image
from sklearn.metrics.pairwise import euclidean_distances
import zipfile
import io


def color_lab_distance(color1, color2):
    '''
    Accepts two RGB colors and returns a measure of similarity between them, 
    converting to LAB color space first.
    '''
    # Convert RGB to LAB
    color1_lab = rgb2lab(np.uint8([[color1]]))[0][0]
    color2_lab = rgb2lab(np.uint8([[color2]]))[0][0]
    
    # Calculate the distance using deltaE_cie76
    delta = deltaE_cie76(color1_lab, color2_lab)
    return delta

def partition_and_average_color(image, rows, cols):
    '''
    Accepts an image and the number of rows and columns to partition it into.
    Returns a list of the average color of each partition.
    '''
    # Partition the image into tiles
    width, height = image.size
    tile_width = width // cols
    tile_height = height // rows
    
    # Create a 2D numpy array to store average colors
    average_colors = np.zeros((rows, cols, 3))
    
    for i in range(rows):
        for j in range(cols):
            left = j * tile_width
            upper = i * tile_height
            right = (j + 1) * tile_width
            lower = (i + 1) * tile_height
            tile = image.crop((left, upper, right, lower))
            # Convert tile to LAB color space
            tile_lab = rgb2lab(np.array(tile))
            # Compute the average color for this tile
            average_color = np.mean(tile_lab, axis=(0, 1))
            average_colors[i, j] = average_color

    return average_colors

def image_from_average_colors(average_colors, size_pixels=50):
    '''
    Accepts a 2D numpy array of average colors and the size of each tile in pixels.
    Returns a new image representing the average colors.
    '''
    # Determine the size of the output image
    rows, cols, _ = average_colors.shape
    width = cols * size_pixels
    height = rows * size_pixels
    
    # Create a new image
    image = Image.new('RGB', (width, height))
    
    for i in range(rows):
        for j in range(cols):
            # Get the average color for this tile
            average_color = average_colors[i, j]
            average_color = lab2rgb(average_color) * 255
            # Create a tile image with the average color
            tile_image = Image.new('RGB', (size_pixels, size_pixels), tuple(average_color.astype(int)))
            # Paste the tile into the output image
            image.paste(tile_image, (j * size_pixels, i * size_pixels))

    return image

def sample_tile_colors(number_of_colors):
    '''
    Generate a list of sample tile colors by sampling the LAB space.
    '''
    # Generate random colors in LAB space
    lab_colors = np.random.rand(number_of_colors, 3) * 100
    # Convert to RGB
    rgb_colors = lab2rgb(lab_colors)
    # convert to 0-255 range and tuples
    rgb_colors = (rgb_colors * 255).astype(int)
    rgb_colors = [tuple(color) for color in rgb_colors]
    return rgb_colors

def image_from_tile_classes(tile_classes, tile_colors, size_pixels=50):
    '''
    Accepts a 2D numpy array of tile classes, a list of tile colors, and the size of each tile in pixels.
    Returns a new image representing the tile classes with the corresponding colors.
    '''
    # Determine the size of the output image
    rows, cols = tile_classes.shape
    width = cols * size_pixels
    height = rows * size_pixels
    
    # Create a new image
    image = Image.new('RGB', (width, height))
    
    for i in range(rows):
        for j in range(cols):
            # Get the class of this tile
            tile_class = tile_classes[i, j]
            # Get the color for this class
            color = tile_colors[tile_class]
            # Create a tile image with the color
            tile_image = Image.new('RGB', (size_pixels, size_pixels), color)
            # Paste the tile into the output image
            image.paste(tile_image, (j * size_pixels, i * size_pixels))

    return image

def tiles_color_knn(tiles_lab_colors: np.ndarray, k=3) -> np.ndarray:
    '''
    Accepts a numpy array (shape: rows*cols*3) of tile colors in LAB space and the number of classes they should be 
    classified into:
        - Infer the representing color for each class by running a k-means clustering algorithm.
        - Classify each tile according to the color distance to the closest representing color.
    Return a 2D numpy array of tile classes.
    '''
    # Step 1: Run k-means clustering
    tlc_rows, tlc_cols, _ = tiles_lab_colors.shape
    flattened_tiles_lab_colors = tiles_lab_colors.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(flattened_tiles_lab_colors)
    
    # The cluster centers (representative colors)
    cluster_centers = kmeans.cluster_centers_

    # Step 2: Compute distances from each tile color to the cluster centers
    distances = euclidean_distances(flattened_tiles_lab_colors, cluster_centers)
    
    # Step 3: Assign each tile to the closest cluster
    tile_classes = np.argmin(distances, axis=1)

    # Reshape the tile classes to match the original shape
    tile_classes = tile_classes.reshape(tlc_rows, tlc_cols)
    
    return tile_classes

def smooth_tile_types(matrix: np.ndarray, threshold: int) -> np.ndarray:
    '''
    GPT Smoothing lol.
    if in neighbors exist a color that occurs more than the threshold value times, change the center color to that color.
    '''
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

def format_to_csv(tiles) -> bytes:
    '''
    Accepts a 2D numpy array of tile classes and returns a string BINARY in CSV format.
    '''
    return '\n'.join([','.join(map(str, row)) for row in tiles]).encode('utf-8')

def create_zip_for_download(tiles, smoothed_tiles) -> io.BytesIO: 
    '''
    Creates an in-memory zip file for download from streamlit. 
    '''
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for file_name, data in [
            ("tiles.csv", io.BytesIO(tiles)), 
            ("smoothed_tiles.csv", io.BytesIO(smoothed_tiles))
        ]:
            zip_file.writestr(file_name, data.getvalue())
    return zip_buffer

# the main function
def image_to_map(image_path: str="./sample_image.jpg", 
                 num_rows: int=10, num_cols: int=10, 
                 num_tile_types: int=3, smoothing_threshold: int=5) -> np.ndarray:
    '''
    Accepts an image path and the number of rows and columns to partition it into, 
    the number of tile types, and the smoothing threshold.
    Returns the tile map in numpy 2D array

    Set smoothing_threshold to -1 for no smoothing and returning raw tile map. 
        Smoothing: if in neighbors exist a color that occurs more than the threshold value times,
        change the center color to that color.

    Parameters:
    - image_path: str, path to the image file
    - num_rows: int, number of rows to partition the image into
    - num_cols: int, number of columns to partition the image into
    - num_tile_types: int, number of tile types to classify the tiles into
    - smoothing_threshold: int, threshold for smoothing the tile types (set to -1 for no smoothing), recommended 3-5
    '''
    # Reads image
    image = Image.open(image_path).convert("RGB")

    # Partition the image into tiles
    tiles_average_colors = partition_and_average_color(image, num_rows, num_cols)
    
    # Classify the tiles into different types
    tiles = tiles_color_knn(tiles_average_colors, num_tile_types)
    if smoothing_threshold == -1:
        return tiles

    # Optional smoothing
    smoothed_tiles = smooth_tile_types(tiles, smoothing_threshold)
    return smoothed_tiles


if __name__ == "__main__":
    print(
        image_to_map()
    )
