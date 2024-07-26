import streamlit as st
from PIL import Image, ImageDraw
from i2m import partition_and_average_color, image_from_average_colors, tiles_color_knn, sample_tile_colors, image_from_tile_classes, smooth_tile_types, format_to_csv, create_zip_for_download

st.set_page_config(
    page_title="Image2Map",
    page_icon="🖼️",
    layout="centered"
)

def draw_grid():
    if 'image' in st.session_state and st.session_state.image is not None:
        image = Image.open(st.session_state.image)
        rows = st.session_state.rows
        cols = st.session_state.cols

        draw = ImageDraw.Draw(image)
        width, height = image.size
        row_height = height / rows
        col_width = width / cols

        # Draw horizontal lines
        for i in range(1, rows):
            y = i * row_height
            draw.line([(0, y), (width, y)], fill="green", width=2)
        # Draw vertical lines
        for i in range(1, cols):
            x = i * col_width
            draw.line([(x, 0), (x, height)], fill="green", width=2)

        image_placeholder.image(image, caption="Image with grid", use_column_width='auto')


st.write("# Image2Map")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key='image')

image = None
if uploaded_image:
    image_placeholder = st.empty()
    image = Image.open(uploaded_image).convert("RGB")
    image_placeholder.image(image, caption="Uploaded image", use_column_width='auto', clamp=True, channels="RGB")

    size_col1, size_col2 = st.columns(2)
    map_rows = size_col1.number_input("Number of rows in the map", min_value=1, value=10, key="rows", on_change=draw_grid)
    map_cols = size_col2.number_input("Number of columns in the map", min_value=1, value=10, key="cols", on_change=draw_grid)

    draw_grid()

    tile_col1, tile_col2, tile_col3 = st.columns(3)
    tile_types_number = tile_col1.number_input("Number of tile types", min_value=1, value=3, key="tile_types")
    vis_tile_size = tile_col2.number_input("Visualization tile size", min_value=10, value=50, key="vis_tile_size")
    smoothing_threshold = tile_col3.number_input("Smoothing threshold", min_value=1, max_value=8, value=5, key="smoothing_threshold")
    if st.button("Generate Map"):
        tiles_average_colors = partition_and_average_color(image, map_rows, map_cols)
        tiles = tiles_color_knn(tiles_average_colors, tile_types_number)
        
        st.header("Tile Map Visualization")
        # display average color for each tile
        average_colors_image = image_from_average_colors(tiles_average_colors, size_pixels=vis_tile_size)
        st.image(average_colors_image, caption="Average colors of each tile", use_column_width='auto', clamp=True, channels="RGB")
        
        # raw tile map
        tile_colors = sample_tile_colors(tile_types_number)
        tile_classes_image = image_from_tile_classes(tiles, tile_colors, size_pixels=vis_tile_size)
        # optional smoothing
        smoothed_tiles = smooth_tile_types(tiles, smoothing_threshold)
        smoothed_tile_classes_image = image_from_tile_classes(smoothed_tiles, tile_colors, size_pixels=vis_tile_size)
        
        # Tile Map Visualization
        img_col1, img_col2 = st.columns(2)
        img_col1.image(tile_classes_image, caption="Tile classes", use_column_width='auto', clamp=True, channels="RGB")
        img_col2.image(smoothed_tile_classes_image, caption="Smoothed tile classes", use_column_width='auto', clamp=True, channels="RGB")
        
        # Tile Map Download
        st.header("Tile Map Download")
        csv_tiles = format_to_csv(tiles)
        csv_smoothed_tiles = format_to_csv(smoothed_tiles)
        st.download_button("Download Both (zip)", 
                           create_zip_for_download(csv_tiles, csv_smoothed_tiles),
                           f"tile_map_{uploaded_image.name}_{map_rows}x{map_cols}.zip")
        # separate
        download_col1, download_col2 = st.columns(2)
        download_col1.text("Raw tile map")
        download_col1.download_button("Download raw tile map (csv)", 
                                      csv_tiles,
                                      f"tile_map_{uploaded_image.name}_{map_rows}x{map_cols}.csv")
        download_col1.write(tiles)
        download_col2.text("Smoothed tile map")
        download_col2.download_button("Download smoothed tile map (csv)", 
                                      csv_smoothed_tiles,
                                      f"smoothed_tile_map_{uploaded_image}_{map_rows}x{map_cols}.csv")
        download_col2.write(smoothed_tiles)

