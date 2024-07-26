# Image2Map
Convert any RGB images (well, RGBA works too since I discard the alpha channel anyways) to a tile map and download them as csv. Ideally for downstream tasks such as becoming maps for a tile-based game. 

Comes with easy-to-use web ui through streamlit. 

## Quick Start
### Installation
1. (Optional) consider creating a virtual environment using your favouritae package manager.
2. Install dependencies with `pip install -r requirements.txt`.
### Usage
3. Open web ui by running `streamlit run ./i2m_ui.py` in the project directory.
4. Upload an image to be processed as a tile map. 
![upload_image](md_upload.png)
5. Adjust settings as needed.
![interface](md_interface.png)
6. Click `Generate Map` and wait for `Tile Map Visualization` and `Tile Map Download` sections to show up.
![visualization](md_vis.png)
![download](md_download.png)
7. Download the zip file of two csv files or separate csv files directly as needed. They are both in "utf-8" format. 
![csv](md_csv.png)

## API Usage
Simply copy paste i2m.py and import the `image_to_map` method. Check docstring for details. 
