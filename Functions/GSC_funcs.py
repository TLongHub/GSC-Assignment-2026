import pystac_client
import planetary_computer
import odc.stac
import rioxarray as rxr
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely import box
import numpy as np
from pystac import ItemCollection
from datetime import datetime, timedelta


def connect_MPC():
    # Connect to the MPC
    stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
    
    catalog = pystac_client.Client.open(
        stac_url,
        modifier=planetary_computer.sign_inplace,  # only needed for MPC
    
        )
    return catalog

def search_Landsat(bbox, daterange, max_cloud_percent, catalog):
    # Search for cloud-free Landsat 8/9 Surface Reflectance (c2l2-sr) imagery in Summer
    collection = "landsat-c2-l2"
    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=daterange,
        query={
            "eo:cloud_cover": {"lt": max_cloud_percent},  # Cloud cover less than 15%
            "platform": {"in": ["landsat-5", "landsat-8", "landsat-9"]},  # Return only Landsat 5, 8, and 9. 
            }, 
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}]  # Sort by cloud cover in ascending order
    )
    
    # Convert the search result object into a list of `pystac.Item` objects
    items = search.item_collection()
    # print(f"Found {len(items)} matching scene(s).")

    return items

def filter_items(bbox, items, aoi_fraction_threshold):
    # convert aoi to a geopandas dataframe
    aoi_gdf = gpd.GeoDataFrame(geometry=[box(*bbox)], crs="EPSG:4326")
    
    # convert search results to a geopandas dataframe
    search_gdf = gpd.GeoDataFrame.from_features(items.to_dict(), crs="EPSG:4326")
    
    # convert both to an equal area projection (EPSG:8857) so we can work in metres
    aoi_gdf = aoi_gdf.to_crs(8857)
    search_gdf = search_gdf.to_crs(8857)
    
    # clip search gdf to aoi gdf
    search_gdf = search_gdf.clip(aoi_gdf)
    
    # Calculate the fraction of the AOI covered by each scene
    search_gdf = search_gdf.sort_index()
    search_gdf["frac"] = search_gdf.area / aoi_gdf.area.values[0]
    
    # Include a boolean column indicating whether the scene exceeds the AOI fraction threshold
    search_gdf["exceed_frac"] = np.where(search_gdf["frac"].values >= aoi_fraction_threshold, True, False)
    
    # Reconstruct the ItemCollection with only the items that exceed the AOI fraction threshold
    items_list = []
    for i, exceed_frac in zip(items, search_gdf.exceed_frac.values):
        if exceed_frac == True:
            items_list.append(i)
    
    items = ItemCollection(items=items_list)
    # print(f"Afer AOI filtering, returns {len(items)} Items")

    return items

def item_details(item):
    date = item.properties['datetime'][:10]
    epsg = item.properties['proj:code']
    satellite = item.properties['platform']
    
    print(f"Selected Scene ID: {item.id}")
    print(f"Acquisition Date: {item.properties['datetime']}")
    print(f"Cloud Cover: {item.properties['eo:cloud_cover']}%")
    print(f"Projection: {item.properties['proj:code']}")

    return date, epsg, satellite

def load_scene(item, bands, bbox):
    ds = odc.stac.load(
        [item],
        bands=bands,
        bbox=bbox,
        # crs="EPSG:XXXX",  # Uncomment if you want a different projection: by default, will use the native UTM zone CRS
        # resolution=100, # Uncomment if you want a different resolution: by default, will use the native resolution of the imagery
    ).squeeze()
    
    # xarray is 'lazy' by default, meaning it doesn't (down)load the data into memory until you explicitly ask for it.
    # The `.compute()` method forces the data to be loaded into memory, which can take some time depending on the size of the dataset.
    ds.compute()
    
    # stack variables into a band dimension
    ds = ds.to_array(dim="band").squeeze()
    
    # add the band names as a long_name attribute to the dataset
    ds.attrs["long_name"] = list(ds["band"].values)
    
    # set zero to NaN
    ds = ds.where(ds > 0)
    
    # Surface reflectance scaling values
    ds = ds * 0.0000275 - 0.2  # convert to SR
    
    return ds

def plot_image(ds):
    rgb = ds.sel(band=["red", "green", "blue"])
    
    fig, ax = plt.subplots()
    rgb.plot.imshow(ax=ax, robust=True)
    ax.set_aspect("equal")

def ndbi(ds):
    ndbi = (ds.sel(band="blue") - ds.sel(band="red")) / (ds.sel(band="blue") + ds.sel(band="red"))
    #ndbi.plot.imshow(cmap="BrBG", robust=True)

    return ndbi


def ndvi(ds):
    """Normalised difference vegetation index."""
    ndvi = (ds.sel(band="nir08") - ds.sel(band="red")) / (ds.sel(band="nir08") + ds.sel(band="red"))
    #ndvi.plot.imshow(cmap="BrBG", robust=True)

    return ndvi

def nbr(ds):
    """Normalised burn ratio."""
    nbr = (ds.sel(band="nir08") - ds.sel(band="swir16")) / (ds.sel(band="nir08") + ds.sel(band="swir16"))
    #nbr.plot.imshow(cmap="BrBG", robust=True)

    return nbr

def date_ranges(start_date, end_date, period):
    """Creates a list of date ranges from start to approx end with each range being 'period' long. Period is in days."""
    date_ranges = []
    
    current_date = start_date
    
    while current_date <= end_date:
        next_date = current_date + timedelta(days=period)
    
        date_ranges.append(
            f"{(current_date + timedelta(days=1)).strftime('%Y-%m-%d')}/{next_date.strftime('%Y-%m-%d')}"
        )
    
        current_date = next_date

    return date_ranges


def fetch_sat_data(date_ranges, aoi_bb, catalog, bands, max_cloud_percent, aoi_percentage_threshold):
    """Returns a list of geo-DataArrays of the area of interest."""
    ds_list = [] # Initiate empty list
    for rng in date_ranges: # Loop over date ranges
        items = search_Landsat(aoi_bb, rng, max_cloud_percent, catalog) # Obtain items from landsat
        if len(items) != 0:
            filtered_items = filter_items(aoi_bb, items, aoi_percentage_threshold/100) # Filter to disragard any that don't capture enough of aoi
            if len(filtered_items) != 0:
                ds = load_scene(filtered_items[0], bands, aoi_bb) # Load the scene
                ds_list.append(ds) # Save the scene to list
    return ds_list