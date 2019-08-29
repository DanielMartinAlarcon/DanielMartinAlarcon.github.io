---
title: The GeoPandas Cookbook
subtitle: Simple recipes for beautiful data maps
image: /img/12_geopandas/geo1.png
---

For anyone used to data science with pandas, GeoPandas is the simplest way to perform geospatial operations and (most importantly) visualize your geographic data. GeoPandas saves you from needing to use specialized spatial databases such as PostGIS. This cookbook contains the recipes that I've found myself using over and over again with GeoPandas, many of them cobbled together from bits scattered throughout Stack Overflow and the official documentation.

When you start out, make sure to scan the [official GeoPandas documentation](http://geopandas.org/index.html) and this great [course from the University of Helsinki](https://automating-gis-processes.github.io/2017/index.html). Happy mapping!

# [Under construction]
I'm still working on the content below, please come back later.

# Make a GeoDataFrame from coordinates
If your pandas DataFrame contains columns for latitude and longitude, you can turn them into a list of `Shapely.point` objects and use that as the `geometry` column of a new Geopandas `GeoDataFrame`.

```python
def make_geodf(df, lat_col_name='latitude', lon_col_name='longitude'):
    """
    Take a dataframe with latitude and longitude columns, and turn
    it into a geopandas df.
    """
    from geopandas import GeoDataFrame
    from geopandas import points_from_xy
    df = df.copy()
    lat = df['latitude']
    lon = df['longitude']
    return GeoDataFrame(df, geometry=points_from_xy(lon, lat))
```
If you have a set of points that you want to turn into a polygon, use the Shapely library. The Polygon constructor can take an array of lat/lon tuples and produce a Polygon. A list of Polygons, much like a list of Points, can be used to make the geometry column of a GeoDataFrame.

```python
from geopandas import GeoDataFrame
from shapely.geometry import Polygon

# Turns array of lat/lon tuples into a single Polygon
poly = Polygon([array_of_lat_lon_tuples])

# Turns a DataFrame into a GeoDataFrame, using an array of Polygons as a geometry column.
geodf = GeoDataFrame(df2, geometry=array_of_polygons)
```

# Import pre-made maps
GeoPandas comes with three pre-made maps that are already on your computer if you installed the package. Once loaded as GeoDataFrames, these maps contain the Points for several major cities or low-resolution Polygons for the borders of all countries. Make sure to check how your particular country is spelled.
The relevant commands are: 

```python
# Returns names of available maps
geopandas.datasets.available
# Returns path of a particular map
geopandas.datasets.get_path('naturalearth_lowres')
# Opens the map as a GeoDataFrame
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
# Subsets the GeoDataFrame
usa = world[world.name == "United States of America"]
```

![geopandas](/img/12_geopandas/geo2.png)

For most tasks, though, you'll want to upload a map of your own. If it is defined as a GEOJSON file or a SHP file, GeoPandas can read them directly. Feed either the path to the GeoJSON file itself or the path to the *folder* of SHP files.

```python
geodf = geopandas.read_file('PATH/TO/custom_map.geojson/')
geodf = geopandas.read_file('PATH/TO/folder_with_SHP_files/')
```

# Basic plotting and Chloropleths
A GeoDataFrame can contain points or polygons. Both will be rendered automatically with `geodf.plot()`.  If you plot a GeoDataFrame with Polygons in the geometry column, you can use the parameter `column` to assign each polygon a color based on the (numerical) value of that column. Here I also used a trick from the [AxesGrid toolkit](https://matplotlib.org/mpl_toolkits/axes_grid/users/overview.html) to add a custom colorbar.

The GeoDataFrame plotted below:

![geopandas](/img/12_geopandas/geo3.png)


```python
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import StrMethodFormatter

fig, ax = plt.subplots(figsize=(8,8))

# Sets axes limits in lat/lon
ax.set_xlim(-123.5,-121)
ax.set_ylim(37,38.5)

# Removes ticks and labels for lat/lon
ax.tick_params(
    axis='both', bottom=False, left=False,         
    labelbottom=False, labelleft=False) 

# make_axes_locatable returns an instance of the AxesLocator class, 
# derived from the Locator. It provides append_axes method that 
# creates a new axes on the given side of (“top”, “right”, 
#“bottom” and “left”) of the original axes.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.1, label='Title')


# Plot the GeoDataFrame as a chloropleth
quakes.plot(ax=ax, 
            column='pga', # Column that determines color
            legend=True,  
            cax=cax,      # Add a colorbar
            cmap='OrRd',  # Colormap
            edgecolor='black')

# Format colorbar tick labels
cax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

# Set the fontsize for each colorbar tick label
for l in cax.yaxis.get_ticklabels():
    l.set_fontsize(14)

ax.set_title('Earthquake Risk in the SF Bay Area', fontsize=20, pad=10);
```

![geopandas](/img/12_geopandas/geo4.png)

# Stacking polygon layers

<!-- ![geopandas](/img/12_geopandas/geo1.png) -->


# Bubble plots



```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

```python
```

