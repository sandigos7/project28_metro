import pandas as pd
import geopandas as gpd
import networkx as nx
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
import os
from pathlib import Path
import zipfile

# -----------------------
# 1. Define Paths
# -----------------------
# Get the script directory and construct paths relative to project root
script_dir = Path(__file__).parent
project_root = script_dir.parent
gtfs_bus_zip = project_root / "data" / "gtfs_bus" / "gtfs_bus.zip"
gtfs_rail_zip = project_root / "data" / "gtfs_rail" / "gtfs_rail.zip"

# -----------------------
# 2. Load GTFS Files from ZIP archives
# -----------------------
# Check if ZIP files exist
if not gtfs_bus_zip.exists():
    raise FileNotFoundError(f"Bus GTFS file not found: {gtfs_bus_zip}")
if not gtfs_rail_zip.exists():
    raise FileNotFoundError(f"Rail GTFS file not found: {gtfs_rail_zip}")

# Read CSV files from ZIP archives
def read_csv_from_zip(zip_path, csv_file):
    """Read a CSV file from a ZIP archive."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        if csv_file not in z.namelist():
            raise FileNotFoundError(f"File {csv_file} not found in {zip_path}")
        with z.open(csv_file) as f:
            return pd.read_csv(f, low_memory=False)

# Load bus GTFS data
bus_stops = read_csv_from_zip(gtfs_bus_zip, "stops.txt")
bus_trips = read_csv_from_zip(gtfs_bus_zip, "trips.txt")
bus_stop_times = read_csv_from_zip(gtfs_bus_zip, "stop_times.txt")
bus_routes = read_csv_from_zip(gtfs_bus_zip, "routes.txt")

# Load rail GTFS data
rail_stops = read_csv_from_zip(gtfs_rail_zip, "stops.txt")
rail_trips = read_csv_from_zip(gtfs_rail_zip, "trips.txt")
rail_stop_times = read_csv_from_zip(gtfs_rail_zip, "stop_times.txt")
rail_routes = read_csv_from_zip(gtfs_rail_zip, "routes.txt")

# -----------------------
# 3. Combine Stops
# -----------------------
# Check required columns exist
required_cols = ['stop_id', 'stop_name', 'stop_lat', 'stop_lon']
for col in required_cols:
    if col not in bus_stops.columns:
        raise ValueError(f"Required column '{col}' not found in bus stops")
    if col not in rail_stops.columns:
        raise ValueError(f"Required column '{col}' not found in rail stops")

# Combine bus and rail stops, add mode identifier
bus_stops['mode'] = 'bus'
rail_stops['mode'] = 'rail'
stops_all = pd.concat([bus_stops, rail_stops], ignore_index=True)

# Create GeoDataFrame with point geometry
stops_gdf = gpd.GeoDataFrame(stops_all,
                             geometry=gpd.points_from_xy(stops_all.stop_lon, stops_all.stop_lat),
                             crs="EPSG:4326")

# -----------------------
# 4. Build Transit Network
# -----------------------
G = nx.DiGraph()

# Add stops as nodes
nodes_data = stops_all[['stop_id', 'stop_name', 'stop_lat', 'stop_lon', 'mode']].to_dict('records')
for node_data in nodes_data:
    G.add_node(node_data['stop_id'], 
               name=node_data['stop_name'], 
               lat=node_data['stop_lat'], 
               lon=node_data['stop_lon'], 
               mode=node_data['mode'])

# Add edges between consecutive stops for each trip
def add_edges(trips_df, stop_times_df, mode_name):
    """Add edges from trips and stop_times dataframes."""
    # Filter and sort stop_times by trip_id and stop_sequence
    valid_trip_ids = set(trips_df['trip_id'].unique())
    stop_times_sorted = stop_times_df[stop_times_df['trip_id'].isin(valid_trip_ids)].sort_values(['trip_id', 'stop_sequence'])
    
    # Group by trip_id and add edges between consecutive stops
    for trip_id, group in stop_times_sorted.groupby('trip_id'):
        if len(group) < 2:
            continue
        stop_ids = group['stop_id'].values
        for j in range(len(stop_ids) - 1):
            G.add_edge(stop_ids[j], stop_ids[j + 1], trip_id=trip_id, mode=mode_name)

# Build network edges
add_edges(bus_trips, bus_stop_times, 'bus')
add_edges(rail_trips, rail_stop_times, 'rail')

# -----------------------
# 5. Plot Stops
# -----------------------
# Create visualization map
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

fig, ax = plt.subplots(figsize=(12, 12))
bus_stops_gdf = stops_gdf[stops_gdf['mode'] == 'bus']
rail_stops_gdf = stops_gdf[stops_gdf['mode'] == 'rail']

if len(bus_stops_gdf) > 0:
    bus_stops_gdf.plot(ax=ax, color='blue', markersize=5, alpha=0.6, label='Bus Stops')
if len(rail_stops_gdf) > 0:
    rail_stops_gdf.plot(ax=ax, color='red', markersize=8, alpha=0.8, label='Rail Stops')

ax.set_title("LA Metro Bus + Rail Stops", fontsize=16, fontweight='bold')
ax.set_xlabel("Longitude", fontsize=12)
ax.set_ylabel("Latitude", fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# Save plot
output_dir = project_root / "outputs"
output_dir.mkdir(exist_ok=True)
plot_path = output_dir / "metro_stops_map.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()

