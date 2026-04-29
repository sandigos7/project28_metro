"""
Enhanced Transit Accessibility Analysis for Los Angeles County
Service-aware, population-weighted, and equity-focused analysis.
Prepares baseline for Project 28 counterfactual analysis.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import Point
import warnings
import zipfile
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Set matplotlib backend
import matplotlib
matplotlib.use('Agg')

# Project paths
script_dir = Path(__file__).parent
project_root = script_dir.parent
output_dir = project_root / "outputs"
output_dir.mkdir(exist_ok=True)


# ============================================================================
# 1. DATA LOADING FUNCTIONS
# ============================================================================

def read_csv_from_zip(zip_path, csv_file):
    """Read a CSV file from a ZIP archive."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        if csv_file not in z.namelist():
            raise FileNotFoundError(f"File {csv_file} not found in {zip_path}")
        with z.open(csv_file) as f:
            return pd.read_csv(f, low_memory=False)


def load_census_block_groups():
    """Load Census Block Group shapefiles for Los Angeles County."""
    data_dir = project_root / "data" / "acs_2022_la_county"
    data_dir.mkdir(exist_ok=True)
    cbg_path = data_dir / "cbg_2022_06_037.shp"
    
    if not cbg_path.exists():
        print("Census Block Group shapefiles not found. Please run basic accessibility_analysis.py first.")
        raise FileNotFoundError(f"CBG shapefile not found: {cbg_path}")
    
    cbg = gpd.read_file(cbg_path)
    print(f"Loaded {len(cbg)} Census Block Groups")
    return cbg


def load_acs_attributes():
    """Load ACS 2022 attributes from CSV."""
    acs_path = project_root / "data" / "acs_2022_la_county.csv"
    acs = pd.read_csv(acs_path)
    
    # Rename columns for clarity
    acs = acs.rename(columns={
        'B01003_001E': 'total_pop',
        'B19013_001E': 'median_income',
        'B25046_001E': 'vehicles_available',
        'B08126_001E': 'transit_var1',
        'B08301_001E': 'transit_var2'
    })
    
    # Create GEOID
    if 'GEOID' not in acs.columns:
        acs['GEOID'] = acs['state'].astype(str).str.zfill(2) + \
                      acs['county'].astype(str).str.zfill(3) + \
                      acs['tract'].astype(str).str.zfill(6)
    
    return acs


def load_gtfs_data():
    """Load GTFS data for frequency calculations."""
    gtfs_bus_zip = project_root / "data" / "gtfs_bus" / "gtfs_bus.zip"
    gtfs_rail_zip = project_root / "data" / "gtfs_rail" / "gtfs_rail.zip"
    
    print("Loading GTFS data...")
    bus_stops = read_csv_from_zip(gtfs_bus_zip, "stops.txt")
    bus_trips = read_csv_from_zip(gtfs_bus_zip, "trips.txt")
    bus_stop_times = read_csv_from_zip(gtfs_bus_zip, "stop_times.txt")
    bus_calendar = read_csv_from_zip(gtfs_bus_zip, "calendar.txt")
    try:
        bus_calendar_dates = read_csv_from_zip(gtfs_bus_zip, "calendar_dates.txt")
    except:
        bus_calendar_dates = pd.DataFrame()
    
    rail_stops = read_csv_from_zip(gtfs_rail_zip, "stops.txt")
    rail_trips = read_csv_from_zip(gtfs_rail_zip, "trips.txt")
    rail_stop_times = read_csv_from_zip(gtfs_rail_zip, "stop_times.txt")
    rail_calendar = read_csv_from_zip(gtfs_rail_zip, "calendar.txt")
    try:
        rail_calendar_dates = read_csv_from_zip(gtfs_rail_zip, "calendar_dates.txt")
    except:
        rail_calendar_dates = pd.DataFrame()
    
    return {
        'bus': {
            'stops': bus_stops,
            'trips': bus_trips,
            'stop_times': bus_stop_times,
            'calendar': bus_calendar,
            'calendar_dates': bus_calendar_dates
        },
        'rail': {
            'stops': rail_stops,
            'trips': rail_trips,
            'stop_times': rail_stop_times,
            'calendar': rail_calendar,
            'calendar_dates': rail_calendar_dates
        }
    }


# ============================================================================
# 2. SERVICE FREQUENCY COMPUTATION
# ============================================================================

def compute_stop_frequency(gtfs_data, mode_name):
    """
    Compute weekday service frequency for each stop.
    Returns stops with trips_per_day and trips_per_hour.
    """
    print(f"Computing {mode_name} stop frequencies...")
    
    stops = gtfs_data['stops'].copy()
    trips = gtfs_data['trips'].copy()
    stop_times = gtfs_data['stop_times'].copy()
    calendar = gtfs_data['calendar'].copy()
    calendar_dates = gtfs_data['calendar_dates']
    
    # Identify weekday service
    # GTFS calendar: monday=1, tuesday=2, ..., sunday=0
    # For typical weekday, check monday=1
    weekday_service = calendar[calendar['monday'] == 1].copy()
    
    # Get service_ids that operate on weekdays
    if len(weekday_service) > 0:
        service_ids = weekday_service['service_id'].unique()
        weekday_trips = trips[trips['service_id'].isin(service_ids)].copy()
    else:
        # Fallback: assume all trips are weekday if calendar doesn't specify
        weekday_trips = trips.copy()
    
    # Merge stop_times with trips to get trip-level info
    stop_times_with_trips = stop_times.merge(
        weekday_trips[['trip_id', 'service_id']],
        on='trip_id',
        how='inner'
    )
    
    # Count trips per stop per day
    trips_per_stop = stop_times_with_trips.groupby('stop_id').size().reset_index(name='trips_per_day')
    
    # Ensure consistent data types for merging
    stops['stop_id'] = stops['stop_id'].astype(str)
    trips_per_stop['stop_id'] = trips_per_stop['stop_id'].astype(str)
    
    # Merge back to stops
    stops_with_freq = stops.merge(trips_per_stop, on='stop_id', how='left')
    stops_with_freq['trips_per_day'] = stops_with_freq['trips_per_day'].fillna(0).astype(int)
    
    # Convert to trips per hour (assuming 18-hour service day: 6 AM - 12 AM)
    service_hours = 18
    stops_with_freq['trips_per_hour'] = stops_with_freq['trips_per_day'] / service_hours
    stops_with_freq['trips_per_hour'] = stops_with_freq['trips_per_hour'].round(2)
    
    # Add mode identifier
    stops_with_freq['mode'] = mode_name
    
    print(f"  {mode_name}: {len(stops_with_freq)} stops, "
          f"mean {stops_with_freq['trips_per_hour'].mean():.1f} trips/hour")
    
    return stops_with_freq


def create_stops_with_frequency():
    """Create combined stops GeoDataFrame with frequency metrics."""
    gtfs_data = load_gtfs_data()
    
    # Compute frequencies for bus and rail
    bus_stops_freq = compute_stop_frequency(gtfs_data['bus'], 'bus')
    rail_stops_freq = compute_stop_frequency(gtfs_data['rail'], 'rail')
    
    # Combine
    stops_all = pd.concat([bus_stops_freq, rail_stops_freq], ignore_index=True)
    
    # Create GeoDataFrame
    stops_gdf = gpd.GeoDataFrame(
        stops_all,
        geometry=gpd.points_from_xy(stops_all.stop_lon, stops_all.stop_lat),
        crs="EPSG:4326"
    )
    
    # Ensure stop_id is string for grouping
    stops_gdf['stop_id'] = stops_gdf['stop_id'].astype(str)
    
    # For stops that appear in both bus and rail, aggregate frequencies
    stops_gdf = stops_gdf.groupby('stop_id').agg({
        'stop_name': 'first',
        'stop_lat': 'first',
        'stop_lon': 'first',
        'trips_per_day': 'sum',
        'trips_per_hour': 'sum',
        'mode': lambda x: '/'.join(sorted(set(str(v) for v in x))),
        'geometry': 'first'
    }).reset_index()
    
    # Recreate GeoDataFrame after aggregation
    stops_gdf = gpd.GeoDataFrame(stops_gdf, geometry='geometry', crs="EPSG:4326")
    
    print(f"Total stops with frequency: {len(stops_gdf)}")
    return stops_gdf


# ============================================================================
# 3. BASIC ACCESSIBILITY METRICS (for compatibility)
# ============================================================================

def compute_basic_accessibility_metrics(cbg_gdf, stops_gdf):
    """
    Compute basic distance-based accessibility metrics.
    Used when existing metrics file is not available.
    """
    print("Computing basic accessibility metrics...")
    
    cbg_proj = cbg_gdf.to_crs("EPSG:3310")
    stops_proj = stops_gdf.to_crs("EPSG:3310")
    
    centroids = gpd.GeoDataFrame(
        geometry=cbg_proj['centroid'],
        index=cbg_proj.index,
        crs=cbg_proj.crs
    )
    
    all_stops = stops_proj.copy()
    rail_stops = stops_proj[stops_proj['mode'].str.contains('rail', case=False, na=False)].copy()
    
    cbg_proj['dist_nearest_stop'] = np.nan
    cbg_proj['dist_nearest_rail'] = np.nan
    cbg_proj['stops_400m'] = 0
    cbg_proj['stops_800m'] = 0
    
    try:
        nearest_all = gpd.sjoin_nearest(
            centroids,
            all_stops,
            how='left',
            max_distance=50000,
            distance_col='distance'
        )
        dists_all = nearest_all.groupby(nearest_all.index)['distance'].min()
        cbg_proj.loc[dists_all.index, 'dist_nearest_stop'] = dists_all.values
        
        if len(rail_stops) > 0:
            nearest_rail = gpd.sjoin_nearest(
                centroids,
                rail_stops,
                how='left',
                max_distance=50000,
                distance_col='distance'
            )
            dists_rail = nearest_rail.groupby(nearest_rail.index)['distance'].min()
            cbg_proj.loc[dists_rail.index, 'dist_nearest_rail'] = dists_rail.values
    except (AttributeError, TypeError):
        from shapely.strtree import STRtree
        tree = STRtree(all_stops.geometry.values)
        all_stops_array = np.array(all_stops.geometry.values)
        for idx in centroids.index:
            centroid_geom = centroids.loc[idx, 'geometry']
            nearest_idx = tree.nearest(centroid_geom)
            cbg_proj.loc[idx, 'dist_nearest_stop'] = centroid_geom.distance(all_stops_array[nearest_idx])
        
        if len(rail_stops) > 0:
            tree_rail = STRtree(rail_stops.geometry.values)
            rail_stops_array = np.array(rail_stops.geometry.values)
            for idx in centroids.index:
                centroid_geom = centroids.loc[idx, 'geometry']
                nearest_idx = tree_rail.nearest(centroid_geom)
                cbg_proj.loc[idx, 'dist_nearest_rail'] = centroid_geom.distance(rail_stops_array[nearest_idx])
    
    # Count stops within buffers
    try:
        centroids_400 = centroids.copy()
        centroids_400.geometry = centroids_400.geometry.buffer(400)
        centroids_800 = centroids.copy()
        centroids_800.geometry = centroids_800.geometry.buffer(800)
        
        stops_in_400 = gpd.sjoin(centroids_400, all_stops, how='left', predicate='contains')
        stops_in_800 = gpd.sjoin(centroids_800, all_stops, how='left', predicate='contains')
        
        cbg_proj['stops_400m'] = stops_in_400.groupby(stops_in_400.index).size().fillna(0).astype(int)
        cbg_proj['stops_800m'] = stops_in_800.groupby(stops_in_800.index).size().fillna(0).astype(int)
    except Exception as e:
        print(f"  Warning: Could not compute buffer counts: {e}")
    
    return cbg_proj


# ============================================================================
# 4. FREQUENCY-AWARE ACCESSIBILITY METRICS
# ============================================================================

def compute_frequency_aware_accessibility(cbg_gdf, stops_gdf):
    """
    Compute accessibility metrics using frequency thresholds.
    For each CBG:
    - Distance to nearest stop with ≥2 trips/hour
    - Distance to nearest stop with ≥4 trips/hour
    - Count of frequent stops (≥2, ≥4 trips/hour) within 400m and 800m
    """
    print("Computing frequency-aware accessibility metrics...")
    
    # Project to EPSG:3310 for distance calculations
    cbg_proj = cbg_gdf.to_crs("EPSG:3310")
    stops_proj = stops_gdf.to_crs("EPSG:3310")
    
    # Get centroids
    centroids = gpd.GeoDataFrame(
        geometry=cbg_proj['centroid'],
        index=cbg_proj.index,
        crs=cbg_proj.crs
    )
    
    # Filter stops by frequency thresholds
    stops_2ph = stops_proj[stops_proj['trips_per_hour'] >= 2].copy()
    stops_4ph = stops_proj[stops_proj['trips_per_hour'] >= 4].copy()
    
    # Initialize new columns
    cbg_proj['dist_nearest_2ph'] = np.nan
    cbg_proj['dist_nearest_4ph'] = np.nan
    cbg_proj['stops_2ph_400m'] = 0
    cbg_proj['stops_2ph_800m'] = 0
    cbg_proj['stops_4ph_400m'] = 0
    cbg_proj['stops_4ph_800m'] = 0
    
    # Compute distances to frequent stops
    try:
        # Distance to nearest ≥2 trips/hour stop
        if len(stops_2ph) > 0:
            nearest_2ph = gpd.sjoin_nearest(
                centroids,
                stops_2ph,
                how='left',
                max_distance=50000,
                distance_col='distance'
            )
            dists_2ph = nearest_2ph.groupby(nearest_2ph.index)['distance'].min()
            cbg_proj.loc[dists_2ph.index, 'dist_nearest_2ph'] = dists_2ph.values
        
        # Distance to nearest ≥4 trips/hour stop
        if len(stops_4ph) > 0:
            nearest_4ph = gpd.sjoin_nearest(
                centroids,
                stops_4ph,
                how='left',
                max_distance=50000,
                distance_col='distance'
            )
            dists_4ph = nearest_4ph.groupby(nearest_4ph.index)['distance'].min()
            cbg_proj.loc[dists_4ph.index, 'dist_nearest_4ph'] = dists_4ph.values
    except (AttributeError, TypeError):
        # Fallback: use STRtree
        from shapely.strtree import STRtree
        
        if len(stops_2ph) > 0:
            tree_2ph = STRtree(stops_2ph.geometry.values)
            stops_2ph_array = np.array(stops_2ph.geometry.values)
            for idx in centroids.index:
                centroid_geom = centroids.loc[idx, 'geometry']
                nearest_idx = tree_2ph.nearest(centroid_geom)
                cbg_proj.loc[idx, 'dist_nearest_2ph'] = centroid_geom.distance(stops_2ph_array[nearest_idx])
        
        if len(stops_4ph) > 0:
            tree_4ph = STRtree(stops_4ph.geometry.values)
            stops_4ph_array = np.array(stops_4ph.geometry.values)
            for idx in centroids.index:
                centroid_geom = centroids.loc[idx, 'geometry']
                nearest_idx = tree_4ph.nearest(centroid_geom)
                cbg_proj.loc[idx, 'dist_nearest_4ph'] = centroid_geom.distance(stops_4ph_array[nearest_idx])
    
    # Count frequent stops within buffers
    print("  Counting frequent stops within buffers...")
    try:
        centroids_400 = centroids.copy()
        centroids_400.geometry = centroids_400.geometry.buffer(400)
        centroids_800 = centroids.copy()
        centroids_800.geometry = centroids_800.geometry.buffer(800)
        
        # Count ≥2 trips/hour stops
        if len(stops_2ph) > 0:
            stops_2ph_400 = gpd.sjoin(centroids_400, stops_2ph, how='left', predicate='contains')
            stops_2ph_800 = gpd.sjoin(centroids_800, stops_2ph, how='left', predicate='contains')
            cbg_proj['stops_2ph_400m'] = stops_2ph_400.groupby(stops_2ph_400.index).size().fillna(0).astype(int)
            cbg_proj['stops_2ph_800m'] = stops_2ph_800.groupby(stops_2ph_800.index).size().fillna(0).astype(int)
        
        # Count ≥4 trips/hour stops
        if len(stops_4ph) > 0:
            stops_4ph_400 = gpd.sjoin(centroids_400, stops_4ph, how='left', predicate='contains')
            stops_4ph_800 = gpd.sjoin(centroids_800, stops_4ph, how='left', predicate='contains')
            cbg_proj['stops_4ph_400m'] = stops_4ph_400.groupby(stops_4ph_400.index).size().fillna(0).astype(int)
            cbg_proj['stops_4ph_800m'] = stops_4ph_800.groupby(stops_4ph_800.index).size().fillna(0).astype(int)
    except Exception as e:
        print(f"  Warning: Could not compute buffer counts: {e}")
    
    print(f"  Completed frequency-aware metrics for {len(cbg_proj)} block groups")
    return cbg_proj


# ============================================================================
# 4. UPDATED TRANSIT DESERT CLASSIFICATIONS
# ============================================================================

def classify_transit_deserts_enhanced(cbg_gdf):
    """
    Create enhanced transit desert indicators using frequency-aware metrics.
    """
    print("Classifying transit deserts with frequency thresholds...")
    
    # Original distance-based deserts
    cbg_gdf['desert_any_800m'] = cbg_gdf['dist_nearest_stop'] > 800
    
    # Frequency-based deserts
    cbg_gdf['desert_freq_800m'] = (
        (cbg_gdf['dist_nearest_4ph'] > 800) | 
        (cbg_gdf['dist_nearest_4ph'].isna())
    )
    
    # Combined: no frequent service AND far from any service
    cbg_gdf['desert_severe_freq'] = (
        cbg_gdf['desert_freq_800m'] & 
        (cbg_gdf['dist_nearest_stop'] > 800)
    )
    
    # Population-weighted severity (if population data available)
    if 'total_pop' in cbg_gdf.columns and 'geometry' in cbg_gdf.columns:
        if 'area_km2' not in cbg_gdf.columns:
            cbg_gdf['area_km2'] = cbg_gdf.geometry.area / 1e6
        if 'pop_density' not in cbg_gdf.columns:
            cbg_gdf['pop_density'] = cbg_gdf['total_pop'] / cbg_gdf['area_km2']
        
        # High population density + frequency desert = priority desert
        median_density = cbg_gdf['pop_density'].median()
        cbg_gdf['desert_priority'] = (
            cbg_gdf['desert_freq_800m'] & 
            (cbg_gdf['pop_density'] > median_density)
        )
    else:
        cbg_gdf['desert_priority'] = cbg_gdf['desert_freq_800m']
    
    return cbg_gdf


# ============================================================================
# 5. POPULATION-WEIGHTED DESERT METRICS
# ============================================================================

def compute_population_weighted_metrics(cbg_gdf):
    """
    Compute population-weighted transit desert statistics.
    Returns summary table.
    """
    print("Computing population-weighted desert metrics...")
    
    if 'total_pop' not in cbg_gdf.columns:
        print("  Warning: No population data available")
        return pd.DataFrame()
    
    # Filter out block groups with missing population
    cbg_valid = cbg_gdf[cbg_gdf['total_pop'].notna() & (cbg_gdf['total_pop'] > 0)].copy()
    
    total_pop = cbg_valid['total_pop'].sum()
    
    summary = []
    
    # Desert type: Any (distance-based)
    if 'desert_any_800m' in cbg_valid.columns:
        desert_any = cbg_valid[cbg_valid['desert_any_800m']]
        pop_desert_any = desert_any['total_pop'].sum()
        summary.append({
            'desert_type': 'Distance-based (>800m)',
            'block_groups': len(desert_any),
            'population': pop_desert_any,
            'pct_population': (pop_desert_any / total_pop * 100) if total_pop > 0 else 0,
            'pct_block_groups': (len(desert_any) / len(cbg_valid) * 100) if len(cbg_valid) > 0 else 0
        })
    
    # Desert type: Frequency-based
    if 'desert_freq_800m' in cbg_valid.columns:
        desert_freq = cbg_valid[cbg_valid['desert_freq_800m']]
        pop_desert_freq = desert_freq['total_pop'].sum()
        summary.append({
            'desert_type': 'Frequency-based (no ≥4 trips/hr within 800m)',
            'block_groups': len(desert_freq),
            'population': pop_desert_freq,
            'pct_population': (pop_desert_freq / total_pop * 100) if total_pop > 0 else 0,
            'pct_block_groups': (len(desert_freq) / len(cbg_valid) * 100) if len(cbg_valid) > 0 else 0
        })
    
    # Desert type: Severe (frequency + distance)
    if 'desert_severe_freq' in cbg_valid.columns:
        desert_severe = cbg_valid[cbg_valid['desert_severe_freq']]
        pop_desert_severe = desert_severe['total_pop'].sum()
        summary.append({
            'desert_type': 'Severe (frequency + distance)',
            'block_groups': len(desert_severe),
            'population': pop_desert_severe,
            'pct_population': (pop_desert_severe / total_pop * 100) if total_pop > 0 else 0,
            'pct_block_groups': (len(desert_severe) / len(cbg_valid) * 100) if len(cbg_valid) > 0 else 0
        })
    
    # Desert type: Priority (high population density)
    if 'desert_priority' in cbg_valid.columns:
        desert_priority = cbg_valid[cbg_valid['desert_priority']]
        pop_desert_priority = desert_priority['total_pop'].sum()
        summary.append({
            'desert_type': 'Priority (high pop density + frequency desert)',
            'block_groups': len(desert_priority),
            'population': pop_desert_priority,
            'pct_population': (pop_desert_priority / total_pop * 100) if total_pop > 0 else 0,
            'pct_block_groups': (len(desert_priority) / len(cbg_valid) * 100) if len(cbg_valid) > 0 else 0
        })
    
    # Population density comparison
    if 'pop_density' in cbg_valid.columns:
        desert_any_density = cbg_valid[cbg_valid['desert_any_800m']]['pop_density'].mean() if 'desert_any_800m' in cbg_valid.columns else np.nan
        non_desert_density = cbg_valid[~cbg_valid['desert_any_800m']]['pop_density'].mean() if 'desert_any_800m' in cbg_valid.columns else np.nan
        
        summary.append({
            'desert_type': 'Mean pop density - Desert areas',
            'block_groups': np.nan,
            'population': np.nan,
            'pct_population': np.nan,
            'pct_block_groups': np.nan,
            'pop_density_mean': desert_any_density
        })
        
        summary.append({
            'desert_type': 'Mean pop density - Non-desert areas',
            'block_groups': np.nan,
            'population': np.nan,
            'pct_population': np.nan,
            'pct_block_groups': np.nan,
            'pop_density_mean': non_desert_density
        })
    
    summary_df = pd.DataFrame(summary)
    return summary_df


# ============================================================================
# 6. EQUITY ANALYSIS
# ============================================================================

def perform_equity_analysis(cbg_gdf):
    """
    Analyze how transit deserts intersect with equity variables.
    Returns equity summary and creates visualizations.
    """
    print("Performing equity analysis...")
    
    if 'total_pop' not in cbg_gdf.columns:
        print("  Warning: No ACS data available for equity analysis")
        return pd.DataFrame()
    
    # Prepare equity variables
    equity_data = cbg_gdf[cbg_gdf['total_pop'].notna() & (cbg_gdf['total_pop'] > 0)].copy()
    
    # Compute vehicle ownership rate (if available)
    # Note: vehicles_available might need interpretation based on ACS variable definition
    if 'vehicles_available' in equity_data.columns:
        # Assuming this is total vehicles, compute households with no vehicle
        # This is a simplification - actual ACS has B25044 for zero-vehicle households
        equity_data['has_vehicle'] = equity_data['vehicles_available'] > 0
        equity_data['no_vehicle'] = ~equity_data['has_vehicle']
    
    # Grouped comparisons: Desert vs Non-Desert
    equity_summary = []
    
    if 'desert_freq_800m' in equity_data.columns:
        desert_mask = equity_data['desert_freq_800m']
        
        # Median income comparison (with data cleaning)
        if 'median_income' in equity_data.columns:
            # Filter to reasonable income values
            valid_income = (equity_data['median_income'] > 0) & (equity_data['median_income'] < 500000)
            equity_data_clean = equity_data[valid_income].copy()
            desert_mask_clean = equity_data_clean['desert_freq_800m']
            
            if len(equity_data_clean) > 0:
                income_desert = equity_data_clean[desert_mask_clean]['median_income'].median()
                income_non_desert = equity_data_clean[~desert_mask_clean]['median_income'].median()
                equity_summary.append({
                    'variable': 'Median Household Income ($)',
                    'desert_value': income_desert,
                    'non_desert_value': income_non_desert,
                    'difference_pct': ((income_desert - income_non_desert) / income_non_desert * 100) if income_non_desert > 0 else np.nan
                })
        
        # Vehicle ownership comparison
        if 'no_vehicle' in equity_data.columns:
            no_vehicle_desert = equity_data[desert_mask]['no_vehicle'].mean() * 100
            no_vehicle_non_desert = equity_data[~desert_mask]['no_vehicle'].mean() * 100
            equity_summary.append({
                'variable': '% Households with No Vehicle',
                'desert_value': no_vehicle_desert,
                'non_desert_value': no_vehicle_non_desert,
                'difference_pct': no_vehicle_desert - no_vehicle_non_desert
            })
        
        # Population density comparison
        if 'pop_density' in equity_data.columns:
            density_desert = equity_data[desert_mask]['pop_density'].mean()
            density_non_desert = equity_data[~desert_mask]['pop_density'].mean()
            equity_summary.append({
                'variable': 'Mean Population Density (people/km²)',
                'desert_value': density_desert,
                'non_desert_value': density_non_desert,
                'difference_pct': ((density_desert - density_non_desert) / density_non_desert * 100) if density_non_desert > 0 else np.nan
            })
    
    equity_df = pd.DataFrame(equity_summary)
    
    # Create equity visualizations
    if len(equity_data) > 0 and 'desert_freq_800m' in equity_data.columns:
        create_equity_visualizations(equity_data)
    
    return equity_df


def create_equity_visualizations(cbg_gdf):
    """Create equity analysis figures."""
    print("  Creating equity visualizations...")
    
    desert_mask = cbg_gdf['desert_freq_800m']
    
    # Clean income data - filter out invalid values
    if 'median_income' in cbg_gdf.columns:
        # Filter to reasonable income range (0 to $500k)
        valid_income = (cbg_gdf['median_income'] > 0) & (cbg_gdf['median_income'] < 500000)
        cbg_clean = cbg_gdf[valid_income].copy()
        desert_mask_clean = cbg_clean['desert_freq_800m']
        
        if len(cbg_clean) > 0:
            # Figure 1: Income comparison boxplot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            income_desert = cbg_clean[desert_mask_clean]['median_income'].dropna()
            income_non_desert = cbg_clean[~desert_mask_clean]['median_income'].dropna()
            
            if len(income_desert) > 0 and len(income_non_desert) > 0:
                bp = ax.boxplot([income_non_desert, income_desert], 
                                labels=['Non-Desert', 'Transit Desert'],
                                patch_artist=True,
                                showmeans=True)
                bp['boxes'][0].set_facecolor('#2ecc71')
                bp['boxes'][1].set_facecolor('#e74c3c')
                
                ax.set_ylabel('Median Household Income ($)', fontsize=12)
                ax.set_title('Household Income: Transit Deserts vs Non-Deserts', 
                             fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add legend explaining boxplot elements
                from matplotlib.patches import Patch, Rectangle
                legend_elements = [
                    Patch(facecolor='#2ecc71', label='Non-Desert Areas'),
                    Patch(facecolor='#e74c3c', label='Transit Desert Areas'),
                    Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='black', label='Box: IQR'),
                    Rectangle((0, 0), 1, 1, facecolor='white', edgecolor='red', linestyle='--', label='Mean (dashed)')
                ]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
                
                # Add statistics text
                med_desert = income_desert.median()
                med_non_desert = income_non_desert.median()
                ax.text(0.02, 0.02, f'Median (Non-Desert): ${med_non_desert:,.0f}\nMedian (Desert): ${med_desert:,.0f}',
                       transform=ax.transAxes, verticalalignment='bottom',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
                
                plt.tight_layout()
                plt.savefig(output_dir / "equity_income_comparison.png", dpi=300, bbox_inches='tight')
                plt.close()
                print("    Created: equity_income_comparison.png")
            
            # Figure 2: Income vs Distance scatter
            if 'dist_nearest_4ph' in cbg_clean.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                scatter_data = cbg_clean[['median_income', 'dist_nearest_4ph', 'desert_freq_800m']].dropna()
                
                if len(scatter_data) > 0:
                    colors = scatter_data['desert_freq_800m'].map({True: '#e74c3c', False: '#2ecc71'})
                    ax.scatter(scatter_data['dist_nearest_4ph'], 
                              scatter_data['median_income'],
                              c=colors, alpha=0.4, s=15, edgecolors='none')
                    
                    ax.set_xlabel('Distance to Nearest Frequent Transit (meters)', fontsize=12)
                    ax.set_ylabel('Median Household Income ($)', fontsize=12)
                    ax.set_title('Income vs Transit Accessibility', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    
                    # Add legend
                    from matplotlib.patches import Patch
                    legend_elements = [
                        Patch(facecolor='#2ecc71', label='Non-Desert'),
                        Patch(facecolor='#e74c3c', label='Transit Desert')
                    ]
                    ax.legend(handles=legend_elements, loc='upper right')
                    
                    plt.tight_layout()
                    plt.savefig(output_dir / "equity_income_distance.png", dpi=300, bbox_inches='tight')
                    plt.close()
                    print("    Created: equity_income_distance.png")
        
        # Figure 3: Income distribution histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        
        income_desert = cbg_clean[desert_mask_clean]['median_income'].dropna()
        income_non_desert = cbg_clean[~desert_mask_clean]['median_income'].dropna()
        
        if len(income_desert) > 0 and len(income_non_desert) > 0:
            ax.hist([income_non_desert, income_desert], 
                   bins=30, alpha=0.7, label=['Non-Desert', 'Transit Desert'],
                   color=['#2ecc71', '#e74c3c'], edgecolor='black', linewidth=0.5)
            ax.set_xlabel('Median Household Income ($)', fontsize=12)
            ax.set_ylabel('Number of Block Groups', fontsize=12)
            ax.set_title('Distribution of Household Income', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(output_dir / "equity_income_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("    Created: equity_income_distribution.png")
    
    # Figure 4: Population density comparison
    if 'pop_density' in cbg_gdf.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        density_desert = cbg_gdf[desert_mask]['pop_density'].dropna()
        density_non_desert = cbg_gdf[~desert_mask]['pop_density'].dropna()
        
        if len(density_desert) > 0 and len(density_non_desert) > 0:
            # Use log scale for better visualization
            bp = ax.boxplot([np.log10(density_non_desert + 1), np.log10(density_desert + 1)], 
                           labels=['Non-Desert', 'Transit Desert'],
                           patch_artist=True,
                           showmeans=True)
            bp['boxes'][0].set_facecolor('#2ecc71')
            bp['boxes'][1].set_facecolor('#e74c3c')
            
            ax.set_ylabel('Log10(Population Density + 1) (people/km²)', fontsize=12)
            ax.set_title('Population Density: Transit Deserts vs Non-Deserts', 
                         fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#2ecc71', label='Non-Desert Areas'),
                Patch(facecolor='#e74c3c', label='Transit Desert Areas')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
            
            # Add note about log scale
            mean_desert = density_desert.mean()
            mean_non_desert = density_non_desert.mean()
            ax.text(0.02, 0.02, f'Mean (Non-Desert): {mean_non_desert:,.0f} people/km²\nMean (Desert): {mean_desert:,.0f} people/km²\n(Log scale for visualization)',
                   transform=ax.transAxes, verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)
            
            plt.tight_layout()
            plt.savefig(output_dir / "equity_pop_density_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("    Created: equity_pop_density_comparison.png")


# ============================================================================
# 7. SENSITIVITY ANALYSIS
# ============================================================================

def sensitivity_analysis_distance_thresholds(cbg_gdf):
    """
    Test transit desert definitions at different distance thresholds.
    Returns sensitivity summary.
    """
    print("Performing sensitivity analysis on distance thresholds...")
    
    if 'total_pop' not in cbg_gdf.columns or 'dist_nearest_stop' not in cbg_gdf.columns:
        print("  Warning: Insufficient data for sensitivity analysis")
        return pd.DataFrame()
    
    cbg_valid = cbg_gdf[cbg_gdf['total_pop'].notna() & (cbg_gdf['total_pop'] > 0)].copy()
    total_pop = cbg_valid['total_pop'].sum()
    
    thresholds = [600, 800, 1000]
    sensitivity_results = []
    
    for threshold in thresholds:
        desert_mask = cbg_valid['dist_nearest_stop'] > threshold
        pop_desert = cbg_valid[desert_mask]['total_pop'].sum()
        
        sensitivity_results.append({
            'threshold_meters': threshold,
            'block_groups_desert': desert_mask.sum(),
            'pct_block_groups': (desert_mask.sum() / len(cbg_valid) * 100),
            'population_desert': pop_desert,
            'pct_population': (pop_desert / total_pop * 100) if total_pop > 0 else 0
        })
    
    sensitivity_df = pd.DataFrame(sensitivity_results)
    
    # Interpretation comments (printed, not in DataFrame)
    print("\n  Sensitivity Analysis Interpretation:")
    print("  - 600m: More restrictive, identifies more areas as deserts")
    print("  - 800m: Standard threshold (5-10 min walk)")
    print("  - 1000m: Less restrictive, fewer areas classified as deserts")
    print("  - Threshold choice should balance policy goals with practical walking distances")
    
    return sensitivity_df


# ============================================================================
# 8. SPATIAL AUTOCORRELATION ANALYSIS
# ============================================================================

def compute_spatial_autocorrelation(cbg_gdf):
    """
    Compute Moran's I for transit accessibility metrics.
    Identifies spatial clustering of transit deserts.
    """
    print("Computing spatial autocorrelation (Moran's I)...")
    
    try:
        from libpysal.weights import Queen
        from esda.moran import Moran
    except ImportError:
        print("  Warning: libpysal/esda not available. Install with: pip install libpysal esda")
        print("  Skipping spatial autocorrelation analysis")
        return None
    
    # Project to appropriate CRS for spatial weights
    cbg_proj = cbg_gdf.to_crs("EPSG:3310")
    
    # Create spatial weights matrix (Queen contiguity)
    print("  Creating spatial weights matrix...")
    w = Queen.from_dataframe(cbg_proj)
    w.transform = 'r'  # Row-standardize
    
    # Compute Moran's I for key accessibility metrics
    moran_results = {}
    
    metrics_to_test = ['dist_nearest_stop', 'dist_nearest_4ph', 'desert_freq_800m']
    
    for metric in metrics_to_test:
        if metric in cbg_proj.columns:
            values = cbg_proj[metric].values
            
            # Handle NaN values
            valid_mask = ~np.isnan(values)
            if metric == 'desert_freq_800m':
                # Convert boolean to numeric
                values = values.astype(float)
            
            if valid_mask.sum() > 10:  # Need sufficient valid values
                try:
                    # Create subset weights matrix
                    valid_indices = np.where(valid_mask)[0]
                    w_subset = w[valid_indices][:, valid_indices]
                    values_subset = values[valid_mask]
                    
                    moran = Moran(values_subset, w_subset)
                    moran_results[metric] = {
                        'I': moran.I,
                        'p_value': moran.p_norm,
                        'z_score': moran.z_norm,
                        'interpretation': 'Clustered' if moran.I > 0 else 'Dispersed'
                    }
                    print(f"  {metric}: I={moran.I:.4f}, p={moran.p_norm:.4f} ({moran_results[metric]['interpretation']})")
                except Exception as e:
                    print(f"  Warning: Could not compute Moran's I for {metric}: {e}")
    
    return moran_results


def identify_hot_cold_spots(cbg_gdf):
    """
    Identify hot spots (poor accessibility clusters) and cold spots (good accessibility clusters).
    """
    print("Identifying hot/cold spots...")
    
    try:
        from libpysal.weights import Queen
        from esda.getisord import G_Local
    except ImportError:
        print("  Warning: libpysal/esda not available for hot spot analysis")
        return cbg_gdf
    
    cbg_proj = cbg_gdf.to_crs("EPSG:3310")
    
    # Create spatial weights
    w = Queen.from_dataframe(cbg_proj)
    w.transform = 'r'
    
    # Compute Getis-Ord Gi* for distance to frequent transit
    if 'dist_nearest_4ph' in cbg_proj.columns:
        values = cbg_proj['dist_nearest_4ph'].fillna(cbg_proj['dist_nearest_4ph'].max())
        
        try:
            g_local = G_Local(values.values, w, star=True)
            
            # Classify hot/cold spots (using 95% confidence)
            cbg_proj['hotspot'] = (g_local.Zs > 1.96) & (g_local.p_sim < 0.05)
            cbg_proj['coldspot'] = (g_local.Zs < -1.96) & (g_local.p_sim < 0.05)
            cbg_proj['gistar_z'] = g_local.Zs
            
            print(f"  Hot spots (poor accessibility): {cbg_proj['hotspot'].sum()}")
            print(f"  Cold spots (good accessibility): {cbg_proj['coldspot'].sum()}")
            
            # Merge back to original CRS
            cbg_gdf['hotspot'] = cbg_proj['hotspot'].values
            cbg_gdf['coldspot'] = cbg_proj['coldspot'].values
            cbg_gdf['gistar_z'] = cbg_proj['gistar_z'].values
            
        except Exception as e:
            print(f"  Warning: Could not compute hot spots: {e}")
    
    return cbg_gdf


# ============================================================================
# 9. TABLE VISUALIZATION FUNCTIONS
# ============================================================================

def create_summary_table(df, title, output_path):
    """Create a formatted table visualization from a DataFrame."""
    if len(df) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(14, max(6, len(df) * 0.5)))
    ax.axis('tight')
    ax.axis('off')
    
    # Format the DataFrame for display
    df_display = df.copy()
    
    # Format numeric columns
    for col in df_display.columns:
        if df_display[col].dtype in [np.float64, np.int64]:
            if 'pct' in col.lower() or 'percent' in col.lower():
                df_display[col] = df_display[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "")
            elif 'income' in col.lower() or 'value' in col.lower():
                df_display[col] = df_display[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) and x > 0 else "")
            elif 'population' in col.lower():
                df_display[col] = df_display[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
            elif 'density' in col.lower():
                df_display[col] = df_display[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
            else:
                df_display[col] = df_display[col].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
    
    # Replace NaN with empty string
    df_display = df_display.fillna("")
    
    # Create table
    table = ax.table(cellText=df_display.values,
                    colLabels=df_display.columns,
                    cellLoc='left',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Header styling
    for i in range(len(df_display.columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(0, i)].set_height(0.08)
    
    # Row styling (alternating colors)
    for i in range(1, len(df_display) + 1):
        for j in range(len(df_display.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            else:
                table[(i, j)].set_facecolor('white')
            table[(i, j)].set_height(0.06)
    
    # Title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Created: {output_path.name}")


# ============================================================================
# 10. VISUALIZATION FUNCTIONS
# ============================================================================

def plot_frequency_based_accessibility(cbg_gdf, output_path):
    """Create map of distance to frequent transit (≥4 trips/hour)."""
    cbg_vis = cbg_gdf.to_crs("EPSG:4326")
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot with colorbar
    cbg_vis.plot(
        ax=ax,
        column='dist_nearest_4ph',
        cmap='YlOrRd',
        legend=True,
        legend_kwds={
            'label': 'Distance to Nearest Frequent Transit (≥4 trips/hr) (meters)',
            'shrink': 0.8,
            'orientation': 'vertical',
            'pad': 0.02
        },
        edgecolor='gray',
        linewidth=0.1,
        missing_kwds={'color': 'lightgray', 'label': 'No frequent transit'}
    )
    
    ax.set_title("Distance to Frequent Transit Service\nLos Angeles County Census Block Groups", 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.axis('off')
    
    # Add note about frequent transit definition
    ax.text(0.02, 0.02, 'Frequent transit: ≥4 trips per hour\nData: LA Metro GTFS 2022',
           transform=ax.transAxes, fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_frequency_deserts(cbg_gdf, output_path):
    """Create map of frequency-based transit deserts."""
    cbg_vis = cbg_gdf.to_crs("EPSG:4326")
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Create classification
    cbg_vis['desert_class'] = 'Not a Desert'
    cbg_vis.loc[cbg_vis['desert_freq_800m'], 'desert_class'] = 'Frequency Desert'
    cbg_vis.loc[cbg_vis['desert_priority'], 'desert_class'] = 'Priority Desert'
    
    colors = {
        'Not a Desert': '#2ecc71',
        'Frequency Desert': '#f39c12',
        'Priority Desert': '#e74c3c'
    }
    
    # Plot each class with proper labels
    for class_name, color in colors.items():
        subset = cbg_vis[cbg_vis['desert_class'] == class_name]
        if len(subset) > 0:
            subset.plot(ax=ax, color=color, edgecolor='gray', linewidth=0.1, label=class_name)
    
    ax.set_title("Frequency-Based Transit Desert Classification\nLos Angeles County Census Block Groups", 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    
    # Enhanced legend with descriptions
    legend_labels = [
        'Not a Desert\n(≥4 trips/hr within 800m)',
        'Frequency Desert\n(No ≥4 trips/hr within 800m)',
        'Priority Desert\n(High pop density + frequency desert)'
    ]
    ax.legend(legend_labels, loc='upper right', frameon=True, fancybox=True, 
             shadow=True, fontsize=10, title='Desert Classification', title_fontsize=11)
    
    ax.axis('off')
    
    # Add note
    ax.text(0.02, 0.02, 'Threshold: 800m from frequent transit\nData: LA Metro GTFS 2022',
           transform=ax.transAxes, fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_population_weighted_deserts(cbg_gdf, output_path):
    """Create map showing population-weighted desert intensity."""
    cbg_vis = cbg_gdf.to_crs("EPSG:4326")
    
    if 'total_pop' not in cbg_vis.columns:
        print("  Warning: No population data for population-weighted map")
        return
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Create intensity metric: population * desert indicator
    cbg_vis['desert_intensity'] = 0
    cbg_vis.loc[cbg_vis['desert_freq_800m'], 'desert_intensity'] = cbg_vis.loc[cbg_vis['desert_freq_800m'], 'total_pop']
    
    cbg_vis.plot(
        ax=ax,
        column='desert_intensity',
        cmap='Reds',
        legend=True,
        legend_kwds={
            'label': 'Population in Transit Deserts (people)',
            'shrink': 0.8,
            'orientation': 'vertical',
            'pad': 0.02
        },
        edgecolor='gray',
        linewidth=0.1,
        missing_kwds={'color': 'lightgray', 'label': 'No data'}
    )
    
    ax.set_title("Population-Weighted Transit Desert Intensity\nLos Angeles County Census Block Groups", 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.axis('off')
    
    # Add statistics
    total_pop_desert = cbg_vis[cbg_vis['desert_freq_800m']]['total_pop'].sum()
    total_pop = cbg_vis['total_pop'].sum()
    pct_desert = (total_pop_desert / total_pop * 100) if total_pop > 0 else 0
    
    ax.text(0.02, 0.02, f'Population in deserts: {total_pop_desert:,.0f} ({pct_desert:.1f}%)\nData: ACS 2022, LA Metro GTFS 2022',
           transform=ax.transAxes, fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================================
# 10. PROJECT 28 HOOKS (Placeholders for Counterfactual Analysis)
# ============================================================================

def add_proposed_stops_to_network(stops_gdf, proposed_stops_path):
    """
    PLACEHOLDER FOR PROJECT 28 COUNTERFACTUAL ANALYSIS
    
    This function will be reused for Project 28 counterfactual analysis.
    It adds proposed D Line stations to the transit network.
    
    Args:
        stops_gdf: Current stops GeoDataFrame with frequency metrics
        proposed_stops_path: Path to GeoJSON/CSV with proposed stops
    
    Returns:
        Combined stops GeoDataFrame with proposed stops added
    """
    # TODO: Implement when Project 28 data is available
    # 1. Load proposed stops from GeoJSON/CSV
    # 2. Assign assumed service frequency (e.g., 4 trips/hour for new rail)
    # 3. Merge with existing stops
    # 4. Return combined network
    
    print("Project 28 counterfactual: add_proposed_stops_to_network() - Not yet implemented")
    return stops_gdf


def recompute_accessibility_with_proposed(cbg_gdf, stops_with_proposed):
    """
    PLACEHOLDER FOR PROJECT 28 COUNTERFACTUAL ANALYSIS
    
    This function will be reused for Project 28 counterfactual analysis.
    Recomputes all accessibility metrics with proposed infrastructure.
    
    Args:
        cbg_gdf: Current Census Block Group GeoDataFrame
        stops_with_proposed: Stops GeoDataFrame including proposed stops
    
    Returns:
        Updated CBG GeoDataFrame with counterfactual metrics
    """
    # TODO: Implement when Project 28 data is available
    # 1. Use compute_frequency_aware_accessibility() with new stops
    # 2. Reclassify deserts
    # 3. Compare baseline vs. proposed
    # 4. Compute population impact
    
    print("Project 28 counterfactual: recompute_accessibility_with_proposed() - Not yet implemented")
    return cbg_gdf


# ============================================================================
# 11. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 70)
    print("Enhanced Transit Accessibility Analysis")
    print("Service-aware, Population-weighted, Equity-focused Baseline")
    print("=" * 70)
    
    # Load data
    print("\n1. Loading data...")
    cbg = load_census_block_groups()
    acs = load_acs_attributes()
    
    # Prepare Census Block Groups
    print("\n2. Preparing Census Block Groups...")
    cbg['tract_geoid'] = cbg['GEOID'].str[:11]
    acs['tract_geoid'] = acs['GEOID'].str[:11]
    cbg = cbg.merge(acs[['tract_geoid', 'total_pop', 'median_income', 'vehicles_available']], 
                   on='tract_geoid', how='left')
    
    # Project and compute centroids
    cbg_proj = cbg.to_crs("EPSG:3310")
    cbg_proj['centroid'] = cbg_proj.geometry.centroid
    cbg_proj['area_km2'] = cbg_proj.geometry.area / 1e6
    cbg_proj['pop_density'] = cbg_proj['total_pop'] / cbg_proj['area_km2']
    
    # Load existing accessibility metrics (if available) or compute basic ones
    existing_metrics_path = output_dir / "accessibility_metrics.geojson"
    if existing_metrics_path.exists():
        print("  Loading existing accessibility metrics...")
        existing = gpd.read_file(existing_metrics_path)
        # Merge existing metrics
        if 'dist_nearest_stop' in existing.columns:
            cbg_proj = cbg_proj.merge(
                existing[['GEOID', 'dist_nearest_stop', 'dist_nearest_rail', 
                         'stops_400m', 'stops_800m']],
                on='GEOID',
                how='left',
                suffixes=('', '_existing')
            )
            # Use existing if new not available
            if 'dist_nearest_stop_existing' in cbg_proj.columns:
                cbg_proj['dist_nearest_stop'] = cbg_proj['dist_nearest_stop'].fillna(
                    cbg_proj['dist_nearest_stop_existing']
                )
    else:
        print("  Computing basic accessibility metrics...")
        # Compute basic metrics using stops without frequency
        stops_basic = create_stops_with_frequency()  # Will compute frequency but we can use for basic metrics too
        cbg_proj = compute_basic_accessibility_metrics(cbg_proj, stops_basic)
    
    # Compute stop frequencies
    print("\n3. Computing service frequencies...")
    stops_with_freq = create_stops_with_frequency()
    
    # Compute frequency-aware accessibility
    print("\n4. Computing frequency-aware accessibility...")
    cbg_proj = compute_frequency_aware_accessibility(cbg_proj, stops_with_freq)
    
    # Classify deserts
    print("\n5. Classifying transit deserts...")
    cbg_proj = classify_transit_deserts_enhanced(cbg_proj)
    
    # Population-weighted metrics
    print("\n6. Computing population-weighted metrics...")
    pop_summary = compute_population_weighted_metrics(cbg_proj)
    if len(pop_summary) > 0:
        pop_summary.to_csv(output_dir / "desert_population_summary.csv", index=False)
        print(f"  Saved: {output_dir / 'desert_population_summary.csv'}")
        # Create formatted table visualization
        create_summary_table(pop_summary, "Transit Desert Population Summary", 
                           output_dir / "desert_population_summary_table.png")
        print("\n  Population Summary:")
        print(pop_summary.to_string())
    
    # Equity analysis
    print("\n7. Performing equity analysis...")
    equity_summary = perform_equity_analysis(cbg_proj)
    if len(equity_summary) > 0:
        equity_summary.to_csv(output_dir / "equity_analysis_summary.csv", index=False)
        print(f"  Saved: {output_dir / 'equity_analysis_summary.csv'}")
        # Create formatted table visualization
        create_summary_table(equity_summary, "Equity Analysis Summary", 
                           output_dir / "equity_analysis_summary_table.png")
        print("\n  Equity Summary:")
        print(equity_summary.to_string())
    
    # Sensitivity analysis
    print("\n8. Sensitivity analysis...")
    sensitivity_df = sensitivity_analysis_distance_thresholds(cbg_proj)
    if len(sensitivity_df) > 0:
        sensitivity_df.to_csv(output_dir / "sensitivity_analysis.csv", index=False)
        print(f"  Saved: {output_dir / 'sensitivity_analysis.csv'}")
        # Create formatted table visualization
        create_summary_table(sensitivity_df, "Sensitivity Analysis: Distance Thresholds", 
                           output_dir / "sensitivity_analysis_table.png")
    
    # Spatial autocorrelation
    print("\n9. Spatial autocorrelation analysis...")
    moran_results = compute_spatial_autocorrelation(cbg_proj)
    if moran_results:
        import json
        with open(output_dir / "spatial_autocorrelation.json", 'w') as f:
            json.dump(moran_results, f, indent=2)
        print(f"  Saved: {output_dir / 'spatial_autocorrelation.json'}")
    
    # Hot/cold spots
    print("\n10. Identifying hot/cold spots...")
    cbg_proj = identify_hot_cold_spots(cbg_proj)
    
    # Create visualizations
    print("\n11. Creating visualizations...")
    plot_frequency_based_accessibility(cbg_proj, output_dir / "distance_to_frequent_transit.png")
    plot_frequency_deserts(cbg_proj, output_dir / "frequency_transit_deserts.png")
    plot_population_weighted_deserts(cbg_proj, output_dir / "population_weighted_deserts.png")
    
    # Save final baseline
    print("\n12. Saving final baseline...")
    cbg_final = cbg_proj.drop(columns=['centroid']).to_crs("EPSG:4326")
    baseline_path = output_dir / "baseline_accessibility_enhanced.geojson"
    cbg_final.to_file(baseline_path, driver='GeoJSON')
    print(f"  Saved: {baseline_path}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("Enhanced Baseline Analysis Complete!")
    print("=" * 70)
    print(f"\nTotal Block Groups: {len(cbg_final)}")
    if 'desert_freq_800m' in cbg_final.columns:
        desert_count = cbg_final['desert_freq_800m'].sum()
        print(f"Frequency-based transit deserts: {desert_count} ({100*desert_count/len(cbg_final):.1f}%)")
    if 'total_pop' in cbg_final.columns:
        total_pop = cbg_final['total_pop'].sum()
        if 'desert_freq_800m' in cbg_final.columns:
            pop_in_desert = cbg_final[cbg_final['desert_freq_800m']]['total_pop'].sum()
            print(f"Population in frequency deserts: {pop_in_desert:,.0f} ({100*pop_in_desert/total_pop:.1f}%)")
    
    print("\nReady for Project 28 counterfactual analysis!")
    print("Use add_proposed_stops_to_network() and recompute_accessibility_with_proposed()")


if __name__ == "__main__":
    main()

