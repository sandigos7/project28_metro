"""
Transit Accessibility Analysis for Los Angeles County
Computes baseline accessibility metrics and identifies transit deserts.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import Point
import warnings
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

def load_census_block_groups():
    """
    Load Census Block Group shapefiles for Los Angeles County.
    Downloads from Census Bureau if not present locally.
    """
    data_dir = project_root / "data" / "acs_2022_la_county"
    data_dir.mkdir(exist_ok=True)
    cbg_path = data_dir / "cbg_2022_06_037.shp"
    
    # Check if shapefile exists, if not download it
    if not cbg_path.exists():
        print("Census Block Group shapefiles not found. Downloading...")
        try:
            import urllib.request
            import zipfile
            
            # LA County FIPS: 06 (state) + 037 (county)
            # Download all CA block groups (will filter to LA County)
            url = "https://www2.census.gov/geo/tiger/TIGER2022/BG/tl_2022_06_bg.zip"
            zip_path = data_dir / "tl_2022_06_bg.zip"
            
            if not zip_path.exists():
                print(f"  Downloading from Census Bureau...")
                urllib.request.urlretrieve(url, zip_path)
                print(f"  Download complete")
            
            # Extract
            print("  Extracting shapefiles...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(data_dir)
            
            # Load all block groups and filter to LA County (COUNTYFP = '037')
            shp_file = data_dir / "tl_2022_06_bg.shp"
            if shp_file.exists():
                cbg_all = gpd.read_file(shp_file)
                cbg_la = cbg_all[cbg_all['COUNTYFP'] == '037'].copy()
                
                if len(cbg_la) == 0:
                    raise ValueError("No block groups found for LA County. Check COUNTYFP values.")
                
                # Save filtered version
                cbg_la.to_file(cbg_path)
                print(f"  Loaded and saved {len(cbg_la)} Census Block Groups for LA County")
            else:
                raise FileNotFoundError(f"Shapefile not found after extraction: {shp_file}")
                
        except Exception as e:
            print(f"  Error downloading Block Groups: {e}")
            print("  Attempting to use Census Tracts as fallback...")
            # Fallback to tracts
            return load_census_tracts_fallback()
    
    cbg = gpd.read_file(cbg_path)
    print(f"Loaded {len(cbg)} Census Block Groups")
    return cbg


def load_census_tracts_fallback():
    """Fallback: Load Census Tracts if Block Groups unavailable."""
    data_dir = project_root / "data" / "acs_2022_la_county"
    tract_path = data_dir / "tract_2022_06_037.shp"
    
    if not tract_path.exists():
        print("  Downloading Census Tract shapefiles...")
        import urllib.request
        import zipfile
        
        url = "https://www2.census.gov/geo/tiger/TIGER2022/TRACT/tl_2022_06_tract.zip"
        zip_path = data_dir / "tl_2022_06_tract.zip"
        
        if not zip_path.exists():
            urllib.request.urlretrieve(url, zip_path)
        
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(data_dir)
        
        tract_all = gpd.read_file(data_dir / "tl_2022_06_tract.shp")
        tract_la = tract_all[tract_all['COUNTYFP'] == '037'].copy()
        tract_la.to_file(tract_path)
        print(f"  Loaded {len(tract_la)} Census Tracts for LA County")
    
    return gpd.read_file(tract_path)


def load_acs_attributes():
    """Load ACS 2022 attributes from CSV."""
    acs_path = project_root / "data" / "acs_2022_la_county.csv"
    acs = pd.read_csv(acs_path)
    
    # Rename columns for clarity
    # B01003_001E: Total population
    # B19013_001E: Median household income
    # B25046_001E: Vehicles available
    # B08126_001E, B08301_001E: Transportation variables
    acs = acs.rename(columns={
        'B01003_001E': 'total_pop',
        'B19013_001E': 'median_income',
        'B25046_001E': 'vehicles_available',
        'B08126_001E': 'transit_var1',
        'B08301_001E': 'transit_var2'
    })
    
    # Extract GEOID from NAME or create from state+county+tract
    # For Block Groups, we'll need to join on GEOID
    # For now, create GEOID from tract (assuming tract-level data)
    if 'GEOID' not in acs.columns:
        acs['GEOID'] = acs['state'].astype(str).str.zfill(2) + \
                      acs['county'].astype(str).str.zfill(3) + \
                      acs['tract'].astype(str).str.zfill(6)
    
    return acs


def load_transit_stops():
    """Load transit stops from build_network.py output or recreate."""
    # Import from build_network to reuse the loading logic
    import sys
    sys.path.insert(0, str(script_dir))
    
    # Reuse the GTFS loading logic
    import zipfile
    gtfs_bus_zip = project_root / "data" / "gtfs_bus" / "gtfs_bus.zip"
    gtfs_rail_zip = project_root / "data" / "gtfs_rail" / "gtfs_rail.zip"
    
    def read_csv_from_zip(zip_path, csv_file):
        with zipfile.ZipFile(zip_path, 'r') as z:
            with z.open(csv_file) as f:
                return pd.read_csv(f, low_memory=False)
    
    # Load stops
    bus_stops = read_csv_from_zip(gtfs_bus_zip, "stops.txt")
    rail_stops = read_csv_from_zip(gtfs_rail_zip, "stops.txt")
    
    bus_stops['mode'] = 'bus'
    rail_stops['mode'] = 'rail'
    stops_all = pd.concat([bus_stops, rail_stops], ignore_index=True)
    
    # Create GeoDataFrame
    stops_gdf = gpd.GeoDataFrame(
        stops_all,
        geometry=gpd.points_from_xy(stops_all.stop_lon, stops_all.stop_lat),
        crs="EPSG:4326"
    )
    
    return stops_gdf[['stop_id', 'stop_name', 'mode', 'geometry']]


# ============================================================================
# 2. DATA PREPARATION FUNCTIONS
# ============================================================================

def prepare_census_block_groups(cbg, acs):
    """
    Prepare Census Block Groups: filter, project, compute centroids.
    """
    # Filter to Los Angeles County (should already be filtered, but double-check)
    # LA County GEOID starts with 06037
    cbg = cbg[cbg['GEOID'].str.startswith('06037')].copy()
    
    # Join ACS attributes on GEOID (first 11 digits for tract-level, full for BG)
    # For tract-level ACS data, join on tract GEOID (first 11 chars)
    if len(acs) > 0:
        acs['tract_geoid'] = acs['GEOID'].str[:11]  # Tract-level GEOID
        cbg['tract_geoid'] = cbg['GEOID'].str[:11]
        cbg = cbg.merge(acs[['tract_geoid', 'total_pop', 'median_income', 'vehicles_available']], 
                       on='tract_geoid', how='left')
    
    # Project to California Albers (EPSG:3310) for distance calculations
    cbg_projected = cbg.to_crs("EPSG:3310")
    
    # Compute population-weighted centroids (or geometric centroids)
    if 'total_pop' in cbg_projected.columns:
        # Use geometric centroid for now (population weighting requires more complex calculation)
        cbg_projected['centroid'] = cbg_projected.geometry.centroid
    else:
        cbg_projected['centroid'] = cbg_projected.geometry.centroid
    
    return cbg_projected


def prepare_transit_stops(stops_gdf):
    """Prepare transit stops: project to same CRS as CBGs."""
    # Project to EPSG:3310 (California Albers)
    stops_projected = stops_gdf.to_crs("EPSG:3310")
    return stops_projected


# ============================================================================
# 3. ACCESSIBILITY METRICS COMPUTATION
# ============================================================================

def compute_accessibility_metrics(cbg_gdf, stops_gdf):
    """
    Compute accessibility metrics for each Census Block Group.
    Uses spatial indexing (sjoin_nearest) for efficiency.
    """
    print("Computing accessibility metrics...")
    
    # Get centroids as GeoDataFrame
    centroids = gpd.GeoDataFrame(
        geometry=cbg_gdf['centroid'],
        index=cbg_gdf.index,
        crs=cbg_gdf.crs
    )
    
    # Separate bus and rail stops
    all_stops = stops_gdf.copy()
    rail_stops = stops_gdf[stops_gdf['mode'] == 'rail'].copy()
    
    # Initialize metrics columns
    cbg_gdf['dist_nearest_stop'] = np.nan
    cbg_gdf['dist_nearest_rail'] = np.nan
    cbg_gdf['stops_400m'] = 0
    cbg_gdf['stops_800m'] = 0
    
    # Use sjoin_nearest for efficient nearest neighbor (requires geopandas >= 0.11)
    try:
        # Nearest any stop
        print("  Using sjoin_nearest for distance calculations...")
        nearest_all = gpd.sjoin_nearest(
            centroids, 
            all_stops, 
            how='left',
            max_distance=50000,  # 50km max search distance
            distance_col='distance'
        )
        # Group by original index and get minimum distance
        dists_all = nearest_all.groupby(nearest_all.index)['distance'].min()
        cbg_gdf.loc[dists_all.index, 'dist_nearest_stop'] = dists_all.values
        
        # Nearest rail stop
        if len(rail_stops) > 0:
            nearest_rail = gpd.sjoin_nearest(
                centroids,
                rail_stops,
                how='left',
                max_distance=50000,
                distance_col='distance'
            )
            dists_rail = nearest_rail.groupby(nearest_rail.index)['distance'].min()
            cbg_gdf.loc[dists_rail.index, 'dist_nearest_rail'] = dists_rail.values
    except (AttributeError, TypeError) as e:
        # Fallback: use STRtree for efficient nearest neighbor
        print(f"  Using STRtree for distance calculation ({type(e).__name__})...")
        from shapely.strtree import STRtree
        
        # Build spatial index for stops
        tree = STRtree(all_stops.geometry.values)
        all_stops_array = np.array(all_stops.geometry.values)
        
        print("    Computing distances to all stops...")
        for idx in centroids.index:
            centroid_geom = centroids.loc[idx, 'geometry']
            # Find nearest stop using spatial index
            nearest_idx = tree.nearest(centroid_geom)
            cbg_gdf.loc[idx, 'dist_nearest_stop'] = centroid_geom.distance(all_stops_array[nearest_idx])
        
        if len(rail_stops) > 0:
            print("    Computing distances to rail stops...")
            tree_rail = STRtree(rail_stops.geometry.values)
            rail_stops_array = np.array(rail_stops.geometry.values)
            for idx in centroids.index:
                centroid_geom = centroids.loc[idx, 'geometry']
                nearest_idx = tree_rail.nearest(centroid_geom)
                cbg_gdf.loc[idx, 'dist_nearest_rail'] = centroid_geom.distance(rail_stops_array[nearest_idx])
    
    # Count stops within buffers using spatial join
    print("  Counting stops within buffers...")
    try:
        # Create buffers
        centroids_400 = centroids.copy()
        centroids_400.geometry = centroids_400.geometry.buffer(400)
        centroids_800 = centroids.copy()
        centroids_800.geometry = centroids_800.geometry.buffer(800)
        
        # Count stops within buffers
        stops_in_400 = gpd.sjoin(centroids_400, all_stops, how='left', predicate='contains')
        stops_in_800 = gpd.sjoin(centroids_800, all_stops, how='left', predicate='contains')
        
        cbg_gdf['stops_400m'] = stops_in_400.groupby(stops_in_400.index).size()
        cbg_gdf['stops_800m'] = stops_in_800.groupby(stops_in_800.index).size()
        cbg_gdf['stops_400m'] = cbg_gdf['stops_400m'].fillna(0).astype(int)
        cbg_gdf['stops_800m'] = cbg_gdf['stops_800m'].fillna(0).astype(int)
    except Exception as e:
        print(f"  Warning: Could not compute buffer counts efficiently: {e}")
    
    print(f"  Completed metrics for {len(cbg_gdf)} block groups")
    
    return cbg_gdf


# ============================================================================
# 4. TRANSIT DESERT CLASSIFICATION
# ============================================================================

def classify_transit_deserts(cbg_gdf):
    """Define transit desert indicators."""
    # Binary indicators
    cbg_gdf['desert_800m'] = cbg_gdf['dist_nearest_stop'] > 800
    cbg_gdf['desert_rail_1600m'] = cbg_gdf['dist_nearest_rail'] > 1600
    
    # Combined with population density (if available)
    if 'total_pop' in cbg_gdf.columns and 'geometry' in cbg_gdf.columns:
        # Compute area in square meters, convert to km²
        cbg_gdf['area_km2'] = cbg_gdf.geometry.area / 1e6
        cbg_gdf['pop_density'] = cbg_gdf['total_pop'] / cbg_gdf['area_km2']
        
        # High population + far from transit = severe desert
        cbg_gdf['severe_desert'] = (
            (cbg_gdf['desert_800m']) & 
            (cbg_gdf['pop_density'] > cbg_gdf['pop_density'].median())
        )
    else:
        cbg_gdf['severe_desert'] = cbg_gdf['desert_800m']
    
    return cbg_gdf


# ============================================================================
# 5. VISUALIZATION FUNCTIONS
# ============================================================================

def plot_distance_to_nearest_stop(cbg_gdf, output_path):
    """Create map of distance to nearest transit stop."""
    # Project back to WGS84 for visualization
    cbg_vis = cbg_gdf.to_crs("EPSG:4326")
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plot with color scale
    cbg_vis.plot(
        ax=ax,
        column='dist_nearest_stop',
        cmap='YlOrRd',
        legend=True,
        legend_kwds={'label': 'Distance to Nearest Stop (meters)', 'shrink': 0.8},
        edgecolor='gray',
        linewidth=0.1,
        missing_kwds={'color': 'lightgray', 'label': 'No data'}
    )
    
    ax.set_title("Distance to Nearest Transit Stop\nLos Angeles County Census Block Groups", 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_transit_deserts(cbg_gdf, output_path):
    """Create map of transit desert classification."""
    # Project back to WGS84 for visualization
    cbg_vis = cbg_gdf.to_crs("EPSG:4326")
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Create binary classification map
    cbg_vis['desert_class'] = 'Not a Desert'
    cbg_vis.loc[cbg_vis['desert_800m'], 'desert_class'] = 'Transit Desert (>800m)'
    cbg_vis.loc[cbg_vis['severe_desert'], 'desert_class'] = 'Severe Desert'
    
    # Plot
    colors = {'Not a Desert': '#2ecc71', 'Transit Desert (>800m)': '#f39c12', 'Severe Desert': '#e74c3c'}
    for class_name, color in colors.items():
        subset = cbg_vis[cbg_vis['desert_class'] == class_name]
        if len(subset) > 0:
            subset.plot(ax=ax, color=color, edgecolor='gray', linewidth=0.1, label=class_name)
    
    ax.set_title("Transit Desert Classification\nLos Angeles County Census Block Groups", 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 60)
    print("LA Metro Transit Accessibility Analysis")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    try:
        cbg = load_census_block_groups()
        print(f"   Loaded {len(cbg)} Census Block Groups")
    except Exception as e:
        print(f"   Error loading CBGs: {e}")
        print("   Attempting to use Census Tracts as fallback...")
        # Fallback: use Census Tracts if Block Groups unavailable
        # This would require downloading tract shapefiles
        raise
    
    acs = load_acs_attributes()
    print(f"   Loaded {len(acs)} ACS attribute records")
    
    stops_gdf = load_transit_stops()
    print(f"   Loaded {len(stops_gdf)} transit stops ({len(stops_gdf[stops_gdf['mode']=='bus'])} bus, {len(stops_gdf[stops_gdf['mode']=='rail'])} rail)")
    
    # Prepare data
    print("\n2. Preparing data...")
    cbg_prepared = prepare_census_block_groups(cbg, acs)
    stops_prepared = prepare_transit_stops(stops_gdf)
    
    # Compute metrics
    print("\n3. Computing accessibility metrics...")
    cbg_with_metrics = compute_accessibility_metrics(cbg_prepared, stops_prepared)
    
    # Classify deserts
    print("\n4. Classifying transit deserts...")
    cbg_final = classify_transit_deserts(cbg_with_metrics)
    
    # Save results
    print("\n5. Saving results...")
    output_geojson = output_dir / "accessibility_metrics.geojson"
    
    # Drop centroid column (keep only main geometry) before saving
    cbg_to_save = cbg_final.drop(columns=['centroid']).to_crs("EPSG:4326")
    cbg_to_save.to_file(output_geojson, driver='GeoJSON')
    print(f"   Saved: {output_geojson}")
    
    # Create visualizations
    print("\n6. Creating visualizations...")
    plot_distance_to_nearest_stop(cbg_final, output_dir / "distance_to_nearest_stop.png")
    plot_transit_deserts(cbg_final, output_dir / "transit_deserts.png")
    
    # Summary statistics
    print("\n7. Summary Statistics:")
    print(f"   Total Block Groups: {len(cbg_final)}")
    print(f"   Mean distance to nearest stop: {cbg_final['dist_nearest_stop'].mean():.0f} meters")
    print(f"   Block groups >800m from transit: {cbg_final['desert_800m'].sum()} ({100*cbg_final['desert_800m'].sum()/len(cbg_final):.1f}%)")
    print(f"   Block groups >1600m from rail: {cbg_final['desert_rail_1600m'].sum()} ({100*cbg_final['desert_rail_1600m'].sum()/len(cbg_final):.1f}%)")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    
    # ========================================================================
    # PLACEHOLDER FOR PROJECT 28 SCENARIOS
    # ========================================================================
    # Future enhancement: Counterfactual analysis with Project 28 proposals
    # 
    # To add Project 28 scenarios:
    # 1. Load proposed stops/lines from GeoJSON:
    #    project28_stops = gpd.read_file("data/project28_proposed_stops.geojson")
    # 
    # 2. Combine with existing stops:
    #    all_stops_with_proposed = pd.concat([stops_gdf, project28_stops])
    # 
    # 3. Recompute accessibility metrics:
    #    cbg_proposed = compute_accessibility_metrics(cbg_prepared, all_stops_with_proposed)
    # 
    # 4. Compare baseline vs. proposed:
    #    - Compute difference in metrics
    #    - Identify newly served areas
    #    - Calculate population impact
    # 
    # 5. Generate counterfactual maps:
    #    - Before/after comparison maps
    #    - Change in transit desert classification
    #    - Population served by new infrastructure
    # ========================================================================


if __name__ == "__main__":
    main()

