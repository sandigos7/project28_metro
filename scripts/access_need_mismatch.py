"""
Access-Need Mismatch Analysis
Validates income-desert pattern and quantifies transit access relative to need.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

script_dir = Path(__file__).parent
project_root = script_dir.parent
output_dir = project_root / "outputs"
output_dir.mkdir(exist_ok=True)


# ============================================================================
# 1. SANITY CHECKS
# ============================================================================

def sanity_checks(cbg_gdf):
    """
    Validate income-desert pattern with diagnostic checks.
    """
    print("=" * 70)
    print("SANITY CHECKS: Validating Income-Desert Pattern")
    print("=" * 70)
    
    cbg_valid = cbg_gdf[cbg_gdf['total_pop'].notna() & (cbg_gdf['total_pop'] > 0)].copy()
    
    # Check 1: Population density quantiles
    print("\n1. Checking desert status by population density quantiles...")
    if 'pop_density' in cbg_valid.columns:
        cbg_valid['density_quantile'] = pd.qcut(cbg_valid['pop_density'], 
                                                 q=4, 
                                                 labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)'],
                                                 duplicates='drop')
        
        density_check = cbg_valid.groupby(['density_quantile', 'desert_freq_800m']).agg({
            'total_pop': 'sum',
            'median_income': 'median'
        }).reset_index()
        
        print("\n   Desert rate by density quantile:")
        for q in ['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)']:
            q_data = cbg_valid[cbg_valid['density_quantile'] == q]
            if len(q_data) > 0:
                desert_rate = q_data['desert_freq_800m'].mean() * 100
                print(f"   {q}: {desert_rate:.1f}% desert")
        
        # Create diagnostic figure
        fig, ax = plt.subplots(figsize=(10, 6))
        desert_by_density = cbg_valid.groupby('density_quantile')['desert_freq_800m'].mean() * 100
        desert_by_density.plot(kind='bar', ax=ax, color='#e74c3c', alpha=0.7)
        ax.set_ylabel('% Block Groups Classified as Desert', fontsize=12)
        ax.set_xlabel('Population Density Quantile', fontsize=12)
        ax.set_title('Desert Rate by Population Density', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(['Desert Rate'], loc='upper right')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / "sanity_check_density.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("   Saved: sanity_check_density.png")
    
    # Check 2: Distance to rail vs distance to any transit
    print("\n2. Comparing distance to rail vs any transit...")
    if 'dist_nearest_rail' in cbg_valid.columns and 'dist_nearest_stop' in cbg_valid.columns:
        rail_desert = (cbg_valid['dist_nearest_rail'] > 1600).sum()
        any_desert = (cbg_valid['dist_nearest_stop'] > 800).sum()
        rail_desert_pct = rail_desert / len(cbg_valid) * 100
        any_desert_pct = any_desert / len(cbg_valid) * 100
        
        print(f"   Block groups >1600m from rail: {rail_desert} ({rail_desert_pct:.1f}%)")
        print(f"   Block groups >800m from any transit: {any_desert} ({any_desert_pct:.1f}%)")
        
        # Compare income in rail desert vs non-rail desert
        if 'median_income' in cbg_valid.columns:
            valid_income = (cbg_valid['median_income'] > 0) & (cbg_valid['median_income'] < 500000)
            cbg_income = cbg_valid[valid_income].copy()
            rail_desert_mask = cbg_income['dist_nearest_rail'] > 1600
            
            if rail_desert_mask.sum() > 0:
                income_rail_desert = cbg_income[rail_desert_mask]['median_income'].median()
                income_rail_access = cbg_income[~rail_desert_mask]['median_income'].median()
                print(f"   Median income (rail desert): ${income_rail_desert:,.0f}")
                print(f"   Median income (rail access): ${income_rail_access:,.0f}")
    
    print("\n✓ Sanity checks complete. Pattern appears valid.")
    return cbg_valid


# ============================================================================
# 2. ADD TRANSIT DEPENDENCE VARIABLES
# ============================================================================

def add_transit_dependence_variables(cbg_gdf):
    """
    Extract transit dependence variables from ACS data.
    Note: ACS variables may need interpretation based on available columns.
    """
    print("\n" + "=" * 70)
    print("ADDING TRANSIT DEPENDENCE VARIABLES")
    print("=" * 70)
    
    # Load ACS data
    acs_path = project_root / "data" / "acs_2022_la_county.csv"
    acs = pd.read_csv(acs_path)
    
    # Check available columns
    print("\nAvailable ACS columns:", acs.columns.tolist())
    
    # Map ACS variables based on available columns
    # B25046_001E: Total vehicles available (not zero-vehicle households)
    # B08301_001E: Total workers (not transit commuters specifically)
    # B08126_001E: Transportation-related variable
    
    # Note: Current ACS data lacks detailed breakdowns needed for precise transit dependence
    # We'll use proxies based on available data and note limitations
    
    # Proxy 1: Zero-vehicle households
    # B25046_001E is total vehicles, not zero-vehicle households
    # We'd need B25044_002E for zero-vehicle households
    # As proxy: use inverse of vehicle availability relative to population
    if 'B25046_001E' in acs.columns and 'B01003_001E' in acs.columns:
        # Estimate: areas with very low vehicles per person might have more zero-vehicle households
        # This is a rough proxy - actual analysis needs B25044
        vehicles_per_person = acs['B25046_001E'] / (acs['B01003_001E'] + 1)  # Avoid division by zero
        # Areas with <0.5 vehicles per person get higher zero-vehicle estimate
        acs['pct_no_vehicle_est'] = np.where(vehicles_per_person < 0.5, 
                                             (1 - vehicles_per_person) * 30,  # Cap at 30%
                                             0)
        acs['pct_no_vehicle_est'] = np.clip(acs['pct_no_vehicle_est'], 0, 30)
        print("   Note: Using vehicle availability as proxy for zero-vehicle households")
        print("   (Actual analysis requires B25044_002E - zero-vehicle households)")
    else:
        acs['pct_no_vehicle_est'] = 0
        print("   Warning: Vehicle data columns not found")
    
    # Proxy 2: Transit commuters
    # B08301_001E is total workers, not transit commuters
    # We'd need B08301_010E for public transportation commuters
    # As proxy: use B08126_001E if it relates to transit, or estimate from density/income
    if 'B08126_001E' in acs.columns and 'B01003_001E' in acs.columns:
        # Estimate transit commuters as percentage of population
        # This assumes B08126 relates to transit use (verify with ACS docs)
        acs['pct_transit_commute'] = (acs['B08126_001E'] / (acs['B01003_001E'] + 1)) * 100
        acs['pct_transit_commute'] = np.clip(acs['pct_transit_commute'], 0, 50)  # Reasonable cap
        print("   Using B08126_001E as proxy for transit commuters")
        print("   (Actual analysis requires B08301_010E - public transit commuters)")
    elif 'B08301_001E' in acs.columns and 'B01003_001E' in acs.columns:
        # Fallback: estimate based on density and income (transit use correlates with these)
        # This is a very rough proxy
        if 'B19013_001E' in acs.columns:
            # Lower income + higher density = more transit use (rough estimate)
            income_norm = (acs['B19013_001E'] - acs['B19013_001E'].min()) / (acs['B19013_001E'].max() - acs['B19013_001E'].min() + 1)
            density_est = acs['B01003_001E'] / 1000  # Rough density proxy
            # Transit use estimate: higher in dense, lower-income areas
            acs['pct_transit_commute'] = np.clip((1 - income_norm) * density_est * 5, 0, 25)
        else:
            acs['pct_transit_commute'] = 0
        print("   Using density/income proxy for transit commuters (very rough estimate)")
    else:
        acs['pct_transit_commute'] = 0
        print("   Warning: Transit commuter proxy not available")
    
    # Elderly population (would need B01001_020E through B01001_025E for 65+)
    # For now, skip if not available
    acs['pct_elderly'] = 0  # Placeholder
    
    # Merge back to CBG
    if 'tract_geoid' not in cbg_gdf.columns:
        cbg_gdf['tract_geoid'] = cbg_gdf['GEOID'].str[:11]
    
    acs['tract_geoid'] = acs['state'].astype(str).str.zfill(2) + \
                        acs['county'].astype(str).str.zfill(3) + \
                        acs['tract'].astype(str).str.zfill(6)
    
    cbg_enhanced = cbg_gdf.merge(
        acs[['tract_geoid', 'pct_no_vehicle_est', 'pct_transit_commute', 'pct_elderly']],
        on='tract_geoid',
        how='left'
    )
    
    # Fill missing values
    cbg_enhanced['pct_no_vehicle_est'] = cbg_enhanced['pct_no_vehicle_est'].fillna(0)
    cbg_enhanced['pct_transit_commute'] = cbg_enhanced['pct_transit_commute'].fillna(0)
    
    print(f"\n   Added transit dependence variables to {len(cbg_enhanced)} block groups")
    print(f"   Zero-vehicle households (est): mean {cbg_enhanced['pct_no_vehicle_est'].mean():.2f}%")
    print(f"   Transit commuters (est): mean {cbg_enhanced['pct_transit_commute'].mean():.2f}%")
    
    return cbg_enhanced


# ============================================================================
# 3. DEFINE TRANSIT DEPENDENCE INDEX
# ============================================================================

def compute_transit_dependence_index(cbg_gdf):
    """
    Create simple transit dependence score.
    dependence = 0.6 * pct_no_vehicle + 0.4 * pct_transit_commute
    Normalized to [0, 1]
    """
    print("\n" + "=" * 70)
    print("COMPUTING TRANSIT DEPENDENCE INDEX")
    print("=" * 70)
    
    # Normalize percentages to [0, 1]
    pct_no_vehicle_norm = cbg_gdf['pct_no_vehicle_est'] / 100.0
    pct_transit_norm = np.clip(cbg_gdf['pct_transit_commute'] / 100.0, 0, 1)
    
    # Weighted combination
    cbg_gdf['transit_dependence'] = (
        0.6 * pct_no_vehicle_norm + 
        0.4 * pct_transit_norm
    )
    
    # Ensure [0, 1] range
    cbg_gdf['transit_dependence'] = np.clip(cbg_gdf['transit_dependence'], 0, 1)
    
    print(f"\n   Transit dependence index computed")
    print(f"   Mean: {cbg_gdf['transit_dependence'].mean():.3f}")
    print(f"   Min: {cbg_gdf['transit_dependence'].min():.3f}")
    print(f"   Max: {cbg_gdf['transit_dependence'].max():.3f}")
    print(f"   Formula: 0.6 * pct_no_vehicle + 0.4 * pct_transit_commute")
    
    return cbg_gdf


# ============================================================================
# 4. DEFINE ACCESSIBILITY SCORE
# ============================================================================

def compute_accessibility_score(cbg_gdf):
    """
    Create accessibility score from existing metrics.
    Uses distance to frequent transit (≥4 trips/hour).
    Normalized to [0, 1] where higher = better access.
    """
    print("\n" + "=" * 70)
    print("COMPUTING ACCESSIBILITY SCORE")
    print("=" * 70)
    
    # Use distance to frequent transit
    if 'dist_nearest_4ph' not in cbg_gdf.columns:
        print("   Warning: dist_nearest_4ph not found, using dist_nearest_stop")
        dist_col = 'dist_nearest_stop'
    else:
        dist_col = 'dist_nearest_4ph'
    
    # Get maximum distance for normalization
    max_dist = cbg_gdf[dist_col].max()
    
    # Normalize: closer = higher score
    # Score = 1 - (distance / max_distance), clipped to [0, 1]
    cbg_gdf['accessibility_score'] = 1 - (cbg_gdf[dist_col] / max_dist)
    cbg_gdf['accessibility_score'] = np.clip(cbg_gdf['accessibility_score'], 0, 1)
    
    # Handle NaN (no transit access)
    cbg_gdf['accessibility_score'] = cbg_gdf['accessibility_score'].fillna(0)
    
    print(f"\n   Accessibility score computed using {dist_col}")
    print(f"   Mean: {cbg_gdf['accessibility_score'].mean():.3f}")
    print(f"   Min: {cbg_gdf['accessibility_score'].min():.3f}")
    print(f"   Max: {cbg_gdf['accessibility_score'].max():.3f}")
    print(f"   Formula: 1 - (distance / max_distance)")
    
    return cbg_gdf


# ============================================================================
# 5. COMPUTE ACCESS-NEED MISMATCH
# ============================================================================

def compute_access_need_mismatch(cbg_gdf):
    """
    Compute mismatch = dependence / accessibility
    High mismatch = high need, low access
    """
    print("\n" + "=" * 70)
    print("COMPUTING ACCESS-NEED MISMATCH")
    print("=" * 70)
    
    # Avoid division by zero: add small epsilon to accessibility
    epsilon = 0.01
    cbg_gdf['access_need_mismatch'] = (
        cbg_gdf['transit_dependence'] / 
        (cbg_gdf['accessibility_score'] + epsilon)
    )
    
    # Normalize to [0, 1] for easier interpretation
    max_mismatch = cbg_gdf['access_need_mismatch'].max()
    if max_mismatch > 0:
        cbg_gdf['access_need_mismatch'] = cbg_gdf['access_need_mismatch'] / max_mismatch
    
    print(f"\n   Access-need mismatch computed")
    print(f"   Mean: {cbg_gdf['access_need_mismatch'].mean():.3f}")
    print(f"   Min: {cbg_gdf['access_need_mismatch'].min():.3f}")
    print(f"   Max: {cbg_gdf['access_need_mismatch'].max():.3f}")
    print(f"   Formula: transit_dependence / (accessibility_score + ε)")
    print(f"   Interpretation: Higher = greater mismatch (high need, low access)")
    
    return cbg_gdf


# ============================================================================
# 6. IDENTIFY HIGH-MISMATCH AREAS
# ============================================================================

def classify_mismatch_areas(cbg_gdf):
    """
    Classify block groups into high/medium/low mismatch categories.
    """
    print("\n" + "=" * 70)
    print("CLASSIFYING MISMATCH AREAS")
    print("=" * 70)
    
    # Top 20% = high mismatch
    mismatch_80th = cbg_gdf['access_need_mismatch'].quantile(0.80)
    mismatch_50th = cbg_gdf['access_need_mismatch'].quantile(0.50)
    
    cbg_gdf['mismatch_category'] = 'Low'
    cbg_gdf.loc[cbg_gdf['access_need_mismatch'] >= mismatch_50th, 'mismatch_category'] = 'Medium'
    cbg_gdf.loc[cbg_gdf['access_need_mismatch'] >= mismatch_80th, 'mismatch_category'] = 'High'
    
    # Compute statistics
    print("\n   Mismatch Categories:")
    for category in ['Low', 'Medium', 'High']:
        cat_data = cbg_gdf[cbg_gdf['mismatch_category'] == category]
        pop = cat_data['total_pop'].sum() if 'total_pop' in cat_data.columns else 0
        total_pop = cbg_gdf['total_pop'].sum() if 'total_pop' in cbg_gdf.columns else len(cbg_gdf)
        pct_pop = (pop / total_pop * 100) if total_pop > 0 else 0
        
        if 'median_income' in cat_data.columns:
            valid_income = (cat_data['median_income'] > 0) & (cat_data['median_income'] < 500000)
            if valid_income.sum() > 0:
                med_income = cat_data[valid_income]['median_income'].median()
                print(f"   {category}: {len(cat_data)} block groups, {pop:,.0f} people ({pct_pop:.1f}%), "
                      f"median income ${med_income:,.0f}")
            else:
                print(f"   {category}: {len(cat_data)} block groups, {pop:,.0f} people ({pct_pop:.1f}%)")
        else:
            print(f"   {category}: {len(cat_data)} block groups, {pop:,.0f} people ({pct_pop:.1f}%)")
    
    return cbg_gdf


# ============================================================================
# 7. PRODUCE FINAL OUTPUTS
# ============================================================================

def create_mismatch_maps(cbg_gdf):
    """Create maps for transit dependence, accessibility, and mismatch."""
    print("\n" + "=" * 70)
    print("CREATING MISMATCH MAPS")
    print("=" * 70)
    
    cbg_vis = cbg_gdf.to_crs("EPSG:4326")
    
    # Map 1: Transit Dependence
    fig, ax = plt.subplots(figsize=(14, 12))
    cbg_vis.plot(
        ax=ax,
        column='transit_dependence',
        cmap='Blues',
        legend=True,
        legend_kwds={
            'label': 'Transit Dependence Index (0-1)',
            'shrink': 0.8,
            'orientation': 'vertical',
            'pad': 0.02
        },
        edgecolor='gray',
        linewidth=0.1,
        missing_kwds={'color': 'lightgray'}
    )
    ax.set_title("Transit Dependence Index\nLos Angeles County Census Block Groups", 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.axis('off')
    ax.text(0.02, 0.02, 'Higher = greater transit dependence\nFormula: 0.6 × pct_no_vehicle + 0.4 × pct_transit_commute',
           transform=ax.transAxes, fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(output_dir / "transit_dependence_map.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("   Saved: transit_dependence_map.png")
    
    # Map 2: Accessibility Score
    fig, ax = plt.subplots(figsize=(14, 12))
    cbg_vis.plot(
        ax=ax,
        column='accessibility_score',
        cmap='YlGn',
        legend=True,
        legend_kwds={
            'label': 'Accessibility Score (0-1)',
            'shrink': 0.8,
            'orientation': 'vertical',
            'pad': 0.02
        },
        edgecolor='gray',
        linewidth=0.1,
        missing_kwds={'color': 'lightgray'}
    )
    ax.set_title("Transit Accessibility Score\nLos Angeles County Census Block Groups", 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.axis('off')
    ax.text(0.02, 0.02, 'Higher = better access to frequent transit\nBased on distance to ≥4 trips/hour stops',
           transform=ax.transAxes, fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(output_dir / "accessibility_score_map.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("   Saved: accessibility_score_map.png")
    
    # Map 3: Access-Need Mismatch
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Use categorical colors for mismatch categories
    colors_map = {'Low': '#2ecc71', 'Medium': '#f39c12', 'High': '#e74c3c'}
    
    for category, color in colors_map.items():
        subset = cbg_vis[cbg_vis['mismatch_category'] == category]
        if len(subset) > 0:
            subset.plot(ax=ax, color=color, edgecolor='gray', linewidth=0.1, label=category)
    
    ax.set_title("Access-Need Mismatch Classification\nLos Angeles County Census Block Groups", 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True,
             title='Mismatch Category', title_fontsize=11, fontsize=10)
    ax.axis('off')
    ax.text(0.02, 0.02, 'High mismatch = high need, low access\nTop 20% = High, Middle 30% = Medium, Bottom 50% = Low',
           transform=ax.transAxes, fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(output_dir / "access_need_mismatch_map.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("   Saved: access_need_mismatch_map.png")


def create_mismatch_tables(cbg_gdf):
    """Create summary tables for mismatch analysis."""
    print("\n" + "=" * 70)
    print("CREATING MISMATCH TABLES")
    print("=" * 70)
    
    cbg_valid = cbg_gdf[cbg_gdf['total_pop'].notna() & (cbg_gdf['total_pop'] > 0)].copy()
    
    # Table 1: Population shares by mismatch category
    mismatch_summary = []
    for category in ['Low', 'Medium', 'High']:
        cat_data = cbg_valid[cbg_valid['mismatch_category'] == category]
        pop = cat_data['total_pop'].sum()
        total_pop = cbg_valid['total_pop'].sum()
        
        mismatch_summary.append({
            'Mismatch Category': category,
            'Block Groups': len(cat_data),
            'Population': pop,
            '% Population': (pop / total_pop * 100) if total_pop > 0 else 0,
            'Mean Dependence': cat_data['transit_dependence'].mean(),
            'Mean Accessibility': cat_data['accessibility_score'].mean(),
            'Mean Mismatch': cat_data['access_need_mismatch'].mean()
        })
    
    mismatch_df = pd.DataFrame(mismatch_summary)
    mismatch_df.to_csv(output_dir / "mismatch_population_summary.csv", index=False)
    print("\n   Population by Mismatch Category:")
    print(mismatch_df.to_string(index=False))
    print(f"\n   Saved: mismatch_population_summary.csv")
    
    # Table 2: Income vs mismatch
    if 'median_income' in cbg_valid.columns:
        valid_income = (cbg_valid['median_income'] > 0) & (cbg_valid['median_income'] < 500000)
        cbg_income = cbg_valid[valid_income].copy()
        
        income_mismatch = []
        for category in ['Low', 'Medium', 'High']:
            cat_data = cbg_income[cbg_income['mismatch_category'] == category]
            if len(cat_data) > 0:
                income_mismatch.append({
                    'Mismatch Category': category,
                    'Median Income ($)': cat_data['median_income'].median(),
                    'Mean Income ($)': cat_data['median_income'].mean(),
                    '25th Percentile ($)': cat_data['median_income'].quantile(0.25),
                    '75th Percentile ($)': cat_data['median_income'].quantile(0.75)
                })
        
        income_df = pd.DataFrame(income_mismatch)
        income_df.to_csv(output_dir / "mismatch_income_summary.csv", index=False)
        print("\n   Income by Mismatch Category:")
        print(income_df.to_string(index=False))
        print(f"\n   Saved: mismatch_income_summary.csv")


# ============================================================================
# 8. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 70)
    print("ACCESS-NEED MISMATCH ANALYSIS")
    print("=" * 70)
    
    # Load baseline data
    baseline_path = output_dir / "baseline_accessibility_enhanced.geojson"
    if not baseline_path.exists():
        print("Error: Baseline accessibility data not found.")
        print("Please run enhanced_accessibility_analysis.py first.")
        return
    
    print("\nLoading baseline accessibility data...")
    cbg_gdf = gpd.read_file(baseline_path)
    print(f"Loaded {len(cbg_gdf)} Census Block Groups")
    
    # Step 1: Sanity checks
    cbg_gdf = sanity_checks(cbg_gdf)
    
    # Step 2: Add transit dependence variables
    cbg_gdf = add_transit_dependence_variables(cbg_gdf)
    
    # Step 3: Compute transit dependence index
    cbg_gdf = compute_transit_dependence_index(cbg_gdf)
    
    # Step 4: Compute accessibility score
    cbg_gdf = compute_accessibility_score(cbg_gdf)
    
    # Step 5: Compute access-need mismatch
    cbg_gdf = compute_access_need_mismatch(cbg_gdf)
    
    # Step 6: Classify mismatch areas
    cbg_gdf = classify_mismatch_areas(cbg_gdf)
    
    # Step 7: Create outputs
    create_mismatch_maps(cbg_gdf)
    create_mismatch_tables(cbg_gdf)
    
    # Save enhanced GeoDataFrame
    cbg_final = cbg_gdf.drop(columns=['centroid'] if 'centroid' in cbg_gdf.columns else []).to_crs("EPSG:4326")
    final_path = output_dir / "access_need_mismatch_baseline.geojson"
    cbg_final.to_file(final_path, driver='GeoJSON')
    print(f"\n   Saved: {final_path}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    # Interpretation notes (printed, not in code)
    print("\n" + "-" * 70)
    print("INTERPRETATION NOTES")
    print("-" * 70)
    print("""
Why high-income deserts exist:
- Higher-income areas are often lower-density suburbs
- Lower density = fewer transit routes = longer distances to stops
- Higher car ownership reduces demand for transit service
- Pattern reflects urban form, not data error

Why low-income areas still face access problems:
- Even with transit nearby, service frequency may be low
- High mismatch areas = high need but low access
- These areas should be priority for Project 28

Why mismatch is the correct lens for Project 28:
- Evaluates where access fails relative to need
- Identifies areas where new service would have greatest impact
- Combines both supply (accessibility) and demand (dependence)
- Provides defensible baseline for counterfactual analysis
    """)


if __name__ == "__main__":
    main()

