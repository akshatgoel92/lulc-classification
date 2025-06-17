import os
import io
import base64
import tempfile
import time
import json
from typing import Optional
import numpy as np
import requests
import xml.etree.ElementTree as ET
from PIL import Image
from shapely.geometry import Polygon
from fastapi import FastAPI, HTTPException
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from rasterio.mask  import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pathlib import Path

app = FastAPI()

# Configuration for pre-stored TIFF files
class_mapping = {
    0: "Water",
    1: "Crop land", 
    2: "Barren",
    3: "Forest",
    4: "Built-up"
}

# Update these paths to match your Google Drive structure
DRIVE_ROOT = Path("/app/data")  # Adjust this path

lulc_tif_paths = {
    2013: DRIVE_ROOT / "LULC_2013_improved.tif",
    2014: DRIVE_ROOT / "LULC_2014_improved.tif", 
    2015: DRIVE_ROOT / "LULC_2015_improved.tif",
    2016: DRIVE_ROOT / "LULC_2016_improved.tif",
    2017: DRIVE_ROOT / "LULC_2017_improved.tif",
    2018: DRIVE_ROOT / "LULC_2018_improved.tif",
    2019: DRIVE_ROOT / "LULC_2019_improved_sentinel.tif",
    2020: DRIVE_ROOT / "LULC_2020_improved_sentinel.tif", 
    2021: DRIVE_ROOT / "LULC_2021_improved_sentinel.tif",
    2022: DRIVE_ROOT / "LULC_2022_improved_sentinel.tif",
    2023: DRIVE_ROOT / "LULC_2023_improved_sentinel.tif",
    2024: DRIVE_ROOT / "LULC_2024_improved_sentinel.tif"
}

# Global storage for preloaded TIFF data
loaded_tiffs = {}

def load_all_tiffs_into_memory():
    """Load all available TIFF files into memory at startup"""
    global loaded_tiffs
    print("🚀 Loading all TIFF files into memory...")
    
    for year, tif_path in lulc_tif_paths.items():
        if tif_path.exists():
            try:
                print(f"📂 Loading {tif_path}...")
                with rasterio.open(tif_path) as src:
                    # Store both data and metadata
                    loaded_tiffs[year] = {
                        'data': src.read(1),  # Read first band
                        'transform': src.transform,
                        'crs': src.crs,
                        'shape': src.shape,
                        'bounds': src.bounds,
                        'nodata': src.nodata
                    }
                print(f"✅ Loaded {year}: shape={loaded_tiffs[year]['shape']}, CRS={loaded_tiffs[year]['crs']}")
            except Exception as e:
                print(f"❌ Failed to load {tif_path}: {e}")
        else:
            print(f"⚠️ TIFF file not found: {tif_path}")
    
    print(f"🎯 Successfully loaded {len(loaded_tiffs)} TIFF files into memory")
    return loaded_tiffs

def parse_kml_coordinates(kml_content: str):
    """Parse KML content and extract coordinates"""
    try:
        root = ET.fromstring(kml_content)
        
        # Handle different KML namespaces
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}
        
        # Try to find coordinates
        coords_elem = root.find('.//kml:coordinates', ns)
        if coords_elem is None:
            # Try without namespace
            coords_elem = root.find('.//coordinates')
        
        if coords_elem is not None:
            coords_text = coords_elem.text.strip()
            coordinates = []
            
            for coord in coords_text.split():
                parts = coord.split(',')
                if len(parts) >= 2:
                    lon, lat = float(parts[0]), float(parts[1])
                    coordinates.append((lon, lat))
            
            return coordinates
        else:
            print("❌ No coordinates found in KML")
            return None
            
    except Exception as e:
        print(f"❌ Error parsing KML: {e}")
        return None

def get_stats(masked_lulc, year):
    """Calculate LULC statistics for a given masked array"""
    try:
        # Remove NaN values for calculation
        valid_data = masked_lulc[~np.isnan(masked_lulc)]
        
        if len(valid_data) == 0:
            return {"error": "No valid data found in the region"}
        
        # Count pixels for each class
        unique_values, counts = np.unique(valid_data, return_counts=True)
        
        total_pixels = len(valid_data)
        
        # Calculate pixel area (assuming 30m resolution for Landsat)
        pixel_area_m2 = 30 * 30  # 900 m² per pixel
        total_area_m2 = total_pixels * pixel_area_m2
        
        # Build class statistics
        classes = []
        for value, count in zip(unique_values, counts):
            class_name = class_mapping.get(int(value), f"Class {int(value)}")
            area_m2 = count * pixel_area_m2
            percentage = (count / total_pixels) * 100
            
            classes.append({
                "class_value": int(value),
                "class_name": class_name,
                "pixel_count": int(count),
                "area_m2": float(area_m2),
                "percentage": float(percentage)
            })
        
        return {
            "year": year,
            "total_pixels": int(total_pixels),
            "total_area_m2": float(total_area_m2),
            "classes": classes
        }
        
    except Exception as e:
        print(f"❌ Error calculating statistics: {e}")
        return {"error": str(e)}

def get_lulc_analysis_from_memory(kml_content: str, year: int):
    """Get LULC analysis from preloaded TIFF data"""
    try:
        # Check if data is available
        if year not in loaded_tiffs:
            print(f"❌ No TIFF data available for year {year}")
            return {"error": f"No data available for year {year}"}
        
        tiff_data = loaded_tiffs[year]
        
        # Parse KML and extract geometry
        coords = parse_kml_coordinates(kml_content)
        if not coords:
            print("❌ Failed to parse KML coordinates")
            return {"error": "Failed to parse KML coordinates"}
        
        # Create GeoDataFrame from coordinates
        polygon = Polygon(coords)
        gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs="EPSG:4326")
        
        print(f"📊 Analyzing LULC for year {year}")
        
        # Use the preloaded data
        lulc_data = tiff_data['data']
        lulc_transform = tiff_data['transform']
        gdf = gdf.to_crs(tiff_data['crs'])

        # Create mask
        mask = geometry_mask(
            geometries=gdf.geometry,
            transform=lulc_transform,
            invert=True,
            out_shape=lulc_data.shape
        )

        # Apply mask
        masked_lulc = np.where(mask, lulc_data, np.nan)
        
        # Calculate statistics
        stats = get_stats(masked_lulc, year)
        return stats
                
    except Exception as e:
        print(f"❌ Error in LULC analysis: {e}")
        return {"error": str(e)}

# Load TIFFs at startup
@app.on_event("startup")
async def startup_event():
    load_all_tiffs_into_memory()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "loaded_tiffs": list(loaded_tiffs.keys()),
        "tiff_count": len(loaded_tiffs),
        "memory_usage": "TIFFs preloaded in memory",
        "available_years": sorted(loaded_tiffs.keys())
    }

@app.get("/available_years")
async def get_available_years():
    """Get list of available years for analysis"""
    available_years = sorted(loaded_tiffs.keys())
    return {
        "success": True,
        "years": available_years,
        "count": len(available_years)
    }

@app.post("/lulc_analysis")
async def get_lulc_analysis(request: dict):
    """Get LULC analysis for a KML polygon and year"""
    try:
        kml_content = request.get("kml_content")
        year = request.get("year")
        
        if not kml_content or not year:
            raise HTTPException(status_code=400, detail="Missing kml_content or year")
        
        analysis = get_lulc_analysis_from_memory(kml_content, year)
        
        if "error" not in analysis:
            return {
                "success": True,
                "message": f"Successfully analyzed LULC for {year}",
                "data": analysis
            }
        else:
            return {
                "success": False,
                "message": f"Failed to analyze LULC for {year}",
                "data": analysis
            }
            
    except Exception as e:
        print(f"❌ Error in lulc_analysis endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/multi_year_analysis")
async def get_multi_year_analysis(request: dict):
    """Get multi-year LULC trend analysis"""
    try:
        kml_content = request.get("kml_content")
        start_year = request.get("start_year")
        end_year = request.get("end_year")
        
        if not kml_content or not start_year or not end_year:
            raise HTTPException(status_code=400, detail="Missing kml_content, start_year, or end_year")
        
        if start_year > end_year:
            raise HTTPException(status_code=400, detail="start_year must be less than or equal to end_year")
        
        results = []
        years_analyzed = []
        print(results)
        for year in range(start_year, end_year + 1):
            if year in loaded_tiffs:
                print(f"📊 Analyzing year {year}...")
                analysis = get_lulc_analysis_from_memory(kml_content, year)
                
                if "error" not in analysis:
                    results.append(analysis)
                    years_analyzed.append(year)
                else:
                    print(f"⚠️ Skipping year {year}: {analysis['error']}")
            else:
                print(f"⚠️ No data available for year {year}")
        
        if len(results) > 0:
            return {
                "success": True,
                "message": f"Successfully analyzed LULC trends from {start_year} to {end_year}",
                "data": results,
                "years_analyzed": years_analyzed,
                "total_years": len(results)
            }
        else:
            return {
                "success": False,
                "message": f"No data available for the requested year range ({start_year}-{end_year})",
                "data": [],
                "years_analyzed": [],
                "total_years": 0
            }
            
    except Exception as e:
        print(f"❌ Error in multi_year_analysis endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))