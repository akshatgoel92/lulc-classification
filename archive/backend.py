from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import xml.etree.ElementTree as ET
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
import requests
from typing import List, Tuple, Optional
import tempfile
import os
from datetime import datetime, timedelta
import base64
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import ee
import geemap

app = FastAPI(title="Satellite Imagery API")



# Initialize Google Earth Engine
def initialize_ee():
    """Initialize Google Earth Engine with service account or interactive auth."""
    try:
        # Get project ID from environment or use default
        project_id = os.getenv('GOOGLE_CLOUD_PROJECT') or os.getenv('GEE_PROJECT')
        # Option 1: Try service account authentication
        service_account_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        
        if service_account_path and os.path.exists(service_account_path):
            print(f"Using service account: {service_account_path}")
            # Read the service account file to get the email and project
            import json
            with open(service_account_path, 'r') as f:
                service_account_info = json.load(f)
            
            # 
            service_account_email = service_account_info['client_email']
            project_id = service_account_info.get('project_id', project_id)
            
            # Create credentials object
            credentials = ee.ServiceAccountCredentials(service_account_email, service_account_path)
            ee.Initialize(credentials, project=project_id)
            print(f"Earth Engine initialized with service account, project: {project_id}")
            return True
            
        # Option 2: Try existing user credentials with project
        else:
            print(f"Trying user authentication with project: {project_id}")
            ee.Initialize(project=project_id)
            print(f"Earth Engine initialized with user credentials, project: {project_id}")
            return True
            
    except Exception as e:
        error_msg = str(e).lower()
        if "serviceusage.services.use" in error_msg or "serviceusageConsumer" in error_msg:
            print("❌ Permission Error: Service account needs 'Service Usage Consumer' role")
            print("Solutions:")
            print("1. Go to: https://console.developers.google.com/iam-admin/iam/project?project=amiable-raceway-342517")
            print("2. Find your service account and add 'Service Usage Consumer' role")
            print("3. Or use personal authentication: earthengine authenticate --project=amiable-raceway-342517")
            print("4. Or enable required APIs: https://console.developers.google.com/apis/library?project=amiable-raceway-342517")
        else:
            print(f"Earth Engine initialization failed: {e}")
            print("Please either:")
            print("1. Set GOOGLE_APPLICATION_CREDENTIALS environment variable")
            print("2. Set GOOGLE_CLOUD_PROJECT or GEE_PROJECT environment variable")
            print("3. Run 'earthengine authenticate --project=amiable-raceway-342517' for user authentication")
            print("4. Check IAM permissions for your service account")
        return False

# Global variable to track EE initialization
EE_INITIALIZED = initialize_ee()

class ImageryRequest(BaseModel):
    kml_content: str
    start_date: str = "2023-01-01"
    end_date: str = "2023-12-31"
    satellite: str = "sentinel-2"  # or "landsat-8"

def parse_kml_polygon(kml_content: str) -> Polygon:
    """Parse KML content and extract polygon coordinates."""
    try:
        # Check if content is empty
        if not kml_content or not kml_content.strip():
            raise ValueError("KML content is empty")
        
        # Remove BOM if present
        if kml_content.startswith('\ufeff'):
            kml_content = kml_content[1:]
        
        # Basic validation
        kml_content = kml_content.strip()
        if not (kml_content.startswith('<?xml') or kml_content.startswith('<kml')):
            raise ValueError("Content doesn't appear to be valid XML/KML")
        
        # Parse XML
        try:
            root = ET.fromstring(kml_content)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML structure: {str(e)}")
        
        # Find coordinates in the KML - try multiple approaches
        coords_element = None
        
        # Method 1: Look for any coordinates element
        for elem in root.iter():
            if elem.tag.endswith('coordinates'):
                coords_element = elem
                break
        
        # Method 2: Try specific namespace patterns
        if coords_element is None:
            for elem in root.iter():
                if 'coordinates' in elem.tag.lower():
                    coords_element = elem
                    break
        
        # Method 3: Look in common KML structure
        if coords_element is None:
            # Try kml/Document/Placemark/Polygon/outerBoundaryIs/LinearRing/coordinates
            for placemark in root.iter():
                if placemark.tag.endswith('Placemark'):
                    for polygon in placemark.iter():
                        if polygon.tag.endswith('Polygon'):
                            for elem in polygon.iter():
                                if elem.tag.endswith('coordinates'):
                                    coords_element = elem
                                    break
                    if coords_element is not None:
                        break
        
        if coords_element is None:
            # Debug: Print available elements
            available_tags = [elem.tag for elem in root.iter()]
            raise ValueError(f"No coordinates found in KML. Available elements: {available_tags[:10]}")
        
        # Parse coordinates
        coords_text = coords_element.text
        if not coords_text:
            raise ValueError("Coordinates element is empty")
        
        coords_text = coords_text.strip()
        coords_list = []
        
        # Handle different coordinate formats
        for coord_line in coords_text.split('\n'):
            coord_line = coord_line.strip()
            if not coord_line:
                continue
                
            for coord in coord_line.split():
                coord = coord.strip()
                if not coord:
                    continue
                    
                parts = coord.split(',')
                if len(parts) >= 2:
                    try:
                        lon = float(parts[0])
                        lat = float(parts[1])
                        coords_list.append((lon, lat))
                    except ValueError:
                        continue
        
        if len(coords_list) < 3:
            raise ValueError(f"Not enough valid coordinates found. Got {len(coords_list)}, need at least 3")
        
        # Ensure polygon is closed
        if coords_list[0] != coords_list[-1]:
            coords_list.append(coords_list[0])
        
        return Polygon(coords_list)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing KML: {str(e)}")

def validate_polygon_for_imagery(polygon: Polygon) -> dict:
    """Validate if polygon is suitable for satellite imagery download."""
    bounds = polygon.bounds
    width = abs(bounds[2] - bounds[0])  # longitude
    height = abs(bounds[3] - bounds[1])  # latitude
    area = polygon.area
    
    warnings = []
    
    # Check size
    if width > 5 or height > 5:
        warnings.append("Area is very large - this may cause download failures")
    elif width < 0.001 or height < 0.001:
        warnings.append("Area is very small - imagery may have low resolution")
    
    # Check if area is over water (rough check)
    center_lon, center_lat = polygon.centroid.x, polygon.centroid.y
   
    # Check longitude bounds
    if abs(center_lon) > 180:
        warnings.append("Invalid longitude coordinates")
    if abs(center_lat) > 90:
        warnings.append("Invalid latitude coordinates")
    
    return {
        "valid": len(warnings) == 0,
        "warnings": warnings,
        "dimensions": f"{width:.4f}° x {height:.4f}°",
        "area": area,
        "center": [center_lon, center_lat]
    }

def get_sentinel2_imagery(polygon: Polygon, start_date: str, end_date: str) -> Optional[np.ndarray]:
    """Download Sentinel-2 imagery using Google Earth Engine."""
    if not EE_INITIALIZED:
        raise HTTPException(status_code=500, detail="Google Earth Engine not initialized")
    
    try:
        # Validate polygon first
        validation = validate_polygon_for_imagery(polygon)
        print(f"📐 Polygon validation:")
        print(f"   Dimensions: {validation['dimensions']}")
        print(f"   Center: {validation['center']}")
        if validation['warnings']:
            for warning in validation['warnings']:
                print(f"   ⚠️  {warning}")
        
        # Convert shapely polygon to Earth Engine geometry
        coords = list(polygon.exterior.coords)
        ee_polygon = ee.Geometry.Polygon([coords])
        
        print(f"🔍 Searching for Sentinel-2 imagery...")
        print(f"📅 Date range: {start_date} to {end_date}")
        print(f"🗺️  Polygon bounds: {polygon.bounds}")
        
        # Define Sentinel-2 collection with better filtering
        s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                        .filterBounds(ee_polygon)
                        .filterDate(start_date, end_date)
                        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                        .sort('CLOUDY_PIXEL_PERCENTAGE'))
        
        # Check if any images are available
        collection_size = s2_collection.size().getInfo()
        print(f"Found {collection_size} Sentinel-2 images")
        
        if collection_size == 0:
            return None
        
        # Get the best image
        image = s2_collection.first()
        image_info = image.getInfo()
        print(f"Selected image ID: {image_info['id']}")
        
        # Select RGB bands and apply scaling - FIXED: No client-side operations
        rgb_image = (image.select(['B4', 'B3', 'B2'])
                    .clip(ee_polygon))
        
        # Apply simple contrast enhancement - Server-side only
        # Calculate min/max values for the region
        minMax = rgb_image.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=ee_polygon,
            scale=100,
            maxPixels=1e6
        )
        
        # Get min/max values (these are client-side calls)
        try:
            minMax_info = minMax.getInfo()
            print(f"Image min/max values: {minMax_info}")
            
            # Extract min/max values for stretching
            b4_min = minMax_info.get('B4_min', 0)
            b4_max = minMax_info.get('B4_max', 0.3)
            b3_min = minMax_info.get('B3_min', 0)
            b3_max = minMax_info.get('B3_max', 0.3)
            b2_min = minMax_info.get('B2_min', 0)
            b2_max = minMax_info.get('B2_max', 0.3)
            
            def linear_stretch():
                # Apply linear stretch using server-side operations only
                b4_stretched = rgb_image.select('B4').subtract(b4_min).divide(ee.Number(b4_max).subtract(b4_min)).clamp(0, 1)
                b3_stretched = rgb_image.select('B3').subtract(b3_min).divide(ee.Number(b3_max).subtract(b3_min)).clamp(0, 1)
                b2_stretched = rgb_image.select('B2').subtract(b2_min).divide(ee.Number(b2_max).subtract(b2_min)).clamp(0, 1)
            
                enhanced_image = ee.Image.cat([b4_stretched, b3_stretched, b2_stretched]).rename(['B4', 'B3', 'B2'])
            
        except Exception as e:
            print(f"Enhancement failed, using simple scaling: {e}")
            # Fallback to simple scaling
            enhanced_image = rgb_image.clamp(0, 0.3).divide(0.3)
        
        # Calculate appropriate scale based on polygon size
        bounds = polygon.bounds
        width = abs(bounds[2] - bounds[0])
        height = abs(bounds[3] - bounds[1])
        max_dimension = max(width, height)
        
        if max_dimension > 1:
            scale = 60
        elif max_dimension > 0.1:
            scale = 30
        else:
            scale = 10
        
        print(f"Using scale: {scale}m, polygon size: {width:.4f}° x {height:.4f}°")
        
        # Get the image URL for download with better error handling
        download_params = {
            'scale': scale,
            'crs': 'EPSG:4326',
            'region': ee_polygon,
            'format': 'GEO_TIFF'
        }
        
        print("🔗 Generating download URL...")
        try:
            url = enhanced_image.getDownloadURL(download_params)
            print(url)
            print(f"✅ Download URL generated successfully")
            print(f"URL length: {len(url)} characters")
        except Exception as url_error:
            print(f"❌ Failed to generate download URL: {url_error}")

            
            # Try with simpler parameters
            print("🔄 Retrying with simpler parameters...")
            try:
                simple_params = {
                    'scale': max(scale, 30),  # Use larger scale
                    'crs': 'EPSG:4326',
                    'region': ee_polygon.bounds(),  # Use bounding box instead of polygon
                    'format': 'GEO_TIFF'
                }
                url = enhanced_image.getDownloadURL(simple_params)
                print(f"✅ Download URL generated with simpler parameters")
            except Exception as retry_error:
                print(f"❌ Retry also failed: {retry_error}")
                return None
        
        print(f"🌐 Starting download...")
        
        # Open image directly from URL with PIL
        try:
            print(f"🖼️ Opening image directly from URL...")
            with Image.open(requests.get(url, stream=True).raw) as img:
                print(f"✅ Successfully opened image: mode={img.mode}, size={img.size}")
                
                # Check if image has any actual data
                if img.size[0] == 0 or img.size[1] == 0:
                    print("❌ Downloaded image has zero dimensions")
                    return None
                
                rgb_array = np.array(img)
                print(f"   Array shape: {rgb_array.shape}, dtype: {rgb_array.dtype}")
                print(f"   Array value range: {rgb_array.min()} to {rgb_array.max()}")
                
                # Check if all values are zero (black image)
                if rgb_array.max() == 0:
                    print("❌ Image appears to be completely black (all zeros)")
                    return None
                
                # Handle different image modes
                if len(rgb_array.shape) == 3 and rgb_array.shape[2] >= 3:
                    rgb_array = rgb_array[:, :, :3]
                elif len(rgb_array.shape) == 2:
                    rgb_array = np.stack([rgb_array] * 3, axis=2)
                elif len(rgb_array.shape) == 3 and rgb_array.shape[2] == 1:
                    rgb_array = np.repeat(rgb_array, 3, axis=2)
                else:
                    print(f"❌ Unexpected array shape: {rgb_array.shape}")
                    return None
                
                # Convert to float and normalize
                if rgb_array.dtype == np.uint16:
                    rgb_array = rgb_array.astype(np.float32) / 65535.0
                elif rgb_array.dtype == np.uint8:
                    rgb_array = rgb_array.astype(np.float32) / 255.0
                else:
                    rgb_array = rgb_array.astype(np.float32)
                
                rgb_array = np.clip(rgb_array, 0, 1)
                
                # Final validation
                non_zero_pixels = np.count_nonzero(rgb_array)
                total_pixels = rgb_array.size
                percentage_non_zero = (non_zero_pixels / total_pixels) * 100
                
                print(f"✅ Final image stats:")
                print(f"   Shape: {rgb_array.shape}")
                print(f"   Value range: {rgb_array.min():.4f} to {rgb_array.max():.4f}")
                print(f"   Non-zero pixels: {non_zero_pixels:,}/{total_pixels:,} ({percentage_non_zero:.1f}%)")
                
                if percentage_non_zero < 1:
                    print("⚠️  Warning: Image is mostly black/empty")
                else:
                    print("✅ Image contains valid data!")
                
                return rgb_array
                
        except Exception as img_error:
            print(f"❌ Error opening image directly from URL: {img_error}")
            return None
        
    except Exception as e:
        print(f"Error downloading Sentinel-2 imagery: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def get_landsat8_imagery(polygon: Polygon, start_date: str, end_date: str) -> Optional[np.ndarray]:
    """Download Landsat-8 imagery using Google Earth Engine."""
    if not EE_INITIALIZED:
        raise HTTPException(status_code=500, detail="Google Earth Engine not initialized")
    
    try:
        # Convert shapely polygon to Earth Engine geometry
        coords = list(polygon.exterior.coords)
        ee_polygon = ee.Geometry.Polygon([coords])
        
        print(f"Searching for Landsat-8 imagery...")
        print(f"Date range: {start_date} to {end_date}")
        
        # Define Landsat-8 collection (Surface Reflectance)
        l8_collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                        .filterBounds(ee_polygon)
                        .filterDate(start_date, end_date)
                        .filter(ee.Filter.lt('CLOUD_COVER', 50))
                        .sort('CLOUD_COVER'))
        
        # Check collection size
        collection_size = l8_collection.size().getInfo()
        print(f"Found {collection_size} Landsat-8 images")
        
        if collection_size == 0:
            l8_collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                            .filterBounds(ee_polygon)
                            .filterDate(start_date, end_date)
                            .filter(ee.Filter.lt('CLOUD_COVER', 80))
                            .sort('CLOUD_COVER'))
            collection_size = l8_collection.size().getInfo()
            print(f"With relaxed filters: {collection_size} images")
        
        if collection_size == 0:
            return None
        
        # Get the least cloudy image
        image = l8_collection.first()
        image_info = image.getInfo()
        print(f"Selected Landsat image ID: {image_info['id']}")
        
        # Apply scaling factors for Landsat Collection 2 Surface Reflectance - Server-side only
        optical_bands = image.select(['SR_B4', 'SR_B3', 'SR_B2']).multiply(0.0000275).add(-0.2)
        rgb_image = optical_bands.clip(ee_polygon)
        
        # Simple contrast enhancement - Server-side operations only
        try:
            # Get min/max values for the region
            minMax = rgb_image.reduceRegion(
                reducer=ee.Reducer.minMax(),
                geometry=ee_polygon,
                scale=100,
                maxPixels=1e6
            ).getInfo()
            
            print(f"Landsat min/max values: {minMax}")
            
            # Extract min/max for each band
            b4_min = minMax.get('SR_B4_min', 0)
            b4_max = minMax.get('SR_B4_max', 0.3)
            b3_min = minMax.get('SR_B3_min', 0)
            b3_max = minMax.get('SR_B3_max', 0.3)
            b2_min = minMax.get('SR_B2_min', 0)
            b2_max = minMax.get('SR_B2_max', 0.3)
            
            # Apply linear stretch
            b4_stretched = rgb_image.select('SR_B4').subtract(b4_min).divide(ee.Number(b4_max).subtract(b4_min)).clamp(0, 1)
            b3_stretched = rgb_image.select('SR_B3').subtract(b3_min).divide(ee.Number(b3_max).subtract(b3_min)).clamp(0, 1)
            b2_stretched = rgb_image.select('SR_B2').subtract(b2_min).divide(ee.Number(b2_max).subtract(b2_min)).clamp(0, 1)
            
            enhanced_image = ee.Image.cat([b4_stretched, b3_stretched, b2_stretched]).rename(['SR_B4', 'SR_B3', 'SR_B2'])
            
        except Exception as e:
            print(f"Landsat enhancement failed, using simple scaling: {e}")
            # Fallback to simple clipping
            enhanced_image = rgb_image.clamp(0, 0.3).divide(0.3)
        
        # Calculate appropriate scale
        bounds = polygon.bounds
        max_dimension = max(abs(bounds[2] - bounds[0]), abs(bounds[3] - bounds[1]))
        scale = 60 if max_dimension > 0.5 else 30
        
        print(f"Using scale: {scale}m")
        
        # Get the image URL for download
        url = enhanced_image.getDownloadURL({
            'scale': scale,
            'crs': 'EPSG:4326',
            'region': ee_polygon,
            'format': 'GEO_TIFF'
        })
        
        # Download the image
        response = requests.get(url, timeout=600)
        if response.status_code != 200:
            print(f"Download failed with status code: {response.status_code}")
            return None
        
        print(f"Downloaded {len(response.content)} bytes")
        
        # Save to temporary file and read with PIL
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name
        
        try:
            # Open with PIL and convert to numpy array
            with Image.open(tmp_path) as img:
                print(f"Landsat image: mode={img.mode}, size={img.size}")
                rgb_array = np.array(img)
                print(f"Array shape: {rgb_array.shape}, dtype: {rgb_array.dtype}")
                print(f"Array min/max: {rgb_array.min()}/{rgb_array.max()}")
                
                # Handle different image modes
                if len(rgb_array.shape) == 3 and rgb_array.shape[2] >= 3:
                    rgb_array = rgb_array[:, :, :3]
                elif len(rgb_array.shape) == 2:
                    rgb_array = np.stack([rgb_array] * 3, axis=2)
                
                # Convert to float and normalize based on data type
                if rgb_array.dtype == np.uint16:
                    rgb_array = rgb_array.astype(np.float32) / 65535.0
                elif rgb_array.dtype == np.uint8:
                    rgb_array = rgb_array.astype(np.float32) / 255.0
                else:
                    rgb_array = rgb_array.astype(np.float32)
                
                rgb_array = np.clip(rgb_array, 0, 1)
                
                print(f"Final Landsat array min/max: {rgb_array.min()}/{rgb_array.max()}")
                print(f"Non-zero pixels: {np.count_nonzero(rgb_array)}/{rgb_array.size}")
                
                return rgb_array
        finally:
            os.unlink(tmp_path)
        
    except Exception as e:
        print(f"Error downloading Landsat-8 imagery: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_visualization(rgb_array: np.ndarray, title: str) -> str:
    """Create a visualization of the RGB array and return as base64 string."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(rgb_array)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Convert plot to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
    buffer.seek(0)
    
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close(fig)
    
    return img_base64

@app.post("/process_imagery")
async def process_imagery(request: ImageryRequest):
    """Process KML polygon and download satellite imagery."""
    try:
        # Parse KML polygon
        polygon = parse_kml_polygon(request.kml_content)
        
        # Download imagery based on satellite type
        if request.satellite.lower() == "sentinel-2":
            rgb_array = get_sentinel2_imagery(polygon, request.start_date, request.end_date)
            title = f"Sentinel-2 RGB Imagery ({request.start_date} to {request.end_date})"
        elif request.satellite.lower() == "landsat-8":
            rgb_array = get_landsat8_imagery(polygon, request.start_date, request.end_date)
            title = f"Landsat-8 RGB Imagery ({request.start_date} to {request.end_date})"
        else:
            raise HTTPException(status_code=400, detail="Satellite must be 'sentinel-2' or 'landsat-8'")
        
        if rgb_array is None:
            raise HTTPException(status_code=404, detail="No imagery found for the specified area and time range")
        
        # Create visualization
        img_base64 = create_visualization(rgb_array, title)
        
        return JSONResponse(content={
            "success": True,
            "image": img_base64,
            "title": title,
            "shape": rgb_array.shape,
            "bounds": polygon.bounds
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Satellite Imagery API is running",
        "earth_engine_initialized": EE_INITIALIZED
    }

@app.post("/test_raw_download")
async def test_raw_download():
    """Test what Earth Engine actually returns for a simple request."""
    if not EE_INITIALIZED:
        return {"error": "Earth Engine not initialized"}
    
    try:
        # Use a very simple test case - San Francisco
        coords = [[-122.5, 37.7], [-122.4, 37.7], [-122.4, 37.8], [-122.5, 37.8], [-122.5, 37.7]]
        ee_polygon = ee.Geometry.Polygon(coords)
        
        # Get the most recent Sentinel-2 image
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterBounds(ee_polygon)
                     .filterDate('2024-01-01', '2024-12-31')
                     .sort('system:time_start', False)  # Most recent first
                     .limit(1))
        
        image = collection.first()
        if image is None:
            return {"error": "No images found"}
        
        # Just try to get one band to start simple
        single_band = image.select('B4').clip(ee_polygon)
        
        # Try the simplest possible download
        try:
            url = single_band.getDownloadURL({
                'scale': 100,  # Large scale for speed
                'crs': 'EPSG:4326',
                'region': ee_polygon,
                'format': 'GEO_TIFF'
            })
            
            print(f"Generated URL: {url[:100]}...")
            
            # Download and examine
            response = requests.get(url, timeout=60)
            
            result = {
                "success": True,
                "url_length": len(url),
                "status_code": response.status_code,
                "content_length": len(response.content),
                "content_type": response.headers.get('content-type', 'unknown'),
                "first_bytes_hex": response.content[:20].hex() if response.content else "empty",
                "first_chars": response.content[:100].decode('utf-8', errors='ignore') if response.content else "empty"
            }
            
            # Save for inspection
            debug_path = f"/tmp/ee_debug_{int(time.time())}.dat"
            with open(debug_path, 'wb') as f:
                f.write(response.content)
            result["debug_file"] = debug_path
            
            return result
            
        except Exception as download_error:
            return {
                "success": False,
                "error": str(download_error),
                "error_type": type(download_error).__name__
            }
        
    except Exception as e:
        return {"error": str(e)}

@app.post("/test_simple_imagery")
async def test_simple_imagery():
    """Test endpoint with a known working area."""
    if not EE_INITIALIZED:
        return {"error": "Earth Engine not initialized"}
    
    try:
        # Use San Francisco Bay area - known to have good imagery
        coords = [[-122.5, 37.7], [-122.3, 37.7], [-122.3, 37.9], [-122.5, 37.9], [-122.5, 37.7]]
        ee_polygon = ee.Geometry.Polygon(coords)
        
        # Get a recent image
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterBounds(ee_polygon)
                     .filterDate('2024-01-01', '2024-12-31')
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                     .sort('CLOUDY_PIXEL_PERCENTAGE')
                     .limit(1))
        
        count = collection.size().getInfo()
        if count == 0:
            return {"error": "No test images found"}
        
        image = collection.first()
        rgb_image = image.select(['B4', 'B3', 'B2']).multiply(0.0001).clip(ee_polygon)
        
        # Try to get download URL
        try:
            url = rgb_image.getDownloadURL({
                'scale': 60,
                'crs': 'EPSG:4326',
                'region': ee_polygon,
                'format': 'GEO_TIFF'
            })
            
            # Test download
            response = requests.get(url, timeout=60)
            
            return {
                "success": True,
                "collection_size": count,
                "download_status": response.status_code,
                "content_length": len(response.content),
                "content_type": response.headers.get('content-type', 'unknown'),
                "first_bytes": response.content[:20].hex() if response.content else "empty"
            }
            
        except Exception as download_error:
            return {
                "success": False,
                "collection_size": count,
                "error": str(download_error)
            }
        
    except Exception as e:
        return {"error": str(e)}

@app.post("/debug_imagery")
async def debug_imagery(request: dict):
    """Debug endpoint to check imagery availability."""
    try:
        kml_content = request.get("kml_content", "")
        start_date = request.get("start_date", "")
        end_date = request.get("end_date", "")
        
        polygon = parse_kml_polygon(kml_content)
        coords = list(polygon.exterior.coords)
        ee_polygon = ee.Geometry.Polygon([coords])
        
        # Check Sentinel-2 availability
        s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                        .filterBounds(ee_polygon)
                        .filterDate(start_date, end_date))
        
        s2_count = s2_collection.size().getInfo()
        s2_info = []
        
        if s2_count > 0:
            s2_list = s2_collection.limit(5).getInfo()
            for item in s2_list['features']:
                props = item['properties']
                s2_info.append({
                    'id': props.get('system:id', 'Unknown'),
                    'date': props.get('system:time_start', 0),
                    'cloud_cover': props.get('CLOUDY_PIXEL_PERCENTAGE', 'Unknown')
                })
        
        # Check Landsat-8 availability
        l8_collection = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
                        .filterBounds(ee_polygon)
                        .filterDate(start_date, end_date))
        
        l8_count = l8_collection.size().getInfo()
        l8_info = []
        
        if l8_count > 0:
            l8_list = l8_collection.limit(5).getInfo()
            for item in l8_list['features']:
                props = item['properties']
                l8_info.append({
                    'id': props.get('system:id', 'Unknown'),
                    'date': props.get('system:time_start', 0),
                    'cloud_cover': props.get('CLOUD_COVER', 'Unknown')
                })
        
        return {
            "polygon_bounds": polygon.bounds,
            "polygon_area": polygon.area,
            "date_range": f"{start_date} to {end_date}",
            "sentinel2": {
                "count": s2_count,
                "images": s2_info
            },
            "landsat8": {
                "count": l8_count,
                "images": l8_info
            }
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.post("/validate_kml")
async def validate_kml(request: dict):
    """Validate KML content and return polygon info."""
    try:
        kml_content = request.get("kml_content", "")
        
        if not kml_content:
            return {"valid": False, "error": "No KML content provided"}
        
        # Try to parse the polygon
        polygon = parse_kml_polygon(kml_content)
        
        return {
            "valid": True,
            "bounds": polygon.bounds,
            "area": polygon.area,
            "centroid": [polygon.centroid.x, polygon.centroid.y],
            "num_points": len(list(polygon.exterior.coords))
        }
        
    except HTTPException as he:
        return {"valid": False, "error": he.detail}
    except Exception as e:
        return {"valid": False, "error": str(e)}

@app.get("/ee_status")
async def ee_status():
    """Check Google Earth Engine initialization status."""
    return {"earth_engine_initialized": EE_INITIALIZED}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)