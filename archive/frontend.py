import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image
import folium
from streamlit_folium import st_folium
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
import geopandas as gpd
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Satellite Imagery Viewer",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ Satellite Imagery Viewer")
st.markdown("Upload a KML polygon and visualize Sentinel-2 or Landsat-8 RGB imagery")

# Sidebar for inputs
with st.sidebar:
    st.header("Configuration")
    
    # File upload
    uploaded_file = st.file_uploader("Upload KML file", type=['kml'])
    
    # Satellite selection
    satellite = st.selectbox(
        "Select Satellite",
        ["sentinel-2", "landsat-8"],
        help="Choose between Sentinel-2 (10m resolution) or Landsat-8 (30m resolution)"
    )
    
    # Date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=365),
            max_value=datetime.now()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            max_value=datetime.now()
        )
    
    # API endpoint
    api_url = st.text_input(
        "API Endpoint",
        value="http://localhost:8000",
        help="FastAPI backend URL"
    )
    
    process_button = st.button("🚀 Process Imagery", type="primary")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📍 KML Polygon Preview")
    
    if uploaded_file is not None:
        try:
            # Read KML content properly
            uploaded_file.seek(0)  # Reset file pointer
            bytes_data = uploaded_file.read()
            
            # Try different encodings
            try:
                kml_content = bytes_data.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    kml_content = bytes_data.decode('utf-8-sig')  # Handle BOM
                except UnicodeDecodeError:
                    kml_content = bytes_data.decode('latin-1')
            
            # Debug: Show file info
            st.write(f"**File Info:** {uploaded_file.name} ({len(bytes_data)} bytes)")
            st.write(f"**Content preview:** {kml_content[:200]}...")
            
            # Parse KML to extract coordinates for preview
            root = ET.fromstring(kml_content)
            coords_element = None
            for elem in root.iter():
                if elem.tag.endswith('coordinates'):
                    coords_element = elem
                    break
            
            if coords_element is not None:
                coords_text = coords_element.text.strip()
                coords_list = []
                
                for coord in coords_text.split():
                    if coord.strip():
                        lon, lat, *alt = coord.split(',')
                        coords_list.append([float(lat), float(lon)])  # Folium uses [lat, lon]
                
                if coords_list:
                    # Create folium map
                    center_lat = sum(coord[0] for coord in coords_list) / len(coords_list)
                    center_lon = sum(coord[1] for coord in coords_list) / len(coords_list)
                    
                    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
                    
                    # Add polygon to map
                    folium.Polygon(
                        locations=coords_list,
                        color='red',
                        weight=2,
                        fillColor='red',
                        fillOpacity=0.2
                    ).add_to(m)
                    
                    # Display map
                    st_folium(m, width=400, height=300)
                    
                    st.success(f"✅ Polygon loaded with {len(coords_list)} points")
                else:
                    st.error("❌ No valid coordinates found in KML")
            else:
                st.error("❌ No coordinates found in KML file")
                
        except Exception as e:
            st.error(f"❌ Error parsing KML: {str(e)}")
    else:
        st.info("👆 Upload a KML file to preview the polygon")
        
        # Add sample KML for testing
        st.subheader("🧪 Test with Sample KML")
        if st.button("Create Sample KML File"):
            sample_kml = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <Placemark>
      <name>Sample Polygon</name>
      <description>A sample polygon for testing</description>
      <Polygon>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>
              -122.4194,37.7749,0 -122.4094,37.7749,0 -122.4094,37.7849,0 -122.4194,37.7849,0 -122.4194,37.7749,0
            </coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>
  </Document>
</kml>"""
            
            st.download_button(
                label="📥 Download Sample KML",
                data=sample_kml,
                file_name="sample_polygon.kml",
                mime="application/vnd.google-earth.kml+xml"
            )
            st.success("Sample KML created! Download it and upload to test the app.")
            
        # Add a KML content tester
        st.subheader("🔍 KML Content Tester")
        if st.checkbox("Show raw file content (for debugging)"):
            if uploaded_file is not None:
                uploaded_file.seek(0)
                raw_bytes = uploaded_file.read()
                st.write(f"**Raw file bytes ({len(raw_bytes)}):**")
                st.code(repr(raw_bytes[:200]) + "..." if len(raw_bytes) > 200 else repr(raw_bytes))
                
                try:
                    decoded = raw_bytes.decode('utf-8')
                    st.write(f"**UTF-8 decoded ({len(decoded)} chars):**")
                    st.code(decoded[:500] + "..." if len(decoded) > 500 else decoded)
                except UnicodeDecodeError as e:
                    st.error(f"UTF-8 decode error: {e}")
                    try:
                        decoded = raw_bytes.decode('latin-1')
                        st.write("**Latin-1 decoded:**")
                        st.code(decoded[:500] + "..." if len(decoded) > 500 else decoded)
                    except Exception as e2:
                        st.error(f"All decoding failed: {e2}")
                
                uploaded_file.seek(0)  # Reset for other operations

with col2:
    st.subheader("🛰️ Satellite Imagery")
    
    if process_button and uploaded_file is not None:
        try:
            with st.spinner(f"Processing {satellite} imagery..."):
                # IMPORTANT: Reset file pointer and re-read
                uploaded_file.seek(0)  
                bytes_data = uploaded_file.read()
                
                # Proper decoding with error handling
                try:
                    kml_content = bytes_data.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        kml_content = bytes_data.decode('utf-8-sig')  # Handle BOM
                    except UnicodeDecodeError:
                        kml_content = bytes_data.decode('latin-1')
                
                # Debug information
                st.write("**Debug Info:**")
                st.write(f"- File name: {uploaded_file.name}")
                st.write(f"- Raw bytes: {len(bytes_data)} bytes")
                st.write(f"- Decoded string: {len(kml_content)} characters")
                st.write(f"- Content starts with: `{repr(kml_content[:50])}`")
                
                if not kml_content.strip():
                    st.error("❌ KML file appears to be empty after decoding")
                    st.stop()
                
                # Check for valid KML/XML start
                content_start = kml_content.strip()
                if not (content_start.startswith('<?xml') or content_start.startswith('<kml')):
                    st.error("❌ File doesn't appear to be valid KML/XML")
                    st.write(f"Content starts with: `{content_start[:100]}`")
                    
                    # Show raw bytes for debugging
                    st.write(f"Raw bytes start: `{repr(bytes_data[:100])}`")
                    st.stop()
                
                # Validate KML with backend
                try:
                    validation_response = requests.post(
                        f"{api_url}/validate_kml", 
                        json={"kml_content": kml_content},
                        timeout=30
                    )
                    if validation_response.status_code == 200:
                        validation_result = validation_response.json()
                        if not validation_result["valid"]:
                            st.error(f"❌ KML validation failed: {validation_result['error']}")
                            st.stop()
                        else:
                            st.success("✅ KML file is valid!")
                            st.write(f"- Polygon bounds: {validation_result['bounds']}")
                            st.write(f"- Number of points: {validation_result['num_points']}")
                    else:
                        st.warning("⚠️ Could not validate KML with backend, proceeding anyway...")
                except requests.exceptions.RequestException as e:
                    st.warning(f"⚠️ Backend validation failed: {e}, proceeding anyway...")
                
                payload = {
                    "kml_content": kml_content,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "satellite": satellite
                }
                
                # Make API request
                response = requests.post(f"{api_url}/process_imagery", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result["success"]:
                        # Decode and display image
                        img_data = base64.b64decode(result["image"])
                        img = Image.open(BytesIO(img_data))
                        
                        st.image(img, caption=result["title"], use_column_width=True)
                        
                        # Show metadata
                        st.info(f"""
                        **Image Details:**
                        - Shape: {result['shape']}
                        - Bounds: {[f"{x:.4f}" for x in result['bounds']]}
                        - Satellite: {satellite.upper()}
                        - Date Range: {start_date} to {end_date}
                        """)
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Image",
                            data=img_data,
                            file_name=f"{satellite}_imagery_{start_date}_{end_date}.png",
                            mime="image/png"
                        )
                    else:
                        st.error("❌ Processing failed")
                else:
                    error_detail = response.json().get("detail", "Unknown error")
                    st.error(f"❌ API Error: {error_detail}")
                    
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Make sure the FastAPI backend is running.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    elif process_button and uploaded_file is None:
        st.warning("⚠️ Please upload a KML file first")
    
    else:
        st.info("👆 Configure settings and click 'Process Imagery' to start")

# Footer
st.markdown("---")
st.markdown("""
**Instructions:**
1. **Authenticate Google Earth Engine:**
   ```bash
   earthengine authenticate
   ```
2. Start the FastAPI backend: `python backend.py`
3. Run this Streamlit app: `streamlit run frontend.py`
4. Upload a KML polygon file
5. Select satellite type and date range
6. Click 'Process Imagery' to download and visualize

**Note:** This app uses Google Earth Engine for satellite data access. Make sure you have authenticated your GEE account.
""")