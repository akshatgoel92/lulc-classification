import streamlit as st
import requests
import base64
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



# API base URL
api_url = "http://localhost:8000"

# Define LULC class colors (matching backend)
LULC_COLORS = {
    "Water": "#0000FF",      # Blue
    "Crop land": "#00FF00",  # Green
    "Barren": "#CC9966",     # Brown
    "Forest": "#008000",     # Dark Green
    "Built-up": "#FF0000"    # Red
}

st.title("🛰️ Satellite Imagery Viewer")
st.write("Upload a KML file to view satellite imagery and LULC analysis from pre-stored TIFF files")

# File upload
uploaded_file = st.file_uploader("Choose a KML file", type=['kml'])

if uploaded_file is not None:
    kml_content = uploaded_file.read().decode('utf-8')
    st.success("✅ KML file uploaded successfully!")
    
    # Controls
    st.subheader("📊 LULC Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        start_year = st.selectbox("Start Year", list(range(2013, 2025)), index=11, key="start_year_select")  # Default to 2024
    
    with col2:
        end_year = st.selectbox("End Year", list(range(2013, 2025)), index=11, key="end_year_select")  # Default to 2024
    
    # LULC Analysis
    if st.button("📊 Get LULC Analysis", key="lulc_analysis_button"):
        if start_year == end_year:
            # Single year analysis - show bar chart
            with st.spinner(f"Analyzing LULC for {start_year}..."):
                payload = {
                    "kml_content": kml_content,
                    "year": start_year
                }
                
                try:
                    response = requests.post(f"{api_url}/lulc_analysis", json=payload)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if data.get("success"):
                            stats = data["data"]
                            
                            if "error" in stats:
                                st.error(f"❌ LULC Analysis Error: {stats['error']}")
                            else:
                                st.subheader(f"📊 LULC Analysis - {start_year}")
                                st.write(f"**Total Area:** {stats['total_area_m2']:,.0f} m²")
                                
                                # Create DataFrame for display
                                df = pd.DataFrame(stats["classes"])
                                st.dataframe(df, use_container_width=True)
                                
                                # Pie chart with matching colors
                                fig, ax = plt.subplots(figsize=(8, 6))
                                colors = [LULC_COLORS.get(name, '#808080') for name in df["class_name"]]
                                ax.pie(df["percentage"], labels=df["class_name"], autopct='%1.1f%%', colors=colors)
                                ax.set_title(f"LULC Distribution - {start_year}")
                                st.pyplot(fig)
                        else:
                            st.error(f"Backend returned error: {data.get('message', 'Unknown error')}")
                            st.write("Full LULC response:", data)
                    else:
                        st.error(f"Failed to get LULC analysis: {response.status_code}")
                        st.error(f"Response: {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend API. Is the FastAPI server running?")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        else:
            # Multi-year analysis - show line chart
            with st.spinner(f"Analyzing LULC trends from {start_year} to {end_year}..."):
                payload = {
                    "kml_content": kml_content,
                    "start_year": start_year,
                    "end_year": end_year
                }
                
                try:
                    response = requests.post(f"{api_url}/multi_year_analysis", json=payload)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if data.get("success"):
                            results = data["data"]
                            
                            st.subheader(f"📈 LULC Trends ({start_year}-{end_year})")
                            st.write(f"**Years analyzed:** {data.get('years_analyzed', [])}")
                            
                            if len(results) == 0:
                                st.warning("No data available for the selected year range")
                            else:
                                # Create trend visualization
                                trend_data = []
                                all_years = set()
                                all_classes = set()
                                
                                # First pass: collect data and identify all years and classes
                                for result in results:
                                    all_years.add(result["year"])
                                    for cls in result["classes"]:
                                        all_classes.add(cls["class_name"])
                                        trend_data.append({
                                            "Year": result["year"],
                                            "Class": cls["class_name"],
                                            "Percentage": cls["percentage"] if cls["percentage"] is not None else 0,
                                            "Area_m2": cls["area_m2"] if cls["area_m2"] is not None else 0
                                        })
                                
                                # Convert to DataFrame
                                trend_df = pd.DataFrame(trend_data)
                                
                                # Create a complete DataFrame with all year-class combinations
                                all_years = sorted(list(all_years))
                                all_classes = sorted(list(all_classes))
                                
                                # Create a complete index with all combinations
                                complete_index = pd.MultiIndex.from_product([all_years, all_classes], names=['Year', 'Class'])
                                
                                # Set index on trend_df and reindex to include all combinations
                                trend_df_indexed = trend_df.set_index(['Year', 'Class'])
                                trend_df_complete = trend_df_indexed.reindex(complete_index, fill_value=0).reset_index()
                                
                                # Line chart showing trends with matching colors
                                fig, ax = plt.subplots(figsize=(12, 6))
                                for class_name in all_classes:
                                    class_data = trend_df_complete[trend_df_complete["Class"] == class_name]
                                    color = LULC_COLORS.get(class_name, '#808080')
                                    ax.plot(class_data["Year"], class_data["Percentage"], 
                                           marker='o', label=class_name, linewidth=2, color=color)
                                
                                ax.set_xlabel("Year")
                                ax.set_ylabel("Percentage (%)")
                                ax.set_title("LULC Change Over Time")
                                ax.legend()
                                ax.grid(True, alpha=0.3)
                                st.pyplot(fig)
                                
                                # Data table
                                st.subheader("📋 Detailed Data")
                                pivot_df = trend_df_complete.pivot(index="Year", columns="Class", values="Percentage")
                                # Format the DataFrame to prevent year formatting issues
                                pivot_df.index = pivot_df.index.astype(str)
                                st.dataframe(pivot_df, use_container_width=True)
                        else:
                            st.error(f"Backend returned error: {data.get('message', 'Unknown error')}")
                            st.write("Full multi-year response:", data)
                    else:
                        st.error(f"Failed to get multi-year analysis: {response.status_code}")
                        st.error(f"Response: {response.text}")
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend API. Is the FastAPI server running?")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Satellite Imagery Section
    st.markdown("---")
    st.subheader("🛰️ Satellite Imagery")
    
    # Get available years for satellite imagery
    available_years = []
    try:
        response = requests.get(f"{api_url}/available_years")
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                available_years = data.get("years", list(range(2013, 2025)))
            else:
                available_years = list(range(2013, 2025))
        else:
            available_years = list(range(2013, 2025))
    except:
        available_years = list(range(2013, 2025))
    
    # Year selector for satellite imagery
    imagery_year = st.selectbox(
        "Select Year for Imagery", 
        available_years, 
        index=len(available_years)-1 if available_years else 0,
        key="imagery_year_select"
    )
    
    # Get satellite imagery button
    if st.button("🛰️ Get Satellite Imagery", key="satellite_imagery_button"):
        with st.spinner(f"Loading satellite imagery for {imagery_year}..."):
            payload = {
                "kml_content": kml_content,
                "year": imagery_year,
                "use_raw": False
            }
            
            try:
                response = requests.post(f"{api_url}/satellite_imagery", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get("success"):
                        st.success(data["message"])
                        
                        # Display imagery
                        if data.get("imagery_base64"):
                            st.subheader(f"📸 Satellite Imagery - {imagery_year}")
                            st.write(f"**Source:** {data.get('source', 'Unknown')} | **Shape:** {data.get('shape', 'Unknown')}")
                            
                            # Decode and display image
                            img_data = base64.b64decode(data["imagery_base64"])
                            st.image(img_data, caption=f"Satellite Imagery - {imagery_year}", use_column_width=True)
                            
                            # Display legend under the image
                            st.markdown("**🎨 LULC Classification Legend:**")
                            
                            # Create columns for the legend
                            legend_cols = st.columns(len(LULC_COLORS))
                            for i, (class_name, color) in enumerate(LULC_COLORS.items()):
                                with legend_cols[i]:
                                    # Create a colored square using matplotlib
                                    fig, ax = plt.subplots(figsize=(1, 1))
                                    ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=color))
                                    ax.set_xlim(0, 1)
                                    ax.set_ylim(0, 1)
                                    ax.axis('off')
                                    st.pyplot(fig, use_container_width=True)
                                    plt.close()
                                    st.caption(class_name)
                        else:
                            st.warning("No imagery data available in response")
                            st.write("Available keys in response:", list(data.keys()))
                    else:
                        st.error(f"Backend returned error: {data.get('message', 'Unknown error')}")
                        st.write("Full response:", data)
                else:
                    st.error(f"Failed to get satellite imagery: {response.status_code}")
                    st.error(f"Response: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend API. Is the FastAPI server running?")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Add API Health Check section at the bottom
st.markdown("---")
if st.button("🔧 Check Backend Status", key="backend_status_button"):
    try:
        response = requests.get(f"{api_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            st.success("✅ Backend API is running")
            st.json(health_data)
        else:
            st.error(f"❌ Backend API error: {response.status_code}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend API")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")