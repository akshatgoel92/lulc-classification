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

st.title("🛰️ LULC Analysis Tool")
st.write("Upload a KML file to analyze Land Use Land Cover (LULC) data from pre-stored TIFF files")

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
        with st.spinner(f"Analyzing LULC from {start_year} to {end_year}..."):
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
                        
                        st.subheader(f"📈 Forest Coverage Analysis ({start_year}-{end_year})")
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
                            
                            # Line chart showing only Forest trend
                            fig, ax = plt.subplots(figsize=(12, 6))
                            
                            # Create a complete year range from start to end
                            full_year_range = list(range(start_year, end_year + 1))
                            
                            # Get forest data for analyzed years
                            forest_data = trend_df_complete[trend_df_complete["Class"] == "Forest"]
                            forest_percentages = {}
                            for _, row in forest_data.iterrows():
                                forest_percentages[row["Year"]] = row["Percentage"]
                            
                            # Get actually analyzed years from the response
                            analyzed_years = set(data.get('years_analyzed', []))
                            
                            # Create complete forest data with 0 for missing years
                            years = []
                            percentages = []
                            for year in full_year_range:
                                years.append(year)
                                percentages.append(forest_percentages.get(year, 0))
                            
                            # Plot forest data
                            color = LULC_COLORS.get("Forest", '#008000')
                            ax.plot(years, percentages, 
                                   marker='o', label="Forest", linewidth=2, color=color)
                            
                            # Mark years where TIFF was not available (not just 0% forest)
                            no_data_shown = False
                            for i, year in enumerate(years):
                                if year not in analyzed_years:
                                    # Only show red dot if TIFF was not available
                                    label = 'No data' if not no_data_shown else ""
                                    ax.plot(year, percentages[i], 'o', color='red', markersize=8, label=label)
                                    no_data_shown = True
                            
                            ax.set_xlabel("Year")
                            ax.set_ylabel("Forest Coverage (%)")
                            ax.set_title("Forest Coverage Change Over Time")
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                            ax.set_xticks(full_year_range)
                            ax.set_xticklabels(full_year_range, rotation=45)
                            st.pyplot(fig)
                            
                            # Eligibility Panel
                            st.markdown("---")
                            
                            # Check eligibility: forest coverage must be < 10% for all years with data
                            eligible = True
                            forest_percentages_with_data = []
                            
                            for year in analyzed_years:
                                if year in forest_percentages:
                                    forest_percent = forest_percentages[year]
                                    forest_percentages_with_data.append(forest_percent)
                                    if forest_percent >= 10:
                                        eligible = False
                            
                            # Display eligibility status
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                                if eligible:
                                    st.markdown(
                                        """
                                        <div style='background-color: #d4edda; border: 2px solid #28a745; 
                                                    border-radius: 10px; padding: 20px; text-align: center;'>
                                            <h2 style='color: #28a745; margin: 0;'>✓ Eligible</h2>
                                        </div>
                                        """, 
                                        unsafe_allow_html=True
                                    )
                                    st.info(f"Forest coverage is below 10% for all {len(analyzed_years)} years with available data")
                                else:
                                    st.markdown(
                                        """
                                        <div style='background-color: #f8d7da; border: 2px solid #dc3545; 
                                                    border-radius: 10px; padding: 20px; text-align: center;'>
                                            <h2 style='color: #dc3545; margin: 0;'>✗ Not Eligible</h2>
                                        </div>
                                        """, 
                                        unsafe_allow_html=True
                                    )
                                    # Find years where forest coverage >= 10%
                                    years_above_threshold = []
                                    for year in analyzed_years:
                                        if year in forest_percentages and forest_percentages[year] >= 10:
                                            years_above_threshold.append(f"{year} ({forest_percentages[year]:.1f}%)")
                                    
                                    st.warning(f"Forest coverage exceeds 10% in: {', '.join(years_above_threshold)}")
                            
                            # Data table
                            st.subheader("📋 Detailed Data")
                            pivot_df = trend_df_complete.pivot(index="Year", columns="Class", values="Percentage")
                            # Format the DataFrame to prevent year formatting issues
                            pivot_df.index = pivot_df.index.astype(str)
                            st.dataframe(pivot_df, use_container_width=True)
                    else:
                        st.error(f"Backend returned error: {data.get('message', 'Unknown error')}")
                        st.write("Full response:", data)
                else:
                    st.error(f"Failed to get analysis: {response.status_code}")
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