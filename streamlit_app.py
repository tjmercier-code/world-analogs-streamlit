import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="🌍 World Analogs Database",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/YOUR_USERNAME/world-analogs-streamlit',
        'Report a bug': 'https://github.com/YOUR_USERNAME/world-analogs-streamlit/issues',
        'About': """
        ## World Analogs Database
        Assessment Unit-Scale Analogs for Oil and Gas Resource Assessment
        
        Based on USGS Open-File Report 2007-1404
        """
    }
)

# Custom CSS
st.markdown("""

    .main-header {
        font-size: 2.5rem;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1e88e5;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        border-radius: 6px;
        height: 3rem;
        background-color: #1e88e5;
        color: white;
        border: none;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #1565c0;
        border: none;
        color: white;
    }
    .sidebar .stButton > button {
        height: 2.5rem;
        font-size: 0.9rem;
    }
    .search-results {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

""", unsafe_allow_html=True)

class WorldAnalogsDatabase:
    """Main database class for World Analogs functionality"""
    
    def __init__(self, excel_file_path: str):
        """Initialize database from Excel file"""
        self.excel_file = excel_file_path
        self.data = {}
        self.current_selection = []
        self.load_data()
        
    def load_data(self):
        """Load all worksheets from Excel file"""
        try:
            with st.spinner("📊 Loading database worksheets..."):
                xl_file = pd.ExcelFile(self.excel_file)
                
                progress_bar = st.progress(0)
                total_sheets = len(xl_file.sheet_names)
                
                for i, sheet_name in enumerate(xl_file.sheet_names):
                    st.write(f"Loading {sheet_name}...")
                    self.data[sheet_name] = pd.read_excel(xl_file, sheet_name=sheet_name)
                    # Clean data
                    self.data[sheet_name] = self.data[sheet_name].dropna(how='all').reset_index(drop=True)
                    progress_bar.progress((i + 1) / total_sheets)
                
                progress_bar.empty()
                
        except Exception as e:
            st.error(f"❌ Error loading Excel file: {e}")
            raise e
    
    def get_classification_variables(self) -> List[str]:
        """Get available classification variables"""
        if 'Geology' not in self.data:
            return []
        
        geology_df = self.data['Geology']
        classification_vars = [
            'Structural Setting', 'Crustal System', 'Architecture', 
            'Trap System (Major)', 'Depositional System', 
            'Source Rock Depositional Environment', 'Kerogen Type',
            'Source Type', 'Reservoir Rock Lithology', 
            'Reservoir Rock Depositional Environment', 'Seal Rock Lithology',
            'Trap Type', 'Status'
        ]
        
        return [var for var in classification_vars if var in geology_df.columns]
    
    def get_variable_values(self, variable: str) -> List[str]:
        """Get unique values for a classification variable"""
        if 'Geology' not in self.data or variable not in self.data['Geology'].columns:
            return []
        
        values = self.data['Geology'][variable].dropna().unique().tolist()
        return sorted([str(v) for v in values if str(v) != 'nan'])
    
    def analog_search(self, search_criteria: Dict[str, Any]) -> pd.DataFrame:
        """Perform analog search"""
        if 'Geology' not in self.data:
            return pd.DataFrame()
        
        geology_df = self.data['Geology']
        mask = pd.Series(True, index=geology_df.index)
        
        for variable, value in search_criteria.items():
            if variable in geology_df.columns:
                if isinstance(value, list):
                    var_mask = geology_df[variable].isin(value)
                else:
                    var_mask = geology_df[variable] == value
                mask = mask & var_mask
        
        result_df = geology_df[mask]
        self.current_selection = result_df.index.tolist()
        return result_df
    
    def create_histogram(self, variable: str, sheet_name: str = 'BOE', n_bins: int = 20) -> go.Figure:
        """Create histogram for selected analogs"""
        if not self.current_selection or sheet_name not in self.data:
            return None
        
        df = self.data[sheet_name]
        if variable not in df.columns:
            return None
        
        selected_df = df[df.index.isin(self.current_selection)]
        values = selected_df[variable].dropna()
        
        if len(values) == 0:
            return None
        
        # Calculate statistics
        stats = {
            'Count': len(values),
            'Min': values.min(),
            'Median': values.median(),
            'Max': values.max(),
            'Mean': values.mean(),
            'Std Dev': values.std()
        }
        
        # Create histogram
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=values,
            nbinsx=n_bins,
            marker_color='rgba(30, 136, 229, 0.7)',
            marker_line_color='rgba(30, 136, 229, 1)',
            marker_line_width=2,
            name='Frequency'
        ))
        
        # Add statistical lines
        fig.add_vline(
            x=stats['Median'], 
            line_dash="dash", 
            line_color="red", 
            line_width=2,
            annotation_text=f"Median: {stats['Median']:.3f}",
            annotation_position="top right"
        )
        
        fig.add_vline(
            x=stats['Mean'], 
            line_dash="dot", 
            line_color="orange", 
            line_width=2,
            annotation_text=f"Mean: {stats['Mean']:.3f}",
            annotation_position="top left"
        )
        
        # Add statistics box
        stats_text = ''.join([f'{k}: {v:.3f}' if isinstance(v, float) else f'{k}: {v}' 
                                 for k, v in stats.items()])
        
        fig.add_annotation(
            text=stats_text,
            xref="paper", yref="paper",
            x=0.98, y=0.98,
            xanchor="right", yanchor="top",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(30, 136, 229, 1)",
            borderwidth=2,
            font=dict(size=11)
        )
        
        fig.update_layout(
            title={
                'text': f'{variable} Distribution{len(values)} Assessment Units',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            xaxis_title=variable,
            yaxis_title='Frequency',
            template='plotly_white',
            height=500,
            showlegend=False
        )
        
        return fig
    
    def get_field_statistics(self) -> Dict:
        """Get comprehensive field statistics"""
        if not self.current_selection or 'BOE' not in self.data:
            return {}
        
        boe_df = self.data['BOE']
        selected_df = boe_df[boe_df.index.isin(self.current_selection)]
        
        # Find relevant columns
        density_cols = [col for col in selected_df.columns if 'Number / 1000 km2' in col]
        size_cols = [col for col in selected_df.columns if 'Median of' in col]
        max_cols = [col for col in selected_df.columns if 'Maximum of' in col]
        
        stats = {
            'density': {},
            'median_sizes': {},
            'max_sizes': {}
        }
        
        # Field density statistics
        for col in density_cols:
            values = selected_df[col].dropna()
            if len(values) > 0:
                stats['density'][col] = {
                    'min': values.min(),
                    'median': values.median(),
                    'max': values.max(),
                    'mean': values.mean(),
                    'std': values.std()
                }
        
        # Field size statistics
        for col in size_cols:
            values = selected_df[col].dropna()
            if len(values) > 0:
                stats['median_sizes'][col] = {
                    'min': values.min(),
                    'median': values.median(),
                    'max': values.max(),
                    'mean': values.mean(),
                    'std': values.std()
                }
        
        for col in max_cols:
            values = selected_df[col].dropna()
            if len(values) > 0:
                stats['max_sizes'][col] = {
                    'min': values.min(),
                    'median': values.median(),
                    'max': values.max(),
                    'mean': values.mean(),
                    'std': values.std()
                }
        
        return stats

def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'database' not in st.session_state:
        st.session_state.database = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'current_selection' not in st.session_state:
        st.session_state.current_selection = []
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []

def display_header():
    """Display application header"""
    st.markdown('🌍 World Analogs Database', unsafe_allow_html=True)
    st.markdown('Assessment Unit-Scale Analogs for Oil and Gas Resource Assessment', unsafe_allow_html=True)
    st.markdown('Based on USGS Open-File Report 2007-1404', unsafe_allow_html=True)
    st.markdown("---")

def file_upload_section():
    """Handle file upload"""
    st.sidebar.header("📁 Data Loading")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload WorldAnalogs.xls",
        type=['xls', 'xlsx'],
        help="Upload the WorldAnalogs.xls file from USGS Open-File Report 2007-1404"
    )
    
    if uploaded_file is not None:
        try:
            # Save uploaded file temporarily
            temp_file_path = "temp_worldanalogs.xls"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Initialize database if not already done
            if st.session_state.database is None:
                st.session_state.database = WorldAnalogsDatabase(temp_file_path)
                st.session_state.data_loaded = True
            
            st.sidebar.success("✅ Database loaded successfully!")
            
            # Display database info
            if 'Geology' in st.session_state.database.data:
                geology_df = st.session_state.database.data['Geology']
                st.sidebar.metric("📊 Assessment Units", len(geology_df))
                st.sidebar.metric("🌍 Provinces", geology_df['Province Name'].nunique())
            
            return True
            
        except Exception as e:
            st.sidebar.error(f"❌ Error loading file: {e}")
            st.session_state.data_loaded = False
            return False
    
    return False

def search_interface():
    """Search interface in sidebar"""
    if not st.session_state.data_loaded:
        return
    
    st.sidebar.header("🔍 Analog Search")
    
    db = st.session_state.database
    
    # Quick search presets
    st.sidebar.subheader("⚡ Quick Searches")
    
    quick_searches = {
        "🌊 Turbidite Systems": {"Depositional System": ["Slope, clinoforms, turbidites"]},
        "🏔️ Carbonate Reservoirs": {"Reservoir Rock Lithology": ["Carbonates"]},
        "🏞️ Lacustrine Sources": {"Source Rock Depositional Environment": ["Lacustrine"]},
        "🌋 Extensional Settings": {"Structural Setting": ["Extensional"]},
        "🏗️ Foreland Basins": {"Architecture": ["Foreland"]}
    }
    
    for search_name, criteria in quick_searches.items():
        if st.sidebar.button(search_name, key=f"quick_{search_name}"):
            results = db.analog_search(criteria)
            st.session_state.current_selection = results.index.tolist()
            st.session_state.search_history.append({
                'name': search_name,
                'criteria': criteria,
                'results': len(results)
            })
            st.rerun()
    
    # Custom search
    st.sidebar.subheader("🔧 Custom Search")
    
    variables = db.get_classification_variables()
    selected_var = st.sidebar.selectbox("Select variable:", [""] + variables)
    
    if selected_var:
        values = db.get_variable_values(selected_var)
        selected_values = st.sidebar.multiselect(
            f"Select values for {selected_var}:",
            values,
            key=f"multiselect_{selected_var}"
        )
        
        if selected_values:
            if st.sidebar.button("🔍 Execute Search", type="primary"):
                search_criteria = {selected_var: selected_values}
                results = db.analog_search(search_criteria)
                st.session_state.current_selection = results.index.tolist()
                st.session_state.search_history.append({
                    'name': f'Custom: {selected_var}',
                    'criteria': search_criteria,
                    'results': len(results)
                })
                st.sidebar.success(f"Found {len(results)} analogs")
                st.rerun()
    
    # Search history
    if st.session_state.search_history:
        st.sidebar.subheader("📜 Recent Searches")
        for search in st.session_state.search_history[-3:]:  # Show last 3
            st.sidebar.text(f"• {search['name']}: {search['results']} results")

def main_content():
    """Main content area"""
    if not st.session_state.data_loaded:
        display_help_content()
        return
    
    if not st.session_state.current_selection:
        st.info("👈 Please use the sidebar to search for analogs")
        return
    
    db = st.session_state.database
    
    # Display current selection
    display_current_selection(db)
    
    # Analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Selection Details", 
        "📊 Field Analysis", 
        "📈 Visualizations", 
        "📄 Assessment Report"
    ])
    
    with tab1:
        selection_details_tab(db)
    
    with tab2:
        field_analysis_tab(db)
    
    with tab3:
        visualizations_tab(db)
    
    with tab4:
        assessment_report_tab(db)

def display_current_selection(db):
    """Display current selection summary"""
    num_selected = len(st.session_state.current_selection)
    
    st.markdown(f'', unsafe_allow_html=True)
    st.subheader(f"🎯 Current Selection: {num_selected} Assessment Units")
    
    # Quick statistics
    geology_df = db.data['Geology']
    selected_data = geology_df[geology_df.index.isin(st.session_state.current_selection)]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🌍 Provinces", selected_data['Province Name'].nunique())
    with col2:
        st.metric("🏗️ Structural Settings", selected_data['Structural Setting'].nunique())
    with col3:
        st.metric("🪨 Depositional Systems", selected_data['Depositional System'].nunique())
    with col4:
        st.metric("🎯 Selection %", f"{num_selected/len(geology_df)*100:.1f}%")
    
    st.markdown('', unsafe_allow_html=True)

def selection_details_tab(db):
    """Selection details tab content"""
    st.subheader("📋 Selected Assessment Units")
    
    geology_df = db.data['Geology']
    selected_data = geology_df[geology_df.index.isin(st.session_state.current_selection)]
    
    # Display key columns
    display_cols = [
        'AU_Code', 'AU_Name', 'Province Name', 
        'Structural Setting', 'Depositional System', 'Status'
    ]
    available_cols = [col for col in display_cols if col in selected_data.columns]
    
    st.dataframe(
        selected_data[available_cols], 
        use_container_width=True,
        height=400
    )
    
    # Download options
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = selected_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Selection (CSV)",
            data=csv_data,
            file_name="world_analogs_selection.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Detailed CSV with all columns
        detailed_csv = selected_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Detailed (CSV)",
            data=detailed_csv,
            file_name="world_analogs_detailed.csv",
            mime="text/csv",
            use_container_width=True
        )

def field_analysis_tab(db):
    """Field analysis tab content"""
    st.subheader("📊 Field Analysis")
    
    stats = db.get_field_statistics()
    
    if not stats:
        st.warning("⚠️ No field statistics available for current selection")
        return
    
    # Field Density Analysis
    if stats.get('density'):
        st.markdown("### 🏗️ Field Density Analysis")
        st.markdown("*Fields per 1,000 km²*")
        
        for var, data in stats['density'].items():
            threshold = "5 MMBOE" if "> 5" in var else "50 MMBOE" if "> 50" in var else "all sizes"
            
            with st.expander(f"📊 Field Density > {threshold}", expanded=True):
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Minimum", f"{data['min']:.3f}")
                with col2:
                    st.metric("Median", f"{data['median']:.3f}")
                with col3:
                    st.metric("Mean", f"{data['mean']:.3f}")
                with col4:
                    st.metric("Maximum", f"{data['max']:.3f}")
                with col5:
                    st.metric("Std Dev", f"{data['std']:.3f}")
    
    # Field Size Analysis
    if stats.get('median_sizes'):
        st.markdown("### 📏 Field Size Analysis")
        st.markdown("*Median field sizes in MMBOE*")
        
        for var, data in stats['median_sizes'].items():
            threshold = "5 MMBOE" if "> 5" in var else "50 MMBOE" if "> 50" in var else "all sizes"
            
            with st.expander(f"📏 Median Field Size > {threshold}", expanded=True):
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Minimum", f"{data['min']:.1f}")
                with col2:
                    st.metric("Median", f"{data['median']:.1f}")
                with col3:
                    st.metric("Mean", f"{data['mean']:.1f}")
                with col4:
                    st.metric("Maximum", f"{data['max']:.1f}")
                with col5:
                    st.metric("Std Dev", f"{data['std']:.1f}")

def visualizations_tab(db):
    """Visualizations tab content"""
    st.subheader("📈 Interactive Visualizations")
    
    # Configuration options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sheet_choice = st.selectbox("Data Sheet:", ["BOE", "Oil", "Gas"])
    
    with col2:
        n_bins = st.slider("Number of Bins:", 10, 50, 20)
    
    # Get available variables for plotting
    if sheet_choice in db.data:
        sheet_df = db.data[sheet_choice]
        plot_vars = [col for col in sheet_df.columns if any(keyword in col.lower() 
                    for keyword in ['number', 'median', 'maximum', 'density', 'ratio'])]
        
        with col3:
            if plot_vars:
                selected_var = st.selectbox("Variable to Plot:", plot_vars)
            else:
                selected_var = None
                st.warning("No plottable variables found")
        
        if selected_var:
            # Create and display visualization
            fig = db.create_histogram(selected_var, sheet_choice, n_bins)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                # Additional analysis
                st.markdown("### 📊 Variable Analysis")
                
                # Get the actual data for additional insights
                selected_df = db.data[sheet_choice][db.data[sheet_choice].index.isin(st.session_state.current_selection)]
                values = selected_df[selected_var].dropna()
                
                if len(values) > 1:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Distribution Shape:**")
                        skewness = values.skew()
                        kurtosis = values.kurtosis()
                        
                        if abs(skewness) < 0.5:
                            skew_desc = "Approximately symmetric"
                        elif skewness > 0.5:
                            skew_desc = "Right-skewed (positive)"
                        else:
                            skew_desc = "Left-skewed (negative)"
                        
                        st.write(f"• Skewness: {skewness:.3f} ({skew_desc})")
                        st.write(f"• Kurtosis: {kurtosis:.3f}")
                    
                    with col2:
                        st.markdown("**Percentiles:**")
                        st.write(f"• 25th percentile: {values.quantile(0.25):.3f}")
                        st.write(f"• 75th percentile: {values.quantile(0.75):.3f}")
                        st.write(f"• 90th percentile: {values.quantile(0.90):.3f}")
                        st.write(f"• 95th percentile: {values.quantile(0.95):.3f}")
            else:
                st.warning("⚠️ No data available for the selected variable")

def assessment_report_tab(db):
    """Assessment report tab content"""
    st.subheader("📄 Assessment Report Generator")
    
    # Report parameters
    col1, col2 = st.columns(2)
    
    with col1:
        target_area = st.number_input(
            "Target Area (km²):",
            min_value=1000,
            max_value=1000000,
            value=100000,
            step=10000,
            help="Enter the area of your prospect/assessment unit"
        )
    
    with col2:
        min_field_size = st.selectbox(
            "Minimum Field Size:",
            [5, 50],
            format_func=lambda x: f"{x} MMBOE",
            help="Minimum field size threshold for assessment"
        )
    
    # Generate report
    if st.button("📊 Generate Assessment Report", type="primary"):
        generate_assessment_report(db, target_area, min_field_size)

def generate_assessment_report(db, target_area: int, min_field_size: int):
    """Generate comprehensive assessment report"""
    st.markdown("---")
    st.markdown("## 📋 Assessment Report")
    
    # Report header
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 Target Area", f"{target_area:,} km²")
    with col2:
        st.metric("📏 Min Field Size", f"{min_field_size} MMBOE")
    with col3:
        st.metric("📊 Analogs Used", len(st.session_state.current_selection))
    
    # Get field statistics
    stats = db.get_field_statistics()
    
    # Field count estimates
    st.markdown("### 🎯 Field Count Estimates")
    
    density_var = f'Number / 1000 km2 for > {min_field_size}'
    
    if stats.get('density') and density_var in stats['density']:
        density_stats = stats['density'][density_var]
        
        # Calculate field estimates
        low_estimate = density_stats['min'] * (target_area / 1000)
        modal_estimate = density_stats['median'] * (target_area / 1000)
        high_estimate = density_stats['max'] * (target_area / 1000)
        mean_estimate = density_stats['mean'] * (target_area / 1000)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Low Estimate",
                f"{low_estimate:.0f} fields",
                help=f"Based on minimum density: {density_stats['min']:.3f}/1000km²"
            )
        
        with col2:
            st.metric(
                "Modal Estimate",
                f"{modal_estimate:.0f} fields",
                help=f"Based on median density: {density_stats['median']:.3f}/1000km²"
            )
        
        with col3:
            st.metric(
                "Mean Estimate",
                f"{mean_estimate:.0f} fields",
                help=f"Based on mean density: {density_stats['mean']:.3f}/1000km²"
            )
        
        with col4:
            st.metric(
                "High Estimate",
                f"{high_estimate:.0f} fields",
                help=f"Based on maximum density: {density_stats['max']:.3f}/1000km²"
            )
        
        # Recommendation
        st.info(f"💡 **Recommended range:** {modal_estimate:.0f} to {high_estimate:.0f} fields (>{min_field_size} MMBOE) in {target_area:,} km²")
    
    else:
        st.warning("⚠️ No field density data available for selected analogs")
    
    # Field size estimates
    st.markdown("### 📏 Field Size Recommendations")
    
    size_var = f'Median of > {min_field_size}'
    max_var = f'Maximum of > {min_field_size}'
    
    if stats.get('median_sizes') and size_var in stats['median_sizes']:
        size_stats = stats['median_sizes'][size_var]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Median Field Size:**")
            st.success(f"🎯 **Recommended:** {size_stats['median']:.1f} MMBOE")
            st.write(f"• Range: {size_stats['min']:.1f} - {size_stats['max']:.1f} MMBOE")
            st.write(f"• Mean: {size_stats['mean']:.1f} MMBOE")
        
        with col2:
            if stats.get('max_sizes') and max_var in stats['max_sizes']:
                max_stats = stats['max_sizes'][max_var]
                st.markdown("**Maximum Field Size:**")
                st.success(f"🔝 **Recommended:** {max_stats['median']:.0f} MMBOE")
                st.write(f"• Range: {max_stats['min']:.0f} - {max_stats['max']:.0f} MMBOE")
                st.write(f"• Mean: {max_stats['mean']:.0f} MMBOE")
    
    # Assessment summary
    st.markdown("### 📋 Assessment Summary")
    
    geology_df = db.data['Geology']
    selected_data = geology_df[geology_df.index.isin(st.session_state.current_selection)]
    
    summary_text = f"""
    **Assessment Parameters:**
    - Target Area: {target_area:,} km²
    - Minimum Field Size: {min_field_size} MMBOE
    - Number of Analogs: {len(st.session_state.current_selection)}
    - Provinces Represented: {selected_data['Province Name'].nunique()}
    
    **Key Findings:**
    - Modal field count estimate: {modal_estimate:.0f} fields
    - Recommended median field size: {size_stats['median']:.1f} MMBOE
    - Assessment based on global analog database (USGS 2007-1404)
    """
    
    st.markdown(summary_text)
    
    # Download report
    report_data = f"""
WORLD ANALOGS DATABASE - ASSESSMENT REPORT
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

ASSESSMENT PARAMETERS:
Target Area: {target_area:,} km²
Minimum Field Size: {min_field_size} MMBOE
Number of Analogs: {len(st.session_state.current_selection)}
Provinces Represented: {selected_data['Province Name'].nunique()}

FIELD COUNT ESTIMATES:
Low Estimate: {low_estimate:.0f} fields
Modal Estimate: {modal_estimate:.0f} fields
High Estimate: {high_estimate:.0f} fields

FIELD SIZE RECOMMENDATIONS:
Median Field Size: {size_stats['median']:.1f} MMBOE
Maximum Field Size: {max_stats['median']:.0f} MMBOE

SELECTED ANALOGS:
{selected_data[['AU_Code', 'AU_Name', 'Province Name']].to_string(index=False)}

DATA SOURCE:
Charpentier, R.R., Klett, T.R., and Attanasi, E.D., 2008, Database for assessment 
unit-scale analogs (exclusive of the United States): U.S. Geological Survey 
Open-File Report 2007-1404, 61 p.
    """
    
    st.download_button(
        label="📥 Download Assessment Report",
        data=report_data,
        file_name=f"assessment_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt",
        mime="text/plain",
        use_container_width=True
    )

def display_help_content():
    """Display help content when no data is loaded"""
    st.markdown("""
    ## 📖 Welcome to the World Analogs Database
    
    This application replicates the functionality of the Excel macros in the USGS World Analogs Database 
    (Open-File Report 2007-1404) for assessment unit-scale analogs.
    
    ### 🚀 Getting Started:
    1. **📁 Upload Data**: Use the sidebar to upload the WorldAnalogs.xls file
    2. **🔍 Search Analogs**: Use quick searches or custom criteria to find relevant analogs
    3. **📊 Analyze Results**: Examine field densities, sizes, and relationships
    4. **📄 Generate Reports**: Create comprehensive assessment reports
    
    ### 🔧 Key Features:
    
    #### ⚡ Quick Search Presets:
    - **🌊 Turbidite Systems**: Find deep-water turbidite analogs
    - **🏔️ Carbonate Reservoirs**: Search carbonate reservoir analogs
    - **🏞️ Lacustrine Sources**: Find lacustrine source rock analogs
    - **🌋 Extensional Settings**: Search extensional tectonic settings
    - **🏗️ Foreland Basins**: Find foreland basin analogs
    
    #### 🔧 Custom Search:
    - Multi-criteria searches on geological variables
    - Boolean logic for refined analog selection
    - Search history tracking
    
    #### 📊 Analysis Tools:
    - **Field Density Analysis**: Calculate fields per 1000 km²
    - **Field Size Analysis**: Examine median and maximum field sizes
    - **Interactive Visualizations**: Histograms with statistical overlays
    - **Assessment Reports**: Generate field count and size estimates
    
    ### 📊 Data Source:
    **Charpentier, R.R., Klett, T.R., and Attanasi, E.D., 2008**  
    *Database for assessment unit-scale analogs (exclusive of the United States)*  
    U.S. Geological Survey Open-File Report 2007-1404, 61 p.
    
    ### 💡 Tips for Best Results:
    - Start with quick search presets to explore the database
    - Use custom searches to combine multiple geological criteria
    - Analyze both field density and size statistics
    - Generate assessment reports for documentation
    - Download results for further analysis in other tools
    
    ### 🔗 Additional Resources:
    - [USGS Energy Resources Program](https://energy.usgs.gov/)
    - [Original Excel Database](https://pubs.usgs.gov/of/2007/1404/)
    - [Assessment Methodology](https://pubs.usgs.gov/dds/dds-060/)
    """)

# Main application
def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Display header
    display_header()
    
    # File upload section
    data_loaded = file_upload_section()
    
    # Search interface (if data loaded)
    if data_loaded:
        search_interface()
    
    # Main content
    main_content()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    
        Built with ❤️ using Streamlit • Data from USGS Open-File Report 2007-1404
        🌍 Deployed on Streamlit Community Cloud
    
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
```




