import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from datetime import datetime, timedelta
import pycountry
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from collections import Counter
from github_user_analyzer import ORG_NAME_VARIATIONS

# Helper function to extract country from location string
def extract_country(location):
    """Extract country from a location string using pattern matching and common names"""
    if not location:
        return None
    
    # Common country names and abbreviations
    country_mapping = {
        'us': 'United States',
        'usa': 'United States',
        'u.s.': 'United States',
        'u.s.a.': 'United States',
        'united states': 'United States',
        'america': 'United States',
        'uk': 'United Kingdom',
        'u.k.': 'United Kingdom',
        'britain': 'United Kingdom',
        'england': 'United Kingdom',
        'great britain': 'United Kingdom',
        'deutschland': 'Germany',
        'bundesrepublik deutschland': 'Germany',
        'deutschland/germany': 'Germany',
        'españa': 'Spain',
        'italia': 'Italy',
        'polska': 'Poland',
        'россия': 'Russia',
        'schweiz': 'Switzerland',
        'suisse': 'Switzerland',
        'svizzera': 'Switzerland',
        '中国': 'China',
        '日本': 'Japan',
        'भारत': 'India',
        'المملكة العربية السعودية': 'Saudi Arabia',
        'مصر': 'Egypt',
        '대한민국': 'South Korea',
        'ประเทศไทย': 'Thailand',
    }
    
    # Try direct mapping first
    loc_lower = location.lower()
    if loc_lower in country_mapping:
        return country_mapping[loc_lower]
    
    # Handle cases like "City, Country"
    parts = [p.strip() for p in location.split(',')]
    if len(parts) > 1:
        last_part = parts[-1].lower()
        if last_part in country_mapping:
            return country_mapping[last_part]
    
    try:
        # Try to find country codes or names in the string
        for country in pycountry.countries:
            # Check for country code
            if country.alpha_2.lower() == loc_lower or country.alpha_3.lower() == loc_lower:
                return country.name
            # Check for country name
            if country.name.lower() == loc_lower:
                return country.name
            # Check within the location string
            if country.name.lower() in loc_lower or (hasattr(country, 'common_name') and country.common_name.lower() in loc_lower):
                return country.name
    except ImportError:
        pass
    
    return None

# Helper function to normalize company/organization names
def normalize_org_name(org_name):
    """Normalize organization names for more accurate matching"""
    if not org_name:
        return ""
    
    # Convert to lowercase and remove common punctuation
    normalized = org_name.lower().replace(',', ' ').replace('.', ' ').replace('-', ' ')
    
    # Replace common abbreviations and prefixes/suffixes
    replacements = {
        'univ': 'university',
        'univ.': 'university',
        'dept': 'department',
        'dept.': 'department',
        'inc': '',
        'inc.': '',
        'llc': '',
        'llc.': '',
        'ltd': '',
        'ltd.': '',
        'co.': 'company',
        'corp.': 'corporation',
        'corp': 'corporation',
        '@': '',
    }
    
    for old, new in replacements.items():
        normalized = normalized.replace(f" {old} ", f" {new} " if new else " ")
    
    # Remove extra whitespace
    normalized = ' '.join(normalized.split())
    
    return normalized

# Import ORG_NAME_VARIATIONS from github_user_analyzer to ensure consistent mappings
from github_user_analyzer import ORG_NAME_VARIATIONS

# Function to check if org names match using word comparison
def org_names_match(org1, org2, min_word_length=3):
    """
    Check if organization names match by comparing individual words
    Only considers words of at least min_word_length characters
    Returns match score (higher = better match)
    """
    if not org1 or not org2:
        return 0
    
    # First check if org1 or org2 matches a canonical organization name or its variations
    org1_lower = org1.lower().strip() if org1 else ""
    org2_lower = org2.lower().strip() if org2 else ""
    
    # Check if either name directly maps to a canonical form
    for standard_name, variations in ORG_NAME_VARIATIONS.items():
        # If org1 matches standard name or its variations
        if (org1_lower == standard_name or org1_lower in variations) and \
           (org2_lower == standard_name or org2_lower in variations):
            return 1.0  # Perfect match - both map to same canonical form
    
    # Normalize both names
    norm1 = normalize_org_name(org1)
    norm2 = normalize_org_name(org2)
    
    # Split into words and filter for minimum length
    words1 = [w for w in norm1.split() if len(w) >= min_word_length]
    words2 = [w for w in norm2.split() if len(w) >= min_word_length]
    
    # If either has no qualifying words, no match
    if not words1 or not words2:
        return 0
    
    # Count matching words
    common_words = set(words1).intersection(set(words2))
    
    # Calculate matching score - percentage of words that match
    match_score1 = len(common_words) / len(words1)
    match_score2 = len(common_words) / len(words2)
    
    # Return the higher of the two scores - if either one has most of its words match
    return max(match_score1, match_score2)

# Function to get canonical organization name from any variation
def get_canonical_org_name(org_name):
    """
    Return the canonical (standard) form of an organization name
    """
    if not org_name:
        return ""
    
    org_lower = org_name.lower().strip()
    
    # Check direct matches to standards or variations
    for standard_name, variations in ORG_NAME_VARIATIONS.items():
        if org_lower == standard_name or org_lower in variations:
            return standard_name.title()
    
    # Return original if no match
    return org_name.strip()

# Page configuration
st.set_page_config(
    page_title="Energy Model Analysis Dashboard",
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and introduction
st.title("Energy Model Analysis Dashboard")
st.markdown("""
This dashboard analyzes energy modeling tools from the G-PST Open Tools Portal,
focusing on GitHub metrics to evaluate **usage**, **maturity**, and **user-friendliness**.

The dashboard helps answer questions like:
- Which energy models are most popular in the community?
- Which models are actively maintained and developed?
- How do different models compare across multiple dimensions?
- Which models have the strongest community support?

Use the sidebar to customize how different factors are weighted in the overall score.
""")

# Add a divider
st.markdown("---")

# Load data
@st.cache_data(ttl=3600)
def load_model_data():
    """Load the energy models base data from CSV"""
    data = pd.read_csv("gpst_open_tools.csv")
    # Clean up categories
    data['categories'] = data['categories'].str.split(';').apply(
        lambda x: [item.strip() for item in x] if isinstance(x, list) else [])
    return data

@st.cache_data(ttl=3600)
def load_github_data():
    """Load the GitHub metrics data from JSON file"""
    try:
        with open("github_data.json", "r") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except FileNotFoundError:
        st.error("GitHub data not found. Please run `github_data_fetcher.py` first to fetch the data.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading GitHub data: {str(e)}")
        return pd.DataFrame()

# Main app
def main():
    # Load base model data 
    base_data = load_model_data()
    
    # Check if GitHub data exists
    if not os.path.exists("github_data.json"):
        st.warning("""
        GitHub data not found. Please run the data fetcher script first:
        ```
        python github_data_fetcher.py
        ```
        """)
        st.stop()
    
    # Load GitHub data
    github_df = load_github_data()
    
    if github_df.empty:
        st.error("Failed to load GitHub data.")
        st.stop()
    
    # Show data overview
    st.header("Dataset Overview")
    st.write(f"Total models in dataset: {len(base_data)}")
    st.write(f"Models with GitHub metrics: {len(github_df)}")
    
    # Get last updated date from file modification time
    try:
        last_updated = datetime.fromtimestamp(os.path.getmtime("github_data.json"))
        st.write(f"Data last updated: {last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
    except:
        pass
    
    # Extract unique categories
    all_categories = []
    for cats in github_df['categories']:
        if isinstance(cats, list):
            all_categories.extend(cats)
        elif isinstance(cats, str):
            # Handle string format from JSON
            all_categories.extend([c.strip() for c in cats.split(';')])
    
    unique_categories = sorted(list(set(filter(None, all_categories))))
    st.write(f"Model categories: {', '.join(unique_categories)}")
    
    # Sidebar for scoring weights
    st.sidebar.title("Scoring Weights")
    st.sidebar.write("Customize weights for different metrics (must sum to 100%)")

    # Add explanation for the metrics
    with st.sidebar.expander("ℹ️ What do these metrics mean?"):
        st.markdown("""
        **Popularity (stars, forks)**
        - Based on GitHub stars (70%) and forks (30%)
        - Recent star growth (past 90 days) receives extra weight
        - Indicates how well-known and frequently used the model is

        **Maturity (age, maintenance, releases)**
        - Based on repository age (40%), recent updates (40%), and release count (20%)
        - Indicates how established and well-maintained the model is

        **Developer Activity (commit frequency)**
        - Based on the average number of weekly commits over the past year
        - Indicates how actively the model is being developed

        **Developer Community (contributors)**
        - Based on the number of unique contributors to the repository
        - Indicates how broad the development community is

        **User-friendliness (documentation, wiki, issues)**
        - Based on issue resolution ratio (50%), documentation presence (50%)
        - Documentation includes GitHub Wiki, GitHub Pages, and external docs (e.g., ReadTheDocs)
        - "Other Docs" column shows external documentation like ReadTheDocs sites
        - Indicates how well-documented and supported the model is
        """)

    # Add information section
    with st.sidebar.expander("🔍 Methodology & Data Sources"):
        st.markdown("""
        ### Data Sources
        - Base model information comes from the G-PST Open Tools Portal dataset
        - GitHub metrics are retrieved using the GitHub API
        - Last data refresh: {}
        
        ### Scoring Methodology
        Each model receives scores in 5 dimensions:
        
        1. **Popularity**: Reflects the model's adoption in the community
           - GitHub stars (70%)
           - GitHub forks (30%)
           - Recent star growth provides a bonus weight
        
        2. **Maturity**: Reflects how established and maintained the model is
           - Repository age (40%) - older repos get higher scores
           - Recent maintenance (40%) - recently updated repos get higher scores
           - Release count (20%) - more releases indicates better versioning
        
        3. **Developer Activity**: Reflects ongoing development
           - Average weekly commits over the past year (100%)
        
        4. **Developer Community**: Reflects community engagement
           - Number of unique contributors (100%)
        
        5. **User-friendliness**: Reflects documentation and support
           - Issue resolution ratio (50%)
           - Has GitHub pages (30%)
           - Has wiki (20%)
        
        All scores are normalized to a 0-100 scale for easier comparison.
        
        The overall score is calculated as a weighted combination of these five dimensions, 
        with weights that can be customized in the sidebar above.
        """.format(
            datetime.fromtimestamp(os.path.getmtime("github_data.json")).strftime('%Y-%m-%d') 
            if os.path.exists("github_data.json") else "N/A"
        ))

    # Default weights
    default_weights = {
        "popularity": 25,
        "maturity": 25,
        "developer_activity": 20,
        "developer_community": 15,
        "user_friendliness": 15
    }
    
    # Get weights from session state or use defaults
    if 'weights' not in st.session_state:
        st.session_state['weights'] = default_weights.copy()
        # For backward compatibility
        if "activity" in st.session_state['weights']:
            st.session_state['weights']["developer_activity"] = st.session_state['weights'].pop("activity")
        if "community" in st.session_state['weights']:
            st.session_state['weights']["developer_community"] = st.session_state['weights'].pop("community")
    
    # Create sliders for weights
    total_weight = 0
    weights = {}
    
    for metric, default in default_weights.items():
        weight = st.sidebar.slider(
            f"{metric.replace('_', ' ').title()} (%)",
            min_value=0,
            max_value=50,
            value=st.session_state['weights'][metric],
            step=5,
            key=f"weight_{metric}"
        )
        weights[metric] = weight
        total_weight += weight
    
    # Display total and warning if not 100%
    st.sidebar.write(f"Total weight: {total_weight}%")
    if total_weight != 100:
        st.sidebar.warning("⚠️ Weights should sum to 100%. Scores will be normalized.")
    
    # Normalize weights if needed
    if total_weight > 0:
        for metric in weights:
            weights[metric] = weights[metric] / total_weight
    else:
        # If all weights are 0, use default weights
        for metric, default in default_weights.items():
            weights[metric] = default / 100
    
    # Store weights in session state
    st.session_state['weights'] = {k: int(v * 100) for k, v in weights.items()}
    
    # Function to recalculate overall score with custom weights
    def recalculate_overall_score(df, weights):
        """Recalculate overall score with custom weights"""
        df = df.copy()
        df['overall_score'] = (
            weights['popularity'] * df['popularity_score'] +
            weights['maturity'] * df['maturity_score'] +
            weights['developer_activity'] * df['developer_activity_score'] +
            weights['developer_community'] * df['developer_community_score'] +
            weights['user_friendliness'] * df['user_friendliness_score']
        )
        return df
    
    # Apply custom weights to recalculate overall score
    if st.sidebar.button("Apply Custom Weights"):
        with st.spinner("Recalculating scores with new weights..."):
            github_df = recalculate_overall_score(github_df, weights)
            st.success("Scores recalculated with custom weights!")
    
    # Category selection for filtering
    st.header("Analyze by Category")
    selected_category = st.selectbox(
        "Select a category to analyze:", 
        ["All Categories"] + unique_categories
    )
    
    # Filter by category if needed
    if selected_category != "All Categories":
        filtered_df = github_df[github_df['categories'].apply(lambda x: selected_category in x if isinstance(x, list) else selected_category in str(x))]
    else:
        filtered_df = github_df.copy()
        
    if filtered_df.empty:
        st.warning(f"No models found in the '{selected_category}' category with GitHub metrics.")
    else:
        # Display top models by overall score
        st.subheader(f"Top Models in {selected_category} Category")
        top_models = filtered_df.sort_values('overall_score', ascending=False).head(10)
        
        # Create a bar chart of top models by overall score
        fig = px.bar(
            top_models,
            x='name',
            y='overall_score',
            labels={'name': 'Model', 'overall_score': 'Overall Score (0-100)'},
            title=f"Top Models by Overall Score in {selected_category} Category",
            color='overall_score',
            color_continuous_scale='Viridis',
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, key="top_models_chart")
        
        # Radar chart for top 5 models
        st.subheader("Multi-Dimensional Comparison")
        top5 = filtered_df.sort_values('overall_score', ascending=False).head(5)
        
        # Ensure we have unique models by taking the first occurrence of each model name
        top5 = top5.drop_duplicates(subset=['name'])
        
        # Prepare radar chart data
        metrics = [
            'popularity_score', 'maturity_score', 'developer_activity_score',
            'developer_community_score', 'user_friendliness_score'
        ]
        
        fig = go.Figure()
        
        for i, row in top5.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[m] for m in metrics],
                theta=[m.replace('_score', '').replace('_', ' ').capitalize() for m in metrics],
                fill='toself',
                name=row['name']
            ))
            
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title="Model Comparison Across Different Dimensions"
        )
        st.plotly_chart(fig, key="radar_chart_top5")
        
        # Detailed metrics table
        st.subheader("Detailed Metrics")
        
        # Calculate issue resolution ratio
        filtered_df['issue_resolution_ratio'] = filtered_df.apply(
            lambda row: round((row.get('closed_issues', 0) / max(row.get('total_issues', 1), 1)) * 100, 1) 
            if 'closed_issues' in row and 'total_issues' in row and row.get('total_issues', 0) > 0
            else "N/A", 
            axis=1
        )
        
        # Add a more descriptive message for cases with no issues
        filtered_df['issue_resolution_display'] = filtered_df.apply(
            lambda row: f"{row['issue_resolution_ratio']}%" 
            if row['issue_resolution_ratio'] != "N/A"
            else ("N/A" if row.get('total_issues', 0) == 0 else "No data"), 
            axis=1
        )
        
        # Make sure has_external_docs is available (if not, use False as default)
        if 'has_external_docs' not in filtered_df.columns:
            filtered_df['has_external_docs'] = False
        
        metrics_display = filtered_df[[
            'name', 'language', 'stars', 'forks', 'contributors_count',
            'open_issues', 'total_issues', 'closed_issues', 'releases_count', 
            'has_wiki', 'has_pages', 'has_external_docs', 
            'issue_resolution_display', 'overall_score'
        ]].sort_values('overall_score', ascending=False)
        
        metrics_display.columns = [
            'Model', 'Language', 'Stars', 'Forks', 'Contributors',
            'Open Issues', 'Total Issues', 'Closed Issues', 'Releases', 
            'Wiki', 'GitHub Pages', 'Other Docs', 
            'Issue Resolution (%)', 'Overall Score'
        ]
        
        # Convert boolean columns to more readable format
        if 'Wiki' in metrics_display.columns:
            metrics_display['Wiki'] = metrics_display['Wiki'].map({True: '✓', False: '✗'})
        if 'GitHub Pages' in metrics_display.columns:
            metrics_display['GitHub Pages'] = metrics_display['GitHub Pages'].map({True: '✓', False: '✗'})
        if 'Other Docs' in metrics_display.columns:
            metrics_display['Other Docs'] = metrics_display['Other Docs'].map({True: '✓', False: '✗'})
        
        st.dataframe(metrics_display)
        
        # Model comparison section
        st.header("Compare Specific Models")
        st.write("Select models to compare directly against each other")
        
        # Get unique model names from filtered data
        model_names = sorted(filtered_df['name'].unique().tolist())
        
        # Multi-select for models to compare
        selected_models = st.multiselect(
            "Select models to compare:",
            options=model_names,
            default=model_names[:2] if len(model_names) >= 2 else model_names
        )
        
        if len(selected_models) >= 2:
            # Filter data for selected models and ensure no duplicates
            compare_df = filtered_df[filtered_df['name'].isin(selected_models)].drop_duplicates(subset=['name'])
            
            # Create radar chart for comparison
            st.subheader("Model Comparison")
            
            # Prepare radar chart data
            metrics = [
                'popularity_score', 'maturity_score', 'developer_activity_score',
                'developer_community_score', 'user_friendliness_score'
            ]
            
            radar_fig = go.Figure()
            
            for i, row in compare_df.iterrows():
                radar_fig.add_trace(go.Scatterpolar(
                    r=[row[m] for m in metrics],
                    theta=[m.replace('_score', '').replace('_', ' ').capitalize() for m in metrics],
                    fill='toself',
                    name=row['name']
                ))
                
            radar_fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                title="Model Comparison Across Different Dimensions"
            )
            st.plotly_chart(radar_fig, key="model_comparison_radar")
            
            # Bar chart comparison for core metrics
            core_metrics = ['stars', 'forks', 'contributors_count', 'releases_count']
            bar_data = []
            
            for model in selected_models:
                model_data = compare_df[compare_df['name'] == model].iloc[0]
                for metric in core_metrics:
                    bar_data.append({
                        'Model': model,
                        'Metric': metric.replace('_count', '').capitalize(),
                        'Value': model_data[metric]
                    })
            
            bar_df = pd.DataFrame(bar_data)
            
            bar_fig = px.bar(
                bar_df,
                x='Model',
                y='Value',
                color='Metric',
                barmode='group',
                title="Comparison of Key Metrics"
            )
            st.plotly_chart(bar_fig, key="model_comparison_bar")
            
            # Growth Analysis section (time series data)
            st.subheader("Growth Analysis")
            st.write("Analyze how models have grown and evolved over time")
            
            # Check if time series data is available
            has_time_series = False
            for i, row in compare_df.iterrows():
                if 'time_series' in row and row['time_series'] and isinstance(row['time_series'], dict):
                    for key in ['stars_over_time', 'commits_by_week', 'code_frequency', 'issues_by_month']:
                        if key in row['time_series'] and row['time_series'][key]:
                            has_time_series = True
                            break
                    if has_time_series:
                        break
            
            if not has_time_series:
                st.info("Time series data is not available. Please run the data fetcher script with the latest version to collect time series data.")
            else:
                # Select which time series to display
                ts_metric = st.selectbox(
                    "Select time series metric:",
                    ["Stars Growth", "Commit Activity", "Code Changes", "Issue Activity"]
                )
                
                if ts_metric == "Stars Growth":
                    # Stars over time
                    st.write("### Stars Growth Over Time")
                    stars_fig = go.Figure()
                    
                    has_stars_data = False
                    for i, row in compare_df.iterrows():
                        if ('time_series' in row and 
                            isinstance(row['time_series'], dict) and 
                            'stars_over_time' in row['time_series'] and 
                            row['time_series']['stars_over_time']):
                            try:
                                stars_data = pd.DataFrame(row['time_series']['stars_over_time'])
                                if len(stars_data) > 0:
                                    stars_data['date'] = pd.to_datetime(stars_data['date'])
                                    stars_data = stars_data.sort_values('date')
                                    
                                    stars_fig.add_trace(go.Scatter(
                                        x=stars_data['date'],
                                        y=stars_data['total_stars'],
                                        mode='lines',
                                        name=row['name']
                                    ))
                                    has_stars_data = True
                            except Exception as e:
                                st.error(f"Error processing stars data for {row['name']}: {str(e)}")
                    
                    if has_stars_data:
                        stars_fig.update_layout(
                            title="Cumulative Stars Over Time",
                            xaxis_title="Date",
                            yaxis_title="Total Stars",
                            legend_title="Models",
                            hovermode="x unified"
                        )
                        st.plotly_chart(stars_fig, key="stars_growth_chart")
                    else:
                        st.info("No star growth data available for selected models.")
                    
                    # Calculate star growth rates
                    st.write("### Star Growth Rates")
                    growth_data = []
                    
                    for i, row in compare_df.iterrows():
                        if 'time_series' in row and 'stars_over_time' in row['time_series'] and row['time_series']['stars_over_time']:
                            stars_data = pd.DataFrame(row['time_series']['stars_over_time'])
                            if len(stars_data) > 0:
                                stars_data['date'] = pd.to_datetime(stars_data['date'])
                                stars_data = stars_data.sort_values('date')
                                
                                # Calculate days since first star
                                first_date = stars_data['date'].min()
                                last_date = stars_data['date'].max()
                                total_days = (last_date - first_date).days or 1  # Avoid division by zero
                                
                                # Calculate growth metrics
                                total_stars = stars_data['total_stars'].max()
                                avg_stars_per_day = total_stars / total_days
                                avg_stars_per_month = avg_stars_per_day * 30
                                avg_stars_per_year = avg_stars_per_day * 365
                                
                                # Calculate recent growth rate (last 90 days)
                                cutoff_date = pd.Timestamp(datetime.now() - timedelta(days=90)).tz_localize('UTC')
                                recent_stars = len(stars_data[stars_data['date'] >= cutoff_date])
                                recent_stars_per_day = recent_stars / 90
                                recent_stars_per_month = recent_stars_per_day * 30
                                
                                growth_data.append({
                                    'Model': row['name'],
                                    'Total Stars': total_stars,
                                    'First Star': first_date.strftime('%Y-%m-%d'),
                                    'Stars/Day (Overall)': round(avg_stars_per_day, 2),
                                    'Stars/Month (Overall)': round(avg_stars_per_month, 2),
                                    'Stars/Year (Overall)': round(avg_stars_per_year, 2),
                                    'Stars/Month (Last 90 Days)': round(recent_stars_per_month, 2)
                                })
                    
                    if growth_data:
                        growth_df = pd.DataFrame(growth_data)
                        st.dataframe(growth_df)
                    else:
                        st.info("No star growth data available for selected models.")
                    
                elif ts_metric == "Commit Activity":
                    # Commits over time
                    st.write("### Commit Activity Over Time")
                    commit_fig = go.Figure()
                    
                    has_commit_data = False
                    for i, row in compare_df.iterrows():
                        if ('time_series' in row and 
                            isinstance(row['time_series'], dict) and 
                            'commits_by_week' in row['time_series'] and 
                            row['time_series']['commits_by_week']):
                            try:
                                commit_data = pd.DataFrame(row['time_series']['commits_by_week'])
                                if len(commit_data) > 0:
                                    commit_data['date'] = pd.to_datetime(commit_data['date'])
                                    commit_data = commit_data.sort_values('date')
                                    
                                    commit_fig.add_trace(go.Scatter(
                                        x=commit_data['date'],
                                        y=commit_data['total_commits'],
                                        mode='lines',
                                        name=row['name']
                                    ))
                                    has_commit_data = True
                            except Exception as e:
                                st.error(f"Error processing commit data for {row['name']}: {str(e)}")
                    
                    if has_commit_data:
                        commit_fig.update_layout(
                            title="Weekly Commits Over Time",
                            xaxis_title="Date",
                            yaxis_title="Commits per Week",
                            legend_title="Models",
                            hovermode="x unified"
                        )
                        st.plotly_chart(commit_fig, key="commits_time_chart")
                    else:
                        st.info("No commit activity data available for selected models.")
                    
                    # Weekly average commits
                    st.write("### Average Weekly Commits")
                    avg_commit_data = []
                    
                    for i, row in compare_df.iterrows():
                        if ('time_series' in row and 
                            isinstance(row['time_series'], dict) and 
                            'commits_by_week' in row['time_series'] and 
                            row['time_series']['commits_by_week']):
                            try:
                                commit_data = pd.DataFrame(row['time_series']['commits_by_week'])
                                if len(commit_data) > 0:
                                    avg_commits = commit_data['total_commits'].mean()
                                    max_commits = commit_data['total_commits'].max()
                                    recent_commits = commit_data['total_commits'].iloc[-12:].mean() if len(commit_data) >= 12 else float('nan')
                                    
                                    avg_commit_data.append({
                                        'Model': row['name'],
                                        'Avg Weekly Commits': round(avg_commits, 1),
                                        'Max Weekly Commits': max_commits,
                                        'Recent Avg (3 months)': round(recent_commits, 1)
                                    })
                            except Exception as e:
                                st.error(f"Error processing commit statistics for {row['name']}: {str(e)}")
                    
                    if avg_commit_data:
                        avg_commit_df = pd.DataFrame(avg_commit_data)
                        st.dataframe(avg_commit_df)
                    else:
                        st.info("No commit statistics available for selected models.")
                    
                elif ts_metric == "Code Changes":
                    # Code frequency - additions/deletions
                    st.write("### Code Changes Over Time")
                    code_fig = go.Figure()
                    
                    has_code_data = False
                    for i, row in compare_df.iterrows():
                        if ('time_series' in row and 
                            isinstance(row['time_series'], dict) and 
                            'code_frequency' in row['time_series'] and 
                            row['time_series']['code_frequency']):
                            try:
                                code_data = pd.DataFrame(row['time_series']['code_frequency'])
                                if len(code_data) > 0:
                                    code_data['date'] = pd.to_datetime(code_data['date'])
                                    code_data = code_data.sort_values('date')
                                    
                                    code_fig.add_trace(go.Scatter(
                                        x=code_data['date'],
                                        y=code_data['additions'],
                                        mode='lines',
                                        name=f"{row['name']} - Additions",
                                        line=dict(color=px.colors.qualitative.Plotly[i*2 % len(px.colors.qualitative.Plotly)])
                                    ))
                                    
                                    code_fig.add_trace(go.Scatter(
                                        x=code_data['date'],
                                        y=-code_data['deletions'],  # Negative for deletions
                                        mode='lines',
                                        name=f"{row['name']} - Deletions",
                                        line=dict(color=px.colors.qualitative.Plotly[(i*2+1) % len(px.colors.qualitative.Plotly)], dash='dot')
                                    ))
                                    has_code_data = True
                            except Exception as e:
                                st.error(f"Error processing code changes data for {row['name']}: {str(e)}")
                    
                    if has_code_data:
                        code_fig.update_layout(
                            title="Weekly Code Additions and Deletions",
                            xaxis_title="Date",
                            yaxis_title="Lines Changed",
                            legend_title="Models",
                            hovermode="x unified"
                        )
                        st.plotly_chart(code_fig, key="code_changes_chart")
                    else:
                        st.info("No code changes data available for selected models.")
                    
                elif ts_metric == "Issue Activity":
                    # Issues over time
                    st.write("### Issue Activity Over Time")
                    issue_fig = go.Figure()
                    
                    has_issue_data = False
                    for i, row in compare_df.iterrows():
                        if ('time_series' in row and 
                            isinstance(row['time_series'], dict) and 
                            'issues_by_month' in row['time_series'] and 
                            row['time_series']['issues_by_month']):
                            try:
                                issue_data = pd.DataFrame(row['time_series']['issues_by_month'])
                                if len(issue_data) > 0:
                                    issue_data['month'] = pd.to_datetime(issue_data['month'] + '-01')
                                    issue_data = issue_data.sort_values('month')
                                    
                                    issue_fig.add_trace(go.Scatter(
                                        x=issue_data['month'],
                                        y=issue_data['opened'],
                                        mode='lines',
                                        name=f"{row['name']} - Opened",
                                        line=dict(color=px.colors.qualitative.Plotly[i*2 % len(px.colors.qualitative.Plotly)])
                                    ))
                                    
                                    issue_fig.add_trace(go.Scatter(
                                        x=issue_data['month'],
                                        y=issue_data['closed'],
                                        mode='lines',
                                        name=f"{row['name']} - Closed",
                                        line=dict(color=px.colors.qualitative.Plotly[(i*2+1) % len(px.colors.qualitative.Plotly)], dash='dot')
                                    ))
                                    has_issue_data = True
                            except Exception as e:
                                st.error(f"Error processing issue activity data for {row['name']}: {str(e)}")
                    
                    if has_issue_data:
                        issue_fig.update_layout(
                            title="Monthly Issues Opened and Closed",
                            xaxis_title="Month",
                            yaxis_title="Number of Issues",
                            legend_title="Models",
                            hovermode="x unified"
                        )
                        st.plotly_chart(issue_fig, key="issue_activity_chart")
                    
                    # Issue response metrics
                    st.write("### Issue Response Metrics")
                    issue_metrics = []
                    
                    for i, row in compare_df.iterrows():
                        if ('time_series' in row and 
                            isinstance(row['time_series'], dict) and 
                            'issues' in row['time_series'] and 
                            row['time_series']['issues']):
                            try:
                                issues = pd.DataFrame(row['time_series']['issues'])
                                if len(issues) > 0:
                                    # Calculate metrics
                                    total_issues = len(issues)
                                    closed_issues = len(issues[issues['state'] == 'closed'])
                                    
                                    # Calculate resolution time for closed issues
                                    if 'closed_at' in issues.columns:
                                        issues['created_at'] = pd.to_datetime(issues['created_at'])
                                        issues['closed_at'] = pd.to_datetime(issues['closed_at'])
                                        
                                        # Calculate resolution time in days
                                        resolution_times = []
                                        for _, issue in issues[issues['state'] == 'closed'].iterrows():
                                            if pd.notna(issue['closed_at']):
                                                resolution_time = (issue['closed_at'] - issue['created_at']).total_seconds() / (24 * 3600)  # days
                                                resolution_times.append(resolution_time)
                                                
                                        if resolution_times:
                                            avg_resolution_time = sum(resolution_times) / len(resolution_times)
                                            median_resolution_time = sorted(resolution_times)[len(resolution_times) // 2]
                                        else:
                                            avg_resolution_time = float('nan')
                                            median_resolution_time = float('nan')
                                    else:
                                        avg_resolution_time = float('nan')
                                        median_resolution_time = float('nan')
                                        
                                    issue_metrics.append({
                                        'Model': row['name'],
                                        'Total Issues': total_issues,
                                        'Closed Issues': closed_issues,
                                        'Close Rate': f"{round(closed_issues / total_issues * 100, 1)}%" if total_issues > 0 else "N/A",
                                        'Avg Resolution (days)': round(avg_resolution_time, 1) if not np.isnan(avg_resolution_time) else "N/A",
                                        'Median Resolution (days)': round(median_resolution_time, 1) if not np.isnan(median_resolution_time) else "N/A"
                                    })
                            except Exception as e:
                                st.error(f"Error processing issue metrics for {row['name']}: {str(e)}")
                    
                    if issue_metrics:
                        issue_metrics_df = pd.DataFrame(issue_metrics)
                        st.dataframe(issue_metrics_df)
                    else:
                        st.info("No issue metrics data available for selected models.")
            
            # Detailed comparison table
            st.subheader("Detailed Comparison")
            
            # Select columns for comparison
            comparison_cols = [
                'name', 'language', 'stars', 'forks', 'contributors_count',
                'open_issues', 'releases_count', 'overall_score',
                'popularity_score', 'maturity_score', 'developer_activity_score',
                'developer_community_score', 'user_friendliness_score'
            ]
            
            # Create comparison table
            compare_display = compare_df[comparison_cols].sort_values('overall_score', ascending=False)
            
            # Rename columns for display
            compare_display.columns = [
                'Model', 'Language', 'Stars', 'Forks', 'Contributors',
                'Open Issues', 'Releases', 'Overall Score',
                'Popularity Score', 'Maturity Score', 'Developer Activity Score',
                'Developer Community Score', 'User-Friendliness Score'
            ]
            
            st.dataframe(compare_display)
        
        else:
            st.info("Please select at least 2 models to compare.")
        
        # Download data button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name=f"energy_models_{selected_category.replace(' ', '_')}.csv",
            mime="text/csv",
        )

        # User Analysis Section
        st.header("User Analysis")

        # Check if user analysis data exists
        user_analysis_file = "user_analysis_results.json"
        if os.path.exists(user_analysis_file):
            with open(user_analysis_file, 'r') as f:
                user_analysis = json.load(f)
            
            if user_analysis:
                st.write("Analysis of GitHub users who have interacted with the repositories")
                
                # Select repository to view
                available_repos = [analysis["repo_name"] for analysis in user_analysis]
                selected_repo = st.selectbox(
                    "Select repository to analyze:", 
                    options=["All Repositories"] + available_repos
                )
                
                # Initialize locations and loc_counts before the if/else branches
                locations = []
                loc_counts = []
                
                if selected_repo == "All Repositories":
                    # Combine data from all repos
                    total_users = sum(analysis["total_users"] for analysis in user_analysis)
                    
                    # Combine classifications
                    combined_class_counts = {}
                    for analysis in user_analysis:
                        for cls, count in analysis["classification_counts"].items():
                            if cls in combined_class_counts:
                                combined_class_counts[cls] += count
                            else:
                                combined_class_counts[cls] = count
                    
                    # Create pie chart of classifications
                    st.subheader("User Types Across All Repositories")
                    classes = list(combined_class_counts.keys())
                    counts = list(combined_class_counts.values())
                    
                    fig = px.pie(
                        values=counts,
                        names=classes,
                        title=f"Distribution of {total_users} Users by Type",
                        color_discrete_sequence=px.colors.qualitative.Plotly
                    )
                    st.plotly_chart(fig, key="all_repos_user_types_pie")
                    
                    # Combine organizations
                    combined_orgs = {}
                    for analysis in user_analysis:
                        for org, count in analysis["top_organizations"].items():
                            if org in combined_orgs:
                                combined_orgs[org] += count
                            else:
                                combined_orgs[org] = count
                    
                    # Show top organizations
                    st.subheader("Top Organizations Engaging with Repositories")
                    
                    # Sort organizations by count
                    sorted_orgs = sorted(combined_orgs.items(), key=lambda x: x[1], reverse=True)
                    top_orgs = sorted_orgs[:15]
                    
                    fig = px.bar(
                        x=[org for org, _ in top_orgs],
                        y=[count for _, count in top_orgs],
                        labels={'x': 'Organization', 'y': 'Number of Users'},
                        title="Top 15 Organizations",
                        color=[count for _, count in top_orgs],
                        color_continuous_scale=px.colors.sequential.Viridis
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, key="all_repos_top_orgs_bar")
                    
                    # Breakdown of contributors
                    st.subheader("Contributors by Type")
                    combined_contributor_class = {}
                    for analysis in user_analysis:
                        for cls, count in analysis.get("contributor_class_counts", {}).items():
                            if cls in combined_contributor_class:
                                combined_contributor_class[cls] += count
                            else:
                                combined_contributor_class[cls] = count
                    
                    contributor_classes = list(combined_contributor_class.keys())
                    contributor_counts = list(combined_contributor_class.values())
                    
                    fig = px.pie(
                        values=contributor_counts,
                        names=contributor_classes,
                        title="Distribution of Contributors by Type",
                        color_discrete_sequence=px.colors.qualitative.Plotly
                    )
                    st.plotly_chart(fig, key="all_repos_contributor_pie")
                    
                    # Top Organizations by Type
                    st.header("Top Organizations by Type")
                    
                    academic_tab, professional_tab, industry_tab, financial_tab, utility_tab, government_tab = st.tabs([
                        "Academic", "Professional", "Industry", "Financial", "Utility", "Government"
                    ])
                    
                    # Get combined organizations by type
                    combined_orgs_by_type = get_all_organizations_by_type(user_analysis)
                    
                    if combined_orgs_by_type:
                        # Create tabs for each organization type
                        org_types = ['academic', 'professional', 'industry', 'financial', 'utility', 'government']
                        tabs = [academic_tab, professional_tab, industry_tab, financial_tab, utility_tab, government_tab]
                        
                        for i, org_type in enumerate(org_types):
                            tab = tabs[i]
                            orgs_data = combined_orgs_by_type.get(org_type, {})
                            with tab:
                                if orgs_data:
                                    # Create a DataFrame for the organizations
                                    orgs_df = pd.DataFrame({
                                        'Organization': list(orgs_data.keys()),
                                        'Users': list(orgs_data.values())
                                    }).sort_values('Users', ascending=False)
                                    
                                    # Display as a bar chart
                                    fig = px.bar(
                                        orgs_df,
                                        x='Organization',
                                        y='Users',
                                        title=f"Top {org_type.capitalize()} Organizations Across All Repositories",
                                        color='Users',
                                        color_continuous_scale=px.colors.sequential.Viridis
                                    )
                                    fig.update_layout(xaxis_tickangle=-45)
                                    st.plotly_chart(fig, key=f"org_type_bar_{org_type}")
                                    
                                    # Also display as a table
                                    st.dataframe(orgs_df.head(10))
                                else:
                                    st.info(f"No {org_type} organizations found in the dataset.")
                    else:
                        st.info("No organization by type data available.")
                    
                    # Locations
                    st.subheader("Geographic Distribution")
                    # Combine locations from all repositories when All Repositories is selected
                    combined_locations = {}
                    for analysis in user_analysis:
                        for location, count in analysis["top_locations"].items():
                            if location in combined_locations:
                                combined_locations[location] += count
                            else:
                                combined_locations[location] = count
                    
                    # Sort locations by count and take top 20
                    sorted_locations = sorted(combined_locations.items(), key=lambda x: x[1], reverse=True)
                    top_locations = dict(sorted_locations[:20])
                    
                    locations = list(top_locations.keys())
                    loc_counts = list(top_locations.values())
                else:
                    # Find the analysis for the selected repository
                    repo_analysis = next((analysis for analysis in user_analysis if analysis["repo_name"] == selected_repo), None)
                    
                    if not repo_analysis:
                        st.error(f"No analysis data found for {selected_repo}")
                    else:
                        # Show basic statistics
                        st.subheader(f"Analysis for {selected_repo}")
                        st.write(f"Total users: {repo_analysis.get('total_users', 0)}")
                        
                        # Locations
                        st.subheader("Geographic Distribution")
                        if "top_locations" in repo_analysis and repo_analysis["top_locations"]:
                            locations = list(repo_analysis["top_locations"].keys())
                            loc_counts = list(repo_analysis["top_locations"].values())
                
                # Now locations and loc_counts are always defined
                if locations:
                    # Bar chart of top locations
                    fig = px.bar(
                        x=locations,
                        y=loc_counts,
                        labels={'x': 'Location', 'y': 'Number of Users'},
                        title="Top Locations",
                        color=loc_counts,
                        color_continuous_scale=px.colors.sequential.Viridis
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, key=f"top_locations_bar_{selected_repo}")
                    
                    # Add a world map visualization
                    st.subheader("Geographic Map")
                    try:
                        # Use the pre-processed country data
                        if selected_repo == "All Repositories":
                            # Combine country counts from all repositories
                            combined_country_counts = {}
                            for analysis in user_analysis:
                                if "country_counts" in analysis:
                                    for country, count in analysis["country_counts"].items():
                                        if country in combined_country_counts:
                                            combined_country_counts[country] += count
                                        else:
                                            combined_country_counts[country] = count
                            country_counts = combined_country_counts
                        else:
                            # Use the country data from the selected repository
                            country_counts = repo_analysis.get("country_counts", {})
                        
                        if country_counts:
                            # Create dataframe for choropleth map
                            country_df = pd.DataFrame({
                                'country': list(country_counts.keys()),
                                'users': list(country_counts.values())
                            })
                            
                            # Create choropleth map
                            map_title = f"User Distribution by Country ({selected_repo})" if selected_repo != "All Repositories" else "User Distribution by Country (All Repositories)"
                            fig = px.choropleth(
                                country_df,
                                locations='country',
                                locationmode='country names',
                                color='users',
                                hover_name='country',
                                color_continuous_scale=px.colors.sequential.Viridis,
                                title=map_title
                            )
                            fig.update_layout(
                                geo=dict(
                                    showframe=True,
                                    showcoastlines=True,
                                    projection_type='equirectangular',
                                    landcolor='rgb(243, 243, 243)',  # Light gray land
                                    oceancolor='rgb(220, 240, 255)',  # Light blue ocean
                                    coastlinecolor='rgb(80, 80, 80)',  # Darker coast lines
                                    countrycolor='rgb(150, 150, 150)'  # Gray country borders
                                ),
                                margin=dict(l=0, r=0, t=50, b=0),  # Tight margins
                                paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
                                plot_bgcolor='rgba(0,0,0,0)'    # Transparent plot area
                            )
                            st.plotly_chart(fig, key=f"country_map_{selected_repo}")
                        else:
                            st.info("No country data available. Run the GitHub user analysis script to collect this data.")
                    except Exception as e:
                        st.error(f"Error creating map visualization: {str(e)}")
                        st.info("Try running the GitHub user analysis script with the latest version.")
                else:
                    st.info("No location data available.")

                # User Types - show for both All Repositories and individual repos but with different layouts
                if selected_repo == "All Repositories":
                    # Already handled in the All Repositories section
                    pass
                else:
                    # Classification by organization type for individual repositories
                    st.subheader("User Types")
                    if repo_analysis and "classification_counts" in repo_analysis and repo_analysis["classification_counts"]:
                        # Define custom colors for the classifications
                        color_map = {
                            'academic': '#4285F4',  # blue
                            'industry': '#34A853',  # green
                            'utility': '#FBBC05',   # yellow
                            'government': '#EA4335',  # red
                            'financial': '#8205B4',  # purple
                            'ngo': '#00BCD4',       # cyan
                            'professional': '#FF9800',  # orange
                            'research_organization': '#4DB6AC',  # teal
                            'rto': '#FF5722',       # deep orange
                            'unknown': '#9E9E9E'    # grey
                        }
                        
                        # Create a dataframe for better visualization
                        class_df = pd.DataFrame({
                            'Type': list(repo_analysis["classification_counts"].keys()),
                            'Users': list(repo_analysis["classification_counts"].values())
                        })
                        
                        # Sort by number of users
                        class_df = class_df.sort_values('Users', ascending=False)
                        
                        # Create a nicer bar chart with custom colors
                        colors = [color_map.get(cls, '#9E9E9E') for cls in class_df['Type']]
                        
                        fig = px.bar(
                            class_df,
                            x='Type',
                            y='Users',
                            color='Type',
                            title="User Types by Count",
                            color_discrete_sequence=colors,
                            labels={'Type': 'User Type', 'Users': 'Number of Users'}
                        )
                        st.plotly_chart(fig, key=f"user_types_bar_{selected_repo}")
                    else:
                        st.info("No classification data available for this repository.")

                # Organization Search Section - Always show regardless of repository selection
                st.header("Organization Search")
                st.write("Search for organizations and see associated users and their model interactions")
                
                # Organization search
                st.subheader("Search for an organization:")
                search_term = st.text_input("Search for an organization:", "")
                
                if search_term:
                    # Find matching organizations
                    matching_orgs = search_organizations(search_term, user_analysis)
                    
                    if matching_orgs:
                        # Allow user to select from matching organizations
                        st.subheader("Select an organization:")
                        org_options = [org['name'] for org in matching_orgs]
                        selected_org = st.selectbox("", org_options)
                        
                        if selected_org:
                            # Display users from this organization across all repositories
                            users = []
                            
                            # Find all variations that map to the selected canonical name
                            selected_canonical = get_canonical_org_name(selected_org)
                            matching_variations = [selected_canonical.lower()]
                            
                            # Add all variations of this canonical name
                            for std_name, variations in ORG_NAME_VARIATIONS.items():
                                if std_name.title() == selected_canonical:
                                    matching_variations.extend(variations)
                                    break
                            
                            for repo_analysis in user_analysis:
                                # Handle both new and old data formats
                                if 'users_data' in repo_analysis:
                                    users_data = repo_analysis['users_data']
                                    for interaction_type in ['stargazers', 'forkers', 'contributors', 'watchers']:
                                        if interaction_type in users_data:
                                            for user in users_data[interaction_type]:
                                                # Check if user's company matches any variation of the selected org
                                                if user.get('company'):
                                                    user_company = user['company'].lower().strip()
                                                    
                                                    # Check against variations or direct match to canonical
                                                    if (user.get('dashboard_org_name') and user['dashboard_org_name'].lower() == selected_canonical.lower()) or \
                                                       any(var in user_company for var in matching_variations):
                                                        # Avoid duplicates
                                                        if not any(u['login'] == user['login'] for u in users):
                                                            users.append({
                                                                'login': user['login'],
                                                                'name': user.get('name', ''),
                                                                'company': user.get('company', ''),
                                                                'match_quality': user.get('org_match_confidence', 1.0) if 'org_match_confidence' in user else 1.0,
                                                                'models': set()
                                                            })
                                # Older data format
                                elif 'data' in repo_analysis:
                                    for interaction_type in ['stargazers', 'forkers', 'contributors', 'watchers']:
                                        if interaction_type in repo_analysis['data']:
                                            for user in repo_analysis['data'][interaction_type]:
                                                # Check if user's company matches any variation of the selected org
                                                if user.get('company'):
                                                    user_company = user['company'].lower().strip()
                                                    
                                                    # Check against variations or direct match to canonical
                                                    if (user.get('dashboard_org_name') and user['dashboard_org_name'].lower() == selected_canonical.lower()) or \
                                                       any(var in user_company for var in matching_variations):
                                                        # Avoid duplicates
                                                        if not any(u['login'] == user['login'] for u in users):
                                                            users.append({
                                                                'login': user['login'],
                                                                'name': user.get('name', ''),
                                                                'company': user.get('company', ''),
                                                                'match_quality': user.get('org_match_confidence', 1.0) if 'org_match_confidence' in user else 1.0,
                                                                'models': set()
                                                            })
                            
                            # Get models for each user
                            for repo_analysis in user_analysis:
                                repo_name = repo_analysis['repo_name']
                                # Handle both data formats
                                if 'users_data' in repo_analysis:
                                    users_data = repo_analysis['users_data']
                                    for interaction_type in ['stargazers', 'forkers', 'contributors', 'watchers']:
                                        if interaction_type in users_data:
                                            for user in users_data[interaction_type]:
                                                # Find matching user in our list
                                                for u in users:
                                                    if u['login'] == user['login']:
                                                        u['models'].add(repo_name)
                                elif 'data' in repo_analysis:
                                    for interaction_type in ['stargazers', 'forkers', 'contributors', 'watchers']:
                                        if interaction_type in repo_analysis['data']:
                                            for user in repo_analysis['data'][interaction_type]:
                                                # Find matching user in our list
                                                for u in users:
                                                    if u['login'] == user['login']:
                                                        u['models'].add(repo_name)
                            
                            # Convert sets to lists for display
                            for u in users:
                                u['models'] = ', '.join(sorted(u['models']))
                            
                            # Display as table
                            if users:
                                st.subheader(f"Users from {selected_org}")
                                st.write(f"Found {len(users)} users from this organization")
                                
                                # Create DataFrame for display
                                users_df = pd.DataFrame(users)
                                st.dataframe(users_df)
                            else:
                                st.info("No users found for this organization")
                    else:
                        st.info("No matching organizations found")
                else:
                    st.info("Enter an organization name to search")
            else:
                st.info("No user analysis data available.")
        else:
            st.info("""
            User analysis data not available. Run the GitHub user analysis script to collect this data:
            ```
            python github_user_analysis.py
            ```
            """)

    # Add refresh data button
    st.sidebar.title("Data Management")
    st.sidebar.write("To refresh the GitHub data, run the data fetcher script:")
    st.sidebar.code("python github_data_fetcher.py")
    st.sidebar.write("To analyze GitHub users, run the user analysis script:")
    st.sidebar.code("python github_user_analysis.py")

def search_organizations(search_term, user_analysis, threshold=0.6):
    """Search for organizations across all repositories"""
    matching_orgs = []
    
    for repo_analysis in user_analysis:
        # Process organizations from organization_name_mapping
        if 'organization_name_mapping' in repo_analysis:
            org_map = repo_analysis['organization_name_mapping']
            for org_name in org_map.values():
                # Get canonical name for consistency
                canonical_name = get_canonical_org_name(org_name)
                
                # Match against canonical name or original
                match_score = max(
                    org_names_match(search_term, canonical_name),
                    org_names_match(search_term, org_name)
                )
                
                if match_score >= threshold and canonical_name not in [o['name'] for o in matching_orgs]:
                    matching_orgs.append({
                        'name': canonical_name,
                        'score': match_score
                    })
        
        # Also search through all users - newer format
        if 'users_data' in repo_analysis:
            for interaction_type in ['stargazers', 'forkers', 'contributors', 'watchers']:
                if interaction_type in repo_analysis['users_data']:
                    for user in repo_analysis['users_data'][interaction_type]:
                        if user.get('company'):
                            org_name = user['company']
                            canonical_name = get_canonical_org_name(org_name)
                            
                            match_score = max(
                                org_names_match(search_term, canonical_name),
                                org_names_match(search_term, org_name)
                            )
                            
                            if match_score >= threshold and canonical_name not in [o['name'] for o in matching_orgs]:
                                matching_orgs.append({
                                    'name': canonical_name,
                                    'score': match_score
                                })
        # Older format
        elif 'data' in repo_analysis:
            for interaction_type in ['stargazers', 'forkers', 'contributors', 'watchers']:
                if interaction_type in repo_analysis['data']:
                    for user in repo_analysis['data'][interaction_type]:
                        if user.get('company'):
                            org_name = user['company']
                            canonical_name = get_canonical_org_name(org_name)
                            
                            match_score = max(
                                org_names_match(search_term, canonical_name),
                                org_names_match(search_term, org_name)
                            )
                            
                            if match_score >= threshold and canonical_name not in [o['name'] for o in matching_orgs]:
                                matching_orgs.append({
                                    'name': canonical_name,
                                    'score': match_score
                                })
    
    # Sort by match score
    matching_orgs.sort(key=lambda x: x['score'], reverse=True)
    return matching_orgs

def get_all_organizations_by_type(user_analysis):
    """Gather organizations by type across all repositories"""
    combined_orgs_by_type = {}
    
    for analysis in user_analysis:
        if "organizations_by_type" in analysis:
            for org_type, orgs in analysis["organizations_by_type"].items():
                if org_type not in combined_orgs_by_type:
                    combined_orgs_by_type[org_type] = Counter()
                combined_orgs_by_type[org_type].update(orgs)
        
        # If we don't have pre-calculated organizations_by_type, process from raw data
        # This is a fallback for older data formats
        elif 'users_data' in analysis or 'data' in analysis:
            # First gather all users
            all_users = []
            if 'users_data' in analysis:
                for interaction_type in ['stargazers', 'forkers', 'contributors', 'watchers']:
                    if interaction_type in analysis['users_data']:
                        for user in analysis['users_data'][interaction_type]:
                            if user.get('login') and user.get('classification') and user.get('classification') != 'unknown':
                                all_users.append(user)
            elif 'data' in analysis:
                for interaction_type in ['stargazers', 'forkers', 'contributors', 'watchers']:
                    if interaction_type in analysis['data']:
                        for user in analysis['data'][interaction_type]:
                            if user.get('login') and user.get('classification') and user.get('classification') != 'unknown':
                                all_users.append(user)
            
            # Process users to get organizations by type
            orgs_by_type = {}
            for user in all_users:
                org_name = None
                # Use high-confidence dashboard org name when available
                if user.get('dashboard_org_name') and user.get('org_match_confidence', 0) >= 0.8:
                    org_name = user.get('dashboard_org_name')
                # Otherwise use company name directly, but only if no dashboard name was assigned
                elif user.get('company') and not user.get('dashboard_org_name'):
                    org_name = user.get('company', '').strip()
                    
                # Only proceed if we have both an org name and a classification
                if org_name and user['classification'] != 'unknown':
                    cls = user['classification']
                    
                    if cls not in orgs_by_type:
                        orgs_by_type[cls] = Counter()
                    
                    orgs_by_type[cls][org_name] += 1
            
            # Add to combined organizations
            for org_type, orgs in orgs_by_type.items():
                if org_type not in combined_orgs_by_type:
                    combined_orgs_by_type[org_type] = Counter()
                combined_orgs_by_type[org_type].update(orgs)
    
    return combined_orgs_by_type

if __name__ == "__main__":
    main() 