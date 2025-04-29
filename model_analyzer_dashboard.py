import streamlit as st
import pandas as pd
import requests
import time
import re
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    print("Error importing seaborn, trying to install...")
    import subprocess
    subprocess.check_call(["pip", "install", "seaborn"])
    import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from github import Github
import os
from dotenv import load_dotenv

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
focusing on GitHub metrics to evaluate usage, maturity, and user-friendliness.
""")

# Load data
@st.cache_data(ttl=3600)
def load_data():
    """Load the energy models data from CSV"""
    data = pd.read_csv("gpst_open_tools.csv")
    # Clean up categories
    data['categories'] = data['categories'].str.split(';').apply(
        lambda x: [item.strip() for item in x] if isinstance(x, list) else [])
    return data

# GitHub API functions
@st.cache_data(ttl=3600)
def get_github_token():
    """Get GitHub token from environment or from user input"""
    load_dotenv()  # Load from .env file if available
    token = os.getenv("GITHUB_TOKEN")
    if token:
        return token
    
    # Try to get from Streamlit secrets, handle the case when secrets are not set up
    try:
        return st.secrets.get("GITHUB_TOKEN", "")
    except:
        return ""  # Return empty string if no secrets found

def extract_github_info(url):
    """Extract username and repo name from GitHub URL"""
    if not isinstance(url, str) or 'github.com' not in url:
        return None, None
    
    # Extract username/repo from GitHub URL
    pattern = r'github\.com/([^/]+)/([^/]+)'
    match = re.search(pattern, url)
    if match:
        username, repo = match.groups()
        # Remove .git extension if present
        repo = repo.replace('.git', '')
        return username, repo
    return None, None

@st.cache_data(ttl=3600)
def get_github_metrics(username, repo, token):
    """Get GitHub metrics for a repository"""
    if not username or not repo:
        return None
    
    # Initialize GitHub API client with token
    g = Github(token)
    
    try:
        # Get repository
        repository = g.get_repo(f"{username}/{repo}")
        
        # Get metrics
        metrics = {
            "stars": repository.stargazers_count,
            "forks": repository.forks_count,
            "open_issues": repository.open_issues_count,
            "watchers": repository.subscribers_count,
            "created_at": repository.created_at.isoformat(),
            "updated_at": repository.updated_at.isoformat(),
            "pushed_at": repository.pushed_at.isoformat(),
            "size": repository.size,
            "language": repository.language,
            "has_wiki": repository.has_wiki,
            "has_pages": repository.has_pages,
            "license": repository.license.name if repository.license else "None",
        }
        
        # Get contributors count
        contributors = list(repository.get_contributors())
        metrics["contributors_count"] = len(contributors)
        
        # Get commit activity (last 52 weeks)
        commit_activity = list(repository.get_stats_commit_activity())
        if commit_activity:
            metrics["weekly_commits"] = [week.total for week in commit_activity]
            metrics["total_commits_year"] = sum(metrics["weekly_commits"])
            metrics["avg_weekly_commits"] = np.mean(metrics["weekly_commits"])
        else:
            metrics["weekly_commits"] = []
            metrics["total_commits_year"] = 0
            metrics["avg_weekly_commits"] = 0
        
        # Get recent releases
        releases = list(repository.get_releases())
        metrics["releases_count"] = len(releases)
        if releases:
            metrics["latest_release_date"] = releases[0].published_at.isoformat()
        else:
            metrics["latest_release_date"] = None
            
        return metrics
    
    except Exception as e:
        st.error(f"Error fetching GitHub metrics for {username}/{repo}: {str(e)}")
        return None

# Process all data and get GitHub metrics
@st.cache_data(ttl=3600)
def process_github_data(data, token):
    """Process all repositories and get their GitHub metrics"""
    github_data = []
    
    for idx, row in data.iterrows():
        if pd.isna(row['source_code_url']) or 'github.com' not in row['source_code_url']:
            continue
            
        username, repo = extract_github_info(row['source_code_url'])
        if username and repo:
            metrics = get_github_metrics(username, repo, token)
            if metrics:
                # Add model info to metrics
                metrics['name'] = row['name']
                metrics['categories'] = row['categories']
                metrics['description'] = row['description']
                metrics['programming_languages'] = row['programming_languages']
                metrics['url'] = row['url']
                metrics['source_code_url'] = row['source_code_url']
                github_data.append(metrics)
                
            # Avoid rate limiting
            time.sleep(0.5)
    
    return pd.DataFrame(github_data)

# Calculate model scores
def calculate_scores(github_df):
    """Calculate various scores for each model"""
    if github_df.empty:
        return github_df
    
    # Create a copy to avoid modifying the original
    df = github_df.copy()
    
    # Age score (0-1): older repositories get higher scores
    if 'created_at' in df.columns:
        df['created_at_dt'] = pd.to_datetime(df['created_at'])
        oldest = df['created_at_dt'].min()
        newest = df['created_at_dt'].max()
        time_range = (newest - oldest).total_seconds()
        if time_range > 0:
            df['age_score'] = df['created_at_dt'].apply(
                lambda x: 1 - ((newest - x).total_seconds() / time_range)
            )
        else:
            df['age_score'] = 1.0
    else:
        df['age_score'] = 0.0
        
    # Maintenance score (0-1): recently updated repositories get higher scores
    if 'pushed_at' in df.columns:
        df['pushed_at_dt'] = pd.to_datetime(df['pushed_at'])
        now = datetime.now()
        one_year_ago = now - timedelta(days=365)
        df['maintenance_score'] = df['pushed_at_dt'].apply(
            lambda x: max(0, min(1, 1 - (now - x).total_seconds() / (now - one_year_ago).total_seconds()))
        )
    else:
        df['maintenance_score'] = 0.0
        
    # Popularity score (0-1): normalized based on stars and forks
    if 'stars' in df.columns and 'forks' in df.columns:
        max_stars = df['stars'].max() if df['stars'].max() > 0 else 1
        max_forks = df['forks'].max() if df['forks'].max() > 0 else 1
        df['popularity_score'] = (
            0.7 * (df['stars'] / max_stars) + 
            0.3 * (df['forks'] / max_forks)
        )
    else:
        df['popularity_score'] = 0.0
        
    # Community score (0-1): based on contributors
    if 'contributors_count' in df.columns:
        max_contributors = df['contributors_count'].max() if df['contributors_count'].max() > 0 else 1
        df['community_score'] = df['contributors_count'] / max_contributors
    else:
        df['community_score'] = 0.0
        
    # Activity score (0-1): based on commit frequency
    if 'avg_weekly_commits' in df.columns:
        max_commits = df['avg_weekly_commits'].max() if df['avg_weekly_commits'].max() > 0 else 1
        df['activity_score'] = df['avg_weekly_commits'] / max_commits
    else:
        df['activity_score'] = 0.0
        
    # Maturity score: combination of age, releases, and maintenance
    df['maturity_score'] = (
        0.4 * df['age_score'] +
        0.4 * df['maintenance_score'] +
        0.2 * (df['releases_count'] / (df['releases_count'].max() if df['releases_count'].max() > 0 else 1))
    )
    
    # User-friendliness score: based on documentation and issues
    df['user_friendliness_score'] = (
        0.5 * (df['has_wiki'] * 1.0) +
        0.3 * (df['has_pages'] * 1.0) +
        0.2 * (1 - (df['open_issues'] / (df['open_issues'].max() if df['open_issues'].max() > 0 else 1)))
    )
    
    # Overall score: weighted combination of all scores
    df['overall_score'] = (
        0.25 * df['popularity_score'] +
        0.25 * df['maturity_score'] +
        0.2 * df['activity_score'] +
        0.15 * df['community_score'] +
        0.15 * df['user_friendliness_score']
    )
    
    # Scale all scores to 0-100 for easier interpretation
    score_columns = [
        'age_score', 'maintenance_score', 'popularity_score', 
        'community_score', 'activity_score', 'maturity_score',
        'user_friendliness_score', 'overall_score'
    ]
    
    for col in score_columns:
        df[col] = df[col] * 100
        
    return df

# Main app
def main():
    # Load data
    data = load_data()
    
    # Sidebar for token input
    st.sidebar.title("GitHub API Settings")
    token_input = st.sidebar.text_input(
        "GitHub API Token (needed for higher rate limits)",
        value=get_github_token(),
        type="password"
    )
    
    # Add scoring weights customization in sidebar
    st.sidebar.title("Scoring Weights")
    st.sidebar.write("Customize weights for different metrics (must sum to 100%)")
    
    # Default weights
    default_weights = {
        "popularity": 25,
        "maturity": 25,
        "activity": 20,
        "community": 15,
        "user_friendliness": 15
    }
    
    # Get weights from session state or use defaults
    if 'weights' not in st.session_state:
        st.session_state['weights'] = default_weights.copy()
    
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
    
    # Show data overview
    st.header("Dataset Overview")
    st.write(f"Total models: {len(data)}")
    
    # Extract unique categories
    all_categories = []
    for cats in data['categories']:
        if isinstance(cats, list):
            all_categories.extend(cats)
        elif isinstance(cats, str):
            all_categories.extend([c.strip() for c in cats.split(';')])
    
    unique_categories = sorted(list(set(all_categories)))
    st.write(f"Model categories: {', '.join(unique_categories)}")
    
    # Update calculate_scores function to use custom weights
    def calculate_custom_scores(github_df, weights):
        """Calculate scores with custom weights"""
        df = calculate_scores(github_df)  # Get base scores
        
        if not df.empty:
            # Apply custom weights to overall score
            df['overall_score'] = (
                weights['popularity'] * df['popularity_score'] +
                weights['maturity'] * df['maturity_score'] +
                weights['activity'] * df['activity_score'] +
                weights['community'] * df['community_score'] +
                weights['user_friendliness'] * df['user_friendliness_score']
            )
        return df
    
    # GitHub data processing
    if st.button("Fetch GitHub Metrics"):
        with st.spinner("Fetching GitHub metrics for all repositories..."):
            github_df = process_github_data(data, token_input)
            
            if github_df.empty or len(github_df) == 0:
                st.error("Could not fetch any GitHub metrics. Check your API token and try again.")
            else:
                st.session_state['github_df'] = github_df
                # Calculate scores with custom weights
                scored_df = calculate_custom_scores(github_df, weights)
                st.session_state['scored_df'] = scored_df
                st.success(f"Successfully fetched metrics for {len(scored_df)} GitHub repositories!")
    
    # If there's already scored data and weights changed, recalculate
    elif 'github_df' in st.session_state and st.sidebar.button("Apply Custom Weights"):
        with st.spinner("Recalculating scores with new weights..."):
            github_df = st.session_state['github_df']
            scored_df = calculate_custom_scores(github_df, weights)
            st.session_state['scored_df'] = scored_df
            st.success("Scores recalculated with custom weights!")
    
    # If we have processed data in the session state, show visualizations
    if 'scored_df' in st.session_state:
        df = st.session_state['scored_df']
        
        # Category selection for filtering
        st.header("Analyze by Category")
        selected_category = st.selectbox(
            "Select a category to analyze:", 
            ["All Categories"] + unique_categories
        )
        
        # Filter by category if needed
        if selected_category != "All Categories":
            filtered_df = df[df['categories'].apply(lambda x: selected_category in x if isinstance(x, list) else selected_category in str(x))]
        else:
            filtered_df = df.copy()
            
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
            st.plotly_chart(fig)
            
            # Radar chart for top 5 models
            st.subheader("Multi-Dimensional Comparison")
            top5 = filtered_df.sort_values('overall_score', ascending=False).head(5)
            
            # Prepare radar chart data
            metrics = [
                'popularity_score', 'maturity_score', 'activity_score',
                'community_score', 'user_friendliness_score'
            ]
            
            fig = go.Figure()
            
            for i, row in top5.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[row[m] for m in metrics],
                    theta=[m.replace('_score', '').capitalize() for m in metrics],
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
            st.plotly_chart(fig)
            
            # Detailed metrics table
            st.subheader("Detailed Metrics")
            metrics_display = filtered_df[[
                'name', 'language', 'stars', 'forks', 'contributors_count',
                'open_issues', 'releases_count', 'overall_score'
            ]].sort_values('overall_score', ascending=False)
            
            metrics_display.columns = [
                'Model', 'Language', 'Stars', 'Forks', 'Contributors',
                'Open Issues', 'Releases', 'Overall Score'
            ]
            
            st.dataframe(metrics_display)
            
            # Model comparison section
            st.header("Compare Specific Models")
            st.write("Select models to compare directly against each other")
            
            # Get unique model names from filtered data
            model_names = filtered_df['name'].unique().tolist()
            
            # Multi-select for models to compare
            selected_models = st.multiselect(
                "Select models to compare:",
                options=model_names,
                default=model_names[:2] if len(model_names) >= 2 else model_names
            )
            
            if len(selected_models) >= 2:
                # Filter data for selected models
                compare_df = filtered_df[filtered_df['name'].isin(selected_models)]
                
                # Create radar chart for comparison
                st.subheader("Model Comparison")
                
                # Prepare radar chart data
                metrics = [
                    'popularity_score', 'maturity_score', 'activity_score',
                    'community_score', 'user_friendliness_score'
                ]
                
                radar_fig = go.Figure()
                
                for i, row in compare_df.iterrows():
                    radar_fig.add_trace(go.Scatterpolar(
                        r=[row[m] for m in metrics],
                        theta=[m.replace('_score', '').capitalize() for m in metrics],
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
                st.plotly_chart(radar_fig)
                
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
                st.plotly_chart(bar_fig)
                
                # Detailed comparison table
                st.subheader("Detailed Comparison")
                
                # Select columns for comparison
                comparison_cols = [
                    'name', 'language', 'stars', 'forks', 'contributors_count',
                    'open_issues', 'releases_count', 'overall_score',
                    'popularity_score', 'maturity_score', 'activity_score',
                    'community_score', 'user_friendliness_score'
                ]
                
                # Create comparison table
                compare_display = compare_df[comparison_cols].sort_values('overall_score', ascending=False)
                
                # Rename columns for display
                compare_display.columns = [
                    'Model', 'Language', 'Stars', 'Forks', 'Contributors',
                    'Open Issues', 'Releases', 'Overall Score',
                    'Popularity Score', 'Maturity Score', 'Activity Score',
                    'Community Score', 'User-Friendliness Score'
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

if __name__ == "__main__":
    main() 