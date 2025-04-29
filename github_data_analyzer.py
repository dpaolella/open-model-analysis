import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import time
import copy
import re

def load_raw_github_data():
    """Load raw GitHub data"""
    # Try the data directory first
    data_file = "github_raw_data/github_data.json"
    
    if os.path.exists(data_file):
        with open(data_file, 'r') as f:
            return json.load(f)
    else:
        # Try the old location
        data_file = "github_data.json"
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                return json.load(f)
        else:
            print("No GitHub data found. Please run github_data_fetcher.py first.")
            return []

def is_documentation_url(url):
    """
    Check if a URL is likely to be documentation based on common patterns.
    
    Args:
        url: The URL to check
        
    Returns:
        bool: True if the URL appears to be documentation
    """
    if not url or not isinstance(url, str):
        return False
    
    # Convert to lowercase for case-insensitive matching
    url_lower = url.lower()
    
    # Common documentation hosting services
    doc_hosts = [
        'readthedocs.io',
        'readthedocs.org',
        'rtfd.io',
        'rtfd.org',
        'gitbook.io',
        'github.io/docs',
        'github.io/doc',
        'github.io/wiki',
        'docs.github.io',
        'documentation.',
        'docs.',
    ]
    
    # Common documentation patterns in URLs
    doc_patterns = [
        '/docs/',
        '/doc/',
        '/documentation/',
        '/wiki/',
        '/manual/',
        '/guide/',
        '/reference/',
        '/api/',
        '/help/',
        '.readthedocs.',
        'readthedocs',  # More general pattern to catch variations
        'docs.',
        'documentation.',
        '/sphinx/',
        'en/latest',    # Common ReadTheDocs pattern
        'en/stable',    # Common ReadTheDocs pattern
    ]
    
    # Check for documentation hosts
    if any(host in url_lower for host in doc_hosts):
        return True
    
    # Check for documentation patterns
    if any(pattern in url_lower for pattern in doc_patterns):
        return True
    
    return False

def extract_doc_urls_from_repo_data(repo_data):
    """
    Extract potential documentation URLs from repository data.
    
    Args:
        repo_data: Dictionary containing GitHub repository data
        
    Returns:
        list: List of potential documentation URLs
    """
    urls = []
    
    # Check repository description for URLs
    if repo_data.get('description'):
        description = repo_data['description'] or ""
        # Extract URLs using a regex pattern
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        urls.extend(re.findall(url_pattern, description))
    
    # Check homepage URL
    if repo_data.get('homepage') and repo_data['homepage']:
        urls.append(repo_data['homepage'])
    
    # Check README for documentation links if available
    if repo_data.get('readme_content'):
        readme = repo_data['readme_content'] or ""
        url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
        readme_urls = re.findall(url_pattern, readme)
        urls.extend(readme_urls)
    
    return list(set(urls))  # Remove duplicates

def calculate_scores(github_data):
    """Calculate various scores for each model"""
    if not github_data:
        return github_data
    
    # Create a copy to avoid modifying the original
    df = pd.DataFrame(github_data)
    
    # Calculate recent star growth from time series data if available
    df['recent_star_growth'] = 0.0  # Default value
    
    for i, row in df.iterrows():
        if ('time_series' in row and 
            isinstance(row['time_series'], dict) and 
            'stars_over_time' in row['time_series'] and 
            row['time_series']['stars_over_time']):
            try:
                # Convert to dataframe for easier processing
                stars_data = pd.DataFrame(row['time_series']['stars_over_time'])
                if len(stars_data) > 0:
                    # Sort by date to ensure chronological order
                    stars_data['date'] = pd.to_datetime(stars_data['date'])
                    stars_data = stars_data.sort_values('date')
                    
                    # Get stars in the past 90 days
                    cutoff_date = pd.Timestamp(datetime.now() - timedelta(days=90)).tz_localize('UTC')
                    recent_data = stars_data[stars_data['date'] >= cutoff_date]
                    
                    if len(recent_data) > 0:
                        # Calculate growth - either from total_stars or by counting rows
                        if 'total_stars' in recent_data.columns:
                            if len(recent_data) > 1:
                                recent_growth = recent_data['total_stars'].iloc[-1] - recent_data['total_stars'].iloc[0]
                            else:
                                recent_growth = 0
                        else:
                            recent_growth = len(recent_data)
                        
                        df.at[i, 'recent_star_growth'] = recent_growth
            except Exception as e:
                print(f"Error calculating recent star growth for {row.get('name', 'unknown')}: {str(e)}")
    
    # Check for external documentation
    df['has_external_docs'] = False
    
    # Check if repository homepage is a documentation URL
    for i, row in df.iterrows():
        # First check the repository homepage from GitHub data
        if pd.notna(row.get('homepage')):
            homepage_url = row.get('homepage')
            df.at[i, 'has_external_docs'] = is_documentation_url(homepage_url)
        
        # Next check documentation_url field
        if not df.at[i, 'has_external_docs'] and pd.notna(row.get('documentation_url')):
            doc_url = row.get('documentation_url')
            df.at[i, 'has_external_docs'] = is_documentation_url(doc_url)
            
        # Last check website_url field
        if not df.at[i, 'has_external_docs'] and pd.notna(row.get('website_url')):
            website_url = row.get('website_url')
            df.at[i, 'has_external_docs'] = is_documentation_url(website_url)
    
    # Print out documentation status for reference
    for i, row in df.iterrows():
        if row['has_external_docs']:
            doc_urls = []
            if pd.notna(row.get('homepage')):
                doc_urls.append(f"Homepage: {row.get('homepage')}")
            if pd.notna(row.get('documentation_url')):
                doc_urls.append(f"Doc URL: {row.get('documentation_url')}")
            if pd.notna(row.get('website_url')):
                doc_urls.append(f"Website: {row.get('website_url')}")
            
            print(f"Found external documentation for {row.get('name')}: {' | '.join(doc_urls)}")
        else:
            print(f"No external documentation detected for {row.get('name')}")
    
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
        
        # Fix timezone issue by converting all dates to naive datetime (removing timezone info)
        df['pushed_at_naive'] = df['pushed_at_dt'].dt.tz_localize(None)
        
        df['maintenance_score'] = df['pushed_at_naive'].apply(
            lambda x: max(0, min(1, 1 - (now - x).total_seconds() / (now - one_year_ago).total_seconds()))
        )
    else:
        df['maintenance_score'] = 0.0
        
    # Popularity score (0-1): weighted more towards recent stars if available
    if 'stars' in df.columns and 'forks' in df.columns:
        max_stars = df['stars'].max() if df['stars'].max() > 0 else 1
        max_forks = df['forks'].max() if df['forks'].max() > 0 else 1
        
        # Basic popularity score
        df['base_popularity_score'] = (0.7 * (df['stars'] / max_stars) + 0.3 * (df['forks'] / max_forks))
        
        # Add recent growth bonus if available
        if 'recent_star_growth' in df.columns:
            max_recent_growth = df['recent_star_growth'].max() if df['recent_star_growth'].max() > 0 else 1
            # Combine base score (70%) with recent growth (30%)
            df['popularity_score'] = (0.7 * df['base_popularity_score'] + 
                                     0.3 * (df['recent_star_growth'] / max_recent_growth))
        else:
            df['popularity_score'] = df['base_popularity_score']
    else:
        df['popularity_score'] = 0.0
        
    # Community score (0-1): based on contributors
    if 'contributors_count' in df.columns:
        max_contributors = df['contributors_count'].max() if df['contributors_count'].max() > 0 else 1
        df['developer_community_score'] = df['contributors_count'] / max_contributors
    else:
        df['developer_community_score'] = 0.0
        
    # Activity score (0-1): based on commit frequency
    if 'avg_weekly_commits' in df.columns:
        max_commits = df['avg_weekly_commits'].max() if df['avg_weekly_commits'].max() > 0 else 1
        df['developer_activity_score'] = df['avg_weekly_commits'] / max_commits
    else:
        df['developer_activity_score'] = 0.0
        
    # Maturity score: combination of age, releases, and maintenance
    df['maturity_score'] = (
        0.4 * df['age_score'] +
        0.4 * df['maintenance_score'] +
        0.2 * (df['releases_count'] / (df['releases_count'].max() if df['releases_count'].max() > 0 else 1))
    )
    
    # Calculate total_issues and closed_issues from the data
    df['total_issues'] = df['open_issues']  # Start with open issues
    df['closed_issues'] = 0  # Default value
    
    # If we have time_series issue data, use it to get more accurate counts
    for i, row in df.iterrows():
        if ('time_series' in row and 
            isinstance(row['time_series'], dict) and 
            'issues' in row['time_series'] and 
            row['time_series']['issues']):
            
            issues = pd.DataFrame(row['time_series']['issues'])
            if len(issues) > 0:
                # Count total and closed issues from time series data
                time_series_total = len(issues)
                time_series_closed = len(issues[issues['state'] == 'closed'])
                
                # Ensure total_issues is at least equal to open_issues
                df.at[i, 'total_issues'] = max(time_series_total, row['open_issues'])
                
                # If time series shows more closed issues, use that value
                if time_series_closed > 0:
                    df.at[i, 'closed_issues'] = time_series_closed
                    # Adjust total_issues if needed to ensure total = open + closed
                    if df.at[i, 'open_issues'] + time_series_closed > df.at[i, 'total_issues']:
                        df.at[i, 'total_issues'] = df.at[i, 'open_issues'] + time_series_closed
            
        # Always ensure closed_issues is calculated correctly
        # total_issues should be at least open_issues + closed_issues
        df.at[i, 'closed_issues'] = max(df.at[i, 'closed_issues'], 
                                        df.at[i, 'total_issues'] - df.at[i, 'open_issues'])
        
        # Final sanity check - ensure total issues is at least equal to open_issues
        if df.at[i, 'total_issues'] < df.at[i, 'open_issues']:
            df.at[i, 'total_issues'] = df.at[i, 'open_issues']
            df.at[i, 'closed_issues'] = 0  # Reset closed issues if we had to correct total
                
    # User-friendliness score: based on documentation and issues
    # Previously: wiki (20%), GitHub pages (30%), issue resolution (50%)
    # Now: wiki OR external docs (20%), GitHub pages OR external docs (30%), issue resolution (50%)
    
    # First, calculate the issue resolution ratio
    if 'total_issues' in df.columns:
        df['issue_resolution_ratio'] = df.apply(
            lambda row: (row['closed_issues']) / row['total_issues'] 
            if row['total_issues'] > 0 else 0.5,  # Default to middle if no issues
            axis=1
        )
    else:
        # If we only have open_issues count, use a simpler heuristic
        max_issues = df['open_issues'].max() if 'open_issues' in df.columns and df['open_issues'].max() > 0 else 1
        df['issue_resolution_ratio'] = 1 - (df['open_issues'] / max_issues)
    
    # Combine GitHub documentation with external documentation
    df['has_documentation'] = df.apply(
        lambda row: bool(row.get('has_wiki', False) or row.get('has_external_docs', False)), 
        axis=1
    )
    
    df['has_published_docs'] = df.apply(
        lambda row: bool(row.get('has_pages', False) or row.get('has_external_docs', False)), 
        axis=1
    )
    
    df['user_friendliness_score'] = (
        0.2 * df['has_documentation'].astype(float) +  # Wiki or external docs
        0.3 * df['has_published_docs'].astype(float) +  # GitHub Pages or external docs
        0.5 * df['issue_resolution_ratio']  # Issue resolution still 50%
    )
    
    # Print documentation status for debugging
    for i, row in df.iterrows():
        doc_sources = []
        if row.get('has_wiki', False): doc_sources.append('GitHub Wiki')
        if row.get('has_pages', False): doc_sources.append('GitHub Pages')
        if row.get('has_external_docs', False): doc_sources.append('External Documentation')
        
        print(f"{row.get('name', 'unknown')}: Documentation sources: {', '.join(doc_sources) or 'None'}")
        print(f"  → User-friendliness score: {row['user_friendliness_score']*100:.1f}/100")
    
    # Overall score: weighted combination of all scores
    df['overall_score'] = (
        0.25 * df['popularity_score'] +
        0.25 * df['maturity_score'] +
        0.2 * df['developer_activity_score'] +
        0.15 * df['developer_community_score'] +
        0.15 * df['user_friendliness_score']
    )
    
    # Scale all scores to 0-100 for easier interpretation
    score_columns = [
        'age_score', 'maintenance_score', 'popularity_score', 
        'developer_community_score', 'developer_activity_score', 'maturity_score',
        'user_friendliness_score', 'overall_score'
    ]
    
    for col in score_columns:
        df[col] = df[col] * 100
    
    # Remove timestamp columns that can't be serialized to JSON
    columns_to_drop = ['created_at_dt', 'pushed_at_dt', 'pushed_at_naive']
    for col in columns_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])
        
    return df.to_dict('records')

def analyze_github_data():
    """Analyze GitHub data and save scored results"""
    print("Loading raw GitHub data...")
    github_data = load_raw_github_data()
    
    if not github_data:
        print("No GitHub data found. Please run github_data_fetcher.py first.")
        return
    
    print(f"Loaded data for {len(github_data)} repositories.")
    
    # Calculate scores
    print("Calculating scores...")
    github_data_with_scores = calculate_scores(github_data)
    
    # Save to JSON file
    output_file = "github_data.json"
    with open(output_file, 'w') as f:
        json.dump(github_data_with_scores, f, indent=2)
    
    # Save metadata
    metadata = {
        "analysis_timestamp": time.time(),
        "analysis_date": datetime.now().isoformat(),
        "repos_count": len(github_data_with_scores)
    }
    
    metadata_file = "github_data_analysis_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Analysis complete! Results saved to {output_file}")
    print(f"Analysis metadata saved to {metadata_file}")
    print("You can now run the dashboard to view the results.")

if __name__ == "__main__":
    analyze_github_data() 