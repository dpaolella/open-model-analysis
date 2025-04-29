import pandas as pd
import requests
import time
import re
import numpy as np
from datetime import datetime, timedelta
from github import Github
import os
import json
from dotenv import load_dotenv

def get_github_token():
    """Get GitHub token from environment or file"""
    load_dotenv()  # Load from .env file if available
    token = os.getenv("GITHUB_TOKEN")
    
    if token:
        return token
    
    # Try to read from github_api_key file
    try:
        with open("github_api_key", "r") as f:
            return f.read().strip()
    except:
        return ""  # Return empty string if no token found

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

def get_github_metrics(username, repo, token):
    """Get GitHub metrics for a repository"""
    if not username or not repo:
        return None
    
    print(f"Fetching metrics for {username}/{repo}...")
    
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
            "homepage": repository.homepage,
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
        
        # Get time series data
        print(f"Fetching time series data for {username}/{repo}...")
        metrics["time_series"] = get_repository_time_series(repository)
            
        return metrics
    
    except Exception as e:
        print(f"Error fetching GitHub metrics for {username}/{repo}: {str(e)}")
        return None

def get_repository_time_series(repository):
    """Get time series data for a repository"""
    time_series = {}
    
    # Get stars over time
    try:
        print("Fetching stargazers time series...")
        stars_history = []
        try:
            for sg in repository.get_stargazers_with_dates():
                stars_history.append({
                    "date": sg.starred_at.isoformat(),
                    "user": sg.user.login
                })
        except Exception as e:
            print(f"Warning: Could not get stargazers with dates. This may be due to API limitations: {str(e)}")
            # Try to at least get the count if detailed data isn't available
            stars_history = [{"date": repository.created_at.isoformat(), "total_stars": repository.stargazers_count}]
        
        # Convert to cumulative count by date
        if stars_history and "total_stars" not in stars_history[0]:
            stars_cumulative = []
            dates = sorted([sg["date"] for sg in stars_history])
            for i, date in enumerate(dates):
                stars_cumulative.append({
                    "date": date,
                    "total_stars": i + 1
                })
            time_series["stars_over_time"] = stars_cumulative
        elif stars_history:
            # Use the simplified stars history
            time_series["stars_over_time"] = stars_history
        else:
            time_series["stars_over_time"] = []
    except Exception as e:
        print(f"Error fetching stars time series: {str(e)}")
        time_series["stars_over_time"] = []
    
    # Get commit frequency over time (by week)
    try:
        print("Fetching commit activity time series...")
        commit_activity = None
        attempt_count = 0
        
        # Sometimes GitHub API needs a few calls before it returns data
        while attempt_count < 3 and not commit_activity:
            commit_activity = list(repository.get_stats_commit_activity() or [])
            if not commit_activity:
                print("Waiting for GitHub API to generate commit activity stats...")
                time.sleep(2)  # Wait a bit before trying again
                attempt_count += 1
        
        if commit_activity:
            # Get the start date (52 weeks ago from now)
            now = datetime.now()
            start_date = now - timedelta(days=52*7)
            
            commits_by_week = []
            for i, week in enumerate(commit_activity):
                try:
                    week_date = start_date + timedelta(days=i*7)
                    commits_by_week.append({
                        "date": week_date.isoformat(),
                        "total_commits": week.total,
                        "days": week.days  # This is a list of commits per day [Sun, Mon, ..., Sat]
                    })
                except Exception as e:
                    print(f"Error processing commit week data: {str(e)}")
                    continue
            time_series["commits_by_week"] = commits_by_week
        else:
            print("No commit activity data available")
            time_series["commits_by_week"] = []
    except Exception as e:
        print(f"Error fetching commit activity time series: {str(e)}")
        time_series["commits_by_week"] = []
    
    # Get code frequency (additions/deletions per week)
    try:
        print("Fetching code frequency time series...")
        code_freq = None
        attempt_count = 0
        
        # Sometimes GitHub API needs a few calls before it returns data
        while attempt_count < 3 and not code_freq:
            code_freq = list(repository.get_stats_code_frequency() or [])
            if not code_freq:
                print("Waiting for GitHub API to generate code frequency stats...")
                time.sleep(2)  # Wait a bit before trying again
                attempt_count += 1
        
        if code_freq:
            code_frequency = []
            for week in code_freq:
                try:
                    # Handle different types that week.week might be
                    if isinstance(week.week, int):
                        # It's already a timestamp
                        week_timestamp = week.week
                    elif hasattr(week.week, 'timestamp'):
                        # It's a datetime object
                        week_timestamp = int(week.week.timestamp())
                    else:
                        # Try to convert to int as a fallback
                        week_timestamp = int(week.week)
                    
                    week_date = datetime.fromtimestamp(week_timestamp)
                    code_frequency.append({
                        "date": week_date.isoformat(),
                        "additions": week.additions,
                        "deletions": week.deletions
                    })
                except Exception as e:
                    print(f"Error processing code frequency week data: {str(e)}")
                    # Skip this week if there's an error
                    continue
            time_series["code_frequency"] = code_frequency
        else:
            print("No code frequency data available")
            time_series["code_frequency"] = []
    except Exception as e:
        print(f"Error fetching code frequency time series: {str(e)}")
        time_series["code_frequency"] = []
    
    # Get issue data for last year
    try:
        print("Fetching issues time series...")
        try:
            issues = list(repository.get_issues(state='all', since=datetime.now() - timedelta(days=365)))
        except Exception as e:
            print(f"Error getting issues, will try with more limited data: {str(e)}")
            issues = list(repository.get_issues(state='all', since=datetime.now() - timedelta(days=90)))
        
        if issues:
            issues_data = []
            for issue in issues:
                try:
                    issue_data = {
                        "number": issue.number,
                        "created_at": issue.created_at.isoformat(),
                        "state": issue.state
                    }
                    if issue.closed_at:
                        issue_data["closed_at"] = issue.closed_at.isoformat()
                    issues_data.append(issue_data)
                except Exception as e:
                    print(f"Error processing issue data: {str(e)}")
                    continue
            
            time_series["issues"] = issues_data
            
            # Compute issues opened and closed over time (monthly)
            monthly_issues = {}
            for issue in issues_data:
                try:
                    month = issue["created_at"][:7]  # YYYY-MM format
                    if month not in monthly_issues:
                        monthly_issues[month] = {"opened": 0, "closed": 0}
                    monthly_issues[month]["opened"] += 1
                    
                    if "closed_at" in issue:
                        close_month = issue["closed_at"][:7]
                        if close_month not in monthly_issues:
                            monthly_issues[close_month] = {"opened": 0, "closed": 0}
                        monthly_issues[close_month]["closed"] += 1
                except Exception as e:
                    print(f"Error processing monthly issue data: {str(e)}")
                    continue
            
            # Convert to list sorted by month
            issues_by_month = []
            for month in sorted(monthly_issues.keys()):
                issues_by_month.append({
                    "month": month,
                    "opened": monthly_issues[month]["opened"],
                    "closed": monthly_issues[month]["closed"]
                })
            time_series["issues_by_month"] = issues_by_month
        else:
            print("No issue data available")
            time_series["issues"] = []
            time_series["issues_by_month"] = []
    except Exception as e:
        print(f"Error fetching issues time series: {str(e)}")
        time_series["issues"] = []
        time_series["issues_by_month"] = []
    
    return time_series

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
    
    return github_data

def load_data():
    """Load the energy models data from CSV"""
    data = pd.read_csv("gpst_open_tools.csv")
    # Clean up categories
    data['categories'] = data['categories'].str.split(';').apply(
        lambda x: [item.strip() for item in x] if isinstance(x, list) else [])
    return data

def fetch_and_save_github_data():
    """Main function to fetch GitHub data and save to a JSON file"""
    # Get GitHub token
    token = get_github_token()
    
    if not token:
        print("No GitHub token found. Please set the GITHUB_TOKEN environment variable "
              "or create a github_api_key file with your token.")
        return
    
    print("Loading model data from CSV...")
    data = load_data()
    
    print(f"Found {len(data)} models. Fetching GitHub metrics...")
    github_data = process_github_data(data, token)
    
    if not github_data:
        print("Could not fetch any GitHub metrics. Check your API token and try again.")
        return
    
    print(f"Successfully fetched metrics for {len(github_data)} GitHub repositories.")
    
    # Create data directory if it doesn't exist
    os.makedirs("github_raw_data", exist_ok=True)
    
    # Save raw data (without scoring)
    raw_data_file = "github_raw_data/github_data_raw.json"
    with open(raw_data_file, 'w') as f:
        json.dump(github_data, f, indent=2)
    
    # Copy to the regular location for backward compatibility
    compat_file = "github_raw_data/github_data.json"
    with open(compat_file, 'w') as f:
        json.dump(github_data, f, indent=2)
    
    # Save timestamp metadata
    metadata = {
        "fetch_timestamp": time.time(),
        "fetch_date": datetime.now().isoformat(),
        "repos_count": len(github_data)
    }
    
    metadata_file = "github_raw_data/github_data_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Raw data saved to {raw_data_file}")
    print(f"Metadata saved to {metadata_file}")
    print("\nTo analyze this data and generate scores, run:")
    print("python github_data_analyzer.py")

if __name__ == "__main__":
    fetch_and_save_github_data() 