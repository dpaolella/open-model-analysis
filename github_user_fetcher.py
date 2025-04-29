import pandas as pd
import time
import re
import json
import os
from github import Github
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_github_token():
    """Get GitHub token from environment or file"""
    token = os.getenv("GITHUB_TOKEN")
    
    if token:
        return token
    
    # Try to read from github_api_key file
    try:
        with open("github_api_key", "r") as f:
            return f.read().strip()
    except:
        return ""  # Return empty string if no token found

def load_github_data():
    """Load existing GitHub data from JSON file"""
    try:
        with open("github_data.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("GitHub data not found. Please run github_data_fetcher.py first.")
        return []

def collect_user_data(repo_owner, repo_name, g):
    """Collect data on users who starred, forked, or contributed to a repository"""
    try:
        # Get repository
        repo = g.get_repo(f"{repo_owner}/{repo_name}")
        
        # Initialize user data containers
        stargazers = []
        forkers = []
        contributors = []
        watchers = []
        
        # Get stargazers (with rate limit handling)
        print(f"Collecting stargazers for {repo_owner}/{repo_name}...")
        try:
            for user in tqdm(repo.get_stargazers()):
                try:
                    user_data = extract_user_info(user)
                    if user_data:
                        user_data['interaction_type'] = 'star'
                        stargazers.append(user_data)
                except Exception as e:
                    print(f"Error processing stargazer: {str(e)}")
                time.sleep(0.2)  # Avoid rate limiting
        except Exception as e:
            print(f"Error collecting stargazers: {str(e)}")
        
        # Get forkers
        print(f"Collecting forkers for {repo_owner}/{repo_name}...")
        try:
            for fork in tqdm(repo.get_forks()):
                try:
                    if fork.owner:
                        user_data = extract_user_info(fork.owner)
                        if user_data:
                            user_data['interaction_type'] = 'fork'
                            forkers.append(user_data)
                except Exception as e:
                    print(f"Error processing forker: {str(e)}")
                time.sleep(0.2)  # Avoid rate limiting
        except Exception as e:
            print(f"Error collecting forkers: {str(e)}")
        
        # Get contributors
        print(f"Collecting contributors for {repo_owner}/{repo_name}...")
        try:
            for user in tqdm(repo.get_contributors()):
                try:
                    user_data = extract_user_info(user)
                    if user_data:
                        user_data['interaction_type'] = 'contributor'
                        user_data['contributions'] = user.contributions
                        contributors.append(user_data)
                except Exception as e:
                    print(f"Error processing contributor: {str(e)}")
                time.sleep(0.2)  # Avoid rate limiting
        except Exception as e:
            print(f"Error collecting contributors: {str(e)}")
        
        # Get watchers
        print(f"Collecting watchers for {repo_owner}/{repo_name}...")
        try:
            for user in tqdm(repo.get_watchers()):
                try:
                    user_data = extract_user_info(user)
                    if user_data:
                        user_data['interaction_type'] = 'watcher'
                        watchers.append(user_data)
                except Exception as e:
                    print(f"Error processing watcher: {str(e)}")
                time.sleep(0.2)  # Avoid rate limiting
        except Exception as e:
            print(f"Error collecting watchers: {str(e)}")
            
        return {
            'stargazers': stargazers,
            'forkers': forkers,
            'contributors': contributors,
            'watchers': watchers
        }
        
    except Exception as e:
        print(f"Error analyzing repo {repo_owner}/{repo_name}: {str(e)}")
        return {
            'stargazers': [],
            'forkers': [],
            'contributors': [],
            'watchers': []
        }

def extract_user_info(user):
    """Extract raw user information without classification"""
    if not user:
        return None
    
    try:
        user_data = {
            'login': user.login,
            'name': user.name if user.name else "",
            'email': user.email if user.email else "",
            'company': user.company if user.company else "",
            'bio': user.bio if user.bio else "",
            'location': user.location if user.location else "",
            'user_type': user.type,
            'followers': user.followers,
            'public_repos': user.public_repos
        }
        return user_data
    except Exception as e:
        print(f"Error extracting user info for {user.login if hasattr(user, 'login') else 'unknown'}: {str(e)}")
        return None

def main():
    """Main function to collect GitHub user data"""
    # Get GitHub token
    token = get_github_token()
    if not token:
        print("No GitHub token found. Please set the GITHUB_TOKEN environment variable or create a github_api_key file.")
        return
    
    # Initialize GitHub client
    g = Github(token)
    
    # Load existing GitHub data
    github_data = load_github_data()
    if not github_data:
        return
    
    # Ask user which repositories to analyze
    print("\nAvailable repositories:")
    repos = []
    for i, repo_data in enumerate(github_data):
        if 'source_code_url' in repo_data and 'github.com' in repo_data['source_code_url']:
            repos.append({
                'id': i,
                'name': repo_data['name'],
                'url': repo_data['source_code_url']
            })
            print(f"{i+1}. {repo_data['name']} ({repo_data['source_code_url']})")
    
    # Get repo selection from user
    selection = input("\nEnter repository numbers to fetch (comma-separated), or 'all' for all repos: ")
    
    selected_repos = []
    if selection.lower() == 'all':
        selected_repos = repos
    else:
        try:
            indices = [int(idx.strip()) - 1 for idx in selection.split(',')]
            selected_repos = [repos[i] for i in indices if 0 <= i < len(repos)]
        except:
            print("Invalid selection. Please enter comma-separated numbers.")
            return
    
    if not selected_repos:
        print("No repositories selected.")
        return
    
    # Create data directory if it doesn't exist
    os.makedirs("github_raw_data", exist_ok=True)
    
    # Process each selected repository
    for repo in selected_repos:
        print(f"\nFetching data for {repo['name']}...")
        
        # Extract owner and repo name from URL
        url_match = re.search(r'github\.com/([^/]+)/([^/]+)', repo['url'])
        if not url_match:
            print(f"Could not parse GitHub URL: {repo['url']}")
            continue
            
        owner, name = url_match.groups()
        name = name.replace('.git', '')  # Remove .git extension if present
        
        # Collect user data
        users_data = collect_user_data(owner, name, g)
        
        # Save raw user data for this repository
        output_file = f"github_raw_data/raw_user_data_{repo['name'].replace(' ', '_')}.json"
        with open(output_file, 'w') as f:
            json.dump(users_data, f, indent=2)
            
        print(f"Raw user data saved to {output_file}")
        
        # Save metadata about the fetch
        metadata = {
            "repo_name": repo['name'],
            "repo_url": repo['url'],
            "github_owner": owner,
            "github_repo": name,
            "fetch_timestamp": time.time(),
            "stargazers_count": len(users_data['stargazers']),
            "forkers_count": len(users_data['forkers']),
            "contributors_count": len(users_data['contributors']),
            "watchers_count": len(users_data['watchers'])
        }
        
        metadata_file = f"github_raw_data/metadata_{repo['name'].replace(' ', '_')}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Avoid rate limiting
        time.sleep(1)
    
    print("\nData fetching complete! Raw data saved to github_raw_data/ directory")
    print("You can now run github_user_analyzer.py to analyze this data")

if __name__ == "__main__":
    main() 