import pandas as pd
import numpy as np
import time
import re
import json
import os
from github import Github
from tqdm import tqdm
from dotenv import load_dotenv
from collections import Counter

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
    """Extract and classify user information"""
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
            'public_repos': user.public_repos,
            'classification': classify_user(user)
        }
        return user_data
    except Exception as e:
        print(f"Error extracting user info for {user.login if hasattr(user, 'login') else 'unknown'}: {str(e)}")
        return None

def classify_user(user):
    """Classify users into categories: academic, industry, utility, etc."""
    # Default classification
    classification = 'unknown'
    
    # Company/organization hints
    if user.company:
        company = user.company.lower()
        # Check for academic institutions
        academic_keywords = ['university', 'college', 'institute of technology', 'polytechnic', 
                            'academia', 'education', 'research institute', 'laboratory', 'lab', 
                            'école', 'universität', 'università', 'universidad']
        
        # Check for utilities
        utility_keywords = ['utility', 'power', 'electric', 'electricity', 'energy', 'grid operator', 
                          'transmission', 'distribution', 'generation']
        
        # Check for industry
        industry_keywords = ['inc', 'corp', 'llc', 'ltd', 'limited', 'company', 'gmbh', 'ag', 'co.', 
                           'consulting', 'solutions', 'software', 'technologies', 'plc']
        
        # Check for RTO (Regional Transmission Operators)
        rto_keywords = ['rto', 'iso', 'miso', 'pjm', 'caiso', 'ercot', 'iso-ne', 'nyiso', 'spp', 
                      'independent system operator', 'regional transmission', 'ieso', 'aeso']
        
        # Check for research organizations
        research_keywords = ['research', 'technology organization', 'r&d', 'innovation', 'laboratory', 
                      'national lab', 'science', 'nrel', 'pnnl', 'ornl', 'lbnl', 'inl', 'anl']
        
        # Classify based on keywords
        if any(keyword in company for keyword in academic_keywords):
            classification = 'academic'
        elif any(keyword in company for keyword in rto_keywords):
            classification = 'rto'
        elif any(keyword in company for keyword in utility_keywords):
            classification = 'utility'
        elif any(keyword in company for keyword in research_keywords):
            classification = 'research_organization'
        elif any(keyword in company for keyword in industry_keywords):
            classification = 'industry'
    
    # Email hints
    if user.email and classification == 'unknown':
        email_domain = user.email.split('@')[-1].lower()
        
        # Academic email domains
        academic_domains = ['edu', 'ac.uk', 'ac.jp', 'edu.au', 'ac.nz', 'edu.sg', 'uni-']
        if any(domain in email_domain for domain in academic_domains):
            classification = 'academic'
    
    # Bio hints (if still unknown)
    if user.bio and classification == 'unknown':
        bio = user.bio.lower()
        if any(word in bio for word in ['professor', 'phd', 'student', 'researcher', 'postdoc']):
            classification = 'academic'
        elif any(word in bio for word in ['engineer', 'developer', 'scientist', 'consultant']):
            classification = 'professional'
    
    return classification

def analyze_repo_users(repo_name, users_data):
    """Analyze user data for a specific repository"""
    # Combine all users (removing duplicates)
    all_users = []
    user_logins = set()
    
    for interaction_type in ['stargazers', 'forkers', 'contributors', 'watchers']:
        for user in users_data[interaction_type]:
            if user['login'] not in user_logins:
                all_users.append(user)
                user_logins.add(user['login'])
    
    # Count users by classification
    classifications = [user['classification'] for user in all_users]
    class_counts = Counter(classifications)
    
    # Count contributors by classification
    contributor_class = [user['classification'] for user in users_data['contributors']]
    contributor_class_counts = Counter(contributor_class)
    
    # Identify top organizations
    companies = [user['company'] for user in all_users if user['company']]
    companies = [company.strip() for company in companies]
    company_counts = Counter(companies)
    
    # Identify locations
    locations = [user['location'] for user in all_users if user['location']]
    location_counts = Counter(locations)
    
    return {
        'repo_name': repo_name,
        'total_users': len(all_users),
        'classification_counts': dict(class_counts),
        'contributor_class_counts': dict(contributor_class_counts),
        'top_organizations': dict(company_counts.most_common(10)),
        'top_locations': dict(location_counts.most_common(10)),
    }

def main():
    """Main function to collect and analyze GitHub user data"""
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
    selection = input("\nEnter repository numbers to analyze (comma-separated), or 'all' for all repos: ")
    
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
    
    # Initialize results container
    results = []
    
    # Process each selected repository
    for repo in selected_repos:
        print(f"\nAnalyzing {repo['name']}...")
        
        # Extract owner and repo name from URL
        url_match = re.search(r'github\.com/([^/]+)/([^/]+)', repo['url'])
        if not url_match:
            print(f"Could not parse GitHub URL: {repo['url']}")
            continue
            
        owner, name = url_match.groups()
        name = name.replace('.git', '')  # Remove .git extension if present
        
        # Collect user data
        users_data = collect_user_data(owner, name, g)
        
        # Analyze the data
        analysis = analyze_repo_users(repo['name'], users_data)
        
        # Save detailed user data for this repository
        repo_user_file = f"user_data_{name}.json"
        with open(repo_user_file, 'w') as f:
            json.dump(users_data, f, indent=2)
        print(f"Detailed user data saved to {repo_user_file}")
        
        # Add to results
        results.append(analysis)
        
        # Avoid rate limiting
        time.sleep(1)
    
    # Save analysis results
    with open("user_analysis_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print("\nAnalysis complete! Results saved to user_analysis_results.json")
    
    # Generate summary report
    generate_summary_report(results)

def generate_summary_report(results):
    """Generate a summary report of the user analysis"""
    if not results:
        return
        
    print("\n===== SUMMARY REPORT =====")
    
    # Overall metrics
    total_users = sum(r['total_users'] for r in results)
    print(f"Total unique users across all repositories: {total_users}")
    
    # Combine classification counts across repos
    all_classifications = Counter()
    for r in results:
        all_classifications.update(r['classification_counts'])
    
    print("\nUser classification across all repositories:")
    for cls, count in all_classifications.most_common():
        print(f"  {cls}: {count} ({count/total_users*100:.1f}%)")
    
    # Top organizations
    all_orgs = Counter()
    for r in results:
        all_orgs.update(r['top_organizations'])
    
    print("\nTop organizations:")
    for org, count in all_orgs.most_common(15):
        if org:  # Skip empty org names
            print(f"  {org}: {count}")
    
    # Top academic institutions
    print("\nRepository breakdown:")
    for r in results:
        print(f"\n{r['repo_name']}:")
        print(f"  Total users: {r['total_users']}")
        print("  User classifications:")
        for cls, count in Counter(r['classification_counts']).most_common():
            print(f"    {cls}: {count} ({count/r['total_users']*100:.1f}%)")
    
    print("\nReport saved to user_analysis_report.txt")
    
    # Save to file
    with open("user_analysis_report.txt", 'w') as f:
        f.write("===== USER ANALYSIS REPORT =====\n\n")
        
        f.write(f"Total unique users across all repositories: {total_users}\n\n")
        
        f.write("User classification across all repositories:\n")
        for cls, count in all_classifications.most_common():
            f.write(f"  {cls}: {count} ({count/total_users*100:.1f}%)\n")
        
        f.write("\nTop organizations:\n")
        for org, count in all_orgs.most_common(15):
            if org:
                f.write(f"  {org}: {count}\n")
        
        f.write("\nRepository breakdown:\n")
        for r in results:
            f.write(f"\n{r['repo_name']}:\n")
            f.write(f"  Total users: {r['total_users']}\n")
            f.write("  User classifications:\n")
            for cls, count in Counter(r['classification_counts']).most_common():
                f.write(f"    {cls}: {count} ({count/r['total_users']*100:.1f}%)\n")

if __name__ == "__main__":
    main() 