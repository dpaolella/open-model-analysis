import pandas as pd
import numpy as np
import json
import os
import glob
import time
from collections import Counter

def load_raw_user_data():
    """Load all raw user data files"""
    data_dir = "github_raw_data"
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' not found. Please run github_user_fetcher.py first.")
        return []
    
    data_files = glob.glob(f"{data_dir}/raw_user_data_*.json")
    if not data_files:
        print(f"No raw user data files found in '{data_dir}'. Please run github_user_fetcher.py first.")
        return []
    
    repos_data = []
    
    for file_path in data_files:
        try:
            # Extract repository name from filename
            repo_name = os.path.basename(file_path).replace("raw_user_data_", "").replace(".json", "").replace("_", " ")
            
            # Load data
            with open(file_path, 'r') as f:
                user_data = json.load(f)
            
            # Load metadata if available
            metadata_path = file_path.replace("raw_user_data_", "metadata_")
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            repos_data.append({
                "repo_name": metadata.get("repo_name", repo_name),
                "repo_url": metadata.get("repo_url", ""),
                "data": user_data
            })
            
            print(f"Loaded data for {repo_name}: {len(user_data['stargazers'])} stargazers, {len(user_data['contributors'])} contributors")
            
        except Exception as e:
            print(f"Error loading data from {file_path}: {str(e)}")
    
    return repos_data

def classify_user(user_data):
    """Classify users into categories: academic, industry, utility, etc."""
    # Default classification
    classification = 'unknown'
    
    # Company/organization hints
    if user_data.get('company'):
        company = user_data['company'].lower()
        
        # Known major organizations (prioritize these matches first)
        notable_companies = {
            # Tech companies
            'tesla': 'industry',
            'google': 'industry',
            'microsoft': 'industry',
            'apple': 'industry',
            'amazon': 'industry',
            'facebook': 'industry',
            'meta': 'industry',
            'ibm': 'industry',
            'intel': 'industry',
            'oracle': 'industry',
            'cisco': 'industry',
            'hp': 'industry',
            'dell': 'industry',
            'nvidia': 'industry',
            'amd': 'industry',
            'salesforce': 'industry',
            'sap': 'industry',
            'tencent': 'industry',
            'alibaba': 'industry',
            'baidu': 'industry',
            'samsung': 'industry',
            'huawei': 'industry',
            'qualcomm': 'industry',
            'broadcom': 'industry',
            'adobe': 'industry',
            'vmware': 'industry',
            
            # Energy utilities
            'xcel energy': 'utility',
            'duke energy': 'utility',
            'engie': 'utility',
            'edf': 'utility',
            'rwe': 'utility',
            'e.on': 'utility',
            'iberdrola': 'utility',
            'vattenfall': 'utility',
            'enel': 'utility',
            'national grid': 'utility',
            'ørsted': 'utility',
            'orsted': 'utility',
            'nextera': 'utility',
            'dominion': 'utility', 
            'southern company': 'utility',
            'exelon': 'utility',
            'berkshire hathaway energy': 'utility',
            'firstenergy': 'utility',
            'consolidated edison': 'utility',
            'con edison': 'utility',
            'nrg energy': 'utility',
            'vistra': 'utility',
            'cms energy': 'utility',
            'pge': 'utility',
            'pg&e': 'utility',
            'sce': 'utility',
            'sdge': 'utility',
            'dte energy': 'utility',
            'ameren': 'utility',
            
            # Oil and gas / energy
            'exxon': 'industry',
            'exxonmobil': 'industry',
            'shell': 'industry',
            'total': 'industry',
            'totalenergies': 'industry',
            'bp': 'industry',
            'chevron': 'industry',
            'conocophillips': 'industry',
            'marathon': 'industry',
            'valero': 'industry',
            'occidental': 'industry',
            'saudi aramco': 'industry',
            'aramco': 'industry',
            'equinor': 'industry',
            'gazprom': 'industry',
            'rosneft': 'industry',
            'lukoil': 'industry',
            'petrobras': 'industry',
            'pemex': 'industry',
            'petrochina': 'industry',
            'sinopec': 'industry',
            'cnooc': 'industry',
            'eni': 'industry',
            'phillips 66': 'industry',
            
            # Financial institutions, banks, and investment firms
            'blackrock': 'financial',
            'black rock': 'financial',
            'blackstone': 'financial',
            'kkr': 'financial',
            'carlyle': 'financial',
            'apollo': 'financial',
            'tpg': 'financial',
            'bain capital': 'financial',
            'vanguard': 'financial',
            'fidelity': 'financial',
            'state street': 'financial',
            'wellington': 'financial',
            'capital group': 'financial',
            'jpmorgan': 'financial',
            'jp morgan': 'financial',
            'goldman sachs': 'financial',
            'morgan stanley': 'financial',
            'bank of america': 'financial',
            'bofa': 'financial',
            'citigroup': 'financial',
            'citi': 'financial',
            'wells fargo': 'financial',
            'hsbc': 'financial',
            'barclays': 'financial',
            'deutsche bank': 'financial',
            'bnp paribas': 'financial',
            'credit suisse': 'financial',
            'ubs': 'financial',
            'bank of china': 'financial',
            'icbc': 'financial',
            'mitsubishi ufj': 'financial',
            'mufg': 'financial',
            'royal bank of canada': 'financial',
            'rbc': 'financial',
            'toronto dominion': 'financial',
            'td bank': 'financial',
            'scotiabank': 'financial',
            
            # Development Banks and Multilateral Financial Institutions
            'world bank': 'financial',
            'idb': 'financial',
            'inter-american development bank': 'financial',
            'adb': 'financial',
            'asian development bank': 'financial',
            'afdb': 'financial',
            'african development bank': 'financial',
            'ebrd': 'financial',
            'european bank for reconstruction': 'financial',
            'eib': 'financial',
            'european investment bank': 'financial',
            'isdb': 'financial',
            'islamic development bank': 'financial',
            'ndb': 'financial',
            'new development bank': 'financial',
            'brics bank': 'financial',
            'aiib': 'financial',
            'asian infrastructure investment bank': 'financial',
            'cdb': 'financial',
            'caribbean development bank': 'financial',
            'cabei': 'financial',
            'central american bank': 'financial',
            'caf': 'financial',
            'development bank of latin america': 'financial',
            'dbsa': 'financial',
            'development bank of southern africa': 'financial',
            'imf': 'financial',
            'international monetary fund': 'financial',
            'ifc': 'financial', 
            'international finance corporation': 'financial',
            'kfw': 'financial',
            'gef': 'financial',
            'global environment facility': 'financial',
            'green climate fund': 'financial',
            'gcf': 'financial',
            
            # Government and NGOs
            'eia': 'government',
            'iea': 'ngo',
            'irena': 'ngo',
            
            # Professional services
            'catalyst.coop': 'professional',
            'catalyst cooperative': 'professional',
            'fermata energy': 'industry',
            'mckinsey': 'professional',
            'boston consulting': 'professional',
            'bcg': 'professional',
            'bain & company': 'professional',
            'deloitte': 'professional',
            'pwc': 'professional',
            'kpmg': 'professional',
            'ernst & young': 'professional',
            'ey': 'professional',
            'accenture': 'professional',
        }
        
        # Specific academic institutions
        academic_institutions = [
            'eth z', 'fraunhofer', 'tsinghua', 'tu delft', 'unc charlotte',
            'technical university of berlin', 'tu berlin', 'imperial college',
            'harvard', 'mit', 'stanford', 'berkeley', 'cambridge', 'oxford',
            'caltech', 'princeton', 'yale', 'columbia', 'epfl', 'eth zurich',
            'delft', 'rwth aachen', 'tu munich', 'kyoto', 'tokyo', 'copenhagen',
            'mcgill', 'toronto', 'waterloo', 'eth zürich', 'uzh', 'karlsruhe',
            'helmholtz', 'max planck', 'cnrs', 'csic'
        ]
        
        # Specific NGOs and think tanks
        ngos = [
            'stockholm environment institute', 'international energy agency', 
            'world resources institute', 'natural resources defense council',
            'greenpeace', 'wwf', 'sierra club', 'climate works', 'rocky mountain institute',
            'rmi', 'environmental defense fund', 'c2es', 'brookings', 'pew', 'edf'
        ]
        
        # Specific government agencies
        gov_agencies = [
            'doe', 'department of energy', 'epa', 'ferc', 'nrel', 'lbnl', 'pnnl', 'ornl',
            'anl', 'inl', 'bnl', 'lanl', 'sandia', 'srnl', 'netl', 'llnl'
        ]
        
        # Check for notable companies first
        for notable, cls in notable_companies.items():
            if notable in company:
                return cls
                
        # Check for specific academic institutions
        if any(academic in company for academic in academic_institutions):
            return 'academic'
            
        # Check for specific NGOs
        if any(ngo in company for ngo in ngos):
            return 'ngo'
            
        # Check for specific government agencies
        if any(agency in company for agency in gov_agencies):
            return 'government'
        
        # Check for academic institutions with broader keywords
        academic_keywords = ['university', 'college', 'institute of technology', 'polytechnic', 
                            'academia', 'education', 'research institute', 'laboratory', 'lab', 
                            'école', 'universität', 'università', 'universidad', 'universit',
                            'school of', 'dept.', 'department of']
        
        # Check for utilities - but don't rely just on "energy" keyword
        utility_keywords = ['utility', 'power company', 'electric company', 'electricity provider', 
                          'grid operator', 'transmission company', 'distribution company', 
                          'generation company', 'power supplier']
        
        # Check for industry
        industry_keywords = ['inc', 'corp', 'llc', 'ltd', 'limited', 'company', 'gmbh', 'ag', 'co.', 
                           'consulting', 'solutions', 'software', 'technologies', 'plc', 
                           'partners', 'group', 'inc.', 'holdings']
        
        # Check for financial institutions
        financial_keywords = ['bank', 'capital', 'investment', 'asset management', 'financial',
                             'insurance', 'fund', 'hedge fund', 'private equity', 'venture capital',
                             'wealth management', 'asset manager']
        
        # Check for RTO (Regional Transmission Operators)
        rto_keywords = ['rto', 'iso', 'miso', 'pjm', 'caiso', 'ercot', 'iso-ne', 'nyiso', 'spp', 
                      'independent system operator', 'regional transmission', 'ieso', 'aeso']
        
        # Check for research organizations
        research_keywords = ['research', 'technology organization', 'r&d', 'innovation', 'laboratory', 
                      'national lab', 'science', 'scientific', 'institute', 'technology center']
        
        # Classify based on keywords
        if any(keyword in company for keyword in academic_keywords):
            classification = 'academic'
        elif any(keyword in company for keyword in rto_keywords):
            classification = 'rto'
        elif any(keyword in company for keyword in utility_keywords):
            classification = 'utility'
        elif any(keyword in company for keyword in financial_keywords):
            classification = 'financial'
        elif any(keyword in company for keyword in research_keywords):
            classification = 'research_organization'
        elif any(keyword in company for keyword in industry_keywords):
            classification = 'industry'
    
    # Email hints
    if user_data.get('email') and classification == 'unknown':
        email = user_data['email'].lower()
        email_domain = email.split('@')[-1].lower()
        
        # Student email detection
        if 'student' in email:
            return 'academic'
        
        # Academic email domains
        academic_domains = ['edu', 'ac.uk', 'ac.jp', 'edu.au', 'ac.nz', 'edu.sg', 'uni-', 
                          '.edu.', '.ac.', '.universit']
        if any(domain in email_domain for domain in academic_domains):
            classification = 'academic'
            
        # Financial institutions
        if 'bank' in email:
            classification = 'financial'
    
    # Bio hints (if still unknown)
    if user_data.get('bio') and classification == 'unknown':
        bio = user_data['bio'].lower()
        if any(word in bio for word in ['professor', 'phd', 'student', 'researcher', 'postdoc']):
            classification = 'academic'
        elif any(word in bio for word in ['engineer', 'developer', 'scientist', 'consultant']):
            classification = 'professional'
        elif 'bank' in bio:
            classification = 'financial'
    
    return classification

def normalize_company_name(company):
    """Normalize company name to handle different forms of the same organization"""
    if not company:
        return ""
    
    # Convert to lowercase and strip spaces
    normalized = company.lower().strip()
    
    # Remove common prefixes/suffixes and symbols
    normalized = normalized.replace('@', '')
    normalized = normalized.strip('.,;:()-[]{}/')
    
    # Define common name variations
    name_variations = {
        # Research Labs
        'nrel': 'national renewable energy laboratory',
        'national renewable energy lab': 'national renewable energy laboratory',
        'pnnl': 'pacific northwest national laboratory',
        'pacific northwest national lab': 'pacific northwest national laboratory',
        'lbnl': 'lawrence berkeley national laboratory',
        'lawrence berkeley national lab': 'lawrence berkeley national laboratory',
        'ornl': 'oak ridge national laboratory',
        'oak ridge national lab': 'oak ridge national laboratory',
        'anl': 'argonne national laboratory',
        'argonne national lab': 'argonne national laboratory',
        'lanl': 'los alamos national laboratory',
        'los alamos national lab': 'los alamos national laboratory',
        'inl': 'idaho national laboratory',
        'idaho national lab': 'idaho national laboratory',
        'bnl': 'brookhaven national laboratory',
        'brookhaven national lab': 'brookhaven national laboratory',
        
        # Universities
        'mit': 'massachusetts institute of technology',
        'ethz': 'eth zurich',
        'eth zürich': 'eth zurich',
        'eth zurich': 'eth zurich',
        'tu berlin': 'technical university of berlin',
        'tu delft': 'delft university of technology',
        'tu munich': 'technical university of munich',
        'unc': 'university of north carolina',
        'tum': 'technical university of munich',
        'rwth': 'rwth aachen university',
        'tsinghua': 'tsinghua university',
        
        # Energy Organizations
        'iea': 'international energy agency',
        'irena': 'international renewable energy agency',
        'eia': 'energy information administration',
        'epri': 'electric power research institute',
        
        # Financial Institutions
        'wb': 'world bank',
        'world bank group': 'world bank',
        'imf': 'international monetary fund',
        'idb': 'inter-american development bank',
        'adb': 'asian development bank',
        'afdb': 'african development bank',
        'ebrd': 'european bank for reconstruction and development',
        'eib': 'european investment bank',
        
        # Companies
        'ms': 'microsoft',
        'msft': 'microsoft',
        'fb': 'facebook',
        'amzn': 'amazon',
        'goog': 'google',
        'aapl': 'apple',
        'ge': 'general electric',
        'bp': 'british petroleum',
    }
    
    # Check for direct match in name variations
    if normalized in name_variations:
        return name_variations[normalized]
    
    # Check for company name inside larger string
    for short_form, full_form in name_variations.items():
        # Don't replace if it's just a short substring that could match many things
        if len(short_form) > 2:
            if short_form in normalized:
                # Only replace if it's a standalone word or surrounded by non-alphanumeric characters
                import re
                pattern = r'(^|\W)' + re.escape(short_form) + r'($|\W)'
                if re.search(pattern, normalized):
                    normalized = re.sub(pattern, r'\1' + full_form + r'\2', normalized)
    
    return normalized

def analyze_repo_users(repo_name, users_data):
    """Analyze user data for a specific repository"""
    # Apply classification to all users
    for interaction_type in users_data:
        for user in users_data[interaction_type]:
            user['classification'] = classify_user(user)
    
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
    
    # Identify top organizations (with normalization)
    organizations = []
    for user in all_users:
        if user.get('company'):
            # Normalize company name
            company = normalize_company_name(user.get('company'))
            if company:
                organizations.append({
                    'name': company,
                    'original_name': user.get('company'),
                    'classification': user.get('classification', 'unknown')
                })
    
    # Count organizations
    org_names = [org['name'] for org in organizations]
    company_counts = Counter(org_names)
    
    # Group organizations by type
    orgs_by_type = {}
    for org in organizations:
        org_type = org['classification']
        org_name = org['name']
        if org_type not in orgs_by_type:
            orgs_by_type[org_type] = Counter()
        orgs_by_type[org_type][org_name] += 1
    
    # Identify locations
    locations = [user['location'] for user in all_users if user.get('location')]
    location_counts = Counter(locations)
    
    return {
        'repo_name': repo_name,
        'total_users': len(all_users),
        'classification_counts': dict(class_counts),
        'contributor_class_counts': dict(contributor_class_counts),
        'top_organizations': dict(company_counts.most_common(10)),
        'organizations_by_type': {k: dict(v.most_common(5)) for k, v in orgs_by_type.items()},
        'top_locations': dict(location_counts.most_common(10)),
    }

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
    
    # Top organizations by type
    all_orgs_by_type = {}
    for r in results:
        orgs_by_type = r.get('organizations_by_type', {})
        for org_type, orgs in orgs_by_type.items():
            if org_type not in all_orgs_by_type:
                all_orgs_by_type[org_type] = Counter()
            all_orgs_by_type[org_type].update(orgs)
    
    print("\nTop organizations by type:")
    for org_type, orgs in sorted(all_orgs_by_type.items()):
        if orgs:  # Skip empty categories
            print(f"\n  {org_type.capitalize()}:")
            for org, count in orgs.most_common(5):
                if org:  # Skip empty org names
                    print(f"    {org}: {count}")
    
    # Repository breakdown
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
        
        f.write("\nTop organizations by type:\n")
        for org_type, orgs in sorted(all_orgs_by_type.items()):
            if orgs:
                f.write(f"\n  {org_type.capitalize()}:\n")
                for org, count in orgs.most_common(5):
                    if org:
                        f.write(f"    {org}: {count}\n")
        
        f.write("\nRepository breakdown:\n")
        for r in results:
            f.write(f"\n{r['repo_name']}:\n")
            f.write(f"  Total users: {r['total_users']}\n")
            f.write("  User classifications:\n")
            for cls, count in Counter(r['classification_counts']).most_common():
                f.write(f"    {cls}: {count} ({count/r['total_users']*100:.1f}%)\n")

def main():
    """Main function for analyzing GitHub user data"""
    print("GitHub User Analyzer")
    print("This script analyzes previously fetched GitHub user data")
    
    # Load raw user data
    repos_data = load_raw_user_data()
    if not repos_data:
        return
    
    print(f"\nFound data for {len(repos_data)} repositories")
    
    # Ask if user wants to analyze specific repositories
    repo_names = [repo['repo_name'] for repo in repos_data]
    print("\nAvailable repositories:")
    for i, name in enumerate(repo_names):
        print(f"{i+1}. {name}")
    
    selection = input("\nEnter repository numbers to analyze (comma-separated), or 'all' for all repos: ")
    
    selected_repos = []
    if selection.lower() == 'all':
        selected_repos = repos_data
    else:
        try:
            indices = [int(idx.strip()) - 1 for idx in selection.split(',')]
            selected_repos = [repos_data[i] for i in indices if 0 <= i < len(repos_data)]
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
        print(f"\nAnalyzing {repo['repo_name']}...")
        
        # Analyze the data
        analysis = analyze_repo_users(repo['repo_name'], repo['data'])
        
        # Add to results
        results.append(analysis)
    
    # Save analysis results
    with open("user_analysis_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print("\nAnalysis complete! Results saved to user_analysis_results.json")
    print("The dashboard will use this file for visualization")
    
    # Generate summary report
    generate_summary_report(results)

if __name__ == "__main__":
    main() 