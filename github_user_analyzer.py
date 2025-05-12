import pandas as pd
import numpy as np
import json
import os
import glob
import time
from collections import Counter
from github import Github
from tqdm import tqdm
from dotenv import load_dotenv
import pycountry
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
import argparse

# Load environment variables
load_dotenv()

# Common organization name variations for standardization
ORG_NAME_VARIATIONS = {
    "national renewable energy laboratory": ["nrel", "national renewable energy lab", "nrel.gov", "@nrel", "nrel/", "nrel,", "@nrel , uw"],
    "lawrence berkeley national laboratory": ["lbnl", "berkeley lab", "lawrence berkeley lab", "lbl", "@lbl", "@lbnl"],
    "pacific northwest national laboratory": ["pnnl", "pacific northwest lab", "pnnl.gov", "@pnnl"],
    "oak ridge national laboratory": ["ornl", "oak ridge lab", "@ornl", "ornl.gov"],
    "argonne national laboratory": ["anl", "argonne lab", "@anl", "anl.gov"],
    "lawrence livermore national laboratory": ["llnl", "livermore lab", "@llnl", "llnl.gov"],
    "los alamos national laboratory": ["lanl", "los alamos lab", "@lanl", "lanl.gov"],
    "sandia national laboratories": ["sandia", "snl", "@sandia", "snl.gov", "sandia.gov"],
    "brookhaven national laboratory": ["bnl", "brookhaven lab", "@bnl", "bnl.gov"],
    "idaho national laboratory": ["inl", "idaho lab", "@inl", "inl.gov"],
    "slac national accelerator laboratory": ["slac", "@slac", "slac.gov"],
    "fermi national accelerator laboratory": ["fermilab", "fnal", "@fermilab", "fnal.gov"],
    "national energy technology laboratory": ["netl", "@netl", "netl.gov"],
    "electric power research institute": ["epri", "@epri", "epri.org", "electric power research inst"],
    "university of california berkeley": ["uc berkeley", "berkeley", "ucb", "cal", "university of california at berkeley", "uc-berkeley", "berkeley.edu"],
    "massachusetts institute of technology": ["mit", "m.i.t.", "massachusetts institute of tech", "mit.edu", "@mit"],
    "stanford university": ["stanford", "stanford.edu", "@stanford"],
    "eth zurich": ["ethz", "eth zürich", "eidgenössische technische hochschule zürich", "eth", "ethz.ch", "@eth"],
    "imperial college london": ["imperial college", "imperial", "ic london", "icl", "imperial.ac.uk"],
    "technical university of denmark": ["dtu", "danmarks tekniske universitet", "dtu.dk", "@dtu"],
    "university of cambridge": ["cambridge", "cam", "cambridge.ac.uk", "@cambridge"],
    "university of oxford": ["oxford", "ox", "oxford.ac.uk", "@oxford"],
    "tsinghua university": ["tsinghua", "清华大学", "tsinghua.edu.cn", "@tsinghua", "tsinghua university - tbsi", "tsinghua tbsi"],
    "california independent system operator": ["caiso", "california iso", "ca iso", "caiso.com"],
    "midcontinent independent system operator": ["miso", "midwest iso", "miso.org"],
    "pjm interconnection": ["pjm", "pjm.com", "@pjm"],
    "independent system operator of new england": ["iso-ne", "iso new england", "iso-ne.com"],
    "independent electricity system operator": ["ieso", "ieso.ca"],
    "alberta electric system operator": ["aeso", "aeso.ca"],
    "united states department of energy": ["doe", "us department of energy", "department of energy", "u.s. department of energy", "energy.gov", "doe.gov", "@doe", "us doe"],
    "siemens": ["siemens energy", "siemens ag", "siemens.com"],
    "general electric": ["ge", "ge energy", "ge.com"],
    "abb": ["abb group", "asea brown boveri", "abb.com"],
    "schneider electric": ["schneider", "schneider.com"],
    "tesla": ["tesla energy", "tesla inc", "tesla, inc.", "tesla.com"],
    "power systems engineering research center": ["pserc", "pserc.org"],
    "energy systems integration group": ["esig", "esig.energy"],
    "google": ["google.com", "google llc", "google inc", "alphabet", "@google"],
    "microsoft": ["microsoft corporation", "microsoft corp", "msft", "microsoft.com", "@microsoft"],
    "meta": ["facebook", "meta platforms", "fb", "meta.com", "facebook.com", "@meta", "@facebook"],
    "university of north carolina at charlotte": ["unc charlotte", "uncc", "charlotte", "uncc.edu", "@uncc"],
    # Updated financial institutions section - use more explicit name formats with enhanced Rabobank entry
    "rabobank": [
        "rabobank international", "rabobank nederland", "rabobank group", 
        "rabobank.com", "@rabobank.com", "@rabobank", "coöperatieve rabobank",
        "rabobank n.a.", "rabobank na", "rabobank banking", "rabo bank"
    ],
    "ing bank": ["ing group", "ing bank n.v.", "ing.com", "@ing.com", "@ing"],
    "abn amro": ["abn amro bank", "abn-amro", "abnamro.com", "@abnamro.com", "@abnamro"],
    "société générale": ["societe generale", "socgen", "societegenerale.com", "@societegenerale.com", "@socgen.com", "@socgen"],
    "credit agricole": ["crédit agricole", "credit-agricole", "credit-agricole.com", "@creditagricole.com", "@creditagricole"],
    "natixis": ["natixis investment managers", "natixis bank", "natixis.com", "@natixis.com", "@natixis"],
    "natwest": ["natwest group", "natwest bank", "natwest.com", "@natwest.com", "@natwest"],
    "lloyds bank": ["lloyds banking group", "lloyds", "lloydsbank.com", "@lloyds.com", "@lloyds"],
    "santander": ["banco santander", "santander bank", "santander.com", "@santander.com", "@santander"],
    # Add explicit entry for IIASA to prevent false matches
    "international institute for applied systems analysis": [
        "iiasa", "@iiasa", "iiasa.ac.at", "@iiasa.ac.at", 
        "international institute for applied systems", "applied systems analysis",
        "iiasa schlossplatz", "iiasa laxenburg", "laxenburg iiasa"
    ],
    # Add more consultancies and professional services
    "accenture": ["accenture plc", "accenture consulting", "accenture.com", "@accenture.com", "@accenture"],
    "mckinsey": ["mckinsey & company", "mckinsey & co", "mckinsey.com", "@mckinsey.com", "@mckinsey"],
    "deloitte": ["deloitte touche tohmatsu", "deloitte consulting", "deloitte.com", "@deloitte.com", "@deloitte"],
    "boston consulting group": ["bcg", "the boston consulting group", "bcg.com", "@bcg.com", "@bcg"],
    "kpmg": ["kpmg international", "kpmg llp", "kpmg.com", "@kpmg.com", "@kpmg"],
    "pwc": ["pricewaterhousecoopers", "price waterhouse coopers", "pwc.com", "@pwc.com", "@pwc"],
    "ey": ["ernst & young", "ernst and young", "ey.com", "@ey.com", "@ey"],
    "bain & company": ["bain and company", "bain & co", "bain.com", "@bain.com", "@bain"],
    # Add more independent/professional individuals
    "independent consultant": ["freelance consultant", "independent professional", "self-employed", "freelancer"],
    "independent researcher": ["independent research", "independent scholar", "research consultant"],
    "independent developer": ["freelance developer", "independent software developer", "software consultant"]
}

# Define distinctive keyword patterns and anti-patterns for organization matching
# This provides a more systematic approach than one-off special handling
ORG_KEYWORDS = {
    "international institute for applied systems analysis": {
        "required": ["iiasa"],
        "exclusions": ["rabobank", "bank", "financial", "credit"]
    },
    "rabobank": {
        "required": ["rabobank"],
        "exclusions": ["iiasa", "systems analysis", "applied systems"]
    },
    "university of california berkeley": {
        "required": ["berkeley"],
        "preferred": ["uc", "california"]
    },
    "lawrence berkeley national laboratory": {
        "required": ["berkeley", "lab"],
        "preferred": ["lawrence", "national", "lbnl"]
    },
    "national renewable energy laboratory": {
        "required": ["renewable", "energy"],
        "preferred": ["national", "laboratory", "nrel"]
    },
    # Add more distinctive patterns for other organizations as needed
}

# Define a more comprehensive mapping of organizations to their types
ORG_TYPE_MAPPING = {
    # Government - National Labs and Agencies
    "national renewable energy laboratory": "government",
    "lawrence berkeley national laboratory": "government",
    "pacific northwest national laboratory": "government",
    "oak ridge national laboratory": "government", 
    "argonne national laboratory": "government",
    "lawrence livermore national laboratory": "government", 
    "los alamos national laboratory": "government",
    "sandia national laboratories": "government",
    "brookhaven national laboratory": "government",
    "idaho national laboratory": "government",
    "slac national accelerator laboratory": "government",
    "fermi national accelerator laboratory": "government",
    "national energy technology laboratory": "government",
    "united states department of energy": "government",
    
    # Research Organizations
    "international institute for applied systems analysis": "research_organization",
    
    # Utilities and System Operators
    "electric power research institute": "utility",
    "california independent system operator": "utility",
    "midcontinent independent system operator": "utility",
    "pjm interconnection": "utility",
    "independent system operator of new england": "utility",
    "independent electricity system operator": "utility", 
    "alberta electric system operator": "utility",
    
    # Universities
    "university of north carolina at charlotte": "academic",
    "university of california berkeley": "academic",
    "massachusetts institute of technology": "academic",
    "stanford university": "academic",
    "eth zurich": "academic",
    "imperial college london": "academic",
    "technical university of denmark": "academic",
    "university of cambridge": "academic",
    "university of oxford": "academic",
    "tsinghua university": "academic",
    
    # Financial Institutions
    "rabobank": "financial",
    "ing bank": "financial",
    "abn amro": "financial",
    "société générale": "financial",
    "credit agricole": "financial",
    "natixis": "financial",
    "natwest": "financial",
    "lloyds bank": "financial",
    "santander": "financial",
    
    # Professional Services / Consultancies
    "accenture": "professional",
    "mckinsey": "professional",
    "deloitte": "professional",
    "boston consulting group": "professional",
    "kpmg": "professional",
    "pwc": "professional",
    "ey": "professional",
    "bain & company": "professional",
    "independent consultant": "professional",
    "independent researcher": "professional",
    "independent developer": "professional",
    
    # Industry / Technology
    "siemens": "industry",
    "general electric": "industry",
    "abb": "industry",
    "schneider electric": "industry",
    "tesla": "industry",
    "google": "industry",
    "microsoft": "industry",
    "meta": "industry",
}

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
    
    # Check for company/organization hints
    if not user_data.get('company'):
        # If no company info, try other methods like email or bio
        return check_other_classification_hints(user_data)
    
    # Get normalized company name
    company = user_data['company'].lower().strip()
    
    # Skip very short company names
    if len(company) <= 1:
        return check_other_classification_hints(user_data)
        
    # First normalize the company name to check for known variations
    normalized_name, dashboard_org_name = normalize_org_name(company)
    
    # Check if we have a direct mapping for this organization
    if normalized_name in ORG_TYPE_MAPPING:
        return ORG_TYPE_MAPPING[normalized_name]
        
    # Check if any variation matches our known organizations
    standard_name = None
    for std_name, variations in ORG_NAME_VARIATIONS.items():
        if normalized_name == std_name or normalized_name in variations:
            standard_name = std_name
            break
    
    if standard_name and standard_name in ORG_TYPE_MAPPING:
        return ORG_TYPE_MAPPING[standard_name]
    
    # Use the organization keywords system for classification
    for std_name, keywords in ORG_KEYWORDS.items():
        if std_name in ORG_TYPE_MAPPING:
            # Get required keywords for this organization
            required = keywords.get('required', [])
            # Check if all required keywords are present
            if required and all(keyword in company for keyword in required):
                # Check if any exclusion keywords are present
                exclusions = keywords.get('exclusions', [])
                if not exclusions or not any(keyword in company for keyword in exclusions):
                    # Get any preferred keywords (additional signals that strengthen the match)
                    preferred = keywords.get('preferred', [])
                    # If we have preferred keywords and at least one is present, or if there are no preferred keywords
                    if not preferred or any(keyword in company for keyword in preferred):
                        return ORG_TYPE_MAPPING[std_name]
    
    # For organizations not in our specific pattern list, use category keywords
    # Organized by classification type for maintainability
    category_keywords = {
        'government': [
            'national laboratory', 'national lab', 'national laboratories',
            'government', 'federal', 'ministry', 'department of', 'agency',
            'nrel', 'lbnl', 'pnnl', 'ornl', 'anl', 'lanl', 'llnl', 'bnl', 'inl', 'snl',
            'slac', 'fermilab', 'netl', 'doe', 'regulatory', 'commission'
        ],
        'academic': [
            'university', 'college', 'school', 'institute of technology',
            'polytechnic', 'faculty', 'department', 'laboratory', 'research center'
        ],
        'utility': [
            'utility', 'utilities', 'power company', 'energy company', 'electric company',
            'system operator', 'transmission operator', 'grid operator', 'iso', 'rto', 'tso', 'dso'
        ],
        'financial': [
            'bank', 'banking', 'financial', 'finance', 'investment', 'investor',
            'credit', 'venture', 'capital', 'securities', 'insurance'
        ],
        'professional': [
            'consultant', 'consulting', 'consultancy', 'advisor', 'freelance',
            'independent', 'contractor', 'self-employed', 'llc', 'ltd', 'gmbh', 'partner'
        ],
        'industry': [
            'technology', 'software', 'hardware', 'engineering', 'solutions',
            'corporation', 'corp', 'inc', 'company', 'manufacturer', 'systems'
        ]
    }
    
    # Check each category for keyword matches
    for category, terms in category_keywords.items():
        for term in terms:
            if term in company:
                return category
                
    # Check if it's just "independent" or similar
    if company.strip() in ["independent", "consultant", "self-employed", "freelance"]:
        return 'professional'
    
    # If still no match, check other hints like email or bio
    return check_other_classification_hints(user_data)

# Add a helper function to check other classification hints
def check_other_classification_hints(user_data):
    """Check email and bio for classification hints when company name isn't sufficient"""
    # Check email for academic or government domains
    if user_data.get('email'):
        email = user_data['email'].lower()
        parts = email.split('@')
        if len(parts) > 1:
            email_domain = parts[1]
            
            # Student/academic email detection
            if 'student' in email or 'edu' in email_domain or 'ac.' in email_domain:
                return 'academic'
            
            # Government domains
            if 'gov' in email_domain or 'mil' in email_domain:
                return 'government'
    
    # Check bio for classification hints
    if user_data.get('bio'):
        bio = user_data['bio'].lower()
        
        # Academic hints
        if any(term in bio for term in ['professor', 'phd', 'student', 'researcher', 'postdoc', 'faculty']):
            return 'academic'
        
        # Professional hints
        if any(term in bio for term in ['consultant', 'advisor', 'freelance', 'independent']):
            return 'professional'
        
        # Industry hints
        if any(term in bio for term in ['engineer', 'developer', 'scientist', 'software']):
            return 'industry'
            
        # Government hints
        if any(term in bio for term in ['government', 'ministry', 'department', 'agency', 'regulator']):
            return 'government'
    
    # Default if no classification could be determined
    return 'unknown'

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

def normalize_org_name(org_name):
    """
    Normalize an organization name to a standard form.
    Returns tuple of (normalized_name, dashboard_org_name)
    """
    if not org_name:
        return None, None
    
    # Create normalized version (lowercase, no punctuation, normalized whitespace)
    normalized = org_name.lower().strip()
    normalized = normalized.replace(',', ' ').replace('.', ' ').replace('-', ' ')
    normalized = ' '.join(normalized.split())  # Normalize whitespace
    
    # Skip very short organization names (likely initials that will cause false matches)
    if len(normalized) <= 1:
        return normalized, org_name.strip()
    
    # Handle common prefixes in email/web format
    for prefix in ['@', 'http://', 'https://', 'www.']:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
    
    # Handle common suffixes like .com, .org, .edu
    for suffix in ['.com', '.org', '.edu', '.gov', '.net', '.io', '.ac.uk', '.de', '.fr', '.cn', '.ac.at']:
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)]
    
    # Remove common department/division indicators that cause missed matches
    common_suffixes = [
        ' department of', ' dept of', ' school of', ' college of',
        ' laboratory', ' lab', ' labs', ' center', ' centre',
        ' institute', ' division', ' dept', ' research group', 
        ' group', ' tbsi', ' team', ' university - '
    ]
    
    for suffix in common_suffixes:
        suffix_pos = normalized.find(suffix)
        if suffix_pos > 0:
            # Only trim the suffix if it's not the entire name
            if suffix_pos > 3:  # At least 3 chars before the suffix
                # Keep the main part of the name
                original_normalized = normalized
                normalized = normalized[:suffix_pos].strip()
                # But if the result is too short, revert
                if len(normalized) < 3:
                    normalized = original_normalized
    
    # Apply the organization matching based on keyword patterns and anti-patterns
    # This replaces special-case handling with a more systematic approach
    
    # Check against our dictionary of organization variations
    for std_name, variations in ORG_NAME_VARIATIONS.items():
        # First check for exact matches (highest confidence)
        if normalized == std_name or normalized in variations:
            return std_name, std_name.title()
        
        # Then check for strong keyword matches
        # Look for distinctive keywords that strongly identify this organization
        if ORG_KEYWORDS.get(std_name):
            required_keywords = ORG_KEYWORDS[std_name].get('required', [])
            if required_keywords:
                # Check if all required keywords are present
                if all(keyword in normalized for keyword in required_keywords):
                    # Check if any exclusion keywords are present
                    exclusion_keywords = ORG_KEYWORDS[std_name].get('exclusions', [])
                    if not exclusion_keywords or not any(keyword in normalized for keyword in exclusion_keywords):
                        return std_name, std_name.title()
    
    # Try acronym matching
    # 1. Check if the normalized name might be a standalone acronym (all caps in original)
    original_words = org_name.split()
    if len(original_words) == 1 and original_words[0].isupper() and len(original_words[0]) >= 3:
        possible_acronym = normalized
        # Look for this acronym in all standard names and variations
        for standard_name, variations in ORG_NAME_VARIATIONS.items():
            # Generate acronym from standard name
            standard_acronym = ''.join(word[0] for word in standard_name.split() if word)
            if possible_acronym == standard_acronym.lower():
                return standard_name, standard_name.title()
            # Also check the first two letters of two-word organizations (like "LA" for "Los Angeles")
            if len(standard_name.split()) == 2 and len(possible_acronym) == 2:
                alt_acronym = standard_name.split()[0][:1] + standard_name.split()[1][:1]
                if possible_acronym == alt_acronym.lower():
                    return standard_name, standard_name.title()
    
    # If no exact match in variations, try more careful word-based matching with higher standards
    for standard_name, variations in ORG_NAME_VARIATIONS.items():
        # Split the names into words
        std_words = [w for w in standard_name.split() if len(w) >= 3]
        norm_words = [w for w in normalized.split() if len(w) >= 3]
        
        # Skip if either has no qualifying words
        if not std_words or not norm_words:
            continue
        
        # Check for organizations with specific keyword requirements
        if standard_name in ORG_KEYWORDS:
            # Get the required keywords
            required = ORG_KEYWORDS[standard_name].get('required', [])
            if required:
                # Check if all required keywords are present
                if all(keyword in normalized for keyword in required):
                    # Check for exclusions
                    exclusions = ORG_KEYWORDS[standard_name].get('exclusions', [])
                    if not exclusions or not any(keyword in normalized for keyword in exclusions):
                        # Check for preferred keywords (if defined)
                        preferred = ORG_KEYWORDS[standard_name].get('preferred', [])
                        if not preferred or any(keyword in normalized for keyword in preferred):
                            return standard_name, standard_name.title()
            continue
            
        # Standard word overlap approach for organizations without specific patterns
        common_words = set(std_words) & set(norm_words)
        
        # Check for distinctive words that strongly identify an organization
        distinctive_words = set()
        for std_word in std_words:
            # Skip common words like "university", "institute", etc.
            if std_word not in ['university', 'institute', 'college', 'school', 'laboratory', 'department', 'center']:
                distinctive_words.add(std_word)
        
        # If there's a match on distinctive words (non-common terms), increase confidence
        distinctive_matches = distinctive_words & set(norm_words)
        
        # Require stronger matching criteria - increased thresholds
        # If we have multiple distinctive word matches, or a very high overlap
        if (distinctive_matches and len(distinctive_matches) >= 2) or \
           (common_words and (len(common_words) / len(std_words) >= 0.8 or len(common_words) / len(norm_words) >= 0.8)):
            # Additional safety check for problematic cases
            # Make sure Rabobank and IIASA aren't confused
            if standard_name == "rabobank" and ("iiasa" in normalized or "institute" in normalized or "applied" in normalized):
                continue
            if standard_name == "international institute for applied systems analysis" and "rabobank" in normalized:
                continue
                
            return standard_name, standard_name.title()
        
        # Handle cases where the normalized name is a known acronym
        # Generate acronym from standard name words
        if len(normalized) >= 3 and normalized.isalnum():  # Potential acronym
            standard_acronym = ''.join(word[0] for word in standard_name.split())
            if normalized == standard_acronym.lower():
                return standard_name, standard_name.title()
            
            # Also check if normalized name appears in variations
            for variation in variations:
                if normalized == variation or normalized in variation.split():
                    return standard_name, standard_name.title()
    
    # If no match found, use the original name but still lowercase
    return normalized, org_name.strip()

# Function to geocode locations
def geocode_locations(locations):
    """Geocode a dictionary of location strings to countries"""
    print("Geocoding locations...")
    
    # Initialize geolocator
    geolocator = Nominatim(user_agent="energy-models-analyzer")
    
    # Process locations
    country_counts = {}
    location_country_map = {}
    
    # First try to extract countries using pattern matching
    for location in tqdm(locations.keys()):
        country = extract_country(location)
        location_country_map[location] = country
    
    # Then try geocoding for locations without a country
    for location, country in tqdm(list(location_country_map.items())):
        if not country and len(location) > 3:
            try:
                # Try geocoding with a timeout
                geo = geolocator.geocode(location, timeout=5)
                if geo and geo.raw.get('display_name'):
                    # Extract country from geocoded result
                    addr_parts = geo.raw.get('display_name', '').split(',')
                    if addr_parts:
                        country = addr_parts[-1].strip()
                        location_country_map[location] = country
            except (GeocoderTimedOut, GeocoderUnavailable):
                # Skip if geocoding fails
                pass
            # Be nice to the geocoding service
            time.sleep(1)
    
    # Create the country counts dictionary
    for location, count in locations.items():
        country = location_country_map.get(location)
        if country:
            if country in country_counts:
                country_counts[country] += count
            else:
                country_counts[country] = count
    
    return country_counts, location_country_map

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

def analyze_repo_users(repo_name, users_data):
    """Analyze user data for a specific repository"""
    # Combine all users (removing duplicates)
    all_users = []
    user_logins = set()
    
    # First, ensure all users have classification and normalized organization fields
    for interaction_type in ['stargazers', 'forkers', 'contributors', 'watchers']:
        if interaction_type not in users_data:
            users_data[interaction_type] = []
            
        for user in users_data[interaction_type]:
            # Add classification if not present
            if 'classification' not in user:
                user['classification'] = classify_user(user)
            
            # Add normalized organization name if not present
            if 'company' in user and user['company'] and 'dashboard_org_name' not in user:
                normalized_name, dashboard_org_name = normalize_org_name(user['company'])
                user['normalized_org_name'] = normalized_name
                user['dashboard_org_name'] = dashboard_org_name
                
                # Add a match confidence score to filter low-quality matches
                if dashboard_org_name and dashboard_org_name != user['company'].strip():
                    # Calculate match confidence with improved algorithm
                    user['org_match_confidence'] = calculate_org_name_match_confidence(
                        normalized_name, 
                        user['company'].lower().strip()
                    )
                else:
                    # Direct match or no organization
                    user['org_match_confidence'] = 1.0 if dashboard_org_name else 0.0
            
            # If we have a dashboard_org_name, ensure consistency of classification
            # This ensures a single org is only associated with one type
            if user.get('dashboard_org_name'):
                # Re-classify to ensure consistency - now that we have the normalized org name
                user['classification'] = classify_user(user)
    
    # Now gather unique users - use login as the unique identifier
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
    
    # Identify top organizations using dashboard_org_name with match confidence threshold
    # Only consider matches with high confidence (85%+)
    # We also now track which users belong to each organization 
    companies = []
    companies_to_users = {}  # Track which users belong to each organization
    high_confidence_threshold = 0.85
    
    for user in all_users:
        org_name = None
        # Use high-confidence dashboard org name when available
        if user.get('dashboard_org_name') and user.get('org_match_confidence', 0) >= high_confidence_threshold:
            org_name = user['dashboard_org_name']
        # Otherwise use company name directly, but only if no dashboard name was assigned
        elif user.get('company') and not user.get('dashboard_org_name'):
            org_name = user['company'].strip()
            
        if org_name:
            companies.append(org_name)
            # Track which users belong to each org to check for suspicious counts
            if org_name not in companies_to_users:
                companies_to_users[org_name] = []
            companies_to_users[org_name].append(user['login'])
    
    company_counts = Counter(companies)

    # Debug: If any organization has suspiciously high counts, log the users
    for org, count in company_counts.most_common(10):
        if count > 3:  # Arbitrary threshold for checking
            print(f"Organization '{org}' has {count} users: {', '.join(companies_to_users[org][:5])}{'...' if len(companies_to_users[org]) > 5 else ''}")

    # Group organizations by classification - with high confidence filter
    orgs_by_type = {}
    for user in all_users:
        org_name = None
        # Use high-confidence dashboard org name when available
        if user.get('dashboard_org_name') and user.get('org_match_confidence', 0) >= high_confidence_threshold:
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
    
    # Convert to dictionary of dictionaries
    orgs_by_type_dict = {}
    for cls, counter in orgs_by_type.items():
        orgs_by_type_dict[cls] = dict(counter.most_common(10))
    
    # Identify locations
    locations = [user['location'] for user in all_users if user.get('location')]
    location_counts = Counter(locations)
    
    # Geocode locations to countries (process only top 50 locations to avoid rate limiting)
    print(f"Geocoding locations for {repo_name}...")
    country_counts, location_country_map = geocode_locations(dict(location_counts.most_common(50)))
    
    # Create a standard organization mapping for dashboard use
    org_mapping = {}
    for user in all_users:
        if user.get('company') and user.get('dashboard_org_name') and user.get('org_match_confidence', 0) >= high_confidence_threshold:
            org_mapping[user['company']] = user['dashboard_org_name']
    
    return {
        'repo_name': repo_name,
        'total_users': len(all_users),
        'classification_counts': dict(class_counts),
        'contributor_class_counts': dict(contributor_class_counts),
        'top_organizations': dict(company_counts.most_common(15)),
        'organizations_by_type': orgs_by_type_dict,
        'top_locations': dict(location_counts.most_common(20)),
        'country_counts': country_counts,
        'location_country_map': location_country_map,
        'organization_name_mapping': org_mapping,
        'high_confidence_threshold': high_confidence_threshold,
        'users_data': users_data,  # Include the processed user data
        'companies_to_users': companies_to_users  # For debugging
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

def calculate_org_name_match_confidence(normalized_name, original_name):
    """
    Calculate a confidence score for organization name matching.
    Returns a score between 0.0 and 1.0, where 1.0 is highest confidence.
    
    Args:
        normalized_name: The normalized organization name (from the ORG_NAME_VARIATIONS dictionary)
        original_name: The original user-provided company name
    
    Returns:
        float: Confidence score between 0.0 and 1.0
    """
    if not normalized_name or not original_name:
        return 0.0
    
    # Exact match gets highest confidence
    if normalized_name == original_name:
        return 1.0
    
    # Check for exact substring match (for cases like "@org")
    if normalized_name in original_name or original_name in normalized_name:
        # If one is a full substring of the other, give high confidence
        shorter = normalized_name if len(normalized_name) < len(original_name) else original_name
        longer = original_name if len(normalized_name) < len(original_name) else normalized_name
        
        # If the shorter string is very short, require it to be a prefix, suffix or surrounded by non-alphanumeric
        if len(shorter) <= 3:
            import re
            # Check if it's a whole word match
            pattern = r'(^|\W)' + re.escape(shorter) + r'($|\W)'
            if re.search(pattern, longer):
                return 0.9
            else:
                return 0.0  # Too short for a reliable partial match
    
    # Split into words for more detailed comparison
    norm_words = [w for w in normalized_name.split() if len(w) >= 3]
    orig_words = [w for w in original_name.replace(',', ' ').replace('.', ' ').split() if len(w) >= 3]
    
    if not norm_words or not orig_words:
        return 0.0
    
    # Calculate word overlap
    common_words = set(norm_words) & set(orig_words)
    
    # If no common significant words, it's not a match
    if not common_words:
        return 0.0
    
    norm_overlap = len(common_words) / len(norm_words) if norm_words else 0
    orig_overlap = len(common_words) / len(orig_words) if orig_words else 0
    
    # Use the higher of the two overlaps, but require significant overlap
    max_overlap = max(norm_overlap, orig_overlap)
    
    # Check organization-specific exclusion patterns from our keyword system
    for std_name, keywords in ORG_KEYWORDS.items():
        # If either name matches this organization's pattern
        if std_name in normalized_name or std_name in original_name:
            # Check for exclusion terms that would invalidate the match
            exclusions = keywords.get('exclusions', [])
            for exclusion in exclusions:
                # If exclusion term is in the other name, return very low confidence
                if (std_name in normalized_name and exclusion in original_name) or \
                   (std_name in original_name and exclusion in normalized_name):
                    return 0.1  # Very low confidence
    
    # Fall back to general term matching for organizations not in our specific patterns
    # Check for generally incompatible terms that shouldn't appear together
    incompatible_term_pairs = [
        ("bank", "laboratory"),
        ("financial", "laboratory"),
        ("university", "corporation"),
        ("academic", "industry"),
        ("government", "private")
    ]
    
    for term_a, term_b in incompatible_term_pairs:
        if ((term_a in normalized_name and term_b in original_name) or 
            (term_b in normalized_name and term_a in original_name)):
            return max_overlap * 0.2  # Significantly reduce confidence
    
    # Calculate Levenshtein distance for shorter names (handle spelling variations)
    # More similar = smaller distance
    try:
        import Levenshtein
        # Only consider Levenshtein for reasonably sized strings to avoid performance issues
        if len(normalized_name) < 50 and len(original_name) < 50:
            # Calculate normalized distance (0 to 1, where 0 is identical)
            normalized_distance = Levenshtein.distance(normalized_name, original_name) / max(len(normalized_name), len(original_name))
            # Convert to similarity (1 - distance)
            string_similarity = 1 - normalized_distance
            
            # If string similarity is very high, boost the confidence
            if string_similarity > 0.8:
                return min(1.0, max_overlap * 0.5 + string_similarity * 0.5 + 0.1)
    except ImportError:
        # Fall back to just word overlap if Levenshtein is not available
        pass
    
    # Boost confidence for organizations with specific preferred keywords
    for std_name, keywords in ORG_KEYWORDS.items():
        if std_name in normalized_name or std_name in original_name:
            # If we have preferred keywords defined for this organization
            preferred = keywords.get('preferred', [])
            if preferred:
                # Check if preferred keywords appear in both names
                matches = sum(1 for kw in preferred if 
                             (kw in normalized_name and kw in original_name))
                if matches > 0:
                    # Boost confidence based on number of matching preferred keywords
                    boost = 0.1 * matches
                    return min(1.0, max_overlap + boost)  # Cap at 1.0
    
    # Return base confidence from word overlap but make it more strict
    # We now require at least 85% overlap for organizations with few words
    if len(norm_words) <= 2 or len(orig_words) <= 2:
        return max_overlap if max_overlap >= 0.85 else 0.0
        
    # For longer names, use a curve that rewards higher overlap more strongly
    return pow(max_overlap, 1.3)  # Steeper exponential curve to reward higher overlap

def main():
    """Main function for analyzing GitHub user data"""
    print("GitHub User Analyzer")
    print("This script analyzes previously fetched GitHub user data")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze GitHub user data')
    parser.add_argument('--save_raw', action='store_true', help='Save raw user data for debugging')
    parser.add_argument('--reprocess', action='store_true', help='Reprocess all users with improved matching logic')
    args = parser.parse_args()
    
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
    
    print(f"\nAnalyzing {len(selected_repos)} repositories...")
    
    # Process each repository
    results = []
    for repo_data in selected_repos:
        repo_name = repo_data['repo_name']
        users_data = repo_data['data']
        
        print(f"\nAnalyzing {repo_name}...")
        
        # For reprocessing: reset classifications and organization names
        if args.reprocess:
            print(f"Reprocessing all users for {repo_name} with improved matching logic")
            for category in ['stargazers', 'forkers', 'contributors', 'watchers']:
                if category in users_data:
                    for user in users_data[category]:
                        # Remove existing classification and normalized org name
                        if 'classification' in user:
                            del user['classification']
                        if 'dashboard_org_name' in user:
                            del user['dashboard_org_name']
        
        # Analyze users
        repo_results = analyze_repo_users(repo_name, users_data)
        results.append(repo_results)
        
        print(f"Completed analysis for {repo_name}")
    
    # If raw data was updated (reprocessing), save it back to disk
    if args.reprocess:
        print("\nSaving updated user data with improved organization matching...")
        for repo_data in selected_repos:
            repo_name = repo_data['repo_name']
            users_data = repo_data['data']
            
            # Create filename (replace spaces with underscores)
            filename = f"github_raw_data/raw_user_data_{repo_name.replace(' ', '_')}.json"
            
            try:
                with open(filename, 'w') as f:
                    json.dump(users_data, f, indent=2)
                print(f"Saved updated data for {repo_name}")
            except Exception as e:
                print(f"Error saving updated data for {repo_name}: {str(e)}")
    
    # Generate and save combined results
    print("\nSaving combined results...")
    with open("user_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate summary report
    generate_summary_report(results)
    
    print("\nAnalysis complete. Results saved to user_analysis_results.json")
    
    if args.save_raw:
        print("Note: Using --save_raw will preserve the raw data used for debugging.")
    else:
        print("Note: For more detailed analysis, run with --save_raw to preserve raw user data.")
    
    if args.reprocess:
        print("User data was reprocessed with improved organization matching.")
        print("The new systematic approach uses configurable keyword patterns for more robust matching.")
    else:
        print("\nIMPORTANT: To update organization matching with the new systematic approach,")
        print("           run this script with the --reprocess flag:")
        print("           python github_user_analyzer.py --reprocess")
        print("\nThis will reanalyze user data with the configurable keyword-based matching system.")
        print("To customize matching patterns, add entries to the ORG_KEYWORDS dictionary.")
        print("This is more maintainable than one-off special cases.")

if __name__ == "__main__":
    main() 