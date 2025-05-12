# Energy Model Analysis Dashboard

A Streamlit dashboard for analyzing energy modeling tools from the G-PST Open Tools Portal, focusing on GitHub metrics to evaluate usage, maturity, and user-friendliness.

## Features

- Fetches GitHub metrics for energy models
- Calculates scores for:
  - Popularity (stars, forks)
  - Maturity (age, maintenance, releases)
  - Developer Activity (commit frequency)
  - Developer Community (contributors)
  - User-friendliness (documentation, issues)
- Creates visualizations for comparing models
- Filters analysis by model category
- Ranks models with a combined index
- Analyzes growth trends with time series data:
  - Star growth over time and growth rates
  - Commit activity patterns
  - Code changes (additions/deletions)
  - Issue activity and resolution metrics
- Provides user analysis by:
  - Organization type (academic, industry, utility, etc.)
  - Geographic distribution
  - Engagement patterns across repositories

## Requirements

- Python 3.8+
- GitHub API token for accessing repo metrics

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - On macOS/Linux: `source venv/bin/activate`
   - On Windows: `venv\Scripts\activate`
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Data Pipeline Architecture

The application is organized into a modular data pipeline with multiple scripts that each handle a specific part of the data collection, processing, and visualization:

```
Data Sources → Collection → Processing → Analysis → Visualization
 (GitHub API)     ↓            ↓           ↓          ↓
                Fetchers → Processors → Analyzers → Dashboard
```

### Scripts Overview

1. **github_data_fetcher.py** - Collects repository metrics and time series data from GitHub
2. **github_data_analyzer.py** - Processes repository data and calculates metrics and scores
3. **github_user_fetcher.py** - Collects user data (stars, forks, watchers, contributors)
4. **github_user_analyzer.py** - Analyzes and classifies users by organization type and location
5. **model_dashboard.py** - Streamlit dashboard that visualizes all processed data

### Data Flow & Processing Order

```
      ┌───────────────────┐        ┌───────────────────┐
      │github_data_fetcher│  →→→   │github_data_analyzer│
      └───────────────────┘        └───────────────────┘
               ↓                             ↓
         github_data.json             analyzed metrics
               ↓                             ↓
      ┌───────────────────┐        ┌───────────────────┐
      │github_user_fetcher│  →→→   │github_user_analyzer│
      └───────────────────┘        └───────────────────┘
               ↓                             ↓
        user_raw_data files        user_analysis_results.json
               ↓                             ↓
                      ┌───────────────┐
                      │model_dashboard│
                      └───────────────┘
```

## Script Details

### 1. GitHub Repository Data Collection

#### github_data_fetcher.py
```
python github_data_fetcher.py
```

- **Purpose**: Fetches GitHub metrics for repositories and collects time series data
- **Inputs**: `gpst_open_tools.csv` (list of models and their repositories)
- **Outputs**: `github_data.json` (repository metrics and time series data)
- **Details**:
  - Collects stars, forks, contributors, issues, releases, etc.
  - Gathers time series data for stars, commits, code frequency, issues
  - Requires GitHub API token
  - Run this first before any other scripts

#### github_data_analyzer.py
```
python github_data_analyzer.py
```

- **Purpose**: Processes GitHub data to calculate scores and metrics
- **Inputs**: `github_data.json` from github_data_fetcher.py
- **Outputs**: Updates `github_data.json` with calculated scores
- **Details**:
  - Calculates popularity, maturity, activity, community scores
  - Detects external documentation
  - Normalizes metrics for comparison
  - Run this after github_data_fetcher.py

### 2. GitHub User Data Collection

#### github_user_fetcher.py
```
python github_user_fetcher.py
```

- **Purpose**: Collects detailed data about users who interact with repositories
- **Inputs**: `github_data.json` from github_data_fetcher.py
- **Outputs**: Raw user data files in `github_raw_data/` directory
- **Details**:
  - Collects stargazers, forkers, contributors, watchers
  - Saves user profile data including company, location, email
  - Requires GitHub API token
  - Can be time-intensive due to API rate limits

#### github_user_analyzer.py
```
python github_user_analyzer.py [--reprocess] [--save_raw]
```

- **Purpose**: Analyzes and classifies users by organization type and location
- **Inputs**: Raw user data from github_user_fetcher.py
- **Outputs**: 
  - `user_analysis_results.json` (processed user classifications and statistics)
  - `user_analysis_report.txt` (human-readable summary report)
- **Details**:
  - Classifies users as academic, industry, utility, etc.
  - Geocodes user locations to countries
  - Standardizes organization names
  - Generates statistics on user types by repository
  - Optional flags:
    - `--reprocess`: Reapply improved organization matching logic to existing data
    - `--save_raw`: Preserve raw data for debugging

### 3. Dashboard Visualization

#### model_dashboard.py
```
streamlit run model_dashboard.py
```

- **Purpose**: Interactive dashboard for visualizing all collected and processed data
- **Inputs**:
  - `github_data.json` (repository metrics and scores)
  - `user_analysis_results.json` (user analysis results)
  - `gpst_open_tools.csv` (base model information)
- **Details**:
  - Provides interactive visualizations of all metrics
  - Allows filtering by model category
  - Presents time series data for tracking growth
  - Displays user analysis by organization type and location
  - Offers customizable scoring weights
  - Supports model comparison

## Complete Workflow

For a full analysis from scratch, follow these steps in order:

1. **Initial Setup**:
   ```
   # Install dependencies
   pip install -r requirements.txt
   
   # Set up GitHub token (create github_api_key file or use .env)
   echo "your_github_token" > github_api_key
   ```

2. **Repository Data Collection**:
   ```
   # Fetch GitHub repository metrics
   python github_data_fetcher.py
   
   # Process and calculate scores
   python github_data_analyzer.py
   ```

3. **User Data Collection**:
   ```
   # Fetch user data
   python github_user_fetcher.py
   
   # Analyze and classify users
   python github_user_analyzer.py
   ```

4. **Launch Dashboard**:
   ```
   # Run the Streamlit dashboard
   streamlit run model_dashboard.py
   ```

## Refreshing Data

To update the data:

- For repository metrics: `python github_data_fetcher.py`
- For user data: `python github_user_fetcher.py` followed by `python github_user_analyzer.py`

The dashboard will automatically use the updated data next time it's loaded.

## Scoring Methodology

The overall score is calculated as a weighted combination of:
- 25% Popularity (stars, forks)
- 25% Maturity (age, maintenance, releases)
- 20% Developer Activity (commit frequency)
- 15% Developer Community (contributors)
- 15% User-friendliness (documentation, issues)

You can adjust these weights in the dashboard sidebar.

## Deployment

To deploy on Streamlit Cloud:
1. Push this repository to GitHub
2. Connect to Streamlit Cloud
3. Add your GitHub token as a secret in the Streamlit dashboard 
4. Make sure to run the data fetcher scripts first and commit the generated data files 