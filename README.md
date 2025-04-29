# Energy Model Analysis Dashboard

A Streamlit dashboard for analyzing energy modeling tools from the G-PST Open Tools Portal, focusing on GitHub metrics to evaluate usage, maturity, and user-friendliness.

## Features

- Fetches GitHub metrics for energy models
- Calculates scores for:
  - Popularity (stars, forks)
  - Maturity (age, maintenance, releases)
  - Activity (commit frequency)
  - Community engagement (contributors)
  - User-friendliness (documentation, issues)
- Creates visualizations for comparing models
- Filters analysis by model category
- Ranks models with a combined index
- Analyzes growth trends with time series data:
  - Star growth over time and growth rates
  - Commit activity patterns
  - Code changes (additions/deletions)
  - Issue activity and resolution metrics

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
   pip install streamlit pandas numpy matplotlib seaborn plotly PyGithub python-dotenv requests
   ```
   or create a requirements.txt file with these packages and run:
   ```
   pip install -r requirements.txt
   ```

## Architecture

The application is split into two separate scripts:

1. **github_data_fetcher.py** - Fetches GitHub metrics and calculates scores, saving the data to a JSON file
2. **model_dashboard.py** - Streamlit dashboard that reads the pre-fetched data and creates visualizations

This separation allows you to:
- Run the data fetching once and reuse the data in multiple dashboard sessions
- Reduce API calls to GitHub (avoiding rate limits)
- Update the data on your own schedule

## Usage

### Step 1: Set up GitHub API Token

Get a GitHub personal access token:
- Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
- Generate a new token with `repo` scope
- Save the token in one of these ways:
  1. Create a `.env` file with `GITHUB_TOKEN=your_token_here`
  2. Save your token in a file named `github_api_key`
  3. Set an environment variable named `GITHUB_TOKEN`

### Step 2: Fetch GitHub Data

Run the data fetcher:
```
python github_data_fetcher.py
```

This will:
- Load data from the `gpst_open_tools.csv` file
- Fetch GitHub metrics for each repository
- Collect time series data (stars, commits, code changes, issues)
- Calculate scores 
- Save the results to `github_data.json`

### Step 3: Launch the Dashboard

Run the Streamlit dashboard:
```
streamlit run model_dashboard.py
```

## Time Series Analysis

The dashboard includes a "Growth Analysis" section that allows you to:

1. **Analyze Star Growth**
   - View cumulative star growth over time
   - Compare growth rates (stars per day/month/year)
   - Identify trending models based on recent growth

2. **Track Commit Activity**
   - View commit patterns over time
   - Compare development velocity across models
   - Identify consistently maintained projects

3. **Monitor Code Changes**
   - Track code additions and deletions
   - Identify major refactoring or development periods
   - Compare development intensity

4. **Analyze Issue Activity**
   - Track issues opened and closed over time
   - Calculate issue resolution times and rates
   - Evaluate responsiveness to user feedback

## Refreshing Data

To update the GitHub metrics, simply run the data fetcher again:
```
python github_data_fetcher.py
```

The dashboard will automatically use the updated data next time it's loaded.

## Scoring Methodology

The overall score is calculated as a weighted combination of:
- 25% Popularity (stars, forks)
- 25% Maturity (age, maintenance, releases)
- 20% Activity (commit frequency)
- 15% Community (contributors)
- 15% User-friendliness (documentation, issues)

You can adjust these weights in the dashboard sidebar.

## Deployment

To deploy on Streamlit Cloud:
1. Push this repository to GitHub
2. Connect to Streamlit Cloud
3. Add your GitHub token as a secret in the Streamlit dashboard
4. Make sure to run the data fetcher script first and commit the generated JSON file 