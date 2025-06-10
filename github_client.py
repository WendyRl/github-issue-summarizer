import os
import requests
from dotenv import load_dotenv

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

def get_open_issues(repo):
    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

    response = requests.get(url, headers=headers, params={"state": "open"})
    response.raise_for_status()

    return [
        {
            "title": issue["title"],
            "body": issue.get("body", ""),
            "url": issue["html_url"]
        }
        for issue in response.json()
        if "pull_request" not in issue  # Exclude PR
    ]
