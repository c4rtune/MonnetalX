from dotenv import load_dotenv
load_dotenv()

import os
from concurrent.futures import ThreadPoolExecutor

from .helpers import fetch_pr_data, post_comment, sanitize
from app.AI_model_class.gemini import (
    extract_markdown_links,
    fetch_link_metadata,
    summarize_link,
)
from app.AI_model_class.monnetalRanker import MonnetalRanker

def rank_links_with_model(links, pr_body, repo_name, github_token):
    if not links:
        return []

    ranker = MonnetalRanker(
        model_path="model.txt",  # 🔥 update path
        gh_token=github_token,
        vectorizer=None,
        repo_name=repo_name
    )

    predictions = ranker.predict({
        "content": pr_body,
        "links": links
    })

    # Pair links with scores
    ranked = list(zip(links, predictions))

    # Sort by score DESC
    ranked.sort(key=lambda x: x[1], reverse=True)

    # Convert to rank order (1,2,3...)
    ranked_with_position = [
        (link, idx + 1)
        for idx, (link, _) in enumerate(ranked)
    ]

    return ranked_with_position

# ENV
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
PR_NUMBER = int(os.getenv("PR_NUMBER"))
OWNER = os.getenv("REPO_OWNER")
REPO = os.getenv("REPO_NAME")

# FETCH PR
pr_data = fetch_pr_data(GITHUB_TOKEN, OWNER, REPO, PR_NUMBER)
pr_body = pr_data.get("body", "")

# 🚫 1. No PR description
if not pr_body.strip():
    print("❌ No PR description. Skipping bot.")
    exit(0)

# 🚫 2. No links in description
import re
if not re.search(r'https?://|\[[^\]]+\]\((https?://[^\)]+)\)|#\d+', pr_body):
    print("❌ No links found in PR description. Skipping bot.")
    exit(0)
    
# EXTRACT LINKS
link_map = extract_markdown_links(pr_body, REPO)
links = list(link_map)

if not links:
    message = "## 🔍 PR Link Analysis\n\n_No valid links found_"
    post_comment(GITHUB_TOKEN, OWNER, REPO, PR_NUMBER, message)
    exit()

# PARALLEL METADATA FETCH
metadata = {}
with ThreadPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(fetch_link_metadata, links))

for link, data in zip(links, results):
    metadata[link] = data

# RANK LINKS
ranked_links = ranked_links = rank_links_with_model(
    links,
    pr_body,
    REPO,
    GITHUB_TOKEN
)

# BUILD OUTPUT
lines = []

for link, rank in ranked_links:
    title, desc, body = metadata.get(link, ("", "", ""))

    # ✅ Use original markdown text if available
    display_text = link_map.get(link, link).replace("[", "").replace("]", "")
    markdown_link = f"[{display_text}]({link})"

    summary = summarize_link(
        link,
        pr_data["title"],
        pr_body,
        REPO,
        pr_data.get("repo_description", ""),
        title,
        desc,
        body,
    )

    lines.append(
        f"### #{rank} {markdown_link}\n"
        f"🔍 {summary}\n"
    )

message = f"""
## 🔍 PR Link Analysis

### 🔗 Ranked Links
{chr(10).join(lines)}
"""

post_comment(GITHUB_TOKEN, OWNER, REPO, PR_NUMBER, message)

print("🚀 Done")