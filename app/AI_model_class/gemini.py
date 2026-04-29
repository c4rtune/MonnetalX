import re
import requests
import os
from bs4 import BeautifulSoup
from openai import OpenAI
from functools import lru_cache

# -----------------------------------
# OPTIONAL GITHUB GRAPHQL SUPPORT
# -----------------------------------
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; PR-Link-Bot/1.0)"
}

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")


# =====================================================
# URL FETCH
# =====================================================
@lru_cache(maxsize=100)
def fetch_url(url: str):
    try:
        response = requests.get(url, headers=HEADERS, timeout=5)

        if response.status_code >= 400:
            return None

        return response

    except Exception:
        return None


def is_image(url: str) -> bool:
    url = url.lower().split("?")[0]
    return url.endswith(IMAGE_EXTENSIONS)


def is_text_response(response) -> bool:
    content_type = response.headers.get("Content-Type", "").lower()

    return (
        content_type.startswith("text/")
        or "json" in content_type
        or "xml" in content_type
    )


# =====================================================
# TEXT CLEANING
# =====================================================
def clean_pr_text(text: str) -> str:
    text = str(text or "")

    # remove HTML comments
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)

    # remove checklist section onward
    text = re.split(
        r"^\s*####\s*Checklist\b.*$",
        text,
        flags=re.MULTILINE
    )[0]

    return text


# =====================================================
# GITHUB GRAPHQL
# =====================================================
def github_graphql(query: str, variables: dict):
    if not GITHUB_TOKEN:
        return None

    try:
        r = requests.post(
            "https://api.github.com/graphql",
            json={
                "query": query,
                "variables": variables
            },
            headers={
                "Authorization": f"Bearer {GITHUB_TOKEN}",
                "Content-Type": "application/json"
            },
            timeout=8
        )

        if r.status_code >= 400:
            return None

        return r.json()

    except Exception:
        return None


def parse_github_pr_url(url: str):
    m = re.match(
        r"https://github\.com/([^/]+)/([^/]+)/pull/(\d+)",
        str(url)
    )

    if not m:
        return None

    owner, repo, num = m.groups()
    return owner, repo, int(num)


def extract_links_from_bodyhtml(html: str):
    if not html:
        return []

    try:
        soup = BeautifulSoup(html, "html.parser")
        results = []

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()

            if href.startswith("/"):
                href = "https://github.com" + href

            if href.startswith("http"):
                results.append(href)

        return results

    except Exception:
        return []


def fetch_github_pr_links(pr_url: str):
    """
    Uses GraphQL to fetch:
    - bodyHTML
    - closingIssuesReferences
    """

    parsed = parse_github_pr_url(pr_url)

    if not parsed:
        return []

    owner, repo, number = parsed

    query = """
    query($owner:String!, $repo:String!, $num:Int!) {
      repository(owner:$owner, name:$repo) {
        pullRequest(number:$num) {
          bodyHTML
          closingIssuesReferences(first:50) {
            nodes {
              url
            }
          }
        }
      }
    }
    """

    data = github_graphql(
        query,
        {
            "owner": owner,
            "repo": repo,
            "num": number
        }
    )

    if not data:
        return []

    try:
        pr = data["data"]["repository"]["pullRequest"]

        links = []

        # bodyHTML anchor refs
        links.extend(
            extract_links_from_bodyhtml(pr.get("bodyHTML", ""))
        )

        # closing refs
        for node in pr["closingIssuesReferences"]["nodes"]:
            url = node.get("url")
            if url:
                links.append(url)

        # dedupe
        seen = set()
        out = []

        for x in links:
            if x not in seen:
                seen.add(x)
                out.append(x)

        return out

    except Exception:
        return []


# =====================================================
# LINK EXTRACTION
# =====================================================
def extract_markdown_links(text, repo_name):
    """
    Extracts from PR text:
    - Markdown links [text](url)
    - Raw URLs
    - GitHub issue refs anywhere: #123, (#123), (#12, #34)
    - Keywords like Fixes/Closes/Resolves #123

    Excludes:
    - Anything inside HTML comments <!-- ... -->

    Returns:
        dict {url: display_text}
    """

    links = {}

    # ---------------------------------
    # 0. Remove HTML comments first
    # ---------------------------------
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

    # ---------------------------------
    # 1. Markdown links
    # ---------------------------------
    md_pattern = r'\[([^\]]+)\]\((https?://[^\s)]+(?:\)[^\s)]*)?)\)'

    for display_text, url in re.findall(md_pattern, text):
        clean_url = url.strip().rstrip('.,')
        display_text = display_text.strip().rstrip(").,;:")
        links[clean_url] = display_text

    # Remove markdown links to avoid duplicate raw URL matches
    text = re.sub(md_pattern, '', text)

    # ---------------------------------
    # 2. GitHub issue refs (#123)
    # Supports:
    #   #123
    #   (#123)
    #   (#123, #456)
    #   Fixes #123
    # ---------------------------------
    issue_pattern = r'#(\d+)'

    for issue_number in re.findall(issue_pattern, text):
        url = f"https://github.com/{repo_name}/issues/{issue_number}"
        links[url] = f"#{issue_number}"

    # Remove issue refs
    text = re.sub(issue_pattern, '', text)

    # ---------------------------------
    # 3. Raw URLs
    # ---------------------------------
    raw_pattern = r'https?://[^\s)>]+'

    for url in re.findall(raw_pattern, text):
        clean_url = url.strip().rstrip('.,')
        if clean_url not in links:
            links[clean_url] = clean_url

    return links

# -------------------------
# HTML PARSING
# =====================================================
def fetch_link_metadata(url: str):
    response = fetch_url(url)

    if not response:
        return "", "", ""

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(
        ["script", "style", "nav", "footer", "header"]
    ):
        tag.decompose()

    title = ""

    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        title = og_title["content"].strip()

    description = ""

    meta_desc = soup.find(
        "meta",
        attrs={"name": "description"}
    )

    if meta_desc and meta_desc.get("content"):
        description = meta_desc["content"].strip()

    og_desc = soup.find(
        "meta",
        property="og:description"
    )

    if og_desc and og_desc.get("content"):
        description = og_desc["content"].strip()

    paragraphs = soup.find_all("p")
    body = " ".join(
        p.get_text().strip()
        for p in paragraphs[:20]
    )

    return title, description, body


# =====================================================
# SCORING
# =====================================================
def score_link(link_title: str, pr_title: str, pr_body: str) -> float:
    text = (link_title or "").lower()
    pr_text = (pr_title + " " + pr_body).lower()

    score = 0

    for word in pr_text.split():
        if word in text:
            score += 1

    return min(score / 10, 1.0)


def rank_links(links, metadata, pr_title, pr_body):
    ranked = []

    for link in links:
        title, _, _ = metadata.get(link, ("", "", ""))
        score = score_link(
            title,
            pr_title,
            pr_body
        )

        ranked.append((link, score))

    return sorted(
        ranked,
        key=lambda x: x[1],
        reverse=True
    )


# =====================================================
# AI SUMMARY
# =====================================================
def summarize_link(
    url,
    pr_title,
    pr_body,
    repo_name,
    repo_description,
    link_title="",
    link_description="",
    link_body="",
):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful assistant tasked with writing a concise summary of content referenced in a GitHub pull request. The reference is to an external webpage, and your goal is to summarize the most relevant part of the content based on the pull request context, helping developers understand it without opening the link.
            You will always be provided with:
            - Pull request title
            - Pull request description
            - Repository title
            - Repository description
            - Link metadata (title, description, body)

            Your Task:
            Use the following reasoning steps before writing the final summary:

            1. Identify the main purpose of the pull request using its title and description.
            2. Examine the webpage content to find sections most relevant to that purpose.
            3. Factor in the surrounding PR discussion (comment context, where the link was placed).
            4. Determine the core insight or fact from the webpage that supports or explains the PR’s purpose.
            5. Write a 1-sentence summary of that insight. Ensure the summary:

            - Is standalone and factual
            - Is phrased as if it’s part of the pull request itself
            - Avoids references to external links or hosting

            Important: Output only the final 1-sentence summary, with no explanation, no headings, and no formatting.""",
                },
                {
                    "role": "user",
                    "content": f"""
PR title: {pr_title}
PR body: {pr_body}

Repo: {repo_name}
Repo desc: {repo_description}

Link title: {link_title}
Link desc: {link_description}
Link content: {link_body}
"""
                }
            ],
            temperature=0.3,
            max_tokens=60
        )

        return (
            response.choices[0]
            .message.content.strip()
            .replace("\n", " ")
        )

    except Exception:
        return (
            link_description
            or link_title
            or "No summary available"
        )


# =====================================================
# UI HELPERS
# =====================================================
def generate_percentage_bar(score: float, length: int = 12):
    percentage = int(score * 100)
    filled = int(length * score)

    if percentage < 30:
        block = "🟥"
    elif percentage < 60:
        block = "🟨"
    else:
        block = "🟩"

    return (
        f"[{block * filled}"
        f"{'⬜' * (length - filled)}] "
        f"{percentage}%"
    )


def get_color_indicator(score: float):
    if score >= 0.6:
        return "🟢"
    elif score >= 0.3:
        return "🟡"
    return "🔴"