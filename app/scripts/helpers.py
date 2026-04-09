import requests

GRAPHQL_URL = "https://api.github.com/graphql"


def fetch_pr_data(token, owner, repo, pr_number):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    query = """
    query($owner: String!, $repo: String!, $prNumber: Int!) {
      repository(owner: $owner, name: $repo) {
        description
        pullRequest(number: $prNumber) {
          title
          body
          author {
            login
          }
        }
      }
    }
    """

    variables = {
        "owner": owner,
        "repo": repo,
        "prNumber": pr_number,
    }

    response = requests.post(
        GRAPHQL_URL,
        json={"query": query, "variables": variables},
        headers=headers,
        timeout=10,
    )

    if response.status_code != 200:
        raise Exception(f"GitHub API error: {response.status_code} - {response.text}")

    data = response.json()

    if "errors" in data:
        messages = [e.get("message", "Unknown error") for e in data["errors"]]
        raise Exception("GraphQL errors: " + " | ".join(messages))

    repo_data = data["data"]["repository"]
    pr = repo_data["pullRequest"]

    return {
        "title": pr["title"],
        "body": pr["body"] or "_No description provided_",
        "author": pr["author"]["login"],
        "repo_description": repo_data.get("description", ""),
    }

def sanitize(text: str) -> str:
    return text.replace("[", "").replace("]", "").replace("(", "").replace(")", "")


def post_comment(token, owner, repo, pr_number, message):
    url = f"https://api.github.com/repos/{owner}/{repo}/issues/{pr_number}/comments"

    headers = {
        "Authorization": f"token {token}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        url,
        json={"body": message},
        headers=headers,
        timeout=10,
    )

    if response.status_code >= 400:
        raise Exception(f"Failed to post comment: {response.text}")

    print("✅ Comment posted successfully")