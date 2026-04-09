import os

import lightgbm as lgb
import re
from github import Github, GithubException, Auth
from urllib.request import urlopen
from sentence_transformers import SentenceTransformer
       
import numpy as np
import pandas as pd


class MonnetalRanker:
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "model.txt")
    
    GITHUB_LINK_PATTERN = re.compile(r"https://github\.com/(?P<repo>[^/]+/[^/]+)/(pull|issues)/(?P<pr>\d+)")
    COMMENT_PATTERN = re.compile(
        r"https://github\.com/(?P<repo>[^/]+/[^/]+)/(pull|issues)/(?P<pr>\d+)#(?P<type>pullrequestreview|issuecomment)-(?P<comment_id>\d+)"
    )
    def __init__(self, model_path: str, gh_token: str, vectorizer=None, repo_name=None):
        """
        Initializes the MonnetalRanker.

        Args:
            model_path (str): Path to the LightGBM model file (relative or absolute).
            gh_token (str): GitHub Personal Access Token.
            vectorizer: Optional preloaded embedding model.
            repo_name (str): Repo in format 'owner/repo'.
        """
        import os
        import lightgbm as lgb
        from github import Github, Auth
        from sentence_transformers import SentenceTransformer

        # ✅ Store repo name
        self.repo = repo_name  # expected: "owner/repo"

        # ✅ Resolve model path RELATIVE to this file (fixes GitHub Actions issue)
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        if not os.path.isabs(model_path):
            model_path = os.path.join(BASE_DIR, model_path)

        print(f"📦 Loading model from: {model_path}")  # helpful debug

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model not found at: {model_path}")

        # ✅ Load LightGBM model
        self.ranker = lgb.Booster(model_file=model_path)

        # ✅ GitHub client
        auth = Auth.Token(gh_token)
        self.g = Github(auth=auth)

        # ✅ Use passed vectorizer OR create one (fix bug)
        if vectorizer is not None:
            self.vectorizer = vectorizer
        else:
            self.vectorizer = SentenceTransformer('all-MiniLM-L6-v2')

        print("✅ MonnetalRanker initialized successfully")

    def _clean_html(self, text: str) -> str:
        text = re.sub(r"<script.*?>.*?</script>", " ", text, flags=re.DOTALL)
        text = re.sub(r"<style.*?>.*?</style>", " ", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def getNum(self, url: str):
        match = self.GITHUB_LINK_PATTERN.search(url)
        if match:
            try:
                return match.group("repo"), int(match.group("pr"))
            except ValueError:
                return None, None
        return None, None

    def scrap(self, url: str) -> str:
        
        if not url:
            return "404 no body found"

        try:
          
            comment_match = self.COMMENT_PATTERN.match(url)
            if comment_match:
                repo_name = comment_match.group("repo")
                num = int(comment_match.group("pr"))
                comment_id = int(comment_match.group("comment_id"))
                comment_type = comment_match.group("type")
                repo = self.g.get_repo(repo_name)

                try:
                    if comment_type == "issuecomment":
                        issue = repo.get_issue(num)
                        return issue.get_comment(comment_id).body
                    elif comment_type == "pullrequestreview":
                        # For PR review comments (line comments on code)
                        try:
                            pr = repo.get_pull(num)
                            return pr.get_review_comment(comment_id).body
                        except GithubException:
                            # If it's not a review comment, it might be the body of a review itself
                            # PyGithub's get_review() takes a review ID for its body
                            return repo.get_pull(num).get_review(comment_id).body
                    else:
                        raise ValueError("Unknown comment type in URL")
                except GithubException:
                    # Fallback if specific comment/review type fails, try general issue comment
                    try:
                        issue = repo.get_issue(num)
                        return issue.get_comment(comment_id).body
                    except GithubException:
                        pass # Will fall to general error at the end of the try block

            github_link_match = self.GITHUB_LINK_PATTERN.match(url)
            if github_link_match:
                repo_name = github_link_match.group("repo")
                num = int(github_link_match.group("pr"))
                repo = self.g.get_repo(repo_name)
                try:
                    pr = repo.get_pull(num)
                    return pr.body
                except GithubException: 
                    issue = repo.get_issue(num)
                    return issue.body

            with urlopen(url) as response:
                return self._clean_html(response.read().decode('utf-8'))

        except Exception:
  
            return "404 no body found"

    def extract_ranking_features(self,content:str,link:str) -> np.ndarray:
        """
        Extracts a feature vector from a single data row (dictionary)
        for use with the ranking model.

        Args:
            content(str): content of the PR
            link(str): link contained in the PR
        Return:
            arrays of features [ cosine , same repo , user match , code change ]
              ( to be feeded into model )
        """
        # ---- SAFE DEFAULTS ----
        cosine = 0.0
        same_repo_flag = 0
        user_match_flag = 0
        code_change_flag = 0

        # ---- SCRAPE CONTENT ----
        repo = self.repo
        query_content = self.scrap(link)

        # ---- COSINE SIMILARITY FEATURE ----
        try:
            doc_v = self.vectorizer.encode(content)
            query_v = self.vectorizer.encode(query_content)
            sim = self.vectorizer.similarity(doc_v, query_v)
            cosine = float(sim[0][0])
        except Exception:
            cosine = 0.0

        # ---- SAME REPO CHECK FEATURE ----
        try:
            # Check if the 'repo' field (e.g., "owner/repo") is present in the 'link' URL
            same_repo_flag = 1 if str(repo) in str(link) else 0
        except Exception:
            same_repo_flag = 0

        # ---- USER MATCH FEATURE ----
        # Extract user of the PR/Issue from 'pr_link'
        pr_user_login = ""
        repo_name_pr, num_pr = self.getNum(link)
        if repo_name_pr and num_pr:
            try:
                repo_obj = self.g.get_repo(repo_name_pr)
                try:
                    pr = repo_obj.get_pull(num_pr)
                    pr_user_login = getattr(pr.user, "login", "")
                except GithubException: # Not a PR, try as an issue
                    issue = repo_obj.get_issue(num_pr)
                    pr_user_login = getattr(issue.user, "login", "")
            except GithubException:
                pass # pr_user_login remains empty

        # Extract assignees from the 'link' URL (which could be another PR/Issue)
        link_assignees_logins = []
        repo_name_link, num_link = self.getNum(link)
        if repo_name_link and num_link:
            try:
                repo_obj_link = self.g.get_repo(repo_name_link)
                try:
                    pr_link = repo_obj_link.get_pull(num_link)
                    link_assignees_logins = [u.login for u in pr_link.assignees if hasattr(u, "login")]
                except GithubException: # Not a PR, try as an issue
                    issue_link = repo_obj_link.get_issue(num_link)
                    link_assignees_logins = [u.login for u in issue_link.assignees if hasattr(u, "login")]
            except GithubException:
                pass # link_assignees_logins remains empty
        
        try:
            user_match_flag = 1 if pr_user_login and pr_user_login in link_assignees_logins else 0
        except Exception:
            user_match_flag = 0

        # ---- CODE CHANGE FEATURE ----
        # Check if the 'link' URL points to a commit within the *same* repository
        commit_re = re.compile(r'https://github\.com/([^/]+)/([^/]+)/commit/([a-f0-9]{7,40})')
        commit_match = commit_re.match(link)

        if commit_match and repo_name_pr: # Only proceed if link is a commit and we identified the PR's repo
            try:
                owner_commit, repo_name_commit, sha = commit_match.groups()
                # Verify commit existence and if it belongs to the *same* repo as the 'pr_link'
                if repo_name_pr.lower() == f"{owner_commit}/{repo_name_commit}".lower():
                    repo_obj_commit = self.g.get_repo(f"{owner_commit}/{repo_name_commit}")
                    # Just checking existence; will throw GithubException if commit doesn't exist
                    repo_obj_commit.get_commit(sha)
                    code_change_flag = 1
            except Exception:
                pass # code_change_flag remains 0

        return np.array([
            float(cosine),
            int(same_repo_flag),
            int(user_match_flag),
            int(code_change_flag)
        ])

    def predict(self,input ) -> np.ndarray:
        '''
        Args: input expect to receive the format of
        {
            "content": "",
            "links": ["","",""......]
        }
        '''

        features_list = []
        for link in input["links"]:
            features = self.extract_ranking_features(input["content"],link)
            features_list.append(features)
        X_predict = np.vstack(features_list)
        
        return self.ranker.predict(X_predict)
