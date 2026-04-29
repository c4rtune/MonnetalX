import os
import re
from functools import lru_cache
from urllib.request import urlopen

import lightgbm as lgb
import numpy as np
from github import Github, GithubException, Auth
from sentence_transformers import SentenceTransformer


class MonnetalRanker:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "model.txt")

    GITHUB_LINK_PATTERN = re.compile(
        r"https://github\.com/(?P<repo>[^/]+/[^/]+)/(pull|issues)/(?P<num>\d+)"
    )

    COMMENT_PATTERN = re.compile(
        r"https://github\.com/(?P<repo>[^/]+/[^/]+)/(pull|issues)/(?P<num>\d+)"
        r"#(?P<type>pullrequestreview|issuecomment)-(?P<comment_id>\d+)"
    )

    COMMIT_PATTERN = re.compile(
        r"https://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/commit/(?P<sha>[a-f0-9]{7,40})"
    )

    def __init__(
        self,
        model_path: str,
        gh_token: str,
        vectorizer=None,
        repo_name=None,
    ):
        """
        Args:
            model_path: path to LightGBM model
            gh_token: GitHub token
            vectorizer: optional preloaded embedding model
            repo_name: source repo in owner/repo format
        """
        self.repo = repo_name

        if not os.path.isabs(model_path):
            model_path = os.path.join(self.BASE_DIR, model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.ranker = lgb.Booster(model_file=model_path)

        auth = Auth.Token(gh_token)
        self.g = Github(auth=auth)

        self.vectorizer = (
            vectorizer
            if vectorizer is not None
            else SentenceTransformer("all-MiniLM-L6-v2")
        )

        print("MonnetalRanker initialized")

    # ==========================================================
    # TEXT HELPERS
    # ==========================================================

    def _clean_html(self, text: str) -> str:
        text = re.sub(r"<script.*?>.*?</script>", " ", text, flags=re.DOTALL)
        text = re.sub(r"<style.*?>.*?</style>", " ", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _tokenize(self, text: str):
        return re.findall(r"\w+", str(text).lower())

    def get_ngrams(self, text: str, n: int):
        tokens = self._tokenize(text)
        return [
            " ".join(tokens[i : i + n])
            for i in range(len(tokens) - n + 1)
        ]

    def ngram_overlap(self, q: str, d: str, n: int = 2) -> float:
        q_ngrams = set(self.get_ngrams(q, n))
        d_ngrams = set(self.get_ngrams(d, n))

        if not q_ngrams:
            return 0.0

        return float(len(q_ngrams & d_ngrams)) / float(len(q_ngrams))

    # ==========================================================
    # GITHUB HELPERS
    # ==========================================================

    def getNum(self, url: str):
        match = self.GITHUB_LINK_PATTERN.search(str(url))
        if not match:
            return None, None

        try:
            return match.group("repo"), int(match.group("num"))
        except Exception:
            return None, None

    @lru_cache(maxsize=256)
    def _get_repo(self, repo_name: str):
        return self.g.get_repo(repo_name)

    def _get_author_login(self, repo_name: str, num: int) -> str:
        try:
            repo = self._get_repo(repo_name)

            try:
                pr = repo.get_pull(num)
                return getattr(pr.user, "login", "")
            except GithubException:
                issue = repo.get_issue(num)
                return getattr(issue.user, "login", "")

        except Exception:
            return ""

    def _get_assignees(self, repo_name: str, num: int):
        try:
            repo = self._get_repo(repo_name)

            try:
                pr = repo.get_pull(num)
                return [u.login for u in pr.assignees if hasattr(u, "login")]
            except GithubException:
                issue = repo.get_issue(num)
                return [u.login for u in issue.assignees if hasattr(u, "login")]

        except Exception:
            return []

    # ==========================================================
    # SCRAPER
    # ==========================================================

    @lru_cache(maxsize=512)
    def scrap(self, url: str) -> str:
        if not url:
            return ""

        try:
            comment_match = self.COMMENT_PATTERN.match(url)

            if comment_match:
                repo_name = comment_match.group("repo")
                num = int(comment_match.group("num"))
                comment_id = int(comment_match.group("comment_id"))
                comment_type = comment_match.group("type")

                repo = self._get_repo(repo_name)

                if comment_type == "issuecomment":
                    issue = repo.get_issue(num)
                    return issue.get_comment(comment_id).body or ""

                if comment_type == "pullrequestreview":
                    try:
                        pr = repo.get_pull(num)
                        return pr.get_review_comment(comment_id).body or ""
                    except GithubException:
                        return repo.get_pull(num).get_review(comment_id).body or ""

            gh_match = self.GITHUB_LINK_PATTERN.match(url)

            if gh_match:
                repo_name = gh_match.group("repo")
                num = int(gh_match.group("num"))

                repo = self._get_repo(repo_name)

                try:
                    return repo.get_pull(num).body or ""
                except GithubException:
                    return repo.get_issue(num).body or ""

            with urlopen(url) as response:
                html = response.read().decode("utf-8", errors="ignore")
                return self._clean_html(html)

        except Exception:
            return ""

    # ==========================================================
    # FEATURES
    # ==========================================================

    def extract_ranking_features(self, content: str, link: str) -> np.ndarray:
        """
        Returns:
        [
            cosine,
            same_repo,
            user_match,
            code_change,
            qlen,
            unigram,
            bigram,
            trigram
        ]
        """

        cosine = 0.0
        same_repo = 0
        user_match = 0
        code_change = 0
        qlen = 0
        unigram = 0.0
        bigram = 0.0
        trigram = 0.0

        query_content = self.scrap(link)

        # -------------------------
        # Semantic similarity
        # -------------------------
        try:
            doc_v = self.vectorizer.encode(content)
            query_v = self.vectorizer.encode(query_content)

            sim = self.vectorizer.similarity(doc_v, query_v)
            cosine = float(sim[0][0])

        except Exception:
            cosine = 0.0

        # -------------------------
        # Lexical overlap
        # -------------------------
        try:
            unigram = self.ngram_overlap(query_content, content, 1)
            bigram = self.ngram_overlap(query_content, content, 2)
            trigram = self.ngram_overlap(query_content, content, 3)
        except Exception:
            pass

        # -------------------------
        # Query length
        # -------------------------
        qlen = len(self._tokenize(query_content))

        # -------------------------
        # Same repo
        # -------------------------
        if self.repo and self.repo in str(link):
            same_repo = 1

        # -------------------------
        # User match
        # -------------------------
        repo_name, num = self.getNum(link)

        if repo_name and num:
            author = self._get_author_login(repo_name, num)
            assignees = self._get_assignees(repo_name, num)

            if author and author in assignees:
                user_match = 1

        # -------------------------
        # Code change feature
        # -------------------------
        commit_match = self.COMMIT_PATTERN.match(str(link))

        if commit_match and self.repo:
            try:
                owner = commit_match.group("owner")
                repo = commit_match.group("repo")
                sha = commit_match.group("sha")

                commit_repo = f"{owner}/{repo}"

                if commit_repo.lower() == self.repo.lower():
                    repo_obj = self._get_repo(commit_repo)
                    repo_obj.get_commit(sha)
                    code_change = 1

            except Exception:
                pass

        return np.array(
            [
                float(cosine),
                int(same_repo),
                int(user_match),
                int(code_change),
                int(qlen),
                float(unigram),
                float(bigram),
                float(trigram),
            ]
        )

    # ==========================================================
    # PREDICT
    # ==========================================================

    def predict(self, input_data):
        """
        input_data format:
        {
            "content": "...",
            "links": ["...", "..."]
        }
        """

        features_list = []

        for link in input_data["links"]:
            feats = self.extract_ranking_features(
                input_data["content"],
                link,
            )
            features_list.append(feats)

        X_predict = np.vstack(features_list)

        return self.ranker.predict(X_predict)