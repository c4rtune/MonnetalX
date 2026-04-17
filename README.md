# MonnetalX

MonnetalX is an AI-powered GitHub Action that analyzes links found in Pull Request (PR) descriptions, ranks them by relevance, summarizes their content, and automatically posts the results as a comment on the PR.

It helps reviewers quickly understand referenced issues, documentation, RFCs, related PRs, and external resources without manually opening every link.

---

## 🚀 Features

- 🔍 Detects links inside PR descriptions
- 🧠 Supports:
  - Markdown links (`[text](url)`)
  - Raw URLs (`https://...`)
  - GitHub issue references (`#123`)
  - PR references (`owner/repo#123`)
- 📊 Ranks links using a LightGBM LambdaRank model
- 🤖 Generates contextual summaries using LLMs
- 💬 Posts ranked results directly into PR comments
- ⚡ Runs automatically through GitHub Actions
- 🛑 Skips execution when no PR description or no links are found

---

## 📌 Example Output

```text
## 🔍 PR Link Analysis

### 🔗 Ranked Links

#1 Fixes #19611
🔍 This linked issue tracks the Android JavaScript escaping bug resolved by this PR.

#2 RFC Proposal
🔍 The RFC originally proposed the `provide_context` method name for generic error member access.

#3 Expo Snack Example
🔍 Demonstrates the expected WebView output after applying the fix.




To use the tools, create a yaml files as follow

name: MonnetalX

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  run-bot:
    runs-on: ubuntu-latest

    permissions:
      pull-requests: write
      issues: write
      contents: read

    steps:
      - uses: your-username/MonnetalX@v1
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          deepseek_api_key: ${{ secrets.DEEPSEEK_API_KEY }}
