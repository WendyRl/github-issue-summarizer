# 🤖 GitHub Issue Summarizer AI Agent

This AI-powered agent fetches open issues from any public GitHub repository and summarizes each issue using a locally-run NLP model (`facebook/bart-large-cnn`) from Hugging Face. It provides concise summaries directly in your terminal and saves them as a Markdown report.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Create `.env` for GitHub Token

```
GITHUB_TOKEN=your_github_token_here
```

If you don’t include a GitHub token, unauthenticated API calls are limited to 60/hour.

### 3. Run the Script

```bash
python main.py owner/repo
```

Example:

```bash
python main.py openai/openai-python
```

## 📦 Output

- Summaries printed to the terminal
- Markdown file saved as `owner_repo_summaries.md` (e.g., `openai_openai-python_summaries.md`)

## 🧠 Design Choices

### Why GitHub + Summarization?

GitHub issues often contain long technical discussions. Summarizing them helps developers quickly triage and prioritize work — especially useful for maintainers or contributors reviewing many issues.

### Why Hugging Face Instead of OpenAI?

While OpenAI’s GPT-3.5 API is powerful, it has usage limits and requires billing. To avoid quota errors and improve portability, we use a local summarization model from Hugging Face (`facebook/bart-large-cnn`) which is:

- Free to use
- Runs offline after the initial download
- Widely accepted for summarization tasks

### Why `facebook/bart-large-cnn`?

- Pre-trained on CNN/DailyMail summarization corpus
- Balanced trade-off between quality and speed
- Well-suited for issue-style text (semi-formal, technical)

### Why Markdown Output?

- Easy to share, render in GitHub, or paste into documentation
- Supports integration into automated workflows

## 🧰 Tech Stack

| Component           | Tool/Library                        |
|--------------------|-------------------------------------|
| Language            | Python 3                            |
| GitHub API Access   | `requests`, optional `GITHUB_TOKEN` |
| NLP Model           | `facebook/bart-large-cnn` (Hugging Face) |
| Hugging Face Runtime | `transformers`, `torch`            |
| Env Configuration   | `python-dotenv`                     |

## 🧪 Example Output (Terminal)

```
📌 Issue 1: Unexpected behavior in openai.ChatCompletion.create()
🔗 https://github.com/openai/openai-python/issues/123
📝 Summary: Passing a custom user ID completes successfully but doesn’t appear in logs or metrics as documented.
```

## 💡 What I’d Improve with More Time

- Web-based UI (e.g., Flask or Streamlit interface)
- Interactive CLI (prompt for repo, filters, or labels)
- Batch mode to auto-run daily summaries on selected repos
- Keyword/tag extraction for automated triage
- Post summary as GitHub comment via GitHub API
- Multiple model options (`t5-base`, `pegasus`, `mistral`)
- Model caching and progress bars for better UX

## ✅ Status

| Feature                     | Done |
|----------------------------|------|
| GitHub issue fetching      | ✅   |
| Local summarization (free) | ✅   |
| Markdown export            | ✅   |
| CLI repo selection         | ✅   |
| OpenAI-free runtime        | ✅   |

## 📎 License

MIT License. Use freely, attribution appreciated.
