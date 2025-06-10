import sys
from github_client import get_open_issues
from summarizer import summarize_issue

def main():
    repo = sys.argv[1] if len(sys.argv) > 1 else "openai/openai-python"
    output_file = f"{repo.replace('/', '_')}_summaries.md"

    print(f"\n🔍 Fetching issues from: {repo}\n")
    issues = get_open_issues(repo)

    summaries = []

    for i, issue in enumerate(issues, 1):
        print(f"\n📌 Issue {i}: {issue['title']}")
        print(f"🔗 URL: {issue['url']}")
        summary = summarize_issue(issue['title'], issue['body'])
        print(f"📝 Summary: {summary}")
        summaries.append(f"### Issue {i}: {issue['title']}\n🔗 {issue['url']}\n\n**Summary:** {summary}\n")

    # Save to markdown file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# GitHub Issue Summaries for `{repo}`\n\n")
        f.writelines('\n'.join(summaries))

    print(f"\n✅ Summaries saved to `{output_file}`")

if __name__ == "__main__":
    main()
