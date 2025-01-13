import base64
import os
import time
import csv
import asyncio
import aiohttp
from dotenv import load_dotenv
import logging
from parsers import parse_user_repos
from processing import get_text_stats, perform_topic_modeling, extract_keywords, clean_text

# Load GitHub token for authorization
load_dotenv('secrets.env')
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

headers = {"Authorization": f"token {GITHUB_TOKEN}"}

# program will only look at files with these extensions
text_extensions = (
    ".txt", ".html", ".css", ".js", ".ts", ".jsx", ".tsx",
    ".json", ".yml", ".yaml", ".xml",
    ".py", ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".sh", ".bat", ".sql", ".rb", ".go", ".rs",
    ".ini", ".cfg", ".properties", ".pl", ".lua", ".swift", ".r", ".m", ".php", ".asp", ".jsp",
    ".scss", ".less", ".kt", ".gradle", ".svelte", ".dart", ".vb", ".vbs", ".ipynb", ".tf",
    ".clj", ".erl", ".lhs", ".tex", ".asciidoc", ".adoc", ".rst", ".xhtml", ".vtt", ".srt",
    ".rdf", ".ps1"
)
# banned files
excluded_filenames = (
    "package-lock.json", "yarn.lock", "npm-debug.log", "composer.lock",
    "Gemfile.lock", "Pipfile.lock", "requirements.txt", "Dockerfile",
    ".env", ".env.example", ".DS_Store", ".gitignore"
)


async def fetch_json(session, url):
    async with session.get(url) as response:
        return await response.json()


async def fetch_text(session, url):
    async with session.get(url) as response:
        return await response.text()


async def fetch_user_details(username):
    url = f"https://api.github.com/users/{username}"
    async with aiohttp.ClientSession(headers=headers) as session:
        return await fetch_json(session, url)


async def fetch_user_repos(username):
    url = f"https://api.github.com/users/{username}/repos"
    async with aiohttp.ClientSession(headers=headers) as session:
        return await fetch_json(session, url)


async def fetch_repo_languages(session, username, repo_name):
    url = f"https://api.github.com/repos/{username}/{repo_name}/languages"
    return await fetch_json(session, url)


async def fetch_repo_readme(session, username, repo_name):
    url = f"https://api.github.com/repos/{username}/{repo_name}/readme"
    readme_data = await fetch_json(session, url)
    if readme_data and 'download_url' in readme_data:
        return await fetch_text(session, readme_data['download_url'])
    return None


async def fetch_repo_file_structure(session, owner, repo, branch):
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    return await fetch_json(session, url)


async def fetch_file_content(session, owner, repo, file_path):
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    file_info = await fetch_json(session, url)
    if 'content' in file_info:
        return base64.b64decode(file_info['content']).decode('utf-8')
    return None


async def fetch_repo_data(username):
    async with aiohttp.ClientSession(headers=headers) as session:
        user_repos = await fetch_user_repos(username)
        if not isinstance(user_repos, list):
            logging.error(f"Unexpected response format for repos of {username}: {user_repos}")
            return []

        filtered_repos = []
        for repo in user_repos:
            repo_name = repo["name"]
            filtered_repo = parse_user_repos(repo)
            filtered_repo["languages"] = await fetch_repo_languages(session, username, repo_name)
            filtered_repo["readme"] = await fetch_repo_readme(session, username, repo_name)
            filtered_repo["structure"] = await fetch_repo_file_structure(session, username, repo_name,
                                                                         repo["default_branch"])

            readme_content = filtered_repo["readme"] or ""
            cleaned_readme = clean_text(readme_content) if readme_content.strip() else None

            # Filter and fetch content for each file
            repo_data = []
            if filtered_repo["structure"]:
                tree = filtered_repo["structure"].get("tree", [])
                tasks = [
                    fetch_file_content(session, username, repo_name, item["path"])
                    for item in tree
                    if item["path"].endswith(text_extensions) and not any(
                        excl in item["path"] for excl in excluded_filenames)
                ]
                repo_data = await asyncio.gather(*tasks)

            # Process repository data with text analysis (Sentiment, Topics, Keywords)

            languages_used = filtered_repo["languages"]
            processed_data = []
            for content in repo_data:
                if content and len(content.strip()) > 0:  # Check for non-empty content
                    stats = get_text_stats(content)  # Sentiment analysis, word count, etc.
                    topics = []
                    if cleaned_readme and len(cleaned_readme) > 0:  # Prioritize the README
                        topics = perform_topic_modeling([cleaned_readme], languages_used)  # Topic modeling

                    keywords = extract_keywords([content], languages_used)  # TF-IDF keyword extraction

                    # Proceed with data processing
                    processed_data.append({
                        "content": content,
                        "word_count": stats.get('word_count', ''),
                        "sentence_count": stats.get('sentence_count', ''),
                        "sentiment": stats.get('sentiment', ''),
                        "topics": ', '.join([str(topic) for topic in topics]),
                        "keywords": ', '.join([str(keyword) for keyword in keywords])
                    })
                    # print("topics", ', '.join([str(topic) for topic in topics]),)
                else:
                    logging.warning(f"Skipping empty content in repo: {repo_name}")

            filtered_repo["repo_data"] = processed_data
            filtered_repos.append(filtered_repo)

        # Save the data to CSV
        save_to_csv(filtered_repos, username + "_analysis")
        return filtered_repos


def save_to_csv(filtered_repos, csv_file_name):
    flat_data = []
    for repo in filtered_repos:
        for repo_data in repo.get("repo_data", []):
            flat_data.append({
                "repo_name": repo.get("name"),
                "languages": repo.get("languages"),
                # "readme": repo.get("readme"),
                "word_count": repo_data["word_count"],
                "sentence_count": repo_data["sentence_count"],
                "sentiment": repo_data["sentiment"],
                "topics": repo_data["topics"],
                "keywords": repo_data["keywords"]
            })

    # Write data to CSV
    keys = flat_data[0].keys() if flat_data else []
    with open(csv_file_name + '.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flat_data)

    print("Data saved to " + csv_file_name + ".csv")


async def fetch_repo_commits(session, username, repo_name):
    url = f"https://api.github.com/repos/{username}/{repo_name}/commits"
    return await fetch_json(session, url)


async def check_rate_limit():
    url = "https://api.github.com/rate_limit"
    async with aiohttp.ClientSession(headers=headers) as session:
        async with session.get(url) as response:
            rate_limit_info = {
                "limit": response.headers.get("x-ratelimit-limit"),
                "remaining": response.headers.get("x-ratelimit-remaining"),
                "used": response.headers.get("x-ratelimit-used"),
                "reset_time": time.strftime('%Y-%m-%d %H:%M:%S',
                                            time.gmtime(int(response.headers.get("x-ratelimit-reset", 0))))
            }
            return rate_limit_info
