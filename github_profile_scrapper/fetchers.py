import aiohttp
import asyncio
import os
import base64
from dotenv import load_dotenv
import time

from parsers import parse_user_repos

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
        filtered_repos = []

        if not isinstance(user_repos, list):
            print("Unexpected response format:", user_repos)
            return []

        for repo in user_repos:
            repo_name = repo["name"]
            filtered_repo = parse_user_repos(repo)
            filtered_repo["languages"] = await fetch_repo_languages(session, username, repo_name)
            filtered_repo["readme"] = await fetch_repo_readme(session, username, repo_name)
            filtered_repo["structure"] = await fetch_repo_file_structure(session, username, repo_name,
                                                                         repo["default_branch"])

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

            filtered_repo["repo_data"] = repo_data
            filtered_repos.append(filtered_repo)

        return filtered_repos


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
