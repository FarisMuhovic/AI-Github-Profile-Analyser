import csv


def parse_user_details(user_details):
    return {
        "name": user_details.get("login", "N/A"),
        "avatar_url": user_details.get("avatar_url", ""),
        "profile_url": user_details.get("html_url", ""),
        "repos_url": user_details.get("repos_url", ""),
        "public_repos_count": user_details.get("public_repos", 0),
        "private_repos_count": user_details.get("total_private_repos", 0),
        "total_repos": (
                user_details.get("public_repos", 0) + user_details.get("total_private_repos", 0)
        ),
        "followers_count": user_details.get("followers", 0),
        "following_count": user_details.get("following", 0),
    }


def parse_user_repos(repo):
    return {
        "name": repo.get("name", "N/A"),
        "full_name": repo.get("full_name", "N/A"),
        "default_branch": repo.get("default_branch", "N/A"),
        "html_url": repo.get("html_url", ""),
        "clone_url": repo.get("clone_url", ""),
        "description": repo.get("description", "No description available"),
        "forks_count": repo.get("forks_count", 0),
        "open_issues_count": repo.get("open_issues_count", 0),
        "pushed_at": repo.get("pushed_at", "Never pushed"),
        "topics": repo.get("topics", []),
    }


def user_details_to_csv(data, filename):
    if not data:
        print("No data provided.")
        return

    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)

    print(f"Data written to {filename} successfully.")


def repo_data_to_csv(data, filename):
    if not data:
        print("No data provided.")
        return

    fieldnames = data[0].keys()

    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"Data written to {filename} successfully.")
