import time
import asyncio

from github_profile_scrapper.fetchers import check_rate_limit, fetch_user_details, fetch_repo_data
from github_profile_scrapper.parsers import parse_user_details, user_details_to_csv

if __name__ == "__main__":
    rate_limit_info = asyncio.run(check_rate_limit())
    print("Rate Limit Information:")
    print(rate_limit_info)

    username = input("Enter your github username: ")
    start_time = time.time()
    print("Please wait...")

    # here is the data that needs to be cleaned and then processed by the ollama
    user_details = parse_user_details(asyncio.run(fetch_user_details(username)))
    repo_data = asyncio.run(fetch_repo_data(username))

    print(user_details)
    print(repo_data)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

    rate_limit_info = asyncio.run(check_rate_limit())
    print("Rate Limit Information:")
    print(rate_limit_info)
