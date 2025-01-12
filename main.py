import asyncio
import time

from fetchers import check_rate_limit, fetch_user_details, fetch_repo_data
from parsers import parse_user_details, user_details_to_csv
from processing import get_text_stats
from ai_analysis import clean_and_store_data_for_ai, send_to_ai


async def main():
    try:
        print("------------------------------------------")
        print("Program is starting up...")
        rate_limit_info = await check_rate_limit()
        print("Rate Limit Information (Before Requests):")
        print(rate_limit_info)
        print("------------------------------------------")
        # Get GitHub username input
        username = input("Enter your GitHub username: ")
        start_time = time.time()
        print("------------------------------------------")
        print("Fetching data, please wait...")
        print("------------------------------------------")
        # Fetch user details and repository data concurrently
        user_data, repo_data = await asyncio.gather(
            fetch_user_details(username),
            fetch_repo_data(username)
        )

        # Parse and process fetched data
        user_details = parse_user_details(user_data)
        print("------------------------------------------")
        print("User Details: ")
        print(user_details)
        print("------------------------------------------")
        stats = get_text_stats(repo_data)
        print("Profile statistics: ")
        print(stats)
        print("------------------------------------------")
        # Export user details to a CSV file (if applicable)
        user_details_to_csv(user_details, f"{username}_details.csv")
        print(f"User details exported to {username}_details.csv")
        # Fetch updated rate limit info
        updated_rate_limit_info = await check_rate_limit()
        print("------------------------------------------")
        print("Rate Limit Information (After Requests):")
        print(updated_rate_limit_info)
        print("------------------------------------------")
        clean_and_store_data_for_ai(f"{username}_analysis.csv", f"{username}_analysis_cleaned.csv")
        print("Please wait while the AI answers to your response.")
        print("------------------------------------------")
        send_to_ai(username)
        # Calculate and display elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("------------------------------------------")
        print(f"Time taken: {elapsed_time:.2f} seconds")
        print("------------------------------------------")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
