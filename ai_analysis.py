from collections import defaultdict

import pandas as pd
import ast
from ollama import chat
from ollama import ChatResponse


class ProjectData:
    def __init__(self, repo_name, languages, word_count, sentence_count, sentiment, topics, keywords):
        self.repo_name = repo_name
        self.languages = ast.literal_eval(languages)  # Convert string representation of dictionary to actual dictionary
        self.word_count = word_count
        self.sentence_count = sentence_count
        self.sentiment = sentiment
        self.topics = self.clean_data(self.safe_literal_eval(topics))  # Clean empty keys
        self.keywords = self.clean_data(self.safe_literal_eval(keywords))  # Clean empty keys

    def safe_literal_eval(self, data):
        """
        Safely evaluate the string data, ensuring it is a valid dictionary or empty.
        Returns an empty dictionary if data is NaN or not a valid dictionary.
        """
        try:
            if pd.isna(data) or not data:
                return {}
            return ast.literal_eval(data)  # Evaluate if valid
        except (ValueError, SyntaxError):
            return {}  # Return an empty dictionary if evaluation fails

    def is_empty(self):
        """
        Check if both topics and keywords are empty.
        Returns True if both are empty, otherwise False.
        """
        return not bool(self.topics) and not bool(self.keywords)

    def clean_data(self, data):
        """
        Clean the data by removing keys with empty lists (e.g. 'react': [])
        """
        if isinstance(data, dict):
            # Filter out empty lists
            return {key: value for key, value in data.items() if value}
        return data

    def __str__(self):
        return f"Project: {self.repo_name}, Sentiment: {self.sentiment}, Word Count: {self.word_count}, Sentence Count: {self.sentence_count}, Topics: {self.topics}"


class GitHubUser:
    def __init__(self, name, avatar_url, profile_url, repos_url, public_repos_count, private_repos_count, total_repos,
                 followers_count, following_count):
        self.name = name
        self.avatar_url = avatar_url
        self.profile_url = profile_url
        self.repos_url = repos_url
        self.public_repos_count = public_repos_count
        self.private_repos_count = private_repos_count
        self.total_repos = total_repos
        self.followers_count = followers_count
        self.following_count = following_count

    def __str__(self):
        return f"User: {self.name}, Public Repos: {self.public_repos_count}, Followers: {self.followers_count}, Following: {self.following_count}"


def clean_and_store_data_for_ai(input_csv_filename, output_csv_filename):
    print(f"Reading from: {input_csv_filename} and saving to: {output_csv_filename}")
    print("------------------------------------------")
    data = pd.read_csv(input_csv_filename)

    response_data = []

    for index, row in data.iterrows():
        project = ProjectData(
            repo_name=row['repo_name'],
            languages=row['languages'],
            word_count=row['word_count'],
            sentence_count=row['sentence_count'],
            sentiment=row['sentiment'],
            topics=row['topics'],
            keywords=row['keywords']
        )


        if not project.is_empty():
            response_data.append(
                [project.repo_name, project.sentiment, project.word_count, project.sentence_count, project.sentiment,
                 project.topics, project.keywords])

    # Convert the response data into a DataFrame and save it to the specified output CSV
    response_df = pd.DataFrame(response_data,
                               columns=["Project", "Sentiment", "Word Count", "Sentence Count", "Sentiment", "Topics",
                                        "Keywords"])
    response_df.to_csv(output_csv_filename, index=False)

    print(f"Analysis results saved to {output_csv_filename}")


def read_user_details(csv_filename):
    """
    Reads a CSV file containing GitHub user data and returns a list of GitHubUser objects.
    """
    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(csv_filename)

    users = []

    # Loop through each row in the DataFrame and create GitHubUser objects
    for index, row in data.iterrows():
        user = GitHubUser(
            name=row['name'],
            avatar_url=row['avatar_url'],
            profile_url=row['profile_url'],
            repos_url=row['repos_url'],
            public_repos_count=row['public_repos_count'],
            private_repos_count=row['private_repos_count'],
            total_repos=row['total_repos'],
            followers_count=row['followers_count'],
            following_count=row['following_count']
        )
        users.append(user)

    return users


def read_cleaned_analysis(csv_filename):
    """
    Reads a CSV file containing cleaned analysis data and returns a list of ProjectData objects.
    """
    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(csv_filename)

    projects = []

    # Loop through each row in the DataFrame and create ProjectData objects
    for index, row in data.iterrows():
        project = ProjectData(
            repo_name=row['Project'],
            languages=row['Keywords'],  # You can map these accordingly
            word_count=row['Word Count'],
            sentence_count=row['Sentence Count'],
            sentiment=row['Sentiment'],
            topics=row['Topics'],
            keywords=row['Keywords']
        )
        projects.append(project)

    return projects


def send_to_ai(filename):
    user = read_user_details(filename + "_details.csv")
    projects = read_cleaned_analysis(filename + "_analysis_cleaned.csv")  # Merge files from project into one.
    user = user[0]
    projects_merged = merge_projects_together(projects)

    user_prompt_info = f"""
        Analyze the following GitHub user profile in the context of their development expertise:
        - User: {user.name}
        - Profile URL: {user.profile_url}
        - Avatar: {user.avatar_url}
        - Public Repos Count: {user.public_repos_count}
        - Private Repos Count: {user.private_repos_count}
        - Total Repos: {user.total_repos}
        - Followers: {user.followers_count}
        - Following: {user.following_count}
        
        Analyze the user's activity to infer their development expertise, including their familiarity with different programming languages, tools, and frameworks, and their involvement in various projects. Additionally, consider:
        - The scope and complexity of repositories they contribute to, potentially indicating their level of experience (e.g., beginner, intermediate, or expert).
        - The diversity of the repositories, which may reflect their interests across different technology stacks (e.g., web development, machine learning, DevOps).
            
        Settings:
        - Max Word Count for Analysis: 1000 words.
        - Handle Missing Data: If `topics` or `keywords` are missing, mention it as "No relevant data available" or "No topics found."
        - Tone: Provide a balanced and insightful tone, with an emphasis on constructive feedback.
        - Fallback for Missing Fields: If the sentiment is not available, explain that the sentiment might be unclear due to limited project documentation.
           
        Below is the project data of the current user.  
        Keywords are the important tokens from the codebase of that repository.
        Topics are general concepts in the IT world.  
    """

    # Construct the project data in a concise format
    projects_data = ""
    for project in projects_merged:
        # Summarizing project information
        project_info = f"""
            Project: {project['Project']}
            Sentiment: {project['Sentiment']} (Positive/Negative/Neutral)
            Word Count: {project['Word Count']}
            Sentence Count: {project['Sentence Count']}
            Languages Used: {project["Languages"]}
            Keywords: {', '.join(project['Keywords']) if project['Keywords'] else 'No keywords'}
            Topics: {', '.join([str(topic) for topic in project['Topics'][:10]])} 
            ----------------------------------------
        """
        projects_data += project_info

    # Final consolidated prompt
    user_prompt = f"""
        {user_prompt_info}
        
        Projects Data:
        {projects_data}

        Analyze the user's expertise, project complexity, sentiment, and keywords. For each project:
        - Evaluate the overall complexity based on word count, sentiment, and keywords.
        - Identify any trends in the topics or technologies.
        - Offer insights into the user's development expertise based on their contributions.
        - Provide a summary of their strengths and potential areas for improvement.
    """

    print(user_prompt)
    print("------------------------------------------")

    response: ChatResponse = chat(model='llama3.2', messages=[
        {
            'role': 'user',
            'content': user_prompt
        },
    ])
    print("------------------------------------------")
    print(response.message.content)
    print("------------------------------------------")
    save_response_to_file(filename, user_prompt, response.message.content)


def save_response_to_file(username, prompt, response):
    # Create the filename using the username
    filename = f"{username}_response.txt"

    # Open the file in write mode
    with open(filename, 'w') as file:
        # Write the username and the review statement
        file.write(f"{username} has reviewed their profile.\n\n")
        file.write("------------------------------------------")
        # Write the prompt
        file.write("Prompt:\n")
        file.write(f"{prompt}\n\n")
        file.write("------------------------------------------")
        # Write the response
        file.write("Response:\n")
        file.write(f"{response}\n")

    print(f"Response saved to {filename}")



def merge_projects_together(projects):
    # Grouping data by project name
    grouped_projects = defaultdict(lambda: {
        'sentiment': [],
        'word_count': [],
        'sentence_count': [],
        'topics': set(),
        'keywords': set(),
        'languages': set()  # Field for languages extracted from keywords
    })

    for project in projects:
        project_name = project.repo_name  # Use dot notation to access attributes
        sentiment = project.sentiment
        word_count = project.word_count
        sentence_count = project.sentence_count
        topics = project.topics
        keywords = project.keywords  # This should already be a dictionary

        # Parse the keywords and extract languages
        if keywords:
            grouped_projects[project_name]['languages'].update(keywords.keys())  # Extract languages
            for key, keyword_list in keywords.items():
                grouped_projects[project_name]['keywords'].update(keyword_list)

        # Add topics to the grouped data
        if topics and isinstance(topics, list):
            grouped_projects[project_name]['topics'].update(topics)

        # Add other project data to the grouped data
        grouped_projects[project_name]['sentiment'].append(sentiment)
        grouped_projects[project_name]['word_count'].append(word_count)
        grouped_projects[project_name]['sentence_count'].append(sentence_count)

    # Merging grouped data
    merged_projects = []
    for project_name, data in grouped_projects.items():
        merged_project = {
            'Project': project_name,
            'Sentiment': sum(data['sentiment']) / len(data['sentiment']) if data['sentiment'] else None,
            'Word Count': sum(data['word_count']),
            'Sentence Count': sum(data['sentence_count']),
            'Topics': list(data['topics']),
            'Keywords': list(data['keywords']),
            'Languages': list(data['languages'])  # Include extracted languages
        }
        merged_projects.append(merged_project)

    return merged_projects