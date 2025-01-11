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

    # Read the CSV file locally
    data = pd.read_csv(input_csv_filename)

    response_data = []

    # Loop through each row of the CSV data
    for index, row in data.iterrows():
        print(f"Processing row {index}: {row['repo_name']}")

        # Create a ProjectData object from the row
        project = ProjectData(
            repo_name=row['repo_name'],
            languages=row['languages'],
            word_count=row['word_count'],
            sentence_count=row['sentence_count'],
            sentiment=row['sentiment'],
            topics=row['topics'],
            keywords=row['keywords']
        )

        # Optionally process the data further or analyze it
        print(f"Project data: {project}")

        # Store the cleaned data: project name, sentiment, word count, sentence count, topics, and keywords
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
    projects = read_cleaned_analysis(
        filename + "_analysis_cleaned.csv")  # WE NEED TO MERGE EACH FILE FROM PROJECT INTO ONE.
    user = user[0]

    user_prompt = f"""
        Analyze the following GitHub user profile in the context of their development expertise:
        - User: {user.name}
        - Profile URL: {user.profile_url}
        - Avatar: {user.avatar_url}
        - Public Repos Count: {user.public_repos_count}
        - Private Repos Count: {user.private_repos_count}
        - Total Repos: {user.total_repos}
        - Followers: {user.followers_count}
        - Following: {user.following_count}
        
        The user has {user.public_repos_count} public repositories and {user.followers_count} followers. 
        Analyze the user's activity to infer their development expertise, including their familiarity with different programming languages, tools, and frameworks, and their involvement in various projects. Additionally, consider:
        - The scope and complexity of repositories they contribute to, potentially indicating their level of experience (e.g., beginner, intermediate, or expert).
        - The diversity of the repositories, which may reflect their interests across different technology stacks (e.g., web development, machine learning, DevOps).
        Provide a comprehensive analysis of the user’s expertise, activity, and overall profile based on this data.        
        
        
    **Settings**:
    - **Max Word Count for Analysis**: 300 words.
    - **Handle Missing Data**: If `topics` or `keywords` are missing, mention it as "No relevant data available" or "No topics found."
    - **Tone**: Provide a balanced and insightful tone, with an emphasis on constructive feedback.
    - **Fallback for Missing Fields**: If the sentiment is not available, explain that the sentiment might be unclear due to limited project documentation.
        """

    print(user_prompt)

    # for project in projects:
    #     print(f"Sending project to AI: {project}")
    #     # Prepare prompt for AI to analyze project data
    #     project_prompt = f"""
    #         Analyze the following GitHub project in the context of software engineering practices, project complexity, and technical skills:
    #         - Project Name: {project.repo_name}
    #         - Word Count: {project.word_count}
    #         - Sentence Count: {project.sentence_count}
    #         - Sentiment: {project.sentiment}
    #         - Topics: {project.topics}
    #         - Keywords: {project.keywords}
    #
    #         The project consists of {project.word_count} words and {project.sentence_count} sentences, indicating its size and the depth of documentation or code comments. The sentiment analysis reveals a score of {project.sentiment}, which can provide insights into the tone of the project—whether it's focused on innovation, collaboration, problem-solving, or other themes.
    #
    #         The topics of the project cover areas such as {project.topics}, which are crucial to understanding the technical stack and domain expertise of the repository. Keywords extracted from the content like {project.keywords} can offer additional insight into the core technologies, frameworks, and methodologies being utilized.
    #
    #         Based on this data, analyze:
    #         - The **complexity** of the project, taking into account its size (word count, sentence count) and technical content. Does it appear to be a large-scale project that requires advanced knowledge, or is it a smaller, more focused initiative?
    #         - The **sentiment** of the project, which could indicate the approach or challenges the project is addressing. A negative sentiment might suggest unresolved issues or challenges, while a positive sentiment may point to a well-supported and active project.
    #         - The **relevance and focus of keywords and topics** within the project. How do these align with industry trends and the user’s areas of expertise? Are they primarily focused on modern tools and frameworks, or do they suggest an interest in legacy technologies?
    #
    #         Provide an in-depth analysis of the project’s complexity, sentiment, and technical relevance based on this information, and offer insights into the developer's strengths and areas for improvement.
    #         """
    #
    #     print(f"Project Prompt for AI: {project_prompt}")

    response: ChatResponse = chat(model='llama3.2', messages=[
        {
            'role': 'user',
            'content': user_prompt
        },
    ])
    print(response.message.content)
    save_response_to_file(filename, user_prompt, response.message.content)

def save_response_to_file(username, prompt, response):
    # Create the filename using the username
    filename = f"{username}_response.txt"

    # Open the file in write mode
    with open(filename, 'w') as file:
        # Write the username and the review statement
        file.write(f"{username} has reviewed their profile.\n\n")

        # Write the prompt
        file.write("Prompt:\n")
        file.write(f"{prompt}\n\n")

        # Write the response
        file.write("Response:\n")
        file.write(f"{response}\n")

    print(f"Response saved to {filename}")
