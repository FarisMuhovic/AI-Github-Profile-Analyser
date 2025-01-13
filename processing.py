import logging
import re
from textblob import TextBlob
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.stem import WordNetLemmatizer
from keywords import expertise_keywords, topics_keywords
from nltk import FreqDist, word_tokenize, ngrams
from nltk.corpus import stopwords
import string
from collections import Counter
import matplotlib.pyplot as plt

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    import nltk

    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))


def clean_text(text):
    """
    Preprocess and clean text for analysis.
    - Retain meaningful non-alphabetic content (e.g., numbers, programming terms).
    - Remove stopwords and lemmatize words ( to root form ).
    """
    if not text.strip():  # Skip empty or whitespace-only content
        return ''

    # Lowercase text
    text = text.lower()

    # Retain alphanumeric and special programming characters
    text = re.sub(r'[^\w\s(\)\{\}\[\]\<\>\#\=\+\-\*/\.\,\:\;]', '', text)

    # Tokenize and filter stopwords
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    # Join cleaned words back into a single string
    return ' '.join(cleaned_words)


def get_text_stats(text):
    # Text Statistics Function
    # Gets the sentiment value, positive negative or neutral. Based on words,
    # Example:
    # "I love this" → polarity = 0.5 (positive).
    # "This is bad" → polarity = -0.7 (negative).
    text = str(text)
    word_count = len(text.split())
    sentence_count = text.count('.')  # simple sentence count

    # Sentiment analysis
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity  # Sentiment score between -1 (negative) and 1 (positive)

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "sentiment": sentiment
    }


def analyze_vocabulary(text):
    # Tokenize and preprocess
    text = str(text).lower()
    tokens = word_tokenize(text)

    # Define stopwords and punctuation to filter
    stop_words = set(stopwords.words('english'))
    custom_stop_words = set(string.punctuation)
    custom_stop_words.update(["''", "``", "n't", "'s", "--", "...", "\n", "\t", "\\n"])

    # Additional filters: remove single characters, numbers, and other noise
    def is_valid_token(token):
        return (
                token not in stop_words and
                token not in custom_stop_words and
                len(token) > 1 and  # Exclude single characters
                not token.isdigit() and  # Exclude numbers
                not re.match(r'^\s+$', token)  # Exclude whitespace
        )

    # Filter tokens
    filtered_tokens = [word for word in tokens if is_valid_token(word)]

    # Vocabulary size
    vocab_size = len(set(filtered_tokens))

    # Most common words
    most_common_words = FreqDist(filtered_tokens).most_common(25)

    # Generate bigrams and filter based on custom stopwords
    bigram_list = list(
        filter(lambda x: is_valid_token(x[0]) and is_valid_token(x[1]), ngrams(filtered_tokens, 2))
    )
    bigrams = Counter(bigram_list).most_common(25)

    # Plot distribution
    plot_word_frequency(filtered_tokens)

    return {
        "vocab_size": vocab_size,
        "most_common_words": most_common_words,
        "bigrams": bigrams,
    }


def plot_word_frequency(tokens):
    """
    Plot the frequency distribution of the most common words.
    """
    fdist = FreqDist(tokens)
    most_common = fdist.most_common(25)  # Top 10 most frequent words
    words, counts = zip(*most_common)  # Unpack words and counts

    # Plot bar chart
    plt.figure(figsize=(25, 10))
    plt.bar(words, counts, color='skyblue')
    plt.xlabel('Words', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Top 25 Most Frequent Words', fontsize=15)
    plt.xticks(rotation=45)
    plt.show()


# Topic Modeling Function (LDA)
def perform_topic_modeling(documents, languages_used=None, num_topics=3, num_words=10):
    """
    Performs topic modeling using LDA, focusing on relevant technical topics.
    Arguments:
    - documents: List of text documents (strings).
    - languages_used: List of programming languages used (optional, currently not utilized).
    - num_topics: Number of topics to extract.
    - num_words: Number of words to extract per topic.
    Returns:
    - Relevant topics: List of dictionaries with topic names and top words.
    """
    if not documents or all(len(text.strip()) == 0 for text in documents):
        logging.error("No valid content provided for topic modeling.")
        return []

    # Clean and preprocess documents
    cleaned_documents = [clean_text(doc) for doc in documents]

    if not any(cleaned_documents):
        logging.error("No valid content remains after cleaning.")
        return []

    # Create a CountVectorizer to convert text into token counts
    vectorizer = CountVectorizer(stop_words='english', max_features=2000)
    X = vectorizer.fit_transform(cleaned_documents)

    # Apply Latent Dirichlet Allocation (LDA)
    # LDA is a generative probabilistic model that assumes each document is a mixture of topics,
    # where a topic is a probability distribution over words
    # The goal is to identify patterns in word co-occurrence that suggest topics within the text.
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)

    # Get the top words for each topic
    topic_words = []
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-num_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topic_words.append({"Topic": f"Topic {topic_idx + 1}", "Words": top_words})

    # Filter topics to focus on relevant keywords
    relevant_topics = []
    for topic in topic_words:
        matching_keywords = [
            word for word in topic["Words"]
            if any(word in topics_keywords[key] for key in topics_keywords)
        ]
        if matching_keywords:
            relevant_topics.append({
                "Topic": topic["Topic"],
                "Relevant Words": matching_keywords,
            })

    if not relevant_topics:
        logging.warning("No relevant topics matched the keywords.")
        return []  # Return all topics as a fallback

    return relevant_topics


# Keyword Extraction Function (TF-IDF)
def extract_keywords(documents, languages_used, num_keywords=25):
    expertise_keywords_filtered = filter_keywords(languages_used, expertise_keywords)
    """
    Term Frequency-Inverse Document Frequency
    Extracts keywords based on TF-IDF from a list of documents.
    identifying important words in text
    Arguments:
    - documents: List of text documents (strings).
    - num_keywords: Number of keywords to extract per document.
    Returns:
    - List of lists containing top keywords for each document.
    """
    cleaned_documents = [clean_text(doc) for doc in documents]

    if not cleaned_documents:
        return {"error": "No valid documents to process."}

    # Create a TfidfVectorizer to compute the TF-IDF scores
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(cleaned_documents)

    # Get the top keywords
    feature_names = vectorizer.get_feature_names_out()
    keywords = []

    for doc_idx in range(X.shape[0]):
        scores = X[doc_idx, :].toarray().flatten()
        top_keywords_idx = scores.argsort()[-num_keywords:][::-1]
        top_keywords = [feature_names[i] for i in top_keywords_idx]
        keywords.append(top_keywords)

    # Map keywords to expertise
    expertise_keywords_found = []
    for doc_keywords in keywords:
        expertise_keywords_found.append(
            {topic: [keyword for keyword in doc_keywords if keyword in expertise_keywords_filtered[topic]] for topic in
             expertise_keywords_filtered})

    return expertise_keywords_found


# Function to filter keywords based on detected languages
def filter_keywords(languages, expertise_keywords):
    """
    Filters expertise keywords based on the provided languages. Only the expertise keywords
    for the specified languages are returned.

    Arguments:
    - languages: List of programming languages to filter by.
    - expertise_keywords: Dictionary where the key is the language and the value is the list of expertise keywords.

    Returns:
    - A dictionary where the key is the language and the value is a list of expertise keywords for that language.
    """
    filtered_keywords = {}

    for lang in languages:
        if lang in expertise_keywords:
            filtered_keywords[lang] = expertise_keywords[lang]

    return filtered_keywords
