import logging
import re
from textblob import TextBlob
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keywords import expertise_keywords, topics_keywords

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Define priority keywords for analysis

# Initialize the lemmatizer
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
    - Remove stopwords and lemmatize words.
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


# Text Statistics Function
def get_text_stats(text):
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


# Topic Modeling Function (LDA)
def perform_topic_modeling(documents, languages_used, num_topics=10):
    if not documents or all(len(text.strip()) == 0 for text in documents):
        logging.error("No valid content provided for topic modeling.")
        return []

    """
    Performs topic modeling using LDA, focusing on relevant technical topics.
    Arguments:
    - documents: List of text documents (strings).
    - num_topics: Number of topics to extract.
    Returns:
    - List of topics (each topic represented as a list of words).
    """
    # cleaned_documents = [clean_text(doc) for doc in documents]

    if not documents:
        return {"error": "No valid documents to process."}

    # Create a CountVectorizer to convert text into token counts
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(documents)

    # Apply Latent Dirichlet Allocation (LDA)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)

    # Get the top words for each topic
    topic_words = []
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[:-11:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topic_words.append(top_words)

    # Filter topics to focus on technical jargon

    relevant_topics = [
        topic for topic in topic_words
        if any(word in topics_keywords for word in topic)
    ]

    if not relevant_topics:
        logging.warning("No relevant topics matched the keywords.")
        return topic_words  # Return all topics as a fallback

    return relevant_topics


# Keyword Extraction Function (TF-IDF)
def extract_keywords(documents, languages_used, num_keywords=10):
    expertise_keywords_filtered = filter_keywords(languages_used, expertise_keywords)
    """
    Extracts keywords based on TF-IDF from a list of documents.
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
