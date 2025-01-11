import re
import string
from textblob import TextBlob
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Define priority keywords for analysis
expertise_keywords = {
    'oop': [
        'class', 'object', 'inheritance', 'polymorphism', 'encapsulation', 'abstraction', 'interface',
        'method', 'constructor', 'overloading', 'overriding', 'singleton', 'factory', 'observer',
        'dependency injection', 'design pattern', 'composition', 'aggregation', 'coupling', 'cohesion', 'class diagram',
        'this', 'super', 'private', 'protected', 'public', 'abstract', 'interface', 'constructor', 'method signature',
        'method overloading', 'method overriding', 'overriding', 'delegation', 'composition'
    ],
    'react': [
        'react', 'component', 'jsx', 'useeffect', 'usereducer', 'state', 'props', 'componentDidMount',
        'componentWillUnmount', 'useState', 'useContext', 'React Router', 'Redux', 'context api', 'hook',
        'render', 'memo', 'purecomponent', 'virtual dom', 'react native', 'component lifecycle', 'react hooks',
        'redux saga', 'useRef', 'componentDidUpdate', 'useMemo', 'reducer', 'setState', 'React.createElement',
        'useCallback', 'defaultProps', 'propTypes'
    ],
    'java': [
        'public static void main', 'import', 'class', 'new', 'String[] args', 'extends', 'implements', 'super',
        'try', 'catch', 'finally', 'synchronized', 'interface', 'abstract', 'enum', 'hashmap', 'ArrayList', 'java.util',
        'System.out.println', 'JVM', 'JDK', 'Runnable', 'exception handling', 'package', 'throw', 'throws', 'lambda',
        'stream', 'collections', 'multithreading', 'synchronized', 'javadoc', 'constructor', 'method signature',
        'abstract class',
        'interface', 'private', 'protected', 'public', 'default'
    ],
    'python': [
        'def', 'lambda', 'import', 'class', 'self', 'from', 'try', 'except', 'finally', 'with', 'yield', 'return',
        'async', 'await', 'open', 'read', 'write', 'for', 'in', 'if', 'elif', 'else', 'import os', 'import sys',
        'try-except', 'list comprehension', 'flask', 'django', 'requests', 'pandas', 'numpy', 'matplotlib', 'openCV',
        'json', 'json.loads', 'json.dumps', 'import numpy as np', 'pandas.DataFrame', 'staticmethod', 'classmethod',
        'decorators', 'map', 'filter', 'reduce'
    ],
    'async_programming': [
        'async', 'await', 'asyncio', 'eventloop', 'task', 'future', 'coroutine', 'thread', 'multiprocessing',
        'lock', 'semaphore', 'callback', 'concurrent.futures', 'non-blocking', 'gevent', 'celery', 'threading', 'queue',
        'awaitable', 'event loop', 'await', 'non-blocking', 'asyncio.gather', 'asyncio.create_task', 'async def'
    ],
    'web_development': [
        'html', 'css', 'javascript', 'nodejs', 'express', 'api', 'graphql', 'ajax', 'json', 'rest', 'http', 'get',
        'post', 'put', 'delete', 'fetch', 'axios', 'fetch()', 'document.getElementById', 'document.querySelector',
        'style', 'font-family', 'box-shadow', 'border-radius', 'npm', 'webpack', 'babel', 'vue', 'react', 'angular',
        'jsdelivr', 'require', 'module.exports', 'export default', 'router', 'vuex', 'redux', 'context api',
        'bootstrap',
        'tailwindcss', 'sass', 'semantic-ui', 'nextjs', 'nuxtjs', 'gatsby', 'pug', 'handlebars', 'ejs',
        'express.Router',
        'html5', 'html5 video', 'custom events', 'script tag', 'async defer', 'responsive design', 'media query'
    ],
    'angular': [
        'angular', 'component', 'ngOnInit', 'ngOnDestroy', 'ngModel', 'ngFor', 'ngIf', 'rxjs', 'angular cli',
        'dependency injection',
        'module', 'directive', 'pipe', 'observable', 'ngRoute', 'ngForm', 'ts', 'rxjs.subject', 'observable.subscribe',
        'ngBootstrap', 'angular material', 'angular universal', 'ngSwitch', 'services', 'HttpClient', 'zone.js'
    ],
    'sql': [
        'select', 'from', 'where', 'insert', 'update', 'delete', 'join', 'left join', 'right join', 'inner join',
        'full outer join', 'group by', 'having', 'order by', 'limit', 'distinct', 'union', 'create table',
        'alter table',
        'drop table', 'primary key', 'foreign key', 'index', 'varchar', 'int', 'float', 'decimal', 'timestamp',
        'date', 'boolean', 'insert into', 'update set', 'select count', 'select sum', 'transaction', 'commit',
        'rollback'
    ],
    'nosql': [
        'mongodb', 'document', 'collection', 'db.collection', 'insertOne', 'findOne', 'updateOne', 'deleteOne', 'find',
        'aggregate', 'mapReduce', 'mongod', 'mongoose', 'NoSQL', 'json', 'schema', 'model', 'indexing', 'sharding',
        'replica set',
        'mongodb atlas', 'BSON', 'key-value', 'redis', 'cassandra', 'column-family', 'neo4j', 'graph database',
        'elasticsearch',
        'key-value store', 'document store'
    ],
    'machine_learning': [
        'fit', 'predict', 'cross-validation', 'scikit-learn', 'tensorflow', 'keras', 'pytorch', 'model',
        'classifier', 'regressor', 'feature', 'train_test_split', 'accuracy_score', 'confusion_matrix', 'svm', 'kmeans',
        'decision tree', 'neural network', 'activation', 'gradient descent', 'backpropagation', 'loss function',
        'overfitting', 'underfitting', 'xgboost', 'model.fit', 'y_pred', 'accuracy', 'precision', 'recall', 'roc curve',
        'predict_proba', 'train', 'test', 'validation', 'cross-validation', 'random forest'
    ],
    'cloud_computing': [
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'docker-compose', 'vpc', 'ec2', 's3', 'lambda',
        'cloudwatch', 'cloudformation', 'ci/cd', 'jenkins', 'load balancer', 'microservices', 'api gateway',
        'elasticbeanstalk',
        'container', 'cloud-native', 'cloud-init', 'instance', 'keypair', 'security group', 'serverless', 'ecr',
        'ecr push',
        'k8s', 'cloud sdk', 'jenkins pipeline', 'datadog', 'logging', 'prometheus', 'grafana'
    ],
    'devops': [
        'git', 'gitlab', 'jenkins', 'terraform', 'docker', 'kubernetes', 'ansible', 'ci/cd', 'ci pipeline', 'helm',
        'jenkinsfile', 'docker registry', 'infra as code', 'aws', 'cloudformation', 'ecs', 'azure devops', 'monitoring',
        'prometheus', 'grafana', 'elk stack', 'k8s', 'deployment', 'scaling', 'logging', 'docker swarm',
        'blue-green deployment',
        'scaling groups', 'monitoring', 'logging', 'redis', 'jenkins pipeline', 'continuous integration',
        'continuous delivery'
    ],
    'nlp': [
        'import', 'tokenization', 'stopwords', 'stem', 'lemmatization', 'nltk', 'spaCy', 'pos tagging',
        'named entity recognition',
        'bag of words', 'tf-idf', 'word2vec', 'fasttext', 'bert', 'gpt-3', 'seq2seq', 'attention mechanism',
        'text classification',
        'sentiment analysis', 'topic modeling', 'spacy.load', 'word embeddings', 'BERTTokenizer', 'transformers',
        'seq2seq',
        'latent dirichlet allocation', 'collocations', 'wordnet', 'word2vec', 'nlp pipeline'
    ],
    'c_language': [
        '#include', 'int main()', 'printf', 'scanf', 'malloc', 'free', 'void', 'return', 'struct', 'typedef', 'char',
        'int', 'float', 'double', 'pointer', 'if', 'else', 'for', 'while', 'switch', 'case', 'break', 'continue',
        'sizeof', 'null', 'strlen', 'strcmp', 'memcpy', 'calloc', 'pointer dereference', 'system call'
    ],
    'cpp_language': [
        '#include', 'class', 'public', 'private', 'protected', 'virtual', 'new', 'delete', 'cout', 'cin', 'namespace',
        'template', 'std', 'vector', 'string', 'const', 'int main()', 'for', 'while', 'if', 'else', 'do', 'try',
        'catch',
        'exceptions', 'polymorphism', 'inheritance', 'overloading', 'overriding', 'constructor', 'destructor',
        'std::vector',
        'multithreading', 'mutex', 'atomic'
    ],
    'game_development': [
        'unity', 'unreal engine', 'game loop', 'rigidbody', 'collisions', 'events', 'gameObject', 'transform',
        'mesh', 'shader', 'render', 'fps', 'input', 'audio', 'camera', 'player', 'multiplayer', 'physics', 'ai', 'pbr',
        'game mechanic', 'game design', 'c#', 'cpp', 'game assets', 'collision detection', 'game testing'
    ],
    'blockchain': [
        'blockchain', 'ethereum', 'smart contract', 'solidity', 'web3', 'nft', 'ipfs', 'gas', 'web3.js', 'event',
        'chainlink',
        'ethereum 2.0', 'consensus', 'mining', 'proof of work', 'proof of stake', 'hash', 'block', 'transaction',
        'wallet',
        'private key', 'public key', 'eip', 'truffle', 'hardhat', 'cryptocurrency', 'metamask', 'dao', 'cryptography'
    ],
    'iot': [
        'iot', 'mqtt', 'zigbee', 'raspberry pi', 'arduino', 'esp32', 'sensor', 'actuator', 'edge computing',
        'iot device',
        'home automation', 'node-red', 'digital twin', 'gateway', 'cloud', 'mqtt broker', 'data logger', 'wifi',
        'zigbee',
        'low power devices', 'sensor fusion', 'iot security'
    ]
}

priority_keywords = {
    keyword: 3 if topic in ['oop', 'react', 'java', 'python', 'async_programming', 'machine_learning', 'nlp',
                            'game_development'] else 2
    for topic, keywords in expertise_keywords.items()
    for keyword in keywords
}

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()


# Data Cleaning Function
def clean_text(text):
    text = str(text)
    # Convert text to lowercase
    text = text.lower()

    # Remove non-alphabetic characters (keep spaces between words)
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize the text and remove stopwords
    words = text.split()
    stop_words = set(stopwords.words('english'))
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

    # Rejoin the words to form the cleaned text
    cleaned_text = ' '.join(cleaned_words)

    return cleaned_text


# Analyze Expertise Based on Keywords
def analyze_expertise(text):
    """
    Analyze the user's expertise based on the occurrence of specific keywords.
    Arguments:
    - text: The text content of a file or repository.
    Returns:
    - A dictionary of expertise areas with counts or mentions.
    """
    expertise_analysis = {topic: 0 for topic in expertise_keywords}

    # Tokenize and search for expertise-related keywords
    words = text.lower().split()
    for topic, keywords in expertise_keywords.items():
        for keyword in keywords:
            expertise_analysis[topic] += words.count(keyword)

    # Filter out expertise areas with no mentions
    expertise_results = {topic: count for topic, count in expertise_analysis.items() if count > 0}

    # Generate insights for each expertise category
    insights = []
    for topic, count in expertise_results.items():
        expertise_level = "Beginner"
        if count > 5:
            expertise_level = "Expert"
        elif count > 2:
            expertise_level = "Intermediate"
        insights.append(f"{topic.capitalize()}: {expertise_level} (Mentions: {count})")

    return insights


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
def perform_topic_modeling(documents, num_topics=10):
    """
    Performs topic modeling using LDA, focusing on relevant technical topics.
    Arguments:
    - documents: List of text documents (strings).
    - num_topics: Number of topics to extract.
    Returns:
    - List of topics (each topic represented as a list of words).
    """
    cleaned_documents = [clean_text(doc) for doc in documents]

    # Create a CountVectorizer to convert text into token counts
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(cleaned_documents)

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
    relevant_topics = [topic for topic in topic_words if
                       any(word in topic for word in ['async', 'class', 'react', 'python', 'java'])]

    return relevant_topics


# Keyword Extraction Function (TF-IDF)
def extract_keywords(documents, num_keywords=10):
    """
    Extracts keywords based on TF-IDF from a list of documents.
    Arguments:
    - documents: List of text documents (strings).
    - num_keywords: Number of keywords to extract per document.
    Returns:
    - List of lists containing top keywords for each document.
    """
    cleaned_documents = [clean_text(doc) for doc in documents]

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
            {topic: [keyword for keyword in doc_keywords if keyword in expertise_keywords[topic]] for topic in
             expertise_keywords})

    return expertise_keywords_found


# Prioritize Keywords Based on Custom Priority
def prioritize_keywords(keywords):
    """
    Adjust keyword importance based on predefined priority levels.
    Arguments:
    - keywords: List of extracted keywords.
    Returns:
    - A dictionary of keywords with adjusted priority scores.
    """
    adjusted_keywords = {}

    for keyword in keywords:
        priority = priority_keywords.get(keyword, 1)  # Default to low priority (1)
        adjusted_keywords[keyword] = priority

    return adjusted_keywords
