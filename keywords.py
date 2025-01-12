expertise_keywords = {
    "Python": [
        # Basic Syntax
        'lambda', 'class', 'self', 'yield', 'with', 'return', 'async', 'await', 'for', 'in', 'if', 'elif', 'else',
        'try', 'except', 'finally', 'import', 'from', 'def', 'break', 'continue', 'pass', 'global', 'nonlocal',
        # Built-in Functions
        'print', 'len', 'range', 'map', 'filter', 'reduce', 'sum', 'sorted', 'zip', 'all', 'any', 'min', 'max',
        'abs', 'round', 'open', 'read', 'write', 'input', 'dir', 'help', 'isinstance', 'type', 'id', 'hash',
        # Libraries
        'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'keras', 'pytorch', 'flask', 'django', 'fastapi', 'bokeh',
        'matplotlib', 'seaborn', 'plotly', 'dash', 'sqlalchemy', 'beautifulsoup', 'requests', 'aiohttp', 'scrapy',
        'spacy', 'nltk', 'transformers', 'pydantic', 'typer', 'click', 'argparse', 'pytest', 'pytest-mock',
        'pytest-django',
        'sqlmodel', 'celery', 'gunicorn', 'uvicorn', 'flask-restful', 'asyncpg', 'opencv', 'pyautogui',
        # Functions (specific libraries)
        'pd.dataframe', 'pd.series', 'np.array', 'np.dot', 'np.linalg.inv', 'plt.plot', 'plt.show', 'sns.heatmap',
        'sns.pairplot', 'tf.keras.model', 'torch.nn.linear', 'flask.jsonify', 'flask.request', 'django.urls.path',
        'django.shortcuts.render', 'sqlalchemy.create_engine', 'sqlalchemy.orm.sessionmaker', 'requests.get',
        'requests.post', 'bs4.beautifulsoup', 'spacy.load', 'spacy.tokenizer', 'transformers.automodel', 'nltk',
        'ollama', 'requests'
    ],
    "Java": [
        # Basic Syntax
        'public', 'class', 'static', 'final', 'extends', 'implements', 'try', 'catch', 'throw', 'synchronized',
        'interface', 'abstract', 'enum', 'default', 'switch', 'case', 'while', 'do', 'for', 'if', 'else', 'private',
        'protected', 'package', 'import', 'instanceof', 'void', 'return',
        # Libraries
        'spring', 'spring boot', 'hibernate', 'jpa', 'thymeleaf', 'apache kafka', 'jackson', 'gson', 'lombok',
        'junit', 'mockito', 'slf4j', 'log4j', 'vert.x', 'quarkus', 'micronaut', 'servlets', 'tomcat', 'jetty', 'jsp',
        # Functions (specific libraries)
        'springapplication.run', 'restcontroller', 'requestmapping', 'responseentity', 'entitymanager.persist',
        'sessionfactory.opensession', 'log.debug', 'mockito.when', 'junit.assertequals',
    ],
    "JavaScript": [
        # Basic Syntax
        'function', 'let', 'const', 'var', 'if', 'else', 'switch', 'case', 'while', 'do', 'for', 'this', 'new',
        'try', 'catch', 'finally', 'throw', 'return', 'class', 'extends', 'import', 'export', 'default',
        # Libraries
        'react', 'redux', 'angular', 'vue', 'node.js', 'express', 'axios', 'lodash', 'moment.js', 'd3.js',
        'three.js', 'chart.js', 'socket.io', 'webpack', 'babel', 'jest', 'cypress', 'next.js', 'nuxt.js',
        'tailwindcss', 'material-ui', 'bootstrap', 'ant design', 'semantic-ui',
        # Functions (specific libraries)
        'usestate', 'useeffect', 'usereducer', 'usecontext', 'useref', 'react.createelement', 'reactdom.render',
        'react.memo', 'axios.get', 'axios.post', 'redux.createstore', 'redux.combinereducers', 'angular.module',
        '$scope.$apply', '$http.get', 'vue.component', 'vue.observable', 'express.router', 'res.json', 'res.send',
        'socket.emit', 'socket.on',
    ],
    "C++": [
        # Basic Syntax
        'namespace', 'class', 'public', 'private', 'protected', 'virtual', 'override', 'static', 'template',
        'try', 'catch', 'throw', 'new', 'delete', 'friend', 'inline', 'const', 'constexpr', 'volatile', 'final',
        # Libraries
        'stl', 'boost', 'qt', 'poco', 'opencv', 'eigen', 'tbb', 'cgal', 'abseil',
        # Functions (specific libraries)
        'std::vector', 'std::map', 'std::sort', 'std::find', 'qtwidgets.qapplication', 'opencv.imread',
        'opencv.imshow', 'boost.bind', 'boost.thread', 'eigen.matrix',
    ],
    "Ruby": [
        'def', 'end', 'class', 'module', 'self', 'yield', 'lambda', 'proc', 'if', 'else', 'elsif', 'while', 'until',
        'begin', 'rescue', 'ensure', 'block', 'rails', 'activerecord', 'enumerable', 'map', 'select', 'inject', 'proc',
        'instance_eval', 'rspec', 'rails 5', 'sinatra', 'padrino', 'hanami', 'activejob', 'sidekiq', 'capybara', 'pry',
        'delorean'
    ],
    "Go": [
        'package', 'import', 'func', 'var', 'const', 'defer', 'panic', 'recover', 'go', 'goroutine', 'select',
        'channel',
        'struct', 'interface', 'map', 'slice', 'range', 'fallthrough', 'if', 'else', 'switch', 'type', 'receive',
        'send',
        'gin', 'beego', 'echo', 'revel', 'gorm', 'go-kit', 'go-micro', 'testify', 'cobra', 'cobra-cli'
    ],
    "C#": [
        'namespace', 'using', 'public', 'private', 'protected', 'class', 'interface', 'abstract', 'static', 'delegate',
        'event', 'readonly', 'virtual', 'override', 'new', 'base', 'this', 'try', 'catch', 'finally', 'async', 'await',
        'linq', 'entity framework', 'asp.net', 'xamarin', 'unity', 'signalr', 'azure', 'nunit', 'moq', 'blazor', 'wpf'
    ],
    "PHP": [
        'function', 'class', 'public', 'private', 'protected', 'static', 'abstract', 'interface', 'namespace', 'trait',
        'namespace', 'foreach', 'isset', 'empty', 'global', 'require', 'include', 'echo', 'var_dump', 'serialize',
        'laravel',
        'symfony', 'wordpress', 'composer', 'phalcon', 'zend framework', 'phpunit', 'twig', 'slim', 'guzzle'
    ],
    "Swift": [
        'class', 'struct', 'enum', 'protocol', 'extension', 'func', 'var', 'let', 'guard', 'defer', 'async', 'await',
        'do', 'try', 'catch', 'throw', 'nil', 'closure', 'typealias', 'lazy', 'optional', 'map', 'filter', 'reduce',
        'uikit',
        'swiftui', 'coredata', 'combine', 'cocoa', 'alamofire', 'firebase', 'realm', 'carthage', 'cocoapods'
    ],
    "Kotlin": [
        'val', 'var', 'fun', 'object', 'class', 'interface', 'override', 'sealed', 'data class', 'package', 'try',
        'catch',
        'finally', 'null safety', 'extension function', 'lambda', 'coroutine', 'async', 'await', 'runblocking',
        'companion object',
        'spring boot', 'ktor', 'exposed', 'mockito', 'junit', 'room', 'coroutines', 'firebase', 'anko'
    ],
    "TypeScript": [
        'function', 'let', 'const', 'class', 'interface', 'extends', 'implements', 'async', 'await', 'constructor',
        'type',
        'union type', 'intersection type', 'type alias', 'enum', 'tuple', 'generic', 'namespace', 'module', 'import',
        'export', 'react', 'node.js', 'nestjs', 'express', 'redux', 'typeorm', 'graphql', 'jest'
    ]
}

topics_keywords = [
    # Data Structures
    'data structures', 'linked list', 'singly linked list', 'doubly linked list', 'circular linked list', 'stack',
    'queue', 'deque', 'priority queue', 'heap', 'min-heap', 'max-heap', 'binary search tree', 'avl tree',
    'red-black tree', 'b-tree', 'trie', 'suffix tree', 'suffix array', 'disjoint set', 'union-find', 'segment tree',
    'fenwick tree', 'hash table', 'hash map', 'hash set', 'graph', 'adjacency list', 'adjacency matrix',
    'incidence matrix',

    # Graph Algorithms
    'dfs', 'bfs', 'dijkstra', 'bellman-ford', 'floyd-warshall', 'johnson’s algorithm', 'kruskal', 'prims',
    'topological sort', 'strongly connected components', 'kosaraju’s algorithm', 'tarjan’s algorithm',
    'bridges in graph', 'articulation points', 'eulerian path', 'hamiltonian path', 'shortest path algorithms',
    'minimum spanning tree', 'network flow', 'ford-fulkerson', 'edmonds-karp', 'dinic’s algorithm',

    # Sorting Algorithms
    'sorting algorithms', 'bubble sort', 'selection sort', 'insertion sort', 'merge sort', 'quick sort',
    'heap sort', 'radix sort', 'counting sort', 'bucket sort', 'shell sort', 'tim sort',

    # Searching Algorithms
    'binary search', 'linear search', 'ternary search', 'exponential search', 'jump search', 'interpolation search',

    # Algorithmic Paradigms
    'dynamic programming', 'divide and conquer', 'backtracking', 'greedy algorithms', 'branch and bound',
    'recursion', 'memoization',

    # String Algorithms
    'kmp algorithm', 'z-algorithm', 'rabin-karp', 'aho-corasick', 'manacher’s algorithm',
    'longest common subsequence', 'longest palindromic substring',

    # Computational Geometry
    'convex hull', 'graham’s scan', 'jarvis march', 'closest pair of points', 'line segment intersection',
    'polygon triangulation',

    # Number Theory
    'gcd', 'lcm', 'modular arithmetic', 'modular exponentiation', 'sieve of eratosthenes', 'prime factorization',
    'euler’s totient function', 'fermat’s little theorem', 'chinese remainder theorem',

    # Advanced Concepts
    'big-o notation', 'time complexity', 'space complexity', 'amortized analysis', 'randomized algorithms',
    'probabilistic algorithms', 'parallel algorithms', 'distributed algorithms', 'approximation algorithms',

    # Machine Learning
    'supervised learning', 'unsupervised learning', 'reinforcement learning', 'linear regression',
    'logistic regression',
    'decision trees', 'random forests', 'support vector machines', 'naive bayes', 'k-nearest neighbors',
    'k-means clustering', 'pca', 'lda', 'neural networks', 'deep learning', 'convolutional neural networks',
    'recurrent neural networks', 'transformers', 'natural language processing', 'tokenization', 'stemming',
    'lemmatization',

    # Databases
    'relational database', 'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'oracle', 'sql server',
    'database normalization', 'acid properties', 'database indexing', 'stored procedures', 'triggers',
    'entity-relationship model', 'jpa', 'hibernate', 'optimistic locking', 'pessimistic locking',

    # Web Development
    'frontend development', 'backend development', 'api development', 'restful apis', 'graphql', 'jwt authentication',
    'session management', 'cross-origin resource sharing', 'http methods', 'css frameworks', 'responsive design',
    'progressive web apps', 'web sockets', 'server-side rendering', 'client-side rendering',

    # Cloud and DevOps
    'aws', 'azure', 'google cloud', 'docker', 'kubernetes', 'terraform', 'ansible', 'jenkins', 'ci/cd pipelines',
    'cloudformation', 'helm charts', 'monitoring and logging', 'prometheus', 'grafana', 'splunk',

    # Operating Systems
    'process scheduling', 'threads', 'mutex', 'semaphores', 'deadlocks', 'memory management', 'paging', 'segmentation',
    'file systems', 'virtual memory', 'i/o management', 'scheduling algorithms', 'round robin', 'shortest job next',

    # Software Engineering
    'agile methodology', 'scrum', 'kanban', 'waterfall model', 'spiral model', 'v-model', 'software testing',
    'unit testing', 'integration testing', 'system testing', 'acceptance testing', 'regression testing',
    'test-driven development', 'behavior-driven development',

    # Cybersecurity
    'cryptography', 'hashing algorithms', 'symmetric encryption', 'asymmetric encryption', 'rsa', 'aes',
    'diffie-hellman', 'firewalls', 'vpn', 'penetration testing', 'vulnerability assessment', 'sql injection',
    'cross-site scripting', 'csrf attacks', 'zero-day exploits',

    # Other
    'artificial intelligence', 'internet of things', 'blockchain', 'game development', 'unity', 'unreal engine',
    'robotics', 'embedded systems', 'data visualization', 'data cleaning', 'etl pipelines'
]
