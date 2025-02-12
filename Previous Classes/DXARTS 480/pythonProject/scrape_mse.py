import re
import requests
from nltk.tokenize import sent_tokenize
import string

def get_obvious_answers_math(max_pages=5, page_size=100):
    url = "https://api.stackexchange.com/2.3/search/advanced"
    all_bodies = []

    for page in range(1, max_pages + 1):
        params = {
            "order": "desc",
            "sort": "relevance",
            "q": "obvious easy trivial",  # Updated to include multiple keywords
            "site": "math.stackexchange",
            "is_answer": True,
            "filter": "withbody",
            "page": page,
            "pagesize": page_size
        }

        response = requests.get(url, params=params)
        response.raise_for_status()  # Raises an error if the request failed
        data = response.json()

        # Extract and clean the 'body' field from each item
        bodies = [clean_stack_exchange_post(item["body"]) for item in data.get("items", [])]

        # Filter the cleaned bodies based on your custom filter
        filtered_bodies = [body for body in bodies if body_is_rude(body)]

        # Add the filtered bodies to the overall list
        all_bodies.extend(filtered_bodies)

        # Check if there are more pages available
        if not data.get('has_more', False):
            break  # Stop if there are no more results

    sentences = [replace_punctuation_with_space(sentence) for body in all_bodies for sentence in sent_tokenize(body)]
    return sentences

def body_is_rude(body):
    return body.count("obvious") >= 3 or body.count("trivial") >= 3 or body.count("easy") >= 3

def replace_punctuation_with_space(text):
    # Convert to lowercase
    text = text.lower()

    # Replace punctuation (including backticks and single quotes) with spaces
    text = re.sub(rf"[{re.escape(string.punctuation + '``''')}]", ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Function to clean and split text into sentences using NLTK
def clean_stack_exchange_post(post):
    # Remove everything between <...>
    no_tags = re.sub(r'<[^>]*>', '', post)

    # Remove LaTeX expressions between $$...$$, $...$, or \[...\]
    no_latex = re.sub(r'(\$\$.*?\$\$|\$.*?\$|\\\[.*?\\])', '', no_tags, flags=re.DOTALL)

    # Remove all new lines and replace with spaces
    no_newlines = re.sub(r'\s+', ' ', no_latex).strip()

    # Remove invalid characters that might cause encoding issues
    clean_text = re.sub(r'[^\x00-\x7F]+', '', no_newlines)  # Removes non-ASCII characters

    return clean_text

# Get the list of bodies containing 'obvious' in answers on Math Stack Exchange
sentences = get_obvious_answers_math()
with open("sentences.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(sentences))
