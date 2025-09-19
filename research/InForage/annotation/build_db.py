import sqlite3
from my_own_tools import *
from datetime import datetime
import os
db_path = "search_db.db"
from tqdm import tqdm
import json

def build_search_tables():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create Google search table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS google_search (
            query TEXT PRIMARY KEY,
            result TEXT,
            timestamp TEXT
        )
    """)
    
    # Create News search table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS news_search (
            query TEXT PRIMARY KEY,
            result TEXT,
            timestamp TEXT
        )
    """)
    
    conn.commit()
    conn.close()

def build_claim_data_table():
    """
    Create the claim_data table to store claims with their associated metadata.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create claim_data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS claim_data (
            claim TEXT PRIMARY KEY,
            topic TEXT,
            evidence TEXT,
            target TEXT,
            url TEXT,
            category TEXT
        )
    """)
    
    conn.commit()
    conn.close()

def insert_claim_data(claim, topic, evidence, target, url, category=""):
    """
    Insert claim data into the database.
    
    Args:
        claim (str): The claim statement
        topic (str): The topic of the claim
        evidence (str): The evidence supporting the claim
        target (str): The target of the claim
        url (str): The source URL
        category (str): The category of the claim
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute(
        "INSERT OR REPLACE INTO claim_data (claim, topic, evidence, target, url, category) VALUES (?, ?, ?, ?, ?, ?)",
        (claim, topic, evidence, target, url, category)
    )
    
    conn.commit()
    conn.close()

def get_category_from_webpage(url):
    """
    Get the category for a URL from the web_page table.
    
    Args:
        url (str): The URL to look up
        
    Returns:
        str: The category if found, empty string otherwise
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT category FROM web_page WHERE url = ?", (url,))
    result = cursor.fetchone()
    
    conn.close()
    
    if result and result[0]:
        return result[0]
    return ""

def get_random_claims(num=5, category=None):
    """
    Get random claims from the claim_data table.
    
    Args:
        num (int): Number of random claims to retrieve, defaults to 5
        category (str): The category of the claims to retrieve, defaults to None
    Returns:
        list: List of dictionaries containing claim data
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    if category:
        cursor.execute("SELECT claim, topic, evidence, target, url, category FROM claim_data WHERE category = ? ORDER BY RANDOM() LIMIT ?", (category, num))
    else:
        cursor.execute("SELECT claim, topic, evidence, target, url, category FROM claim_data ORDER BY RANDOM() LIMIT ?", (num,))
    rows = cursor.fetchall()
    
    results = []
    for row in rows:
        results.append({
            "claim": row[0],
            "topic": row[1], 
            "evidence": row[2],
            "target": row[3],
            "url": row[4],
            "category": row[5]
        })
    
    conn.close()
    return results

def populate_claim_data_from_jsonl():
    """
    Populate the claim_data table with data from the target_topic.jsonl file.
    """
    build_claim_data_table()
    
    data_path = "~/InForage/data/target_topic.jsonl"
    claims_data = load_jsonl(data_path)
    
    print(f"Loading {len(claims_data)} claims into database...")
    for item in tqdm(claims_data):
        claim = item.get("claim", "")
        topic = item.get("topic", "")
        evidence = item.get("evidence", "")
        target = item.get("target", "")
        url = item.get("url", "")
        
        # Get category from web_page table if available
        category = get_category_from_webpage(url)
        
        insert_claim_data(claim, topic, evidence, target, url, category)
    
    print("Claim data population completed.")

def build_annotated_data_table():
    """
    Create the annotated_data table to store annotated questions and answers.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create annotated_data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS annotated_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            answer TEXT,
            annotator TEXT,
            records TEXT,
            timestamp TEXT
        )
    """)
    
    conn.commit()
    conn.close()

def insert_annotated_data(query, answer, annotator, records):
    """
    Insert annotated data into the database.
    
    Args:
        query (str): The multi-hop query
        answer (str): The answer to the query
        annotator (str): The name of the annotator
        records (list): The records used to generate the query
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Convert records to JSON string
    records_json = json.dumps(records)
    timestamp = datetime.now().isoformat()
    
    cursor.execute(
        "INSERT INTO annotated_data (query, answer, annotator, records, timestamp) VALUES (?, ?, ?, ?, ?)",
        (query, answer, annotator, records_json, timestamp)
    )
    
    conn.commit()
    conn.close()

def get_annotation_stats():
    """
    Get statistics about annotations grouped by annotator.
    
    Returns:
        dict: A dictionary with annotator names as keys and their annotation counts as values
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT annotator, COUNT(*) as count 
        FROM annotated_data 
        GROUP BY annotator
        ORDER BY count DESC
    """)
    
    results = cursor.fetchall()
    conn.close()
    
    stats = {}
    for annotator, count in results:
        stats[annotator] = count
    
    return stats

def insert_or_update_search(search_type, query, results):
    """
    Insert or update search results in the database.
    
    Args:
        search_type (str): Type of search ('google' or 'news')
        query (str): The search query
        results (list): The search results
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Convert results to JSON string
    results_json = json.dumps(results)
    timestamp = datetime.now().isoformat()
    
    table_name = f"{search_type}_search"
    
    cursor.execute(f"SELECT * FROM {table_name} WHERE query = ?", (query,))
    result = cursor.fetchone()
    if result:
        cursor.execute(f"UPDATE {table_name} SET result = ?, timestamp = ? WHERE query = ?", 
                      (results_json, timestamp, query))
    else:
        cursor.execute(f"INSERT INTO {table_name} (query, result, timestamp) VALUES (?, ?, ?)", 
                      (query, results_json, timestamp))
    
    conn.commit()
    conn.close()

def query_search(search_type, query):
    """
    Query search results from the database.
    
    Args:
        search_type (str): Type of search ('google' or 'news')
        query (str): The search query
        
    Returns:
        list or None: The search results if found, None otherwise
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    table_name = f"{search_type}_search"
    
    cursor.execute(f"SELECT result FROM {table_name} WHERE query = ?", (query,))
    result = cursor.fetchone()
    
    conn.close()
    
    if result:
        return json.loads(result[0])
    else:
        return None


def build_web_page_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # cursor.execute("DROP TABLE IF EXISTS web_page")
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS web_page (
            url TEXT PRIMARY KEY,
            title TEXT,
            date TEXT,
            snippet TEXT,
            content TEXT,
            source TEXT,
            language TEXT,
            category TEXT,
            country TEXT
        )
    """)

    conn.commit()
    conn.close()

def build_claim_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS claim (
            url TEXT PRIMARY KEY,
            claims_json TEXT,
            FOREIGN KEY (url) REFERENCES web_page(url)
        )
    """)
    conn.commit()
    conn.close()


def insert_or_update_claim(url, parsed_claims):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Convert parsed claims to JSON string
    claims_json = json.dumps(parsed_claims)
    
    cursor.execute("SELECT * FROM claim WHERE url = ?", (url,))
    result = cursor.fetchone()
    if result:
        cursor.execute("UPDATE claim SET claims_json = ? WHERE url = ?", (claims_json, url))
    else:
        cursor.execute("INSERT INTO claim (url, claims_json) VALUES (?, ?)", (url, claims_json))
    conn.commit()
    conn.close()
    
def get_claim_by_url(url):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM claim WHERE url = ?", (url,))
    result = cursor.fetchone()
    conn.close()
    return result

def get_page_by_url(url):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT url, title, date, snippet, content FROM web_page WHERE url = ?", (url,))
    result = cursor.fetchone()
    
    conn.close()
    
    if result:
        return {
            "url": result[0],
            "title": result[1],
            "date": result[2],
            "snippet": result[3],
            "content": result[4]
        }
    return None

def insert_or_update_page(url, title, date, snippet, content, source="", language="", category="", country=""):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if the URL already exists
    cursor.execute("SELECT date FROM web_page WHERE url = ?", (url,))
    existing = cursor.fetchone()
    
    if existing is None:
        # Insert new record
        cursor.execute("""
            INSERT INTO web_page (url, title, date, snippet, content, source, language, category, country)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (url, title, date, snippet, content, source, language, category, country))
    elif date:
        existing_date = existing[0]
        # Update if the new date is more recent
        # Convert dates to datetime objects for comparison
        existing_datetime = datetime.datetime.fromisoformat(existing_date) if existing_date else None
        new_datetime = datetime.datetime.fromisoformat(date) if date else None
        
        if not existing_datetime or (existing_datetime and new_datetime and new_datetime > existing_datetime):
            cursor.execute("""
                UPDATE web_page
                SET title = ?, date = ?, snippet = ?, content = ?, source = ?, language = ?, category = ?, country = ?
                WHERE url = ?
            """, (title, date, snippet, content, source, language, category, country, url))
    
    conn.commit()
    conn.close()

def add_data_web_page_to_db():
    url_to_content = load_jsonl("merged_data.jsonl")
    url_to_content = {item["url"]: item["content"] for item in url_to_content}
    data_path = "data/api"
    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)
        print(f"process file: {file}")
        try:
            data = load_json(file_path)
        except:
            data = load_jsonl(file_path)

        for item in tqdm(data):
            try:
                url = item["url"]
                title = item["title"]
                date = item["published_at"]
                snippet = item["description"]
                content = url_to_content[url]
                source = item["source"]
                language = item["language"]
                category = item["category"]
                country = item["country"]   
                insert_or_update_page(url, title, date, snippet, content, source, language, category, country)
            except Exception as e:
                print(f"process url: {url} error: {e}")

def get_claim_categories():
    """
    Get all unique categories from the claim_data table.
    
    Returns:
        list: List of unique categories
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT DISTINCT category FROM claim_data WHERE category != '' AND category IS NOT NULL")
    categories = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    return categories

if __name__ == "__main__":

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Count claims with empty category
    cursor.execute("SELECT COUNT(*) FROM claim_data WHERE category = '' OR category IS NULL")
    empty_count = cursor.fetchone()[0]
    print(f"\nClaims with empty category: {empty_count}")

    # Count claims by category
    cursor.execute("""
        SELECT category, COUNT(*) as count 
        FROM claim_data 
        WHERE category != '' AND category IS NOT NULL
        GROUP BY category
        ORDER BY count DESC
    """)
    
    print("\nClaims by category:")
    for category, count in cursor.fetchall():
        print(f"{category}: {count}")
    
    cursor.execute("SELECT COUNT(*) FROM claim_data")
    count = cursor.fetchone()
    print(count)
    
    conn.close()