import requests
import http.client
import urllib.parse
import json
import re

conn = http.client.HTTPConnection('api.mediastack.com')

def search_news(keywords):
    params = urllib.parse.urlencode({
        'access_key': 'api_key',
        'keywords': keywords,
        'sort': 'published_desc',
        'languages': 'en, -zh,-ar,-de,-es,-fr,-it,-nl,-no,-pt,-ru',
        'limit': 100,
        })
    conn.request('GET', '/v1/news?{}'.format(params))

    res = conn.getresponse()
    data = res.read()
    decoded_data = data.decode('utf-8')
    try:
        json_data = json.loads(decoded_data)
        # if error 
        if 'error' in json_data.keys():
            print(f"API ERROR: {json_data}")
            return False
        # print(json_data)
        data_list = json_data['data']
        return data_list

    except json.JSONDecodeError as e:
        print(f"JSON ERROR: {e}")
        return False

def search_google(query):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": "api_key",
        "cx": ""
    }   
    response_json = requests.get(url, params=params).json()
    response_items = response_json["items"]
    rtn = []
    for item in response_items:
        rtn.append({
            "title": item["title"],
            "link": item["link"],
            "snippet": item["snippet"]
        })
    return rtn

def parse_claim_format(text):
    """
    Parse a claim format text and return a structured dictionary using regular expressions.
    
    Example input:
    ##Evidence: Elon Musk: "As a function of the great policies of President Trump and his administration, and as an act of faith in America, Tesla is going to DOUBLE vehicle output in the United States within the next two years…"
    ##Claims: Tesla will double its vehicle output in the United States within the next two years due to the policies of President Trump and his administration.
    ##Claim Target: Tesla
    ##Claim Topic: vehicle output increase
    
    Returns:
    {
        "evidence": "Elon Musk: \"As a function of the great policies of President Trump and his administration, and as an act of faith in America, Tesla is going to DOUBLE vehicle output in the United States within the next two years…\"",
        "claims": "Tesla will double its vehicle output in the United States within the next two years due to the policies of President Trump and his administration.",
        "claim_target": "Tesla",
        "claim_topic": "vehicle output increase"
    }
    """
    import re
    
    result = {}
    
    # Define patterns for each section
    patterns = {
        "evidence": r"##Evidence:\s*(.*?)(?=##|\Z)",
        "claims": r"##Claims:\s*(.*?)(?=##|\Z)",
        "claim_target": r"##Claim Target:\s*(.*?)(?=##|\Z)",
        "claim_topic": r"##Claim Topic:\s*(.*?)(?=##|\Z)"
    }
    
    # Extract each section using regex
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)
        if match:
            result[key] = match.group(1).strip()
    
    return result


def parse_json_output(text):
    """
    Parse potentially irregular JSON output from LLMs.
    
    This function attempts to extract and parse JSON from text that might contain
    additional content or have formatting issues.
    
    Args:
        text (str): The text containing JSON output from an LLM.
        
    Returns:
        dict or list: The parsed JSON object, or None if parsing fails.
    """
    
    
    # First try direct JSON parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON-like content using regex
    json_pattern = r'(\{.*\}|\[.*\])'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    if matches:
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    # Try to fix common JSON formatting issues
    # Replace single quotes with double quotes
    fixed_text = re.sub(r"'([^']*)'", r'"\1"', text)
    # Fix unquoted keys
    fixed_text = re.sub(r'(\w+):', r'"\1":', fixed_text)
    
    # Try to extract and parse the fixed JSON
    matches = re.findall(json_pattern, fixed_text, re.DOTALL)
    if matches:
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    return None


if __name__ == "__main__":
    print(search_google("What role does gut microbiota play in human health and disease, and how can it be modulated for therapeutic benefits?"))
