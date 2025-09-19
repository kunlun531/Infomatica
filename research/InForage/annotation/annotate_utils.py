import streamlit as st
import asyncio
import time
from typing import List, Dict, Any, Optional
import sys
import random
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from topic_search import TopicSearcher
from my_own_tools import *
from prompts import *
from build_db import *
from semantic_text_splitter import TextSplitter
from utils import parse_claim_format


chunk_size = 1000
spliter = TextSplitter.from_tiktoken_model("gpt-3.5-turbo", chunk_size)

def get_api_dict():
    return load_json("~/InForage/data/api_config.json")

@st.cache_resource
def get_gpt_agent():
    return Agent(
        source="openrouter",
        model="openai/gpt-4o",
        api_dict=get_api_dict(),
    )

@st.cache_data
def load_topic_list():
    print("loading topic list")
    return load_jsonl("~/InForage/data/target_topic.jsonl")

@st.cache_resource
def get_bm25_index():
    print("loading bm25 index")
    return TopicSearcher()

@st.cache_resource
def get_qwen_agent():
    print("loading qwen agent")
    return Agent(
        source="vllm",
        model="Qwen2.5-72B-Instruct",
        base_url="localhost:28706/chat/v1",
    )

@st.cache_resource
def get_tok():
    print("loading tokenizer")
    return get_tokenizer()

@st.cache_resource
def get_db():
    print("loading db")
    return sqlite3.connect("~/InForage/data/search_db.db")

def init_annotate_utils():
    topic_list = load_topic_list()
    gpt_agent = get_gpt_agent()
    qwen_agent = get_gpt_agent()
    tokenizer = get_tok()
    bm25_index = get_bm25_index()
    db = get_db()   
    return topic_list, gpt_agent, qwen_agent, tokenizer, bm25_index, db


def translate_text(agent, text: str, target_language: str="zh-cn") -> str:
    prompt = f"Translate the following text to {target_language}: {text}, only output the translated text."
    return stream_output(agent, prompt, max_completion_tokens=512, format_json=False)

def stream_output(agent, prompt: str, max_completion_tokens: int=512, format_json: bool=True):
    placeholder = st.empty()
    full_response = ""
    messages = [    
        {"role": "user", "content": prompt}
    ]
    for token in agent.stream_completion(messages, max_completion_tokens=max_completion_tokens):
        full_response += token
        if format_json:
            placeholder.markdown(f"```json\n{full_response}\n```")
        else:
            placeholder.markdown(full_response)
    return full_response

def get_claim(content, agent, max_prompt:int=10):
    content = spliter.chunks(content)
    prompts = []
    for chunk in content:
        prompt = claim_prompt.format(context=chunk)
        prompts.append(prompt)
    prompts = prompts[:max_prompt]
    responses = agent.batch_completion(prompts)
    claims = []
    for response in responses:
        parsed_claim = parse_claim_format(response)
        # Skip empty values from parsed claims
        if not all(parsed_claim.values()):
            continue
        # Add all non-empty key-value pairs
        claims.append(parsed_claim)
    return claims

def fetch_google_page(result, agent):
    title = result['title']
    snippet = result['snippet']
    url = result['link']
    # Check if the URL content exists in the database
    content = get_page_by_url(url)
    
    if content:
        # Content exists in database
        content = content['content']
        print(f"Content exists in database: {url}")
    else:
        # Content doesn't exist, fetch and store it
        content = extract_text_from_url(url, jina_api_key=get_api_dict()["jina"]["api_key"])
        date = datetime.now().strftime("%Y-%m-%d")
        insert_or_update_page(url, title, date, snippet, content)
        print(f"Content fetched and stored in database: {url}")
    return url,content

def fetch_claim(content, agent, url):

    claims = get_claim_by_url(url)
    if claims:
        claims = json.loads(claims[1])
        print(f"Claims exist in database: {url}")
    else:
        claims = get_claim(content, agent)
        insert_or_update_claim(url, claims)
        print(f"Claims fetched and stored in database: {url}")

    return claims

def display_google_results(results, agent):
    """Display Google search results with selection buttons"""
    colors = ["#f0f8ff", "#e6e6fa", "#f0fff0", "#fff0f5", "#f5f5dc", "#f0ffff"]
    width = st.session_state.get("width_slider", 3)
    
    for i in range(0, len(results), width):
        cols = st.columns(width)
        for j in range(width):
            if i + j < len(results):
                result = results[i + j]
                result_index = i + j + 1
                with cols[j]:
                    card_color = colors[(i+j) % len(colors)]
                    st.markdown(f"""
                    <div style="background-color: {card_color}; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                        <h3 style="color: #333;">Result #{result_index}</h3>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"<h4 style='font-size: 18px;'>{result['title']}</h4>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size: 16px;'>{result['snippet']}</p>", unsafe_allow_html=True)
                    url = result['link']
                    
                    st.markdown("</div>", unsafe_allow_html=True)

def select_google_result(results):
    width = st.session_state.get("width_slider", 3)
    for i in range(0, len(results), width):
        cols = st.columns(width)
        for j in range(width):
            if i + j < len(results):
                result = results[i + j]
                result_index = i + j + 1
                with cols[j]:
                    if st.button("Use this Result", key=f"select_google_{i+j}"):
                        st.session_state.web_to_claim = result
                        st.session_state.states = "extract"
                        st.rerun()

def extract_claim(result, agent):
    url, content = fetch_google_page(result, agent)
    claims = fetch_claim(content, agent, url)
    # Display claims with checkboxes for selection
    
    st.subheader("Claims extracted from the webpage")
    if not claims:
        st.warning("No valid claims extracted from the webpage")
        return claims
    
    rtn = []
    for claim in claims:
        topic = claim.get('claim_topic', '')
        claim_text = claim.get('claims', '')
        target = claim.get('claim_target', '')
        evidence = claim.get('evidence', '')
        rtn.append({
            'topic': topic,
            'claim': claim_text,
            'target': target,
            'evidence': evidence
        })
    return rtn

def display_news_results(results):
    """Display news search results with selection buttons"""
    if len(results) == 0:
        st.markdown("No relevant news found")
        return
    colors = ["#f0f8ff", "#e6e6fa", "#f0fff0", "#fff0f5", "#f5f5dc", "#f0ffff"]
    width = st.session_state.get("width_slider", 3)
    
    for i in range(0, len(results), width):
        cols = st.columns(width)
        for j in range(width):
            if i + j < len(results):
                result = results[i + j]
                result_index = i + j + 1
                with cols[j]:
                    card_color = colors[(i+j) % len(colors)]
                    st.markdown(f"""
                    <div style="background-color: {card_color}; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                        <h3 style="color: #333;">Result #{result_index}</h3>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"<h4 style='font-size: 18px;'>{result['title']}</h4>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size: 16px;'>{result['description']}</p>", unsafe_allow_html=True)
                    if 'image' in result and result['image']:
                        st.image(result['image'], width=150)
                    st.markdown(f"<p style='font-size: 16px;'><b>Source:</b> {result['source']}</p>", unsafe_allow_html=True)
                    url = result['url']
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Selection checkbox
                    key = f"select_news_{i+j}"
                    if st.checkbox("Use this Result", key=key):
                        pass

def display_cache_results(results, qwen_agent):
    """Display BM25 cache search results with selection buttons"""
    if len(results) == 0:
        st.markdown("No relevant cache found")
        return
    colors = ["#f0f8ff", "#e6e6fa", "#f0fff0", "#fff0f5", "#f5f5dc", "#f0ffff"]
    if len(results) <= 5:
        width = len(results)
    else:
        width = 5
    for i in range(0, len(results), width):
        cols = st.columns(width)
        for j in range(width):
            if i + j < len(results):
                result = results[i + j]
                result_index = i + j + 1
                with cols[j]:
                    card_color = colors[(i+j) % len(colors)]
                    st.markdown(f"""
                    <div style="background-color: {card_color}; padding: 15px; border-radius: 10px; margin-bottom: 15px;">
                        <h3 style="color: #333;">Claim #{result_index}</h3>
                    """, unsafe_allow_html=True)
                    

                    st.markdown(f"<h4 style='font-size: 18px;'>Topic: {result.get('topic', '')}</h4>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size: 16px;'>Claim: {result.get('claim', '')}</p>", unsafe_allow_html=True)

                    st.markdown(f"<p style='font-size: 16px;'>Target: {result.get('target', '')}</p>", unsafe_allow_html=True)
                    url = result.get('url', '')
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    # Selection checkbox
def select_cache_result(results, key_name):
    if len(results) <= 5:
        width = len(results)
    else:
        width = 5
    for i in range(0, len(results), width):
        cols = st.columns(width)
        for j in range(width):
            if i + j < len(results):
                result = results[i + j]
                with cols[j]:
                    key = f"select_cache_{key_name}_{i+j}"
                    if st.button(f"Claim #{i+j+1}", key=key):
                        st.session_state.current_record = result
                        st.session_state.records.append(result)
                        st.success(f"Claim #{i+j+1} added to records")
                        st.session_state.states = "search"
                        st.rerun()


