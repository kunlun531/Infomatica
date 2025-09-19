import streamlit as st
import asyncio
import time
from typing import List, Dict, Any, Optional
import sys
import random
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import search_google, search_news, parse_claim_format
from topic_search import TopicSearcher
from my_own_tools import *
from prompts import *
from process.build_db import *
from annotate_utils import *


def main():
    st.set_page_config(page_title="Deep Search Annotation System", layout="wide")
    topic_list, gpt_agent, qwen_agent, tokenizer, bm25_index, db = init_annotate_utils()


    user_name = "hongjin" #TODO
    width = st.sidebar.slider("Select width", min_value=1, max_value=10, value=3, key="width_slider", help="Adjust display width")

    # Initialize session state for records if not exists
    if 'records' not in st.session_state:
        st.session_state.records = []
        st.session_state.current_record = None
        st.session_state.search_query = None
        st.session_state.states = "sample"
        st.session_state.selected_records = []
        st.session_state.final_question = None
        st.session_state.web_to_claim = None
        st.session_state.save_records = None

    # Create a sidebar section to display records
    with st.sidebar:
        st.subheader("Added Records")
        if len(st.session_state.records) == 0:
            st.info("No records yet")
        else:
            for i, record in enumerate(st.session_state.records):
                with st.expander(f"Record {i+1}: {record['topic'][:30]}...", expanded=False):
                    st.markdown(f"**Topic:** {record['topic']}")
                    st.markdown(f"**Claim:** {record['claim']}")
                    if 'target' in record:
                        st.markdown(f"**Target:** {record['target']}")

    # Add buttons to generate final question and clear all records
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Generate Question", key="generate_final_button", type="primary"):
            st.session_state.states = "generate"
            st.rerun()
    with col2:
        if st.button("Clear All Records", key="clear_records_button", type="secondary"):
            st.session_state.records = []
            st.session_state.current_record = None
            st.session_state.search_query = None
            st.session_state.states = "sample"
            st.session_state.selected_records = []
            st.session_state.save_records = None
            st.sidebar.success("All records cleared")
            st.rerun()

    # Add a refresh button to reset the state to search
    if st.sidebar.button("Refresh", key="refresh_button"):
        st.session_state.states = "search"
        st.rerun()

    # Add a statistics button to show annotation counts by user
    if st.sidebar.button("View Annotation Statistics", key="annotation_stats_button"):
        stats = get_annotation_stats()
        st.sidebar.subheader("Annotation Statistics")
        if stats:
            # Create a formatted display of annotation counts
            for annotator, count in stats.items():
                st.sidebar.markdown(f"**{annotator}**: {count} annotations")
        else:
            st.sidebar.info("No annotation data yet")

    if st.session_state.states == "sample" and st.button(
         "**Step 1: Sample a Topic**", use_container_width=True, key="sample_topic_button", type="primary"):
        result = random.choice(topic_list)
        # Display the sampled topic information
        st.subheader("Sampled Topic")
        st.markdown(f"**Claim:** {result['claim']}")

        st.markdown(f"**Topic:** {result['topic']}")
        st.markdown(f"**Target:** {result['target']}")
        st.session_state.current_record = result

    if st.session_state.current_record and st.session_state.states == "sample" and st.button("Use This Data", key="use_this_data"):
        st.session_state.records.append(st.session_state.current_record)
        st.session_state.states = "search"
        st.rerun()
    
    if st.session_state.states == "search" and st.session_state.current_record:
        st.subheader("Currently Selected Record")
        st.markdown(f"**Claim:** {st.session_state.current_record['claim']}")
        st.markdown(f"**Topic:** {st.session_state.current_record['topic']}")
        if 'target' in st.session_state.current_record:
            st.markdown(f"**Target:** {st.session_state.current_record['target']}")
        
        st.subheader("Select Search Query")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Use Topic as Query", use_container_width=True):
                st.session_state.search_query = st.session_state.current_record['topic']
                st.success(f"Selected Topic as query: {st.session_state.search_query}")
        
        with col2:
            if 'target' in st.session_state.current_record and st.button("Use Target as Query", use_container_width=True):
                st.session_state.search_query = st.session_state.current_record['target']
                st.success(f"Selected Target as query: {st.session_state.search_query}")
        
        st.markdown("**Or enter custom query:**")
        custom_query = st.text_input("Enter custom query", key="custom_query_input")
        if custom_query and st.button("Use Custom Query", use_container_width=True):
            st.session_state.search_query = custom_query
            st.success(f"Selected custom query: {st.session_state.search_query}")
        
        # Display the selected query if it exists
        if 'search_query' in st.session_state and st.session_state.search_query:
            st.info(f"Current query: {st.session_state.search_query}")
        
            # Search method selection
            st.subheader("Select Search Method")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("Cache Search", use_container_width=True):
                    if 'search_query' in st.session_state and st.session_state.search_query:
                        with st.spinner("Searching cache..."):
                            search_results = bm25_index.search(st.session_state.search_query, k=width+1)
                            deduplicated_results = []
                            for result in search_results:
                                if result["claim"] != st.session_state.current_record['claim']:
                                    deduplicated_results.append(result)
                            st.session_state.search_results = deduplicated_results
                            st.session_state.search_type = "cache"
                            st.session_state.states = "branching"
                    else:
                        st.error("Please select or enter a search query first")

            with col2:
                if st.button("Google Search", use_container_width=True):
                    if 'search_query' in st.session_state and st.session_state.search_query:
                        with st.spinner("Searching Google..."):
                            st.session_state.search_results = search_google(st.session_state.search_query)[:width]
                            st.session_state.search_type = "google"
                            st.session_state.states = "branching"
                            
                    else:
                        st.error("Please select or enter a search query first")

    if st.session_state.states == "branching" and 'search_results' in st.session_state and st.session_state.search_results:
        st.subheader("Search Results")
        
        # Choose the appropriate display function based on search type
        if st.session_state.search_type == "google":
            display_google_results(st.session_state.search_results, qwen_agent)
        elif st.session_state.search_type == "cache":
            display_cache_results(st.session_state.search_results, qwen_agent)
        st.session_state.states = "claim"

    if st.session_state.states == "claim":
        if st.session_state.search_type == "cache":
            select_cache_result(st.session_state.search_results, "cache")
        elif st.session_state.search_type == "google":
            select_google_result(st.session_state.search_results)

    if st.session_state.states == "extract":
        st.session_state.search_results = extract_claim(st.session_state.web_to_claim, qwen_agent)[:width]
        display_cache_results(st.session_state.search_results, qwen_agent)
        st.session_state.states = "claim"
        st.session_state.search_type = "cache"
    
    if st.session_state.states == "claim" and st.session_state.web_to_claim and st.session_state.search_type == "cache":
        select_cache_result(st.session_state.search_results, "google_cache")
    
    if st.session_state.states == "generate" and st.session_state.records:
        st.subheader("Selected Claims")
        
        # Initialize selection state if not exists
        if 'selected_records' not in st.session_state or len(st.session_state.selected_records) != len(st.session_state.records):
            st.session_state.selected_records = [True] * len(st.session_state.records)
        
        # Display record cards
        colors = ["#f0f7ff", "#fff0f0", "#f0fff0", "#f7f0ff", "#fffff0"]
        width = 3  # Set to three columns
        for i in range(0, len(st.session_state.records), width):
            cols = st.columns(width)
            for j in range(width):
                if i + j < len(st.session_state.records):
                    record = st.session_state.records[i + j]
                    record_index = i + j
                    with cols[j]:
                        card_color = colors[(i+j) % len(colors)]
                        
                        st.markdown(f"""
                        <div style="background-color: {card_color}; padding: 10px; border-radius: 10px; margin-bottom: 10px;">
                            <h4 style="color: #333;">Claim #{record_index+1}</h4>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"<h4 style='font-size: 18px;'>Topic: {record.get('topic', '')}</h4>", unsafe_allow_html=True)
                        st.markdown(f"<p style='font-size: 16px;'>Claim: {record.get('claim', '')}</p>", unsafe_allow_html=True)
                        # st.markdown(f"<p style='font-size: 16px;'>Chinese Translation: {record.get('claim_zh', '')}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='font-size: 16px;'>Target: {record.get('target', '')}</p>", unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Checkbox
                        st.session_state.selected_records[record_index] = st.checkbox(f"Use this Claim", value=True, key=f"record_checkbox_{record_index}")
        
        # Button area
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Generate Question", use_container_width=True, type="secondary"):
                # Only use selected records
                selected_records = [record for i, record in enumerate(st.session_state.records) 
                                   if st.session_state.selected_records[i]]
                st.session_state.save_records = selected_records
                evidence_str = "\n".join([f"- {record['claim']}" for record in selected_records])
                prompt = generate_multihop_query_prompt.format(evidence_str=evidence_str)
                response = stream_output(gpt_agent, prompt)
                response = json.loads(response)
                st.markdown(f"**Final Question:** {response['query']}")

                st.markdown(f"**Final Answer:** {response['answer']}")

                st.session_state.final_question = response

        
        with col2:
            if st.button("Make Question More Complex", use_container_width=True):
                if st.session_state.final_question:
                    st.session_state.save_records = [record for i, record in enumerate(st.session_state.records)  if st.session_state.selected_records[i]]

                    query = st.session_state.final_question['query']
                    prompt = make_query_more_complex_prompt.format(query=query)
                    response = stream_output(gpt_agent, prompt)
                    st.markdown(f"**Original Question:** {query}")
                    st.markdown(f"**More Complex Question:** {response}")
                    st.markdown(f"**Final Answer:** {st.session_state.final_question['answer']}")
                    if st.button("Apply", use_container_width=True):
                        st.session_state.final_question['query'] = response

                else:
                    st.warning("Please generate a question first before trying to make it more complex.")

        # Save question button
        if st.session_state.final_question:
            if st.button("Save Question", use_container_width=True, type="primary"):
                # Create save data structure
                query = st.session_state.final_question['query']
                answer = st.session_state.final_question['answer']
                records = st.session_state.save_records
                save_records = json.dumps({"metadata": st.session_state.final_question, "records": records})
                insert_annotated_data(query, answer, user_name, save_records)
                st.success("Question saved")

if __name__ == "__main__":
    main()