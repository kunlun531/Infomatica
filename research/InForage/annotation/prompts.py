GENERATE_SOURCE_QUERY = """Generate a random search query for information gathering. The query should be specific and seek non-trivial information.

Examples:
- Which of Bach's compositions is considered his most famous work?
- What was Murray Gell-Mann's most influential speech?
- Where did Yoshua Bengio spend his childhood years?
- Who was Fei-Fei Li's PhD advisor?

Generate 10 search queries following the pattern above. The queries should:
- Not be about common knowledge or easily known facts
- The queries should be diverse and cover different topics
- The queries should be answered one or two words
- be separated by newlines


Output only the query text, without any additional explanation or formatting.
"""

GENERATE_FOLLOW_UP_QUERY = """Given a search query and related text passage, generate a follow-up query that explores a related but different topic mentioned in the text. The follow-up query should:

- Shift focus to a new but connected topic from the passage
- Not directly ask about the original subject
- Build on contextual information found in the text
- Lead to deeper exploration of the broader subject area
- The query should be answered one or two words
- The query should be short
- Explain how this new query relates to the original query

Original Query: {query}

Related Text: {text}

Generate a single follow-up search query that explores a different but related topic mentioned in the text. The query should be specific and seek non-trivial information.

Output the query text followed by a brief explanation of how it relates to the original query, separated by a newline."""
GENERATE_MULTIHOP_QUERY = """Given a series of connected search queries and their relationships, generate a single multi-hop question that captures the entire chain of reasoning.

Example input:
Query 1: "Who created ImageNet?"
Rationale: Found that Fei-Fei Li created ImageNet
Query 2: "Who was Fei-Fei Li's PhD advisor?"
Rationale: Found that Paul Viola was her PhD advisor
Query 3: "Where did Paul Viola get his undergraduate degree?"
Rationale: This will tell us where the advisor of ImageNet's creator studied

Example output:
"Where did the PhD advisor of ImageNet's creator receive their undergraduate degree?"

The multi-hop question should:
- Combine all the individual queries into one fluent question
- Maintain the logical connection between entities
- Be clear and unambiguous
- Be answerable with a specific fact or short phrase
- Preserve the reasoning chain from the original queries

Given queries and rationales:
{hops}

Generate a single multi-hop question that captures this chain of reasoning.

Output only the multi-hop question without any additional explanation."""


claim_prompt = """A "claim" is a statement or assertion made within a text expressing a belief, opinion, or fact. Given evidence from the original context, please extract one claim and its associated topics.

Note: The claim should not contain ambiguous references, such as 'he',' she,' and' it', and should use complete names. If there are multiple topics, give the most dominant one. The target of the claim (one entity)is the specific individual, group, or organization that the statement or assertion within a text is directed towards or about which it is making a case. The topic of the claim should be a simple phrase representing the claim's central argument concept. If there is no claim, please leave it blank. Please generate a claim based on the given evidence. Don't generate the evidence yourself.

Please give the response following this format:
##Evidence: {{original context}}
##Claims: {{extract claim}}
##Claim Target: {{target}}
##Claim Topic: {{topic}}

Here are examples:
##Input: Before Android debuted its first-ever turn-by-turn navigation system with real-time GPS tracking, people used third-party devices offered by TomTom and Garmin, or built-in navigation systems inside vehicles' infotainment systems. Maybe you were the type of person to print out a bunch of Google Maps instructions to keep in your vehicle, but the advent of "free" navigation software that didn't require a monthly subscription or additional application purchase for smartphones, didn't exist before October 28, 2009 when Android 2.0 got released. Since then, everyone has had a personal GPS in their pocket, and it's become almost a foregone conclusion that everyone with an Android or iOS device can navigate to wherever they need to go. But software, at the end of the day, is designed by imperfect beings, and users of navigation apps like Google Maps or Apple Maps have reported strange directions they've been asked to follow when on their routes. Something that TikToker @jpall20 believes has gotten worse over the years. In a video that's acquired over 690,000 views as of Sunday, she theorizes there may be some "powers that be" making drivers question their sanity with the roundabout routes in these navigation apps. 

##Evidence: The advent of "free" navigation software that didn't require a monthly subscription or additional application purchase for smartphones, didn't exist before October 28, 2009 when Android 2.0 got released.
##Claims: Android debuted its first-ever turn-by-turn navigation system with real-time GPS tracking on October 28, 2009.
##Claim Target: navigation system
##Claim Topic: first-ever turn-by-turn navigation system

Now, it's your turn.
##Input: {context}"""

generate_multihop_query_prompt = """Based on the following evidence, generate a complex multi-hop query that requires connecting multiple pieces of information to answer.

Evidence:
{evidence_str}

Create a challenging question that requires reasoning across multiple facts. The question should be specific enough that it can only be answered by connecting several pieces of evidence together.

For example, given the evidence list:
1. Donald Trump is the president of the United States.
2. Donald Trump was born in New York City.
3. Donald Trump is born in 1940s.
4. Donald Trump's predecessor was Barack Obama.
5. Barack Obama was born in Hawaii.
6. Barack Obama's predecessor was George W. Bush.
7. George W. Bush was born in Texas.
8. Trump has been divorced twice.

The multi-hop query could be:
"Where is the birthplace of the predecessor of the U.S. president who was born in 1940s and has been divorced?"

Requirement:
1. Ensure Coherence: Make sure the question flows logically from the combined information and is
clear and unambiguous
2. You don't need to use all the evidence, just use the most relevant ones.
3. Formulate the Question: Create a question that cannot be answered by relying on just one of the sentences but instead requires understanding and linking the information from all of the sources. 
4. Ensure Multi-hop: The question should require at least 3 logical steps to answer.
5. The question should be one single question, not parallel questions link with "and" or "or".
6. The answer should be specific and factual, no longer than 10 words.

Now, based on the following evidence, generate a multi-hop query that requires at least 3 logical steps to answer:

{evidence_str}

Generate a multi-hop query that requires at least 3 logical steps to answer.
Output in the following format:
{{"query": "multi-hop query", "answer": "answer"}}"""


generate_multihop_description_prompt = """Based on the following evidence, generate a complex multi-hop description that requires connecting multiple pieces of information to identify the subject.

Evidence:
{evidence_str}

Create a challenging description that requires reasoning across multiple facts. The description should be specific enough that it can only identify one subject by connecting several pieces of evidence together.

For example, given the evidence list:
1. Donald Trump is the president of the United States.
2. Donald Trump was born in New York City.
3. Donald Trump is born in 1940s.
4. Donald Trump's predecessor was Barack Obama.
5. Barack Obama was born in Hawaii.
6. Barack Obama's predecessor was George W. Bush.
7. George W. Bush was born in Texas.
8. Trump has been divorced twice.

The multi-hop description could be:
"Here is a person,
This person was born in New York City in the 1940s,
This person's predecessor in their position was born in Hawaii,
This person has been divorced multiple times.
Who is this person?"

Requirement:
1. Ensure Coherence: Make sure the description flows logically and builds up clues piece by piece
2. You don't need to use all the evidence, just use the most relevant ones
3. Formulate the Description: Create a description that cannot identify the subject by relying on just one of the sentences but instead requires understanding and linking the information from all of the sources
4. Ensure Multi-hop: The description should require at least 3 logical steps to identify the subject
5. The description should be structured as multiple lines, each adding a new clue
6. The answer should be specific and factual, no longer than 5 words

Now, based on the following evidence, generate a multi-hop description that requires at least 3 logical steps to identify the subject:

{evidence_str}

Generate a multi-hop description that requires at least 3 logical steps to identify the subject.
Output in the following format:
{{"query": "multi-hop description", "answer": "answer"}}"""


make_query_more_complex_prompt = """You are tasked with making a query more complex by replacing specific entities or concepts with more ambiguous descriptions that still uniquely identify them.

For example:
- Replace "Shakespeare" with "the playwright that authored the famous line, 'To be or not to be, that is the question'"
- Replace "Leonardo da Vinci" with "the man who created the enigmatic smiling portrait housed in the Louvre Museum"
- Replace "Hawaii" with "an American state not on the North American mainland"

Original query: {query}

Your task:
1. Identify specific entities (people, places, organizations, etc.) in the query
2. Replace each entity with a more ambiguous but uniquely identifying description
3. Ensure the modified query maintains the same meaning and can be answered with the same answer
4. The description should be specific enough that it can only refer to that entity
5. only output the modified query, without any additional explanation or formatting.

Modified query: """

make_answer_more_complex_prompt = """Given a query and its answer, rewrite the answer to match the complexity level of the query. If the query uses ambiguous descriptions instead of direct entity names, your answer should maintain the same style.

Original query: {query}
Original answer: {answer}

Your task:
1. Analyze the style and complexity of the query
2. Rewrite the answer to match this style
3. If the query uses ambiguous descriptions, use the same or similar descriptions in your answer
4. Ensure the rewritten answer contains the same factual information as the original

Rewritten answer: """

select_claim_and_query_prompt = """You are tasked with selecting a factual claim from multiple claims and generating a bridge query for searching related information.

Given claims in the format:
claim0: {{"claim": "", "target": "", "topic": ""}}
...

Your task:
1. Select one claim that contains clear, factual information (e.g., "Bach died in 1750", "Donald Trump is the president of the United States")
2. Generate a bridge query that can help search for information related to the selected claim
3. When generating the bridge query, prioritize using:
   - The target field if relevant
   - The topic field if relevant
   - Key entities from the claim if neither target nor topic is suitable
4. If current query is provided, please select a diverse claim that is not similar to the current query.

For example, given:
{{"claim": "OpenAI is beginning to test connectors for ChatGPT that will integrate with Google Drive and Slack", 
 "topic": "ChatGPT connectors for Google Drive and Slack",
 "target": "OpenAI"}}

You could select terms like "ChatGPT", "OpenAI", "Slack", or "Google Drive" as the bridge query.

Now, it's your turn.
Current query: {query}
Claims to analyze:
{claims}

Select one claim and generate a bridge query. The bridge query should be diverse from the current query.
Output only a JSON object in the format:
{{"id": "ID of selected claim", "query": "bridge query"}}"""

select_relevant_claim_prompt = """You are a helpful assistant that selects relevant claims from a list of claims. 

For example, for the query "America's vice president" and the following claims:
claim0: America's vice president is Kamala Harris.
claim1: America has the highest GDP in the world.
claim2: France's vice president is Emmanuel Macron.
claim4: America's vice president is the second highest position in the country.

You should select claim0 and claim4 as the most relevant claims.

query: {query}
claims: {claims}

Your task:
1. Select the claim that is most relevant to the query
2. Output the claim in the format:
[0, 4]
"""
