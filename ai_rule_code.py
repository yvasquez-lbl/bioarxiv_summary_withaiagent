import requests
import re
import os
from datetime import datetime, timedelta
from openai import OpenAI
from tools.find_papers import BiorxivAgent
from tools.summarize_papers import PaperSummarizer
from tools.generate_paper_images import PaperImageGenerator

# Initialize OpenAI with LBL specifics
client = OpenAI(
    api_key=os.environ.get('CBORG_API_KEY'),
    base_url="https://api.cborg.lbl.gov"
)

model = "lbl/cborg-chat:latest"

FUNCTION_DEFINITIONS = [
    {
        "name": "find_papers",
        "description": "Retrieve recent papers from bioarxiv from the NeLLi group",
        "parameters": {
            "type": "dict",
            "required": [
                "query"
            ],
            "properties": {
                "query": {
                    "type": "str",
                    "description": "The query to utilize to get papers from bioarxiv api"
                },
            }
        }
    },
    {
        "name": "summarize_papers",
        "description": "Summarize papers for bluesky (300 word count), currently based on abstracts",
        "parameters": {
            "type": "dict",
            "required": [
                "query"
            ],
            "properties": {
                "query": {
                    "type": "str",
                    "description": "The query to utilize to summarize paper abstract in 300 word count"
                },
            }
        }
    },
    {
        "name": "generate_paper_images",
        "description": "Generate images based on the abstract of the paper",
        "parameters": {
            "type": "dict",
            "required": [
                "query"
            ],
            "properties": {
                "query": {
                    "type": "str",
                    "description": "The query to generate image of the paper"
                },
            }
        }
    },
]

def find_papers_codehere(query: str) -> str:
    """Wrapper function for finding papers"""
    agent = BiorxivAgent()
    # Add authors from the query
    authors = [author.strip() for author in query.split(',')]
    for author in authors:
        agent.add_author_of_interest(author)
    
    # Set date range to last 7 days
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    # Search for papers
    found_papers = agent.search_authors_with_cursor(start_date, end_date, authors)
    
    if not found_papers:
        return "No papers found for the specified authors."
    
    # Format the results
    result = "Found papers:\n\n"
    for paper in found_papers:
        result += f"Title: {paper['title']}\n"
        result += "Authors:\n"
        for author in paper['matching_authors']:
            result += f"  - {author['name']} ({author['affiliation']})\n"
        result += f"Date: {paper['date']}\n"
        result += f"DOI: {paper['doi']}\n"
        result += "="*50 + "\n"
    
    return result

def summarize_papers_codehere(query: str) -> str:
    """Wrapper function for summarizing papers"""
    summarizer = PaperSummarizer()
    
    # Extract DOI from query
    doi_pattern = r'10\.\d{4,9}/[-._;()/:\w]+'
    doi_match = re.search(doi_pattern, query)
    
    if not doi_match:
        return "No valid DOI found in the query."
    
    doi = doi_match.group(0)
    paper_data = summarizer.get_paper_by_doi(doi)
    
    if not paper_data:
        return "Could not fetch paper data for the given DOI."
    
    summary = summarizer.summarize_paper(paper_data)
    return summary

def generate_paper_images_codehere(query: str) -> str:
    """Wrapper function for generating paper images"""
    generator = PaperImageGenerator()
    
    # Extract DOI from query
    doi_pattern = r'10\.\d{4,9}/[-._;()/:\w]+'
    doi_match = re.search(doi_pattern, query)
    
    if not doi_match:
        return "No valid DOI found in the query."
    
    doi = doi_match.group(0)
    paper_data = generator.get_paper_by_doi(doi)
    
    if not paper_data:
        return "Could not fetch paper data for the given DOI."
    
    # Generate image prompt and image
    image_prompt = generator.generate_image_prompt(paper_data)
    image_path = generator.generate_image(image_prompt, paper_data['title'])
    
    if not image_path:
        return "Failed to generate image for the paper."
    
    return f"Image generated successfully. Prompt used: {image_prompt}"

class LanguageModelWrapper:

    def _parse_function_call(self, response_text):
        """Parse a function call from the model's response"""
        try:
            # Extract function name and parameters
            func_name = response_text.split('(')[0].strip()
            # Extract everything between parentheses
            params_str = response_text[response_text.find('(')+1:response_text.rfind(')')]
            # Parse parameters
            params = {}
            if 'query="' in params_str:  # Handle quoted string parameters
                params['query'] = params_str.split('query="')[1].split('"')[0]
            return func_name, params
        except Exception as e:
            print(f"Error parsing function call: {e}")
            return None, None

    def _execute_function(self, func_name, params):
        """Execute the specified function with given parameters"""
        function_mapping = {
            'find_papers': find_papers_codehere,
            'summarize_papers': summarize_papers_codehere,
            'generate_paper_images': generate_paper_images_codehere,
            # Add new functions here as they become available
            # 'another_function': another_function,
        }
        
        if func_name in function_mapping:
            try:
                return function_mapping[func_name](**params)
            except Exception as e:
                print(f"Error executing function {func_name}: {e}")
                return None
        return None


    def generate_response(self, prompt, model="lbl/cborg-chat:latest"):
        """Send request to CBORG API"""
        try:
            # First step: Research prompt
            research_prompt = f"""You are a science communicator assistant.
            Based on the following topic, determine if you need to gather additional information.
            If you do, you can use one of these available functions:

            {str(FUNCTION_DEFINITIONS)}

            Use find_papers_codehere when you need:
            - Find recent papers from the NeLLi group authors from bioarxiv

            Use summarize_papers_codehere when you need:
            - Summarize paper abstract in 300 words in order to be ready to publish in bluesky

            Use generate_paper_images when you need:
            - To generate an image based on the abstract


            Format your response exactly like this if you want to call a function:
            FUNCTION: function_name(param_name="param_value")

            If you don't need to gather information, respond with:
            NO_FUNCTION_NEEDED

            Topic: {prompt}

            Note: You can only call one function at a time.
            """

            # Get function call decision
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a science communicator assistant that helps determine when to use specific functions for gathering information."},
                    {"role": "user", "content": research_prompt}
                ]
            )
            function_response = response.choices[0].message.content.strip()
            
            research_results = ""
            func_name = None

            # Check if a single function call is needed
            if function_response.startswith("FUNCTION:"):
                function_call = function_response.replace("FUNCTION:", "").strip()
                func_name, params = self._parse_function_call(function_call)
                
                if func_name and params:
                    research_results = self._execute_function(func_name, params)
                else:
                    print("Failed to parse function call")
                    research_results = ""
            else:
                research_results = ""

            # Second step: Tweet generation
            post_prompt = """You are a member of the NeLLi research group, a group focused on the new lineages of life.
            Write tweets that reflect my voice and expertise in new lineages of life and science.

            Tweet Style Guide:
            - Write in a confident yet approachable tone
            - Use active voice and present tense
            - Keep it conversational and engaging
            - Maximum 300 characters
            
            Format:
            - No quotes
            - No random capitalization
            """

            # Add research results if available
            if research_results:
                post_prompt += f"\nHere is additional context that you can use to write such post:\n{research_results}"
            
            post_prompt += f"\nTopic: {prompt}\n\nRespond with ONLY the tweet text, nothing else."

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a member of the NeLLi research group, focused on writing engaging social media posts about new lineages of life and science."},
                    {"role": "user", "content": post_prompt}
                ]
            )
            return {
                "text": response.choices[0].message.content.strip(),
                "tool_used": func_name if func_name else "",
            }
        
        except Exception as e:
            print(f"Error calling CBORG API: {e}")
            return None