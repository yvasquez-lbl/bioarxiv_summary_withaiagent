import os
from openai import OpenAI
from tools.find_papers import BiorxivAgent
from tools.summarize_papers import PaperSummarizer
from tools.generate_paper_images import PaperImageGenerator

# Initialize OpenAI with LBL specifics
client = OpenAI(
    api_key=os.environ.get('CBORG_API_KEY'),
    base_url="https://api.cborg.lbl.gov"
)

class AIAgent:
    def __init__(self):
        self.model = "lbl/cborg-chat:latest"
        self.biorxiv_agent = BiorxivAgent()
        self.paper_summarizer = PaperSummarizer()
        self.image_generator = PaperImageGenerator()
        self.last_paper_doi = None  # Store the last paper's DOI
        
        # Add default authors of interest
        default_authors = [
            "Schulz, F.",
            "Shrestha, B.",
            "Vasquez, Y.M.",
            "Villada, J. C.",
            "Romero, M. F.",
            "Bowers, R."
        ]
        for author in default_authors:
            self.biorxiv_agent.add_author_of_interest(author)

    def process_query(self, query: str) -> str:
        """Process a natural language query and return appropriate response"""
        try:
            # First, determine what the user wants to do
            system_prompt = """You are a helpful research assistant for the NeLLi group. 
            Your task is to understand what the user wants to do and respond appropriately.
            You can help with:
            1. Finding recent papers from specific authors (with optional date range)
            2. Summarizing papers (requires a DOI)
            3. Generating images for papers (requires a DOI)
            
            For date ranges, you can understand formats like:
            - "last week"
            - "last month"
            - "from 2024-01-01 to 2024-03-31"
            - "between 2024-01-01 and 2024-03-31"
            
            If the user wants to summarize or generate an image for a paper without providing a DOI,
            and there was a previous paper found, use that paper's DOI.
            
            You MUST respond with a valid JSON object in exactly this format:
            {
                "action": "find_papers" | "summarize_paper" | "generate_image",
                "params": {
                    "query": "the actual query to use",
                    "start_date": "YYYY-MM-DD" or null,
                    "end_date": "YYYY-MM-DD" or null,
                    "use_last_paper": true or false
                }
            }
            
            Examples:
            For "find papers by Schulz and Shrestha from last week":
            {
                "action": "find_papers",
                "params": {
                    "query": "Schulz, F., Shrestha, B.",
                    "start_date": "2024-04-16",
                    "end_date": "2024-04-23",
                    "use_last_paper": false
                }
            }
            
            For "summarize this paper" (when a paper was just found):
            {
                "action": "summarize_paper",
                "params": {
                    "query": "",
                    "use_last_paper": true
                }
            }
            
            For "summarize paper with DOI 10.1101/2024.03.15.585123":
            {
                "action": "summarize_paper",
                "params": {
                    "query": "10.1101/2024.03.15.585123",
                    "use_last_paper": false
                }
            }
            
            If you can't determine what the user wants, respond with:
            {
                "action": "unknown",
                "params": {
                    "query": "original query",
                    "use_last_paper": false
                }
            }
            
            IMPORTANT: Your response must be a valid JSON object with no additional text or explanation.
            """

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.1  # Lower temperature for more consistent JSON output
            )
            
            # Parse the response
            import json
            try:
                # Clean the response to ensure it's valid JSON
                response_text = response.choices[0].message.content.strip()
                # Remove any markdown code block markers if present
                response_text = response_text.replace('```json', '').replace('```', '').strip()
                
                result = json.loads(response_text)
                action = result.get("action")
                params = result.get("params", {})
                query = params.get("query", "")
                start_date = params.get("start_date")
                end_date = params.get("end_date")
                use_last_paper = params.get("use_last_paper", False)
                
                if action == "find_papers":
                    return self._find_papers(query, start_date, end_date)
                elif action == "summarize_paper":
                    if use_last_paper and self.last_paper_doi:
                        return self._summarize_paper(self.last_paper_doi)
                    return self._summarize_paper(query)
                elif action == "generate_image":
                    if use_last_paper and self.last_paper_doi:
                        return self._generate_image(self.last_paper_doi)
                    return self._generate_image(query)
                else:
                    return "I'm not sure what you want to do. You can ask me to:\n" + \
                           "1. Find recent papers (e.g., 'find papers by Schulz and Shrestha from last week')\n" + \
                           "2. Summarize a paper (e.g., 'summarize paper with DOI 10.1101/2024.03.15.585123')\n" + \
                           "3. Generate an image for a paper (e.g., 'generate image for paper with DOI 10.1101/2024.03.15.585123')"
            
            except json.JSONDecodeError as e:
                print(f"Debug - Raw response: {response_text}")  # Debug print
                print(f"Debug - JSON error: {str(e)}")  # Debug print
                return "I had trouble understanding your request. Please try again with a clearer query."
                
        except Exception as e:
            print(f"Debug - Error: {str(e)}")  # Debug print
            return f"An error occurred: {str(e)}"

    def _find_papers(self, query: str, start_date: str = None, end_date: str = None) -> str:
        """Find papers based on the query and date range"""
        # Parse authors from the query, handling various formats
        authors = []
        # Split by common conjunctions and punctuation
        for part in query.replace(' and ', ',').replace('&', ',').split(','):
            author = part.strip()
            # Remove common prefixes
            author = author.replace('by ', '').replace('from ', '').replace('author ', '')
            if author:
                authors.append(author)
        
        print(f"Debug - Searching for authors: {authors}")  # Debug print
        
        # Add any new authors to the agent
        for author in authors:
            self.biorxiv_agent.add_author_of_interest(author)
        
        # Set date range if not provided
        from datetime import datetime, timedelta
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        print(f"Debug - Date range: {start_date} to {end_date}")  # Debug print
        
        # Search for papers
        found_papers = self.biorxiv_agent.search_authors_with_cursor(start_date, end_date, authors)
        
        if not found_papers:
            return f"No papers found for authors {', '.join(authors)} between {start_date} and {end_date}."
        
        # Store the DOI of the first paper found
        if found_papers:
            self.last_paper_doi = found_papers[0]['doi']
        
        # Format the results
        result = f"Found {len(found_papers)} papers for authors: {', '.join(authors)}\n"
        result += f"Date range: {start_date} to {end_date}\n\n"
        for paper in found_papers:
            result += f"Title: {paper['title']}\n"
            result += "Authors:\n"
            for author in paper['matching_authors']:
                result += f"  - {author['name']} ({author['affiliation']})\n"
            result += f"Date: {paper['date']}\n"
            result += f"DOI: {paper['doi']}\n"
            result += "="*50 + "\n"
        
        result += "\nYou can now ask me to 'summarize this paper' or 'generate an image for this paper'."
        return result

    def _summarize_paper(self, query: str) -> str:
        """Summarize a paper based on its DOI"""
        import re
        doi_pattern = r'10\.\d{4,9}/[-._;()/:\w]+'
        doi_match = re.search(doi_pattern, query)
        
        if not doi_match:
            return "No valid DOI found in the query."
        
        doi = doi_match.group(0)
        paper_data = self.paper_summarizer.get_paper_by_doi(doi)
        
        if not paper_data:
            return "Could not fetch paper data for the given DOI."
        
        summary = self.paper_summarizer.summarize_paper(paper_data)
        return summary

    def _generate_image(self, query: str) -> str:
        """Generate an image for a paper based on its DOI"""
        import re
        doi_pattern = r'10\.\d{4,9}/[-._;()/:\w]+'
        doi_match = re.search(doi_pattern, query)
        
        if not doi_match:
            return "No valid DOI found in the query."
        
        doi = doi_match.group(0)
        paper_data = self.image_generator.get_paper_by_doi(doi)
        
        if not paper_data:
            return "Could not fetch paper data for the given DOI."
        
        # Generate image prompt and image
        image_prompt = self.image_generator.generate_image_prompt(paper_data)
        image_path = self.image_generator.generate_image(image_prompt, paper_data['title'])
        
        if not image_path:
            return "Failed to generate image for the paper."
        
        return f"Image generated successfully. Prompt used: {image_prompt}"

def main():
    agent = AIAgent()
    print("Welcome to the NeLLi Research Assistant!")
    print("You can ask me to:")
    print("1. Find recent papers (e.g., 'find papers by Schulz and Shrestha from last week')")
    print("2. Summarize a paper (e.g., 'summarize paper with DOI 10.1101/2024.03.15.585123')")
    print("3. Generate an image for a paper (e.g., 'generate image for paper with DOI 10.1101/2024.03.15.585123')")
    print("\nType 'quit' to exit.")
    
    while True:
        query = input("\nWhat would you like me to do? ").strip()
        if query.lower() == 'quit':
            break
            
        response = agent.process_query(query)
        print("\n" + response)

if __name__ == "__main__":
    main() 