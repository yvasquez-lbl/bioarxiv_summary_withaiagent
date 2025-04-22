import os
from openai import OpenAI
import requests
import re
import json
from typing import Dict, List

# Initialize OpenAI with LBL specifics
client = OpenAI(
    api_key=os.environ.get('CBORG_API_KEY'), # Please do not store your API key in the code
    base_url="https://api.cborg.lbl.gov" # Local clients can also use https://api-local.cborg.lbl.gov
)

model = "lbl/cborg-coder:latest"

class PaperSummarizer:
    def __init__(self, log_file: str = "paper_notifications.log"):
        self.log_file = log_file
        self.base_url = "https://api.biorxiv.org"

    def get_paper_by_doi(self, doi: str, server: str = "biorxiv") -> Dict:
        """Fetch paper data from biorxiv using DOI"""
        try:
            # First get the paper details
            details_endpoint = f"{self.base_url}/details/{server}/{doi}/na/json"
            print(f"Fetching paper details from: {details_endpoint}")
            details_response = requests.get(details_endpoint)
            details_response.raise_for_status()
            paper_details = details_response.json()
            
            print(f"Paper details response: {json.dumps(paper_details, indent=2)}")
            
            # Extract data from the collection
            if 'collection' in paper_details and paper_details['collection']:
                paper_info = paper_details['collection'][0]
                
                # Create a new dictionary with the extracted information
                extracted_data = {
                    'title': paper_info.get('title', ''),
                    'abstract': paper_info.get('abstract', ''),
                    'authors': paper_info.get('authors', ''),
                    'doi': paper_info.get('doi', ''),
                    'date': paper_info.get('date', ''),
                    'category': paper_info.get('category', '')
                }
                
                # Try to get the full text if available
                if 'jatsxml' in paper_info:
                    try:
                        xml_url = paper_info['jatsxml']
                        print(f"Fetching XML content from: {xml_url}")
                        xml_response = requests.get(xml_url)
                        xml_response.raise_for_status()
                        extracted_data['full_text'] = xml_response.text
                    except Exception as e:
                        print(f"Could not fetch XML content: {e}")
                        extracted_data['full_text'] = ''
                
                return extracted_data
            else:
                print("No collection found in paper details")
                return {}
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching paper by DOI: {e}")
            if hasattr(e.response, 'text'):
                print(f"Response text: {e.response.text}")
            return {}
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {}

    def summarize_paper(self, paper_data: Dict) -> str:
        """Generate summary using internal AI system"""
        try:
            # Extract relevant information from paper data
            title = paper_data.get('title', '')
            abstract = paper_data.get('abstract', '')
            full_text = paper_data.get('full_text', '')
            
            print(f"Title: {title}")
            print(f"Abstract length: {len(abstract)}")
            print(f"Full text length: {len(full_text)}")
            
            # Create prompt for summarization using only title and abstract
            prompt = f"""Please provide a 300-word summary of the following scientific paper:

Title: {title}

Abstract: {abstract}

#Full Text: full_text add brackets back if using

Summary:"""
            
            # Call the internal AI system
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a scientific paper summarizer. Provide clear, concise summaries of scientific paper provided in exactly 300 words. I want the summary to be for social media, specifically bluesky, so it should be fun and engaging."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Error generating summary"

    def process_log_file(self):
        """Read DOIs from log file and generate summaries"""
        try:
            with open(self.log_file, 'r') as f:
                content = f.read()
                
            # Extract DOIs using regex
            doi_pattern = r'DOI: (10\.\d{4,9}/[-._;()/:\w]+)'
            dois = re.findall(doi_pattern, content)
            
            if not dois:
                print("No DOIs found in log file")
                return
                
            print(f"\nFound {len(dois)} papers to summarize")
            
            # Process each DOI
            for doi in dois:
                print(f"\nProcessing DOI: {doi}")
                paper_data = self.get_paper_by_doi(doi)
                
                if paper_data:
                    summary = self.summarize_paper(paper_data)
                    print("\n" + "="*80)
                    print(f"Title: {paper_data.get('title', 'No title')}")
                    print("\nDOI: ")
                    print(f"{paper_data.get('doi', 'No doi')}")
                    print("\nAuthors: ")
                    print(f"{paper_data.get('authors', 'No authors')}")
                    print("\nSummary:")
                    print(summary)
                    print("="*80)
                else:
                    print(f"Could not fetch paper data for DOI: {doi}")
                    
        except Exception as e:
            print(f"Error processing log file: {e}")

def main():
    summarizer = PaperSummarizer()
    summarizer.process_log_file()

if __name__ == "__main__":
    main() 