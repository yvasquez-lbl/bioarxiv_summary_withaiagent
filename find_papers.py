import os
import openai
import numpy as np
import requests
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import asyncio

# Initialize OpenAI with LBL specifics
#openai.api_key = os.environ.get('CBORG_API_KEY')
#openai.base_url = "https://api.cborg.lbl.gov"
#model = "lbl/llama"  # Make sure this is the correct model name

class BiorxivAgent:
    def __init__(self, base_url: str = "https://api.biorxiv.org", log_file: str = "paper_notifications.log"):
        self.base_url = base_url
        self.authors_of_interest = set()
        self.log_file = log_file
        
    def add_author_of_interest(self, author: str):
        self.authors_of_interest.add(author)
        
    def get_papers_by_date_range(self, start_date: str, end_date: str, 
                                server: str = "biorxiv", cursor: int = 0) -> Dict:
        try:
            endpoint = f"{self.base_url}/details/{server}/{start_date}/{end_date}/{cursor}"
            print(f"Fetching papers from endpoint: {endpoint}")  # Debug print
            response = requests.get(endpoint)
            response.raise_for_status()  # Raise an exception for bad status codes
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching papers: {e}")
            return {"collection": []}

    def get_paper_by_doi(self, doi: str, server: str = "biorxiv") -> Dict:
        try:
            endpoint = f"{self.base_url}/details/{server}/{doi}/na/json"
            response = requests.get(endpoint)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching paper by DOI: {e}")
            return {}

    def filter_papers_by_authors(self, papers_data: Dict) -> List[Dict]:
        filtered_papers = []
        collection = papers_data.get('collection', [])
        print(f"Total papers to filter: {len(collection)}")  # Debug print
        
        for paper in collection:
            authors = paper.get('authors', '').split('; ')
            if any(author in self.authors_of_interest for author in authors):
                filtered_papers.append(paper)
        
        print(f"Found {len(filtered_papers)} papers by authors of interest")  # Debug print
        return filtered_papers

    def process_new_papers(self, days_back: int = 1):
        """
        Process papers from the last n days.
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        print(f"Searching for papers between {start_date} and {end_date}")  # Debug print
        
        papers_data = self.get_papers_by_date_range(start_date, end_date)
        filtered_papers = self.filter_papers_by_authors(papers_data)
        
        processed_papers = []
        for paper in filtered_papers:
            print(f"Processing paper: {paper.get('title', 'No title')}")  # Debug print
            summary = self.generate_summary(paper.get('abstract', ''))
            if summary:
                processed_papers.append({
                    'doi': paper.get('doi'),
                    'title': paper.get('title'),
                    'authors': paper.get('authors'),
                    'abstract': paper.get('abstract'),
                    'summary': summary,
                    'date': paper.get('date')
                })
        
        return processed_papers

    def log_paper_notification(self, papers: List[Dict]):
        """
        Log paper details to a file, skipping papers that have already been logged
        """
        if not papers:
            return
            
        try:
            # Read existing log file to check for already logged titles
            existing_titles = set()
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    content = f.read()
                    # Extract titles from the log file
                    for line in content.split('\n'):
                        if line.startswith('Title: '):
                            existing_titles.add(line[7:].strip())
            
            # Filter out papers that have already been logged
            new_papers = [paper for paper in papers if paper['title'] not in existing_titles]
            
            if not new_papers:
                print("No new papers to log - all titles already exist in the log file")
                return
                
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(self.log_file, 'a') as f:
                f.write(f"\n\n=== New Papers Found at {timestamp} ===\n")
                for paper in new_papers:
                    f.write("\n" + "="*50 + "\n")
                    f.write(f"Title: {paper['title']}\n")
                    f.write("Matching Authors:\n")
                    for author in paper['matching_authors']:
                        f.write(f"  - {author['name']} ({author['affiliation']})\n")
                    f.write(f"Date: {paper['date']}\n")
                    f.write(f"DOI: {paper['doi']}\n")
                    f.write("="*50 + "\n")
                
            print(f"Logged {len(new_papers)} new papers to {self.log_file}")
            
        except Exception as e:
            print(f"Failed to log paper notifications: {e}")

    def search_authors_with_cursor(self, start_date: str, end_date: str, target_authors: List[str], max_cursor: int = 145) -> List[Dict]:
        """
        Search for papers by specific authors using cursor pagination.
        Returns papers that match any of the target authors, showing all matching authors for each paper.
        """
        found_papers = {}  # Using dict to track unique papers by DOI
        
        print(f"\nSearching for papers by authors: {', '.join(target_authors)}")
        
        for cursor in range(143, 146):
            papers_data = self.get_papers_by_date_range(start_date, end_date, cursor=cursor)
            collection = papers_data.get('collection', [])
            
            if not collection:
                break
                
            for paper in collection:
                authors = paper.get('authors', '').split(';')
                affiliations = paper.get('affiliations', '').split(';')
                
                # Create a mapping of authors to their affiliations
                author_affiliations = {}
                for i, author in enumerate(authors):
                    if i < len(affiliations):
                        author_affiliations[author.strip()] = affiliations[i].strip()
                    else:
                        author_affiliations[author.strip()] = "No affiliation listed"
                
                # Find all matching authors for this paper
                matching_authors = []
                for author in authors:
                    author = author.strip()
                    if any(target.strip() in author for target in target_authors):
                        matching_authors.append({
                            'name': author,
                            'affiliation': author_affiliations.get(author, "No affiliation listed")
                        })
                
                # If we found any matching authors, add the paper
                if matching_authors:
                    doi = paper.get('doi')
                    if doi not in found_papers:  # Only add if we haven't seen this paper before
                        found_papers[doi] = {
                            'doi': doi,
                            'title': paper.get('title'),
                            'matching_authors': matching_authors,
                            'date': paper.get('date')
                        }
        
        results = list(found_papers.values())
        
        # Log paper notifications if papers are found
        if results:
            self.log_paper_notification(results)
            
        return results

def main():
    agent = BiorxivAgent()
    
    # Define target authors as a list
    target_authors = ["Schulz, F.", "Shrestha, B.", "Vasquez, Y.M.", "Villada, J. C.","Romero, M. F.","Bowers, R."]
    
    # Set date range
    end_date = '2025-04-01'
    start_date = '2025-04-01'
    
    try:
        # Search for papers with target authors
        found_papers = agent.search_authors_with_cursor(start_date, end_date, target_authors)
        
        # Print results
        if not found_papers:
            print("\nNo papers found for any of the target authors")
        else:
            print(f"\nFound {len(found_papers)} unique papers")
            for paper in found_papers:
                print("\n" + "="*80)
                print(f"Title: {paper['title']}")
                print("Matching Authors:")
                for author in paper['matching_authors']:
                    print(f"  - {author['name']} ({author['affiliation']})")
                print(f"Date: {paper['date']}")
                print(f"DOI: {paper['doi']}")
                print("="*80)
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()