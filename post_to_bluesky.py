import os
import json
import re
import time
from datetime import datetime
from typing import Dict, List, Optional
from atproto import Client, models
from atproto.xrpc_client.models.app.bsky.feed.post import Main

class BlueskyPoster:
    def __init__(self, summary_file: str = "summary_output.log"):
        self.summary_file = summary_file
        self.client = None
        
    def authenticate(self, username: str, password: str):
        """Authenticate with Bluesky"""
        try:
            self.client = Client()
            self.client.login(username, password)
            print("Successfully authenticated with Bluesky")
            return True
        except Exception as e:
            print(f"Authentication failed: {e}")
            return False
            
    def extract_summaries_from_log(self) -> List[Dict]:
        """Extract paper summaries from the log file"""
        summaries = []
        
        try:
            with open(self.summary_file, 'r') as f:
                content = f.read()
                
            # Split content by paper entries
            paper_entries = content.split("="*50)
            
            for entry in paper_entries:
                if not entry.strip():
                    continue
                    
                # Extract title
                title_match = re.search(r'Title: (.*?)(?:\n|$)', entry)
                title = title_match.group(1).strip() if title_match else "Unknown Title"
                
                # Extract DOI
                doi_match = re.search(r'DOI: (.*?)(?:\n|$)', entry)
                doi = doi_match.group(1).strip() if doi_match else "Unknown DOI"
                
                # Extract authors
                authors_match = re.search(r'Authors: (.*?)(?:\n|$)', entry)
                authors = authors_match.group(1).strip() if authors_match else "Unknown Authors"
                
                # Extract summary - look for the Summary: section
                summary_match = re.search(r'Summary:\n(.*?)(?:\n\n|$)', entry, re.DOTALL)
                summary = summary_match.group(1).strip() if summary_match else ""
                
                if summary:
                    summaries.append({
                        'title': title,
                        'doi': doi,
                        'authors': authors,
                        'summary': summary
                    })
                    
            return summaries
            
        except Exception as e:
            print(f"Error extracting summaries from log file: {e}")
            return []
            
    def format_post_content(self, paper: Dict) -> str:
        """Format the paper summary for posting to Bluesky"""
        title = paper['title']
        doi = paper['doi']
        authors = paper['authors']
        summary = paper['summary']
        
        # Format the post content
        post_content = f"📚 New Paper Alert: {title}\n\n"
        post_content += f"👥 Authors: {authors}\n\n"
        post_content += f"🔍 Summary:\n{summary}\n\n"
        post_content += f"🔗 DOI: {doi}\n\n"
        post_content += "#Science #Research #Academic"
        
        return post_content
        
    def post_to_bluesky(self, content: str) -> bool:
        """Post content to Bluesky"""
        try:
            if not self.client:
                print("Not authenticated with Bluesky. Please authenticate first.")
                return False
                
            # Create the post
            post = Main(
                text=content,
                created_at=datetime.now().isoformat()
            )
            
            # Send the post
            response = self.client.send_post(post)
            print(f"Successfully posted to Bluesky: {response.uri}")
            return True
            
        except Exception as e:
            print(f"Error posting to Bluesky: {e}")
            return False
            
    def process_summaries(self, username: str, password: str, delay: int = 60):
        """Process all summaries and post them to Bluesky"""
        # Authenticate with Bluesky
        if not self.authenticate(username, password):
            return
            
        # Extract summaries from the log file
        summaries = self.extract_summaries_from_log()
        
        if not summaries:
            print("No summaries found in the log file")
            return
            
        print(f"Found {len(summaries)} summaries to post")
        
        # Post each summary
        for i, paper in enumerate(summaries):
            print(f"\nProcessing paper {i+1}/{len(summaries)}: {paper['title']}")
            
            # Format the post content
            post_content = self.format_post_content(paper)
            
            # Post to Bluesky
            success = self.post_to_bluesky(post_content)
            
            if success:
                print(f"Successfully posted summary for: {paper['title']}")
            else:
                print(f"Failed to post summary for: {paper['title']}")
                
            # Add a delay between posts to avoid rate limiting
            if i < len(summaries) - 1:  # Don't delay after the last post
                print(f"Waiting {delay} seconds before next post...")
                time.sleep(delay)
                
def main():
    # Get Bluesky credentials from environment variables or prompt the user
    username = os.environ.get('BLUESKY_USERNAME')
    password = os.environ.get('BLUESKY_PASSWORD')
    
    if not username or not password:
        username = input("Enter your Bluesky username: ")
        password = input("Enter your Bluesky password: ")
        
    poster = BlueskyPoster()
    poster.process_summaries(username, password)

if __name__ == "__main__":
    main() 