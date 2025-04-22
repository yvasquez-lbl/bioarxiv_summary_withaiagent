import os
import json
import re
import time
from datetime import datetime
from typing import Dict, List, Optional
from atproto import Client, models
from atproto.exceptions import AtProtocolError

class BlueskyPoster:
    def __init__(self, summary_file: str = "summary_output.log", 
                 image_prompts_dir: str = "paper_images",
                 handle: str = None, 
                 password: str = None):
        """
        Initialize the Bluesky poster
        
        Args:
            summary_file: Path to the file containing paper summaries
            image_prompts_dir: Directory containing image prompt files
            handle: Bluesky handle (username)
            password: Bluesky password
        """
        self.summary_file = summary_file
        self.image_prompts_dir = image_prompts_dir
        self.client = Client()
        
        # Get credentials from environment variables if not provided
        self.handle = handle or os.environ.get('BLUESKY_HANDLE')
        self.password = password or os.environ.get('BLUESKY_PASSWORD')
        
        if not self.handle or not self.password:
            raise ValueError("Bluesky credentials not provided. Set BLUESKY_HANDLE and BLUESKY_PASSWORD environment variables or pass them as arguments.")
        
        # Login to Bluesky
        try:
            self.client.login(self.handle, self.password)
            print(f"Successfully logged in to Bluesky as {self.handle}")
        except AtProtocolError as e:
            print(f"Failed to login to Bluesky: {e}")
            raise
    
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
    
    def find_image_prompt(self, title: str) -> Optional[str]:
        """Find the image prompt file for a given paper title"""
        try:
            # Create a safe filename from the title
            safe_title = re.sub(r'[^\w\s-]', '', title)
            safe_title = re.sub(r'[-\s]+', '_', safe_title).strip('-_')
            prompt_file = os.path.join(self.image_prompts_dir, f"{safe_title[:50]}.txt")
            
            if os.path.exists(prompt_file):
                with open(prompt_file, 'r') as f:
                    return f.read()
            return None
            
        except Exception as e:
            print(f"Error finding image prompt: {e}")
            return None
    
    def format_post(self, paper: Dict) -> str:
        """Format a paper summary for posting to Bluesky"""
        title = paper['title']
        doi = paper['doi']
        authors = paper['authors']
        summary = paper['summary']
        
        # Format the post
        post = f"📚 New Paper Alert: {title}\n\n"
        post += f"Authors: {authors}\n\n"
        post += f"Summary: {summary}\n\n"
        post += f"DOI: {doi}\n\n"
        post += "#Science #Research"
        
        return post
    
    def post_to_bluesky(self, text: str) -> bool:
        """Post text to Bluesky"""
        try:
            # Create the post
            post = models.AppBskyFeedPost.Main(
                text=text,
                created_at=datetime.now().isoformat()
            )
            
            # Send the post
            self.client.send_post(post)
            print("Successfully posted to Bluesky")
            return True
            
        except Exception as e:
            print(f"Error posting to Bluesky: {e}")
            return False
    
    def process_summaries(self, max_posts: int = 5):
        """Process summaries and post them to Bluesky"""
        summaries = self.extract_summaries_from_log()
        
        if not summaries:
            print("No summaries found in the log file")
            return
            
        print(f"Found {len(summaries)} summaries to process")
        
        # Limit the number of posts
        summaries = summaries[:max_posts]
        
        for i, paper in enumerate(summaries):
            title = paper['title']
            print(f"\nProcessing paper {i+1}/{len(summaries)}: {title}")
            
            # Format the post
            post_text = self.format_post(paper)
            
            # Find image prompt if available
            image_prompt = self.find_image_prompt(title)
            if image_prompt:
                print(f"Found image prompt for: {title}")
                # You could add the image prompt to the post if desired
                # post_text += f"\n\nImage prompt: {image_prompt}"
            
            # Post to Bluesky
            success = self.post_to_bluesky(post_text)
            
            if success:
                print(f"Successfully posted summary for: {title}")
            else:
                print(f"Failed to post summary for: {title}")
                
            # Add a delay between posts to avoid rate limiting
            if i < len(summaries) - 1:
                print("Waiting 5 seconds before next post...")
                time.sleep(5)

def main():
    # Get credentials from command line or environment variables
    import argparse
    
    parser = argparse.ArgumentParser(description='Post paper summaries to Bluesky')
    parser.add_argument('--summary-file', default='summary_output.log', help='Path to the summary log file')
    parser.add_argument('--image-prompts-dir', default='paper_images', help='Directory containing image prompt files')
    parser.add_argument('--handle', help='Bluesky handle (username)')
    parser.add_argument('--password', help='Bluesky password')
    parser.add_argument('--max-posts', type=int, default=1, help='Maximum number of posts to make')
    
    args = parser.parse_args()
    
    try:
        poster = BlueskyPoster(
            summary_file=args.summary_file,
            image_prompts_dir=args.image_prompts_dir,
            handle=args.handle,
            password=args.password
        )
        
        poster.process_summaries(max_posts=args.max_posts)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 