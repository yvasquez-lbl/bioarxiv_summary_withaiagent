import os
import json
import requests
from openai import OpenAI
from typing import Dict, List, Optional
import time
import re

class PaperImageGenerator:
    def __init__(self, log_file: str = "paper_notifications.log", output_dir: str = "paper_images"):
        self.log_file = log_file
        self.output_dir = output_dir
        self.base_url = "https://api.biorxiv.org"
        self.client = OpenAI(
            api_key=os.environ.get('CBORG_API_KEY'),
            base_url="https://api.cborg.lbl.gov"
        )
        self.text_model = "lbl/cborg-chat:latest"  # For text generation
        self.image_model = "lbl/cborg-vision:latest"  # For image generation
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
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
            
    def generate_image_prompt(self, paper_data: Dict) -> str:
        """Generate a prompt for image generation based on the paper data"""
        try:
            title = paper_data.get('title', '')
            abstract = paper_data.get('abstract', '')
            
            prompt = f"""Based on the following scientific paper title and abstract, generage an image that visually represents the key concepts:

Title: {title}

Abstract: {abstract}

Create a simple yet graphically pleasing image generation prompt that captures the essence of this scientific paper. The prompt should be specific, descriptive, and suitable for an AI image generation model. Focus on the visual elements that would best represent the paper's main findings or concepts."""

            response = self.client.chat.completions.create(
                model=self.text_model,
                messages=[
                    {"role": "system", "content": "You are an expert at creating detailed image generation prompts for scientific papers. Your prompts should be specific, descriptive, and focus on the visual elements that best represent the paper's main findings."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating image prompt: {e}")
            return f"Scientific illustration of {paper_data.get('title', 'Unknown Title')}"

    def generate_image(self, prompt: str, title: str) -> Optional[str]:
        """Generate an image using the vision model"""
        try:
            # Create a safe filename from the title
            safe_title = re.sub(r'[^\w\s-]', '', title)
            safe_title = re.sub(r'[-\s]+', '_', safe_title).strip('-_')
            image_path = os.path.join(self.output_dir, f"{safe_title[:50]}.png")
            
            print(f"Generating image for: {title}")
            print(f"Using prompt: {prompt}")
            
            # Comment out the actual image generation code
            """
            # Call the vision model to generate the image with minimal parameters
            response = self.client.images.generate(
                model="lbl/cborg-vision:latest",
                prompt=prompt,
                n=1,
                size='256x256'
            )
            
            # Get the image URL from the response
            image_url = response.data[0].url
            
            # Download the image
            print(f"Downloading image from: {image_url}")
            img_response = requests.get(image_url)
            img_response.raise_for_status()
            
            # Save the image to a file
            with open(image_path, 'wb') as f:
                f.write(img_response.content)
            """
            
            # Instead, just save the prompt to a text file
            with open(image_path.replace('.png', '.txt'), 'w') as f:
                f.write(f"Image prompt for: {title}\n\n{prompt}")
                
            print(f"Prompt saved to: {image_path.replace('.png', '.txt')}")
            return image_path.replace('.png', '.txt')
            
        except Exception as e:
            print(f"Error generating image: {e}")
            return None
            
    def process_log_file(self):
        """Read DOIs from log file and generate images for each paper"""
        try:
            with open(self.log_file, 'r') as f:
                content = f.read()
                
            # Extract DOIs using regex
            doi_pattern = r'DOI: (10\.\d{4,9}/[-._;()/:\w]+)'
            dois = re.findall(doi_pattern, content)
            
            if not dois:
                print("No DOIs found in log file")
                return
                
            print(f"\nFound {len(dois)} papers to process")
            
            # Process each DOI
            for doi in dois:
                print(f"\nProcessing DOI: {doi}")
                paper_data = self.get_paper_by_doi(doi)
                
                if paper_data:
                    title = paper_data.get('title', 'No title')
                    print(f"\nProcessing paper: {title}")
                    
                    # Generate image prompt
                    image_prompt = self.generate_image_prompt(paper_data)
                    print(f"Generated prompt: {image_prompt}")
                    
                    # Generate image
                    image_path = self.generate_image(image_prompt, title)
                    
                    if image_path:
                        print(f"Successfully generated image for: {title}")
                    else:
                        print(f"Failed to generate image for: {title}")
                else:
                    print(f"Could not fetch paper data for DOI: {doi}")
                    
                # Add a small delay to avoid rate limiting
                time.sleep(1)
                    
        except Exception as e:
            print(f"Error processing log file: {e}")
            
def main():
    generator = PaperImageGenerator()
    generator.process_log_file()

if __name__ == "__main__":
    main() 