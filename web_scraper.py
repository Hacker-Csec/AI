import asyncio
import aiohttp
import logging
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, quote_plus
from typing import Set, List, Dict, Optional
from dataclasses import dataclass
import json
from pathlib import Path
import re
from datetime import datetime
import time
import random
from fake_useragent import UserAgent
import aiofiles
import tqdm.asyncio as tqdm
import sys
import socks
import socket
import requests
from stem import Signal
from stem.control import Controller
import os

@dataclass
class ScrapingConfig:
    max_depth: int = 2
    max_pages: int = 100
    delay: float = 1.0  # Delay between requests in seconds
    timeout: int = 30
    max_retries: int = 3
    output_file: str = 'scraped.json'  # JSON file for scraped data
    allowed_domains: Optional[List[str]] = None
    excluded_paths: List[str] = None
    min_text_length: int = 100  # Minimum length of text to save
    use_tor: bool = False
    tor_password: Optional[str] = None
    tor_control_port: int = 9051
    tor_socks_port: int = 9050

class WebScraper:
    def __init__(self, config: ScrapingConfig):
        self.config = config
        self.visited_urls: Set[str] = set()
        self.scraped_data: List[Dict] = []
        self.session: Optional[aiohttp.ClientSession] = None
        self.ua = UserAgent()
        self._setup_logging()
        self._setup_tor()
        
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _setup_tor(self):
        """Setup Tor connection if enabled."""
        if self.config.use_tor:
            try:
                # Set up SOCKS proxy
                socks.set_default_proxy(
                    socks.SOCKS5,
                    "127.0.0.1",
                    self.config.tor_socks_port
                )
                socket.socket = socks.socksocket
                
                # Test Tor connection
                response = requests.get('https://check.torproject.org/')
                if 'Congratulations' in response.text:
                    print("✓ Successfully connected to Tor network")
                else:
                    print("! Connected to internet but not through Tor")
            except Exception as e:
                print(f"! Error setting up Tor: {str(e)}")
                print("Continuing without Tor...")
                self.config.use_tor = False

    def _renew_tor_identity(self):
        """Renew Tor identity by requesting a new circuit."""
        if not self.config.use_tor:
            return
            
        try:
            with Controller.from_port(port=self.config.tor_control_port) as controller:
                if self.config.tor_password:
                    controller.authenticate(password=self.config.tor_password)
                else:
                    controller.authenticate()
                controller.signal(Signal.NEWNYM)
                print("✓ Tor identity renewed")
        except Exception as e:
            print(f"! Error renewing Tor identity: {str(e)}")

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and allowed."""
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
            
            # Check if domain is allowed
            if self.config.allowed_domains:
                if not any(domain in parsed.netloc for domain in self.config.allowed_domains):
                    return False
            
            # Check if path is excluded
            if self.config.excluded_paths:
                if any(path in parsed.path for path in self.config.excluded_paths):
                    return False
            
            return True
        except Exception:
            return False

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()

    def _is_disclaimer(self, text: str) -> bool:
        """Check if text is a disclaimer or similar content."""
        disclaimer_phrases = [
            # Legal and ethical disclaimers
            "disclaimer", "terms of service", "terms and conditions",
            "privacy policy", "legal notice", "copyright notice",
            "all rights reserved", "this content is for informational purposes only",
            "the information provided is not legal advice",
            "we are not responsible for", "use at your own risk",
            "by using this site", "by accessing this site",
            "please read these terms carefully", "this website is not affiliated with",
            
            # Ethical and responsibility disclaimers
            "this is not official", "this is not endorsed by",
            "this is not sponsored by", "this is not authorized by",
            "this is not approved by", "this is not verified by",
            "this is not certified by", "this is not validated by",
            "this is not authenticated by", "this is not confirmed by",
            
            # Legal compliance phrases
            "compliance with", "in accordance with", "as required by",
            "in compliance with", "subject to", "pursuant to",
            "under the law", "according to law", "legal requirements",
            
            # Ethical and responsibility phrases
            "ethical considerations", "responsible use", "proper use",
            "appropriate use", "authorized use", "permitted use",
            "allowed use", "acceptable use", "authorized access",
            
            # Warning and caution phrases
            "warning", "caution", "notice", "attention",
            "be aware", "be advised", "please note",
            "important notice", "important information",
            
            # Legal and regulatory phrases
            "regulatory", "compliance", "legal", "lawful",
            "permitted", "authorized", "licensed", "certified",
            "verified", "validated", "authenticated", "confirmed",
            
            # Responsibility and liability phrases
            "responsible for", "liable for", "accountable for",
            "answerable for", "obligated to", "required to",
            "must comply", "should comply", "need to comply",
            
            # Permission and authorization phrases
            "with permission", "with authorization", "with approval",
            "with consent", "with agreement", "with understanding",
            "with acknowledgment", "with recognition", "with acceptance",
            
            # Legal and ethical frameworks
            "legal framework", "ethical framework", "regulatory framework",
            "compliance framework", "governance framework", "policy framework",
            "guideline framework", "standard framework", "requirement framework",
            
            # Additional safety phrases
            "safety first", "safety measures", "safety precautions",
            "safety guidelines", "safety procedures", "safety protocols",
            "safety standards", "safety requirements", "safety compliance",
            
            # Additional warning phrases
            "use with caution", "handle with care", "proceed with caution",
            "exercise caution", "take care", "be careful",
            "be cautious", "be vigilant", "be alert",
            
            # Additional legal phrases
            "legal obligation", "legal requirement", "legal duty",
            "legal responsibility", "legal liability", "legal compliance",
            "legal framework", "legal standard", "legal guideline",
            
            # Additional ethical phrases
            "ethical obligation", "ethical requirement", "ethical duty",
            "ethical responsibility", "ethical liability", "ethical compliance",
            "ethical framework", "ethical standard", "ethical guideline"
        ]
        
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in disclaimer_phrases)

    async def _extract_content(self, html: str, url: str) -> Optional[Dict]:
        """Extract relevant content from HTML."""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove unwanted elements
            for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            # Extract main content
            main_content = []
            
            # Try to find main content area
            main_tags = soup.find_all(['article', 'main', 'div'], class_=re.compile(r'content|main|article', re.I))
            if main_tags:
                for tag in main_tags:
                    text = self._clean_text(tag.get_text())
                    if len(text) >= self.config.min_text_length and not self._is_disclaimer(text):
                        main_content.append(text)
            
            # If no main content found, get all paragraphs
            if not main_content:
                paragraphs = soup.find_all('p')
                for p in paragraphs:
                    text = self._clean_text(p.get_text())
                    if len(text) >= self.config.min_text_length and not self._is_disclaimer(text):
                        main_content.append(text)
            
            if not main_content:
                return None
            
            # Format content for AI training
            content = '\n'.join(main_content)
            word_count = sum(len(text.split()) for text in main_content)
            
            # Generate a generic title
            entry_number = len(self.scraped_data) + 1
            generic_title = f"Data Entry {entry_number}"
            
            # Create a more structured output for AI processing
            return {
                "metadata": {
                    "entry_id": entry_number,
                    "source_url": url,
                    "timestamp": datetime.now().isoformat(),
                    "word_count": word_count
                },
                "content": {
                    "title": generic_title,
                    "main_text": content,
                    "summary": content[:500] + "..." if len(content) > 500 else content
                },
                "analysis": {
                    "key_topics": [topic.strip() for topic in content.split('.')[:5] if len(topic.strip()) > 20],
                    "estimated_reading_time": f"{word_count // 200 + 1} minutes",
                    "content_quality": "high" if word_count > 500 else "medium"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting content from {url}: {str(e)}")
            return None

    async def _fetch_page(self, url: str) -> Optional[str]:
        """Fetch page content with retries and error handling."""
        headers = {'User-Agent': self.ua.random}
        
        # Configure proxy if using Tor
        proxy = None
        if self.config.use_tor:
            proxy = f"socks5://127.0.0.1:{self.config.tor_socks_port}"
        
        for attempt in range(self.config.max_retries):
            try:
                async with self.session.get(
                    url,
                    headers=headers,
                    timeout=self.config.timeout,
                    proxy=proxy
                ) as response:
                    if response.status == 200:
                        return await response.text()
                    elif response.status == 429:  # Too Many Requests
                        wait_time = int(response.headers.get('Retry-After', self.config.delay * 2))
                        await asyncio.sleep(wait_time)
                        if self.config.use_tor:
                            self._renew_tor_identity()
                    else:
                        self.logger.warning(f"Failed to fetch {url}: Status {response.status}")
                        return None
            except Exception as e:
                self.logger.error(f"Error fetching {url} (attempt {attempt + 1}): {str(e)}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.delay * (attempt + 1))
                    if self.config.use_tor:
                        self._renew_tor_identity()
        
        return None

    async def _process_url(self, url: str, depth: int = 0) -> List[str]:
        """Process a single URL and extract links."""
        if depth > self.config.max_depth or len(self.visited_urls) >= self.config.max_pages:
            return []
        
        if url in self.visited_urls or not self._is_valid_url(url):
            return []
        
        self.visited_urls.add(url)
        print(f"\nProcessing {url} (depth: {depth})")
        
        # Add random delay
        await asyncio.sleep(self.config.delay + random.random())
        
        html = await self._fetch_page(url)
        if not html:
            print(f"! Failed to fetch content from {url}")
            return []
        
        # Extract content
        content = await self._extract_content(html, url)
        if content:
            print(f"✓ Extracted content from {url}:")
            print(f"  Title: {content['content']['title']}")
            print(f"  Word count: {content['metadata']['word_count']}")
            self.scraped_data.append(content)
            print(f"Current scraped data size: {len(self.scraped_data)} items")
        else:
            print(f"! No content extracted from {url}")
        
        # Extract links
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            absolute_url = urljoin(url, href)
            if self._is_valid_url(absolute_url):
                links.append(absolute_url)
        
        return links

    async def _save_data(self):
        """Save scraped data to file and merge with training data."""
        try:
            print(f"\nSaving {len(self.scraped_data)} items to {self.config.output_file}")
            
            # Initialize empty list if file doesn't exist
            if not os.path.exists(self.config.output_file):
                print(f"Creating new file: {self.config.output_file}")
                async with aiofiles.open(self.config.output_file, 'w', encoding='utf-8') as f:
                    await f.write('[]')
            
            # Read existing data
            try:
                async with aiofiles.open(self.config.output_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    existing_data = json.loads(content) if content.strip() else []
                print(f"Found {len(existing_data)} existing items in {self.config.output_file}")
            except json.JSONDecodeError as e:
                print(f"Error reading existing data: {str(e)}")
                existing_data = []
            
            # Append new data
            existing_data.extend(self.scraped_data)
            print(f"Total items after adding new data: {len(existing_data)}")
            
            # Write back all data
            async with aiofiles.open(self.config.output_file, 'w', encoding='utf-8') as f:
                json_data = json.dumps(existing_data, ensure_ascii=False, indent=4)
                await f.write(json_data)
                print(f"Successfully wrote {len(json_data)} bytes to {self.config.output_file}")
            
            print(f"\n✓ Added {len(self.scraped_data)} items to {self.config.output_file}")
            
            # Verify the data was written correctly
            try:
                async with aiofiles.open(self.config.output_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    saved_data = json.loads(content)
                    print(f"Verified {len(saved_data)} items in {self.config.output_file}")
            except Exception as e:
                print(f"! Error verifying saved data: {str(e)}")
            
            # Merge with training data
            await self._merge_with_training_data()
            
            # Check for and merge backup files
            await self._merge_backup_files()
            
        except Exception as e:
            print(f"Error saving data: {str(e)}")
            # Try to save to a backup file
            try:
                backup_file = f"backup_{int(time.time())}.json"
                async with aiofiles.open(backup_file, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(self.scraped_data, ensure_ascii=False, indent=4))
                print(f"✓ Saved backup to {backup_file}")
            except Exception as backup_error:
                print(f"Failed to save backup: {str(backup_error)}")

    async def _merge_with_training_data(self):
        """Merge scraped data with training data while preventing duplicates."""
        try:
            training_file = Path("training_data/training_data.json")
            if not training_file.exists():
                training_file.parent.mkdir(exist_ok=True)
                async with aiofiles.open(training_file, 'w', encoding='utf-8') as f:
                    await f.write('[]')
            
            # Read existing training data
            async with aiofiles.open(training_file, 'r', encoding='utf-8') as f:
                content = await f.read()
                training_data = json.loads(content) if content.strip() else []
            
            print(f"\nFound {len(training_data)} existing items in training data")
            
            # Track existing URLs to prevent duplicates
            existing_urls = set()
            
            # First, collect existing URLs from training data
            for item in training_data:
                if isinstance(item, dict):
                    # Check both old and new format
                    url = None
                    if 'metadata' in item and 'source_url' in item['metadata']:
                        url = item['metadata']['source_url']
                    elif 'input' in item and isinstance(item['input'], dict) and 'url' in item['input']:
                        url = item['input']['url']
                    
                    if url:
                        existing_urls.add(url)
            
            # Add new items that aren't duplicates
            new_items = []
            for item in self.scraped_data:
                if not isinstance(item, dict):
                    continue
                
                url = item.get('metadata', {}).get('source_url', '')
                if url and url not in existing_urls:
                    new_items.append(item)
                    existing_urls.add(url)
            
            if new_items:
                # Add new items to training data
                training_data.extend(new_items)
                
                # Write updated training data
                async with aiofiles.open(training_file, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(training_data, ensure_ascii=False, indent=4))
                
                print(f"✓ Added {len(new_items)} new items to training data")
            else:
                print("! No new items to add to training data (all were duplicates)")
            
        except Exception as e:
            print(f"! Error merging with training data: {str(e)}")
            print("Continuing without updating training data...")

    async def _merge_backup_files(self):
        """Merge any backup files with the main data file."""
        try:
            # Find all backup files
            backup_files = [f for f in os.listdir('.') if f.startswith('backup_') and f.endswith('.json')]
            
            if not backup_files:
                return
                
            print("\nChecking for backup files to merge...")
            
            # Read main data file
            try:
                async with aiofiles.open(self.config.output_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    main_data = json.loads(content) if content.strip() else []
            except json.JSONDecodeError:
                main_data = []
            
            # Track URLs to avoid duplicates
            existing_urls = {item.get('input', {}).get('url', '') for item in main_data}
            new_items = []
            
            # Process each backup file
            for backup_file in backup_files:
                try:
                    async with aiofiles.open(backup_file, 'r', encoding='utf-8') as f:
                        backup_content = await f.read()
                        backup_data = json.loads(backup_content) if backup_content.strip() else []
                    
                    # Add items that aren't already in main data
                    for item in backup_data:
                        url = item.get('input', {}).get('url', '')
                        if url and url not in existing_urls:
                            new_items.append(item)
                            existing_urls.add(url)
                    
                    # Remove processed backup file
                    os.remove(backup_file)
                    print(f"✓ Merged and removed {backup_file}")
                    
                except Exception as e:
                    print(f"! Error processing backup file {backup_file}: {str(e)}")
            
            if new_items:
                # Add new items to main data
                main_data.extend(new_items)
                
                # Write back all data
                async with aiofiles.open(self.config.output_file, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(main_data, ensure_ascii=False, indent=4))
                
                print(f"✓ Added {len(new_items)} new items from backup files")
            
        except Exception as e:
            print(f"! Error merging backup files: {str(e)}")

    async def scrape(self, start_url: str):
        """Start the scraping process."""
        async with aiohttp.ClientSession() as session:
            self.session = session
            
            # Process URLs in batches
            urls_to_process = [start_url]
            depth = 0
            
            try:
                while urls_to_process and depth <= self.config.max_depth:
                    current_batch = urls_to_process[:self.config.max_pages - len(self.visited_urls)]
                    urls_to_process = urls_to_process[len(current_batch):]
                    
                    tasks = [self._process_url(url, depth) for url in current_batch]
                    new_links = await asyncio.gather(*tasks)
                    
                    # Add new links to process
                    for links in new_links:
                        urls_to_process.extend(links)
                    
                    depth += 1
                
                # Save data after scraping is complete
                if self.scraped_data:
                    print(f"\nScraping complete. Saving {len(self.scraped_data)} items...")
                    print("Sample of first item:")
                    if self.scraped_data:
                        first_item = self.scraped_data[0]
                        print(f"URL: {first_item['metadata']['source_url']}")
                        print(f"Title: {first_item['content']['title']}")
                        print(f"Content preview: {first_item['content']['main_text'][:200]}...")
                    await self._save_data()
                    # After saving, ensure all data is accessible
                    await self._merge_backup_files()
                else:
                    print("\n! No data was scraped")
            except Exception as e:
                print(f"\nAn error occurred during scraping: {str(e)}")
                # Try to save any data we have
                if self.scraped_data:
                    print(f"Saving {len(self.scraped_data)} items after error...")
                    await self._save_data()
                    await self._merge_backup_files()

def get_user_input() -> str:
    """Get website URL from user input."""
    while True:
        url = input("\nEnter the website URL to scrape (e.g., https://example.com or http://example.onion): ").strip()
        
        # Add https:// if not present
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        try:
            parsed = urlparse(url)
            if parsed.scheme and parsed.netloc:
                return url
            else:
                print("Invalid URL format. Please enter a valid website URL.")
        except Exception:
            print("Invalid URL format. Please enter a valid website URL.")

async def main():
    print("\n=== Web Scraper ===")
    print("This program will scrape content from a website and save it to 'scraped.json'")
    print("The scraper will follow links within the same domain up to 2 levels deep")
    print("Press Ctrl+C at any time to stop the scraping process\n")
    
    # Get URL from user
    start_url = get_user_input()
    
    # Check if it's an .onion address
    is_onion = '.onion' in start_url
    
    # Extract domain for allowed_domains
    domain = urlparse(start_url).netloc
    
    # Create configuration
    config = ScrapingConfig(
        max_depth=2,
        max_pages=100,
        delay=1.0,
        timeout=30,
        max_retries=3,
        output_file='scraped.json',  # JSON file for scraped data
        allowed_domains=[domain],
        excluded_paths=['/login', '/signup', '/admin', '/logout', '/register'],
        min_text_length=100,
        use_tor=is_onion,  # Automatically use Tor for .onion addresses
        tor_password=os.getenv('TOR_PASSWORD'),  # Get Tor password from environment variable
        tor_control_port=9051,
        tor_socks_port=9050
    )
    
    try:
        scraper = WebScraper(config)
        print(f"\nStarting to scrape {start_url}")
        if is_onion:
            print("Using Tor network for .onion address")
        print("This may take a few minutes...\n")
        await scraper.scrape(start_url)
    except KeyboardInterrupt:
        print("\n\nScraping stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main()) 