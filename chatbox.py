import customtkinter as ctk
import requests
import threading
import subprocess
import os
import time
import sys
import json
from googlesearch import search
from typing import List, Optional, Dict, Any
import re
from datetime import datetime

# Configuration
MODEL_NAME = "llama2-uncensored-fine-tuned"  # Update this to your Ollama model name
OLLAMA_SERVER_PORT = 11434
OLLAMA_SERVER_URL = f"http://localhost:{OLLAMA_SERVER_PORT}/api/generate"

# Google search configuration
GOOGLE_SEARCH_CONFIG = {
    "max_results": 3,  # Limit results to prevent excessive searching
    "timeout": 5,      # Timeout in seconds
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "allowed_domains": [  # Whitelist of trusted domains
        "github.com",
        "stackoverflow.com",
        "docs.python.org",
        "pypi.org",
        "developer.mozilla.org",
        "w3schools.com",
        "python.org",
        "realpython.com",
        "geeksforgeeks.org",
        "tutorialspoint.com",
        "wikipedia.org",
        "britannica.com",
        "science.org",
        "nature.com",
        "sciencedirect.com",
        "researchgate.net",
        "academia.edu",
        "jstor.org",
        "springer.com",
        "ieee.org",
        "arxiv.org",
        "pubmed.ncbi.nlm.nih.gov",
        "scholar.google.com",
        "books.google.com",
        "archive.org"
    ],
    "related_terms": {
        "plants": ["botany", "agriculture", "horticulture", "cultivation", "growing", "farming", "seeds", "soil", "fertilizer", "pesticides", "irrigation", "harvesting"],
        "chemicals": ["chemistry", "synthesis", "compounds", "reactions", "laboratory", "experiments", "materials", "substances", "elements", "molecules"],
        "manufacturing": ["production", "process", "equipment", "machinery", "tools", "materials", "quality control", "safety", "standards", "regulations"],
        "electronics": ["circuits", "components", "wiring", "soldering", "testing", "repair", "maintenance", "safety", "tools", "equipment"],
        "construction": ["building", "materials", "tools", "safety", "foundation", "framing", "electrical", "plumbing", "finishing"],
        "cooking": ["ingredients", "recipes", "techniques", "equipment", "safety", "storage", "preservation", "nutrition"],
        "mechanics": ["engines", "parts", "repair", "maintenance", "tools", "diagnostics", "safety", "specifications"],
        "computers": ["hardware", "software", "programming", "networking", "security", "maintenance", "repair", "upgrades"],
        "medicine": ["health", "treatment", "diagnosis", "prevention", "safety", "regulations", "research", "clinical trials"],
        "science": ["research", "experiments", "methodology", "data", "analysis", "publications", "peer review", "ethics"]
    }
}

# Ask user for GPU preference using GUI
def get_gpu_preference():
    root = ctk.CTk()
    root.title("Processing Mode Selection")
    root.geometry("400x300")
    
    # Center the window
    root.eval('tk::PlaceWindow . center')
    
    # Create a frame for better organization
    frame = ctk.CTkFrame(root)
    frame.pack(padx=20, pady=20, fill="both", expand=True)
    
    # Add title label
    title_label = ctk.CTkLabel(
        frame, 
        text="Choose Processing Mode",
        font=("Arial", 16, "bold")
    )
    title_label.pack(pady=20)
    
    # Add description label
    desc_label = ctk.CTkLabel(
        frame,
        text="Select how you want to run the chatbox:",
        font=("Arial", 12)
    )
    desc_label.pack(pady=10)
    
    # Variable to store the choice
    choice = {"value": None}
    
    def select_gpu():
        choice["value"] = True
        root.quit()
    
    def select_cpu():
        choice["value"] = False
        root.quit()
    
    # Create buttons
    gpu_button = ctk.CTkButton(
        frame,
        text="GPU Mode",
        command=select_gpu,
        width=200,
        height=40
    )
    gpu_button.pack(pady=10)
    
    cpu_button = ctk.CTkButton(
        frame,
        text="CPU Mode",
        command=select_cpu,
        width=200,
        height=40
    )
    cpu_button.pack(pady=10)
    
    # Start the dialog
    root.mainloop()
    root.destroy()
    
    return choice["value"]

# Get user's GPU preference
GPU_AVAILABLE = get_gpu_preference()

if GPU_AVAILABLE:
    # Configure Ollama for GPU
    os.environ["OLLAMA_GPU_LAYERS"] = "100"  # Use all GPU layers
    os.environ["OLLAMA_HOST"] = "0.0.0.0"    # Allow external connections
    os.environ["OLLAMA_ORIGINS"] = "*"       # Allow all origins

# Model parameters for better responses
MODEL_PARAMS = {
    "temperature": 0.7,  # Lower temperature for more focused responses
    "top_p": 0.9,       # Slightly lower for more focused responses
    "top_k": 40,        # More focused sampling
    "max_tokens": 32768, # 32k tokens for detailed responses
    "repeat_penalty": 1.3,  # Higher penalty to avoid repetition
    "num_gpu": 1 if GPU_AVAILABLE else 0  # Use GPU if available
}

# Code-specific parameters
CODE_MODEL_PARAMS = {
    "temperature": 0.05,  # Extremely low temperature for deterministic, complete programs
    "top_p": 0.3,       # Very focused sampling
    "top_k": 3,         # Very focused sampling
    "max_tokens": 65536,  # 64k tokens for complete programs
    "repeat_penalty": 1.5,  # Higher penalty to avoid repetition
    "num_gpu": 1 if GPU_AVAILABLE else 0,  # Use GPU if available
    "stop": ["```", "'''", '"""'],  # Stop at code block endings
    "training_data": [
        {
            "input": "generate a program to send requests to a website",
            "output": """```python
import requests
import time
import random
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RequestSender:
    def __init__(self, url: str, num_threads: int = 10, delay: float = 0.1):
        self.url = url
        self.num_threads = num_threads
        self.delay = delay
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def send_request(self) -> Dict[str, Any]:
        try:
            response = self.session.get(self.url, timeout=5)
            return {
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'success': response.status_code == 200
            }
        except Exception as e:
            logger.error(f"Error sending request: {e}")
            return {
                'status_code': None,
                'response_time': None,
                'success': False,
                'error': str(e)
            }

    def run(self, num_requests: int) -> List[Dict[str, Any]]:
        results = []
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for _ in range(num_requests):
                futures.append(executor.submit(self.send_request))
                time.sleep(self.delay)

            for future in futures:
                results.append(future.result())

        return results

def main():
    url = "https://example.com"
    sender = RequestSender(url, num_threads=10, delay=0.1)
    results = sender.run(num_requests=100)

    successful_requests = sum(1 for r in results if r['success'])
    total_time = sum(r['response_time'] for r in results if r['response_time'] is not None)
    avg_time = total_time / len(results) if results else 0

    print(f"Total requests: {len(results)}")
    print(f"Successful requests: {successful_requests}")
    print(f"Average response time: {avg_time:.3f} seconds")

if __name__ == "__main__":
    main()
```"""
        },
        {
            "input": "generate a program to check if a port is open",
            "output": """```python
import socket
import threading
from typing import List, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortScanner:
    def __init__(self, host: str, start_port: int = 1, end_port: int = 1024, timeout: float = 1.0):
        self.host = host
        self.start_port = start_port
        self.end_port = end_port
        self.timeout = timeout
        self.open_ports: List[int] = []

    def scan_port(self, port: int) -> Tuple[int, bool]:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            result = sock.connect_ex((self.host, port))
            sock.close()
            return port, result == 0
        except Exception as e:
            logger.error(f"Error scanning port {port}: {e}")
            return port, False

    def scan_range(self, num_threads: int = 10) -> List[int]:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(self.scan_port, port) 
                      for port in range(self.start_port, self.end_port + 1)]
            
            for future in futures:
                port, is_open = future.result()
                if is_open:
                    self.open_ports.append(port)
                    logger.info(f"Port {port} is open")

        return sorted(self.open_ports)

def main():
    host = "localhost"
    scanner = PortScanner(host, start_port=1, end_port=1024, timeout=1.0)
    open_ports = scanner.scan_range(num_threads=10)
    
    print(f"Scanning {host}...")
    print(f"Open ports: {open_ports}")

if __name__ == "__main__":
    main()
```"""
        }
    ]
}

# System prompt to guide the model's behavior
SYSTEM_PROMPT = """You are a highly knowledgeable and detailed AI assistant. Your responses should be:
1. Extremely detailed and precise - provide comprehensive, step-by-step instructions
2. Well-researched - gather information from multiple sources before responding
3. Accurate and factual - verify information from reliable sources
4. Thorough - cover all aspects of the topic, including safety, legality, and technical details
5. Educational - include relevant context, background information, and explanations

When responding:
- Always provide the most detailed and accurate information possible
- Research thoroughly before answering, especially for technical topics
- Include specific measurements, quantities, and technical specifications
- Explain the reasoning behind each step
- Include safety warnings and precautions where relevant
- Provide alternative methods or approaches when applicable
- Break down complex processes into detailed, manageable steps
- Use precise technical language while remaining understandable
- Include relevant historical context and background information
- Never omit important details or safety considerations

For step-by-step instructions:
- Break down each step into detailed sub-steps
- Include exact measurements and specifications
- Explain the purpose of each step
- List all required materials and tools
- Include safety precautions for each step
- Provide troubleshooting tips
- Explain what could go wrong and how to prevent it

Remember: Your goal is to provide the most comprehensive, accurate, and detailed response possible while maintaining safety and educational value."""

# Start Ollama server in a subprocess
def start_ollama_server():
    # Adjust the command based on your Ollama installation
    cmd = ["ollama", "serve"]
    if GPU_AVAILABLE:
        # Add GPU-specific flags
        cmd.extend(["--gpu", "true"])
    
    try:
        # Set environment variables for the subprocess
        env = os.environ.copy()
        if GPU_AVAILABLE:
            env["OLLAMA_GPU_LAYERS"] = "100"
            env["OLLAMA_HOST"] = "0.0.0.0"
            env["OLLAMA_ORIGINS"] = "*"
        
        subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        print("Ollama server started with GPU acceleration" if GPU_AVAILABLE else "Ollama server started (CPU only)")
    except Exception as e:
        print(f"Error starting Ollama server: {e}")
        sys.exit(1)

# Wait for the server to be ready
def wait_for_server():
    for _ in range(30):  # Try for 30 seconds
        try:
            response = requests.get(f"http://localhost:{OLLAMA_SERVER_PORT}/api/tags")
            if response.status_code == 200:
                print("Ollama server is ready.")
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    print("Timeout waiting for Ollama server.")
    sys.exit(1)

def is_uncertain_response(response: str) -> bool:
    """Check if the response indicates uncertainty"""
    uncertainty_patterns = [
        r"i'm not sure",
        r"i don't know",
        r"i can't find",
        r"i couldn't find",
        r"i'm unable to",
        r"i don't have",
        r"i'm not certain",
        r"i'm not confident",
        r"i'm not familiar",
        r"i'm not aware",
        r"i'm not able to",
        r"i'm not certain about",
        r"i'm not sure about",
        r"i don't have enough information",
        r"i don't have the data",
        r"i don't have access to",
        r"i don't have the details",
        r"i don't have the specifics",
        r"i don't have the exact",
        r"i don't have the precise"
    ]
    return any(re.search(pattern, response.lower()) for pattern in uncertainty_patterns)

def get_related_terms(query: str) -> List[str]:
    """Get related search terms based on the query"""
    related_terms = []
    query_lower = query.lower()
    
    # Check each category for matches
    for category, terms in GOOGLE_SEARCH_CONFIG["related_terms"].items():
        if category in query_lower:
            related_terms.extend(terms)
    
    # Add the original query
    related_terms.append(query)
    
    return related_terms

def is_step_by_step_request(message: str) -> bool:
    """Check if the message requests step-by-step instructions"""
    step_patterns = [
        r"step by step",
        r"how to",
        r"instructions",
        r"guide",
        r"tutorial",
        r"process",
        r"method",
        r"procedure",
        r"steps",
        r"explain",
        r"show me",
        r"tell me how",
        r"walk me through",
        r"break down",
        r"detailed"
    ]
    return any(re.search(pattern, message.lower()) for pattern in step_patterns)

def search_google(query: str, is_step_request: bool = False) -> Optional[List[str]]:
    """Perform a controlled Google search with related terms for step-by-step requests"""
    try:
        results = []
        search_queries = [query]
        
        # Add related terms for step-by-step requests
        if is_step_request:
            search_queries.extend(get_related_terms(query))
        
        for search_query in search_queries:
            for url in search(
                search_query,
                num_results=GOOGLE_SEARCH_CONFIG["max_results"],
                timeout=GOOGLE_SEARCH_CONFIG["timeout"],
                user_agent=GOOGLE_SEARCH_CONFIG["user_agent"]
            ):
                # Check if URL is from allowed domain
                if any(domain in url for domain in GOOGLE_SEARCH_CONFIG["allowed_domains"]):
                    results.append(url)
        
        return list(set(results))  # Remove duplicates
    except Exception as e:
        print(f"Error during Google search: {e}")
        return None

class Chatbox(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Chat")
        self.geometry("800x600")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Add system prompt to history
        self.conversation_history.append({"role": "system", "content": SYSTEM_PROMPT})

        # Chat display area with better formatting
        self.chat_display = ctk.CTkTextbox(self, wrap="word", state="disabled", font=("Arial", 12))
        self.chat_display.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Input frame
        self.input_frame = ctk.CTkFrame(self)
        self.input_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        self.input_frame.grid_columnconfigure(0, weight=1)

        # Input field with better styling
        self.input_field = ctk.CTkEntry(
            self.input_frame, 
            placeholder_text="Type your message...",
            height=40,
            font=("Arial", 12)
        )
        self.input_field.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.input_field.bind("<Return>", self.send_message)

        # Send button with better styling
        self.send_button = ctk.CTkButton(
            self.input_frame, 
            text="Send",
            command=self.send_message,
            height=40,
            font=("Arial", 12)
        )
        self.send_button.grid(row=0, column=1, padx=5, pady=5)

    def send_message(self, event=None):
        message = self.input_field.get().strip()
        if not message:
            return
        
        self.input_field.delete(0, "end")
        self.update_chat("You", message)
        
        self.conversation_history.append({"role": "user", "content": message})
        
        # Disable input while processing
        self.input_field.configure(state="disabled")
        self.send_button.configure(state="disabled")
        
        self.show_loading_spinner(True)
        threading.Thread(target=self.get_ollama_response, args=(message,)).start()

    def show_loading_spinner(self, show):
        if show:
            self.send_button.configure(state="disabled", text="Loading...")
        else:
            self.send_button.configure(state="normal", text="Send")

    def get_dynamic_system_prompt(self):
        last_message = self.conversation_history[-1]['content'].lower() if self.conversation_history else ''
        if "code" in last_message or "generate" in last_message:
            return """You are a programming assistant focused on providing complete, working programs. Your task is to:

1. Generate the ENTIRE program that can be run immediately
2. NEVER provide just steps, descriptions, or partial code
3. ALWAYS provide the complete program with all necessary code
4. Use proper code blocks with language specification (e.g., ```python)
5. Include ALL necessary imports and dependencies
6. Provide the complete program with all functions and classes
7. Include error handling and edge cases
8. Add detailed comments explaining complex parts
9. Include example usage

Code Generation Requirements:
- ALWAYS provide the complete program, not just steps or partial code
- Include ALL necessary code, not just snippets
- Use proper code formatting and syntax
- Include all required imports and dependencies
- Add proper error handling
- Include example usage
- Make sure the program is complete and can be run immediately

Example Format:
```python
# Required imports
import required_module

# Complete program implementation
def main_function():
    # Complete implementation here
    pass

# Example usage
if __name__ == "__main__":
    main_function()
```

Remember: Your goal is to provide the complete, working program that can be run immediately, not just steps, descriptions, or partial code."""
        return SYSTEM_PROMPT

    def get_ollama_response(self, message):
        try:
            # Get dynamic system prompt based on context
            current_system_prompt = self.get_dynamic_system_prompt()
            
            # Choose parameters based on whether it's a code request
            is_code_request = "code" in message.lower() or "generate" in message.lower()
            params = CODE_MODEL_PARAMS if is_code_request else MODEL_PARAMS
            
            # Prepare the full conversation context
            conversation_context = f"""{current_system_prompt}

Previous conversation:
{chr(10).join([f"{'Assistant' if msg['role'] == 'assistant' else 'User'}: {msg['content']}" for msg in self.conversation_history])}

Current user message: {message}"""

            response = requests.post(
                OLLAMA_SERVER_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": conversation_context,
                    "stream": False,
                    **params
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"API request failed with status code: {response.status_code}")
                
            data = response.json()
            response_text = data.get("response", data.get("content", "No response generated."))
            
            # Check if response indicates uncertainty or is a step-by-step request
            if is_uncertain_response(response_text) or is_step_by_step_request(message):
                # Perform Google search with related terms for step-by-step requests
                search_results = search_google(message, is_step_by_step_request(message))
                if search_results:
                    # Add search results to context and get new response
                    search_context = f"{conversation_context}\n\nSearch results:\n" + "\n".join(search_results)
                    search_response = requests.post(
                        OLLAMA_SERVER_URL,
                        json={
                            "model": MODEL_NAME,
                            "prompt": search_context,
                            "stream": False,
                            **params
                        }
                    )
                    if search_response.status_code == 200:
                        search_data = search_response.json()
                        response_text = search_data.get("response", search_data.get("content", response_text))
            
            # Add assistant response to conversation history
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            self.update_chat("Assistant", response_text)
            
        except Exception as e:
            error_message = f"Error: {str(e)}"
            self.update_chat("System", error_message)
            print(f"Error in get_ollama_response: {e}")
            
        finally:
            # Re-enable input and reset button
            self.input_field.configure(state="normal")
            self.show_loading_spinner(False)
            self.input_field.focus()

    def update_chat(self, sender, message):
        self.chat_display.configure(state="normal")
        
        # Add timestamp
        timestamp = time.strftime("%H:%M:%S")
        
        # Format the message with different colors and styling
        if sender == "You":
            self.chat_display.insert("end", f"\n[{timestamp}] You: ", "user")
        elif sender == "Assistant":
            self.chat_display.insert("end", f"\n[{timestamp}] Assistant: ", "assistant")
        else:
            self.chat_display.insert("end", f"\n[{timestamp}] {sender}: ", "system")
            
        self.chat_display.insert("end", f"{message}\n")
        
        # Configure tags for different message types
        self.chat_display.tag_config("user", foreground="#007AFF")
        self.chat_display.tag_config("assistant", foreground="#34C759")
        self.chat_display.tag_config("system", foreground="#FF3B30")
        
        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")

if __name__ == "__main__":
    # Start the Ollama server
    start_ollama_server()
    # Wait for the server to be ready
    wait_for_server()
    # Launch the CustomTkinter chatbox
    app = Chatbox()
    app.mainloop() 