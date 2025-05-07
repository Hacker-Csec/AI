import json
import os
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    filename='scraped_operations.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def add_entry_to_scraped():
    try:
        # Check if temp.txt exists
        if not os.path.exists('temp.txt'):
            logging.warning("temp.txt not found")
            print("Error: temp.txt not found")
            return

        # Load existing data from scraped.json
        try:
            with open('scraped.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            data = []
        except json.JSONDecodeError:
            logging.error("Error decoding scraped.json")
            print("Error: Invalid JSON in scraped.json")
            return

        # Read content from temp.txt
        try:
            with open('temp.txt', 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logging.error(f"Error reading temp.txt: {str(e)}")
            print(f"Error reading temp.txt: {str(e)}")
            return

        # Normalize line endings and count lines/words
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        lines = content.split('\n')
        line_count = len(lines)
        word_count = len(content.split())

        # Create new entry
        new_entry = {
            "id": len(data) + 1,
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "source": "temp.txt",
                "line_count": line_count,
                "word_count": word_count
            },
            "content": content,
            "analysis": {
                "line_count": line_count,
                "word_count": word_count
            }
        }

        # Add to existing data
        data.append(new_entry)

        # Save updated data
        with open('scraped.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logging.info(f"Added entry {new_entry['id']} to scraped.json")
        print(f"Successfully added entry {new_entry['id']} to scraped.json")

        # Clear temp.txt after a short delay
        time.sleep(2)
        with open('temp.txt', 'w', encoding='utf-8') as f:
            f.write('')

    except Exception as e:
        logging.error(f"Error in add_entry_to_scraped: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    add_entry_to_scraped() 