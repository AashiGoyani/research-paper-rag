# import urllib.request
# import json

# def check_available_models(api_key):
#     url = f'https://generativelanguage.googleapis.com/v1beta/models?key={api_key}'
    
#     try:
#         with urllib.request.urlopen(url) as response:
#             result = json.loads(response.read().decode('utf-8'))
            
#             print("API key has access to these models:\n")
            
#             for model in result.get('models', []):
#                 model_name = model['name'].split('/')[-1]
#                 if 'gemini' in model_name.lower():
#                     print(f"  ✓ {model_name}")
            
#             return result
            
#     except Exception as e:
#         print(f"Error: {e}")
#         return None

# # Replace with YOUR actual API key
# api_key = "AIzaSyBptwMC_hzq5s9ySDvCkjYQ_zWyY_XMPX4"
# check_available_models(api_key)

import requests
from bs4 import BeautifulSoup

def decode_secret_message(url):
    """
    Decodes a secret message from a Google Doc containing Unicode characters
    and their positions in a 2D grid.
    
    Args:
        url: String containing the URL for the Google Doc with input data
    """
    # Fetch the document
    response = requests.get(url)
    response.raise_for_status()
    
    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all table rows
    rows = soup.find_all('tr')
    
    # Extract character data from table (skip header row)
    characters = []
    for row in rows[1:]:
        cells = row.find_all('td')
        if len(cells) >= 3:
            try:
                x = int(cells[0].get_text(strip=True))
                char = cells[1].get_text(strip=True)
                y = int(cells[2].get_text(strip=True))
                characters.append((x, y, char))
            except (ValueError, IndexError):
                continue
    
    # Find grid dimensions
    max_x = max(c[0] for c in characters)
    max_y = max(c[1] for c in characters)
    
    # Create 2D grid filled with spaces
    grid = [[' ' for _ in range(max_x + 1)] for _ in range(max_y + 1)]
    
    # Place each character at its (x, y) position
    for x, y, char in characters:
        grid[y][x] = char
    
    # Print the grid to reveal the secret message
    for row in reversed(grid):
        print(''.join(row), end='\n\n') 


if __name__ == "__main__":
    # Test with the provided URL
    url = "https://docs.google.com/document/u/0/d/e/2PACX-1vTMOmshQe8YvaRXi6gEPKKlsC6UpFJSMAk4mQjLm_u1gmHdVVTaeh7nBNFBRlui0sTZ-snGwZM4DBCT/pub?pli=1"
    decode_secret_message(url)