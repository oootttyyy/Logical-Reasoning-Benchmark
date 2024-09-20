import base64
import requests
import os
import json

# OpenAI API Key
api_key = os.getenv('OPENAI_API_KEY')

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

result = []

task_list = os.listdir('../Data/data_sample/')
for task in task_list:
    if 'task_' not in task:
        continue

    task_idx = task.split('_')[-1]
    image_path = f'../Data/data_sample/{task}/image_{task_idx}.png'

    with open(f'../Data/data_sample/{task}/query_{task_idx}.jsonl','r') as f:
        lines = f.readlines()
        entry = json.loads(lines[1])
        prompt = entry['query']
        answer = entry['answer'].strip()

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    payload = {
    "model": "gpt-4o",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": prompt + "output your letter of choice after __LETTER_OF_CHOICE__",
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 600
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    response = response.json()
    response_string = response['choices'][0]['message']['content']
    response_letter = response_string.split('__LETTER_OF_CHOICE__')[-1].strip()
    result.append({
        'success': response_letter==answer, 
        'ground truth': answer, 
        'output': response_letter,
    })
    print(response)

print(result)
