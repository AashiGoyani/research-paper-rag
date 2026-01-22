import json
import boto3
import os
import urllib.request
import numpy as np 

s3 = boto3.client('s3')
papers_cache = None

def load_papers():
    global papers_cache
    if papers_cache is None:
        try:
            bucket = 'research-paper-rec'
            obj = s3.get_object(Bucket=bucket, Key='data/processed/papers_sample_10k.json')
            lines = obj['Body'].read().decode('utf-8').split('\n')
            papers_cache = [json.loads(line) for line in lines if line.strip()]
            print(f"Loaded {len(papers_cache)} papers")
        except Exception as e:
            print(f"Error: {e}")
            papers_cache = []
    return papers_cache

def call_gemini_api(prompt):
    import time
    api_key = os.environ.get('GOOGLE_API_KEY')
    
    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}'
    
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'})
            
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result['candidates'][0]['content']['parts'][0]['text']
                
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            
            if e.code in [503, 429] and attempt < max_retries - 1:
                print(f"Retry attempt {attempt + 1}/{max_retries}")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            else:
                if e.code == 503:
                    raise Exception("The AI service is busy right now. Please wait a moment and try again!")
                elif e.code == 429:
                    raise Exception("Too many requests. Please wait 30 seconds and try again.")
                else:
                    raise Exception(f"AI service error. Please try again. (Error code: {e.code})")
                
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            else:
                raise Exception("Unable to reach AI service. Please check your connection and try again.")

def get_embedding(text):
    api_key = os.environ.get('GOOGLE_API_KEY')
    url = f'https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={api_key}'
    
    data = json.dumps({
        'content': {
            'parts': [{'text': text}]
        }
    }).encode('utf-8')
    
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    
    with urllib.request.urlopen(req, timeout=30) as response:
        result = json.loads(response.read())
        return np.array(result['embedding']['values'])

def semantic_search(query_embedding, paper_embeddings, top_k=10):
    similarities = np.dot(paper_embeddings, query_embedding) / (
        np.linalg.norm(paper_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    return top_indices.tolist(), similarities[top_indices].tolist()

def lambda_handler(event, context):
    try:
        # Handle CORS preflight
        if event.get('httpMethod') == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type'
                },
                'body': ''
            }
        
        print(f"Event: {json.dumps(event)}")
        body = json.loads(event.get('body', '{}')) if event.get('httpMethod') == 'POST' else event
        action = body.get('action')
        papers = load_papers()
        
        if action == 'explain_paper':
            paper = papers[body.get('paper_index')]
            prompt = f"Explain this research paper in simple, clear terms:\n\nTitle: {paper['title_clean']}\n\nAbstract: {paper.get('abstract', 'No abstract available')[:500]}\n\nProvide:\n1. Main idea (2-3 sentences)\n2. Key methods used\n3. Why it matters\n4. Who should read it\n\nKeep it concise and accessible."
            explanation = call_gemini_api(prompt)
            return {'statusCode': 200, 'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'}, 'body': json.dumps({'explanation': explanation, 'paper_title': paper['title_clean']})}
        
        elif action == 'compare_papers':
            indices = body.get('paper_indices', [])
            
            papers_text = ""
            for i, idx in enumerate(indices, 1):
                paper = papers[idx]
                papers_text += f"\n\nPaper {i}:\nTitle: {paper['title_clean']}\nAbstract: {paper.get('abstract', '')[:400]}"
            
            prompt = f"""Compare these {len(indices)} research papers:

        {papers_text}

        Provide a comprehensive comparison that includes:
        1. **Main Similarities** - What do all these papers have in common?
        2. **Key Differences** - How does each paper differ from the others?
        3. **Strengths & Weaknesses** - What are the unique strengths of each paper?
        4. **Best For** - Which paper is best for beginners vs experts?
        5. **Reading Order** - In what order should someone read these papers?

        Make it clear and structured."""
            
            comparison = call_gemini_api(prompt)
            return {'statusCode': 200, 'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'}, 'body': json.dumps({'comparison': comparison})}

        elif action == 'rag_search':
            query = body.get('query', '')
            
            if not query:
                return {
                    'statusCode': 400,
                    'headers': {'Access-Control-Allow-Origin': '*'},
                    'body': json.dumps({'error': 'Query is required'})
                }
            
            if 'relevant_papers' in body and body['relevant_papers']:
                indices = body.get('relevant_papers', [])[:5]
                scores = [1.0] * len(indices)
            else:
                print("Loading embeddings from S3...")
                bucket = 'research-paper-rec'
                embeddings_obj = s3.get_object(Bucket=bucket, Key='data/embeddings/paper_embeddings_10k.npy')
                paper_embeddings = np.load(embeddings_obj['Body'])
                
                print("Getting query embedding...")
                query_embedding = get_embedding(query)
                
                print("Performing semantic search...")
                indices, scores = semantic_search(query_embedding, paper_embeddings, top_k=10)
            
            if not indices:
                return {
                    'statusCode': 200,
                    'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
                    'body': json.dumps({'answer': 'I couldn\'t find papers relevant to your question. Try different keywords!'})
                }
            
            context = "\n---\n".join([
                f"Paper {i+1} (Relevance: {scores[i]:.2f}):\nTitle: {papers[idx]['title_clean']}\nAbstract: {papers[idx].get('abstract', '')[:300]}"
                for i, idx in enumerate(indices[:5])
            ])
            
            prompt = f"""Based on these research papers:

        {context}

        User question: {query}

        Provide a comprehensive answer that:
        1. Directly answers the question
        2. Lists relevant papers with their titles in bold like: **Paper Title Here**
        3. Explains key concepts clearly
        4. Do NOT include any index numbers or paper IDs in your response

        Be helpful and informative!"""
            
            answer = call_gemini_api(prompt)
            
            paper_details = [
                {
                    'index': idx, 
                    'title': papers[idx]['title_clean'],
                    'score': float(scores[i])
                }
                for i, idx in enumerate(indices[:5])
            ]
            
            return {
                'statusCode': 200,
                'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
                'body': json.dumps({
                    'answer': answer,
                    'papers': paper_details
                })
            }
        
        return {'statusCode': 400, 'headers': {'Access-Control-Allow-Origin': '*'}, 'body': json.dumps({'error': 'Invalid action'})}
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(traceback.format_exc())
        return {'statusCode': 500, 'headers': {'Access-Control-Allow-Origin': '*'}, 'body': json.dumps({'error': str(e)})}