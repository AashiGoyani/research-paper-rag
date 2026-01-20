import json
import boto3
import os
import urllib.request

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
    api_key = os.environ.get('GOOGLE_API_KEY')
    
    # CORRECT MODEL: gemini-2.0-flash
    url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={api_key}'
    
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    
    req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result['candidates'][0]['content']['parts'][0]['text']
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        print(f"Gemini error {e.code}: {error_body}")
        raise Exception(f"Gemini failed: {error_body}")
    except Exception as e:
        print(f"Error: {e}")
        raise

def lambda_handler(event, context):
    try:
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
            p1, p2 = papers[indices[0]], papers[indices[1]]
            prompt = f"Compare these two research papers:\n\nPaper 1:\nTitle: {p1['title_clean']}\nAbstract: {p1.get('abstract', '')[:400]}\n\nPaper 2:\nTitle: {p2['title_clean']}\nAbstract: {p2.get('abstract', '')[:400]}\n\nProvide:\n1. Main similarities\n2. Key differences\n3. Which is better for beginners vs experts\n4. Your recommendation"
            comparison = call_gemini_api(prompt)
            return {'statusCode': 200, 'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'}, 'body': json.dumps({'comparison': comparison})}
        
        elif action == 'rag_search':
            indices = body.get('relevant_papers', [])[:5]
            context = "\n---\n".join([f"Paper {i+1}:\nTitle: {papers[idx]['title_clean']}\nAbstract: {papers[idx].get('abstract', '')[:300]}" for i, idx in enumerate(indices)])
            prompt = f"Based on these research papers from our database:\n\n{context}\n\nUser question: {body.get('query')}\n\nProvide a comprehensive answer that:\n1. Directly answers the question\n2. Cites specific papers\n3. Explains key concepts\n4. Suggests which papers to read first"
            answer = call_gemini_api(prompt)
            return {'statusCode': 200, 'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'}, 'body': json.dumps({'answer': answer})}
        
        return {'statusCode': 400, 'headers': {'Access-Control-Allow-Origin': '*'}, 'body': json.dumps({'error': 'Invalid action'})}
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        print(traceback.format_exc())
        return {'statusCode': 500, 'headers': {'Access-Control-Allow-Origin': '*'}, 'body': json.dumps({'error': str(e)})}
