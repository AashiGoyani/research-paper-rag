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
            
            # Build comparison for all selected papers
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
            
            # If user provided specific paper indices, use those
            if 'relevant_papers' in body and body['relevant_papers']:
                indices = body.get('relevant_papers', [])[:5]
            else:
                # Search all papers for relevant ones
                query_lower = query.lower()
                query_words = set(query_lower.split())
                
                relevant = []
                for idx, paper in enumerate(papers):
                    title = paper['title_clean'].lower()
                    abstract = paper.get('abstract', '').lower()
                    matches = sum(1 for word in query_words if word in title or word in abstract)
                    if matches > 0:
                        relevant.append((idx, matches))
                
                relevant.sort(key=lambda x: x[1], reverse=True)
                indices = [idx for idx, _ in relevant[:5]]
            
            if not indices:
                return {
                    'statusCode': 200,
                    'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
                    'body': json.dumps({'answer': 'I couldn\'t find papers relevant to your question. Try different keywords!'})
                }
            
            # Build context with paper details
            context = "\n---\n".join([
                f"Paper {i+1} (Index: {idx}):\nTitle: {papers[idx]['title_clean']}\nAbstract: {papers[idx].get('abstract', '')[:300]}"
                for i, idx in enumerate(indices)
            ])
            
            prompt = f"""Based on these research papers:

        {context}

        User question: {query}

        Provide a comprehensive answer that:
        1. Directly answers the question
        2. Lists the relevant papers with their titles
        3. For each paper mentioned, format it as: **[Paper Title]** (Index: XX) where XX is the paper index number
        4. Explains key concepts clearly
        5. Recommends which papers to read

        Example format:
        - **Convolutional Neural Networks for Classification** (Index: 114)
        - **Deep Learning Approaches** (Index: 68)

        Be helpful and include paper indices!"""
            
            answer = call_gemini_api(prompt)
            
            # Also return paper details so frontend can make them clickable
            paper_details = [
                {'index': idx, 'title': papers[idx]['title_clean']}
                for idx in indices
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
# Force update
