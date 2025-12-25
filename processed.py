import pandas as pd
import json
import re

def process_langsmith_json(input_file, output_file):
    # Load the JSON dataset
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    results = []
    
    for entry in data:
        # 1. Basic Metadata Extraction
        run_id = entry.get('run_id')
        name = entry.get('name')
        start_time = entry.get('start_time')
        
        # 2. Input Content Extraction
        # Path: inputs -> messages -> [list] -> [list] -> kwargs -> content
        input_content = None
        try:
            inputs = entry.get('inputs', {})
            messages = inputs.get('messages', [])
            if messages and isinstance(messages[0], list):
                input_content = messages[0][0].get('kwargs', {}).get('content')
        except (IndexError, AttributeError, TypeError):
            input_content = None
            
        # 3. Output Content Extraction
        # Path: outputs -> generations -> [list] -> [list] -> text
        output_content = None
        score = None
        reason = None
        
        try:
            outputs = entry.get('outputs')
            if outputs and 'generations' in outputs:
                generations = outputs.get('generations', [])
                if generations and isinstance(generations[0], list):
                    output_content = generations[0][0].get('text')
        except (IndexError, AttributeError, TypeError):
            output_content = None
            
        # 4. JSON Parsing from Output String
        if output_content:
            try:
                # Locate JSON block within the text (handles preambles)
                json_match = re.search(r'\{.*\}', output_content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    json_data = json.loads(json_str)
                    
                    if isinstance(json_data, dict):
                        reason = json_data.get('reason')
                        score = json_data.get('score')
                        
                        # Fallback: Extract score from text if not explicitly keyed
                        if score is None and reason:
                            match = re.search(r"(?:score is|score of)\s*([\d\.]+)", str(reason), re.IGNORECASE)
                            if match:
                                score = match.group(1)
            except Exception:
                pass
        
        results.append({
            'run_id': run_id,
            'name': name,
            'start_time': start_time,
            'Input Content': input_content,
            'Output Content': output_content,
            'Score': score,
            'Reason': reason
        })
    
    # Create the final DataFrame and export to Excel
    final_df = pd.DataFrame(results)
    final_df.to_excel(output_file, index=False)
    print(f"Processing complete. {len(results)} rows saved to: {output_file}")

if __name__ == "__main__":
    # Ensure these file names match your local environment
    input_json = 'langsmith_traces_v2.json'
    output_xlsx = 'processed_langsmith_traces_v2.xlsx'
    process_langsmith_json(input_json, output_xlsx)