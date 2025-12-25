import pandas as pd
import asyncio
import tqdm.notebook as tqdm
from evaluation_service import EvaluationAgent

# 1. Load the existing results
file_path = "results_with_nlp_metrics.xlsx"
df = pd.read_excel(file_path)

# 2. Initialize the Evaluator
# Use gpt-5-mini as the judge for consistency with previous runs
evaluator = EvaluationAgent(
    task_type="recovery_summarization", 
    threshold=0.8, 
    model_name="gpt-5-mini"
)

async def recover_reasons(df):
    """
    Identifies missing reasons and re-evaluates specific rows.
    """
    updated_rows = []
    
    # Filter for rows needing recovery (where reasons are missing)
    recovery_queue = df[df['logical_flow_reason'].isna() | df['exec_quality_reason'].isna()]
    
    print(f"🚀 Found {len(recovery_queue)} records requiring reason recovery.")

    for index, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Processing Results"):
        # Check if recovery is needed for this specific row
        if pd.isna(row['logical_flow_reason']) or pd.isna(row['exec_quality_reason']):
            try:
                # Re-run evaluation on the existing output_result
                eval_res = await evaluator.a_evaluate(
                    generated_text=row['output_result'],
                    source_context=row['input_text'],
                    section_topic=f"Reason Recovery: {row['title']}"
                )
                # Extract the new results and update both the score and the reason
                new_flow_score = eval_res['metrics'].get('Summary Coherence & Flow', {}).get('score')
                new_flow_reason = eval_res['metrics'].get('Summary Coherence & Flow', {}).get('feedback')

                new_exec_score = eval_res['metrics'].get('Executive Writing Quality', {}).get('score')
                new_exec_reason = eval_res['metrics'].get('Executive Writing Quality', {}).get('feedback')

                # Mandatory update for consistency
                row['logical_flow'] = new_flow_score
                row['logical_flow_reason'] = new_flow_reason
                row['exec_quality'] = new_exec_score
                row['exec_quality_reason'] = new_exec_reason
                
                # Extract reasons from the new evaluation response
                # Note: 'feedback' contains the qualitative reasoning in your metrics setup
                row['logical_flow_reason'] = eval_res['metrics'].get('Summary Coherence & Flow', {}).get('feedback', "")
                row['exec_quality_reason'] = eval_res['metrics'].get('Executive Writing Quality', {}).get('feedback', "")
                
                # Optional: Update scores if they were also missing
                row['logical_flow'] = eval_res['metrics'].get('Summary Coherence & Flow', {}).get('score', row['logical_flow'])
                row['exec_quality'] = eval_res['metrics'].get('Executive Writing Quality', {}).get('score', row['exec_quality'])
                
            except Exception as e:
                print(f"⚠️ Failed to recover reasons for Article ID {row['article_id']}: {e}")
        
        updated_rows.append(row)
        
    return pd.DataFrame(updated_rows)

# 3. Execute and Save
if __name__ == "__main__":
    updated_df = await recover_reasons(df)
    updated_df.to_excel("evaluation_results_recovered.xlsx", index=False)
    print("✅ Recovery complete. Results saved to 'evaluation_results_recovered.xlsx'.")