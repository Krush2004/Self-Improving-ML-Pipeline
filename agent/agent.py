import os
import uuid
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    """Initialize the OpenRouter LLM using Langchain."""
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key or api_key == "your_openrouter_api_key_here":
        raise ValueError("Missing OPENROUTER_API_KEY in .env file.")
        
    llm = ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        model_name="openai/gpt-3.5-turbo", # Universal fallback, can be adjusted 
        temperature=0.3
    )
    return llm

def get_pinecone_client_and_index():
    """Returns pc client and index, checking for existance."""
    api_key = os.getenv("PINECONE_API_KEY", "")
    if not api_key or api_key == "your_pinecone_api_key_here":
        return None, None
        
    pc = Pinecone(api_key=api_key)
    index_name = os.getenv("PINECONE_INDEX_NAME", "self-improving-agent-memory")
    
    # Check if index exists, map to serverless if needed
    if index_name not in pc.list_indexes().names():
        try:
            pc.create_index(
                name=index_name,
                dimension=1024, # llama-text-embed-v2 dimension
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        except Exception as e:
            print(f"Pinecone create index error: {e}")
            return None, None
            
    return pc, pc.Index(index_name)

def retrieve_past_learnings(query_text):
    """Retrieves similar past critiques from Pinecone VectorDB using Native Inference."""
    pc, idx = get_pinecone_client_and_index()
    if not pc or not idx:
        return ""
    
    try:
        # Embed the query
        embed_resp = pc.inference.embed(
            model='llama-text-embed-v2',
            inputs=[query_text],
            parameters={'input_type': 'query'}
        )
        
        # Search index
        query_result = idx.query(
            vector=embed_resp[0]['values'],
            top_k=3,
            include_metadata=True
        )
        
        memories = []
        for match in query_result['matches']:
            if match['score'] > 0.6: # threshold to only get relevant memory
                memories.append(match['metadata'].get('text', ''))
                
        if memories:
            return "--- Past Memory/Critique retrieved from Vector DB ---\n" + "\n".join(memories) + "\n---------------------------------------------------"
        return ""
    except Exception as e:
        print("Error retrieving past learnings:", e)
        return ""

def save_to_memory(text_to_save):
    """Embeds and saves the text pipeline result into Pinecone."""
    pc, idx = get_pinecone_client_and_index()
    if not pc or not idx:
        return
        
    try:
        embed_resp = pc.inference.embed(
            model='llama-text-embed-v2',
            inputs=[text_to_save],
            parameters={'input_type': 'passage'}
        )
        
        uid = str(uuid.uuid4())
        idx.upsert(vectors=[{
            'id': uid,
            'values': embed_resp[0]['values'],
            'metadata': {'text': text_to_save}
        }])
        print(f"Successfully saved to pinecone memory: {uid}")
    except Exception as e:
        print("Error saving to memory:", e)

def analyze_dataset_initial(df_head, df_info):
    """Agent analyzes initial dataset before ml_pipeline."""
    try:
        llm = get_llm()
        system_msg = SystemMessage(content="You are an expert Data Scientist AI. Analyze the dataset snippet and provide brief preprocessing suggestions.")
        human_msg = HumanMessage(content=f"Dataset Head:\n{df_head}\n\nDataset Info (Summary):\n{df_info}")
        
        response = llm.invoke([system_msg, human_msg])
        return response.content
    except Exception as e:
        return f"Could not analyze dataset with LLM. Error: {e}"

def self_critique_models(results_df_str, best_model_name):
    """Agent looks at the results and suggests improvements dynamically leveraging long-term memory."""
    try:
        llm = get_llm()
        
        # Pull past similar critiques based on current models list
        past_memories = retrieve_past_learnings(f"Pipeline results memory for models. Best was {best_model_name}")
        
        sys_instructions = (
            "You are a Self-Improving AI Data Scientist. You MUST analyze the EXACT numbers in the model results table below.\n"
            "CRITICAL RULES:\n"
            "1. The 'Best Model' has ALREADY been determined by the pipeline code based on the highest Accuracy. Do NOT contradict this selection.\n"
            "2. Reference the ACTUAL metric values from the table (Accuracy, Precision, Recall, F1 Score) in your analysis.\n"
            "3. Explain WHY the best model performed well for THIS specific dataset.\n"
            "4. Point out models that performed poorly (e.g., 0.0 Recall means the model failed to predict the minority class).\n"
            "5. Provide SPECIFIC, ACTIONABLE suggestions: feature engineering ideas, hyperparameter ranges to try, data balancing techniques.\n"
            "6. Each time you run, give DIFFERENT suggestions. Do not repeat generic advice.\n"
        )
        if past_memories:
            sys_instructions += f"\nThese are memories from past runs for context only. Do NOT copy old suggestions — generate fresh analysis based on the CURRENT results:\n{past_memories}\n"

        system_msg = SystemMessage(content=sys_instructions)
        human_msg = HumanMessage(content=f"Model Results Table:\n{results_df_str}\n\nThe pipeline determined Best Model: {best_model_name}\n\nPlease analyze these SPECIFIC results and provide improvement suggestions.")
        response = llm.invoke([system_msg, human_msg])
        
        final_critique = response.content
        
        # Save this result + critique combo to Long Term Memory
        memory_payload = f"Model Results:\n{results_df_str}\nBest: {best_model_name}\nCopilot Critique:\n{final_critique}"
        save_to_memory(memory_payload)
        
        return final_critique
    except Exception as e:
        return f"Could not generate self-critique. Error: {e}"

def chat_with_copilot(messages_list):
    """
    Answers user queries based on full conversation history.
    And injects pinecone knowledge if relevant to the latest query.
    """
    try:
        llm = get_llm()
        
        # Extract the latest query string to do a vector search
        latest_query = messages_list[-1].content
        memories = retrieve_past_learnings(latest_query)
        
        if memories:
            # Inject memory context behind the scenes
            messages_list.insert(0, SystemMessage(content=f"You fetched these relevant past project logs from your Pinecone Long-Term Memory database. Use them if applicable to answering the user:\n{memories}"))
            
        response = llm.invoke(messages_list)
        return response.content
    except Exception as e:
        return f"Could not generate response. Error: {e}"
