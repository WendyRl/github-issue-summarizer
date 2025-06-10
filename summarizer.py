from transformers import pipeline

# Load the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_issue(title, body):
    text = f"{title}. {body}"
    
    # Hugging Face models have a max token limit; truncate if needed
    if len(text) > 1024:
        text = text[:1024]
    # Can edit this line for Clean Up Summary Formatting (shorter, neater summaries), e.g.
    # result = summarizer(text, max_length=40, min_length=10, do_sample=False)
    result = summarizer(text, max_length=60, min_length=20, do_sample=False) 
    return result[0]['summary_text']
