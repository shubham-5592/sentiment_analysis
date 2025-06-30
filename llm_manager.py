from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.runnables import Runnable

def get_hf_sentiment_chain(hf_token: str) -> Runnable:
    """
    LangChain Runnable for sentiment analysis
    Args:
        hf_token: Hugging Face API token.
    Returns:
        A LangChain Runnable that can be invoked with text for sentiment analysis.
    """

    sentiment_pipeline = pipeline(
        task="sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        token=hf_token
    )
    return sentiment_pipeline

    # llm = HuggingFacePipeline(pipeline=sentiment_pipeline)
    # prompt = PromptTemplate.from_template("Analyze the sentiment of this sentence: {text}")
    # chain = prompt | llm
    
    # return chain