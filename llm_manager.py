from transformers import pipeline

def get_hf_binary_sentiment(hf_token: str) -> pipeline:
    """
    Gets Pipeline sentiment analysis
    Args:
        hf_token: Hugging Face API token.
    Returns:
        A Pipeline that can be invoked with text for sentiment analysis.
    """

    sentiment_pipeline = pipeline(
        task="sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        token=hf_token
    )
    return sentiment_pipeline


def get_hf_emotion_sentiment(hf_token: str) -> pipeline:
    """
    Gets Pipeline emotion analysis
    Args:
        hf_token: Hugging Face API token.
    Returns:
        A Pipeline that can be invoked with text for emotion analysis.
    """

    sentiment_pipeline = pipeline(
        task="text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        token=hf_token,
        return_all_scores=True
    )
    return sentiment_pipeline

