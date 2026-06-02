def emotion_to_sentiment(emotion):
    if emotion in ['happy', 'surprise']:
        return "POSITIVE"
    elif emotion == 'neutral':
        return "NEUTRAL"
    else:
        return "NEGATIVE"