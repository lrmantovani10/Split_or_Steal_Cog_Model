import random
from fastai.vision.all import *

# Image path
filepath = 'Ana.png'

# Download these files from https://huggingface.co/spaces/schibsted/facial_expression_classifier

learn_emotion = load_learner('emotions_vgg19.pkl')
learn_emotion_labels = learn_emotion.dls.vocab

learn_sentiment = load_learner('sentiment_vgg19.pkl')
learn_sentiment_labels = learn_sentiment.dls.vocab

def predict(img_path):
    img = PILImage.create(img_path)

    _, _, probs_emotion = learn_emotion.predict(img)
    
    _, _, probs_sentiment = learn_sentiment.predict(img)
    
    emotions = {learn_emotion_labels[i]: float(probs_emotion[i]) for i in range(len(learn_emotion_labels))}
    sentiments = {learn_sentiment_labels[i]: float(probs_sentiment[i]) for i in range(len(learn_sentiment_labels))}
        
    return [emotions, sentiments]

emotions, sentiment = predict(filepath)

print("Facial emotions and sentiments detected:", emotions, sentiment, "\n")

likelihoods = {
    'Split': {'Fear': 0.15, 'Sadness': 0.3, 'Surprise': 0.05, 'Happiness': 0.9, 'Anger': 0.1, 'Disgust': 0.1},
    'Steal': {'Fear': 0.8, 'Sadness': 0.65, 'Surprise': 0.05, 'Happiness': 0.4, 'Anger': 0.7, 'Disgust': 0.6}
}
priors = {'Split': 0.5, 'Steal': 0.5}


def categorize(decision_emotions, decision_sentiment):
    posteriors = {}
    total_posterior = 0

    for decision in likelihoods:
        posterior = priors[decision]
        for emotion, value in decision_emotions.items():
            posterior_multiplier = likelihoods[decision][emotion] if value else (1 - likelihoods[decision][emotion])
            if emotion == "Happiness":
                posterior_multiplier * (1 + decision_sentiment["Positive"] - decision_sentiment["Negative"])
            elif emotion == "Sadness":
                posterior_multiplier * (1 + decision_sentiment["Negative"] - decision_sentiment["Positive"])
            elif emotion == "Anger":
                posterior_multiplier * (2 ** (1 + decision_sentiment["Negative"] - decision_sentiment["Positive"]))
            elif emotion == "Disgust":
                posterior_multiplier * (1.2 ** (1 + decision_sentiment["Negative"] - decision_sentiment["Positive"]))
            elif emotion == "Fear":
                posterior_multiplier * (3 ** (1 + decision_sentiment["Negative"] - decision_sentiment["Positive"]))
            elif emotion == "Surprise":
                posterior_multiplier * (np.e ** (decision_sentiment["Neutral"]))


            posterior *= posterior_multiplier
        posteriors[decision] = posterior
        total_posterior += posterior

    if total_posterior == 0:
        return "Uncategorized", 0

    for decision in posteriors:
        posteriors[decision] /= total_posterior

    categorized_type = max(posteriors, key=posteriors.get)
    return categorized_type, posteriors[categorized_type]

# Function to generate random examples
def generate_emotional_state(emotions):
    features = {
        'Fear': emotions["fear"] > 0.2,
        'Sadness': emotions["sad"] > 0.4,
        'Surprise': emotions["surprise"] > 0.5,
        'Happiness': emotions["happy"] > 0.35,
        'Anger': emotions["angry"] > 0.15,
        'Disgust': emotions["disgust"] > 0.2,
    }
    return features


emotions = generate_emotional_state(emotions)
sentiment = {"Negative": sentiment["negative"], "Positive": sentiment["positive"], "Neutral": sentiment["neutral"]}

print("Facial emotions and sentiments detected:", emotions, sentiment, "\n")
decision = categorize(emotions, sentiment)

print("\nDecision:", decision)