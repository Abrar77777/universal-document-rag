from __future__ import annotations

import re
from collections import Counter
from typing import Any


STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "you", "your", "are", "was",
    "were", "have", "has", "had", "but", "not", "from", "they", "their",
    "our", "can", "will", "would", "there", "what", "when", "where", "which",
    "about", "into", "than", "then", "also", "just", "very", "more", "some",
    "been", "being", "too", "all", "any", "out", "get", "got", "use", "using",
}

POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "love", "liked", "easy", "fast",
    "helpful", "happy", "satisfied", "smooth", "best", "awesome", "perfect",
    "clear", "useful", "recommend", "resolved", "improved", "positive",
}

NEGATIVE_WORDS = {
    "bad", "poor", "slow", "difficult", "confusing", "issue", "problem",
    "angry", "frustrated", "hate", "delay", "delayed", "broken", "error",
    "failed", "failure", "worst", "expensive", "refund", "complaint",
    "negative", "unhappy", "disappointed", "bug", "crash", "stuck",
}

EMOTION_LEXICON = {
    "joy": {"happy", "love", "great", "excellent", "amazing", "awesome", "perfect"},
    "trust": {"helpful", "resolved", "reliable", "smooth", "clear", "recommend"},
    "anger": {"angry", "hate", "worst", "complaint", "refund", "broken"},
    "sadness": {"unhappy", "disappointed", "poor", "bad", "failed"},
    "fear": {"worried", "risk", "unsafe", "concern", "confusing", "stuck"},
}

THEME_KEYWORDS = {
    "Pricing": {"price", "pricing", "cost", "expensive", "cheap", "refund", "bill", "billing"},
    "Support": {"support", "agent", "help", "service", "response", "resolved", "ticket"},
    "Product Quality": {"quality", "feature", "bug", "broken", "crash", "error", "working"},
    "Delivery/Speed": {"slow", "fast", "delay", "delayed", "time", "quick", "speed"},
    "Usability": {"easy", "difficult", "confusing", "simple", "interface", "use", "setup"},
}


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip() for part in parts if len(part.strip()) > 5]


def _tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z']+", text)
        if token.lower() not in STOPWORDS and len(token) > 2
    ]


def _score_sentence(sentence: str) -> tuple[str, int]:
    words = set(_tokens(sentence))
    positive = len(words & POSITIVE_WORDS)
    negative = len(words & NEGATIVE_WORDS)
    score = positive - negative
    if score > 0:
        return "positive", score
    if score < 0:
        return "negative", score
    return "neutral", score


def analyze_feedback_text(text: str, source: str = "uploaded text") -> dict[str, Any]:
    sentences = _sentences(text)
    tokens = _tokens(text)
    keyword_counts = Counter(tokens)

    sentiment_counts = Counter()
    scored_sentences = []
    for sentence in sentences:
        label, score = _score_sentence(sentence)
        sentiment_counts[label] += 1
        scored_sentences.append({"text": sentence, "sentiment": label, "score": score})

    theme_counts = Counter()
    for token in tokens:
        for theme, words in THEME_KEYWORDS.items():
            if token in words:
                theme_counts[theme] += 1

    emotion_counts = Counter()
    for token in tokens:
        for emotion, words in EMOTION_LEXICON.items():
            if token in words:
                emotion_counts[emotion] += 1

    pain_points = [
        item for item in scored_sentences
        if item["sentiment"] == "negative"
    ][:10]
    praise_points = [
        item for item in scored_sentences
        if item["sentiment"] == "positive"
    ][:10]

    total = max(sum(sentiment_counts.values()), 1)
    net_sentiment = round(
        (sentiment_counts["positive"] - sentiment_counts["negative"]) / total,
        4,
    )

    return {
        "source": source,
        "responses": len(sentences),
        "words": len(tokens),
        "net_sentiment": net_sentiment,
        "sentiment": dict(sentiment_counts),
        "keywords": keyword_counts.most_common(25),
        "themes": theme_counts.most_common(10),
        "emotions": emotion_counts.most_common(10),
        "pain_points": pain_points,
        "praise_points": praise_points,
    }
