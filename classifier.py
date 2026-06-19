import json
import os
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TRAIN_FILE, LABELS_FILE

_client = Groq(api_key=GROQ_API_KEY)


def load_labeled_examples() -> list[dict]:
    """
    Load the training episodes and merge them with the student's labels.

    Returns a list of dicts, each with:
      - "id"          : episode ID
      - "title"       : episode title
      - "podcast"     : podcast name
      - "description" : episode description
      - "label"       : the label from my_labels.json (may be None if not yet annotated)

    Only returns episodes where the label is a valid, non-null string.
    Episodes with null labels are silently skipped.
    """
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    labels_path = os.path.join(DATA_PATH, LABELS_FILE)

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            labeled.append({**ep, "label": label})

    return labeled


def build_few_shot_prompt(labeled_examples: list[dict], description: str) -> str:
    """
    Build a few-shot classification prompt using the student's labeled training examples.
    """
    parts = []

    parts.append(
        "You are classifying podcast episodes by their format. "
        "Classify the episode into exactly one of these four labels:\n\n"
        "- interview: a conversation between a host and one or more guests\n"
        "- solo: a single host speaking from memory, experience, or opinion — "
        "no guests, no assembled external sources\n"
        "- panel: multiple guests with roughly equal speaking time, often "
        "debating or discussing a topic together\n"
        "- narrative: a story assembled from external sources — interviews, "
        "archival audio, reporting — with a clear narrative arc\n\n"
        "Return only the label and your reasoning. Do not explain the taxonomy."
    )

    if labeled_examples:
        parts.append("\nHere are labeled examples:\n")
        for ex in labeled_examples:
            parts.append(f"Title: {ex['title']}")
            parts.append(f"Description: {ex['description']}")
            parts.append(f"Label: {ex['label']}")
            parts.append("---")

    parts.append("\nNow classify this episode:\n")
    parts.append(f"Description: {description}")
    parts.append(
        "\nRespond in exactly this format:\n"
        "Label: <one of: interview, solo, panel, narrative>\n"
        "Reasoning: <one sentence explaining why>"
    )

    return "\n".join(parts)


def classify_episode(description: str, labeled_examples: list[dict]) -> dict:
    """
    Classify a single podcast episode description using the few-shot LLM classifier.
    """
    try:
        # Step 1: build prompt
        prompt = build_few_shot_prompt(labeled_examples, description)

        # Step 2: send to LLM
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
        )
        response_text = response.choices[0].message.content

        # Step 3: parse label and reasoning line-by-line
        raw_label = ""
        reasoning = response_text  # fallback: preserve raw output
        for line in response_text.splitlines():
            stripped = line.strip()
            lower = stripped.lower()
            if lower.startswith("label:"):
                raw_label = stripped[len("label:"):].strip().lower()
            elif lower.startswith("reasoning:"):
                reasoning = stripped[len("reasoning:"):].strip()

        # Step 4: validate label
        label = raw_label if raw_label in VALID_LABELS else "unknown"

        return {"label": label, "reasoning": reasoning}

    except Exception as e:
        return {"label": "unknown", "reasoning": f"Error: {e}"}
