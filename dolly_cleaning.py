import csv
import re

INPUT_FILE = "data/Benign_data/dolly_15k.csv"
OUTPUT_FILE = "data/Benign_data/dolly_cleaned.csv"


def is_english(text):
    # keep only mostly alphabetic text
    letters = sum(c.isalpha() for c in text)
    return letters / max(len(text), 1) > 0.7


def has_repetition(text):
    words = text.lower().split()
    if len(words) < 5:
        return False
    return len(set(words)) / len(words) < 0.6


def has_weird_tokens(text):
    # remove garbage tokens
    blacklist = [
        "instrumentedtest",
        "viaf",
        "jeografia",
        "lorem ipsum"
    ]

    t = text.lower()

    for word in blacklist:
        if word in t:
            return True

    # long weird tokens
    for word in text.split():
        if len(word) > 25:
            return True

    return False


def is_valid_response(text):
    if not text:
        return False

    text = text.strip()

    # too short
    if len(text.split()) < 15:
        return False

    # placeholder patterns
    if text.lower().startswith("sure, here is"):
        return False

    # repetition
    if has_repetition(text):
        return False

    # weird tokens
    if has_weird_tokens(text):
        return False

    # non-english
    if not is_english(text):
        return False

    return True


def clean_text(text):
    text = text.strip()

    # normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # remove excessive punctuation
    text = re.sub(r"[^\w\s.,!?'-]", "", text)

    return text


cleaned_rows = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        instruction = (row.get("instruction") or "").strip()
        context = (row.get("context") or "").strip()
        response = (row.get("response") or "").strip()

        if not instruction or not response:
            continue

        if not is_valid_response(response):
            continue

        # combine instruction + context
        if context:
            input_text = f"{instruction}\n{context}"
        else:
            input_text = instruction

        input_text = clean_text(input_text)
        response = clean_text(response)

        cleaned_rows.append({
            "input": input_text,
            "output": response
        })

with open(INPUT_FILE, encoding="utf-8") as f:
    original_size = sum(1 for _ in f)

print(f"Original size: {original_size}")
print(f"Cleaned size: {len(cleaned_rows)}")


# save cleaned dataset
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["input", "output"])
    writer.writeheader()
    writer.writerows(cleaned_rows)

print("✅ Cleaned dataset saved to:", OUTPUT_FILE)