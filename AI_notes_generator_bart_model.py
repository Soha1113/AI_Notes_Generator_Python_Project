"""
AI Notes Generator using BART
--------------------------------
This project uses the pretrained BART model for
abstractive text summarization to generate structured notes.
"""

# Import required libraries
from transformers import BartForConditionalGeneration, BartTokenizer
import torch


# -----------------------------
# Load Pretrained Model
# -----------------------------
def load_model():
    """
    Loads the pretrained BART model and tokenizer.
    """
    try:
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        return tokenizer, model
    except Exception as e:
        print("Error loading model:", e)
        exit()


# -----------------------------
# Generate Notes Function
# -----------------------------
def generate_notes(text, tokenizer, model):
    """
    Generates structured notes from input text using BART.
    """

    if not text.strip():
        return {"Error": "Input text cannot be empty."}

    # Step 1: Tokenize input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=1024,
        truncation=True
    )

    # Step 2: Generate summary using beam search
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Step 3: Structure notes safely
    sentences = [s.strip() for s in summary.split('.') if s.strip()]

    definition = sentences[0] + '.' if len(sentences) > 0 else summary
    key_points = sentences[1:3] if len(sentences) > 1 else []

    notes = {
        "Definition": definition,
        "Key Points": key_points,
        "Advantages": [
            "Time-saving",
            "Improves learning efficiency",
            "Generates human-like summaries"
        ],
        "Applications": [
            "Student revision",
            "Lecture preparation",
            "Meeting summaries",
            "Quick study material"
        ]
    }

    return notes


# -----------------------------
# Main Program
# -----------------------------
def main():
    print("===== AI NOTES GENERATOR USING BART =====\n")
    print("Enter/Paste your text below.")
    print("Type 'END' on a new line to finish input.\n")

    # Multi-line input
    user_input = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        user_input.append(line)

    text = " ".join(user_input)

    # Load model
    tokenizer, model = load_model()

    # Generate notes
    notes = generate_notes(text, tokenizer, model)

    # Display results
    print("\n----- AI GENERATED NOTES -----")
    for section, content in notes.items():
        print(f"\n{section}:")
        if isinstance(content, list):
            for item in content:
                print(f"- {item}")
        else:
            print(content)


# Run program
if __name__ == "__main__":
    main()