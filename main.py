import pandas as pd
from LLM_detection import load_or_create_few_shot_examples
from dotenv import load_dotenv
import os
import openai
import google.generativeai as genai
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import numpy as np

# Load API keys from environment file
load_dotenv('keys.env')

# Set API keys
openai.api_key = os.getenv('OPENAI_API_KEY')
gemini_api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=gemini_api_key)

# Initialize Sentence-BERT for semantic similarity
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

# Explanation of the task
few_shot_explanation = (
    "In this task, you will read an entire GitHub issue conversation. If any part of the conversation "
    "contains rude, offensive, or disrespectful language, respond with 'Yes'. Otherwise, respond with 'No'. "
    "Here are some examples:\n"
)

zero_shot_prompt = (
    "In this task, you will read an entire GitHub issue conversation. If any part of the conversation "
    "contains rude, offensive, or disrespectful language, respond with 'Yes'. Otherwise, respond with 'No'."
)


random_examples_csv = "train_issues.csv"          
examples_few_shot_csv = "examples_few_shot.csv"  
examples = load_or_create_few_shot_examples(
    original_csv_file=random_examples_csv,
    few_shot_csv_file=examples_few_shot_csv,
)

few_shot_prompt = few_shot_explanation + "\n".join(
    [f"Conversation: {ex['comment']}\nToxic: {ex['label']}\n" for ex in examples]
)

def detect_toxicity_with_gpt4o(input_text, few_shot=True):
    try:
        if few_shot:
            prompt = f"{few_shot_prompt}\nComment: {input_text}"
        else:
            prompt = f"{zero_shot_prompt}\nComment: {input_text}"

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a toxicity detection assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        prediction = response.choices[0].message.content.strip()
        return prediction
    except Exception as e:
        print(f"Error detecting toxicity with GPT-4o: {e}")
        return "Error"

# Few-shot examples for mitigation
mitigation_examples = [
    {
        "original": "You are the worst developer ever. This code is garbage!",
        "rewritten": "I'm concerned about the current approach. Could we find a clearer way to write this code?"
    },
    {
        "original": "This fix is so stupid, why are you even trying?",
        "rewritten": "I see potential issues with this fix. Let's discuss possible improvements."
    }
]

# Explanation of the task for rewriting
mitigation_few_shot_explanation = (
    "Your task is to rewrite a given comment to remove rude or offensive language, while preserving its core meaning. "
    "The rewritten comment should be polite, respectful, and constructive. Below are some examples:\n\n"
)

# Build the prompt string for few-shot mitigation
mitigation_few_shot_prompt = mitigation_few_shot_explanation + "\n".join([
    f"Original: {ex['original']}\nRewritten: {ex['rewritten']}\n"
    for ex in mitigation_examples
])

def mitigate_toxicity_with_gemini(input_text):
    """
    Use Gemini in few-shot mode to rewrite the input text, removing toxic or offensive language.
    """
    try:
        prompt = (
            f"{mitigation_few_shot_prompt}"
            f"Original: {input_text}\n"
            "Rewritten:"
        )
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        mitigated_text = response.text.strip()
        return mitigated_text
    except Exception as e:
        print(f"Error mitigating toxicity with Gemini: {e}")
        return "Error"

def determine_similarity_with_bert(original_text, mitigated_text):
    """
    Use Sentence-BERT (LLM C) to determine similarity between two sentences.
    """
    try:
        embeddings1 = similarity_model.encode(original_text, convert_to_tensor=True)
        embeddings2 = similarity_model.encode(mitigated_text, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(embeddings1, embeddings2).item()
        return similarity_score
    except Exception as e:
        print(f"Error determining similarity: {e}")
        return None

def process_dataset(input_file, output_file, max_rows=-1):
    """
    Process the dataset.
    If max_rows > 0, only process that many rows for a proof of concept.
    """
    # Load the dataset
    df = pd.read_csv(input_file)

    # Add new columns for the results if they do not exist
    if 'Mitigated Text' not in df.columns:
        df['Mitigated Text'] = None
    if 'Toxicity Prediction' not in df.columns:
        df['Toxicity Prediction'] = None
    if 'Toxicity After Mitigation' not in df.columns:
        df['Toxicity After Mitigation'] = None
    if 'Similarity Score' not in df.columns:
        df['Similarity Score'] = None
    if 'Mitigation Attempts' not in df.columns:
        df['Mitigation Attempts'] = 0

    row_count = 0
    for index, row in df.iterrows():
        if max_rows > 0 and row_count >= max_rows:
            break

        # Check if row is already mitigated (i.e., "Mitigated Text" is not NaN/None)
        if pd.notna(row['Mitigated Text']) and row['Mitigated Text'] != "":
            print(f"Skipping row {index + 1}/{len(df)}: already mitigated.")
            continue

                # Check if row is already mitigated (i.e., "Mitigated Text" is not NaN/None)
        if pd.notna(row['Toxicity Prediction']) and row['Toxicity Prediction'] != "":
            print(f"Skipping row {index + 1}/{len(df)}: already mitigated.")
            continue

        input_text = row['text']
        print(f"Processing row {index + 1}/{len(df)}: {str(input_text)[:50]}...")

        # Step 1: Predict toxicity for the original text
        toxicity_prediction = detect_toxicity_with_gpt4o(input_text)
        df.at[index, 'Toxicity Prediction'] = toxicity_prediction

        # Step 2: Mitigate toxicity if toxic
        if toxicity_prediction == "Yes":
            mitigated_text = mitigate_toxicity_with_gemini(input_text)
            count = 0
            new_toxicity_prediction = toxicity_prediction

            while count < 3:
                new_toxicity_prediction = detect_toxicity_with_gpt4o(mitigated_text)
                if new_toxicity_prediction == "No":
                    break
                mitigated_text = mitigate_toxicity_with_gemini(mitigated_text)
                count += 1

            # Save mitigation info
            df.at[index, 'Mitigated Text'] = mitigated_text
            df.at[index, 'Toxicity After Mitigation'] = new_toxicity_prediction
            df.at[index, 'Mitigation Attempts'] = count + 1

            # Determine semantic similarity
            similarity_score = determine_similarity_with_bert(input_text, mitigated_text)
            df.at[index, 'Similarity Score'] = similarity_score

        row_count += 1
        if row_count%10 == 0:
            df.to_csv(output_file, index=False)
            print("save until "+ str(row_count))

    # Save the processed dataset
    
    print(f"Processed dataset saved to {output_file}")

def analyze_mitigation_results(csv_file):
    """
    Calculate and print:
    1) Mitigation Success Rate (%)
    2) Average Mitigation Attempts
    3) Average Similarity Score
    4) Failure Cases (%)
    Then plot a histogram of the Similarity Score distribution.
    """
    import matplotlib.pyplot as plt
    
    df = pd.read_csv(csv_file)

    # Filter rows that were predicted toxic initially
    originally_toxic = df[df['Toxicity Prediction'] == "Yes"].copy()
    if originally_toxic.empty:
        print("No originally toxic comments found in the dataset.")
        return

    # 1) Mitigation Success Rate
    mitigated_success = originally_toxic[originally_toxic['Toxicity After Mitigation'] == "No"]
    success_rate = len(mitigated_success) / len(originally_toxic) * 100

    # 2) Average Mitigation Attempts
    avg_attempts = originally_toxic['Mitigation Attempts'].mean()

    # 3) Average Similarity Score
    mitigated_nonempty = originally_toxic.dropna(subset=['Mitigated Text'])
    if not mitigated_nonempty.empty:
        avg_similarity = mitigated_nonempty['Similarity Score'].mean()
    else:
        avg_similarity = 0.0

    # 4) Failure Cases
    failed_after_3 = originally_toxic[
        (originally_toxic['Mitigation Attempts'] == 3) &
        (originally_toxic['Toxicity After Mitigation'] == "Yes")
    ]
    failure_rate = len(failed_after_3) / len(originally_toxic) * 100

    # Print results
    print("\n=== Mitigation Analysis ===")
    print(f"Mitigation Success Rate: {success_rate:.2f}%")
    print(f"Average Mitigation Attempts: {avg_attempts:.2f}")
    print(f"Average Similarity Score: {avg_similarity:.4f}")
    print(f"Failure Cases after 3 attempts: {failure_rate:.2f}%")

    similarity_scores = mitigated_nonempty['Similarity Score'].dropna()
    if not similarity_scores.empty:
        plt.figure(figsize=(8, 5))
        plt.hist(similarity_scores, bins=20, edgecolor='black', alpha=0.7)
        plt.title("Distribution of Similarity Scores")
        plt.xlabel("Similarity Score")
        plt.ylabel("Frequency")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig("similarity_distribution.png", dpi=300)

    else:
        print("No valid similarity scores to plot.")

    # For cumulative histogram
    similarity_scores = mitigated_nonempty['Similarity Score'].dropna()
    if not similarity_scores.empty:
        # Calculate cumulative distribution
        weights = np.ones_like(similarity_scores) * 100.0 / len(similarity_scores)
        
        plt.figure(figsize=(8, 5))
        plt.hist(similarity_scores, bins=20, edgecolor='black', alpha=0.7, weights=weights, cumulative=True)
        plt.title("Cumulative Distribution of Similarity Scores")
        plt.xlabel("Similarity Score")
        plt.ylabel("Cumulative Percentage (%)")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig("cumulative_similarity_distribution.png", dpi=300)
    else:
        print("No valid similarity scores to plot.")



# Main execution
if __name__ == "__main__":
    input_file = "train_comments.csv"           
    output_file = "mitigate_train_comments.csv" 
    process_dataset(input_file, output_file)
    analyze_mitigation_results(output_file)
