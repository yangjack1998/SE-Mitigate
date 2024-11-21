import pandas as pd
from dotenv import load_dotenv
import os
import openai
import google.generativeai as genai
from sklearn.metrics import confusion_matrix, classification_report
import time
import random

# Load API keys
load_dotenv('keys.env')
openai.api_key = os.getenv('OPENAI_API_KEY')
gemini_api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=gemini_api_key)

def load_or_create_few_shot_examples(
    original_csv_file,
    few_shot_csv_file="examples_few_shot.csv"
):
    
    if os.path.exists(few_shot_csv_file):
        print(f"[INFO] Found existing few-shot examples file: {few_shot_csv_file}. Loading...")
        df_saved = pd.read_csv(few_shot_csv_file)
        examples = df_saved.to_dict(orient="records")
    else:
        print(f"[INFO] {few_shot_csv_file} not found. Creating it from: {original_csv_file}")
        df_original = pd.read_csv(original_csv_file).dropna(subset=['total_text', 'toxicity'])
        
        df_toxic = df_original[df_original['toxicity'] == 'y']
        df_non_toxic = df_original[df_original['toxicity'] == 'n']

        sampled_toxic = df_toxic.sample(3, random_state=42)
        sampled_non_toxic = df_non_toxic.sample(3, random_state=42)

        combined = pd.concat([sampled_toxic, sampled_non_toxic]).sample(frac=1, random_state=42)

        data_list = []
        for idx, row in combined.iterrows():
            label = "Yes" if row['toxicity'] == 'y' else "No"
            data_list.append({
                "comment": row['total_text'],
                "label": label
            })

        df_few_shot = pd.DataFrame(data_list)
        df_few_shot.to_csv(few_shot_csv_file, index=False)
        examples = data_list

    return examples

# Few-shot examples
random_examples_csv = "train_issues.csv"          
examples_few_shot_csv = "examples_few_shot.csv"   
examples = load_or_create_few_shot_examples(
    original_csv_file=random_examples_csv,
    few_shot_csv_file=examples_few_shot_csv,
)

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

few_shot_prompt = few_shot_explanation + "\n".join(
    [f"Conversation: {ex['comment']}\nToxic: {ex['label']}\n" for ex in examples]
)

def detect_toxicity_with_gpt4o(input_text, few_shot=False):
    """
    Use GPT-4o to determine if the input text is toxic, with optional few-shot learning.
    """
    try:
        if few_shot:
            # Include few-shot examples in the prompt
            prompt = f"{few_shot_prompt} Here is the conversation you need to determine: {input_text}"
            # print("prompt:"+prompt)
        else:
            # Zero-shot prompt
            prompt = f"{zero_shot_prompt} Here is the onversation you need to determine: {input_text}"

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

# Function to determine toxicity using Gemini (Few-Shot and Zero-Shot)
def detect_toxicity_with_gemini(input_text, few_shot=False):
    """
    Use Gemini to determine if the input text is toxic, with optional few-shot learning.
    """
    try:
        if few_shot:
            # Include few-shot examples in the prompt
            prompt = f"{few_shot_prompt} Here is the onversation you need to determine: {input_text}"
        else:
            # Zero-shot prompt
            prompt = f"{zero_shot_prompt} Here is the onversation you need to determine: {input_text}"

        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            prompt
        )
        prediction = response.text.strip()
        return prediction
    except Exception as e:
        print(f"Error detecting toxicity with Gemini: {e}")
        return "Error"

# Calculate accuracy
def calculate_accuracy(df, ground_truth_column, prediction_column):
    correct_predictions = 0
    total_predictions = 0

    for index, row in df.iterrows():
        ground_truth = "Yes" if row[ground_truth_column] == "y" else "No"
        prediction = row[prediction_column]

        if prediction != "Error":
            total_predictions += 1
            if prediction == ground_truth:
                correct_predictions += 1

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy

# Process the dataset
def process_dataset(input_file, output_file, save_interval=5, start_row=0):
    # Load the dataset
    df = pd.read_csv(input_file)

    for col in ['Toxicity GPT-4o Few-Shot', 'Toxicity GPT-4o Zero-Shot', 'Toxicity Gemini Few-Shot', 'Toxicity Gemini Zero-Shot']:
        if col not in df.columns:
            df[col] = ""  

    # Process each row
    for index, row in df.iterrows():
        if index < start_row:
            continue  
        input_text = row['total_text']
        if pd.isna(input_text) or not isinstance(input_text, str):
            input_text = ""  

        print(f"Processing row {index + 1}/{len(df)}: {input_text[:50]}...")

        # Detect toxicity with GPT-4o (Few-Shot and Zero-Shot)
        gpt4_few_shot_result = detect_toxicity_with_gpt4o(input_text, few_shot=True)
        df.at[index, 'Toxicity GPT-4o Few-Shot'] = gpt4_few_shot_result

        gpt4_zero_shot_result = detect_toxicity_with_gpt4o(input_text, few_shot=False)
        df.at[index, 'Toxicity GPT-4o Zero-Shot'] = gpt4_zero_shot_result

        # Detect toxicity with Gemini (Few-Shot and Zero-Shot)
        gemini_few_shot_result = detect_toxicity_with_gemini(input_text, few_shot=True)
        df.at[index, 'Toxicity Gemini Few-Shot'] = gemini_few_shot_result

        gemini_zero_shot_result = detect_toxicity_with_gemini(input_text, few_shot=False)
        df.at[index, 'Toxicity Gemini Zero-Shot'] = gemini_zero_shot_result
        
        if (index + 1) % save_interval == 0:
            df.to_csv(output_file, index=False)
            time.sleep(10)


    # Save the updated dataset
    df.to_csv(output_file, index=False)
    print(f"Processed dataset saved to {output_file}")

def calculate_metrics(processed_file, ground_truth_column):
    """
    Calculate accuracy, precision, recall, F1-score, and confusion matrix for each LLM.
    """
    df = pd.read_csv(processed_file)

    # Models to evaluate
    models = [
        "Toxicity GPT-4o Few-Shot",
        "Toxicity GPT-4o Zero-Shot",
        "Toxicity Gemini Few-Shot",
        "Toxicity Gemini Zero-Shot"
    ]

    # Iterate through each model's predictions
    for model in models:
        print(f"\nEvaluating {model}...")
        
        # Convert ground truth and predictions to binary (Yes = 1, No = 0)
        y_true = df[ground_truth_column].map(lambda x: 1 if x.lower() == "y" else 0)
        y_pred = df[model].map(lambda x: 1 if (str(x).lower() == "yes" or str(x).lower() == "yes.") else 0)

        # Calculate metrics
        print("Classification Report:")
        print(classification_report(y_true, y_pred, target_names=["Non-Toxic", "Toxic"]))

        # Confusion Matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        print("Confusion Matrix:")
        print(f"True Negatives (TN): {tn}")
        print(f"False Positives (FP): {fp}")
        print(f"False Negatives (FN): {fn}")
        print(f"True Positives (TP): {tp}")


if __name__ == "__main__":
    input_file = "labeled_test_issues.csv"  
    processed_file = "processed_labeled_test_issues.csv"  

    process_dataset(input_file, processed_file)
    calculate_metrics(processed_file, "toxicity")