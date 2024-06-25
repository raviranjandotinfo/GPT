

# make sure open AI model is installed for python 
# !pip install -U openai

#Importing important liabararies 
import pandas as pd
from sklearn.model_selection import train_test_split
from openai import OpenAI
client = OpenAI(api_key="<unique_openAI_key>")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Importing new prompt about crypto currency news
df = pd.read_csv("cryptNews.csv")
df.head()

# Function to change data format to top categories
def convert_to_gpt35_format(dataset):
    fine_tuning_data = []
    for _, row in dataset.iterrows():
        json_response = '{"Top Category": "' + row['Top Category'] + '", "Sub Category": "' + row['Sub Category'] + '"}'
        fine_tuning_data.append({
            "messages": [
                {"role": "user", "content": row['Support Query']},
                {"role": "system", "content": json_response}
            ]
        })
    return fine_tuning_data

# Stratified splitting. Assuming 'Top Category' can be used for stratification
train_data, val_data = train_test_split(
    converted_data,
    test_size=0.3,
    stratify=dataset['Top Category'],
    random_state=42  # for reproducibility
)

# Creating training and test/validation datasets in to json format 
def write_to_jsonl(data, file_path):
    with open(file_path, 'w') as file:
        for entry in data:
            json.dump(entry, file)
            file.write('\n')

training_file_name = "train.jsonl"
validation_file_name = "val.jsonl"

write_to_jsonl(train_data, training_file_name)
write_to_jsonl(val_data, validation_file_name)

# Upload Training and Validation Files
training_file = client.files.create(
    file=open(training_file_name, "rb"), purpose="fine-tune"
)
validation_file = client.files.create(
    file=open(validation_file_name, "rb"), purpose="fine-tune"
)

# Creating Fine-Tuning Job
suffix_name = "crypto_news"
response = client.fine_tuning.jobs.create(
    training_file=training_file.id,
    validation_file=validation_file.id,
    model="gpt-3.5-turbo",
    suffix=suffix_name,
)

def format_test(row):
    formatted_message = [{"role": "user", "content": row['Support Query']}]
    return formatted_message

def predict(test_messages, fine_tuned_model_id):
    response = client.chat.completions.create(
        model=fine_tuned_model_id, messages=test_messages, temperature=0, max_tokens=50
    )
    return response.choices[0].message.content

#Creating results.pdf for predictions
def store_predictions(test_df, fine_tuned_model_id):
    test_df['Prediction'] = None
    for index, row in test_df.iterrows():
        test_message = format_test(row)
        prediction_result = predict(test_message, fine_tuned_model_id)
        test_df.at[index, 'Prediction'] = prediction_result

    test_df.to_csv("results.csv")
