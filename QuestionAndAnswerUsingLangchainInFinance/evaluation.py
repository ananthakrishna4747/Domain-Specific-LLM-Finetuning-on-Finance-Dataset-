from QnA import generate_answer
predictions = []
references = []

import json


# Load the JSON data from the file
with open('sample.json', 'r') as file:
    data = json.load(file)

i = 0
# Iterate through each item in the JSON data
for item in data:
    question = item['Question']
    context = item['Context']
    references_answers = item['Reference']
    prediction = generate_answer(question, context, model_choice='hugging_face')
    predictions.append(prediction)
    references.append(references_answers)
    i += 1

record_metrics = {}

record_metrics["predictions"] = predictions
record_metrics["references"] = references

with open("record_metrics_data_huggingface.json", "w") as fpj:
    json.dump(record_metrics, fpj)

