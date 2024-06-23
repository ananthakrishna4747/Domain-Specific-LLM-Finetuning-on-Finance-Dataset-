import evaluate
import json
from bert_score import score

final_metrics = {
    "hugging_face": {},
    "openai": {}
}
def calculate_metrics(predictions, references, model_type):
    rouge = evaluate.load("rouge")
    # Compute the Rouge score
    results = rouge.compute(predictions=predictions, references=references)
    final_metrics[model_type]["rouge"] = results

    # Compute BLEU Score
    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=predictions, references=references)
    final_metrics[model_type]["bleu"] = results

    # Bert score
    P, R, F1 = score(predictions, references, lang="en", model_type="bert-base-uncased", verbose=True)
    final_metrics[model_type]["bert_score"] = {"Precision": P.mean().item(), "Recall": R.mean().item(), "F1_score": F1.mean().item()}



filenames = [["record_metrics_data_huggingface.json", "hugging_face"], ["record_metrics_data_openai.json", "openai"]]
for file in filenames:
    with open(file[0]) as fp:
        data = json.load(fp)

    predictions = data.get("predictions", None)
    references = data.get("references", None)

    calculate_metrics(predictions, references, model_type=file[1])

print(final_metrics)
with open("metrics.json", 'w') as fm:
    json.dump(final_metrics, fm)







