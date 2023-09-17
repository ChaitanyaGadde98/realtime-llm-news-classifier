import torch
from transformers import RobertaModel, RobertaTokenizer
# from utils.model import RobertaClass


def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model


def inference(model, sentence):
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    token_type_ids = inputs.get("token_type_ids", None)  # token_type_ids may not be used by some models

    with torch.no_grad():
        probabilities = model(input_ids, attention_mask, token_type_ids)

    _, predicted_class = torch.max(probabilities, dim=1)

    return predicted_class.item()


# if __name__ == "__main__":
#     model_path = "/Users/cvsgadde/GSU/SEM2/NLP/Project/code/model/roberta_model/pytorch_roberta_news_1.bin"  
#     sample_sentence = "This is a sample sentence."
#
#     model = load_model(model_path)
#     predicted_class = inference(model, sample_sentence)
#
#     print(f"Predicted class: {predicted_class}")
