from transformers import BertTokenizer, BertModel

class Embedding():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        self.model = BertModel.from_pretrained("bert-large-uncased")


    def cover(self,text):
        encoded_input = self.tokenizer(text, return_tensors='pt')
        output = self.model(**encoded_input)
        return output[1]
