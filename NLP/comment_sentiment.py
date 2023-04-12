from transformers import AutoTokenizer,AutoModelForSequenceClassification
from scipy.special import softmax

comment= "Weather is great today ! :)"

#Precpress tweet

comment_words = []

for word in comment.split(' '):
    if word.startswith('@') and len(word) > 1:
        word = '@user'
        
    elif word.startswith('http'):
        word = "http"
    comment_words.append(word)
    
comment_proc = " ".join(comment_words)
print(comment_proc )

#Load the model and tokenizer

Jarvis = "cardiffnlp/twitter-roberta-base-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(Jarvis)

tokenizer = AutoTokenizer.from_pretrained(Jarvis)

lables = ['Negative', 'Neutral' , 'Positive']

#sentiment_analysis

encoded_comment = tokenizer(comment_proc,return_tensors='pt')

output = model(encoded_comment['input_ids'],encoded_comment['attention_mask'])    

scores = output[0][0].detach().numpy()
scores = softmax(scores)
for i in range(len(scores)):
    l = lables[i]
    s = scores[i]
    print(l,s)

print(scores)
        
            
            
