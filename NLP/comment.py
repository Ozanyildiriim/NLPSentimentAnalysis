import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax



# Load the CSV file
df = pd.read_csv('D:/New Projects/NLP/winemag-data_first150k.csv')


# Load the model and tokenizer

Jarvis = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(Jarvis)
tokenizer = AutoTokenizer.from_pretrained(Jarvis)

# Initialize dictionary to store sentiment scores for each variety


varieties = {}
labels = ['Negative', 'Neutral', 'Positive']
for label in labels:
    varieties[label] = {}

# Process each comment and perform sentiment analysis


for i, row in df.head(20).iterrows():
    
    # Generate the comment string
    
    comment = f"{row['description']} ({row['country']}, {row['variety']}, {row['price']})"
    
    # Preprocess comment
   
    comment_words = []
    for word in comment.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        elif word.startswith('http'):
            word = "http"
        comment_words.append(word)
    comment_proc = " ".join(comment_words)
    
    # Perform sentiment analysis
   
    encoded_comment = tokenizer(comment_proc, return_tensors='pt')
    output = model(encoded_comment['input_ids'], encoded_comment['attention_mask'])    
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    
   
    # Update sentiment scores for the current variety
   
   
    variety = row['variety']
   
    for j in range(len(scores)):
        label = labels[j]
        score = scores[j]
       
       
        if variety in varieties[label]:
            varieties[label][variety]['score'] += score
            varieties[label][variety]['count'] += 1
       
       
        else:
            varieties[label][variety] = {'score': score, 'count': 1}

    
    
    # Print sentiment scores and NLP result for the current comment
   
   
    print(f"Variety: {row['variety']} | Description: {row['description']} | Labels: {labels}")
    print(f"    Scores: {scores}")
    print('-'*50)



# Print sentiment scores for all varieties


for variety in varieties['Positive']:
    
    
    print(f"{variety} sentiment scores:")
    
    for label in labels:
        score_sum = varieties[label][variety]['score']
        count = varieties[label][variety]['count']
        score_avg = score_sum / count
       
       
        print(f"    {label}: {score_avg:.4f}")
   
   
    print('-'*50)
