from transformers import pipeline

generator = pipeline('text-generation', model= 'distilgpt2')
result = generator("Artificial intelligence will", max_length=50)
print(result[0]['generated_text'])
