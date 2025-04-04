import spacy
from textblob import TextBlob
import os


try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Modelo spaCy não encontrado. Instale com: python -m spacy download en_core_web_sm")
    exit()


def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    
    if polarity > 0.1:
        sentiment = "Positivo"
    elif polarity < -0.1:
        sentiment = "Negativo"
    else:
        sentiment = "Neutro"
    
    return sentiment, polarity


def recognize_entities(text):
    doc = nlp(text)
    entities = []
    
    for ent in doc.ents:
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        })
    
    return entities


def main():
   
    input_file = os.path.join("..", "inputs", "exemplos.txt")
    if not os.path.exists(input_file):
        print(f"Arquivo de entrada não encontrado: {input_file}")
        return
    
    
    with open(input_file, 'r', encoding='utf-8') as file:
        documents = [line.strip() for line in file if line.strip()]
    
    print("===== Análise de Texto Local =====")
    print("(Usando TextBlob para sentimentos e spaCy para entidades)\n")
    
    for i, text in enumerate(documents, 1):
        print(f"\nDocumento {i}:")
        print(f"Texto: {text}")
        
        
        sentiment, polarity = analyze_sentiment(text)
        print(f"\nSentimento: {sentiment} (Polaridade: {polarity:.2f})")
        
       
        entities = recognize_entities(text)
        if entities:
            print("\nEntidades encontradas:")
            for entity in entities:
                print(f"- {entity['text']} ({entity['label']})")
        else:
            print("\nNenhuma entidade reconhecida.")
        
        print("\n" + "-"*50)

if __name__ == "__main__":
    main()
