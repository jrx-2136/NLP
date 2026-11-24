import re

def word_tokenizer(sentence):
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+|www\.[-\w.]+(?:\.[a-z]+)+'
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    date_pattern = r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b'
    num_pattern = r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b'
    punct_pattern = r'[.,!?;:()\[\]{}"\'…।॥\-]'
    word_pattern = r'[^\s.,!?;:()\[\]{}"\'…।॥\-]+'
    
    pattern = f'({url_pattern})|({email_pattern})|({date_pattern})|({num_pattern})|({punct_pattern})|({word_pattern})'
    
    tokens = []
    for match in re.finditer(pattern, sentence, re.UNICODE):
        token = match.group(0)
        if token.strip():
            tokens.append(token)
    return tokens

text = "मेरा ईमेल test@example.com है और वेबसाइट https://google.com है। आज 12/03/2024 है और मेरे पास 1,234.50 रुपये हैं!"
print(word_tokenizer(text))
