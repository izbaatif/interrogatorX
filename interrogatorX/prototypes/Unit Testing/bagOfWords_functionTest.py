import nltk
nltk.download('wordnet')

#initialize the tokenizer
tokenizer = nltk.RegexpTokenizer(r"\w+")

#code for all passing
def test_getwords(text):
    #make tokens of everything in the text
    tokens = tokenizer.tokenize(text)
    return tokens


#tokens dictionary for it to check
tokens = ["these" , "are" , "some", "random" , "tokens" , "these" , "are", "just", "for" , "testing"]

#code for all pass
def test_common(word , threshold):
    count=0
    for t in tokens:
        if t==word:
            count += 1
    if count>=threshold:
        return True
    else:
        return False
        
