import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from file_read import read_multiple_csv_from_zipped_csv
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
nltk.download('wordnet')

import glob 
import os 
  
# merging the files 
joined_files = os.path.join("dataset_files", "clean_file*.zip") 
  
# A list of all joined files is returned 
joined_list = glob.glob(joined_files) 
  
# Finally, the files are joined 
df = pd.concat(map(pd.read_csv, joined_list), ignore_index=True) 
print(df)


price_map = {
    'cheap-eats': ('cheap', 'inexpensive', 'low-price', 'low-cost', 'economical',
                   'economic', 'affordable'),
    'mid-range': ('moderate', 'fair', 'mid-price', 'reasonable', 'average'),
    'fine-dining': ('expensive', 'fancy', 'lavish')
}


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def process_sentences(text):
    temp_sent =[]

    # Tokenize words
    words = nltk.word_tokenize(text)

    # Lemmatize each of the words based on their position in the sentence
    tags = nltk.pos_tag(words)
    for i, word in enumerate(words):
        if tags[i][1] in ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'):  # only verbs
            lemmatized = lemmatizer.lemmatize(word, 'v')
        else:
            lemmatized = lemmatizer.lemmatize(word)
        
        # Remove stop words and non alphabet tokens
        if lemmatized not in stop_words and lemmatized.isalpha(): 
            temp_sent.append(lemmatized)

    # Some other clean-up
    full_sentence = ' '.join(temp_sent)
    full_sentence = full_sentence.replace("n't", " not")
    full_sentence = full_sentence.replace("'m", " am")
    full_sentence = full_sentence.replace("'s", " is")
    full_sentence = full_sentence.replace("'re", " are")
    full_sentence = full_sentence.replace("'ll", " will")
    full_sentence = full_sentence.replace("'ve", " have")
    full_sentence = full_sentence.replace("'d", " would")
    return full_sentence

def recommend(input_text):
    input_text = str(input_text.lower())
    data=df.copy()
    #extracting area from input
    area_input = []
    for area in df['location'].unique():
        if area in input_text:
            area_input.append(area)
            input_text = input_text.replace(area, "")
    if area_input:
        data=data[data['location'].isin(area_input)].all()
    #samething for cuisines
    cuisines_input = []
    for cuisi in df['cuisines'].unique():
        if cuisi in input_text:
            cuisines_input.append(cuisi)
            input_text =input_text.replace(cuisi, "")
    if cuisines_input:
        data = data[data['cuisines'].isin(cuisines_input)].all()
       #samething with dishes_liked
    dish_liked_input=[]
    for dish in df['dish_liked'].unique():
        if dish in input_text:
            dish_liked_input.append(dish)
            input_text =input_text.replace(dish, "")
    if dish_liked_input:
        data = data[data['dish_liked'].isin(dish_liked_input)].all()
    #samething with type of restaurant
    rest_type_input=[]
    
    for rest_t in df['rest_type'].unique():
        if rest_t in input_text:
            rest_type_input.append(rest_t)
            input_text =input_text.replace(rest_t, "")
    if rest_type_input:
        data = data[data['rest_type'].isin(rest_type_input)].all()
    
    #cost for dining for two people
    for key, value in price_map.items():
        if any(v in input_text for v in value):
            data = data[data['price'] == key]
            break
        # Process user description text input 
    input_text = process_sentences(input_text)
   
    print('Processed user feedback:', input_text)

        # Init a TF-IDF vectorizer
    tfidfvec = TfidfVectorizer()
        # Fit data on processed reviews
    vec = tfidfvec.fit(data["bag_of_words"])
    features = vec.transform(data["bag_of_words"])
        # Transform user input data based on fitted model
    input_vector =  vec.transform([input_text])
        # Calculate cosine similarities between users processed input and reviews
    cos_sim = linear_kernel(input_vector, features)
        # Add similarities to data frame
    data['similarity'] = cos_sim[0]
        # Sort data frame by similarities
    data.sort_values(by='similarity', ascending=False, inplace=True)
    result= data[['name', 'location', 'cost', 'cuisines', 'reviews_list', 'similarity']][0:10]
    return result.to_html()

