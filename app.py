import re
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
import pickle
import xgboost as xgb
from nltk.corpus import movie_reviews, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
app= Flask(__name__)
#xgb_clf = pickle.load(open('model.pkl', 'rb'))
booster = xgb.Booster()
booster.load_model('test_model.bin')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    file= request.form['qa']
    text_df=pd.read_excel(file)

    #data cleaning code
    #text_df['CX Cat'] = np.where(text_df[' CUSTOMER_RESOLVER_COMM'].str.contains("à",na=False), 'Not Readable' , text_df['CX Cat'])

    stemmer = nltk. stem.SnowballStemmer('english')

    nltk.download('stopwords')
    stop_words = set(nltk.corpus.stopwords.words('english'))

    def tokenize(text):
        tokens = [word for word in nltk.word_tokenize(text) if (len(word) > 3 and len(word.strip('www./'))>2 and len(re.sub('\d+','', word.strip('www./'))))]
        tokens = map(str.lower, tokens) 
        stems = [stemmer. stem(item) for item in tokens if (item not in stop_words)] 
        return stems

    #vectorizer_tf = TfidfVectorizer(tokenizer=tokenize, stop_words=None, max_df=0.75, max_features=1000, lowercase=False, ngram_range=(1,2))
    #vectorizer_tf= pickle.load(open('vectorizer.pkl', 'rb'))
    #train_vectors = vectorizer_tf.fit_transform(X_train[' CUSTOMER_RESOLVER_COMM'].apply(str))
    vectorizer_tf = TfidfVectorizer(tokenizer=tokenize, stop_words=None, max_df=0.75, max_features=1000, lowercase=False, ngram_range=(1,2))
    full_vectors = vectorizer_tf.fit_transform(text_df[' CUSTOMER_RESOLVER_COMM'].apply(str))

    #full_vectors = vectorizer_tf.transform(text_df[' CUSTOMER_RESOLVER_COMM'].apply(str))
    #test_df=pd.DataFrame(full_vectors.toarray(), columns=vectorizer_tf.get_feature_names())
    #test_df=pd.concat([test_df,test_df['Category'].reset_index(drop=True)], axis=1)
    #test_df.head()

    #full_vectors = vectorizer_tf.transform(test_df[' CUSTOMER_RESOLVER_COMM'].apply(str))

    predict_all = booster.predict(full_vectors)

    rev_catogeries = {0:'Proper',1:'Stereo Type Reply',2:'Futuristic',3:'Diverted',4:'Not Readable'}

    #test_df['Category'] = test_df['CX Cat'].map(catogeries)
    #df.head()

    #print('classification_report :\n',classification_report(df['Category'], predict_all))

    pred_df = pd.Series(predict_all).to_frame()
    pred_df.columns = ['Predicted']
    #pred_df

    df_out = pd.merge(text_df,pred_df,how = 'left',left_index = True, right_index = True)
    #df_out

    df_out['Predicted CX Cat'] = df_out['Predicted'].map(rev_catogeries)
    #df_out.head()

    df_out.drop(['Predicted'], axis = 1,inplace=True)
    #df_out.info()

    df_out.to_csv('Data_NOV.csv', header=True, index=False)


    

if __name__ == "__main__":
    app.run(debug=True)