
import streamlit as st
import streamlit.components.v1 as components
from textblob import TextBlob
from PIL import Image
import text2emotion as te
import plotly.graph_objects as go

import pickle
import joblib 

import torch
from torch import nn
from transformers import BertTokenizer, BertModel
import gdown 
from huggingface_hub import hf_hub_download


def plotPie(labels, values):
    fig = go.Figure(
        go.Pie(
        labels = labels,
        values = values,
        hoverinfo = "label+percent",
        textinfo = "value"
    ))
    st.plotly_chart(fig)

    
def getPolarity(userText):
    tb = TextBlob(userText)
    polarity = round(tb.polarity, 2)
    subjectivity = round(tb.subjectivity, 2)
    if polarity>0:
        return polarity, subjectivity, "Positive"
    elif polarity==0:
        return polarity, subjectivity, "Neutral"
    else:
        return polarity, subjectivity, "Negative"


def getSentiments(userText, type, model, tokenizer, device):


    if(type == 'TextBlob built-in library'):

        # using TextBlob
        polarity, subjectivity, status = getPolarity(userText)
        if(status=="Positive"):
            image = Image.open('./images/positive.PNG')
        elif(status == "Negative"):
            image = Image.open('./images/negative.PNG')
        else:
            image = Image.open('./images/neutral.PNG')
        col1, col2, col3 = st.columns(3)
        col1.metric("Polarity", polarity, None)
        col2.metric("Subjectivity", subjectivity, None)
        col3.metric("Result", status, None)
        st.image(image, caption=status)



    elif(type == 'My classical ML model: SVC + Bag of Words'): 

        # using our built ML model?
        with open('vectorizer_bow.pickle', 'rb') as file:
            vectorizer_fit = pickle.load(file)
        # vectorizer_fit = pickle.load('vectorizer_bow.pickle')
        model_ml = joblib.load('svc_model.pkl')

        x_test = vectorizer_fit.transform([userText]) 

        y_test = model_ml.predict(x_test)[0]

        ## convert output to string
        if y_test==0:
            ## i.e. negative 
            status = 'Negative'
            image = Image.open('./images/negative.PNG')

        elif y_test==1:
            ## i.e. neutral 
            status = 'Neutral'
            image = Image.open('./images/neutral.PNG')

        elif y_test==2:
            ## i.e. positive 
            status = 'Positive' 
            image = Image.open('./images/positive.PNG')

        # col1, col2 = st.columns(2)
        # col1.metric("Nothing1", None)
        # col2.metric("Nothing2", None)
        st.image(image, caption=status) 



    elif(type == 'My fine-tune distilBERT model'):


        def predict_sentiment(text, model, tokenizer, device, max_length=128):
            # model.eval()
            encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    _, preds = torch.max(outputs, dim=1)

            print(f'\npreds = {preds}')
            print(f'\npreds.item() = {preds.item()}')
            print("Shape of preds:", preds.shape)
            print("Size of preds:", preds.size())
            print("Data type of preds:", preds.dtype)
            print("Device of preds:", preds.device)
            print("First element of preds:", preds[0].item())
            print("Preds as a NumPy array:", preds.numpy())
            print("Preds as a list:", preds.tolist())

            # print("Mean of preds:", preds.mean().item())
            # print("Max of preds:", preds.max().item())
            # print("Min of preds:", preds.min().item())
            # print("Sum of preds:", preds.sum().item())

            return "Positive" if preds.item() == 2 else "Neutral" if preds.item() == 1 else "Negative"         


        ## prediction 
        sentiment = predict_sentiment(userText, model, tokenizer, device)


        ## convert output to string
        status = sentiment
        if status == 'Negative':
            ## i.e. negative             
            image = Image.open('./images/negative.PNG')

        elif status == 'Neutral':
            ## i.e. neutral             
            image = Image.open('./images/neutral.PNG')

        elif status == 'Positive':
            ## i.e. positive 
            image = Image.open('./images/positive.PNG')

        # col1, col2 = st.columns(2)
        # col1.metric("Nothing1", None)
        # col2.metric("Nothing2", None)
        st.image(image, caption=status) 


    elif(type == 'Happy/Sad/Angry/Fear/Surprise - text2emotion'):
        emotion = dict(te.get_emotion(userText))
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Happy 😊", emotion['Happy'], None)
        col2.metric("Sad 😔", emotion['Sad'], None)
        col3.metric("Angry 😠", emotion['Angry'], None)
        col4.metric("Fear 😨", emotion['Fear'], None)
        col5.metric("Surprise 😲", emotion['Surprise'], None)
        plotPie(list(emotion.keys()), list(emotion.values()))
        

def renderPage():


    # st.title("Sentiment Analysis Demo 😊😐😕😡")
    st.title("Sentiment Analysis Demo")
    components.html("""<hr style="height:3px;border:none;color:#333;background-color:#333; margin-bottom: 10px" /> """)
    # st.markdown("### User Input Text Analysis")
    st.subheader("Text Analysis from User Input")
    # st.text("Analyzing text data (Amazon Kindle reviews) given by an user and find sentiments ")
    # st.text("")
    st.text("Input: text that users input, i.e., Amazon Kindle product reviews, etc")
    st.text("Output: sentiments in either one of three scenarios Positive/Neutral/Negative")    
    userText = st.text_input('User Input', placeholder='Input text HERE')
    st.text("")

    # type = 'TextBlob built-in library'


    type = st.selectbox(
     'Type of analysis',
     ('My fine-tune distilBERT model', 'My classical ML model: SVC + Bag of Words', 'TextBlob built-in library') )    

    # type = st.selectbox(
    #  'Type of analysis',
    #  ('Positive/Negative/Neutral - TextBlob', 'Happy/Sad/Angry/Fear/Surprise - text2emotion'))    

    st.text("")


    ## cache so download BERT model only ONCE 
    # @st.cache(allow_output_mutation=True)
    # @st.cache

    ### 7/13/24 this is for BERT models -- do only ONCE ###
    class BERTClassifier(nn.Module):
        def __init__(self, bert_model_name, num_classes):
            super(BERTClassifier, self).__init__()
            self.bert = BertModel.from_pretrained(bert_model_name)
            self.dropout = nn.Dropout(0.1)
            self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

        def forward(self, input_ids, attention_mask):
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs.pooler_output
                x = self.dropout(pooled_output)
                logits = self.fc(x)
                return logits


    ## define some variables used here
    bert_model_name = 'bert-base-uncased'
    num_classes = 3        


    ## cache so download BERT model only ONCE 
    @st.cache_resource

    ## load saved trained model 
    # if 'model' not in locals() and 'model' not in globals():

    def get_bert_model():
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")
        model  = BERTClassifier(bert_model_name, num_classes).to(device)
        
        # model_path = 'https://drive.google.com/uc?id=1OvacWq2Vem4L7gdO0S_yteCNPKNtBpJ1'

        # model_name = 'sentiment_classifier.pth'
        # gdown.download(model_path, model_name)

        path_model = hf_hub_download(repo_id="chih3/bert_sentiment_analysis", filename="sentiment_classifier_500K_3epoch.pth")

        # model.load_state_dict(torch.load(model_path, map_location=device) )

        model.load_state_dict(torch.load(path_model, map_location=device) )
        
        model.eval()

        tokenizer = BertTokenizer.from_pretrained(bert_model_name)

        return device, tokenizer, model

    device, tokenizer, model = get_bert_model()


    ## when pressing Predict button ## 
    if st.button('Predict'):
        if(userText!="" and type!=None):
            st.text("")
            st.components.v1.html("""
                                <h3 style="color: #0284c7; font-family: Source Sans Pro, sans-serif; font-size: 28px; margin-bottom: 10px; margin-top: 50px;">Result</h3>
                                """, height=100)
            getSentiments(userText, type, model, tokenizer, device)



