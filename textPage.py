
import streamlit as st
import streamlit.components.v1 as components
from textblob import TextBlob
from PIL import Image
import text2emotion as te
import plotly.graph_objects as go

import pickle
import joblib 


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

def getSentiments(userText, type):

    if(type == 'Positive/Negative/Neutral - TextBlob'):

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

    elif(type == 'Positive/Negative/Neutral - Our built ML model'): 

        # using our built ML model?
        with open('vectorizer_bow.pickle', 'rb') as file:
            vectorizer_fit = pickle.load(file)
        # vectorizer_fit = pickle.load('vectorizer_bow.pickle')
        model = joblib.load('svc_model.pkl')

        x_test = vectorizer_fit.transform([userText]) 

        y_test = model.predict(x_test)[0]

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

    # type = 'Positive/Negative/Neutral - TextBlob'

    type = st.selectbox(
     'Type of analysis',
     ('Positive/Negative/Neutral - Our built ML model', 'Positive/Negative/Neutral - TextBlob'))    

    # type = st.selectbox(
    #  'Type of analysis',
    #  ('Positive/Negative/Neutral - TextBlob', 'Happy/Sad/Angry/Fear/Surprise - text2emotion'))    

    st.text("")
    if st.button('Predict'):
        if(userText!="" and type!=None):
            st.text("")
            st.components.v1.html("""
                                <h3 style="color: #0284c7; font-family: Source Sans Pro, sans-serif; font-size: 28px; margin-bottom: 10px; margin-top: 50px;">Result</h3>
                                """, height=100)
            getSentiments(userText, type)

