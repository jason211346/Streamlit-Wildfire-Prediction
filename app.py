import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image  
import numpy as np
from streamlit_folium import folium_static
import folium
import pandas as pd 
import pickle 

#keras==2.3.0

# loading in the model to predict on the data 
pickle_in = open('svc.pkl', 'rb') 
classifier = pickle.load(pickle_in) 

with open("style.css") as f:
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)


st.title("Wild Fire Detection App")




uploaded_image = Image.open('test_images/image2.png')
st.markdown("** Original Raw Image: **")
st.image(uploaded_image, width = 500)
    


#### Show the picture
st.markdown("** The satellite images **: ")
    
from streamlit_folium import folium_static
import folium
    
token = "pk.eyJ1IjoiZTk2MDMxNDEzIiwiYSI6ImNqdDcxbW1kMzBhbWE0M25uYmswaWNnc3EifQ.xUELGj4ak4EIaPPYQUnYug" # your mapbox token
tileurl = 'https://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}@2x.png?access_token=' + str(token)
# center on Liberty Bell
m = folium.Map(location=[22.999727, 121.127028], zoom_start=6, tiles=tileurl, attr='Mapbox')



   

# add marker for Liberty Bell
tooltip = "TW"
folium.Marker(
        [25.042843, 121.560666], popup="TW", tooltip=tooltip
    ).add_to(m)

# call to render Folium map in Streamlit
folium_static(m)




   
st.sidebar.markdown("** Wildfire Prediction **")

st.sidebar.markdown('Please input data')  	 	
 	 	 	 	 	 	 	 	 	 	
def prediction(X,Y,month,day,FFMC,DMC,DC,ISI,temp,RH,wind,rain):   
   
    prediction = classifier.predict([[X,Y,month,day,FFMC,DMC,DC,ISI,temp,RH,wind,rain]]) 
    #prediction = classifier.predict([[1, 4, 9 ,1 ,91.5, 130.1, 807.1, 7.5, 21.3, 35, 2.2, 0]])
    #print(prediction) 
    
    classes={0:'safe',1:'On Fire'}
    #print(classes[prediction[0]])
    prediction2 = classes[prediction[0]]
    st.sidebar.markdown('The total burnt area is:')
    st.sidebar.text(prediction2)
    return prediction2 

def accept_user_data():
    X = st.sidebar.text_input("Enter the X: " ,'1')
    Y = st.sidebar.text_input("Enter the Y: ",'4')
    day = st.sidebar.text_input("Enter the day: ",'9')
    month = st.sidebar.text_input("Enter the month: ",'1')
    FFMC = st.sidebar.text_input("Enter the FFMC: ",'91.5')
    DMC = st.sidebar.text_input("Enter the DMC: ",'130.1')
    DC = st.sidebar.text_input("Enter the DC: ",'807.1')
    ISI = st.sidebar.text_input("Enter the ISI: ",'7.5')
    temp = st.sidebar.text_input("Enter the temp: ",'21.3')
    RH = st.sidebar.text_input("Enter the RH: ",'35')
    wind = st.sidebar.text_input("Enter the wind: ",'2.2')
    rain = st.sidebar.text_input("Enter the rain: ",'0')
    result =""

    user_prediction_data = np.array([X,Y,day,DMC,DC,ISI,temp,RH,wind,rain]).reshape(1,-1)
	
    if st.sidebar.button("Predict"): 
        result = prediction(X,Y,month,day,FFMC,DMC,DC,ISI,temp,RH,wind,rain) 
    st.sidebar.success('The output is {}'.format(result)) 
    return user_prediction_data



 
  



    
if __name__=='__main__': 
    accept_user_data()
	

