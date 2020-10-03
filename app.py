import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import folium
import pandas as pd 
import pickle 
import pydeck as pdk
from streamlit_folium import folium_static

#tensorflow==1.14
#keras==2.3.0
#scikit-learn==0.20.3

    



    

# loading in the model to predict on the data 
pickle_in = open('svc.pkl', 'rb') 
classifier = pickle.load(pickle_in) 


   
st.sidebar.markdown("** Wildfire Prediction **")

st.sidebar.markdown('Try to predict wildfires yourself with Machine Learning')  	 	
 	 	 	 	 	 	 	 	 	 	
def prediction(X,Y,month,day,FFMC,DMC,DC,ISI,temp,RH,wind,rain):   
   
    prediction = classifier.predict([[X,Y,month,day,FFMC,DMC,DC,ISI,temp,RH,wind,rain]]) 
    #prediction = classifier.predict([[1, 4, 9 ,1 ,91.5, 130.1, 807.1, 7.5, 21.3, 35, 2.2, 0]])
    #print(prediction) 
    
    classes={0:'safe',1:'On Fire'}
    #print(classes[prediction[0]])
    prediction2 = classes[prediction[0]]
    st.sidebar.markdown('The area is:')
    st.sidebar.text(prediction2)
    return prediction2 

@st.cache
def showMap():
    
    plotData = pd.read_csv("https://firms.modaps.eosdis.nasa.gov/data/active_fire/c6/csv/MODIS_C6_Global_24h.csv")
    Data = pd.DataFrame()
    Data = plotData[plotData['confidence'] > 50]  
    Data['latitude'] = plotData['latitude']
    Data['longitude'] = plotData['longitude']
    
    return Data

def accept_user_data():
    
    
    X = st.sidebar.text_input("X Coordinate: " ,'1')
    Y = st.sidebar.text_input("Y Coordinate: ",'4')
    day = st.sidebar.text_input("Day: ",'9')
    month = st.sidebar.text_input("Month: ",'1')
    FFMC = st.sidebar.text_input("FFMC: ",'91.5')
    DMC = st.sidebar.text_input("DMC: ",'130.1')
    DC = st.sidebar.text_input("DC: ",'807.1')
    ISI = st.sidebar.text_input("ISI: ",'7.5')
    temp = st.sidebar.text_input("Temperature (°C): ",'21.3')
    RH = st.sidebar.text_input("Relative Humidity (%): ",'35')
    wind = st.sidebar.text_input("Wind Speed (km/h): ",'2.2')
    rain = st.sidebar.text_input("Rain(mm/m2): ",'0')
    result =""
    
    
    #### Show the picture
    st.markdown("** The Latest Fire Location from Our Satellite **: ")
    token = "pk.eyJ1IjoiZTk2MDMxNDEzIiwiYSI6ImNqdDcxbW1kMzBhbWE0M25uYmswaWNnc3EifQ.xUELGj4ak4EIaPPYQUnYug" # your mapbox token
    tileurl = 'https://api.mapbox.com/v4/mapbox.satellite/{z}/{x}/{y}@2x.png?access_token=' + str(token)
    # center on Liberty Bell
    m = folium.Map(location=[22.999727, 121.127028], zoom_start=2, tiles=tileurl, attr='Mapbox')
    # add marker for Liberty Bell
    tooltip = "TW"
    folium.Marker(
            [25.042843, 121.560666], popup="TW", tooltip=tooltip
        ).add_to(m)
    satellitePlot = showMap()
    maps = satellitePlot[['latitude','longitude']]
    locations = maps[['latitude', 'longitude']]
    locationlist = locations.values.tolist()
    for point in range(1, 101):
        folium.Marker(locationlist[-point], icon=folium.Icon(color='orange', icon_color='red', icon='fire',radius=4, angle=0)).add_to(m)
    folium_static(m)
        # call to render Folium map in Streamlit

    
    
    
    #user_prediction_data = np.array([X,Y,day,DMC,DC,ISI,temp,RH,wind,rain]).reshape(1,-1)
    
    if st.sidebar.button("Predict"): 
        result = prediction(X,Y,month,day,FFMC,DMC,DC,ISI,temp,RH,wind,rain) 
    st.sidebar.success('The output is {}'.format(result)) 
    #return user_prediction_data


def main():


    with open("style.css") as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

    st.title("Automated Wild Fire Detection App")

    uploaded_image = Image.open('test_images/image2.png')
    st.markdown("** Fire Observation: **")
    st.image(uploaded_image, width = 500)
    
    accept_user_data()
    
    
    plotData = showMap()
    st.subheader("Fire Visualization for the past 48 hours:")
    st.map(plotData, zoom = 0)
   
  



    
if __name__=='__main__': 
    main()
    

