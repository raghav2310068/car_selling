import gdown
import os
import streamlit as st
import pandas as pd 
import pickle
URL= "https://drive.google.com/uc?id=1ymO321FUcZFkXVzZwIiQdHhW_0GzVEmf"
output = "model.pkl"
if not os.path.exists(output):
    gdown.download(URL, output, quiet=False)

with open(output, "rb") as f:
    model = pickle.load(f)


def space():
    st.markdown("")

# with open("model.pkl", "rb") as f:
#     model=pickle.load(f)
with open("allcars.pkl", "rb") as f:
    allcars=pickle.load(f)
with open("preprosessor.pkl", "rb") as f:
    preprosessor=pickle.load(f)
with open("car_list.pkl", "rb") as f:
    cars = pickle.load(f)
with open("encoder.pkl", "rb") as f:
    encoder=pickle.load(f)
encoder.fit(allcars)

st.title("🚗 Car Selling Price Prediction")
space()
# st.write("Your Car Brand")
brand=st.selectbox("Your Car Brand",list(cars.keys()))
car_model=st.selectbox("Select Your Car Model",list(cars[brand]))
# encoded_car_model=encoder.transform(list(allcars))
encoded_car_model=encoder.transform([car_model])[0]
num_owners=st.slider(label="Number of Owners",min_value=1,max_value=4,step=1)
total_driven=st.number_input("Enter The Total Kilometers Driven")
year=st.number_input(label="Age Of Your Car")
seller_type = st.selectbox("Seller Type", ['Dealer', 'Individual'])
fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
transmission_type = st.selectbox("Transmission", ['Manual', 'Automatic'])
milage=st.slider("Milage",min_value=2.00,max_value=35.00,step=0.2)
engine=st.slider("Engine",min_value=500,max_value=4000,step=1)
power=st.slider("Power",min_value=22.00,max_value=100.00,step=0.15)
setas=int(st.selectbox("Number Of Seats",[f"{i}" for i in range(10)]))
if st.button("Predict My Car Price"):
    input_df = pd.DataFrame([[
    encoded_car_model, year, total_driven, seller_type, fuel_type,
    transmission_type, milage, engine, power, setas
]], columns=[
    "model", "vehicle_age", "km_driven", "seller_type", "fuel_type",
    "transmission_type", "mileage", "engine", "max_power", "seats"
])

    input_transformed=preprosessor.transform(input_df)
    prediction=model.predict(input_transformed)
    st.title(f"{prediction}")