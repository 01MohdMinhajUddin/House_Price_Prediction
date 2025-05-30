import pandas as pd
import streamlit as st
import pickle

with open("House Price Prediction.pkl","rb") as f:
    model=pickle.load(f)

location_list = model.named_steps['columntransformer'].transformers_[0][1].categories_[0]


# named_steps is the shortcut provide by sklearn
# model.named_steps['columntransformer'] -- This give Column Transformer
# .transformers_[0][1] -- This give actual onehotencoder where we fit our columns
# .categories_[0] --  This gives all the columns


# print(model)
print(location_list)



st.title("House Price Prediction")

# st.markdown("Enter the House Detail")

location=st.selectbox(
    "Select Location",
    sorted(location_list)
)

total_sqft=st.number_input(
    "Total Square Feet",
    min_value=300,max_value=10000,step=10
)

bath=st.number_input(
    "Enter Number of Bathrooms",
    max_value=15,min_value=1,step=0
)

bhk=st.number_input(
    "Enter Number of Bedrooms",
    min_value=1,max_value=25,step=1
)

if st.button("Predict Price"):
    df=pd.DataFrame([[location,total_sqft,bath,bhk]],columns=["location", "total_sqft","bath","bhk"])

    price=model.predict(df)[0]
    st.success(f"The Price is **{round(price,2)}** Lakhs rupees")
