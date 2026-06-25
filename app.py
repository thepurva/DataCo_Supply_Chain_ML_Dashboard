import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

st.set_page_config(page_title="DataCo Smart Supply Chain", layout="wide")
st.title("📦 DataCo Smart Supply Chain - Machine Learning Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("data.csv", encoding="latin1")
    return df

try:
    df = load_data()
    st.success(f"✅ SUCCESSFULLY{df.shape[0]:,} RECORDS LOAD!")

   
    tab1, tab2 = st.tabs(["📊 DATA Analytics & GRAPHS", "🤖 MACHINE LEARNING PREDICTION"])

    with tab1:
        st.subheader("📈 SUPPLY CHAIN BUSSINESS GLIMPSE")
        
        
        col1, col2, col3 = st.columns(3)
        col1.metric("TOTAL ORDERS (Rows)", f"{df.shape[0]:,}")
        col2.metric("TOTAL FEATURES(Columns)", df.shape[1])
        if 'Sales' in df.columns:
            col3.metric("Total Sales", f"${df['Sales'].sum():,.2f}")

        st.write("---")

       
        if 'Delivery Status' in df.columns:
            st.subheader("🚚 DELIVERY STATUS DISTRIBUTION")
            status_counts = df['Delivery Status'].value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']
            fig = px.bar(status_counts, x='Status', y='Count', color='Status', text_auto=True,
                         title="WHICH ORDERS ARE LATE AND ON TIME?")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("🤖 LATE DELIVERY PREDICTION MODEL (Machine Learning)")
        st.write("WE USE Random Forest Classifier FOR DELAY PREDICTION...")

       
        required_cols = ['Days for shipping (real)', 'Days for shipment (scheduled)', 'Late_delivery_risk']
        
        
        if all(col in df.columns for col in required_cols):
            
            
            ml_df = df[required_cols].dropna().sample(n=20000, random_state=42)
            
            X = ml_df[['Days for shipping (real)', 'Days for shipment (scheduled)']]
            y = ml_df['Late_delivery_risk']

           
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # मॉडेल ट्रेनिंग
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)

           
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

           
            st.metric(label="🎯 Model Accuracy", value=f"{acc*100:.2f}%")
            st.info("💡 TIP: THIS IS THE REAL WORLD DATA SO THE ACCURACY ARE REAL!")

          
            st.write("---")
            st.subheader("🔮Live Prediction")
            
            with st.form("prediction_form"):
                real_days = st.number_input("Days for shipping real", min_value=0, max_value=20, value=3)
                sched_days = st.number_input("Days for shipment scheduled", min_value=0, max_value=20, value=4)
                
                submit = st.form_submit_button("PREDICT")
                
                if submit:
                    input_data = np.array([[real_days, sched_days]])
                    prediction = model.predict(input_data)
                    
                    if prediction[0] == 1:
                        st.error("🚨 Late Delivery Risk!")
                    else:
                        st.success("✅ AWESOME! THIS ORDER IS ON TIME")
        else:
            st.warning("COLUMNS REQUIRED FOR MACHINE LEARNING ('Days for shipping (real)', 'Days for shipment (scheduled)', 'Late_delivery_risk') ARE NOT IN THIS DATASET")

except Exception as e:
    st.error(f"❌ ERROR OCCUR: {e}")