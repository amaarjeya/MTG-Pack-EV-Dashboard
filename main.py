"""

File for streamlit dashboard

1. Display Pack EV Table

2. Allow user to pick set/ booster and run simulations 


"""


import streamlit as st
import pandas as pd

from functions import *

#Import data- 
# the static files for now, then allow users to upload files themselves

sets, setBoosterContents, setBoosterSheetCards, setBoosterContentWeights, setBoosterSheets, prices, cards, pack_prices = load_data()

# Title
st.title("MTG Pack EV Dashboard")

# Sidebar inputs
st.sidebar.header("Settings")
threshold = st.sidebar.number_input("Price Threshold", value=0.0, step=0.5)
min_date = st.sidebar.date_input("Minimum Release Date", value=pd.to_datetime("2024-06-01"))
# Convert to pandas Timestamp for comparisons
min_date = pd.Timestamp(min_date)


# Get EV table
latest_packs_EV = return_EV_table(threshold=threshold, min_date=min_date)

# Display table
st.subheader("Pack Expected Value")

# Pick a row height (in pixels). Adjust this until it looks good.
row_height = 35


# Calculate height dynamically
dynamic_height = min(len(latest_packs_EV) * row_height + 50, 1000) 

#DataFrame (styled)

st.dataframe(return_EV_table_styled(latest_packs_EV), use_container_width=True, height= dynamic_height)


st.sidebar.header("Simulation Settings")
set_input = st.sidebar.selectbox("Set", options=sets['code'].tolist(), index=0)
booster_input = st.sidebar.selectbox("Booster Type", options=['play','set','draft','collector'], index=0)
sim_size = st.sidebar.number_input("Packs per Simulation", value=100, step=100)
samples = st.sidebar.number_input("Number of Simulations", value=100, step=100)



pack_price, avg_pack_value, pack_value_at_sim_EV = run_simulation(set_input, booster_input, sim_size, samples, threshold)


# ----- Display Plot -----

st.subheader("Simulate Pack Openings")

fig = plot_simulation(avg_pack_value, pack_value_at_sim_EV, pack_price, sim_size, set_input, booster_input, threshold)
st.pyplot(fig)



