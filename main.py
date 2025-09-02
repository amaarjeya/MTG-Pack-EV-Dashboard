import streamlit as st
import pandas as pd
from functions import *

# Load demo data
demo_sets, demo_setBoosterContents, demo_setBoosterSheetCards, demo_setBoosterContentWeights, demo_setBoosterSheets, demo_prices, demo_cards, demo_pack_prices = load_data()

# ----------------------------
# Sidebar Upload
# ----------------------------
with st.sidebar.expander("Upload Files", expanded=False):
    uploaded_sets = st.file_uploader("Upload sets.csv", type="csv")
    uploaded_setBoosterContents = st.file_uploader("Upload setBoosterContents.csv", type="csv")
    uploaded_setBoosterSheetCards = st.file_uploader("Upload setBoosterSheetCards.csv", type="csv")
    uploaded_setBoosterContentWeights = st.file_uploader("Upload setBoosterContentWeights.csv", type="csv")
    uploaded_setBoosterSheets = st.file_uploader("Upload setBoosterSheets.csv", type="csv")
    uploaded_prices = st.file_uploader("Upload prices.csv", type="csv")
    uploaded_cards = st.file_uploader("Upload cards.csv", type="csv")
    uploaded_pack_prices = st.file_uploader("Upload pack_prices.csv", type="csv")

# ----------------------------
# Load uploaded files or fallback to demo
# ----------------------------
sets = pd.read_csv(uploaded_sets) if uploaded_sets else demo_sets
setBoosterContents = pd.read_csv(uploaded_setBoosterContents) if uploaded_setBoosterContents else demo_setBoosterContents
setBoosterSheetCards = pd.read_csv(uploaded_setBoosterSheetCards) if uploaded_setBoosterSheetCards else demo_setBoosterSheetCards
setBoosterContentWeights = pd.read_csv(uploaded_setBoosterContentWeights) if uploaded_setBoosterContentWeights else demo_setBoosterContentWeights
setBoosterSheets = pd.read_csv(uploaded_setBoosterSheets) if uploaded_setBoosterSheets else demo_setBoosterSheets
prices = pd.read_csv(uploaded_prices) if uploaded_prices else demo_prices
cards = pd.read_csv(uploaded_cards) if uploaded_cards else demo_cards
pack_prices = pd.read_csv(uploaded_pack_prices) if uploaded_pack_prices else demo_pack_prices

# ----------------------------
# Sidebar Table
# ----------------------------
st.sidebar.header("Table Settings")
threshold = st.sidebar.number_input("Price Threshold", value=2.0, step=0.5)
min_date = pd.Timestamp(st.sidebar.date_input("Minimum Release Date", value=pd.to_datetime("2025-01-01")))

# ----------------------------
# Sidebar Simulation
# ----------------------------
st.sidebar.header("Simulation Settings")


sim_size = st.sidebar.number_input("Packs per Simulation", value=100, step=100)
samples = st.sidebar.number_input("Number of Simulations", value=100, step=100)


# ----------------------------
# Main dashboard logic as a function
# ----------------------------
def render_dashboard():
    st.title("MTG Pack EV Dashboard")

    # EV Table
    latest_packs_EV = return_EV_table(
        sets, setBoosterContents, setBoosterSheetCards, setBoosterContentWeights,
        setBoosterSheets, prices, cards, pack_prices,
        threshold=threshold, min_date=min_date
    )

    set_input = st.sidebar.selectbox("Set", options=latest_packs_EV['Set Code'].unique())
    booster_input = st.sidebar.selectbox("Booster Type", options = latest_packs_EV[
    latest_packs_EV['Set Code'] == set_input
]['Booster Name'].unique())

    st.subheader("Pack Expected Value")
    row_height = 34
    dynamic_height = min(len(latest_packs_EV) * row_height + 50, 1000)
    st.dataframe(return_EV_table_styled(latest_packs_EV), width='stretch', height=dynamic_height)

    # Preprocessing
    booster_subsets, sheets_subsets, cards_by_sheet = preprocess_for_sim(
        setBoosterSheetCards, setBoosterSheets, setBoosterContents,
        setBoosterContentWeights, prices, cards, threshold=threshold
    )

    # --- Simulation logic ---

    pack_price, avg_pack_value, pack_value_at_sim_EV = run_simulation(
            latest_packs_EV, set_input, booster_input, sim_size, samples, threshold,
            booster_subsets, sheets_subsets, cards_by_sheet
        )

    
    # Display results
    st.subheader("Simulate Pack Openings")
    fig = plot_simulation(avg_pack_value, pack_value_at_sim_EV, pack_price, sim_size, set_input, booster_input, threshold)
    st.pyplot(fig)

    #Card table 
    st.subheader("Simulated Card Prices")
    
    card_table = sim_card_prices(sim_size, pack_price, set_input, booster_input, booster_subsets, sheets_subsets, cards_by_sheet)
    st.dataframe(card_table)

# ----------------------------
# Render dashboard
# ----------------------------
render_dashboard()
