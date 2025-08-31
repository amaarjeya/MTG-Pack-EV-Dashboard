"""
OVERVIEW:

This code is used to create the Pack EV Table and Simulation graphs for MTG pack analysis. 

For both tools, the user can input a threshold, below which all card prices will be treated as $0.00 and not contribute towards EV analysis. 

### FULL DATA MAPPING PROCESS ###

~~~ Stitching foil/ non foil info to setBoosterSheetCards csv ~~~

1. Stitch setBoosterSheets 'sheetIsFoil' to the setBoosterSheetCards dataframe. 


2. Group by boosterName, setCode, sheetName --> Return the sheetIsFoil value



~~Getting Prices mapped to setBoosterSheetCards csv ~~~
3. Prices DB: Filter only to tcgplayer, paper

4. Check if sheet is foil. If foil, take UUID + foil
    Else, take UUID + normal

~~~ Mapping card EV and calculating sheet EV for a given booster Index ~~~


In setBoosterSheetCards, card EV is given by: 
card Weight / SUM [cardWeight by (setcode, sheetName, boosterName ) ]

Map Sum of card EV to setBoosterContents

 
In setBoosterContents, sheet EV column is given by:

SUM card EV * sheet Picks  


~~~Normalizing sheet EV by booster Weights~~~

Map SUM Sheet EV to setBoosterContentWeights

Booster Index EV is given by:

(SUM Sheet EV * Booster Index Weights ) / SUM Booster Index weights 

"""

"Dependencies"


import pandas as pd

import numpy as np


import matplotlib.pyplot as plt


import streamlit as st
import gdown


"""

Read Inputs, Clean Data

"""

#Mapping sets to boosters to sheets to sheet weights

#Heirarchy: Set, Booster, Sheet, Card
#Each of these has a weight



@st.cache_data
def load_data():
    """
    Downloads CSV files from Google Drive and loads them into pandas DataFrames.
    Returns each DataFrame individually.
    """
    # Google Drive file URLs
    urls = {
        "sets": "https://drive.google.com/file/d/1Q94b-uTEr5WGL8SGTpUmaDdTqhgmJcey/view?usp=drive_link",
        "setBoosterContents": "https://drive.google.com/file/d/1g91qFR3Agly5KhY4VEdy9GKRgwKgjU9G/view?usp=drive_link",
        "setBoosterSheetCards": "https://drive.google.com/file/d/1fib0Aj3ZVv4e4LZoDh0ClIew-IdD3gzN/view?usp=drive_link",
        "setBoosterContentWeights": "https://drive.google.com/file/d/1pAnpbd38aJz21pneOB5sEEf8xv6JmwQT/view?usp=drive_link",
        "setBoosterSheets": "https://drive.google.com/file/d/1LTnLfAcQ4AJWCTHZv0LgtC6c9ToFSB29/view?usp=drive_link",
        "prices": "https://drive.google.com/file/d/1xrWWAqUIxz2410YR1DzO3nuGD3iPO3fG/view?usp=drive_link",
        "cards": "https://drive.google.com/file/d/12_CKAzpxbVhO2mI4J7r70blp_2wQttDe/view?usp=drive_link",
        "pack_prices": "https://drive.google.com/file/d/1sAkTd6LS1F1Um4y1sz4aKxlXhuAvtfZd/view?usp=drive_link"
    }

    # Helper to download and read CSV
    def download_csv(url, filename):
        file_id = url.split('/d/')[1].split('/')[0]
        download_url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(download_url, filename, quiet=False)
        return pd.read_csv(filename)
    
    # Load each DataFrame
    sets = download_csv(urls["sets"], "sets.csv")
    setBoosterContents = download_csv(urls["setBoosterContents"], "setBoosterContents.csv")
    setBoosterSheetCards = download_csv(urls["setBoosterSheetCards"], "setBoosterSheetCards.csv")
    setBoosterContentWeights = download_csv(urls["setBoosterContentWeights"], "setBoosterContentWeights.csv")
    setBoosterSheets = download_csv(urls["setBoosterSheets"], "setBoosterSheets.csv")
    prices = download_csv(urls["prices"], "prices.csv")
    cards = download_csv(urls["cards"], "cards.csv")
    pack_prices = download_csv(urls["pack_prices"], "pack_prices.csv")
    
    return sets, setBoosterContents, setBoosterSheetCards, setBoosterContentWeights, setBoosterSheets, prices, cards, pack_prices



sets, setBoosterContents, setBoosterSheetCards, setBoosterContentWeights, setBoosterSheets, prices, cards, pack_prices = load_data()
 
def get_merged_cardprices(
        threshold = 0,
        setBoosterSheetCards = setBoosterSheetCards,
        setBoosterSheets = setBoosterSheets,
        prices = prices, 
        cards = cards,
        ):
    
    merged_setBoosterSheetCards = setBoosterSheetCards.merge(setBoosterSheets[['boosterName', 'setCode', 'sheetName', 'sheetIsFoil']], on=['boosterName', 'setCode', 'sheetName'], how='left')

    # Step 1: Map sheetIsFoil to setBoosterSheetCards df

    #Filter down to tcgplayer, USD, paper

    prices = prices[
        (prices['currency'] == "USD") &
        (prices['gameAvailability'] == "paper") &
        (prices['priceProvider'] == "tcgplayer") 
    ]

    #rename UUID to cardUuid

    prices = prices.rename(columns= {'uuid' : 'cardUuid'})


    cards = cards.rename(columns= {'uuid' : 'cardUuid'})



    #Prices added to merged dataframe

    merged_setBoosterSheetCards["desiredFinish"] = merged_setBoosterSheetCards["sheetIsFoil"].map(
        {True: "foil", False: "normal"}
    )

    merged_cardprices = merged_setBoosterSheetCards.merge(
        prices[["cardUuid", "cardFinish", "price"]],
        left_on=["cardUuid", "desiredFinish"],
        right_on=["cardUuid", "cardFinish"],
        how="left"
    )


    # Get total sheet Weight

    merged_cardprices["totalSheetWeight"] = (
        merged_cardprices
        .groupby(["boosterName", "setCode", "sheetName", 'sheetIsFoil'])["cardWeight"]
        .transform("sum")
    )

    #Add names of cards

    merged_cardprices = merged_cardprices.merge(
        cards[['cardUuid', 'name']].rename(columns={"name": "cardName"}),
        on='cardUuid',
        how='left'
    )


    merged_cardprices['price'] = np.where(
            merged_cardprices['price'] <= threshold, 0,
            (merged_cardprices['price']
            ))


    #Add column for card EV, per sheet

    merged_setBoosterSheetCards[merged_setBoosterSheetCards['setCode'] == "INR"].groupby(['boosterName', 'setCode', 'sheetName']).count()

    merged_cardprices['card EV'] = merged_cardprices['price'] * merged_cardprices['cardWeight'] / merged_cardprices['totalSheetWeight']



    # Get normalized sheet EV 

    merged_cardprices["sheet EV"] = (
        merged_cardprices
        .groupby(["boosterName", "setCode", "sheetName"])["card EV"]
        .transform("sum")
    )


    #Sheet Variance Calc

    merged_cardprices['card contribution to sheet Var'] = (
        merged_cardprices['price'] ** 2 ) * (merged_cardprices['cardWeight'] / merged_cardprices['totalSheetWeight']) 
    - (merged_cardprices['card EV'] ** 2) 


    merged_cardprices["sheet Var"] = (
        merged_cardprices
        .groupby(["boosterName", "setCode", "sheetName"])["card contribution to sheet Var"]
        .transform("sum")
    )

    return merged_cardprices


def get_pack_EV(
                
                setBoosterContents = setBoosterContents, 
                setBoosterSheetCards = setBoosterSheetCards,
                setBoosterContentWeights = setBoosterContentWeights,
                setBoosterSheets = setBoosterSheets, 
                prices = prices, 
                cards = cards,
                threshold = 0
                ):
    
    merged_cardprices = get_merged_cardprices(threshold = threshold,
        setBoosterSheetCards = setBoosterSheetCards,
        setBoosterSheets = setBoosterSheets,
        prices = prices, 
        cards = cards)
    
 

    #Merge Sheet EV to boosterContents dataframe

    sheet_EV = (
        merged_cardprices
        .groupby(['boosterName', 'setCode', 'sheetName'], as_index=False)
        .agg({
            'sheet EV': 'mean',
            'sheet Var': 'mean'
        })
    )

    booster_index_EV = setBoosterContents.merge(
        sheet_EV, on= ['boosterName', 'setCode', 'sheetName'], how='left')




    #Update Sheet EV to be weighted by sheet Picks (given multiple sheet picks in a pack)

    booster_index_EV['sheet EV'] = booster_index_EV['sheet EV'] * booster_index_EV['sheetPicks']


    #Sheet Var Weighted by Sheet Picks

    booster_index_EV['sheet Var'] = booster_index_EV['sheet Var'] * booster_index_EV['sheetPicks']



    #Add booster weight column, using helper df

    booster_index_EV_2 = booster_index_EV.merge(setBoosterContentWeights, on= ['boosterIndex', 'boosterName', 'setCode'], how='left')



    total_booster_weight_df = booster_index_EV_2[['boosterIndex', 'boosterName', 'setCode', 'boosterWeight']]

    total_booster_weight_df = total_booster_weight_df.drop_duplicates()


    total_booster_weight_df["totalboosterWeight"] = (
        total_booster_weight_df
        .groupby(["boosterName", "setCode"])["boosterWeight"]
        .transform("sum")
    )



    total_booster_weight_df = total_booster_weight_df[['boosterName', 'setCode', 'totalboosterWeight']]

    total_booster_weight_df[
        (total_booster_weight_df['setCode'].isin(["MH3"])) &
        
        (total_booster_weight_df['boosterName'].isin(["play"]))
    ]




    # Get total booster Weight added to main dataframe
    booster_index_EV_2 = booster_index_EV_2.merge(
        total_booster_weight_df, 
        on=['boosterName', 'setCode']
    )

    booster_index_EV_2 = booster_index_EV_2.drop_duplicates()




    #Helper column for boosterIndex EV

    booster_index_EV_helper = booster_index_EV_2




    booster_index_EV_helper["boosterIndex EV"] = (
        booster_index_EV_helper
        .groupby(["boosterName", "setCode", 'boosterIndex'])["sheet EV"]
        .transform("sum")
    )

    booster_index_EV_helper["boosterIndex Var"] = (
        booster_index_EV_helper
        .groupby(["boosterName", "setCode", 'boosterIndex'])["sheet Var"]
        .transform("sum")
    )

    booster_index_EV_helper[
        (booster_index_EV_helper['setCode'].isin(["MH3"])) &
        (booster_index_EV_helper['boosterName'].isin(["play"]))
    ]


    # Get total EV and Var per booster
    booster_index_EV_2["booster Index EV"] = (
        booster_index_EV_2
        .groupby(["boosterIndex", "boosterName", "setCode"])["sheet EV"]
        .transform("sum")
    )


    booster_index_EV_2["booster Index Var"] = (
        booster_index_EV_2
        .groupby(["boosterIndex", "boosterName", "setCode"])["sheet Var"]
        .transform("sum")
    )



    #Calculate Pack EV

    booster_index_EV_2['Pack EV'] = booster_index_EV_2['booster Index EV'] * booster_index_EV_2['boosterWeight'] /booster_index_EV_2['totalboosterWeight']

    #Calculate Pack Var

    booster_index_EV_2['Pack Var'] = booster_index_EV_2['boosterWeight'] / booster_index_EV_2['totalboosterWeight'] * (
    booster_index_EV_2['booster Index Var'] + ((booster_index_EV_2['booster Index EV'] - booster_index_EV_2['Pack EV']) ** 2) 
    )



    pack_EV = booster_index_EV_2[['setCode', 'boosterName', 'boosterIndex', 'Pack EV', 'Pack Var']]

    pack_EV = pack_EV.drop_duplicates(
        subset=["setCode", "boosterName", "boosterIndex"]
    )

    pack_EV[
        (pack_EV['setCode'].isin(["MH3"])) &
        (pack_EV['boosterName'].isin(["play"]))
    ]


    pack_EV = (
        pack_EV
        .groupby(["setCode", "boosterName"], as_index=False)
        .agg({
            "Pack EV": "sum",   # or mean, depending on how you defined it
            "Pack Var": "sum"   # sum is correct for variance decomposition across indices
        })
    )


    # Get total EV and Var per pack
    booster_index_EV_2["Pack EV"] = (
        booster_index_EV_2
        .groupby(["boosterName", "setCode"])["Pack EV"]
        .transform("sum")
    )


    booster_index_EV_2["booster Index Var"] = (
        booster_index_EV_2
        .groupby(["boosterName", "setCode"])["Pack Var"]
        .transform("sum")
    )

    pack_EV[
        (pack_EV['setCode'].isin(["MH3"])) &
        (pack_EV['boosterName'].isin(["play"]))]


    return pack_EV



#Get Last N Sets (already released) from sets Dataframe

def last_n_sets(min_date, sets_df = sets):

    # Convert releaseDate to pandas datetime
    sets_df['releaseDate'] = pd.to_datetime(sets_df['releaseDate'], errors='coerce')
    
    # Filter out unreleased sets (releaseDate in the future or NaT)
    released_sets = sets_df[
        (sets_df['releaseDate'] >= min_date) &
        (sets_df['releaseDate'] <= pd.Timestamp.today())
    ].copy()

    
    
    # Sort by releaseDate descending
    released_sets = released_sets.sort_values('releaseDate', ascending=False)
    
    released_sets = released_sets[['name', 'code', 'releaseDate']].rename(columns={'code': 'setCode'})
    # Return the top N
    return released_sets



# Make Final EV Table

def return_EV_table(
        sets, 
        setBoosterContents, 
        setBoosterSheetCards, 
        setBoosterContentWeights, 
        setBoosterSheets, 
        prices, 
        cards, 
        pack_prices,
        threshold = 0, 
        min_date = '2024-06-01'):


    df = last_n_sets(min_date, sets_df = sets)

    main_boosters = ['draft', 'set', 'collector', 'default', 'play']

    pack_EV = get_pack_EV(setBoosterContents = setBoosterContents, 
                setBoosterSheetCards = setBoosterSheetCards,
                setBoosterContentWeights = setBoosterContentWeights,
                setBoosterSheets = setBoosterSheets, 
                prices = prices, 
                cards = cards,
                threshold = threshold)

    pack_EV = pack_EV[pack_EV['boosterName'].isin(main_boosters)]


    latest_sets = df['setCode'].tolist()

    latest_packs_EV = pack_EV[pack_EV['setCode'].isin(latest_sets)]

    latest_packs_EV = latest_packs_EV.merge(
        df,
        on= 'setCode',
        how='left'
    )


    latest_packs_EV = latest_packs_EV.sort_values(by=['releaseDate', 'Pack EV'], ascending=[False, False])

    latest_packs_EV = latest_packs_EV.rename(columns= {'name' : 'Name', 'setCode': 'Set Code', 'boosterName': 'Booster Name', 'releaseDate' : 'Release Date', 'Pack EV' : 'Pack EV ($)'})

    latest_packs_EV['Std Dev'] = np.sqrt(latest_packs_EV['Pack Var'])

    latest_packs_EV = latest_packs_EV[['Name', 'Set Code', 'Booster Name', 'Release Date', 'Pack EV ($)', 'Std Dev']]

    latest_packs_EV = latest_packs_EV.merge(pack_prices, on=['Set Code', 'Booster Name'], how='left')

    latest_packs_EV['Price Spread'] = ((latest_packs_EV['Pack EV ($)'] - latest_packs_EV['Pack Price ($)']) / latest_packs_EV['Pack Price ($)']) * 100

    #Analyze specific sets


    latest_packs_EV = latest_packs_EV[['Name', 'Set Code', 'Booster Name', 'Pack EV ($)', 'Pack Price ($)', 'Price Spread', 'Std Dev']]



    return latest_packs_EV.reset_index(drop=True)

# Format Table 
def return_EV_table_styled(latest_packs_EV):
    # Style the DataFrame
    latest_packs_EV_styled = latest_packs_EV.style \
        .background_gradient(subset=["Price Spread"], cmap="RdYlGn", 
                            vmin=latest_packs_EV["Price Spread"].min(), 
                            vmax=latest_packs_EV["Price Spread"].max()) \
        .background_gradient(subset=["Std Dev"], cmap="RdYlGn_r") \
        .format({"Pack EV ($)": "{:.2f}",
                "Pack Price ($)": "{:.2f}",
                "Price Spread": "{:.2f}%",   # add percent sign
                "Std Dev": "{:.2f}"})  # Round only numeric columns

    return latest_packs_EV_styled






"""
What should the price of a card be? Card prices based on box price

Simulate Random Pack Pulls
Uses distribution for booster Index and Card Weights, per Sheet

"""

# ----- Preprocessing for simulation -----

def preprocess_for_sim(setBoosterSheetCards, 
                       setBoosterSheets, 
                       setBoosterContents, 
                       setBoosterContentWeights, 
                       prices, 
                       cards, 
                       threshold):
    
    merged_cardprices = get_merged_cardprices(threshold,
            setBoosterSheetCards,
            setBoosterSheets,
            prices, 
            cards)

    merged_cardprices_marketsim = merged_cardprices[['cardName', 'cardUuid', 'cardFinish', 'boosterName', 'setCode', 'sheetName', 'cardWeight', 'totalSheetWeight', 'price']]

    # --- Precompute booster probabilities ---
    booster_subsets = {}
    for (setCode, boosterName), group in setBoosterContentWeights.groupby(['setCode', 'boosterName']):
        booster_subsets[(setCode, boosterName)] = (
            group['boosterIndex'].values,
            (group['boosterWeight'] / group['boosterWeight'].sum()).values
        )

    # --- Precompute sheets per booster index ---
    sheets_subsets = {}
    for (setCode, boosterName, boosterIndex), group in setBoosterContents.groupby(['setCode', 'boosterName', 'boosterIndex']):
        sheets_subsets[(setCode, boosterName, boosterIndex)] = group[['sheetName', 'sheetPicks']].values  # array of [sheetName, sheetPicks]

    # --- Precompute cards per sheet ---
    cards_by_sheet = {}
    for (setCode, boosterName, sheetName), group in merged_cardprices_marketsim.groupby(['setCode', 'boosterName', 'sheetName']):
        prices = group['price'].values
        weights = (group['cardWeight'] / group['totalSheetWeight']).values
        cards_by_sheet[(setCode, boosterName, sheetName)] = (prices, weights, group.to_dict(orient='records'))

    
    return booster_subsets, sheets_subsets, cards_by_sheet
    

# --- Simulate Opening a random booster pack ---
def crack_pack(booster_subsets, sheets_subsets, cards_by_sheet, set_input, booster_input):
    # 1. Sample booster index
    booster_indices, booster_weights = booster_subsets[(set_input, booster_input)]

    booster_index = np.random.choice(booster_indices, p=booster_weights) #Picks a booster variant randomly

    # 2. Get sheets for this booster
    sheets = sheets_subsets[(set_input, booster_input, booster_index)]

    # 3. Sample cards
    pack = []
    for sheet_name, sheet_picks in sheets:

        prices, weights, records = cards_by_sheet[(set_input, booster_input, sheet_name)]

        chosen_indices = np.random.choice(len(prices), size=int(sheet_picks), p=weights) #Picks a card randomly
        # Append selected records
        pack.extend([records[i] for i in chosen_indices])

    return pack


def sim_market(n_packs, pack_price, set_input, booster_input, booster_subsets, sheets_subsets, cards_by_sheet):
    #Calcs used in simulation

    total_market_value = n_packs * pack_price

    market = [] #Used to store sim results


    for i in range(n_packs):
        result = crack_pack(booster_subsets, sheets_subsets, cards_by_sheet,set_input, booster_input)
        market.append(result)

    # Step 1: Flatten the outer list
    market_df = [item for sublist in market for item in sublist]

    # Step 2: Convert to DataFrame
    market_df = pd.DataFrame(market_df)

    market_df['count'] = market_df.groupby(
        ['cardName', 'cardUuid', 'cardFinish']
    )['cardName'].transform('count')

    market_df['inv count'] = 1 / market_df['count']

    # Remove duplicates first (keeping only unique rows)
    unique_cards = market_df.drop_duplicates(subset=['cardUuid', 'cardFinish']).copy()

    # Normalize inverse counts
    unique_cards['normalized sim weight'] = unique_cards['inv count'] / unique_cards['inv count'].sum()

    unique_cards['sim price'] = total_market_value * unique_cards['normalized sim weight'] / unique_cards['count']

    unique_cards = unique_cards[['cardName', 'cardUuid', 'setCode', 'cardFinish', 'sheetName', 'count', 'normalized sim weight', 'price', 'sim price']]

    return unique_cards



"""
Simulation CHART

Repeated Simulations + threshold
cut card prices that are less than the threshold sale price, keep the ones that are above

"""


#Inputs 

sim_size = 100 #Packs opened per sim

samples = 100 #Number of simulations run


set_input = "OTJ"
booster_input="play"

def run_simulation(latest_packs_EV, set_input, booster_input, sim_size, samples, threshold, booster_subsets, sheets_subsets, cards_by_sheet):
    pack_price = latest_packs_EV[
        (latest_packs_EV['Set Code'] == set_input) &
        (latest_packs_EV['Booster Name'] == booster_input)
    ]['Pack Price ($)'].iloc[0]

    #Function to run 

    avg_pack_value = []

    pack_value_at_sim_EV = [] 

    for n in range(samples): # n number of packs opened, per sim

        result = sim_market(sim_size, pack_price, set_input, booster_input, booster_subsets, sheets_subsets, cards_by_sheet)

        
        #Only include cards that are worth more than threshold
        result['card contribution to sale'] = np.where(
            result['price'] <= threshold, 0,
            (result['price'] * result['count'])
        )

        #Only include cards to sim EV pack prices if more than threshold
        result['card contribution at EV prices'] = np.where(
            result['sim price'] <= threshold, 0,
            (result['sim price'] * result['count'])

        )

        avg_pack_value.append(result['card contribution to sale'].sum()/ sim_size)

        pack_value_at_sim_EV.append(result['card contribution at EV prices'].sum()/ sim_size)

    return pack_price, avg_pack_value, pack_value_at_sim_EV

def plot_simulation(avg_pack_value, pack_value_at_sim_EV, pack_price, sim_size, set_input, booster_input, threshold):
    fig, ax = plt.subplots(figsize=(10,6))

    pct_success = 100 * np.sum(np.array(avg_pack_value) > pack_price) / len(avg_pack_value)
    ax.plot([], [], ' ', label=f'Odds of Success = {pct_success:.1f}%')

    x = list(range(1, len(avg_pack_value)+1))  # x-axis for plotting

    ax.plot(x, sorted(avg_pack_value), color='red', linestyle='-', 
             label=f'Avg Pack Value (Sale Price) = ${np.mean(avg_pack_value):.2f}')

    ax.axhline(y=pack_price, color='black', linestyle='-', 
                label=f'Pack Price = ${pack_price:.2f}')

    expected_spread = np.mean(avg_pack_value) - pack_price
    ax.plot([], [], ' ', label=f'Expected Spread = ${expected_spread:.2f}')

    ax.plot(x, sorted(pack_value_at_sim_EV), color='green', linestyle='--', 
             label=f'Avg Pack Value (Sim EV Price) = ${np.mean(pack_value_at_sim_EV):.2f}')

    ax.set_title(f'{pct_success:.1f}% chance of Profit Opening {sim_size} {set_input} {booster_input} Packs, only selling ${threshold:.2f}+ cards')
    ax.set_xlabel('Simulation Count (Sorted within each series)')
    ax.set_ylabel('Average Pack Value ($)')
    ax.grid(True)
    ax.legend()

    return fig


def sim_card_prices (n_packs, pack_price, set_input, booster_input, booster_subsets, sheets_subsets, cards_by_sheet):
    table = sim_market(n_packs, pack_price, set_input, booster_input, booster_subsets, sheets_subsets, cards_by_sheet)

    table = table[['cardName', 'price', 'sim price']]

    table['Price Spread'] = table['price'] - table['sim price']

    table = table.rename(columns= {'cardName' : 'Name', 'price': 'Card Price ($)', 'sim price': 'Sim Price ($)'})

    table = table.sort_values(by='Price Spread', ascending= False)

    table = table.style \
        .background_gradient(subset=["Price Spread"], cmap="RdYlGn") \
        .format({
            "Card Price ($)": "{:.2f}",
            "Sim Price ($)": "{:.2f}",
            "Price Spread": "{:.2f}"})
                             
    return table
