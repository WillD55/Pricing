import pandas as pd
import numpy as np
from scipy.stats import norm
import streamlit as st 
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

st.title('Black Scholes Calculator')

Spot_price = st.sidebar.number_input(
    "Spot Price", 
    min_value=0.00, 
    max_value=500.00, 
    value=4.00, 
    step=0.01, 
    format="%.2f"
)

Strike = st.sidebar.number_input(
    "Strike Price", 
    min_value=0.00, 
    max_value=500.00, 
    value=4.00, 
    step=0.01, 
    format="%.2f"
)

Volatility = st.sidebar.slider("Volatility (%): ", 0.0, 50.0, 20.0, step = 5.0)

R = st.sidebar.slider("Risk-free rate (%): ", 2.0, 6.0, 2.0, step = 0.1)

#Storage = st.sidebar.slider("Storage Cost (%): ", 0.00, 3.00, 1.00, step = 0.01)

Time = st.sidebar.slider("Time to maturity (year): ", 0.0, 3.00, 1.0, step = 0.5)

d1 = (np.log(Spot_price / Strike) + (R/100 + 0.5 * Volatility/100 ** 2) * Time) / (Volatility * np.sqrt(Time))
d2 = d1 - Volatility/100 * np.sqrt(Time)

Call = Spot_price * norm.cdf(d1) - Strike * np.exp(-R/100 * Time) * norm.cdf(d2)
Put = Strike * np.exp(-R/100 * Time) * norm.cdf(-d2) - Spot_price * norm.cdf(-d1)

col1, col2 = st.columns(2)

with col1:
    st.success(f"Call Price: ${Call:.2f}")

with col2:
    st.error(f"Put Price: ${Put:.2f}")

#F_t = Spot_price * np.exp((R/100 + Storage/100) * Time)
#F_t = round(F_t, 2)

# st.markdown (
#     f"<h1 style='text-align: left; font-size:20px;'>The fair price of the copper futures contract is ${F_t} per pound</h1>",
#     unsafe_allow_html=True
# )

Strike_price = np.linspace(0.8 * Strike, 1.2 * Strike, 8)

Vol = np.linspace(0.8 * Volatility, 1.2 * Volatility, 8) / 100

call_prices = np.zeros((len(Vol), len(Strike_price)))

put_prices = np.zeros((len(Vol), len(Strike_price)))

for i, sigma in enumerate(Vol):
    for y, X in enumerate(Strike_price):
        d1 = (np.log(Spot_price / X) + (R/100 + 0.5 * sigma ** 2) * Time) / (sigma * np.sqrt(Time))
        d2 = d1 - sigma * np.sqrt(Time)
        C = Spot_price * norm.cdf(d1) - X * np.exp(-R/100 * Time) * norm.cdf(d2)
        call_prices[i, y] = C

for j, sigma in enumerate(Vol):
    for z, X in enumerate(Strike_price):
        d1 = (np.log(Spot_price / X) + (R/100 + 0.5 * sigma ** 2) * Time) / (sigma * np.sqrt(Time))
        d2 = d1 - sigma * np.sqrt(Time)
        P = X * np.exp(-R/100 * Time) * norm.cdf(-d2) - Spot_price * norm.cdf(-d1)
        put_prices[j, z] = P

fig, ax = plt.subplots(figsize=(20, 15), dpi=200)

fig1, ax1 = plt.subplots(figsize=(20, 15), dpi=200)

vol_ax = [f"{v*100:.0f}%" for v in Vol]
strike_ax = [f"${round(s, 2)}" for s in Strike_price]

col3, col4 = st.columns(2)

with col3:
    ax = sns.heatmap(call_prices, annot=True, cmap="viridis", annot_kws={"size": 35, "weight": "bold"}, cbar=False, fmt=".2f", cbar_kws={'label': 'Value of C'}, xticklabels=strike_ax,yticklabels=vol_ax, ax=ax)

    ax.set_xlabel("Strike Price", fontsize = 25, fontweight = "bold")
    ax.set_ylabel("Volatility", fontsize = 25, fontweight = "bold")

    ax.tick_params(axis="x", labelsize=25) 
    ax.tick_params(axis="y", labelsize=25)

    st.markdown (
    f"<h1 style='text-align: left; font-size:20px;'>Call Option:</h1>",
    unsafe_allow_html=True)

    st.pyplot(fig)

with col4:
    ax1 = sns.heatmap(put_prices, annot=True, cmap="viridis", annot_kws={"size": 35, "weight": "bold"},  cbar=False, fmt=".2f", cbar_kws={'label': 'Value of P'}, xticklabels=strike_ax,yticklabels=vol_ax, ax=ax1)

    ax1.set_xlabel("Strike Price")
    ax1.set_ylabel("Volatility")

    ax1.set_xlabel("Strike Price", fontsize = 25, fontweight = "bold")
    ax1.set_ylabel("Volatility", fontsize = 25, fontweight = "bold")

    ax1.tick_params(axis="x", labelsize=25)
    ax1.tick_params(axis="y", labelsize=25)

    st.markdown (
    f"<h1 style='text-align: left; font-size:20px;'>Put Option:</h1>",
    unsafe_allow_html=True)

    st.pyplot(fig1)
