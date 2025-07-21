import pandas as pd
import numpy as np
from scipy.stats import norm
import streamlit as st 
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

st.title('Black-Scholes Model')


# Inputs on sidebar
Spot_price = st.sidebar.number_input(
    "Spot Price", 
    min_value=0.00, 
    max_value=500.00, 
    value=100.00, 
    step=1.00, 
    format="%.2f"
)

Strike = st.sidebar.number_input(
    "Strike Price", 
    min_value=0.00, 
    max_value=500.00, 
    value=101.10, 
    step=1.00, 
    format="%.2f"
)

Volatility = st.sidebar.number_input(
    "Volatility (%)",
    min_value=0.00,
    max_value=100.00,
    value=20.00,
    step=1.00,
    format="%.2f"
)

R = st.sidebar.number_input(
    "Risk-free rate (%)",
    min_value=0.00,
    max_value=5.00,
    value=2.00,
    step=0.10,
    format="%.2f"
)

Time = st.sidebar.number_input(
    "Time to maturity (years)",
    min_value=0.00,
    max_value=10.00,
    value=1.50,
    step=0.25,
    format="%.2f"
)

st.sidebar.markdown("**Developed by Will Drake**")
st.sidebar.markdown("[GitHub](https://github.com/WillD55)")

volatility = Volatility / 100
r = R/100

# Call and Put Boxes with Price
d_1 = (np.log(Spot_price / Strike) + (r + 0.5 * volatility ** 2) * Time) / (volatility * np.sqrt(Time))
d_2 = d_1 - (volatility * np.sqrt(Time))

Call = Spot_price * norm.cdf(d_1) - Strike * np.exp(-r * Time) * norm.cdf(d_2)
Put = Strike * np.exp(-r * Time) * norm.cdf(-d_2) - Spot_price * norm.cdf(-d_1)

col1, col2 = st.columns(2)

with col1:
    st.success(f"Call Price: ${Call:.2f}")

with col2:
    st.error(f"Put Price: ${Put:.2f}")


# Heatmaps of Call and Put Options
st.title('HeatMaps')

Strike_price = np.linspace(0.8 * Strike, 1.2 * Strike, 8)

Vol = np.linspace(0.8 * Volatility, 1.2 * Volatility, 8) / 100

call_prices = np.zeros((len(Vol), len(Strike_price)))

put_prices = np.zeros((len(Vol), len(Strike_price)))

for i, sigma in enumerate(Vol):
    for y, X in enumerate(Strike_price):
        d1 = (np.log(Spot_price / X) + (r + 0.5 * sigma ** 2) * Time) / (sigma * np.sqrt(Time))
        d2 = d1 - sigma * np.sqrt(Time)
        C = Spot_price * norm.cdf(d1) - X * np.exp(r * Time) * norm.cdf(d2)
        call_prices[i, y] = C

for j, sigma in enumerate(Vol):
    for z, X in enumerate(Strike_price):
        d1 = (np.log(Spot_price / X) + (r + 0.5 * sigma ** 2) * Time) / (sigma * np.sqrt(Time))
        d2 = d1 - sigma * np.sqrt(Time)
        P = X * np.exp(-r * Time) * norm.cdf(-d2) - Spot_price * norm.cdf(-d1)
        put_prices[j, z] = P

fig, ax = plt.subplots(figsize=(20, 15), dpi=200)

fig1, ax1 = plt.subplots(figsize=(20, 15), dpi=200)

vol_ax = [f"{v*100:.0f}%" for v in Vol]
strike_ax = [f"${round(s, 2)}" for s in Strike_price]

col3, col4 = st.columns(2)

with col3:
    ax = sns.heatmap(call_prices, annot=True, cmap="Greens", annot_kws={"size": 35, "weight": "bold"}, cbar=False, fmt=".2f", cbar_kws={'label': 'Value of C'}, xticklabels=strike_ax,yticklabels=vol_ax, ax=ax)

    ax.set_xlabel("Strike Price", fontsize = 25, fontweight = "bold")
    ax.set_ylabel("Volatility", fontsize = 25, fontweight = "bold")

    ax.tick_params(axis="x", labelsize=25) 
    ax.tick_params(axis="y", labelsize=25)

    st.markdown (
    f"<h1 style='text-align: left; font-size:20px;'>Call Option:</h1>",
    unsafe_allow_html=True)

    st.pyplot(fig)

with col4:
    ax1 = sns.heatmap(put_prices, annot=True, cmap="Reds", annot_kws={"size": 35, "weight": "bold"},  cbar=False, fmt=".2f", cbar_kws={'label': 'Value of P'}, xticklabels=strike_ax,yticklabels=vol_ax, ax=ax1)

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


# The Greeks
st.title('The Greeks')

# Delta
D_Call = norm.cdf(d_1)
D_Put = -norm.cdf(-d_1)

# Gamma
G_Call_Put = norm.pdf(d_1) / (Spot_price * volatility * np.sqrt(Time))

# Vega
V_Call_Put = (norm.pdf(d_1) * Spot_price * np.sqrt(Time)) / 100 

# Theta
T_Call = (-(Spot_price * volatility * norm.pdf(d_1)) / (2 * np.sqrt(Time)) - (r * Strike * np.exp(-r * Time) * norm.cdf(d_2))) / 365
T_Put = (-(Spot_price * volatility * norm.pdf(d_1)) / (2 * np.sqrt(Time)) + (r * Strike * np.exp(-r * Time) * norm.cdf(-d_2))) / 365

# Rho
R_Call = (Strike * Time * np.exp(-r * Time) * norm.cdf(d_2)) /100
R_Put = (-Strike * Time * np.exp(-r * Time) * norm.cdf(-d_2)) / 100

Greeks = {
    "" : ["Call Option", "Put Option"],
    "Delta" : [D_Call, D_Put],
    "Gamma" : [G_Call_Put, G_Call_Put],
    "Vega" : [V_Call_Put, V_Call_Put],
    "Theta" : [T_Call, T_Put],
    "Rho" : [R_Call, R_Put]
}

greeks_df = pd.DataFrame(Greeks)
st.table(greeks_df)

st.markdown (
    f"<h1 style='text-align: left; font-size:15px;'>Delta: Measure of sensitivity of an option's price to changes in the asset's price. If the price of the asset increases by $1, the price of the option will change by the Delta amount.</h1>",
    unsafe_allow_html=True)

st.markdown (
    f"<h1 style='text-align: left; font-size:15px;'>Gamma: Measure of Delta's change to changes in the price of the asset. If the price of the asset increases by $1, the option's Delta will change by Gamma amount.</h1>",
    unsafe_allow_html=True)

st.markdown (
    f"<h1 style='text-align: left; font-size:15px;'>Vega: Measure of sensitivity of an option's price to change in the volatility. If the volatility of the asset increases by 1%, the option price will change by the Vega amount.</h1>",
    unsafe_allow_html=True)

st.markdown (
    f"<h1 style='text-align: left; font-size:15px;'>Theta: Measure of sensitivity of an option's price to change in the option's time to maturity. If the option's time to maturity decreases by one day, the option's price will change by Theta amount. </h1>",
    unsafe_allow_html=True)

st.markdown (
    f"<h1 style='text-align: left; font-size:15px;'>Rho: Measure of sensitivity of an option's price to change in interest rate. If a benchmark interest rate increases by 1%, the option price will change by the Rho amount.</h1>",
    unsafe_allow_html=True)
