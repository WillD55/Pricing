import pandas as pd
import numpy as np
from scipy.stats import norm
import streamlit as st 
import matplotlib.pyplot as plt
import seaborn as sns

st.title('Future Price Calculator (Copper)')

Spot_price = st.sidebar.slider("Spot Price: ", 3.50, 6.00, 4.00, step = 0.01)

R = st.sidebar.slider("Risk-free rate (%): ", 2.0, 6.0, 2.0, step = 0.1)

Storage = st.sidebar.slider("Storage Cost (%): ", 0.00, 3.00, 1.00, step = 0.01)

Time = st.sidebar.slider("Time to maturity (year): ", 0.0, 3.00, 1.0, step = 0.5)

F_t = Spot_price * np.exp((R/100 + Storage/100) * Time)
F_t = round(F_t, 2)

st.markdown (
    f"<h1 style='text-align: left; font-size:20px;'>The fair price of the copper futures contract is ${F_t} per pound</h1>",
    unsafe_allow_html=True
)

Strike_price = np.linspace(0.8 * F_t, 1.2 * F_t, 8)

Vol = np.linspace(10, 45, 8) / 100

option_prices = np.zeros((len(Vol), len(Strike_price)))

for i, sigma in enumerate(Vol):
    for y, X in enumerate(Strike_price):
        d1 = (np.log(Spot_price / X) + (R/100 + 0.5 * sigma ** 2) * Time) / (sigma * np.sqrt(Time))
        d2 = d1 - sigma * np.sqrt(Time)
        C = Spot_price * norm.cdf(d1) - X * np.exp(-R/100 * Time) * norm.cdf(d2)
        option_prices[i, y] = C

fig, ax = plt.subplots(figsize=(8, 6))

vol_ax = [f"{v*100:.0f}%" for v in Vol]
strike_ax = [f"${round(s, 2)}" for s in Strike_price]

ax = sns.heatmap(option_prices, annot=True, cmap="viridis", cbar=False, fmt=".2f", cbar_kws={'label': 'Value of C'}, xticklabels=strike_ax,yticklabels=vol_ax)

ax.set_xlabel("Strike Price")
ax.set_ylabel("Volatility")


st.pyplot(fig)
