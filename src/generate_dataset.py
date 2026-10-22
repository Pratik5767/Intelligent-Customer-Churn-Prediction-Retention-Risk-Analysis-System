# %%
# Imports
import numpy as np
import pandas as pd

# %%
rng = np.random.default_rng(42)
N = 10000

# %%
# --- Demographics ---
age = np.clip(rng.normal(38, 12, N).round(), 18, 75).astype(int)

gender = rng.choice(["Male", "Female", "Other"], N, p=[0.49, 0.48, 0.03])

location = rng.choice(
    ["Mumbai", "Delhi", "Bengaluru", "Pune", "Hyderabad", "Chennai", "Kolkata", "Ahmedabad"],
    N, 
    p=[0.16, 0.16, 0.16, 0.12, 0.12, 0.10, 0.10, 0.08]
)

# %%
print(age)
print(gender)
print(location)

# %%
# --- Subscription ---
subscription = rng.choice(["Basic", "Standard", "Premium"], N, p=[0.40, 0.38, 0.22])

contract = rng.choice(["Month-to-Month", "One Year", "Two Year"], N, p=[0.55, 0.28, 0.17])

payment = rng.choice(
    ["Credit Card", "Debit Card", "UPI", "Net Banking", "Wallet"], 
    N,
    p=[0.25, 0.20, 0.35, 0.12, 0.08]
)

tenure = np.clip(rng.gamma(2.2, 14, N).round(), 1, 72).astype(int)

base_price = pd.Series(subscription).map({"Basic": 299, "Standard": 599, "Premium": 999}).values

monthly = np.round(base_price * rng.uniform(0.9, 1.15, N), 2)

total = np.round(monthly * tenure * rng.uniform(0.92, 1.0, N), 2)

# %%
print(subscription)
print(contract)
print(payment)
print(tenure)
print(base_price)
print(monthly)
print(total)

# %%
# --- Behaviour ---
login_freq = np.clip(rng.poisson(14, N) - (contract == "Month-to-Month") * 2, 0, 40)  # logins / month

last_login = np.clip(rng.exponential(9, N).round(), 0, 120).astype(int)

usage = np.clip(rng.gamma(3, 6, N) * (login_freq / 14 + 0.3), 0.5, 150).round(1)

tickets = rng.poisson(1.5, N)

complaints = np.clip(rng.poisson(0.4 + 0.25 * tickets), 0, 8)

discount = rng.choice([0, 5, 10, 15, 20, 25], N, p=[0.45, 0.15, 0.15, 0.1, 0.1, 0.05])

# %%
print(login_freq)
print(last_login)
print(usage)
print(tickets)
print(complaints)
print(discount)

# %%
# --- Satisfaction (1-5), driven by experience ---
sat_raw = 4.2 - 0.35 * complaints - 0.12 * tickets - 0.01 * last_login + rng.normal(0, 0.6, N)

satisfaction = np.clip(sat_raw.round(), 1, 5).astype(int)

print(satisfaction)

# %%
# --- Churn logic (hidden relationship the model should discover) ---
z = (-1.4
     + 1.0 * (contract == "Month-to-Month")
     - 0.9 * (contract == "Two Year")
     - 0.03 * tenure
     + 0.035 * last_login
     - 0.045 * login_freq
     + 0.35 * complaints
     + 0.12 * tickets
     - 0.55 * (satisfaction - 3)
     + 0.0008 * (monthly - 600)
     - 0.02 * discount
     + 0.25 * (payment == "Wallet")
     + 0.15 * (subscription == "Premium") * (satisfaction <= 2)
     + rng.normal(0, 0.5, N))

prob = 1 / (1 + np.exp(-z))
churn = (rng.uniform(0, 1, N) < prob).astype(int)

# %%
print(churn)

# %%
df = pd.DataFrame({
    "Customer_ID": [f"CUST{100000 + i}" for i in range(N)],
    "Age": age, 
    "Gender": gender, 
    "Location": location,
    "Tenure_Months": tenure, 
    "Monthly_Charges": monthly, 
    "Total_Charges": total,
    "Subscription_Type": subscription, 
    "Payment_Method": payment,
    "Login_Frequency": login_freq, 
    "Support_Tickets": tickets, 
    "Complaints": complaints,
    "Discount_Received": discount, 
    "Contract_Type": contract,
    "Last_Login_Days": last_login, 
    "Usage_Hours": usage,
    "Customer_Satisfaction": satisfaction, 
    "Churn": churn,
})

# %%
df.head()

# %%
def pick(frac):
    """Random row positions covering `frac` of the dataset (no repeats)."""
    return rng.choice(N, int(N * frac), replace=False)

# %%
missing_plan = {
    "Gender": 0.03, 
    "Location": 0.03, 
    "Payment_Method": 0.02,
    "Contract_Type": 0.04, 
    "Subscription_Type": 0.03,
    "Tenure_Months": 0.03, 
    "Monthly_Charges": 0.02,
    "Login_Frequency": 0.03, 
    "Last_Login_Days": 0.02,
    "Support_Tickets": 0.03, 
    "Discount_Received": 0.02,
}

# %%
# --- adding missing values in a few columns ---
for col, frac in missing_plan.items():
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].astype(float)
    df.loc[pick(frac), col] = np.nan

# %%
# --- placeholders that secretly mean "missing" ---
for col, token in [("Gender", "Unknown"), ("Location", "?"), ("Payment_Method", "N/A")]:
    df.loc[pick(0.01), col] = token
df.loc[pick(0.01), "Last_Login_Days"] = -1

# %%
# --- Realistic messiness: 10% missing values in a few columns ---
for col in ["Age", "Total_Charges", "Customer_Satisfaction", "Usage_Hours"]:
    idx = rng.choice(N, int(N * 0.1), replace=False)
    df[col] = df[col].astype(float)
    df.loc[idx, col] = np.nan

# %%
# --- Outliers ---
def inject(col, values, n):
    idx = rng.choice(N, n, replace=False)
    df.loc[idx, col] = rng.choice(values, n)

inject("Age", [-5, 0, 120, 150, 999], 15)
inject("Tenure_Months", [0, -3, 240, 500], 12)
inject("Monthly_Charges", [-599, 0, 9999, 59900], 12)
inject("Usage_Hours", [-10, 0, 500, 2400], 12)
inject("Customer_Satisfaction", [0, 6, 7, 10], 20)      # valid range is 1-5
inject("Support_Tickets", [-2, 40, 99], 8)
inject("Discount_Received", [-10, 60, 100, 150], 10)

# %%
# Total_Charges far below Monthly_Charges x Tenure --
idx = pick(0.01)
df.loc[idx, "Total_Charges"] = (df.loc[idx, "Monthly_Charges"] * 0.1).round(2)

# %%
# ------ case, spaces, alternate spellings ------
def scramble(col, frac, alt=None):
    for i in pick(frac):
        v = df.at[i, col]
        if not isinstance(v, str):
            continue
        choice = rng.integers(0, 4)
        if choice == 0:
            df.at[i, col] = v.lower()
        elif choice == 1:
            df.at[i, col] = v.upper()
        elif choice == 2:
            df.at[i, col] = f" {v} "
        elif alt and v in alt:
            df.at[i, col] = alt[v]
 
scramble("Gender", 0.08, {"Male": "M", "Female": "F"})
scramble("Location", 0.08, {"Bengaluru": "Bangalore", "Mumbai": "Bombay", "Kolkata": "Calcutta"})
scramble(
    "Payment_Method", 0.08, {"UPI": "Upi", "Credit Card": "CreditCard", "Debit Card": "Debit", "Net Banking": "NetBanking"}
)
scramble("Contract_Type", 0.08, {"Month-to-Month": "Month to Month", "One Year": "1 Year", "Two Year": "2 Year"})
scramble("Subscription_Type", 0.06)

# %%
# -------- numbers stored as text --------
df["Total_Charges"] = df["Total_Charges"].astype(object)

df.loc[pick(0.03), "Total_Charges"] = " "                # blank space, not NaN

for i in pick(0.02):
    v = df.at[i, "Total_Charges"]
    if not isinstance(v, str) and pd.notna(v):
        df.at[i, "Total_Charges"] = f"₹{v:,.2f}"      # e.g. "₹12,345.60"
 
df["Age"] = df["Age"].astype(object)
words = ["thirty-five", "forty", "twenty five", "N/A"]

for i in pick(0.004):
    df.at[i, "Age"] = rng.choice(words)

# %%
# ------- exact copies + same Customer_ID with conflicting values -------
exact = df.sample(60, random_state=7)
conflict = df.sample(10, random_state=11).copy()

conflict["Support_Tickets"] = pd.to_numeric(conflict["Support_Tickets"], errors="coerce").fillna(0) + 1
df = pd.concat([df, exact, conflict], ignore_index=True)
 
# ------- shuffle rows so duplicates are not sitting next to each other -------
df = df.sample(frac=1, random_state=7).reset_index(drop=True)

# %%
df.to_csv("data/raw/customer_churn_dataset.csv", index=False)
print(df.shape)

# %%
print("Churn rate:", round(df["Churn"].mean() * 100, 1), "%")

# %%
print(df.isna().sum())

# %%
print(df.duplicated().sum())

# %%
print(df['Customer_ID'].duplicated(keep=False).sum())

# %%
df.head()