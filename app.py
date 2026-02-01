import streamlit as st
from datetime import date
from collections import defaultdict
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import backend
import os
ASSETS_DIR = "assets"  # folder where hero_finance.png is stored


backend.create_tables()


# ================= PAGE CONFIG =================
st.set_page_config(page_title="MoneyLeak AI", layout="centered")

# ================= GLOBAL CSS =================
st.markdown("""
<style>
/* Background */
.stApp {
    background: linear-gradient(135deg, #e6f0ff, #f9fbff);
    font-family: 'Segoe UI', sans-serif;
}

/* Glass Cards */
.card {
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.12);
    margin-bottom: 25px;
}

/* Section headers */
.section {
    margin-top: 30px;
    margin-bottom: 10px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #4facfe, #00f2fe);
    color: white;
    border-radius: 12px;
    height: 3em;
    border: none;
    font-weight: 600;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.85);
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.1);
}

/* Hide Streamlit clutter */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ================= SESSION STATE =================
if "page" not in st.session_state:
    st.session_state.page = "intro"

if "user" not in st.session_state:
    st.session_state.user = None

if "monthly_budget" not in st.session_state:
    st.session_state.monthly_budget = 0

if "category_budget" not in st.session_state:
    st.session_state.category_budget = {
        "Food": 0,
        "Rent/Bills": 0,
        "Entertainment": 0,
        "Savings": 0
    }

if "expenses" not in st.session_state:
    st.session_state.expenses = []

# ================= NAVIGATION =================
def navigate(page):
    st.session_state.page = page

def ai_badge(text, color):
    st.markdown(f"""
    <div style="
        display:inline-block;
        padding:6px 16px;
        border-radius:25px;
        background:{color};
        color:white;
        font-weight:600;
        margin-bottom:10px;">
        🤖 {text}
    </div>
    """, unsafe_allow_html=True)


# =================================================
# PAGE 1 : INTRO
# =================================================
import os  # Needed for image path

ASSETS_DIR = "assets"  # make sure your folder is named "assets"

if st.session_state.page == "intro":
    
    # ================= HERO SECTION =================
    hero_left, hero_right = st.columns([1.2, 1])

    with hero_left:
        st.markdown(f"""
        <div style="
            padding:60px 40px;
            border-radius:30px;
            background: linear-gradient(145deg, #dbe9ff, #f9fbff);
            box-shadow: 0 15px 40px rgba(0,0,0,0.1);
            text-align:left;
        ">
            <h1 style="font-size:56px; font-weight:900; margin-bottom:20px; color:#0B3D91;">
                💸 MoneyLeak AI
            </h1>
            <p style="font-size:22px; color:#333; font-weight:500; margin-bottom:20px;">
                AI that understands how you spend — not just where
            </p>
            <p style="font-size:18px; color:#555; margin-bottom:25px;">
                Traditional budget apps track numbers.<br>
                <b>MoneyLeak AI</b> analyzes financial behavior, predicts overspending, 
                and guides smarter decisions using AI & ML.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.button(
            "🚀 Start Smart Budgeting",
            on_click=navigate,
            args=("login",),
            key="hero_start_btn",
            help="Click to login or register and start tracking your finances!"
        )

with hero_right:
    image_path = os.path.join(ASSETS_DIR, "hero_finance.png")

    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    else:
        st.warning("⚠ Hero image not found (assets/hero_finance.png)")



    st.markdown("<br><br>", unsafe_allow_html=True)

    # ================= FEATURES GRID =================
    st.markdown("## ✨ What Makes MoneyLeak AI Different")
    f1, f2, f3 = st.columns(3)

    features = [
        ("🧠 Behavioral AI", "Understands your spending personality and habits using machine learning."),
        ("📊 Risk Intelligence", "Detects anomalies, predicts overspending, and warns before damage happens."),
        ("🧮 What-If Simulation", "Test future decisions safely before spending real money.")
    ]

    for col, (title, desc) in zip([f1, f2, f3], features):
        with col:
            st.markdown(f"""
            <div class="card" style="
                transition: transform 0.3s ease;
            " onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
                <h3>{title}</h3>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ================= HOW IT WORKS =================
    st.markdown("## ⚙️ How It Works")
    step_cols = st.columns(4)
    steps = [
        ("1️⃣", "Set Budget"),
        ("2️⃣", "Add Expenses"),
        ("3️⃣", "AI Analysis"),
        ("4️⃣", "Smart Decisions")
    ]

    for col, (icon, text) in zip(step_cols, steps):
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center; padding:20px; transition: transform 0.3s ease;">
                <h2>{icon}</h2>
                <p style="font-weight:600;">{text}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ================= AI HIGHLIGHT =================
    st.markdown(f"""
    <div class="card" style="text-align:center; background: linear-gradient(90deg, #4facfe, #00f2fe); color:white;">
        <h2>🤖 Powered by Real AI</h2>
        <p style="font-size:17px; color:white; margin-bottom:5px;">
            Linear Regression • Trend Detection • Anomaly Detection • Behavior Scoring
        </p>
        <p style="font-size:15px; color:white;">
            Not rules. Not assumptions. <b>Actual machine learning.</b>
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ================= CTA BUTTON =================
    st.button(
        "💡 Enter MoneyLeak AI",
        on_click=navigate,
        args=("login",),
        key="footer_start_btn",
        help="Click to start tracking and optimizing your finances!"
    )


# if st.session_state.page == "intro":
#     st.title("💸 MoneyLeak AI")
#     st.subheader("Smart Financial Behavior Analyzer")

#     st.markdown("""
#     MoneyLeak AI is an intelligent expense monitoring system that goes beyond
#     simple expense tracking.

#     ### 🔍 What makes it different?
#     - Detects bad spending habits  
#     - Scores your financial discipline  
#     - Predicts future risk  
#     - Helps you simulate better decisions  

#     👉 Built for hackathons, real users & judges
#     """)

#     st.button("🚀 Get Started", on_click=navigate, args=("login",))

# =================================================
# PAGE 2 : LOGIN / REGISTER
# =================================================
elif st.session_state.page == "login":
    st.title("🔐 Secure Login")

    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)

        # ✅ Add unique keys
        username = st.text_input("Username", key="login_username")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login / Register", key="login_button"):
            if username and email and password:
                backend.save_user(username, email, password)

                st.session_state.user = {
                    "username": username,
                    "email": email
                }

                # 🔥 Load old expenses
                expenses = backend.get_expenses(email)
                st.session_state.expenses = [
                    {"category": e[0], "amount": e[1], "date": e[2]}
                    for e in expenses
                ]

                # 🔥 Load budget if exists
                budget = backend.get_budget(email)
                if budget:
                    st.session_state.monthly_budget = budget[0]
                    st.session_state.category_budget = {
                        "Food": budget[1],
                        "Rent/Bills": budget[2],
                        "Entertainment": budget[3],
                        "Savings": budget[4]
                    }

                navigate("budget")
            else:
                st.warning("All fields are required")

        st.markdown('</div>', unsafe_allow_html=True)


# =================================================
# PAGE 3 : BUDGET SETUP
# =================================================
elif st.session_state.page == "budget":
    st.title("📅 Monthly Budget Setup")

    total = st.number_input("Total Monthly Budget (₹)", min_value=0, key="budget_total")

    st.subheader("📊 Category-wise Budget Allocation")
    food = st.number_input("Food (₹)", min_value=0, key="budget_food")
    rent = st.number_input("Rent / Bills (₹)", min_value=0, key="budget_rent")
    entertainment = st.number_input("Entertainment (₹)", min_value=0, key="budget_entertainment")
    savings = st.number_input("Savings (₹)", min_value=0, key="budget_savings")

    if st.button("Save Budget", key="save_budget_button"):
        st.session_state.monthly_budget = total
        st.session_state.category_budget = {
            "Food": food,
            "Rent/Bills": rent,
            "Entertainment": entertainment,
            "Savings": savings
        }
        navigate("expense")



# =================================================
# PAGE 4 : ADD EXPENSE
# =================================================
elif st.session_state.page == "expense":
    st.title("➕ Add Daily Expenses")

    st.markdown('<div class="card">', unsafe_allow_html=True)

    exp_date = st.date_input("Date", value=date.today(), key="expense_date")

    st.subheader("Enter amount spent today")
    food = st.number_input("🍔 Food (₹)", min_value=0, key="expense_food")
    rent = st.number_input("🏠 Rent / Bills (₹)", min_value=0, key="expense_rent")
    entertainment = st.number_input("🎮 Entertainment (₹)", min_value=0, key="expense_entertainment")

    if st.button("Save Today's Expenses", key="save_expense_button"):
        email = st.session_state.user["email"]

        if food > 0:
            st.session_state.expenses.append({"category": "Food", "amount": food, "date": exp_date})
            backend.save_expense(email, "Food", food, exp_date)

        if rent > 0:
            st.session_state.expenses.append({"category": "Rent/Bills", "amount": rent, "date": exp_date})
            backend.save_expense(email, "Rent/Bills", rent, exp_date)

        if entertainment > 0:
            st.session_state.expenses.append({"category": "Entertainment", "amount": entertainment, "date": exp_date})
            backend.save_expense(email, "Entertainment", entertainment, exp_date)

        if food == rent == entertainment == 0:
            st.warning("Please enter at least one expense")
        else:
            st.success("✅ Daily expenses saved")

    st.markdown('</div>', unsafe_allow_html=True)
    st.button("📊 Go to Dashboard", on_click=navigate, args=("dashboard",), key="go_dashboard_button")


# =================================================
# PAGE 5 : DASHBOARD
# =================================================
elif st.session_state.page == "dashboard":
    st.title("📊 Financial Dashboard")

    total_spent = sum(e["amount"] for e in st.session_state.expenses)
    actual_savings = st.session_state.category_budget["Savings"] - total_spent * 0.1

    if total_spent > st.session_state.monthly_budget * 0.8:
        st.warning("🚨 ALERT: You are approaching your monthly budget limit!")

    col1, col2, col3 = st.columns(3)
    col1.metric("Monthly Budget", f"₹{st.session_state.monthly_budget}")
    col2.metric("Total Expenses", f"₹{total_spent}")
    col3.metric("Estimated Savings", f"₹{int(actual_savings)}")

    score = 100
    if total_spent > st.session_state.monthly_budget:
        score -= 30
    if actual_savings < st.session_state.category_budget["Savings"]:
        score -= 30
    score = max(score, 0)

    st.subheader("🧠 Financial Discipline Score")
    st.progress(score / 100)
    st.write(f"Score: {score} / 100")

    st.subheader("📉 Category Budget Utilization")
    if st.session_state.expenses:
        util_data = []
        for category, budget in st.session_state.category_budget.items():
            if category == "Savings":
                continue
            spent = sum(float(e["amount"]) for e in st.session_state.expenses if e["category"] == category)
            util = spent / budget if budget > 0 else 0
            util_data.append({"Category": category, "Spent": spent, "Budget": budget, "Util": util})

        df_util = pd.DataFrame(util_data)
        colors = ["green" if v <= 0.8 else "orange" if v <= 1 else "red" for v in df_util["Util"]]

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.barh(df_util["Category"], df_util["Budget"], color="lightgray", alpha=0.3)
        ax.barh(df_util["Category"], df_util["Spent"], color=colors, edgecolor="black")
        for i, (spent, budget) in enumerate(zip(df_util["Spent"], df_util["Budget"])):
            ax.text(spent + 10, i, f"₹{int(spent)}/{int(budget)}", va='center', fontweight='bold')
        ax.set_xlabel("Amount (₹)")
        ax.set_title("Category-wise Budget Utilization")
        st.pyplot(fig)
    else:
        st.info("Add expenses to see category utilization")

    st.subheader("⚠ Monthly Risk Assessment")
    if total_spent > st.session_state.monthly_budget * 0.9:
        st.error("🔴 Next Month Risk: HIGH")
    elif total_spent > st.session_state.monthly_budget * 0.75:
        st.warning("🟡 Next Month Risk: MEDIUM")
    else:
        st.success("🟢 Financial position stable")

    st.subheader("📊 Daily Spend vs Allowed Budget")
    if st.session_state.expenses:
        df = pd.DataFrame(st.session_state.expenses)
        daily_spend = df.groupby("date")["amount"].sum()
        daily_budget = st.session_state.monthly_budget / 30
        chart_df = pd.DataFrame({
            "Actual Spend": daily_spend,
            "Allowed Budget": [daily_budget] * len(daily_spend)
        })
        st.line_chart(chart_df)
    else:
        st.info("Add daily expenses to see budget comparison")

    st.subheader("🤖 ML-Based Weekly Prediction")
    category_totals = defaultdict(list)
    for e in st.session_state.expenses:
        category_totals[e["category"]].append(e["amount"])

    for category, amounts in category_totals.items():
        if len(amounts) >= 3:
            X = np.array(range(len(amounts))).reshape(-1, 1)
            y = np.array(amounts)
            model = LinearRegression()
            model.fit(X, y)
            future_days = np.array(range(len(amounts), len(amounts) + 7)).reshape(-1, 1)
            prediction = model.predict(future_days).sum()
            budget = st.session_state.category_budget[category]
            st.write(f"*{category}* → Predicted next week: ₹{int(prediction)}")
            if prediction > budget * 0.25:
                st.error("🚨 ML Alert: Overspending predicted")
            else:
                st.success("✅ ML predicts safe spending")

    st.divider()
    st.button("🔍 View Smart Insights", on_click=navigate, args=("insights",))
    st.button("🧮 What-If Analysis", on_click=navigate, args=("whatif",))

# =================================================
# PAGE 6 : INSIGHTS
# =================================================

# =================================================
# PAGE 6 : ADVANCED AI INSIGHTS
# =================================================
elif st.session_state.page == "insights":
    st.title("🧠 AI-Powered Financial Intelligence")

    if not st.session_state.expenses:
        st.info("Add expenses to unlock AI insights.")
        st.button("⬅ Back to Dashboard", on_click=navigate, args=("dashboard",))
        st.stop()

    df = pd.DataFrame(st.session_state.expenses)

    # ================= CORE STATS =================
    total_spent = df["amount"].sum()
    daily_spend = df.groupby("date")["amount"].sum()
    avg_daily = daily_spend.mean()
    max_day = daily_spend.idxmax()

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## 📊 Spending Intelligence Summary")

    c1, c2, c3 = st.columns(3)
    c1.metric("💰 Total Spent", f"₹{int(total_spent)}")
    c2.metric("📆 Avg Daily Spend", f"₹{int(avg_daily)}")
    c3.metric("🔥 Highest Spend Day", str(max_day))

    st.markdown('</div>', unsafe_allow_html=True)

    # ================= CATEGORY INTELLIGENCE =================
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## 🔍 Category Behavior Intelligence")
    
    cat_summary = (
        df.groupby("category")["amount"]
        .agg(["sum", "mean", "count"])
        .rename(columns={
            "sum": "Total Spent",
            "mean": "Avg Spend",
            "count": "Transactions"
        })
    )
    st.dataframe(cat_summary.style.format("₹{:.0f}"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.dataframe(cat_summary.style.format("₹{:.0f}"))

    # ================= ANOMALY DETECTION (REAL AI) =================
    st.subheader("🚨 AI Anomaly Detection (Unusual Spending)")

    threshold = avg_daily + 2 * daily_spend.std()

    anomalies = daily_spend[daily_spend > threshold]

    if not anomalies.empty:
        for day, amount in anomalies.items():
            st.error(f"⚠ Unusual spend detected on {day}: ₹{int(amount)}")
    else:
        st.success("✅ No abnormal spending patterns detected")

    # ================= TREND AI =================
    st.subheader("📈 Spending Trend Intelligence")

    X = np.arange(len(daily_spend)).reshape(-1, 1)
    y = daily_spend.values

    trend_strength = "Stable"
    trend_color = "green"

    if len(y) >= 5:
        model = LinearRegression()
        model.fit(X, y)
        slope = model.coef_[0]

        if slope > 8:
            trend_strength = "Aggressively Increasing"
            trend_color = "red"
        elif slope > 2:
            trend_strength = "Gradually Increasing"
            trend_color = "orange"

    st.markdown(f"### Trend Status: :{trend_color}[{trend_strength}]")

    # ================= SPENDING PERSONALITY =================
    st.subheader("🧬 Spending Personality (AI Classification)")

    food_ratio = cat_summary.loc["Food", "Total Spent"] / total_spent if "Food" in cat_summary.index else 0
    entertainment_ratio = cat_summary.loc["Entertainment", "Total Spent"] / total_spent if "Entertainment" in cat_summary.index else 0

    if food_ratio > 0.45:
        st.warning("🍔 **Impulse Spender** — High daily consumption behavior")
    elif entertainment_ratio > 0.35:
        st.warning("🎮 **Lifestyle Spender** — Entertainment-driven spending")
    else:
        st.success("💎 **Balanced Spender** — Healthy distribution detected")

    # ================= AI BEHAVIOR SCORE =================
    st.subheader("🧠 AI Financial Behavior Score")

    score = 100
    if total_spent > st.session_state.monthly_budget:
        score -= 35
    if anomalies.any():
        score -= 20
    if trend_strength != "Stable":
        score -= 15
    if avg_daily > (st.session_state.monthly_budget / 30):
        score -= 15

    score = max(score, 0)

    st.progress(score / 100)
    st.markdown(f"### Score: **{score} / 100**")

    if score >= 80:
        st.success("Elite financial discipline 💎")
    elif score >= 60:
        st.info("Good control, minor optimizations recommended")
    else:
        st.error("High-risk financial behavior detected")

    # ================= AI RECOMMENDATIONS =================
    st.subheader("🤖 AI Recommendations Engine")

    if food_ratio > 0.4:
        st.warning("🍔 Reduce food spend by ₹50/day → Save ₹1500/month")

    if trend_strength != "Stable":
        st.info("📉 Introduce 1 no-spend day per week")

    st.info("💡 Shift unused entertainment budget into savings")

    # ================= VISUAL EXPLANATION =================
    st.subheader("📊 AI Explainability Chart")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cat_summary.index,
        y=cat_summary["Total Spent"],
        marker_color="royalblue"
    ))
    fig.update_layout(
        title="Where Your Money Actually Goes",
        yaxis_title="Amount (₹)",
        xaxis_title="Category",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.button("⬅ Back to Dashboard", on_click=navigate, args=("dashboard",))

    col1, col2 = st.columns(2)
    col1.button("⬅ Dashboard", on_click=navigate, args=("dashboard",))
    col2.button("🧮 What-If Analysis", on_click=navigate, args=("whatif",))


# elif st.session_state.page == "insights":
#     st.title("🔍 Smart Financial Insights")

#     food_spend = sum(e["amount"] for e in st.session_state.expenses if e["category"] == "Food")
#     if food_spend > st.session_state.category_budget["Food"]:
#         st.warning("🍔 Food spending consistently exceeds budget")

#     st.subheader("🤖 AI-style Suggestions")
#     st.info("Reduce food spending by ₹40/day to improve savings")
#     st.info("Shift unused entertainment budget to savings")

#     st.button("⬅ Back to Dashboard", on_click=navigate, args=("dashboard",))

# =================================================
# PAGE 7 : WHAT-IF ANALYSIS
# =================================================
elif st.session_state.page == "whatif":
    st.title("🧮 What-If Decision Support (Interactive)")

    categories = ["Food", "Rent/Bills", "Entertainment"]
    simulation_changes = {}
    for cat in categories:
        simulation_changes[cat] = st.number_input(
            f"Change Daily Spend for {cat} (₹) (negative = reduce, positive = increase)",
            value=0
        )

    if st.button("Run Simulation"):
        predictions = {}
        df = pd.DataFrame(st.session_state.expenses)
        for cat in categories:
            cat_expenses = df[df["category"] == cat]["amount"].tolist()
            avg_daily = sum(cat_expenses) / len(cat_expenses) if cat_expenses else 0
            new_avg = avg_daily + simulation_changes[cat]
            predicted_week = new_avg * 7
            budget_week = st.session_state.category_budget[cat] * 0.25

            predictions[cat] = {
                "Current Avg Daily": avg_daily,
                "New Avg Daily": new_avg,
                "Predicted 7-Day Spend": predicted_week,
                "Weekly Budget Limit": budget_week
            }

        sim_table = pd.DataFrame(predictions).T
        st.dataframe(sim_table.style.format({"Current Avg Daily": "₹{:.0f}",
                                             "New Avg Daily": "₹{:.0f}",
                                             "Predicted 7-Day Spend": "₹{:.0f}",
                                             "Weekly Budget Limit": "₹{:.0f}"}))

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=categories,
            y=[predictions[c]["Weekly Budget Limit"] for c in categories],
            name="Weekly Budget",
            marker_color='lightgray',
            text=[f"₹{int(predictions[c]['Weekly Budget Limit'])}" for c in categories],
            textposition='auto'
        ))
        predicted_colors = ['green' if predictions[c]["Predicted 7-Day Spend"] <= predictions[c]["Weekly Budget Limit"] else 'red' for c in categories]
        fig.add_trace(go.Bar(
            x=categories,
            y=[predictions[c]["Predicted 7-Day Spend"] for c in categories],
            name="Predicted Spend",
            marker_color=predicted_colors,
            text=[f"₹{int(predictions[c]['Predicted 7-Day Spend'])}" for c in categories],
            textposition='auto'
        ))
        fig.update_layout(
            barmode='overlay',
            title="Predicted Weekly Spend vs Budget",
            yaxis_title="Amount (₹)",
            xaxis_title="Category",
            legend=dict(x=0.75, y=1.15),
            template='plotly_white',
            height=450
        )
        st.plotly_chart(fig, use_container_width=True)

        for cat in categories:
            if predictions[cat]["Predicted 7-Day Spend"] > predictions[cat]["Weekly Budget Limit"]:
                st.error(f"⚠ {cat} predicted to exceed weekly budget! 🚨")
            else:
                st.success(f"✅ {cat} is within weekly budget.")

    st.divider()

    st.button("⬅ Back to Dashboard", on_click=navigate, args=("dashboard",))

