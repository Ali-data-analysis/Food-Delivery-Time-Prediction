# %% [markdown]
# Phase 1: Data Collection and Exploratory Data Analysis (EDA)
# 
# %% [markdown]
# 
# Step 1 - Data Import and Preprocessing
# 
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL.ImageOps import scale
from matplotlib.pyplot import axes
from numpy.ma.extras import corrcoef
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import ast,math
# %%
df = pd.read_csv(r"C:\Users\HP\Downloads\Food_Delivery_Time_Prediction.csv")
print(df.head())
# %%
print(df.info())
# %%
print(df.isnull().sum())
# %%
print(df.dtypes)
# %%
print(df.columns)
# %%
# Encode categorical variables
cat_features = ["Weather_Conditions","Traffic_Conditions","Vehicle_Type","Order_Priority","Order_Time"]
df_enc = pd.get_dummies(df,columns=cat_features,drop_first=True)
# %%
# Standardize numeric columns
scale_cols = ["Distance", "Order_Cost", "Delivery_Person_Experience", "Restaurant_Rating", "Customer_Rating", "Tip_Amount"]
scaler = StandardScaler()
df_enc[[c + "_scaled" for c in scale_cols]] = scaler.fit_transform(df_enc[scale_cols])
# %%
# Handle missing values
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(include="object").columns
df[num_cols] = df[num_cols].apply(lambda c: c.fillna(c.median()))
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])
# %% [markdown]
# STEP 2 - Exploratory Data Analysis (EDA)
# 
# %%
###Descriptive Statistics###
numeric_cols = ["Distance", "Delivery_Person_Experience", "Restaurant_Rating",
                "Customer_Rating", "Delivery_Time", "Order_Cost", "Tip_Amount"]
### Descriptive statistics ###
desc =df[numeric_cols].agg(["mean", "median", "var"]).T
desc["mode"] = df[numeric_cols].mode().iloc[0]
print(desc)
# %%
# Correlation analysis
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True,fmt=".2f",cmap="coolwarm",center=0)
plt.title("Correlation Heatmap")
plt.show()
# %%
# Outlier detection (boxplots) + handling (IQR capping)
fig, axes = plt.subplots(2, 4, figsize=(18, 7))

for ax, col in zip(axes.flatten(), numeric_cols):
    sns.boxplot(y=df[col], ax=ax)

# Hide the unused subplot
for ax in axes.flatten()[len(numeric_cols):]:
    ax.set_visible(False)

plt.tight_layout()
plt.show()
# %%
def cap_iqr(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return s.clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

for col in ["Distance", "Delivery_Time", "Order_Cost", "Tip_Amount"]:
    df_enc[col] = cap_iqr(df_enc[col])
# %% [markdown]
# Step 3 - Feature Engineering
# 
# %%
#Distance Calculation

def parse_latlon(s):
    lat, lon = ast.literal_eval(s)
    return float(lat), float(lon)
df_enc[["Cust_Lat", "Cust_Lon"]] = df["Customer_Location"].apply(lambda s: pd.Series(parse_latlon(s)))
df_enc[["Rest_Lat", "Rest_Lon"]] = df["Restaurant_Location"].apply(lambda s: pd.Series(parse_latlon(s)))

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

df_enc["Haversine_Distance_km"] = df_enc.apply(
    lambda r: haversine(r.Cust_Lat, r.Cust_Lon, r.Rest_Lat, r.Rest_Lon), axis=1)

# Time-based feature: rush hour flag
df_enc["Is_Rush_Hour"] = df["Order_Time"].isin(["Afternoon", "Evening"]).astype(int)

# %% [markdown]
# Phase 2: Predictive Modeling
# 
# %% [markdown]
# STEP 4 - Linear Regression Model
# %%
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


lr_features = [c for c in df_enc.columns if c.startswith(("Traffic_Conditions_", "Order_Priority_"))]
lr_features += ["Distance_scaled"]

X_train, X_test, y_train, y_test = train_test_split(
    df_enc[lr_features], df_enc["Delivery_Time"], test_size=0.2, random_state=42)

lr_model = LinearRegression().fit(X_train, y_train)
y_pred = lr_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Linear Regression -> MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.3f}")
# %% [markdown]
#  STEP 5 - Logistic Regression Model (Categorization)
# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


median_t = df_enc["Delivery_Time"].median()
df_enc["Delivery_Status"] = (df_enc["Delivery_Time"] > median_t).astype(int)  # 1 = Delayed

log_features = [c for c in df_enc.columns if c.startswith(("Weather_Conditions_", "Traffic_Conditions_"))]
log_features += ["Delivery_Person_Experience_scaled"]

Xl_train, Xl_test, yl_train, yl_test = train_test_split(
    df_enc[log_features], df_enc["Delivery_Status"], test_size=0.2, random_state=42,
    stratify=df_enc["Delivery_Status"])

log_model = LogisticRegression(max_iter=1000).fit(Xl_train, yl_train)
yl_pred = log_model.predict(Xl_test)
yl_prob = log_model.predict_proba(Xl_test)[:, 1]

acc = accuracy_score(yl_test, yl_pred)
prec = precision_score(yl_test, yl_pred)
rec = recall_score(yl_test, yl_pred)
f1 = f1_score(yl_test, yl_pred)
cm = confusion_matrix(yl_test, yl_pred)
print(f"Logistic Regression -> Acc: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}")
# %% [markdown]
# Phase 3: Reporting and Insights
# 
# %% [markdown]
# STEP 6 - Model Evaluation and Comparison
# %%
from sklearn.metrics import roc_curve, ConfusionMatrixDisplay, auc


ConfusionMatrixDisplay(cm, display_labels=["Fast", "Delayed"]).plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=130); plt.close()

fpr, tpr, _ = roc_curve(yl_test, yl_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.legend(); plt.title("ROC Curve")
plt.savefig("roc_curve.png", dpi=130); plt.close()

plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Delivery Time"); plt.ylabel("Predicted Delivery Time")
plt.title("Linear Regression: Actual vs Predicted")
plt.savefig("lr_actual_vs_pred.png", dpi=130); plt.close()

print("\n=== Comparison ===")
print(f"Linear Regression   -> MSE: {mse:.2f} | MAE: {mae:.2f} | R2: {r2:.3f}")
print(f"Logistic Regression -> Accuracy: {acc:.3f} | AUC: {roc_auc:.3f}")
# %% [markdown]
# Step 7 - Actionable Insights
# %%

# STEP 7 - Actionable Insights


print("=" * 60)
print("ACTIONABLE INSIGHTS & RECOMMENDATIONS")
print("=" * 60)

# 1. Insight from Linear Regression performance
print("\n1. Delivery Time Prediction (Linear Regression)")
if r2 < 0.3:
    print(f"   - R2 = {r2:.3f}: Distance, Traffic, and Order_Priority explain very")
    print("     little of the variation in delivery time. Before trusting this")
    print("     model operationally, verify that Distance/coordinates are")
    print("     correctly linked to the same delivery leg as Delivery_Time.")
else:
    print(f"   - R2 = {r2:.3f}: model captures a meaningful share of delivery-time variation.")

# 2. Insight from feature importance (Linear Regression coefficients)
coef_series = pd.Series(lr_model.coef_, index=lr_features).sort_values(key=abs, ascending=False)
top_driver = coef_series.index[0]
print(f"\n2. Strongest driver in the linear model: '{top_driver}'")
print("   - Prioritize monitoring/optimizing this factor operationally,")
print("     e.g. via routing or dispatch rules.")

# 3. Insight from Logistic Regression performance
print("\n3. Fast vs Delayed Classification (Logistic Regression)")
print(f"   - Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f} | AUC: {roc_auc:.3f}")
if acc < 0.6:
    print("   - Performance is close to random guessing. Available features")
    print("     (weather, traffic, experience) are not yet sufficient signals")
    print("     for reliably flagging delayed deliveries.")
else:
    print("   - Model reliably separates Fast vs Delayed deliveries; safe to")
    print("     use for proactive delay alerts.")

# 4. Rush-hour operational insight (derived from actual data, not assumed)
rush_avg = df_enc.loc[df_enc["Is_Rush_Hour"] == 1, "Delivery_Time"].mean()
non_rush_avg = df_enc.loc[df_enc["Is_Rush_Hour"] == 0, "Delivery_Time"].mean()
print(f"\n4. Rush-hour impact:")
print(f"   - Avg delivery time during rush hour (Afternoon/Evening): {rush_avg:.1f} min")
print(f"   - Avg delivery time otherwise: {non_rush_avg:.1f} min")
if rush_avg > non_rush_avg:
    print("   - Rush-hour orders are slower on average -> increase delivery")
    print("     staffing / prioritize dispatch during these windows.")
else:
    print("   - No clear rush-hour slowdown detected in this sample.")

# 5. Experience-level insight (derived from actual data)
low_exp = df_enc[df_enc["Delivery_Person_Experience"] <= df_enc["Delivery_Person_Experience"].median()]
high_exp = df_enc[df_enc["Delivery_Person_Experience"] > df_enc["Delivery_Person_Experience"].median()]
print(f"\n5. Delivery partner experience impact:")
print(f"   - Avg delivery time, lower-experience partners: {low_exp['Delivery_Time'].mean():.1f} min")
print(f"   - Avg delivery time, higher-experience partners: {high_exp['Delivery_Time'].mean():.1f} min")
print("   - If the gap is meaningful, invest in training/incentives for")
print("     less experienced delivery partners.")

# 6. Traffic condition insight
traffic_group = df.groupby("Traffic_Conditions")["Delivery_Time"].mean().sort_values(ascending=False)
print(f"\n6. Average delivery time by traffic condition:")
print(traffic_group.to_string())
print("   - Use this breakdown to set dynamic ETAs and reroute/dispatch")
print("     decisions by traffic level.")

# 7. Final recommendations (standard operational actions)
print("\n7. Recommended next steps:")
print("   - Optimize delivery routes using real road-network distance/time")
print("     data instead of straight-line coordinates.")
print("   - Adjust staffing levels during identified high-traffic/rush windows.")
print("   - Provide refresher training to lower-experience delivery partners.")
print("   - Re-run this pipeline on a larger, verified dataset before using")
print("     model outputs for operational decisions.")
# %%
