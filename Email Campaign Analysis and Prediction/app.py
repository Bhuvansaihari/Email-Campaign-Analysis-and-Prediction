# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 2: Load the data
email_df = pd.read_csv(r"C:\Users\bhuva\Downloads\step\step\email_table.csv")
opened_df = pd.read_csv(r"C:\Users\bhuva\Downloads\step\step\email_opened_table.csv")
clicked_df = pd.read_csv(r"C:\Users\bhuva\Downloads\step\step\link_clicked_table.csv")

# Step 3: Add open and click information
email_df['opened'] = email_df['email_id'].isin(opened_df['email_id']).astype(int)
email_df['clicked'] = email_df['email_id'].isin(clicked_df['email_id']).astype(int)

# Step 4: Q1 - What % of users opened the email and what % clicked the link?
total_emails = len(email_df)
open_rate = email_df['opened'].mean() * 100
click_rate = email_df['clicked'].mean() * 100

print(f"Q1:")
print(f"Total Emails Sent: {total_emails}")
print(f"Percentage of users who opened the email: {open_rate:.2f}%")
print(f"Percentage of users who clicked the link: {click_rate:.2f}%")

# Step 5: Q2 - Build a model to predict who is likely to click
# Feature Engineering
df = email_df.copy()
df = pd.get_dummies(df, columns=['email_text', 'email_version', 'weekday', 'user_country'], drop_first=True)

X = df.drop(columns=['email_id', 'opened', 'clicked'])
y = df['clicked']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("\nQ2:")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 6: Q3 - How much can your model improve CTR? How would you test that?
y_probs = model.predict_proba(X_test)[:, 1]

# Let's simulate a targeted campaign - top 20% most likely to click
threshold = np.percentile(y_probs, 80)
targeted = y_probs >= threshold

simulated_ctr = y_test[targeted].sum() / targeted.sum()
actual_ctr = y_test.mean()

improvement = ((simulated_ctr - actual_ctr) / actual_ctr) * 100

print("\nQ3:")
print(f"Actual Click-Through Rate: {actual_ctr:.4f}")
print(f"Simulated Click-Through Rate (Top 20% Users): {simulated_ctr:.4f}")
print(f"Estimated CTR Improvement: {improvement:.2f}%")
print("To test this, run an A/B test: send emails only to users with highest predicted scores and compare CTRs.")

# Step 7: Q4 - Interesting patterns in campaign performance
print("\nQ4:")

# Pattern 1: Click rate by email version
click_by_version = email_df.groupby('email_version')['clicked'].mean()
print("Click rate by email version:\n", click_by_version)

# Pattern 2: Click rate by email text
click_by_text = email_df.groupby('email_text')['clicked'].mean()
print("\nClick rate by email text:\n", click_by_text)

# Pattern 3: Click rate by weekday
click_by_day = email_df.groupby('weekday')['clicked'].mean()
print("\nClick rate by weekday:\n", click_by_day)

# Pattern 4: Click rate by user country (top 5)
click_by_country = email_df.groupby('user_country')['clicked'].mean().sort_values(ascending=False).head(5)
print("\nTop 5 countries by click rate:\n", click_by_country)

# Pattern 5: More purchases = higher click rate?
bins = [0, 1, 3, 5, 10, np.inf]
labels = ['0', '1-3', '4-5', '6-10', '10+']
email_df['purchase_bin'] = pd.cut(email_df['user_past_purchases'], bins=bins, labels=labels)
click_by_purchases = email_df.groupby('purchase_bin')['clicked'].mean()
print("\nClick rate by purchase history:\n", click_by_purchases)

# Optional Visualizations (can be commented out if submitting in script format)
sns.barplot(x=click_by_version.index, y=click_by_version.values)
plt.title("Click Rate by Email Version")
plt.show()

sns.barplot(x=click_by_text.index, y=click_by_text.values)
plt.title("Click Rate by Email Text")
plt.show()
