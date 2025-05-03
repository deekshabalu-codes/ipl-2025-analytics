# IPL 2025 Match Outcome Prediction - Complete Script

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------- Step 1: Load Dataset --------------------
file_path = 'ipl_2025_matches.csv'  # Adjust this to match your CSV file name

try:
    df = pd.read_csv(file_path)
    print(f"✅ Loaded dataset: {file_path}")
except FileNotFoundError:
    print(f"❌ File not found: {file_path}")
    exit()

# -------------------- Step 2: Clean Columns --------------------
df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

# -------------------- Step 3: Inspect & Clean Data --------------------
# Preview match_winner values
print("🔎 Unique match_winner values:", df['match_winner'].unique())

# Fill missing run values if present
for col in ['runs_team1', 'runs_team2']:
    if col in df.columns:
        df[col].fillna(df[col].mean(), inplace=True)

# -------------------- Step 4: Encode Categorical Data --------------------
def encode_team_result(row, target_column):
    team1 = str(row['team1']).strip().lower()
    team2 = str(row['team2']).strip().lower()
    value = str(row[target_column]).strip().lower()
    if value == team1:
        return 0
    elif value == team2:
        return 1
    else:
        return np.nan

df['toss_winner'] = df.apply(lambda row: encode_team_result(row, 'toss_winner'), axis=1)
df['match_winner'] = df.apply(lambda row: encode_team_result(row, 'match_winner'), axis=1)

# Drop rows where match_winner couldn't be determined
df.dropna(subset=['match_winner'], inplace=True)

# -------------------- Step 5: Feature Engineering --------------------
df['runs_difference'] = df['runs_team1'] - df['runs_team2']

# Drop unused columns
columns_to_drop = ['match_date', 'venue', 'team1', 'team2']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# -------------------- Step 6: Visualization --------------------
plt.figure(figsize=(6, 4))
sns.countplot(x='match_winner', data=df)
plt.title('Match Winner Distribution')
plt.xlabel('Winner (0 = Team1, 1 = Team2)')
plt.ylabel('Match Count')
plt.show()

plt.figure(figsize=(6, 4))
sns.histplot(df['runs_difference'], kde=True)
plt.title('Runs Difference Distribution')
plt.show()

# -------------------- Step 7: Train Model --------------------
X = df[['toss_winner', 'runs_difference']]
y = df['match_winner']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------- Step 8: Evaluate --------------------
y_pred = model.predict(X_test)
print("\n✅ Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("🧮 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------- Step 9: Feature Importance --------------------
importances = model.feature_importances_
features = ['Toss Winner', 'Runs Difference']
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})

sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance')
plt.show()

# -------------------- Step 10: Prediction --------------------
print("\n🔮 Predict a Match Outcome:")
try:
    toss_input = int(input("Toss Winner (0 for Team1, 1 for Team2): "))
    run_diff_input = int(input("Runs Difference (Team1 - Team2): "))
    new_data = pd.DataFrame({'toss_winner': [toss_input], 'runs_difference': [run_diff_input]})
    pred = model.predict(new_data)[0]
    print("🏆 Predicted Match Winner: Team", 1 if pred == 0 else 2)
except Exception as e:
    print("⚠️ Error in prediction input:", e)
