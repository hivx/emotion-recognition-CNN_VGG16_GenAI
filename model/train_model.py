import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd

# Load data
data_file = "data.txt"
data = np.loadtxt(data_file, delimiter=',')

X = data[:, :-1]
y = data[:, -1].astype(int)  # Đảm bảo nhãn là số nguyên

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
)

# Huấn luyện mô hình
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Dự đoán và đánh giá
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"🎯 Accuracy: {accuracy * 100:.2f}%")

# Hiển thị confusion matrix với tên cảm xúc
emotions = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index=emotions, columns=emotions)
print("📊 Confusion Matrix:")
print(df_cm)

# Lưu mô hình
with open('model.pkl', 'wb') as f:
    pickle.dump(rf_classifier, f)
    print("✅ Mô hình đã được lưu vào model.pkl")
