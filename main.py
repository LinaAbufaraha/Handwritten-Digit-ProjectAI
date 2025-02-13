import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import joblib
import os
import time
import logging

#Setup logging
logging.basicConfig(filename="training_logs.log", level=logging.INFO,
                    format="%(asctime)s - %(message)s", filemode="w")
logging.info("Program started.")

# ----------------------- Load Arabic Dataset 
print("Loading Arabic dataset...")
train_images_ar = pd.read_csv("C:/Users/Lenovo/Desktop/folders/AI_Project/arabicdataset/csvTrainImages 60k x 784.csv", header=None)
train_labels_ar = pd.read_csv("C:/Users/Lenovo/Desktop/folders/AI_Project/arabicdataset/csvTrainLabel 60k x 1.csv", header=None)
test_images_ar = pd.read_csv("C:/Users/Lenovo/Desktop/folders/AI_Project/arabicdataset/csvTestImages 10k x 784.csv", header=None)
test_labels_ar = pd.read_csv("C:/Users/Lenovo/Desktop/folders/AI_Project/arabicdataset/csvTestLabel 10k x 1.csv", header=None)

# Reshape and normalize Arabic dataset
x_train_ar = train_images_ar.values.reshape(-1, 28, 28, 1) / 255.0
x_test_ar = test_images_ar.values.reshape(-1, 28, 28, 1) / 255.0

# Convert labels to one-hot encoding
y_train_ar = to_categorical(train_labels_ar.values)
y_test_ar = to_categorical(test_labels_ar.values)

# Flatten data for Decision Tree and KNN models
train_images_ar_flat = train_images_ar.values
test_images_ar_flat = test_images_ar.values

# ----------------------- Load MNIST Dataset 
print("Loading MNIST dataset...")
(x_train_en, y_train_en), (x_test_en, y_test_en) = mnist.load_data()

# Reshape and normalize MNIST dataset
x_train_en = x_train_en.reshape(-1, 28, 28, 1) / 255.0
x_test_en = x_test_en.reshape(-1, 28, 28, 1) / 255.0

# Convert labels to one-hot encoding
y_train_en = to_categorical(y_train_en, 10)
y_test_en = to_categorical(y_test_en, 10)

# Flatten data for Decision Tree and KNN models
train_images_en_flat = x_train_en.reshape(-1, 28 * 28)
test_images_en_flat = x_test_en.reshape(-1, 28 * 28)

# ----------------------- Train/Load Decision Tree (Arabic) -----------------------
start_time = time.time()
if os.path.exists("decision_tree_model_ar.pkl"):
    decision_tree_model_ar = joblib.load("decision_tree_model_ar.pkl")
    print("Loaded Arabic Decision Tree model.")
else:
    print("Training Arabic Decision Tree model...")
    decision_tree_model_ar = DecisionTreeClassifier()
    decision_tree_model_ar.fit(train_images_ar_flat, train_labels_ar.values.ravel())
    joblib.dump(decision_tree_model_ar, "decision_tree_model_ar.pkl")
    print("Saved Arabic Decision Tree model.")
dt_time_ar = time.time() - start_time

# ----------------------- Train/Optimize KNN (Arabic) -----------------------
def train_knn_model(train_images, train_labels, test_images, test_labels, save_path):
    
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    #k Grid Search
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}  
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=3, scoring='accuracy', verbose=1)
    grid_search.fit(train_images, train_labels)

    # Get the best model
    best_knn = grid_search.best_estimator_

    joblib.dump(best_knn, save_path)
    print(f"Saved optimized KNN model to {save_path}")

    knn_accuracy = accuracy_score(test_labels, best_knn.predict(test_images))
    print(f"KNN Accuracy: {knn_accuracy * 100:.2f}%")
    return knn_accuracy

# Train or load Arabic KNN model
if os.path.exists("knn_model_ar_optimized.pkl"):
    knn_model_ar = joblib.load("knn_model_ar_optimized.pkl")
    print("Loaded optimized Arabic KNN model.")
    knn_accuracy_ar = accuracy_score(test_labels_ar.values.ravel(), knn_model_ar.predict(test_images_ar_flat))
else:
    print("Training and optimizing Arabic KNN model...")
    start_time = time.time()
    knn_accuracy_ar = train_knn_model(
        train_images_ar_flat, train_labels_ar.values.ravel(),
        test_images_ar_flat, test_labels_ar.values.ravel(),
        "knn_model_ar_optimized.pkl"
        
    )
    
print(f"KNN Accuracy (Arabic): {knn_accuracy_ar * 100:.2f}%")
print("Saved Arabic KNN model.")
knn_time_ar = time.time() - start_time

# Train or load Arabic KNN model
if os.path.exists("knn_model_en_optimized.pkl"):
    knn_model_en = joblib.load("knn_model_en_optimized.pkl")
    print("Loaded optimized English KNN model.")
else:
    print("Training and optimizing English KNN m;bodel...")
    start_time = time.time()
    knn_accuracy_en = train_knn_model(
        train_images_en_flat, np.argmax(y_train_en, axis=1),
        test_images_en_flat, np.argmax(y_test_en, axis=1),
        "knn_model_en_optimized.pkl"
    ) 
    print("Saved English KNN model.")
knn_time_en = time.time() - start_time

# ----------------------- Train/Load CNN (Arabic) -----------------------
start_time = time.time()
if os.path.exists("cnn_model_arabic.h5"):
    cnn_model_ar = load_model("cnn_model_arabic.h5")
    print("Loaded Arabic CNN model.")
else:
    print("Training Arabic CNN model...")
    cnn_model_ar = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    cnn_model_ar.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn_model_ar.fit(x_train_ar, y_train_ar, epochs=5, batch_size=32, validation_data=(x_test_ar, y_test_ar))
    cnn_model_ar.save("cnn_model_arabic.h5")
    print("Saved Arabic CNN model.")
cnn_time_ar = time.time() - start_time

# Accuracy for Arabic models
dt_accuracy_ar = accuracy_score(test_labels_ar.values.ravel(), decision_tree_model_ar.predict(test_images_ar_flat))
cnn_accuracy_ar = cnn_model_ar.evaluate(x_test_ar, y_test_ar, verbose=0)[1] * 100

# Collect results
results = {
    "Model": [
        "Decision Tree (Arabic)", "KNN (Arabic)", "CNN (Arabic)"
    ],
    "Accuracy (%)": [
        dt_accuracy_ar * 100, knn_accuracy_ar * 100, cnn_accuracy_ar
    ],
    "Training Time (seconds)": [
        dt_time_ar, knn_time_ar, cnn_time_ar
    ]
}

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("model_results_arabic.csv", index=False)
print("Results saved to model_results_arabic.csv")

# ----------------------- Train/Load Decision Tree (English) -----------------------
start_time = time.time()
if os.path.exists("decision_tree_model_en.pkl"):
    decision_tree_model_en = joblib.load("decision_tree_model_en.pkl")
    print("Loaded English Decision Tree model.")
else:
    print("Training English Decision Tree model...")
    decision_tree_model_en = DecisionTreeClassifier()
    decision_tree_model_en.fit(train_images_en_flat, np.argmax(y_train_en, axis=1))
    joblib.dump(decision_tree_model_en, "decision_tree_model_en.pkl")
    print("Saved English Decision Tree model.")
dt_time_en = time.time() - start_time


# ----------------------- Train/Load CNN (English) -----------------------
start_time = time.time()
if os.path.exists("cnn_model_english.h5"):
    cnn_model_en = load_model("cnn_model_english.h5")
    print("Loaded English CNN model.")
else:
    print("Training English CNN model...")
    cnn_model_en = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    cnn_model_en.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    cnn_model_en.fit(x_train_en, y_train_en, epochs=5, batch_size=32, validation_data=(x_test_en, y_test_en))
    cnn_model_en.save("cnn_model_english.h5")
    print("Saved English CNN model.")
cnn_time_en = time.time() - start_time


# Accuracy for English models
dt_accuracy_en = accuracy_score(np.argmax(y_test_en, axis=1), decision_tree_model_en.predict(test_images_en_flat))
knn_accuracy_en = accuracy_score(np.argmax(y_test_en, axis=1), knn_model_en.predict(test_images_en_flat))
cnn_accuracy_en = cnn_model_en.evaluate(x_test_en, y_test_en, verbose=0)[1] * 100

# Collect results
results_en = {
    "Model": [
        "Decision Tree (English)", "KNN (English)", "CNN (English)"
    ],
    "Accuracy (%)": [
        dt_accuracy_en * 100, knn_accuracy_en * 100, cnn_accuracy_en
    ],
    "Training Time (seconds)": [
        dt_time_en, knn_time_en, cnn_time_en
    ]
}

# Save results to CSV
results_en_df = pd.DataFrame(results_en)
results_en_df.to_csv("model_results_english.csv", index=False)
print("Results saved to model_results_english.csv")


# ----------------------- GUI Functionality -----------------------

def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = Image.eval(img, lambda x: 255 - x)  # Invert colors if necessary
    img = img.resize((28, 28))  # Resize to match the model input
    img_array_flat = np.array(img).reshape(1, -1) / 255.0  # Flattened version
    img_array_cnn = np.array(img).reshape(1, 28, 28, 1) / 255.0  # For CNN
    return img_array_flat, img_array_cnn


def predict_with_model(image_path, dataset, algorithm):
    img_flat, img_cnn = preprocess_image(image_path)
    if dataset == "Arabic":
        if algorithm == "Decision Tree":
            return decision_tree_model_ar.predict(img_flat)[0]
        elif algorithm == "KNN":
            return knn_model_ar.predict(img_flat)[0]
        elif algorithm == "CNN":
            prediction = cnn_model_ar.predict(img_cnn)
            return np.argmax(prediction)
    elif dataset == "English":
        if algorithm == "Decision Tree":
            return decision_tree_model_en.predict(img_flat)[0]
        elif algorithm == "KNN":
            return knn_model_en.predict(img_flat)[0]
        elif algorithm == "CNN":
            prediction = cnn_model_en.predict(img_cnn)
            return np.argmax(prediction)


from tkinter import filedialog, Tk, Button, Label, ttk, messagebox
from PIL import Image, ImageTk

def open_gui():
    def upload_image():
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", ".png;.jpg;*.jpeg")])
        if file_path:
            uploaded_image_label.config(text=file_path)
            global uploaded_image
            uploaded_image = file_path

            img = Image.open(file_path)
            img = img.resize((150, 150))
            img = ImageTk.PhotoImage(img)
            image_label.config(image=img)
            image_label.image = img

    def predict():
        if not uploaded_image:
            messagebox.showerror("Error", "Please upload an image!")
            return
        dataset = dataset_combobox.get()
        algorithm = algorithm_combobox.get()
        if not dataset or not algorithm:
            messagebox.showerror("Error", "Please select dataset and algorithm!")
            return
        result = predict_with_model(uploaded_image, dataset, algorithm)
        result_label.config(text=f"Predicted Number: {result}")

    root = Tk()
    root.title("Digit Recognition")
    root.geometry("500x600")

    # Upload Image Button
    Button(root, text="Upload Image", command=upload_image).pack(pady=10)
    uploaded_image_label = Label(root, text="No Image Uploaded")
    uploaded_image_label.pack()

    # Image Display
    image_label = Label(root)
    image_label.pack()

    # Select Dataset
    Label(root, text="Select Dataset").pack(pady=5)
    dataset_combobox = ttk.Combobox(root, values=["Arabic", "English"], state="readonly")
    dataset_combobox.pack()

    Label(root, text="Select Algorithm").pack(pady=5)
    algorithm_combobox = ttk.Combobox(root, values=["Decision Tree", "KNN", "CNN"], state="readonly")
    algorithm_combobox.pack()

    Button(root, text="Predict", command=predict).pack(pady=20)

    result_label = Label(root, text="Predicted Number: ", font=("Arial", 14), fg="blue")
    result_label.pack(pady=10)

    root.mainloop()

# Run GUI
uploaded_image = None
open_gui()
print("\nArabic Model Results:")
print(results_df)

print("\nEnglish Model Results:")
print(results_en_df)