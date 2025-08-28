# MNIST Digit Recognition API

This project is a web-based digit recognition app using a neural network trained on the MNIST dataset. Users can draw a digit (0-9) in the browser, and the backend predicts the digit using a trained model.

## Features
- Draw digits in a browser grid
- Visualize neural network layers
- Predict digit using a Python backend (Flask + TensorFlow)
- Save and view training accuracy/loss plots

## Project Structure
```
├── train_mnist.py        # Python script to train and save the MNIST model
├── mnist_model.h5        # Saved Keras model (generated after training)
├── app.py                # Flask backend for prediction
├── frontend/             # Frontend code (React/JSX, HTML, CSS)
└── README.md             # Project documentation
```

## Getting Started

### 1. Train the Model
Run the training script to generate `mnist_model.h5`:
```bash
python3 train_mnist.py
```

### 2. Start the Backend
Install dependencies and run the Flask server:
```bash
./my_env/bin/pip install flask flask-cors tensorflow pillow numpy
python3 app.py
```
The backend will run at `http://127.0.0.1:5000`.

### 3. Run the Frontend
Set up your frontend (React or HTML/JS) to send POST requests to `/predict` with the drawn digit as a base64 PNG string.

Example fetch request:
```js
fetch('http://127.0.0.1:5000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ image: base64Image })
})
  .then(res => res.json())
  .then(data => {
    // Show prediction: data.prediction
  });
```

### 4. Deployment
You can host the backend on platforms like Render, Railway, Heroku, AWS, or DigitalOcean. See their docs for Python/Flask deployment.

## API
### POST `/predict`
- **Request:** `{ "image": "<base64 PNG>" }`
- **Response:** `{ "prediction": <digit> }`

## Requirements
- Python 3.10+
- Flask
- Flask-CORS
- TensorFlow
- Pillow
- numpy

## License
MIT

## Author
Your Name
