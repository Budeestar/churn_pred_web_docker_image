import React, { useState } from 'react';

function App() {
  const [inputData, setInputData] = useState({
    SeniorCitizen: '',
    MonthlyCharges: '',
    TotalCharges: '',
    gender: '', // Adjust these fields according to your form inputs and expected model inputs
    Partner: '',
    Dependents: '',
    PhoneService: '',
    MultipleLines: '',
    InternetService: '',
    OnlineSecurity: '',
    OnlineBackup: '',
    DeviceProtection: '',
    TechSupport: '',
    StreamingTV: '',
    StreamingMovies: '',
    Contract: '',
    PaperlessBilling: '',
    PaymentMethod: '',
    tenure: ''
  });
  const [predictions, setPredictions] = useState('');
  const [probability, setProbability] = useState('');

  const handleInputChange = e => {
    setInputData({ ...inputData, [e.target.name]: e.target.value });
  };

  const handlePredict = () => {
    fetch('http://localhost:5000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(inputData)
    })
    .then(response => response.json())
    .then(data => {
      setPredictions(data.prediction[0]); // Assuming a single prediction
      setProbability(`Probability: ${data.probability[0] * 100}%`); // Assuming a single probability
    })
    .catch(error => {
      console.error('Error:', error);
      setPredictions('Error in making prediction');
      setProbability('');
    });
  };

  return (
    <div className="App">
      <h1>Churn Prediction</h1>
      {/* Input fields */}
      <div className="input-container">
        {Object.keys(inputData).map(key => (
          <div key={key}>
            <label htmlFor={key}>{key}:</label>
            <input
              type="text"
              id={key}
              name={key}
              value={inputData[key]}
              onChange={handleInputChange}
            />
          </div>
        ))}
      </div>
      <button onClick={handlePredict}>Predict</button>
      {/* Display predictions */}
      <div id="output">
        <p>Prediction: {predictions}</p>
        <p>{probability}</p>
      </div>
    </div>
  );
}

export default App;
