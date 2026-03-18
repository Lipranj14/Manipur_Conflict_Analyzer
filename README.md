# 🛡️ Manipur Conflict & Humanitarian Impact Analyzer

🔴 **[Live App: Click here to interact with the dashboard!](https://manipur-conflict-analyzer-lipranjdaharwal.streamlit.app/)**

An interactive geospatial and temporal analysis dashboard for visualizing and predicting conflict events in Manipur, India. Built with Python and Streamlit, this application provides actionable insights into the humanitarian impact of the conflict using historical data (ACLED synthetic dataset).

## 🌟 Features

- **Interactive Geospatial Map**: Visualize the distribution of conflict events across different districts with a dark-themed Folium map.
- **Temporal Trend Analysis**: Track the frequency and intensity of events over time with dynamic line charts.
- **Conflict Risk Prediction**: Utilize a trained **Random Forest Classifier** to predict the future risk level (High/Stable) of conflict escalation for specific districts, seasons, and historical lag indicators.
- **Feature Importance**: Understand which factors most heavily influence conflict prediction in the region.
- **Sleek UI/UX**: Dark glassmorphism design for a modern, immersive analytical experience.

## 🛠️ Technologies Used

- **Language**: Python 3.x
- **Frontend/Framework**: [Streamlit](https://streamlit.io/)
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn (Random Forest)
- **Data Visualization**: Plotly Express, Plotly Graph Objects, Folium, Streamlit-Folium

## 📂 Project Structure

- `app.py`: The main Streamlit dashboard application.
- `generate_dataset.py`: Script to generate a synthetic dataset roughly mirroring ACLED reporting styles for the region.
- `data/`: Contains the raw and processed datasets (`manipur_processed.csv`).
- `models/`: Stores the pickled Machine Learning models (`random_forest_conflict_model.pkl`) and label encoders.
- `src/`: Additional source code (data processing and modeling scripts).
- `requirements.txt`: Python package dependencies for the project.

## 🚀 How to Run Locally

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/YourUsername/Manipur_Conflict_Analyzer.git
   cd Manipur_Conflict_Analyzer
   ```

2. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Generate synthetic data and run pipelines**:
   If the processed data or models are missing, run the generation/training scripts first:
   ```bash
   python generate_dataset.py
   # (Run data processing and ML scripts in the src/ folder if applicable)
   ```

4. **Launch the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

## 📈 ML Model Details
The predictive component uses a **Random Forest Classifier** to determine the risk of escalation. It relies on engineered features such as:
- Encoded District and Season data.
- **Lag count metrics** (e.g., number of events/fatalities from the previous month).
- Temporal indicators (Year).

## 📝 Note
*This project was developed for educational and portfolio purposes, initially utilizing synthetic/simulated data modeled after established armed conflict databases to demonstrate data analysis and machine learning capabilities.*
