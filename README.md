# Cricket ODI World Cup Predictor

![Cricket](https://github.com/chandrajithv/World_Cup_Predictor/blob/1a6520b60d7b7b6310cf053d54b96e46c5127c16/Predictor_Web_App_photo.jpg)

This is a Cricket World Cup predictor web application developed using Streamlit. The application combines predictions from three machine learning models to forecast the winner of upcoming matches.

## About

- The first model is trained on international ODI match data from 2018 to September 17, 2023, sourced from ESPNcricinfo.
- The second model utilizes match details, pitch ratings, and outfield ratings from ICC's website for the period between January 2022 and September 2023.
- The third model focuses on matches played in Indian grounds, particularly those used for the World Cup, and considers specific ground data.

## Models and Weights

The final prediction is a weighted combination of these models:
- Model 1: 40%
- Model 2: 30%
- Model 3: 30%

All three models were built using the hyperparameter-tuned XGBClassifier algorithm from scikit-learn.

## Repository Contents

- **`models_n.pkl`:** This file contains all three saved machine learning models used by the application.

- **`encoders_n.pkl`:** This file contains the label encoders used for the models.

- **`dicts.pkl`:** This file contains dictionaries required for model inputs.

- **`support_dicts.pkl`:** Here, you can find support dictionaries used in the application.

- **`rank_dataframe.pkl`:** This file stores data needed for rank calculations.

- **`win_per_dataframe.pkl`:** This file contains win percentage data used in the application.

- **`streamlit_app.py`:** The Streamlit web application code.

- **`requirements.txt`:** Lists the necessary Python packages and dependencies required to run the application.

All these files are present directly in the main branch of this repository.

## Usage

To use the predictor:

1. Clone this repository.
2. Install the required Python packages using `pip install -r requirements.txt`.
3. Run the Streamlit app with `streamlit run app.py`.

Please note that toss details are not considered, which may lead to slight bias when both teams have similar winning probabilities. Additionally, predictions are based on data available up to September 17, 2023.

## Contribution

Feel free to contribute to the project by submitting issues or pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
