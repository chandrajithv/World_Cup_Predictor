import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import plotly.express as px
import sys
print(sys.executable)


# Load Models and there Label Encoders
with open('models_n.pkl', 'rb') as f:
    model1 = pickle.load(f)
    model2 = pickle.load(f)
    model3 = pickle.load(f)

with open('encoders_n.pkl', 'rb') as f:
    le1 = pickle.load(f)
    le2 = pickle.load(f)
    le3 = pickle.load(f)

# Loading saved dictionaries
with open('dicts.pkl', 'rb') as f:
    dict1 = pickle.load(f)
    dict2 = pickle.load(f)
    dict3 = pickle.load(f)

# Loading support dictionaries
with open('support_dicts.pkl', 'rb') as f:
    team_summary_df_dict = pickle.load(f)
    ranking_df_dict = pickle.load(f)
    pitch_details_dict = pickle.load(f)



top_5_odi_batsmen_team=['Pakistan','India','South Africa','Ireland']
top_5_odi_bowler_team=['India','Australia','New Zealand','Afghanistan']

# Function to assign values to the sample dictionary according to the user input
def model_1_dict(team1, team2,Day,Month,Year,Ground,Country, data_dict):
    updated_dict = data_dict.copy()  
    
    if team1 in team_summary_df_dict['Team']:
        index = team_summary_df_dict['Team'].index(team1)
        updated_dict['Team 1 win%']=[team_summary_df_dict['%W'][index]]
    else:
        updated_dict['Team 1 win%']=[0]
        
    if team2 in team_summary_df_dict['Team']:
        index = team_summary_df_dict['Team'].index(team2)
        updated_dict['Team 2 win%']=[team_summary_df_dict['%W'][index]]
    else:
        updated_dict['Team 2 win%']=[0]
        
    if team1 in ranking_df_dict['Team']:
        index = ranking_df_dict['Team'].index(team1)
        updated_dict['Team 1 Ranking']=[ranking_df_dict['Rank'][index]]
    else:
        updated_dict['Team 1 Ranking']=[0]
        
    if team2 in ranking_df_dict['Team']:
        index = ranking_df_dict['Team'].index(team2)
        updated_dict['Team 2 Ranking']=[ranking_df_dict['Rank'][index]]
    else:
        updated_dict['Team 2 Ranking']=[0]
    
    updated_dict['Day']=[int(Day)]
    updated_dict['Month']=[int(Month)]
    updated_dict['Year']=[int(Year)]
    
    updated_dict['top_5_batsmen_team_1']=[1 if team1 in top_5_odi_batsmen_team else 0]
    updated_dict['top_5_batsmen_team_2']=[1 if team2 in top_5_odi_batsmen_team else 0]
    updated_dict['top_5_bowler_team_1']=[1 if team1 in top_5_odi_bowler_team else 0]
    updated_dict['top_5_bowler_team_2']=[1 if team2 in top_5_odi_bowler_team else 0]
    
    for key in updated_dict.keys():
        if key.startswith('Ground_'):
            updated_dict[key] = [1 if key == f'Ground_{Ground}' else 0]
            
    for key in updated_dict.keys():
        if key.startswith('Country_'):
            updated_dict[key] = [1 if key == f'Country_{Country}' else 0]
            
   
    for key in updated_dict.keys():
        if key.startswith('Team 1_'):
            updated_dict[key] = [1 if key == f'Team 1_{team1}' else 0]
    
    
    for key in updated_dict.keys():
        if key.startswith('Team 2_'):
            updated_dict[key] = [1 if key == f'Team 2_{team2}' else 0]

    return pd.DataFrame(updated_dict)

# Function to assign values to the sample dictionary according to the user input
def model_2_dict(team1, team2,Day,Month,Year,Ground,Country, data_dict):
    updated_dict = data_dict.copy()  
    
    if team1 in team_summary_df_dict['Team']:
        index = team_summary_df_dict['Team'].index(team1)
        updated_dict['Team 1 win%']=[team_summary_df_dict['%W'][index]]
    else:
        updated_dict['Team 1 win%']=[0]
        
    if team2 in team_summary_df_dict['Team']:
        index = team_summary_df_dict['Team'].index(team2)
        updated_dict['Team 2 win%']=[team_summary_df_dict['%W'][index]]
    else:
        updated_dict['Team 2 win%']=[0]
        
    if team1 in ranking_df_dict['Team']:
        index = ranking_df_dict['Team'].index(team1)
        updated_dict['Team 1 Ranking']=[ranking_df_dict['Rank'][index]]
    else:
        updated_dict['Team 1 Ranking']=[0]
        
    if team2 in ranking_df_dict['Team']:
        index = ranking_df_dict['Team'].index(team2)
        updated_dict['Team 2 Ranking']=[ranking_df_dict['Rank'][index]]
    else:
        updated_dict['Team 2 Ranking']=[0]
       
    if Ground in pitch_details_dict['Ground']:
        index = pitch_details_dict['Ground'].index(Ground)
        updated_dict['Pitch Rating']=[pitch_details_dict['Pitch Rating'][index]]
        updated_dict['Outfield Rating']=[pitch_details_dict['Outfield Rating'][index]]
    else: 
        updated_dict['Pitch Rating']=[0]
        updated_dict['Outfield Rating']=[0]

    updated_dict['Day']=[int(Day)]
    updated_dict['Month']=[int(Month)]
    updated_dict['Year']=[int(Year)]
    
    updated_dict['top_5_batsmen_team_1']=[1 if team1 in top_5_odi_batsmen_team else 0]
    updated_dict['top_5_batsmen_team_2']=[1 if team2 in top_5_odi_batsmen_team else 0]
    updated_dict['top_5_bowler_team_1']=[1 if team1 in top_5_odi_bowler_team else 0]
    updated_dict['top_5_bowler_team_2']=[1 if team2 in top_5_odi_bowler_team else 0]
    
    for key in updated_dict.keys():
        if key.startswith('Ground_'):
            updated_dict[key] = [1 if key == f'Ground_{Ground}' else 0]
            
    for key in updated_dict.keys():
        if key.startswith('Country_'):
            updated_dict[key] = [1 if key == f'Country_{Country}' else 0]
            
   
    for key in updated_dict.keys():
        if key.startswith('Team 1_'):
            updated_dict[key] = [1 if key == f'Team 1_{team1}' else 0]
    
    
    for key in updated_dict.keys():
        if key.startswith('Team 2_'):
            updated_dict[key] = [1 if key == f'Team 2_{team2}' else 0]

    return pd.DataFrame(updated_dict)


# Function to assign values to the sample dictionary according to the user input
def model_3_dict(team1, team2,Day,Month,Year,Ground, data_dict):
    updated_dict = data_dict.copy()  
        
    updated_dict['Day']=[int(Day)]
    updated_dict['Month']=[int(Month)]
    updated_dict['Year']=[int(Year)]
    
    for key in updated_dict.keys():
        if key.startswith('Ground_'):
            updated_dict[key] = [1 if key == f'Ground_{Ground}' else 0]
       
    for key in updated_dict.keys():
        if key.startswith('Team 1_'):
            updated_dict[key] = [1 if key == f'Team 1_{team1}' else 0]
    
    
    for key in updated_dict.keys():
        if key.startswith('Team 2_'):
            updated_dict[key] = [1 if key == f'Team 2_{team2}' else 0]

    return pd.DataFrame(updated_dict)

# Function to predict the winner
def weighted_prediction(team1, team2,Day,Month,Year,Ground,Country):
    
    predictions = [(le1.inverse_transform(model1.predict(model_1_dict(team1, team2,Day,Month,Year,Ground,Country, dict1)))[0]),
              (le2.inverse_transform(model2.predict(model_2_dict(team1, team2,Day,Month,Year,Ground,Country, dict2)))[0]),
               (le3.inverse_transform(model3.predict(model_3_dict(team1, team2,Day,Month,Year,Ground, dict3)))[0])]
    weights = [0.4, 0.3, 0.3]
    # Create a dictionary to store the total weight for each prediction
    weight_dict = {}
    for pred, weight in zip(predictions, weights):
        if pred in weight_dict:
            weight_dict[pred] += weight
        else:
            weight_dict[pred] = weight

    # Find the prediction with the highest total weight
    max_weight_pred = max(weight_dict, key=weight_dict.get)

    return max_weight_pred

# import the ranks dataframe for display purpose
ranks_df = pd.read_pickle('rank_dataframe.pkl')

# import the win% dataframe for display purpose
win_per_dataframe = pd.read_pickle('win_per_dataframe.pkl')

# import excel containing the schedule
schedule=pd.read_excel("ODI 2023 Schedule.xlsx")
schedule[['Team 1', 'Team 2']] = schedule['Matches'].str.split('vs', expand=True)
schedule['Team 1']=schedule['Team 1'].str.strip().str.title()
schedule['Team 2']=schedule['Team 2'].str.strip().str.title()
schedule['Ground']=schedule['Ground'].str.strip()
schedule[['Month','Day','Year']]=schedule['Date'].str.split(' ', expand=True)
schedule['Month']=schedule['Month'].str.split().str[-1].str.strip()
month_dict = {'Oct': 10, 'Nov': 11}
schedule['Month'] = schedule['Month'].replace(month_dict).astype(int)
schedule['Day']=schedule['Day'].str.replace(',', '').str.strip().astype(int)
schedule['Year']=schedule['Year'].str.strip().astype(int)
schedule.drop(columns=['Matches','Date'],inplace=True)

# List of all teams playing
all_teams=list(set(schedule['Team 1']).union(set(schedule['Team 2'])))

# List of all grounds
Grounds_list=list(set(schedule['Ground']))

# Streamlit app creation
# Define the Streamlit app
def main():
    st.set_page_config(page_title="WC Predictor",page_icon=":trophy:",layout="wide")
    # Set a title for your app
    st.title(":cricket_bat_and_ball: CV ODI WC Predictor")
    st.markdown("##")
    ##########
    # Your model details
    model_details = """
    ## Note: Model Combination and Data Sources
    
    The predicted winner shown is a combination of three models, each trained on different datasets:
    - **Model 1 (Matches from 2018 to September 17, 2023):** This model was trained on international ODI match data spanning from 2018 to September 17, 2023. This dataset was obtained through web scraping from ESPNcricinfo.com.
    - **Model 2 (Matches from January 2022 to September 2023):** Model 2 was trained on data obtained from the ICC website. This dataset includes match details, pitch ratings, and outfield ratings for matches played from January 2022 to September 2023.
    - **Model 3 (Matches Played in Indian Grounds):** Model 3 was trained on data specific to ODI matches played in Indian grounds. This dataset was curated to focus on matches conducted in venues relevant to the World Cup.

    In addition to these primary data sources, several enhancements were made to enrich the datasets:
    - **Win Percentage Data (2020 to 2023):** Win percentage data for various teams in the years 2020 to 2023 was added to the datasets. This information provides insights into each team's performance leading up to the predictions.
    - **ODI Rankings (2018 to 2023):** ODI rankings for various teams spanning from 2018 to 2023 were integrated into the data. These rankings offer an additional dimension of team strength and standing.

    The final prediction is a combination of predictions by these three models, with the following weightage:
    Model 1: 40%
    Model 2: 30%
    Model 3: 30%
    This multi-model approach and the inclusion of diverse datasets aim to provide comprehensive and accurate predictions for upcoming ODI matches, considering various factors that may influence match outcomes.

    **Disclaimer: Toss Details Not Considered**
    Please be aware that the predictions do not factor in toss details, which can influence match outcomes in cricket. This omission may result in a slight bias, especially when both teams have similar winning probabilities. Users should consider toss results and other dynamic factors when using these predictions.
    """
    
    # Add to sidebar
    st.sidebar.markdown(model_details)

    ##################

        
    # Create input widgets for user input
    team1 = st.selectbox('Choose Team 1 Name:', all_teams)
    #st.write(f'You selected: {team1}')
    team2 = st.selectbox('Choose Team 2 Name:', all_teams)
    #st.write(f'You selected: {team2}')
    Day = st.number_input("Enter the day", min_value=1, max_value=31, step=1)
    Month = st.number_input("Enter the month", min_value=1, max_value=12, step=1)
    Year = st.number_input("Enter the year", min_value=2019, max_value=2024, step=1)
    Ground = st.selectbox('Choose Ground Name:', Grounds_list)
    Country = st.selectbox('Select the Country Hosting the Match:', all_teams)

    # Create a button to trigger predictions
    if st.button("Predict Winner"):
        if team1 and team2 and Day and Month and Year and Ground and Country:
            
            # Call your prediction function
            prediction = weighted_prediction(team1, team2,Day,Month,Year,Ground,Country)

            # Display the prediction
            st.success(f"Predicted Winner: {prediction}")
        else:
            st.warning("Please fill in all fields.")

    # Filter the DataFrame based on user input
    filtered_ranks_df_team1 = ranks_df[ranks_df['Team'] == team1]
    filtered_ranks_df_team2 = ranks_df[ranks_df['Team'] == team2]

    filtered_ranks_df_team1['Year'] = filtered_ranks_df_team1['Year'].astype(str)
    filtered_ranks_df_team2['Year'] = filtered_ranks_df_team2['Year'].astype(str)

    # Filter the DataFrame for the selected teams
    selected_teams = win_per_dataframe[win_per_dataframe['Team'].isin([team1, team2])]

    # Set the title of your Streamlit app
    st.title(f"{team1} vs {team2} Analytics")
    #st.subheader("Win Percentage Comparison for 2023")

    # Create a bar chart for win percentages
    fig = px.bar(
        selected_teams,
        x='Team',
        y='%W',
        title="Win Percentage Comparison for 2023",
        labels={'Teams': 'Win Percentage (%)'},
    )

    # Customize the chart layout
    fig.update_layout(xaxis_title="Team Name", yaxis_range=[0, 100])

    # Show the bar chart
    st.plotly_chart(fig)

    

    # Create line charts for Team 1 and Team 2 ranks
    st.subheader(f"{team1}'s Rank Over Time")
    st.line_chart(filtered_ranks_df_team1.set_index('Year')['Rank'])

    st.subheader(f"{team2}'s Rank Over Time")
    st.line_chart(filtered_ranks_df_team2.set_index('Year')['Rank'])

    

if __name__ == "__main__":
    main()


