import streamlit as st
import pandas as pd
import joblib
import pickle
import matplotlib.pyplot as plt
import plotly.express as px

# Load Models and there Label Encoders
model1 = joblib.load('C:/CHANDRAJITH/Info/PROJECT/models/model1.pkl')
le1 = joblib.load('C:/CHANDRAJITH/Info/PROJECT/models/le_1.pkl')

model2 = joblib.load('C:/CHANDRAJITH/Info/PROJECT/models/model2.pkl')
le2 = joblib.load('C:/CHANDRAJITH/Info/PROJECT/models/le_2.pkl')

model3 = joblib.load('C:/CHANDRAJITH/Info/PROJECT/models/model3.pkl')
le3 = joblib.load('C:/CHANDRAJITH/Info/PROJECT/models/le_3.pkl')

# Loading saved dictionaries
with open('C:/CHANDRAJITH/Info/PROJECT/models/dicts.pkl', 'rb') as f:
    dict1 = pickle.load(f)
    dict2 = pickle.load(f)
    dict3 = pickle.load(f)

# Loading support dictionaries
with open('C:/CHANDRAJITH/Info/PROJECT/models/support_dicts.pkl', 'rb') as f:
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
ranks_df = pd.read_pickle('C:/CHANDRAJITH/Info/PROJECT/models/rank_dataframe.pkl')

# import the win% dataframe for display purpose
win_per_dataframe = pd.read_pickle('C:/CHANDRAJITH/Info/PROJECT/models/win_per_dataframe.pkl')

# import excel containing the schedule
schedule=pd.read_excel("C:/CHANDRAJITH/Info/PROJECT/DATA/ODI 2023 Schedule.xlsx")
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

