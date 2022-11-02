import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
# KMeans clustering Method
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
warnings.simplefilter("ignore", category=FutureWarning)

# df = pd.read_csv("Weather.csv")
# print(df.head())
dataframe = pd.read_csv("Weather.csv", index_col=0)
dataframe1 = pd.melt(dataframe, id_vars='YEAR', value_vars=dataframe.columns[1:])
# print(df1.head())
dataframe1['Date'] = dataframe1['variable'] + ' ' + dataframe1['YEAR'].astype(str)
dataframe1.loc[:, 'Date'] = dataframe1['Date'].apply(lambda x: datetime.strptime(x, '%b %Y'))
# print(df1.head())
dataframe1.columns = ['Year', 'Month', 'Temperature', 'Date']
dataframe1.sort_values(by='Date', inplace=True)
fig = go.Figure(layout=go.Layout(yaxis=dict(range=[0, dataframe1['Temperature'].max() + 1])))
fig.add_trace(go.Scatter(x=dataframe1['Date'], y=dataframe1['Temperature']), )
fig.update_layout(title='Temperature Thought Timeline: ', xaxis_title='Time', yaxis_title='Temperature in Degrees')
fig.update_layout(xaxis=go.layout.XAxis(
    rangeselector=dict(
        buttons=list([dict(label="Whole View", step="all"),
                      dict(count=1, label="One Year View", step="year", stepmode="todate")
                      ])),
    rangeslider=dict(visible=True), type="date")
)
fig.show()

# Warmest/Coldest/Average:
fig = px.box(dataframe1, 'Month', 'Temperature')
fig.update_layout(title='Warmest, Coldest and Average Monthly Temperature')
fig.show()

# Evaluation on number of clusters
# sse = []
# target = dataframe1['Temperature'].to_numpy().reshape(-1, 1)
# num_clusters = list(range(1, 10))
# for k in num_clusters:
#     km = KMeans(n_clusters=k)
#     km.fit(target)
#     sse.append(km.inertia_)
#
# fig = go.Figure(data=[
#     go.Scatter(x=num_clusters, y=sse, mode='lines'),
#     go.Scatter(x=num_clusters, y=sse, mode='markers')
# ])
#
# fig.update_layout(title="Evaluation on number of clusters:",
#                   xaxis_title="Number of Clusters",
#                   yaxis_title="Sum of Squared Distance",
#                   showlegend=False
#                   )
# fig.show()

# Seasonal Temperature Graph
# Cluster size of 3
km = KMeans(3)
km.fit(dataframe1['Temperature'].to_numpy().reshape(-1, 1))
dataframe1.loc[:, 'Temp Labels'] = km.labels_
fig = px.scatter(dataframe1, 'Date', 'Temperature', color='Temp Labels')
fig.update_layout(title='Temperature clusters.', xaxis_title="Date", yaxis_title="Temperature")
fig.show()

# Frequency chart of temperature
fig = px.histogram(x=dataframe1['Temperature'], nbins=200, histnorm='density')
fig.update_layout(title='Frequency chart of Temperature:', xaxis_title='Temperature', yaxis_title='Count')
fig.show()

# Yearly average temperature
dataframe['Yearly Mean'] = dataframe.iloc[:, 1:].mean(axis=1)
fig = go.Figure(data=[
    go.Scatter(name='Yearly Temperature', x=dataframe['YEAR'], y=dataframe['Yearly Mean'], mode='lines'),
    go.Scatter(name='Yearly Temperature', x=dataframe['YEAR'], y=dataframe['Yearly Mean'], mode='markers')
])
fig.update_layout(title='Yearly Mean Temperature: ', xaxis_title='Time', yaxis_title='Temperature in Degrees')
fig.show()

# Monthly Temperature through history
fig = px.line(dataframe1, 'Year', 'Temperature', facet_col='Month', facet_col_wrap=4)
fig.update_layout(title='Monthly Temperature through history: ')
fig.show()

# # Seasonal Weather Analysis
# df['Winter'] = df[['DEC', 'JAN', 'FEB']].mean(axis=1)
# df['Summer'] = df[['MAR', 'APR', 'MAY']].mean(axis=1)
# df['Monsoon'] = df[['JUN', 'JUL', 'AUG', 'SEP']].mean(axis=1)
# df['Autumn'] = df[['OCT', 'NOV']].mean(axis=1)
# seasonal_df = df[['YEAR', 'Winter', 'Summer', 'Monsoon', 'Autumn']]
# seasonal_df = pd.melt(seasonal_df, id_vars='YEAR', value_vars=seasonal_df.columns[1:])
# seasonal_df.columns=['Year', 'Season', 'Temperature']
# fig = px.scatter(seasonal_df, 'Year', 'Temperature', facet_col='Season', facet_col_wrap=2, trendline='ols')
# fig.update_layout(title='Seasonal mean temperature through years: ')
# fig.show()

# try to find out something out of an animation
fig = px.scatter(dataframe1, 'Month', 'Temperature', size='Temperature', animation_frame='Year')
fig.show()

# Weather forecasting with Machine learning
# Using Decision Tree regressor for prediction as the data do
dataframe2 = dataframe1[['Year', 'Month', 'Temperature']].copy()
# print(df2.head())
dataframe2 = pd.get_dummies(dataframe2)
y = dataframe2[['Temperature']]
x = dataframe2.drop(columns='Temperature')
dtr = DecisionTreeRegressor()
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3)
dtr.fit(train_x, train_y)
prediction = dtr.predict(test_x)
r2_score(test_y, prediction)    # Output 0.9678939568898212
# high r2 value means that our predictive model is working good

# Forecasted data for 2018
next_Year = dataframe1[dataframe1['Year'] == 2017][['Year', 'Month']]
next_Year.Year.replace(2017, 2018, inplace=True)
next_Year = pd.get_dummies(next_Year)
temp_2018 = dtr.predict(next_Year)
temp_2018 = {'Month': dataframe1['Month'].unique(), 'Temperature': temp_2018}
temp_2018 = pd.DataFrame(temp_2018)
temp_2018['Year'] = 2018
print(temp_2018)

# Forecasted Temperature
forecasted_temp = pd.concat([dataframe1, temp_2018], sort=False).groupby(by='Year')['Temperature'].mean().reset_index()
fig = go.Figure(data=[
    go.Scatter(name='Yearly Mean Temperature', x=forecasted_temp['Year'], y=forecasted_temp['Temperature'],
               mode='lines'),
    go.Scatter(name='Yearly Mean Temperature', x=forecasted_temp['Year'], y=forecasted_temp['Temperature'],
               mode='markers')
])
fig.update_layout(title='Forecasted Temperature:', xaxis_title='Time', yaxis_title='Temperature in Degrees')
fig.show()
