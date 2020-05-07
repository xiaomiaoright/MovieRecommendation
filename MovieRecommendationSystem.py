import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')
cols = ['user_id', 'item_id', 'rating','timestamp']

df = pd.read_csv('u.data', sep ='\t', names=cols)

movie_title = pd.read_csv('Movie_Id_Titles')

df.head()

movie_title.head()

df = pd.merge(df, movie_title, on='item_id')

ratings = pd.DataFrame(df.groupby('title').mean()['rating'])
ratings['num of ratings'] = df.groupby('title').count()['rating']

fig = plt.figure(figsize=(6,4))
sns.distplot(ratings['num of ratings'])

fig = plt.figure(figsize=(6,4))
sns.distplot(ratings['rating'])


sns.jointplot(x='rating',y='num of ratings',data=ratings,kind='scatter',alpha=0.5)


movie_rec = df.pivot_table(values = 'rating', index='user_id',columns='title')
starwars_user_ratings = movie_rec['Star Wars (1977)']

similar_to_starwars = movie_rec.corrwith(starwars_user_ratings)
corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars = corr_starwars.join(ratings[['num of ratings']])
corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False).head()


liarliar_user_ratings = movie_rec['Liar Liar (1997)']
starwars_user_ratings
similar_to_liarliar = movie_rec.corrwith(liarliar_user_ratings)
corr_liarliar= pd.DataFrame(similar_to_liarliar, columns = ['Correlations'])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings[['num of ratings']])
corr_liarliar[corr_liarliar['num of ratings'] > 100].sort_values('Correlations', ascending=False)