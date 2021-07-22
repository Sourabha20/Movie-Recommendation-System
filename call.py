import flask
from flask import Flask,redirect,url_for,render_template
from flask import request
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
 
movies = pd.read_csv("C:/Users/my pc/Desktop/movierecommend/movies.csv")
ratings = pd.read_csv("C:/Users/my pc/Desktop/movierecommend/ratings.csv")


final_dataset = ratings.pivot(index='movieId',columns='userId',values='rating')

final_dataset.fillna(0,inplace=True)

no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')
final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index,:]
final_dataset=final_dataset.loc[:,no_movies_voted[no_movies_voted > 50].index]


csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)

knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

app=Flask(__name__)

@app.route("/")


def home():
    return render_template("first.html")
def get_movie_recommendation(movie_name):
    n_movies_to_reccomend = 10
    movie_list = movies[movies['title'].str.contains(movie_name)]  
    if len(movie_list):        
        movie_idx= movie_list.iloc[0]['movieId']
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
        distances , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=n_movies_to_reccomend+1)    
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        recommend_frame = []
        for val in rec_movie_indices:
            movie_idx = final_dataset.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            #recommend_frame.append({'TITLE':movies.iloc[idx]['title'].values[0],'GENRE':movies.iloc[idx]['genre'].values[0]})
            recommend_frame.append({'TITLE':movies.iloc[idx]['title'].values[0],'GENRE':movies.iloc[idx]['genre'].values[0],'Distance':val[1]})
        df = pd.DataFrame(recommend_frame,index=range(1,n_movies_to_reccomend+1))
        return  df
    else:
        return "No movies found. Please check your input"
    


def get_movie_recommendation_kids(movie_name):
    n_movies_to_reccomend = 20
    movie_list = movies[movies['title'].str.contains(movie_name)]  
    if len(movie_list):        
        movie_idx= movie_list.iloc[0]['movieId']
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
        distances , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=n_movies_to_reccomend+1)    
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        recommend_frame = []
        for val in rec_movie_indices:
            movie_idx = final_dataset.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            #recommend_frame.append({'TITLE':movies.iloc[idx]['title'].values[0],'GENRE':movies.iloc[idx]['genre'].values[0]})
            recommend_frame.append({'TITLE':movies.iloc[idx]['title'].values[0],'GENRE':movies.iloc[idx]['genre'].values[0],'DISTANCE':val[1]})
        df = pd.DataFrame(recommend_frame,index=range(1,n_movies_to_reccomend+1))
        df=pd.DataFrame(df[df['GENRE'].str.contains("Children")])
        
        return df
    else:
        return "No movies found. Please check your input"
@app.route("/",methods=['POST','GET'])
def movie_recommend():
    agegroup=request.form.get('age')
    movie_title=request.form.get('movie')
    if agegroup=="1":
        result=get_movie_recommendation_kids(movie_title)
    else:
        result=get_movie_recommendation(movie_title)
    title = []
    genre = []
    dist=[]
            
    for i in range(len(result)):
        title.append(result.iloc[i][0])
        genre.append(result.iloc[i][1])
        dist.append(result.iloc[i][2])

    return flask.render_template('output.html',movie_names=title,movie_g=genre,movie_dist=dist)

   
    


if __name__ == "__main__":
    app.run(debug=True)