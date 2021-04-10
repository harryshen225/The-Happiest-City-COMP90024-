import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import json
import re
from pandas.io.json import json_normalize
from mpi4py import MPI

comm = MPI.COMM_WORLD


def read_json(path):
    with open(path) as file:
        return json.loads(file.read())
    return None

def get_point_region(coord,mgrid_lite):
    allocated_region = None
    #print(mgrid_lite)
    for id,region in mgrid_lite.iterrows(): 
        left_margin = coord[0]-region.xmin
        right_margin = coord[0]-region.xmax
        top_margin = coord[1]-region.ymax
        bottom_margin = coord[1]-region.ymin
        if((left_margin > 0 and right_margin <= 0) and (top_margin <= 0 and bottom_margin > 0)):
            allocated_region = id
            break
    return allocated_region   

def get_sentiment_score(tweet, sentiment_guide):
    score = 0
    if "can't stand" in tweet: score += sentiment_guide.at["can't stand",'score']
    for token in tweet.split():
        score += sentiment_guide.at[token,'score'] if token in sentiment_guide.index else 0
    return score

def tweet_preprocessing(twitter_path,melbGrid_path,afinn_path,comm):
    # 1. Read tweets data, transform the data from JSON to dataframe, and only keep Text and Coordinates data
    tweets = json_normalize(read_json(twitter_path)['rows'])
    tweets_df = pd.concat([tweets['value.properties.text'].apply(lambda text: text.lower()),tweets['doc.coordinates.coordinates']],axis=1)
    tweets_df.columns = ['Text','Coordinates']

    my_tweets_df = tweets_df[tweets_df.apply(lambda df: df.index%comm.Get_size() == comm.Get_rank())].dropna()
    
    # 2. Clean up Tweet text to only contain legit words
    pattern = re.compile(r"(?:^|\s)([A-Za-z\'\.’]+[A-Za-z])(?=\s|\Z|[!,?.'\"’]+)")
    my_tweets_df['Text'] = my_tweets_df['Text'].apply(lambda text: " ".join(pattern.findall(text)))

    # 3. Read in box Coordinates into a dataframe and constuct a new df with the ingradients making up the square box(xmin, xmax, ymin,ymax)
    mgrid = json_normalize(read_json(melbGrid_path)['features'])
    mgrid_lite = pd.concat([mgrid['properties.id'], mgrid['properties.xmin'], mgrid['properties.xmax'], mgrid['properties.ymin'], mgrid['properties.ymax']],axis=1)
    mgrid_lite.columns = ['id','xmin','xmax','ymin','ymax']
    mgrid_lite=mgrid_lite.set_index('id')

    # 4. Define the region allocation of each tweet
    my_tweets_df['region'] = my_tweets_df['Coordinates'].apply(get_point_region,mgrid_lite=mgrid_lite)

    # 5. Read the sentiment guideline file into a dataframe
    sentiment_guide = pd.read_table(afinn_path,names=['word','score'],index_col='word')
    
    # 6. caculate the sentiment score for each tweet
    my_tweets_df['sentiment'] = my_tweets_df['Text'].apply(get_sentiment_score, sentiment_guide=sentiment_guide)

    # 7. calculate the total number of tweets and the corresponding sentiment score
    region_tweets = my_tweets_df.groupby('region').count()
    region_tweets = region_tweets.drop(columns=['sentiment','Coordinates'])
    region_tweets.columns = ['#Total Tweets']

    region_score = my_tweets_df.groupby('region').sum()
    region_score.columns = ['#Overall Sentiment Score'] 

    result = region_tweets.join(region_score,on="region").sort_values('region')

    return result

def tweets_processor(twitter_path,melbGrid_path,afinn_path,comm):
    rank = comm.Get_rank()
    result = tweet_preprocessing(twitter_path,melbGrid_path,afinn_path,comm).sort_values('#Overall Sentiment Score',ascending=True)
    slave_result = comm.gather(result.to_json(), root=0)
    final_result = None
    if rank==0:
        scattered_result_pd = pd.concat([pd.read_json(row_result) for row_result in slave_result])
        final_result = scattered_result_pd.groupby(scattered_result_pd.index).sum()
        print('final: \n {}'.format(final_result))

def main():
    comm = MPI.COMM_WORLD
    tweets_processor('./data/smallTwitter.json','./data/melbGrid.json','./data/AFINN.txt',comm)
    MPI.Finalize

if __name__ == "__main__":
    main()
