import re

import pandas as pd
from pandas import read_csv
import pathlib
from fuzzywuzzy import fuzz
from fuzzywuzzy import process


def clean_data(x):
    if isinstance(x, str):
        return x.replace(" ", "")
    else:
        print(x)
        return x


def match_fun(_data_frame):
    return re.search(_data_frame['ID'], _data_frame['ID'], re.IGNORECASE) is not None


#Get games data from CSV
locationGamesFile = pathlib.Path(r'../../data/raw_data/steam_games.csv')
dataGames = read_csv(locationGamesFile,
                     usecols=["name", "genre", "game_details", "popular_tags", "publisher", "developer"])

# locationGamesFile = pathlib.Path(r'../../data/raw_data/steam_games_2.csv')
# dataGames = read_csv(locationGamesFile,
#                      usecols=["Name","Short Description","Developer","Publisher","Genre","Tags","Type","Categories"], sep=";")

locationUsersFile = pathlib.Path(r'../../data/raw_data/steam_users_purchase_play.csv')
dataUsers = read_csv(locationUsersFile, header=None, usecols=[0, 1, 2, 3],
                     names=["user_id", "game_name", "behavior", "hours"])

dataGames['name'] = dataGames['name'].fillna('')
#dataGames['name'] = dataGames['name'].fillna('')
# create column ID for game and user dataset
dataGames["ID"] = ""
dataUsers["ID"] = ""

# remove spaces and special character from game name in both dataset
for i, row in dataGames.iterrows():
    clean = re.sub('[^A-Za-z0-9]+', '', row["name"])
    clean = clean.lower()
    dataGames.at[i, 'ID'] = clean
print(len(dataGames))
for i, row in dataUsers.iterrows():
    clean = re.sub('[^A-Za-z0-9]+', '', row["game_name"])
    clean = clean.lower()
    dataUsers.at[i, 'ID'] = clean

# find all the games in the game dataset that match the games in user dataset
gameArrayUsers = dataUsers["ID"].unique()
print(len(gameArrayUsers))
gameDataGames = dataGames["ID"].unique()
criteriaTest = dataGames['ID'].isin(gameArrayUsers)
usedGames = dataGames[criteriaTest]
print(len(usedGames))


# threshold = 70
# df1 = pd.DataFrame(gameArrayUsers)
# for i in myList1:
#     match1.append(process.extractOne(i, myList2, scorer=fuzz.ratio))
# df1['matches'] = match1
# print(match1)
# for j in df1['matches']:
#     if j[1] >= threshold:
#         print(j[0])
#         k.append(j[0])
#     match2.append(",".join(k))
#     k = []
#
# df1['matches'] = match2
# print("\nMatches...")
# print(df1)
from rapidfuzz import process, utils

# for (i, processed_query) in enumerate(myList1):
#     # None is skipped by extractOne, so we set the current element to None an
#     # revert this change after the comparision
#     myList1[i] = None
#     match = process.extractOne(processed_query, myList2, processor=None, score_cutoff=90.5)
#     myList1[i] = processed_query
#     if match:
#         df1.loc[i, 'fuzzy_match'] = myList2[match[2]]
#         df1.loc[i, 'fuzzy_match_score'] = match[1]
# df1.to_csv('tmp3.csv', index=False)
# criteriaTest = dataGames['ID'].isin(df1['fuzzy_match'])
# usedGames = dataGames[criteriaTest]
# print(len(usedGames))
# relevant info for recommendation: genre game_details popular_tags publisher developer
# usedGames.loc[:, 'genre'] =
# usedGames.loc[:, 'game_details'] = usedGames['game_details'].fillna('')
# usedGames.loc[:, 'popular_tags'] = usedGames['popular_tags'].fillna('')
# usedGames.loc[:, 'publisher'] = usedGames['publisher'].fillna('')
# usedGames.loc[:, 'developer'] = usedGames['developer'].fillna('')
usedGames = usedGames.fillna('')


usedGames.loc[:, 'genre'] = usedGames['genre'].apply(clean_data)
usedGames.loc[:, 'game_details'] = usedGames['game_details'].apply(clean_data)
usedGames.loc[:, 'popular_tags'] = usedGames['popular_tags'].apply(clean_data)
usedGames.loc[:, 'publisher'] = usedGames['publisher'].apply(clean_data)
usedGames.loc[:, 'developer'] = usedGames['developer'].apply(clean_data)

# usedGames.loc[:, 'Short Description'] = usedGames['Short Description'].apply(clean_data)
# usedGames.loc[:, 'Genre'] = usedGames['Genre'].apply(clean_data)
# usedGames.loc[:, 'Tags'] = usedGames['Tags'].apply(clean_data)
# usedGames.loc[:, 'Type'] = usedGames['Type'].apply(clean_data)
# usedGames.loc[:, 'Publisher'] = usedGames['Publisher'].apply(clean_data)
# usedGames.loc[:, 'Developer'] = usedGames['Developer'].apply(clean_data)

usedGames["genre_publisher_developer"] = usedGames['genre'] + "," + usedGames['publisher'] + "," + usedGames['developer']
usedGames["genre_popular_tags_developer"] = usedGames['genre'] + ","+ usedGames['popular_tags'] + ","+ usedGames['developer']
usedGames["genre_popular_tags_game_details"] = usedGames['genre'] + ","+ usedGames['popular_tags'] + ","+ usedGames['game_details']
usedGames["genre_publisher_developer_game_details"] = usedGames['genre'] + ","+ usedGames['publisher'] + ","+ usedGames['developer'] + ","+ usedGames['game_details']

# create some column containing a mix of different information
# usedGames["genre_publisher_developer"] = usedGames['Genre'] + "," + usedGames['Publisher'] + "," + usedGames[
#     'Developer']
# usedGames["genre_tags_developer"] = usedGames['Genre'] + "," + usedGames['Tags'] + "," + usedGames[
#     'Developer']
# usedGames["genre_tags_description"] = usedGames['Genre'] + "," + usedGames['Tags'] + "," + usedGames[
#     'Short Description']
# usedGames["genre_publisher_developer_short_description"] = usedGames['Genre'] + "," + usedGames['Publisher'] + "," + \
#                                                       usedGames['Developer'] + "," + usedGames['Short Description']
# usedGames["all_out"] = usedGames['Genre'] + "," + usedGames['Publisher'] + "," + usedGames['Developer'] + "," + usedGames['Short Description'] + "," \
#                        + usedGames['Tags'] + usedGames['Type']

usedGames.drop_duplicates("name")
usedGames.to_csv(pathlib.Path(r'../../data/intermediate_data/processed_games_for_content-based.csv'), index=False)
