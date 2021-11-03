import pickle
import glob
import pandas as pd
import numpy as np
import json
import re
import time
import ast
import argparse
import tqdm
from itertools import permutations as pm
from collections import defaultdict
from operator import itemgetter

#Global Variables
num_participants=10

def create_user_history_dict(match_dataframe):

    # aggregate match-data
    match_dataframe = pd.concat([match_dataframe['train'], match_dataframe['val'], match_dataframe['test']]).reset_index(drop=True)

    # initialize user_to_match_history
    user_to_history = defaultdict(list)
    

    for match_idx in tqdm.tqdm(range(len(match_dataframe))):
        match = match_dataframe.iloc[match_idx]

        match_id = match['match_id']
        time = match['time']

        bans = []
        for p_idx in range(num_participants):
            bans.append(match['User' + str(p_idx + 1) + '_ban'])
        
        for p_idx in range(num_participants):

            user_id = match['User' + str(p_idx + 1) + '_id']
            win = match['User'+str(p_idx+1)+'_win']
            team = match['User'+str(p_idx+1)+'_team']
            champion = match['User' + str(p_idx + 1) + '_champion']
            role = match['User' + str(p_idx + 1) + '_role']
            stat = match['User' + str(p_idx + 1) + '_stat']

            current_match_info_dict = {}
            current_match_info_dict['match_id'] = match_id
            current_match_info_dict['time'] = time
            current_match_info_dict['win'] = win
            current_match_info_dict['team'] = team
            current_match_info_dict['ban'] = bans
            current_match_info_dict['champion'] = champion
            current_match_info_dict['role'] = role
            current_match_info_dict['stat'] = stat

            user_to_history[user_id].append(current_match_info_dict)

    # Sort the dictionary by timestamp
    for key in list(user_to_history.keys()):
        user_to_history[key] = sorted(user_to_history[key], key = lambda x :x['time'], reverse=False)

    return user_to_history

def create_match_dict(match_dataframe, user_to_history):
    # aggregate match-data
    match_dataframe = pd.concat([match_dataframe['train'], match_dataframe['val'], match_dataframe['test']]).reset_index(drop=True)

    new_match_dict =defaultdict(list)

    for match_idx in tqdm.tqdm(range(len(match_dataframe))):
        match = match_dataframe.iloc[match_idx]
        match_id = match['match_id']
        time = match['time']
        new_match_dict[match_id].append(time)
        for p_idx in range(num_participants):
            user_id = match['User' + str(p_idx + 1) + '_id']
            user_history_dict = user_to_history[user_id]
            for idx, match_dict in enumerate(user_history_dict):
                if match_dict['match_id'] == match_id:
                    user_history_idx = idx
                    break
            new_match_dict[match_id].append(tuple([user_id, user_history_idx]))

    return new_match_dict


def remove_matches_with_unk_user(user_to_history, new_match_dict, unk_user_threshold=30, unk_match_threshold=1):
    
    # Figure out how many users who don't have sufficient amount of history are in within each problematic match
    match_cnt = {}
    for key in user_to_history.keys():
        if len(user_to_history[key]) <= unk_user_threshold:
            for match_info in user_to_history[key]:
                if match_info['match_id'] not in match_cnt.keys():
                    match_cnt[match_info['match_id']] = 1
                else:
                    match_cnt[match_info['match_id']] += 1

    # Exclude matches which shouldn't be considered problematic
    unk_match_set = set()
    for key, value in match_cnt.items():
        if value > unk_match_threshold :
            unk_match_set.add(key)
    
    # Exclude matches with too many unknowns from new_match_dict
    for match in unk_match_set:
        del new_match_dict[match]

    # Exclude users who do not have a single match in new_match_dict
    empty_user = []
    for key in user_to_history.keys():
        user_matches = set([match['match_id'] for match in user_to_history[key]])
        if len(user_matches - unk_match_set) == 0:
            empty_user.append(key)

    for user_id in empty_user:
        del user_to_history[user_id]
    
    return user_to_history, new_match_dict

def create_match_dataframe(new_match_dict):

    #TODO: Convert the dictionaries to pandas DataFrames

    match_columns = ['match_id', 'time']
    for participant_id in range(1,num_participants+1):
        match_columns.append('User'+str(participant_id))
    new_match_dataframe = pd.DataFrame.from_dict(new_match_dict).T.reset_index()
    new_match_dataframe.columns = match_columns

    match_dataframe={}
    match_dataframe['train'] = new_match_dataframe[(new_match_dataframe['time'] < args.train_end_time)].reset_index().drop(columns=['index'])
    match_dataframe['val'] = new_match_dataframe[(args.train_end_time <= new_match_dataframe['time']) &
                                    (new_match_dataframe['time'] < args.val_end_time)].reset_index().drop(columns=['index'])
    match_dataframe['test'] = new_match_dataframe[(new_match_dataframe['time'] > args.val_end_time)].reset_index().drop(columns=['index'])

    return match_dataframe

def cnt_unk_match(user):
    match_dict={}
    for key in user.keys():
        if len(user[key]) <=30:
            for match_info in user[key]:
                if match_info['match_id'] not in match_dict.keys():
                    match_dict[match_info['match_id']] =1
                else:
                    match_dict[match_info['match_id']] +=1
    return match_dict


#TODO: Need to convert NAN values to 0!!
def normalize_stats(user_to_history,  args):

    train_stats = []
    val_stats = []
    test_stats = []

    for user_idx, match in user_to_history.items():
        for match_info in match:
            if float(match_info['time']) < args.train_end_time:
                train_stats.append(np.array([match_info['stat']]))
            elif float(match_info['time']) < args.val_end_time:
                val_stats.append(np.array([match_info['stat']]))
            else:
                test_stats.append(np.array([match_info['stat']]))

    train_stats = np.concatenate(train_stats)
    val_stats = np.concatenate(val_stats)
    test_stats = np.concatenate(test_stats)

    import pdb
    pdb.set_trace()
    
    mu = train_stats.mean(axis=0)
    std = train_stats.std(axis=0)
    train_stats = np.nan_to_num((train_stats - mu) / std)
    val_stats = np.nan_to_num((val_stats - mu) / std)
    test_stats = np.nan_to_num((test_stats - mu) / std)

    train_idx = 0
    val_idx = 0
    test_idx = 0
    for user_idx, match in user_to_history.items():
        for match_info in match:
            if float(match_info['time']) < args.train_end_time:
                match_info['stat'] = tuple(train_stats[train_idx])
                train_idx += 1
            elif float(match_info['time']) < args.val_end_time:
                match_info['stat'] = tuple(val_stats[val_idx])
                val_idx += 1
            else:
                match_info['stat'] = tuple(test_stats[test_idx])
                test_idx += 1

    return user_to_history


def create_interaction_data4DF(data, categorical_ids):
    num_participants = 10
    num_users = len(categorical_ids['user'])
    num_champions = len(categorical_ids['champion'])
    interaction_matrix = np.zeros((num_users, num_champions), dtype=int)

    for i in tqdm.tqdm(range(len(data))):
        match = data.iloc[i]
        for p_idx in range(num_participants):
            userIdx = int(match['User'+str(p_idx+1)+'_id'])
            championIdx = int(match['User'+str(p_idx+1)+'_champion'])
            interaction_matrix[userIdx][championIdx] += 1
    # Convert to the ratio of interaction with champion for each user: not used for memory issue
    # interaction_matrix = interaction_matrix / (np.sum(interaction_matrix, 1, keepdims=True) + 1e-6)

    return interaction_matrix

def create_interaction_data4dict(user_dict, categorical_ids):
    num_participants = 10
    num_users = len(user_dict.keys())
    num_champions = len(categorical_ids['champion'])
    interaction_matrix = np.zeros((num_users, num_champions), dtype=int)

    for user_idx, match_list in tqdm.tqdm(enumerate(user_dict.values())):
        for match in match_list:
            champion_idx = int(match['champion'])
            interaction_matrix[user_idx][champion_idx] += 1
    # Convert to the ratio of interaction with champion for each user: not used for memory issue
    # interaction_matrix = interaction_matrix / (np.sum(interaction_matrix, 1, keepdims=True) + 1e-6)

    return interaction_matrix

def create_interaction_data4dataframe(match_data, user_dict,categorical_ids):
    num_participants = 10
    num_users = len(user_dict.keys())
    user_idx_query_list = list(user_dict.keys())
    num_champions = len(categorical_ids['champion'])
    interaction_matrix = np.zeros((num_users, num_champions), dtype=int)

    for i in tqdm.tqdm(range(len(match_data))):
        match = match_data.iloc[i]
        for p_idx in range(num_participants):
            (user_idx, history_idx) = match['User'+str(p_idx+1)]
            championIdx = user_dict[user_idx][history_idx]['champion']
            interaction_matrix[user_idx_query_list.index(user_idx)][championIdx] += 1
    # Convert to the ratio of interaction with champion for each user: not used for memory issue
    # interaction_matrix = interaction_matrix / (np.sum(interaction_matrix, 1, keepdims=True) + 1e-6)

    return interaction_matrix

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='arguments for preprocessing')
    parser.add_argument('--data_dir', type=str, default='/home/dongyoonhwang/draftrec/preprocess') 
    parser.add_argument('--categorical_ids_file', type=float, default=None) 
    parser.add_argument('--match_dataframe_file', type=float, default=None)
    parser.add_argument('--train_end_time', type=float, default=1447750992) # train:21.06.01~21.08.28
    parser.add_argument('--val_end_time', type=float, default=1447781344)   # val: 21.08.28 ~ 21.09.01
    parser.add_argument('--test_end_time', type=float, default=1447829216)  # test: 21.09.01 ~ 21.09.09
    args = parser.parse_args()
    with open(args.data_dir + '/dota_categorical_ids.pickle', 'rb') as f:
        categorical_ids = pickle.load(f)

    with open(args.data_dir + '/dota_match_dataframe.pickle', 'rb') as f:
        match_dataframe = pickle.load(f)
    

    #1. Create user-history data and modified match_dataframe
    print('[1. Start creating user-history data]')
    user_history_dict = create_user_history_dict(match_dataframe)
    old_match_dict = create_match_dict(match_dataframe, user_history_dict)

    #new_user_history_dict, new_match_dict = remove_matches_with_unk_user(user_history_dict, old_match_dict)
    new_match_dataframe = create_match_dataframe(old_match_dict)

    # normalize stats
    new_user_history_dict = normalize_stats(user_history_dict, args)

    with open('./dota_user_history_data.pickle', 'wb') as f:
        pickle.dump(new_user_history_dict, f)
    with open('./dota_new_match_dict.pickle', 'wb') as f:
        pickle.dump(old_match_dict, f)
    with open('./dota_new_match_dataframe.pickle', 'wb') as f:
        pickle.dump(new_match_dataframe, f)

    print('Finish creating user-history data')
    # 2. Create interaction data
    #print('[4. Start creating interaction data]')
    #interaction_data = {}
    #interaction_data['train'] = create_interaction_data4dataframe(new_match_dataframe['train'], new_user_history_dict,categorical_ids)
    #interaction_data['val'] = create_interaction_data4dataframe(new_match_dataframe['val'], new_user_history_dict,categorical_ids)
    #interaction_data['test'] = create_interaction_data4dataframe(new_match_dataframe['test'], new_user_history_dict,categorical_ids)
    #with open('./dota_interaction_data.pickle', 'wb') as f:
    #    pickle.dump(interaction_data, f)
    #print('Finish creating interaction data')


