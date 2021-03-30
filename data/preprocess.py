import pickle
import glob
import pandas as pd
import numpy as np
import json
import re
import time
import ast
import argparse


def preprocess(raw_data):
    # Extract the unique matches
    raw_data = raw_data.drop_duplicates()
    # Only consider the rank games
    raw_data = raw_data[(raw_data['queueId'] == 420) | (raw_data['queueId'] == 440)]
    # Extract the necessary features
    raw_data = raw_data[['gameVersion', 'gameCreation', 'teams', 'participants', 'participantIdentities']]
    raw_data = raw_data[raw_data['gameVersion'] >= '10.23.343.2581'].reset_index(drop=True)
    return raw_data


def init_dictionary():
    categorical_ids = {}
    # User dictionary
    user_to_idx = {}
    user_to_idx['UNK'] = 0
    user_to_idx['MASK'] = 1
    # Champion dictionary
    champion_to_idx = {}
    champion_to_idx['UNK'] = 0
    champion_to_idx['MASK'] = 1
    # Version dictionary
    version_to_idx = {}
    version_to_idx['UNK'] = 0
    version_to_idx['MASK'] = 1
    # Lane dictionary
    lane_to_idx = {'UNK': 0,
                   'MASK':1,
                   'TOP': 2,
                   'JUNGLE': 3,
                   'MIDDLE': 4,
                   'DUO_CARRY': 5,
                   'DUO_SUPPORT': 6}

    categorical_ids['user'] = user_to_idx
    categorical_ids['interaction_per_user'] = {}
    categorical_ids['champion'] = champion_to_idx
    categorical_ids['version'] = version_to_idx
    categorical_ids['lane'] = lane_to_idx
    return categorical_ids


def append_item_to_dictionary(data, categorical_ids):
    # Append user dictionary
    user_to_idx = categorical_ids['user']
    interaction_per_user = categorical_ids['interaction_per_user']
    user_cnt = len(user_to_idx)
    for i in range(len(data)):
        participantIdentities = ast.literal_eval(data['participantIdentities'][i])
        for participant in participantIdentities:
            accountId = participant['player']['accountId']
            if accountId not in interaction_per_user:
                interaction_per_user[accountId] = 1
            else:
                interaction_per_user[accountId] += 1

    # Append champion dictionary
    champion_to_idx = categorical_ids['champion']
    champion_cnt = len(champion_to_idx)
    for i in range(len(data)):
        participants = ast.literal_eval(data['participants'][i])
        for participant in participants:
            championId = participant['championId']
            if championId not in champion_to_idx:
                champion_to_idx[championId] = champion_cnt
                champion_cnt += 1

    # Append version dictionary
    version_to_idx = categorical_ids['version']
    version_cnt = len(version_to_idx)
    version_list = data['gameVersion'].unique()
    for version in version_list:
        if version not in version_to_idx:
            version_to_idx[version] = version_cnt
            version_cnt += 1
        
    categorical_ids['user'] = user_to_idx
    categorical_ids['champion'] = champion_to_idx
    categorical_ids['version'] = version_to_idx

    return categorical_ids


def create_match_data(data, categorical_ids, columns):
    user_to_idx = categorical_ids['user']
    champion_to_idx = categorical_ids['champion']
    version_to_idx = categorical_ids['version']
    lane_to_idx = categorical_ids['lane']

    # Initialize dataframe
    match_data = pd.DataFrame(index=data.index, columns=columns)
    
    # Fill the values in match_data
    for i in range(len(data)):
        match = data.iloc[i]
        # Version
        version = match['gameVersion']
        if version in version_to_idx:
            match_data.loc[i, 'version'] = version_to_idx[version]
        else:
            match_data.loc[i, 'version'] = version_to_idx['UNK']
        
        # Win & Ban
        teams = ast.literal_eval(match['teams'])
        blue_team = teams[0]
        red_team = teams[1]
        match_data.loc[i, 'win'] = blue_team['win']
        user_cnt = 0
        for ban in blue_team['bans']:
            if ban['championId'] in champion_to_idx:
                match_data.loc[i, 'User'+str(user_cnt+1)+'_ban'] = champion_to_idx[ban['championId']]
            else:
                match_data.loc[i, 'User'+str(user_cnt+1)+'_ban'] = champion_to_idx['UNK']
            user_cnt += 1
        for ban in red_team['bans']:
            if ban['championId'] in champion_to_idx:
                match_data.loc[i, 'User'+str(user_cnt+1)+'_ban'] = champion_to_idx[ban['championId']]
            else:
                match_data.loc[i, 'User'+str(user_cnt+1)+'_ban'] = champion_to_idx['UNK']
            user_cnt += 1
            
        # User
        user_cnt = 0
        participantIdentities = ast.literal_eval(match['participantIdentities'])
        for participant in participantIdentities:
            accountId = participant['player']['accountId']
            if accountId in user_to_idx:
                match_data.loc[i, 'User'+str(user_cnt+1)+'_id'] = user_to_idx[accountId]
            else:
                match_data.loc[i, 'User'+str(user_cnt+1)+'_id'] = user_to_idx['UNK']
            user_cnt += 1
            
        # Champion & Lane
        user_cnt = 0
        participants = ast.literal_eval(match['participants'])
        for participant in participants:
            championId = participant['championId']
            if championId in champion_to_idx:
                match_data.loc[i, 'User'+str(user_cnt+1)+'_champion'] = champion_to_idx[championId]
            else:
                match_data.loc[i, 'User'+str(user_cnt+1)+'_champion'] = champion_to_idx['UNK']
                    
            timeline = participant['timeline']
            role = timeline['role']
            lane = timeline['lane']
            if lane in lane_to_idx:
                match_data.loc[i, 'User'+str(user_cnt+1)+'_lane'] = lane_to_idx[lane]
            elif role in lane_to_idx:
                match_data.loc[i, 'User'+str(user_cnt+1)+'_lane'] = lane_to_idx[role]
            else:
                match_data.loc[i, 'User'+str(user_cnt+1)+'_lane'] = lane_to_idx['UNK']
            user_cnt += 1
    
    return match_data


def create_interaction_data(data, categorical_ids):
    num_participants = 10
    num_users = len(categorical_ids['user'])
    num_champions = len(categorical_ids['champion'])
    interaction_matrix = np.zeros((num_users, num_champions), dtype=int)

    for i in range(len(data)):
        match = data.iloc[i]
        for p_idx in range(num_participants):
            userIdx = match['User'+str(p_idx+1)+'_id']
            championIdx = match['User'+str(p_idx+1)+'_champion']
            interaction_matrix[userIdx][championIdx] += 1

    # Convert to the ratio of interaction with champion for each user: not used for memory issue
    # interaction_matrix = interaction_matrix / (np.sum(interaction_matrix, 1, keepdims=True) + 1e-6)

    return interaction_matrix


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='arguments for preprocessing')
    parser.add_argument('--data_dir', type=str, default='C:/Users/leehojoon/Desktop/projects/draftRec/data/')
    # Train: 20.11.12 ~ 21.02.09 / Val: 21.02.10 ~ 21.02.12 / Test: 21.02.13 ~ 21.02.15
    parser.add_argument('--train_end_time', type=float, default=1612882800000) # 21.02.10
    parser.add_argument('--val_end_time', type=float, default=1613142000000) # 21.02.13
    parser.add_argument('--interaction_threshold', type=int, default=10)
    args = parser.parse_args()
    file_list = glob.glob(args.data_dir + '*.csv')

    # 1. Build dictionary
    print('[1. Start building dictionary]')
    categorical_ids = init_dictionary()
    for idx, file in enumerate(file_list):
        print('File index:', idx)
        # 1.1 Preprocess raw_data
        raw_data = pd.read_csv(file, index_col=0).reset_index(drop=True)
        preprocessed_data = preprocess(raw_data)
        print('Num data:', len(preprocessed_data))
        del raw_data
        # 1.2 Split into train & test data
        train_data = preprocessed_data[(preprocessed_data['gameCreation'] < args.train_end_time)].reset_index(drop=True)
        val_data = preprocessed_data[(preprocessed_data['gameCreation'] >= args.train_end_time)
                                    &((preprocessed_data['gameCreation'] < args.val_end_time))].reset_index(drop=True)
        test_data = preprocessed_data[preprocessed_data['gameCreation'] > args.val_end_time].reset_index(drop=True)
        print('Num train data:', len(train_data))
        print('Num val data:', len(val_data))
        print('Num test data:', len(test_data))
        del preprocessed_data

        # 1.3 Append dictionary for designated file
        categorical_ids = append_item_to_dictionary(train_data, categorical_ids)
        print('Finish appending dictionary')
        del train_data
        del val_data
        del test_data

    # 1.4 Only keep track of the users who have more than interaction-thershold with champions
    for user, num_interaction in categorical_ids['interaction_per_user'].items():
        if num_interaction >= args.interaction_threshold:
            categorical_ids['user'][user] = len(categorical_ids['user'])

    # 2. Create match data
    print('[2. Start creating match data]')
    num_participants = 10
    columns = ['win', 'version']
    for participant_id in range(1,num_participants+1):
        columns.append('User'+str(participant_id)+'_id')
        columns.append('User'+str(participant_id)+'_lane')
        columns.append('User'+str(participant_id)+'_champion')
        columns.append('User'+str(participant_id)+'_ban')
    train_match_data = pd.DataFrame(columns=columns)
    val_match_data = pd.DataFrame(columns=columns)
    test_match_data = pd.DataFrame(columns=columns)

    for idx, file in enumerate(file_list):
        print('File index:', idx)
        raw_data = pd.read_csv(file, index_col=0).reset_index(drop=True)
        preprocessed_data = preprocess(raw_data)
        del raw_data

        train_data = preprocessed_data[(preprocessed_data['gameCreation'] < args.train_end_time)].reset_index(drop=True)
        val_data = preprocessed_data[(preprocessed_data['gameCreation'] >= args.train_end_time)
                                    &((preprocessed_data['gameCreation'] < args.val_end_time))].reset_index(drop=True)
        test_data = preprocessed_data[preprocessed_data['gameCreation'] > args.val_end_time].reset_index(drop=True)
        del preprocessed_data

        new_train_match_data = create_match_data(train_data, categorical_ids, columns)
        new_val_match_data = create_match_data(val_data, categorical_ids, columns)
        new_test_match_data = create_match_data(test_data, categorical_ids, columns)
        train_match_data = pd.concat([train_match_data, new_train_match_data], ignore_index=True, sort=False)
        val_match_data = pd.concat([val_match_data, new_val_match_data], ignore_index=True, sort=False)
        test_match_data = pd.concat([test_match_data, new_test_match_data], ignore_index=True, sort=False)

    match_data = {}
    match_data['train'] = train_match_data.drop_duplicates().reset_index(drop=True)
    match_data['val'] = val_match_data.drop_duplicates().reset_index(drop=True)
    match_data['test'] = test_match_data.drop_duplicates().reset_index(drop=True)
    with open('./match_data.pickle', 'wb') as f:
        pickle.dump(match_data, f)
    print('Num train data:', match_data['train'])
    print('Num val data:', match_data['val'])
    print('Num test data:', match_data['test'])
    print('Finish creating match data')

    # 3. Create interaction data
    print('[3. Start creating interaction data]')
    interaction_data = {}
    interaction_data['train'] = create_interaction_data(match_data['train'], categorical_ids)
    interaction_data['val'] = create_interaction_data(match_data['val'], categorical_ids)
    interaction_data['test'] = create_interaction_data(match_data['test'], categorical_ids)
    with open('./interaction_data.pickle', 'wb') as f:
        pickle.dump(interaction_data, f)
    print('Finish creating interaction data')

    with open('categorical_ids.pickle', 'wb') as f:
        pickle.dump(categorical_ids, f)
