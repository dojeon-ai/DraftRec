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

def init_dictionary():
    categorical_ids = {}
    # User dictionary
    user_to_idx = {}
    user_to_idx['PAD'] = 0
    user_to_idx['MASK'] = 1
    user_to_idx['CLS'] = 2
    user_to_idx['UNK'] = 3
    # Champion dictionary
    champion_to_idx = {}
    champion_to_idx['PAD'] = 0
    champion_to_idx['MASK'] = 1
    champion_to_idx['CLS'] = 2
    champion_to_idx['UNK'] = 3

    # role dictionary
    role_to_idx = {}
    role_to_idx['PAD'] = 0
    role_to_idx['MASK'] = 1
    role_to_idx['CLS'] = 2
    role_to_idx['UNK'] = 3

    # Team dictionary
    team_to_idx = {}
    team_to_idx['PAD'] = 0
    team_to_idx['MASK'] = 1
    team_to_idx['CLS'] = 2
    team_to_idx['UNK'] = 3
    team_to_idx['BLUE'] = 4
    team_to_idx['RED'] = 5
    # Win dictionary
    win_to_idx = {}
    win_to_idx['PAD'] = 0
    win_to_idx['MASK'] = 1
    win_to_idx['CLS'] = 2
    win_to_idx['UNK'] = 3
    win_to_idx['False'] = 4  # Blue-team lose
    win_to_idx['True'] = 5  # Blue-team win

    categorical_ids['user'] = user_to_idx
    categorical_ids['champion'] = champion_to_idx
    categorical_ids['role'] = role_to_idx
    categorical_ids['team'] = team_to_idx
    categorical_ids['win'] = win_to_idx
    return categorical_ids


def build_match_dataframe(match, players):
    columns = ['match_id', 'time']
    for participant_id in range(1,11):
        columns.append('User'+str(participant_id)+'_id')
        columns.append('User'+str(participant_id)+'_win')
        columns.append('User'+str(participant_id)+'_role')
        columns.append('User'+str(participant_id)+'_champion')
        columns.append('User'+str(participant_id)+'_ban')
        columns.append('User'+str(participant_id)+'_team')
        columns.append('User'+str(participant_id)+'_stat')

    data = pd.DataFrame(columns=columns)
    data['time'] = match['start_time']
    data['match_id'] = data.index
    
    blue_team_order = [1,4,5,8,9]
    red_team_order= [2,3,6,7,10]

    account_ids = defaultdict(list)
    champion_ids = defaultdict(list)
    
    win_ids = defaultdict(list)
    stat_ids = defaultdict(list)
    for match_idx in tqdm.tqdm(range(len(match))):
        for participant_id in range(1, 11):
            idx = match_idx*10 + participant_id-1
            row = players.iloc[idx]
            account_id = row['account_id']
            champion_id = row['hero_id']

            if participant_id <6:
                win_id = match.iloc[match_idx]['radiant_win']
                win_ids[participant_id].append(str(win_id))
            else:
                win_id = ~(match.iloc[match_idx]['radiant_win'])
                win_ids[participant_id].append(str(win_id))

            if account_id == 0:
                account_ids[participant_id].append('UNK')
            else:
                account_ids[participant_id].append(account_id) 
            
            champion_ids[participant_id].append(champion_id)
            
            # Append Stat information


            stat_features = row[['gold','gold_spent','gold_per_min','xp_per_min','kills',\
                                'deaths','assists','denies','last_hits','stuns','hero_damage',\
                                'hero_healing','tower_damage','level','xp_hero','xp_creep','xp_roshan',\
                                'xp_other','gold_other','gold_death','gold_buyback','gold_abandon',\
                                'gold_sell','gold_destroying_structure','gold_killing_heros',\
                                'gold_killing_creeps','gold_killing_roshan']]
            stat_features.values[stat_features.values == 'None'] = np.NAN 

            stat_ids[participant_id].append(stat_features)
    
    user_cnt=0
    for participant_id in range(1, 11):
        if user_cnt <5:    
            data['User'+str(blue_team_order[user_cnt])+'_id'] = account_ids[participant_id]
            data['User'+str(blue_team_order[user_cnt])+'_win'] = win_ids[participant_id]
            data['User'+str(blue_team_order[user_cnt])+'_role'] = 'PAD'
            data['User'+str(blue_team_order[user_cnt])+'_champion'] = champion_ids[participant_id]
            data['User'+str(blue_team_order[user_cnt])+'_ban'] = 'PAD'
            data['User'+str(blue_team_order[user_cnt])+'_stat'] = stat_ids[participant_id]
        else:
            data['User'+str(red_team_order[user_cnt-5])+'_id'] = account_ids[participant_id]
            data['User'+str(red_team_order[user_cnt-5])+'_win'] = win_ids[participant_id]
            data['User'+str(red_team_order[user_cnt-5])+'_role'] = 'PAD'
            data['User'+str(red_team_order[user_cnt-5])+'_champion'] = champion_ids[participant_id]
            data['User'+str(red_team_order[user_cnt-5])+'_ban'] = 'PAD'
            data['User'+str(red_team_order[user_cnt-5])+'_stat'] = stat_ids[participant_id]

        if user_cnt <5:
            data['User'+str(blue_team_order[user_cnt])+'_team'] = 'BLUE'
        else:
            data['User'+str(red_team_order[user_cnt-5])+'_team'] = 'RED'
        
        user_cnt = user_cnt + 1
    return data




def build_dictionary(data, categorical_ids):
    user_to_idx = categorical_ids['user']
    champion_to_idx = categorical_ids['champion']
    
    for i in tqdm.tqdm(range(len(data))):
        row = data.iloc[i]
        for participant_id in range(1, 11):
            account_id = row['User'+str(participant_id)+'_id']
            champion_id = row['User'+str(participant_id)+'_champion']
            if account_id not in user_to_idx:
                user_to_idx[account_id] = len(user_to_idx)
        
            if champion_id not in champion_to_idx:
                champion_to_idx[champion_id] = len(champion_to_idx)

    categorical_ids['user'] = user_to_idx
    categorical_ids['champion'] = champion_to_idx
    
    return categorical_ids

def convert_unk_user(data, categorical_ids):
    user_to_idx = categorical_ids['user']
    max_user_idx = np.array(list(categorical_ids['user'].keys())[4:]).max()
    for i in tqdm.tqdm(range(len(data))):
        row = data.iloc[i]
        for participant_id in range(1, 11):
            account_id = row['User'+str(participant_id)+'_id']
            if account_id == 'UNK':
                max_user_idx = max_user_idx + 1
                data.at[i, 'User'+str(participant_id)+'_id']= max_user_idx
                user_to_idx[max_user_idx] = len(user_to_idx)

    categorical_ids['user'] = user_to_idx
    
    return data, categorical_ids

def convert_idx(dictionary, idx):
    if idx in dictionary:
        return dictionary[idx]
    else:
        return dictionary['UNK']
def convert_user_idx(dictionary, idx):
    if idx in dictionary:
        return dictionary[idx]
    else:
        import pdb
        pdb.set_trace()
        return dictionary['UNK']

def convert_match_data(data, categorical_ids):
    match_data = pd.DataFrame(index=data.index, columns=data.columns)

    win_to_idx = categorical_ids['win']
    user_to_idx = categorical_ids['user']
    champion_to_idx = categorical_ids['champion']
    role_to_idx = categorical_ids['role']
    team_to_idx = categorical_ids['team']
    
    for i in tqdm.tqdm(range(len(data))):
        match = data.iloc[i]
        # Match ID
        match_data.loc[i, 'match_id'] = match['match_id']
        
        # Time
        match_data.loc[i, 'time'] = match['time']
                
        for participant_id in range(1,11):
            account_id = match['User'+str(participant_id)+'_id']
            win = match['User'+str(participant_id)+'_win']
            role = match['User'+str(participant_id)+'_role']
            champion = match['User'+str(participant_id)+'_champion']
            ban = match['User'+str(participant_id)+'_ban']
            team = match['User'+str(participant_id)+'_team']
            stat = match['User'+str(participant_id)+'_stat']
            stat = tuple(np.nan_to_num(stat.values.astype('float')))
            if account_id == 'UNK':
                account_id = len(user_to_idx)
                

            match_data.loc[i, 'User'+str(participant_id)+'_id'] = convert_user_idx(user_to_idx, account_id)
            match_data.loc[i, 'User'+str(participant_id)+'_win'] = convert_idx(win_to_idx, win)
            match_data.loc[i, 'User'+str(participant_id)+'_role'] = convert_idx(role_to_idx, role)
            match_data.loc[i, 'User'+str(participant_id)+'_champion'] = convert_idx(champion_to_idx, champion)
            match_data.loc[i, 'User'+str(participant_id)+'_ban'] = convert_idx(champion_to_idx, ban)
            match_data.loc[i, 'User'+str(participant_id)+'_team'] = convert_idx(team_to_idx,team)
            match_data.loc[i, 'User'+str(participant_id)+'_stat'] = stat
            
    return match_data

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='arguments for preprocessing')
    parser.add_argument('--data_dir', type=str, default='/home/nas1_userC/hojoonlee/draftrec/dota/')
    args = parser.parse_args()
    
    match_data = pd.read_csv(args.data_dir + 'match.csv', index_col=0)
    player_data = pd.read_csv(args.data_dir + 'players.csv', index_col=0)

    # 1. Build match_dataframe
    print('[1. Start building match_dataframe]')
    match_dataframe = build_match_dataframe(match_data, player_data)
    train_data = match_dataframe.iloc[:42500].reset_index(drop=True)
    val_data = match_dataframe.iloc[42500:45000].reset_index(drop=True)
    test_data = match_dataframe.iloc[45000:].reset_index(drop=True)

    print(train_data.shape)
    print(val_data.shape)
    print(test_data.shape)
    
    print('[2. Start building dictionary]')
    categorical_ids = init_dictionary()
    categorical_ids = build_dictionary(train_data, categorical_ids)
    import pdb
    pdb.set_trace()
    print('[3. Start Converting Unk users within built match_dataframe]')
    train_data, categorical_ids = convert_unk_user(train_data, categorical_ids)
    val_data, categorical_ids = convert_unk_user(val_data, categorical_ids)
    test_data, categorical_ids = convert_unk_user(test_data, categorical_ids)

    print('[4. Start Indexing the match_dataframe according to dictionary]')
    match_data = {}
    match_data['train'] = convert_match_data(train_data, categorical_ids)
    match_data['val'] = convert_match_data(val_data, categorical_ids)
    match_data['test'] = convert_match_data(test_data, categorical_ids)

    with open('./dota_categorical_ids.pickle', 'wb') as f:
        pickle.dump(categorical_ids, f)

    with open('./dota_match_dataframe.pickle', 'wb') as f:
        pickle.dump(match_data, f)
    #match_dataframe.to_csv('lol_match_dataframe.csv')
    print('Finish creating match data')
