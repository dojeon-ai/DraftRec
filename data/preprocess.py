import pickle
import glob
import pandas as pd
import numpy as np
import json
import re
import time
import ast
import argparse
from itertools import permutations as pm


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
    # Version dictionary
    version_to_idx = {}
    version_to_idx['PAD'] = 0
    version_to_idx['MASK'] = 1
    version_to_idx['CLS'] = 2
    version_to_idx['UNK'] = 3
    # Lane dictionary
    lane_to_idx = {'PAD': 0,
                   'MASK': 1,
                   'CLS': 2,
                   'UNK': 3,
                   'TOP': 4,
                   'JUNGLE': 5,
                   'MIDDLE': 6,
                   'DUO_CARRY': 7,
                   'DUO_SUPPORT': 8}
    # Team dictionary
    team_to_idx = {'PAD': 0,
                   'MASK': 1,
                   'CLS': 2,
                   'UNK': 3,
                   'BLUE': 4,
                   'RED': 5}
    categorical_ids['user'] = user_to_idx
    categorical_ids['interaction_per_user'] = {}
    categorical_ids['champion'] = champion_to_idx
    categorical_ids['version'] = version_to_idx
    categorical_ids['lane'] = lane_to_idx
    categorical_ids['team'] = team_to_idx
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
    team_to_idx = categorical_ids['team']

    # Initialize dataframe
    match_data = pd.DataFrame(index=data.index, columns=columns)
    
    # Fill the values in match_data
    for i in range(len(data)):
        match = data.iloc[i]
        # Creation
        match_data.loc[i, 'time'] = match['gameCreation']

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
            match_data.loc[i, 'User' + str(user_cnt + 1) + '_team'] = team_to_idx['BLUE']
            if ban['championId'] in champion_to_idx:
                match_data.loc[i, 'User'+str(user_cnt+1)+'_ban'] = champion_to_idx[ban['championId']]
            else:
                match_data.loc[i, 'User'+str(user_cnt+1)+'_ban'] = champion_to_idx['UNK']
            user_cnt += 1
        for ban in red_team['bans']:
            match_data.loc[i, 'User' + str(user_cnt + 1) + '_team'] = team_to_idx['RED']
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


def create_user_history_data(args, match_data, categorical_ids):
    num_participants = 10
    user_to_idx = categorical_ids['user']
    # aggregate match-data
    match_data_df = pd.concat([match_data['train'], match_data['val'], match_data['test']]).reset_index(drop=True)
    for participant_id in range(1, num_participants + 1):
        match_data_df['User' + str(participant_id) + '_stat'] = 0

    # initialize user_to_match_history
    user_to_match_history = {}
    for user_id in range(len(user_to_idx)):
        user_to_match_history[user_id] = []

    # append stat history to user_to_match_history
    stat_id = 0
    for match_idx in range(len(match_data_df)):
        match = match_data_df.iloc[match_idx]
        win = int(match['win'] == 'Win')
        time = match['time']
        if time < args.train_end_time:
            data_type = 'train'
        elif args.train_end_time <= time < args.val_end_time:
            data_type = 'val'
        else:
            data_type = 'test'

        bans = []
        for p_idx in range(num_participants):
            banIdx = match['User' + str(p_idx + 1) + '_ban']
            bans.append(banIdx)

        for p_idx in range(num_participants):
            user = match['User' + str(p_idx + 1) + '_id']
            item = match['User' + str(p_idx + 1) + '_champion']
            lane = match['User' + str(p_idx + 1) + '_lane']

            user_stat_in_match = {}
            user_stat_in_match['id'] = stat_id
            user_stat_in_match['time'] = time
            user_stat_in_match['lane'] = lane
            user_stat_in_match['item'] = item
            user_stat_in_match['bans'] = bans
            user_stat_in_match['data_type'] = data_type

            # user belongs to blue-team so win should be identical
            if p_idx < 5:
                user_stat_in_match['win'] = win
            # user belongs to red-team so win should be reversed
            else:
                user_stat_in_match['win'] = 1 - win
            if user != user_to_idx['UNK']:
                user_to_match_history[user].append(user_stat_in_match)
                match_data_df.loc[match_idx, 'User' + str(p_idx + 1) + '_stat'] = stat_id
                stat_id += 1

    # re-order the matches
    for user, match_histories in user_to_match_history.items():
        user_to_match_history[user] = sorted(match_histories, key=lambda x: x['time'])

    # check whether the item is propery ordered in sequence
    for user, match_histories in user_to_match_history.items():
        cur_time = 0
        for match_history in match_histories:
            time = match_history['time']
            assert time > cur_time
            cur_time = time

    # return the match data with stat-id
    match_data['train'] = match_data_df[(match_data_df['time'] < args.train_end_time)].reset_index(drop=True)
    match_data['val'] = match_data_df[(args.train_end_time <= match_data_df['time']) &
                                      (match_data_df['time'] < args.val_end_time)].reset_index(drop=True)
    match_data['test'] = match_data_df[(match_data_df['time'] > args.val_end_time)].reset_index(drop=True)
    return match_data, user_to_match_history


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


def create_chmp_lane_ratio(dataset):
    def create_champ_lane_info(dataset):
        num_participants = 10
        train_champion_tmp = []
        train_lane_tmp = []

        for participant_id in range(1, num_participants + 1):
            train_champion_tmp.append(dataset['User' + str(participant_id) + '_champion'].to_numpy())
            train_lane_tmp.append(dataset['User' + str(participant_id) + '_lane'].to_numpy())

        # Get Each Champion's lane count
        train_champion = train_champion_tmp[0]
        train_lane = train_lane_tmp[0]

        for i in range(len(train_champion_tmp) - 1):
            train_champion = np.append(train_champion, train_champion_tmp[i + 1])
            train_lane = np.append(train_lane, train_lane_tmp[i + 1])
        return train_champion, train_lane

    train_champion, train_lane = create_champ_lane_info(dataset)

    def create_ratio_dict(train_champion, train_lane):
        train_champion_lane_dict = {}
        for i in range(train_champion.shape[0]):
            champ_idx = train_champion[i]
            lane_idx = train_lane[i]
            if champ_idx not in train_champion_lane_dict:
                train_champion_lane_dict[champ_idx] = {}
                for i in range(len(categorical_ids['lane'])):
                    train_champion_lane_dict[champ_idx][i] = 0
            train_champion_lane_dict[champ_idx][lane_idx] += 1
        for i in train_champion_lane_dict.keys():
            sum = 0
            for j in range(len(train_champion_lane_dict[i])):
                sum += train_champion_lane_dict[i][j]
            train_champion_lane_dict[i]['sum'] = sum
            for j in range(len(train_champion_lane_dict[i]) - 1):
                train_champion_lane_dict[i][j] = train_champion_lane_dict[i][j] / sum
        return train_champion_lane_dict

    train_champion_lane_dict = create_ratio_dict(train_champion, train_lane)
    del train_champion, train_lane
    return train_champion_lane_dict


def fill_missing_lane(dataset, train_champion_lane_dict):
    lane_df = dataset.filter(regex='lane')
    champ_df = dataset.filter(regex='champ')
    team1_lane_champ_df = pd.concat([lane_df.iloc[:, :5], champ_df.iloc[:, :5]], axis=1)
    team2_lane_champ_df = pd.concat([lane_df.iloc[:, 5:], champ_df.iloc[:, 5:]], axis=1)
    team1_lane_champ_np = team1_lane_champ_df.to_numpy()
    team2_lane_champ_np = team2_lane_champ_df.to_numpy()

    def get_missing_rows(team1_lane_champ_np, team2_lane_champ_np):

        # Get the problematic rows for each team individually
        team1_problematic_rows = []
        for i in range(team1_lane_champ_np.shape[0]):
            team1_lane_set = set((4, 5, 6, 7, 8))
            for j in team1_lane_champ_np[i, :5]:
                if j in team1_lane_set:
                    team1_lane_set.remove(j)
            if len(team1_lane_set) != 0:
                team1_problematic_rows.append(i)

        team2_problematic_rows = []
        for i in range(team2_lane_champ_np.shape[0]):
            team2_lane_set = set((4, 5, 6, 7, 8))
            for j in team2_lane_champ_np[i, :5]:
                if j in team2_lane_set:
                    team2_lane_set.remove(j)
            if len(team2_lane_set) != 0:
                team2_problematic_rows.append(i)
        return team1_problematic_rows, team2_problematic_rows

    team1_problematic_rows, team2_problematic_rows = get_missing_rows(team1_lane_champ_np, team2_lane_champ_np)

    # consider only the problematic rows for computational ease
    team1_prob_df = team1_lane_champ_df.iloc[team1_problematic_rows]
    team1_prob_np = team1_prob_df.to_numpy()
    team2_prob_df = team2_lane_champ_df.iloc[team2_problematic_rows]
    team2_prob_np = team2_prob_df.to_numpy()

    def fill_missing_rows(team_prob_np, team_prob_df, train_champion_lane_dict):
        for i in range(team_prob_np.shape[0]):
            unique_vals_tmp, unique_idx, unique_counts = np.unique(team_prob_np[i, :5], False, True, True)

            duplicate_vals = set(unique_vals_tmp[np.where(unique_counts > 1)])

            # minus set(3) since 3==Unk token, and may look unique!
            unique_vals = set(unique_vals_tmp[np.where(unique_counts == 1)])

            # Initialize problematic_idx with idx of Unk tokens, if there are any
            problematic_idx = []
            UNK = 3
            if UNK in unique_vals:
                problematic_idx.append(np.where(team_prob_np[i, :5] == UNK))
                if UNK in duplicate_vals:
                    duplicate_vals.remove(UNK)
            for k in range(len(duplicate_vals)):
                problematic_idx.append(np.where(unique_idx == np.where(unique_counts > 1)[0][k]))

            # Now need to know what lanes are missing!
            # We can't know this by only from the duplicate vals since That lane itself may not exist at all
            missing_lanes = set((4, 5, 6, 7, 8)) - unique_vals
            missing_idx = np.concatenate(problematic_idx, axis=1).squeeze()

            # Now we know which lanes and which users are problematic
            missing_chmp = team_prob_np[i, 5:][missing_idx]

            probs = []
            combinations = {}
            for idx, cmb in enumerate(pm(missing_lanes)):
                combinations[idx] = cmb
                prob = 1
                for champ_idx, lane_idx in enumerate(cmb):
                    prob = prob * (train_champion_lane_dict[missing_chmp[champ_idx]][lane_idx] + 1e-4)
                probs.append(prob)
            probs = np.array(probs)
            max_prob_idx = probs.argmax()
            team_prob_df.iloc[i, missing_idx] = combinations[max_prob_idx]

        return team_prob_df
    team1_filled_df = fill_missing_rows(team1_prob_np, team1_prob_df, train_champion_lane_dict)
    team2_filled_df = fill_missing_rows(team2_prob_np, team2_prob_df, train_champion_lane_dict)

    team1_lane_champ_df.iloc[team1_problematic_rows] = team1_filled_df
    team2_lane_champ_df.iloc[team2_problematic_rows] = team2_filled_df

    # Modify original dataset with most probable lane permutation for missing lane values
    dataset[team1_lane_champ_df.columns[:5]] = team1_lane_champ_df[team1_lane_champ_df.columns[:5]]
    dataset[team2_lane_champ_df.columns[:5]] = team2_lane_champ_df[team2_lane_champ_df.columns[:5]]

    return dataset


def checksum_missing_values(dataset):
    lane_df = dataset.filter(regex='lane')
    checksum = 0
    check_set = set((4, 5, 6, 7, 8))

    for idx in range(len(dataset)):
        for col in lane_df.columns[:5]:
            check_set = check_set - {dataset.iloc[idx][col]}
        if len(check_set) != 0:
            checksum += 1
    return checksum


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='arguments for preprocessing')
    parser.add_argument('--data_dir', type=str, default='C:/Users/leehojoon/Desktop/projects/draftRec/data/')
    # Train: 20.11.12 ~ 21.02.09 / Val: 21.02.10 ~ 21.02.12 / Test: 21.02.13 ~ 21.02.15
    parser.add_argument('--train_end_time', type=float, default=1612882800000) # 21.02.10
    parser.add_argument('--val_end_time', type=float, default=1613142000000) # 21.02.13
    parser.add_argument('--interaction_threshold', type=int, default=20)
    args = parser.parse_args()
    file_list = glob.glob(args.data_dir + '*.csv')[10:15]

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
        #print('Num train data:', len(train_data))
        #print('Num val data:', len(val_data))
        #print('Num test data:', len(test_data))
        del preprocessed_data

        # 1.3 Append dictionary for designated file
        categorical_ids = append_item_to_dictionary(train_data, categorical_ids)
        #print('Finish appending dictionary')
        del train_data
        del val_data
        del test_data

    # 1.4 Only keep track of the users who have more than interaction-thershold with champions
    for user, num_interaction in categorical_ids['interaction_per_user'].items():
        if num_interaction >= args.interaction_threshold:
            categorical_ids['user'][user] = len(categorical_ids['user'])

    # 2.a) Create match data
    print('[2. Start creating match data]')
    num_participants = 10
    columns = ['win', 'version', 'time']
    for participant_id in range(1,num_participants+1):
        columns.append('User'+str(participant_id)+'_id')
        columns.append('User'+str(participant_id)+'_lane')
        columns.append('User'+str(participant_id)+'_champion')
        columns.append('User'+str(participant_id)+'_ban')
        columns.append('User'+str(participant_id)+'_team')
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

    # 2.b) Fill in missing lane values with most probable permutation.
    print("Start filling in missing lanes")
    train_champion_lane_dict = create_chmp_lane_ratio(match_data['train'])

    match_data['train'] = fill_missing_lane(match_data['train'], train_champion_lane_dict)
    match_data['val'] = fill_missing_lane(match_data['val'], train_champion_lane_dict)
    match_data['test'] = fill_missing_lane(match_data['test'], train_champion_lane_dict)

    # Checksum to see if there are any reisdua errors within the lane-info
    train_checksum = checksum_missing_values(match_data['train'])
    val_checksum = checksum_missing_values(match_data['val'])
    test_checksum = checksum_missing_values(match_data['test'])

    assert train_checksum == 0
    assert val_checksum == 0
    assert test_checksum == 0

    print('Num train data:', match_data['train'])
    print('Num val data:', match_data['val'])
    print('Num test data:', match_data['test'])
    print('Finish creating match data')

    # 3. Create user-history data
    match_data, user_history_data = create_user_history_data(args, match_data, categorical_ids)
    with open('./match_data.pickle', 'wb') as f:
        pickle.dump(match_data, f)
    with open('./user_history_data.pickle', 'wb') as f:
        pickle.dump(user_history_data, f)
    print('Finish creating user-history data')

    # 4. Create interaction data
    print('[4. Start creating interaction data]')
    interaction_data = {}
    interaction_data['train'] = create_interaction_data(match_data['train'], categorical_ids)
    interaction_data['val'] = create_interaction_data(match_data['val'], categorical_ids)
    interaction_data['test'] = create_interaction_data(match_data['test'], categorical_ids)
    with open('./interaction_data.pickle', 'wb') as f:
        pickle.dump(interaction_data, f)
    print('Finish creating interaction data')

    with open('categorical_ids.pickle', 'wb') as f:
        pickle.dump(categorical_ids, f)
