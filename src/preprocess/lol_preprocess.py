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


def extract_matches_from_raw_data(raw_data, creationtime):
    # Extract the unique matches
    raw_data = raw_data.drop_duplicates()
    # Only consider the rank games
    raw_data = raw_data[(raw_data['queueId'] == 420)]
    # Extract the necessary features
    raw_data = raw_data[['gameId', 'gameVersion', 'gameCreation', 'gameDuration', 'teams', 'participants', 'participantIdentities']]
    raw_data = raw_data[raw_data['gameCreation'] >= creationtime].reset_index(drop=True)
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

    # role dictionary
    role_to_idx = {}
    role_to_idx['PAD'] = 0
    role_to_idx['MASK'] = 1
    role_to_idx['CLS'] = 2
    role_to_idx['UNK'] = 3

    #TODO: Change to an ambiguous term ==> 어차피 롤 특화 정보인데 솔직히 건드리고 싶지 않음 난!
    role_to_idx['TOP'] = 4
    role_to_idx['JUNGLE'] = 5
    role_to_idx['MIDDLE'] = 6
    role_to_idx['DUO_CARRY'] = 7
    role_to_idx['DUO_SUPPORT'] = 8
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
    win_to_idx['Fail'] = 4  # Blue-team lose
    win_to_idx['Win'] = 5  # Blue-team win

    categorical_ids['user'] = user_to_idx
    categorical_ids['champion'] = champion_to_idx
    categorical_ids['role'] = role_to_idx
    categorical_ids['team'] = team_to_idx
    categorical_ids['win'] = win_to_idx
    return categorical_ids


def append_userid_to_dictionary(data, categorical_ids):

    # Append user dictionary
    user_to_idx = categorical_ids['user']
    idx_to_user = {}
    for i in range(len(data)):
        participantIdentities = ast.literal_eval(data['participantIdentities'][i])
        for participant in participantIdentities:
            accountId = participant['player']['summonerName']
            if accountId not in user_to_idx:
                user_to_idx[accountId] = len(user_to_idx)
                idx_to_user[len(user_to_idx)-1] = accountId

    categorical_ids['user'] = user_to_idx
    categorical_ids['idx2user'] = idx_to_user

    return categorical_ids

def append_itemid_to_dictionary(data, categorical_ids):

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

    categorical_ids['champion'] = champion_to_idx

    return categorical_ids

def create_match_dataframe(data, categorical_ids, columns):
    user_to_idx = categorical_ids['user']
    champion_to_idx = categorical_ids['champion']
    role_to_idx = categorical_ids['role']
    team_to_idx = categorical_ids['team']
    win_to_idx = categorical_ids['win']

    # Initialize dataframe
    match_dataframe = pd.DataFrame(index=data.index, columns=columns)
    
    # Fill the values in match_dataframe
    for i in range(len(data)):
        match = data.iloc[i]

        # Match_Id
        match_dataframe.loc[i, 'match_id'] = match['gameId']

        # Win
        teams = ast.literal_eval(match['teams'])
        blue_team = teams[0]
        red_team = teams[1]

        blue_team_order = [1,4,5,8,9]
        red_team_order= [2,3,6,7,10]

        # Time
        match_dataframe.loc[i, 'time'] = match['gameCreation']
        if match['gameDuration'] != 0:
            duration = match['gameDuration']
        else:
            duration = 1

        # User Id
        user_cnt = 0
        participantIdentities = ast.literal_eval(match['participantIdentities'])
        for participant in participantIdentities:
            accountId = participant['player']['summonerName']
            if user_cnt<5:
                if accountId in user_to_idx:
                    match_dataframe.loc[i, 'User'+str(blue_team_order[user_cnt])+'_id'] = user_to_idx[accountId]
                else:
                    match_dataframe.loc[i, 'User'+str(blue_team_order[user_cnt])+'_id'] = user_to_idx['UNK']
            else:
                if accountId in user_to_idx:
                    match_dataframe.loc[i, 'User'+str(red_team_order[user_cnt-5])+'_id'] = user_to_idx[accountId]
                else:
                    match_dataframe.loc[i, 'User'+str(red_team_order[user_cnt-5])+'_id'] = user_to_idx['UNK']
            user_cnt += 1

        # Win & Team & Ban
        user_cnt = 0
        for ban in blue_team['bans']:
            match_dataframe.loc[i, 'User' + str(blue_team_order[user_cnt]) + '_win'] = win_to_idx[blue_team['win']]
            match_dataframe.loc[i, 'User' + str(blue_team_order[user_cnt]) + '_team'] = team_to_idx['BLUE']
            if ban['championId'] in champion_to_idx:
                match_dataframe.loc[i, 'User'+str(blue_team_order[user_cnt])+'_ban'] = champion_to_idx[ban['championId']]
            else:
                match_dataframe.loc[i, 'User'+str(blue_team_order[user_cnt])+'_ban'] = champion_to_idx['UNK']
            user_cnt += 1

        user_cnt=0
        for ban in red_team['bans']:
            match_dataframe.loc[i, 'User' + str(red_team_order[user_cnt]) + '_win'] = win_to_idx[red_team['win']]
            match_dataframe.loc[i, 'User' + str(red_team_order[user_cnt]) + '_team'] = team_to_idx['RED']
            if ban['championId'] in champion_to_idx:
                match_dataframe.loc[i, 'User'+str(red_team_order[user_cnt])+'_ban'] = champion_to_idx[ban['championId']]
            else:
                match_dataframe.loc[i, 'User'+str(red_team_order[user_cnt])+'_ban'] = champion_to_idx['UNK']
            user_cnt += 1
            
        # Champion & role
        user_cnt = 0
        participants = ast.literal_eval(match['participants'])
        for participant in participants:
            championId = participant['championId']
            if user_cnt<5:
                if championId in champion_to_idx:
                    match_dataframe.loc[i, 'User'+str(blue_team_order[user_cnt])+'_champion'] = champion_to_idx[championId]
                else:
                    match_dataframe.loc[i, 'User'+str(blue_team_order[user_cnt])+'_champion'] = champion_to_idx['UNK']
                    
                timeline = participant['timeline']
                role = timeline['role']
                lane = timeline['lane']
                if lane in role_to_idx:
                    match_dataframe.loc[i, 'User'+str(blue_team_order[user_cnt])+'_role'] = role_to_idx[lane]
                elif role in role_to_idx:
                    match_dataframe.loc[i, 'User'+str(blue_team_order[user_cnt])+'_role'] = role_to_idx[role]
                else:
                    match_dataframe.loc[i, 'User'+str(blue_team_order[user_cnt])+'_role'] = role_to_idx['UNK']

            else:
                if championId in champion_to_idx:
                    match_dataframe.loc[i, 'User'+str(red_team_order[user_cnt-5])+'_champion'] = champion_to_idx[championId]
                else:
                    match_dataframe.loc[i, 'User'+str(red_team_order[user_cnt-5])+'_champion'] = champion_to_idx['UNK']
                    
                timeline = participant['timeline']
                role = timeline['role']
                lane = timeline['lane']
                if lane in role_to_idx:
                    match_dataframe.loc[i, 'User'+str(red_team_order[user_cnt-5])+'_role'] = role_to_idx[lane]
                elif role in role_to_idx:
                    match_dataframe.loc[i, 'User'+str(red_team_order[user_cnt-5])+'_role'] = role_to_idx[role]
                else:
                    match_dataframe.loc[i, 'User'+str(red_team_order[user_cnt-5])+'_role'] = role_to_idx['UNK']

            user_cnt += 1

        # Stats
        user_cnt = 0
        participants = ast.literal_eval(match['participants'])
        for participant in participants:
            stats = participant['stats']
            kda = (stats['kills'] + stats['assists']) / (stats['deaths'] + 1.0)
            largestkillingspree = stats['largestKillingSpree']
            largestmultikill = stats['largestMultiKill']
            killingsprees = stats['killingSprees']
            longesttimespentliving = stats['longestTimeSpentLiving']
            doublekills = stats['doubleKills']
            triplekills = stats['tripleKills']
            quadrakills = stats['quadraKills']
            pentakills = stats['pentaKills']
            unrealkills = stats['unrealKills']
            totaldamagedealt = stats['totalDamageDealt'] / duration
            magicdamagedealt = stats['magicDamageDealt'] / duration
            physicaldamagedealt = stats['physicalDamageDealt'] / duration
            truedamagedealt = stats['trueDamageDealt'] / duration
            largestcriticalstrike = stats['largestCriticalStrike']
            totaldamagedealttochampions = stats['totalDamageDealtToChampions'] / duration
            magicdamagedealttochampions = stats['magicDamageDealtToChampions'] / duration
            physicaldamagedealttochampions = stats['physicalDamageDealtToChampions'] / duration
            truedamagedealttochampions = stats['trueDamageDealtToChampions'] / duration
            totalheal = stats['totalHeal'] / duration
            totalunitshealed = stats['totalUnitsHealed']
            damageselfmitigated = stats['damageSelfMitigated'] / duration
            damagedealttoobjectives = stats['damageDealtToObjectives'] / duration
            damagedealttoturrets = stats['damageDealtToTurrets']
            visionscore = stats['visionScore']  / duration
            timeccingothers = stats['timeCCingOthers']  / duration
            totaldamagetaken = stats['totalDamageTaken'] / duration
            magicaldamagetaken = stats['magicalDamageTaken']  / duration
            physicaldamagetaken = stats['physicalDamageTaken']  / duration
            truedamagetaken = stats['trueDamageTaken'] / duration
            goldearned = stats['goldEarned'] / duration
            goldspent = stats['goldSpent'] / duration
            turretkills = stats['turretKills']
            inhibitorkills = stats['inhibitorKills']
            totalminionskilled = stats['totalMinionsKilled'] / duration
            neutralminionskilled = stats['neutralMinionsKilled'] / duration
            neutralminionskilledteamjungle = stats['neutralMinionsKilledTeamJungle'] / duration
            neutralminionskilledenemyjungle = stats['neutralMinionsKilledEnemyJungle'] / duration
            totaltimecrowdcontroldealt = stats['totalTimeCrowdControlDealt'] / duration
            visionwardsboughtingame = stats['visionWardsBoughtInGame'] / duration
            sightwardsboughtingame = stats['sightWardsBoughtInGame'] / duration
            wardsplaced = stats['wardsPlaced'] / duration
            wardskilled = stats['wardsKilled'] / duration
            #champlevel = stats['champLevel']
            #firstbloodkill = stats['firstBloodKill']
            #firstbloodassist = stats['firstBloodAssist']
            #firsttowerkill = stats['firstTowerKill']
            #firsttowerassist = stats['firstTowerAssist']
            #gold_per_sec = stats['goldEarned'] / duration
            #minion_per_sec = stats['totalMinionsKilled'] / duration
            #enemy_jungle_per_sec = stats['neutralMinionsKilledEnemyJungle'] / duration
            #damage_dealt_per_sec = stats['totalDamageDealt'] / (duration * 100)
            #damage_taken_per_sec = stats['totalDamageTaken'] / (duration * 100)
            #vision_score = stats['visionScore']
            stat = tuple((kda, 
                    largestkillingspree,
                    largestmultikill,
                    killingsprees,
                    longesttimespentliving,
                    doublekills,
                    triplekills,
                    quadrakills,
                    pentakills,
                    unrealkills,
                    totaldamagedealt,
                    magicdamagedealt,
                    physicaldamagedealt,
                    truedamagedealt,
                    largestcriticalstrike,
                    totaldamagedealttochampions,
                    magicdamagedealttochampions,
                    physicaldamagedealttochampions,
                    truedamagedealttochampions,
                    totalheal,
                    totalunitshealed,
                    damageselfmitigated,
                    damagedealttoobjectives,
                    damagedealttoturrets,
                    visionscore,
                    timeccingothers,
                    totaldamagetaken,
                    magicaldamagetaken,
                    physicaldamagetaken,
                    truedamagetaken,
                    goldearned,
                    goldspent,
                    turretkills,
                    inhibitorkills,
                    totalminionskilled,
                    neutralminionskilled,
                    neutralminionskilledteamjungle,
                    neutralminionskilledenemyjungle,
                    totaltimecrowdcontroldealt,
                    visionwardsboughtingame,
                    sightwardsboughtingame,
                    wardsplaced,
                    wardskilled))
            if user_cnt<5:
                match_dataframe.at[i, 'User'+str(blue_team_order[user_cnt])+'_stat'] = stat
            else:
                match_dataframe.at[i, 'User'+str(red_team_order[user_cnt-5])+'_stat'] = stat
            user_cnt += 1
    
    return match_dataframe

#TODO: 함수들 간에는 두줄 띄기~!~!@~!@~!@~!@~!@
def create_champ_role_matrix(dataset, num_participants):
    train_champion_tmp = []
    train_role_tmp = []

    for participant_id in range(1, num_participants + 1):
        train_champion_tmp.append(dataset['User' + str(participant_id) + '_champion'].to_numpy())
        train_role_tmp.append(dataset['User' + str(participant_id) + '_role'].to_numpy())

    # Get Each Champion's role count
    train_champion = train_champion_tmp[0]
    train_role = train_role_tmp[0]

    for i in range(len(train_champion_tmp) - 1):
        train_champion = np.append(train_champion, train_champion_tmp[i + 1])
        train_role = np.append(train_role, train_role_tmp[i + 1])
    return train_champion, train_role


def create_ratio_dict(train_champion, train_role):
    train_champion_role_dict = {}
    for i in range(train_champion.shape[0]):
        champ_idx = train_champion[i]
        role_idx = train_role[i]
        if champ_idx not in train_champion_role_dict:
            train_champion_role_dict[champ_idx] = {}
            for i in range(len(categorical_ids['role'])):
                train_champion_role_dict[champ_idx][i] = 0
        train_champion_role_dict[champ_idx][role_idx] += 1
    for i in train_champion_role_dict.keys():
        sum = 0
        for j in range(len(train_champion_role_dict[i])):
            sum += train_champion_role_dict[i][j]
        train_champion_role_dict[i]['sum'] = sum
        for j in range(len(train_champion_role_dict[i]) - 1):
            train_champion_role_dict[i][j] = train_champion_role_dict[i][j] / sum
    return train_champion_role_dict

#TODO: 이제 USer의 순서가 픽순으로 바뀌는 바람에 해당 코드에 대수정이 필요!!! 
# 원래 코드는 (블루팀, 레드팀) 으로 나뉜 거스올 생각했었던것 ㅠㅠ
def fill_NA_roles(dataset, train_champion_role_dict, categorical_ids):
    role_df = dataset.filter(regex='role')
    champ_df = dataset.filter(regex='champ')

    blue_team_order = [0,3,4,7,8]
    red_team_order = [1,2,5,6,9]

    team1_role_champ_df = pd.concat([role_df.iloc[:, blue_team_order], champ_df.iloc[:, :5]], axis=1)
    team1_role_champ_np = team1_role_champ_df.to_numpy()
    
    team2_role_champ_df = pd.concat([role_df.iloc[:, red_team_order], champ_df.iloc[:, 5:]], axis=1)
    team2_role_champ_np = team2_role_champ_df.to_numpy()

    team1_roleNA_rows, team2_roleNA_rows = get_missing_rows(team1_role_champ_np, team2_role_champ_np, categorical_ids)

    # consider only the problematic rows for computational ease
    team1_prob_df = team1_role_champ_df.iloc[team1_roleNA_rows]
    team1_prob_np = team1_prob_df.to_numpy()
    team2_prob_df = team2_role_champ_df.iloc[team2_roleNA_rows]
    team2_prob_np = team2_prob_df.to_numpy()


    team1_filled_rows = fill_missing_rows(team1_prob_np, team1_prob_df, train_champion_role_dict, categorical_ids)
    team2_filled_rows = fill_missing_rows(team2_prob_np, team2_prob_df, train_champion_role_dict, categorical_ids)

    team1_role_champ_df.iloc[team1_roleNA_rows] = team1_filled_rows
    team2_role_champ_df.iloc[team2_roleNA_rows] = team2_filled_rows

    # Modify original dataset with most probable role permutation for missing role values
    dataset[team1_role_champ_df.columns[:5]] = team1_role_champ_df[team1_role_champ_df.columns[:5]]
    dataset[team2_role_champ_df.columns[:5]] = team2_role_champ_df[team2_role_champ_df.columns[:5]]

    return dataset

def get_missing_rows(team1_role_champ_np, team2_role_champ_np, categorical_ids):

    role_to_idx = categorical_ids['role']

    # Get the problematic rows for each team individually
    team1_roleNA_rows = []
    for i in range(team1_role_champ_np.shape[0]):
        team1_role_set = set([role_to_idx['TOP'], role_to_idx['JUNGLE'], role_to_idx['MIDDLE'], role_to_idx['DUO_CARRY'], role_to_idx['DUO_SUPPORT']])
        for j in team1_role_champ_np[i, :5]:
            if j in team1_role_set:
                team1_role_set.remove(j)
        if len(team1_role_set) != 0:
            team1_roleNA_rows.append(i)

    team2_roleNA_rows = []
    for i in range(team2_role_champ_np.shape[0]):
        team2_role_set = set([role_to_idx['TOP'], role_to_idx['JUNGLE'], role_to_idx['MIDDLE'], role_to_idx['DUO_CARRY'], role_to_idx['DUO_SUPPORT']])
        for j in team2_role_champ_np[i, :5]:
            if j in team2_role_set:
                team2_role_set.remove(j)
        if len(team2_role_set) != 0:
            team2_roleNA_rows.append(i)
    return team1_roleNA_rows, team2_roleNA_rows

def fill_missing_rows(team_prob_np, team_prob_df, train_champion_role_dict, categorical_ids):

    role_to_idx = categorical_ids['role']
    UNK = role_to_idx['UNK']

    for i in range(team_prob_np.shape[0]):
        unique_vals_tmp, unique_idx, unique_counts = np.unique(team_prob_np[i, :5], False, True, True)

        duplicate_vals = set(unique_vals_tmp[np.where(unique_counts > 1)])

        # minus set(3) since 3==Unk token, and may look unique!
        unique_vals = set(unique_vals_tmp[np.where(unique_counts == 1)])

        # Initialize problematic_idx with idx of Unk tokens, if there are any
        problematic_idx = []

        if UNK in unique_vals:
            problematic_idx.append(np.where(team_prob_np[i, :5] == UNK))
            if UNK in duplicate_vals:
                duplicate_vals.remove(UNK)
        for k in range(len(duplicate_vals)):
            problematic_idx.append(np.where(unique_idx == np.where(unique_counts > 1)[0][k]))

        # Now need to know what roles are missing!
        # We can't know this by only from the duplicate vals since That role itself may not exist at all
        missing_roles = set([role_to_idx['TOP'], role_to_idx['JUNGLE'], role_to_idx['MIDDLE'], role_to_idx['DUO_CARRY'], role_to_idx['DUO_SUPPORT']]) - unique_vals
        missing_idx = np.concatenate(problematic_idx, axis=1).squeeze()

        # Now we know which roles and which users are problematic
        missing_chmp = team_prob_np[i, 5:][missing_idx]

        probs = []
        combinations = {}
        for idx, cmb in enumerate(pm(missing_roles)):
            combinations[idx] = cmb
            prob = 1
            for champ_idx, role_idx in enumerate(cmb):
                prob = prob * (train_champion_role_dict[missing_chmp[champ_idx]][role_idx] + 1e-4)
            probs.append(prob)
        probs = np.array(probs)
        max_prob_idx = probs.argmax()
        team_prob_df.iloc[i, missing_idx] = combinations[max_prob_idx]

    return team_prob_df

def delete_UNK_champion_rows(match_dataframe, categorical_ids):
    champ_columns = match_dataframe.filter(regex='champion').columns
    champion_to_idx = categorical_ids['champion']
    match_dataframe[champ_columns] = match_dataframe[champ_columns].replace(champion_to_idx['UNK'], np.NaN)
    match_dataframe = match_dataframe.dropna().reset_index(drop=True)
    return match_dataframe

def checksum_NA_role_values(dataframe):
    role_df = dataframe.filter(regex='role')
    checksum = 0
    check_set = set((4, 5, 6, 7, 8))

    blue_team_order = [0,3,4,7,8]
    for idx in range(len(dataframe)):
        for col in role_df.columns[blue_team_order]:
            check_set = check_set - {dataframe.iloc[idx][col]}
        if len(check_set) != 0:
            checksum += 1

    check_set = set((4, 5, 6, 7, 8))

    red_team_order = [1,2,5,6,9]
    for idx in range(len(dataframe)):
        for col in role_df.columns[blue_team_order]:
            check_set = check_set - {dataframe.iloc[idx][col]}
        if len(check_set) != 0:
            checksum += 1

    return checksum


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='arguments for preprocessing')
    parser.add_argument('--data_dir', type=str, default='/home/nas1_userC/hojoonlee/draftrec/210917/')
    parser.add_argument('--data_start_time', type=float, default=1622505600000) # 21.06.01~21.09.09
    parser.add_argument('--train_end_time', type=float, default=1630162800000) # train:21.06.01~21.08.28
    parser.add_argument('--val_end_time', type=float, default=1630458000000)   # val: 21.08.28 ~ 21.09.01
    parser.add_argument('--test_end_time', type=float, default=1631152598612)  # test: 21.09.01 ~ 21.09.09
    args = parser.parse_args()
    file_list = glob.glob(args.data_dir + '*.csv')

    # 1. Build dictionary
    print('[1. Start building dictionary]')
    categorical_ids = init_dictionary()
    for idx, file in enumerate(file_list):
        print('File index:', idx)
        # 1.1 Preprocess raw_data
        raw_data = pd.read_csv(file, index_col=0).reset_index(drop=True)
        raw_match_data = extract_matches_from_raw_data(raw_data, args.data_start_time)
        print('Num data:', len(raw_match_data))
        del raw_data
        # 1.2 Split into train & test data

        train_data = raw_match_data[(raw_match_data['gameCreation'] < args.train_end_time)].reset_index(drop=True)
        val_data = raw_match_data[(raw_match_data['gameCreation'] >= args.train_end_time)
                                    &((raw_match_data['gameCreation'] < args.val_end_time))].reset_index(drop=True)
        test_data = raw_match_data[(raw_match_data['gameCreation'] > args.val_end_time)
                                    &((raw_match_data['gameCreation'] < args.test_end_time))].reset_index(drop=True)
        print('Num train data:', len(train_data))
        print('Num val data:', len(val_data))
        print('Num test data:', len(test_data))
        del raw_match_data

        # 1.3 Append dictionary for designated file
        categorical_ids = append_userid_to_dictionary(train_data, categorical_ids)
        categorical_ids = append_userid_to_dictionary(val_data, categorical_ids)
        categorical_ids = append_userid_to_dictionary(test_data, categorical_ids)

        categorical_ids = append_itemid_to_dictionary(train_data, categorical_ids)
        del train_data
        del val_data
        del test_data

    # 2.a) Create match data
    print('[2.1. Start creating match data]')
    
    num_participants = 10

    columns = ['match_id', 'time']
    #TODO(완료): User입장에서 각자의 승패 정보 집어넣기.
    for participant_id in range(1,num_participants+1):
        columns.append('User'+str(participant_id)+'_id')
        columns.append('User'+str(participant_id)+'_win')
        columns.append('User'+str(participant_id)+'_role') 
        columns.append('User'+str(participant_id)+'_champion')
        columns.append('User'+str(participant_id)+'_ban') 
        columns.append('User'+str(participant_id)+'_team') 
        columns.append('User'+str(participant_id)+'_stat')
    train_match_dataframe = pd.DataFrame(columns=columns)
    val_match_dataframe = pd.DataFrame(columns=columns)
    test_match_dataframe = pd.DataFrame(columns=columns)

    for idx, file in enumerate(file_list):
        print('File index:', idx)
        raw_data = pd.read_csv(file, index_col=0).reset_index(drop=True)
        raw_match_data = extract_matches_from_raw_data(raw_data, args.data_start_time)
        del raw_data

        train_data = raw_match_data[(raw_match_data['gameCreation'] < args.train_end_time)].reset_index(drop=True)
        val_data = raw_match_data[(raw_match_data['gameCreation'] >= args.train_end_time)
                                    &((raw_match_data['gameCreation'] < args.val_end_time))].reset_index(drop=True)
        test_data = raw_match_data[(raw_match_data['gameCreation'] > args.val_end_time)
                                    &((raw_match_data['gameCreation'] < args.test_end_time))].reset_index(drop=True)
        del raw_match_data

        current_file_train_match_dataframe = create_match_dataframe(train_data, categorical_ids, columns)
        current_file_val_match_dataframe = create_match_dataframe(val_data, categorical_ids, columns)
        current_file_test_match_dataframe = create_match_dataframe(test_data, categorical_ids, columns)

        train_match_dataframe = pd.concat([train_match_dataframe, current_file_train_match_dataframe], ignore_index=True, sort=False)
        val_match_dataframe = pd.concat([val_match_dataframe, current_file_val_match_dataframe], ignore_index=True, sort=False)
        test_match_dataframe = pd.concat([test_match_dataframe, current_file_test_match_dataframe], ignore_index=True, sort=False)

    match_dataframe = {}
    match_dataframe['train'] = train_match_dataframe.drop_duplicates().reset_index(drop=True)
    match_dataframe['val'] = val_match_dataframe.drop_duplicates().reset_index(drop=True)
    match_dataframe['test'] = test_match_dataframe.drop_duplicates().reset_index(drop=True)

    # Drop rows where champ_idx is UNK=3 -> This case only appears in val & test set since there exist no UNK in train
    match_dataframe['val'] = delete_UNK_champion_rows(match_dataframe['val'], categorical_ids)
    match_dataframe['test'] = delete_UNK_champion_rows(match_dataframe['test'], categorical_ids)

    # 2.b) Fill in missing role values with most probable permutation.
    print("2.2. Start filling missing roles")
    train_champion, train_role = create_champ_role_matrix(match_dataframe['train'], num_participants)
    train_champion_role_dict = create_ratio_dict(train_champion, train_role)
    del train_champion, train_role

    match_dataframe['train'] = fill_NA_roles(match_dataframe['train'], train_champion_role_dict, categorical_ids)
    match_dataframe['val'] = fill_NA_roles(match_dataframe['val'], train_champion_role_dict, categorical_ids)
    match_dataframe['test'] = fill_NA_roles(match_dataframe['test'], train_champion_role_dict, categorical_ids)

    # Checksum to see if there are any reisdual errors within the role-info
    train_checksum = checksum_NA_role_values(match_dataframe['train'])
    val_checksum = checksum_NA_role_values(match_dataframe['val'])
    test_checksum = checksum_NA_role_values(match_dataframe['test'])

    assert train_checksum == 0
    assert val_checksum == 0
    assert test_checksum == 0

    # TODO(완료): User들의 순서를 픽순으로 바꾸기 
    print('Num train data:', match_dataframe['train'])
    print('Num val data:', match_dataframe['val'])
    print('Num test data:', match_dataframe['test'])

    with open('./lol_categorical_ids.pickle', 'wb') as f:
        pickle.dump(categorical_ids, f)

    with open('./lol_match_dataframe.pickle', 'wb') as f:
        pickle.dump(match_dataframe, f)
    #match_dataframe.to_csv('lol_match_dataframe.csv')
    print('Finish creating match data')
