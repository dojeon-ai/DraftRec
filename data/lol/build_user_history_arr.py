import os
import sys
import pickle
import tqdm
import numpy as np


def main():
    # Dataset
    print('[Start loading the dataset]')
    with open('/home/dongyoonhwang/draftRec/data/lol_user_history_data.pickle', 'rb') as f:
        user_history_dict = pickle.load(f)
    print('[Finish loading the dataset]')
    
    num_users = 62466
    max_history_len = 1145
    num_features = 57
    
    user_history_array = np.zeros((num_users, max_history_len, num_features), dtype=np.float32)
    user_id_to_array_idx = {}
    feature_to_array_idx = {
        'champion': (0, 1),
        'role': (1, 2),
        'team': (2, 3),
        'ban': (3, 13),
        'win': (13, 14),
        'stat': (14, 57)
    }
    
    user_idx = 0
    for user_id, user_histories in tqdm.tqdm(user_history_dict.items()):
        user_id_to_array_idx[user_id] = user_idx
        for history_idx, user_history in enumerate(user_histories):
            champion = user_history['champion']
            role = user_history['role']
            team = user_history['team']
            ban = user_history['ban']
            win = user_history['win']
            stat = user_history['stat']
            
            user_history_array[user_idx][history_idx][0:1] = champion
            user_history_array[user_idx][history_idx][1:2] = role
            user_history_array[user_idx][history_idx][2:3] = team
            user_history_array[user_idx][history_idx][3:13] = ban
            user_history_array[user_idx][history_idx][13:14] = win
            user_history_array[user_idx][history_idx][14:57] = stat
        
        user_idx += 1
    
    with open('feature_to_array_idx.pickle', 'wb') as f:
        pickle.dump(feature_to_array_idx, f)
        
    with open('user_id_to_array_idx.pickle', 'wb') as f:
        pickle.dump(user_id_to_array_idx, f)
        
    np.save('user_history_array', user_history_array)

    
if __name__ == "__main__":
    main()



