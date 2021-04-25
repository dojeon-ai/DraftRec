"""
def _create_statistical_label(self, match_data):
    statistical_dict = {4: {},
                        5: {},
                        6: {},
                        7: {},
                        8: {}}
    B = self.board_len
    team_ids = match_data[:, :B]
    item_ids = match_data[:, 3 * B:4 * B]
    lane_ids = match_data[:, 4 * B:5 * B]
    win_labels = match_data[:, 6 * B]

    for lane, _ in statistical_dict.items():
        matchups = item_ids[lane_ids == lane].reshape(-1, 2)
        teams = team_ids[lane_ids == lane].reshape(-1, 2)
        matchup_dict = defaultdict(list)
        for matchup, team, win in zip(matchups, teams, win_labels):
            chmp1 = matchup[0]
            chmp2 = matchup[1]
            chmp1_team = team[0]
            chmp2_team = team[1]
            BLUE = 4
            if (chmp1, chmp2) in matchup_dict:
                if chmp1_team == BLUE:
                    matchup_dict[(chmp1, chmp2)].append(win)
                    matchup_dict[chmp1].append(win)
                    matchup_dict[chmp2].append(1-win)
                else:
                    matchup_dict[(chmp1, chmp2)].append(1-win)
                    matchup_dict[chmp1].append(1-win)
                    matchup_dict[chmp2].append(win)
            elif (chmp2, chmp1) in matchup_dict:
                if chmp2_team == BLUE:
                    matchup_dict[(chmp2, chmp1)].append(win)
                    matchup_dict[chmp2].append(win)
                    matchup_dict[chmp1].append(1-win)
                else:
                    matchup_dict[(chmp2, chmp1)].append(1-win)
                    matchup_dict[chmp2].append(1-win)
                    matchup_dict[chmp1].append(win)
            else:
                if chmp1_team == BLUE:
                    matchup_dict[(chmp1, chmp2)] = [win]
                    matchup_dict[chmp1].append(win)
                    matchup_dict[chmp2].append(1-win)
                else:
                    matchup_dict[(chmp1, chmp2)] = [1-win]
                    matchup_dict[chmp1].append(1-win)
                    matchup_dict[chmp2].append(win)

        statistical_dict[lane] = matchup_dict

    return statistical_dict

def get_statistical_label(self, lane, chmp1, chmp2=None):
    reversed_key = False
    if chmp2 == None:
        stats = self.statistics_dict[lane][chmp1]
    else:
        if (chmp1, chmp2) in self.statistics_dict:
            stats = self.statistics_dict[lane][(chmp1, chmp2)]
        elif (chmp2, chmp1) in self.statistics_dict:
            stats = self.statistics_dict[lane][(chmp2, chmp1)]
            reversed_key = True
        else:
            raise AssertionError

    if len(stats) >= self.args.statistic_threshold:
        if reversed_key:
            return 1 - np.mean(stats)
        else:
            return np.mean(stats)
    else:
        return 0.5

def get_statistical_label_from_board(self, item_ids, lane_ids):
    pass
"""