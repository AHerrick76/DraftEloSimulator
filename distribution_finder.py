'''
Uses 17lands data to find winrate distribution among Magic players.

Note that this data only captures winrates among players who use the 17lands MTG Arena extension -
These players are typically more enfranchised than the average player, and this is reflected in
their average winrate, which generally hovers between 58-60%, when the overall winrate
must be 50%*. I use this data to estimate the standard deviation of winrate within the Magic
drafting community.

* This is not 100% true. If players who are more/less skilled play a disproportionately high number
of overall games, it's possible for the average player-specific winrate to be different than 50%.
'''

__author__ = 'Austin Herrick'


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import os
import pickle


def process_all_data():
	'''
	Process both bo1 and bo3 data
	'''

	# create output files for bo1 and bo3 data
	df_bo3 = parse_data('TradDraft', 'bo3')
	df_bo1 = parse_data('PremierDraft', 'bo1')


def parse_data(gameplay_str, output_type):
	'''
	Loads best of 3 data to find implied winrate distribution
	'''

	# load all datasets within the target folder type
	data_folder = os.listdir('Data')
	data_files = [x for x in data_folder if gameplay_str in x]

	# load each dataset, trimming columns and restricting to unique drafts
	set_df_holder = []
	for file in data_files:
		set_df = pd.read_csv(join('Data', file))
		set_df = trim_dataset(set_df)
		set_df_holder.append(set_df)
	
	# concatenate all datasets
	df = pd.concat(set_df_holder)

	# find distribution, weighted by bucket size
	summary_stats = {}
	summary_stats['OverallStandardDeviation'] = round(np.std(df.user_win_rate_bucket), 4)
	summary_stats['OverallWinRate'] = round(df.user_win_rate_bucket.mean(), 4)

	# find maximum & minimum winrates with > 100 observations
	df['BucketCount'] = df.groupby('user_win_rate_bucket').draft_id.transform('count')
	summary_stats['MinWinPercentage'] = df[df.BucketCount >= 50].user_win_rate_bucket.min()
	summary_stats['MaxWinPercentage'] = df[df.BucketCount >= 50].user_win_rate_bucket.max()

	# find set-specific std deviations
	for exp in df.expansion.unique():
		summary_stats['StandardDeviation' + str(exp)] = round(np.std(df[df.expansion == exp].user_win_rate_bucket), 4)
		summary_stats['WinRate' + str(exp)] = round(df[df.expansion == exp].user_win_rate_bucket.mean(), 4)
	
	# create winrate by set graphs
	create_report_graphs(df, output_type)

	# dump results to pickle
	with open(output_type + '_stats.pkl', 'wb') as h:
		pickle.dump(summary_stats, h, protocol=pickle.HIGHEST_PROTOCOL)

	# save output dataframe for comparison later
	df.to_pickle(output_type + '_data.pkl')

	return df

def trim_dataset(df):
	'''
	Trim columns of each dataframe to speed up concatenation, and drop redundant logs (i.e,
	multiple observations of the same draft)
	'''

	# most columns correspond to specific card information, which we aren't interested in
	cols = df.columns
	sub_cols = [x for x in cols if 
		'sideboard_' not in x and
		'deck_' not in x and
		'opening_hand_' not in x and
		'drawn_' not in x]	
	df = df[sub_cols]

	# drop multiple observations of the same draft (since this reflects the same user, and therefore
	# same winrate)
	df.drop_duplicates('draft_id', inplace=True)

	# restrict to users who played 10+ matches
	df = df[df.user_n_games_bucket >= 10]

	return df


def create_report_graphs(df, draft_type):
	'''
	Given a dataset for bo1 or bo3 drafts, creates necessary graphs
	'''

	# create output folder, if it doesn't exist
	if not os.path.exists('Figures'):
		os.makedirs('Figures')

	# create a graph of winrate distribution by set
	df['WinrateByExpansion'] = df.groupby(['user_win_rate_bucket', 'expansion']).draft_id.transform('count')
	df['GamesInExpansion'] = df.groupby('expansion').draft_id.transform('count')
	df['WinrateByExpansion'] /= df.GamesInExpansion
	winrates = df[['user_win_rate_bucket', 'expansion', 'WinrateByExpansion']].drop_duplicates(['user_win_rate_bucket', 'expansion'])
	winrates.rename(columns={'user_win_rate_bucket': 'Win Rate', 'expansion': 'Expansion'}, inplace=True)
	winrates.sort_values('Win Rate', inplace=True)
	winrates = winrates.pivot(index='Win Rate', columns='Expansion', values='WinrateByExpansion')
	winrates.plot(ylabel='Frequency of Win Rate')
	plt.savefig(join('Figures', draft_type + '_winrate_by_set.png'), bbox_inches='tight')
	plt.close()



