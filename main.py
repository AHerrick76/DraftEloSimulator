'''
Master script to handle all processing
'''

__author__ = 'Austin Herrick'


import numpy as np
import pandas as pd
import pickle
import math
import os
from os.path import join
import matplotlib.pyplot as plt

from distribution_finder import process_all_data
from visualizations import figure_manager
from tournament_client import TournamentClient
from tournament_client import TournamentType

def main(mode):
	'''
	Handles all processing. Depending on function call, does one of the following:
	
	- Find summary statistics from all 17lands data
	- Creates & caches large draft simulations
	- Creates all figures / results from draft simulations
	'''

	# loads all 17lands data, finds summary statistics, and produces required charts
	if mode == 1:
		process_all_data()
	
	# creates 100k player tournament clients, runs 5000 drafts in each, and caches results
	elif mode == 2:
		prepare_simulation('bo3')
		prepare_simulation('bo1')

	# create all simulation-related charts necessary for article
	elif mode == 3:
		figure_manager()	


def prepare_simulation(draft_type):
	'''
	Creates output from large-scale draft simulation, to be analyzed later
	'''

	if draft_type == 'bo3':
		t_type = TournamentType.BEST_OF_3
	elif draft_type == 'bo1':
		t_type = TournamentType.BEST_OF_1
	else:
		raise Exception("Invalid draft type!")
	
	# simulate drafts for both bo1 and bo3
	df = generate_player_distribution(draft_type, 'normal', 100000, std_dev_scalar=1.5)
	tourney = TournamentClient(df, t_type)
	tourney.play_n_drafts(5000)

	# cache files
	tourney.players.to_pickle(draft_type + '_draft_results.pkl')


def generate_player_distribution(tournament_type, distribution_type, player_count, std_dev_scalar=1):
	'''
	Load summary statistics from 17lands data according to format type, and retrieve standard deviation.

	Use standard deviation to find implied distribution scale from Chess ELO formula.
	Recall that formula is WinPercentage = 1 / (1 + 10^((Elo_p1 - Elo_p2) / 400).
	We define a spread n such that a player 1 standard deviation above the mean (with Elo 1000 + n)
	will win against the average player 50 + StandardDeviation percent of the time. Player 1 here
	has Elo 1000 + n, whereas player 2 has Elo 1000.

	See below algebra:
	.5 + StdDev = 1 / (1 + 10^(-n / 400))
	1 + 10^(-n/400) = 1 / (.5 + StdDev)
	10^(-n/400) = (.5 - StdDev) / (.5 + StdDev)
	log(10^(-n/400)) = log((.5 - StdDev) / (.5 + StdDev))
	(-n/400) = log((.5 - StdDev) / (.5 + StdDev))
	-n = 400 * log((.5 - StdDev) / (.5 + StdDev))

	Next, use the identified parameter to create a distribution of player ELOs and initialize
	the player dataframe
	'''

	# load summary statistics from 17lands data
	with open(tournament_type + '_stats.pkl', 'rb') as h:
		stats = pickle.load(h)

	# find distribution scalar
	# it's reasonable to think that the data's standard deviation estimate is biased downward,
	# because we're disproportinately observing more skilled players with already above-average winrates
	std_dev = stats['OverallStandardDeviation']
	std_dev *= std_dev_scalar
	target_scale = -1 * (400 * math.log((0.5 - std_dev) / (0.5 + std_dev), 10))

	# initialize ELO distribution
	if distribution_type == 'normal':
		elo_dist = np.random.normal(1000, target_scale, player_count)
	elif distribution_type == 'logistic':
		elo_dist = np.random.logistic(1000, target_scale, player_count)

	# initialize player distribution
	player_df = pd.DataFrame(elo_dist, columns=['Elo'])
	for col in [
		'CurrentWins', 'CurrentLosses', 'TotalWins', 'TotalLosses', 'Drafts',
		'GamesPlayed', 'Gems', 'Packs', 'FirstRoundWins', 'FirstRoundLosses', 'FinishedRares',
		'FinishedMythics', 'CollectedRares', 'CollectedMythics', 'DraftsAtRareCompletion',
		'GemsAtRareCompletion', 'DraftsAtMythicCompletion', 'GemsAtMythicCompletion',
		'EloDeltaFirstRound', 'EloDeltaFinalRound', 'MeanEloDeltaFirstRound', 'MeanEloDeltaFinalRound'
	]:
		player_df[col] = 0
		
	player_df['PlayerID'] = player_df.index
	# find implied winrate versus an average player from ELO
	player_df['ImpliedWinrate'] = 1 / (1 + 10**((1000 - player_df.Elo) / 400))

	# find percentile of player skill
	player_df['EloPercentile'] = player_df.Elo.rank(pct=True).round(3) * 100

	return player_df
