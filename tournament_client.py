'''
Holds class for the Tournament Client, which manages all drafts
'''

__author__ = 'Austin Herrick'

import numpy as np
import pandas as pd
from enum import Enum
import time
import pickle
import math
from os.path import join

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
	with open(join('Data', tournament_type + '_stats.pkl'), 'rb') as h:
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
		'EloDeltaFirstRound', 'EloDeltaFinalRound', 'MeanEloDeltaFirstRound', 'MeanEloDeltaFinalRound',
		'GameWins', 'GameLosses'
	]:
		player_df[col] = 0
		
	player_df['PlayerID'] = player_df.index
	# find implied winrate versus an average player from ELO
	player_df['ImpliedWinrate'] = 1 / (1 + 10**((1000 - player_df.Elo) / 400))
	player_df['ImpliedWinrateMatch'] = (player_df.ImpliedWinrate ** 2) * (3 - 2*player_df.ImpliedWinrate)

	# find percentile of player skill
	player_df['EloPercentile'] = player_df.Elo.rank(pct=True).round(3) * 100

	return player_df

# define enums for tournament type
class TournamentType(Enum):
	BEST_OF_3 = 1
	BEST_OF_1 = 2

# define tournament client
class TournamentClient():
	'''
	Controls players within a client
	'''

	def __init__(self, players, tournament_type, deck_quality_advantage=0.075):

		# define basic parameters
		self.players = players
		self.tournament_type = tournament_type

		# define round parameters
		# round holder is used to store subsets of players during round execution
		self.round_holder = []
		self.round = 1

		# represents the percent improvement (as a scalar) in win percentage
		# gained from being a tier above your opponent in deck quality
		self.deck_quality_advantage = deck_quality_advantage


	# plays a full draft
	def play_draft(self, verbose=True):
		
		# subtract entry fee for all players
		self.players['Gems'] -= 1500

		# assign each player an in-draft deck quality
		# most decks are "average", but we identify the top 20% and bottom 20% of deck quality
		# these decks have an advantage when calculating winrates
		self.players['DeckQualityWeight'] = np.random.randint(1, 11, len(self.players))
		self.players['DeckQuality'] = 0
		self.players.loc[self.players.DeckQualityWeight <= 2, 'DeckQuality'] = -1
		self.players.loc[self.players.DeckQualityWeight >= 9, 'DeckQuality'] = 1

		# in bo3 touraments, drafts are 3 rounds for all players
		if self.tournament_type == TournamentType.BEST_OF_3:
			while self.round <= 3:
				if verbose:
					print('Beginning Round {}.....'.format(self.round))
				self.play_round()
		
			# give out prizes for bo3
			self.players.loc[self.players.CurrentWins == 3, 'Gems'] += 3000
			self.players.loc[self.players.CurrentWins == 2, 'Gems'] += 1000
			self.players.loc[self.players.CurrentWins == 3, 'Packs'] += 6
			self.players.loc[self.players.CurrentWins == 2, 'Packs'] += 4
			self.players.loc[self.players.CurrentWins <= 1, 'Packs'] += 1

		# in bo1 tournaments, players finish a draft in 3-9 rounds, depending on outcomes
		# (players play until they reach 7 wins or 3 losses, so reaching a 6-2 record allows 9 rounds)
		if self.tournament_type == TournamentType.BEST_OF_1:
			while self.round <= 9:
				if verbose:
					print('Beginning Round {}.....'.format(self.round))
				self.play_round()
			
			# give out prizes for bo1
			gem_payouts = [50, 100, 250, 1000, 1400, 1600, 1800, 2200]
			pack_payouts = [1, 1, 2, 2, 3, 4, 5, 6]
			for win_count in range(8):
				self.players.loc[self.players.CurrentWins == win_count, 'Gems'] += gem_payouts[win_count]
				self.players.loc[self.players.CurrentWins == win_count, 'Packs'] += pack_payouts[win_count]
			
		# increment tracking
		self.players['Drafts'] += 1
		self.players['GamesPlayed'] += self.players.CurrentWins + self.players.CurrentLosses

		# increment elo delta statistics
		self.players['MeanEloDeltaFirstRound'] = self.players.MeanEloDeltaFirstRound + \
			abs((self.players.EloDeltaFirstRound / self.players.Drafts))
		self.players['MeanEloDeltaFinalRound'] = self.players.MeanEloDeltaFinalRound + \
			abs((self.players.EloDeltaFinalRound / self.players.Drafts))

		# add collected rares/mythics to a player's collection (3 packs per draft, with 1/8 of
		# rares upgraded to mythic)
		self.players['CollectedRares'] += 3 * (7/8)
		self.players['CollectedMythics'] += 3 * (1/8)

		# prize packs' rare upgrades to mythic at approx. 1:8 odds (with some per-set variance)
		# each rare/mythic is replaced with a wildcard approx. 1:30 odds (with some variance due to
		# scaling probability)
		# see detailed odds information here: https://magic.wizards.com/en/mtgarena/drop-rates
		# note that drafted rares are not duplicate protected, but pack-opened rares are. Players
		# can collect up to 4 copies of each card, and each set has an average of 60 rares and 20 mythics
		# This method assumes players do not craft/obtain rares in other ways (such as daily/event ICRs,
		# progression, etc), assumes rares per draft are evenly distributed, and assumes effective duplicate
		# protection in draft (though this can be approximated by not opening any packs until a user has enough
		# to finish the set)
		just_finished_rares = (self.players.FinishedRares == 0) & \
			(self.players.CollectedRares + ((7/8) * (29/30) * self.players.Packs) >= 60 * 4)
		self.players.loc[just_finished_rares, 'DraftsAtRareCompletion'] = self.players.loc[just_finished_rares, 'Drafts']
		self.players.loc[just_finished_rares, 'GemsAtRareCompletion'] = self.players.loc[just_finished_rares, 'Gems']
		self.players.loc[just_finished_rares, 'FinishedRares'] = 1

		just_finished_mythics = (self.players.FinishedMythics == 0) & \
			(self.players.CollectedMythics + ((1/8) * (29/30) * self.players.Packs) >= 20 * 4)
		self.players.loc[just_finished_mythics, 'DraftsAtMythicCompletion'] = self.players.loc[just_finished_mythics, 'Drafts']
		self.players.loc[just_finished_mythics, 'GemsAtMythicCompletion'] = self.players.loc[just_finished_mythics, 'Gems']
		self.players.loc[just_finished_mythics, 'FinishedMythics'] = 1


		# reset round counters, and current win/loss statistics
		self.round = 1
		self.players['CurrentWins'] = 0
		self.players['CurrentLosses'] = 0

	# plays a tournament round
	def play_round(self):

		# generate outcome draws for all players
		self.players['MatchOutcome'] = np.random.random(len(self.players))

		# in initial round, pair all players randomly
		if self.round == 1:
			self.execute_pairings(self.players.copy())
		# otherwise, iterate through possible records and pair accordingly
		else:
			# in Best of 3 tournaments, all players play all rounds,
			# so players can be paired entirely based on # of wins
			if self.tournament_type == TournamentType.BEST_OF_3:
				
				# retrieve subsets and execute round
				for win_count in [0, 1, 2]:
					subset = self.players[self.players.CurrentWins == win_count]
					self.execute_pairings(subset)

			# In Best of 1 tournaments, players keep playing until they reach 7 wins or 3 losses
			if self.tournament_type == TournamentType.BEST_OF_1:
				
				# retrieve possible subsets based on all combinations of 0-6 wins, 0-2 losses
				self.players['PlayedRound'] = 0
				for loss_count in [0, 1, 2]:
					for win_count in range(7):
						subset_condition = (self.players.CurrentWins == win_count) & (self.players.CurrentLosses == loss_count)
						self.players.loc[subset_condition, 'PlayedRound'] = 1
						subset = self.players[subset_condition]
						self.execute_pairings(subset)

				# hold all players who did not play this round
				self.round_holder.append(self.players[self.players.PlayedRound == 0])

		# increment round, and reconstitute player dataframe
		self.round += 1
		self.players = pd.concat(self.round_holder)
		self.round_holder = []

	def execute_pairings(self, subset):
		'''
		Given a subset of players with matching records, pair players and determine match
		winners
		'''

		# if no players are passed, skip function (this happens in earlier rounds)
		if len(subset) == 0:
			return

		# to pair players, shuffle the dataframe, and assign opponents based on odd/even index pairs
		# players left without a partner are assigned a hypothetical average player
		pairings = subset.sample(len(subset))
		pairings['WonGame'] = 0
		pairings.reset_index(inplace=True, drop=True)
		pairings['NextElo'] = pairings.Elo.shift(-1)
		pairings.NextElo.fillna(1000, inplace=True)
		pairings['NextQuality'] = pairings.DeckQuality.shift(-1)
		pairings.NextQuality.fillna(0, inplace=True)

		# identify elo difference between players
		if self.round == 1:
			pairings['EloDeltaFirstRound'] = pairings.NextElo - pairings.Elo
		# final delta is identified every round, because the last round in bo1 drafts
		# isn't known until match outcome is determined
		else:
			pairings['EloDeltaFinalRound'] = pairings.NextElo - pairings.Elo

		# calculate win probability based on elo difference, and decide match outcome
		pairings['WinProbability'] = 1 / (1 + 10 ** ((pairings.NextElo - pairings.Elo) / 400))
		
		# bound win probability at 15-85%
		pairings.loc[pairings.WinProbability < 0.15, 'WinProbability'] = 0.15
		pairings.loc[pairings.WinProbability > 0.85, 'WinProbability'] = 0.85

		# scale win probability based on deck quality delta
		pairings['DeckQualityDelta'] = pairings.DeckQuality - pairings.NextQuality
		pairings['WinProbability'] *= (1 + (pairings.DeckQualityDelta * self.deck_quality_advantage))

		# in best of 3 tournaments, scale game win rate to match win rate (winning 2 games out of 3)
		# a player can win a bo3 round with WW, WLW, or LWW combinations, or p^2 + 2(p^2)(1-p),
		# or p^2(3 - 2p)
		# game wins/losses also tracks the likelihood of particular match outcomes
		if self.tournament_type == TournamentType.BEST_OF_3:

			# track game outcomes by cumulative probability
			# (lose in 3 isn't tracked explicitly, since its cumulative probability is always 1)
			pairings['WinIn3'] = (pairings.WinProbability ** 2) * (1 - pairings.WinProbability) * 2
			pairings['WinIn2'] = pairings.WinIn3 + pairings.WinProbability ** 2
			pairings['LoseIn2'] = pairings.WinIn2 + (1 - pairings.WinProbability)**2
		
			pairings.loc[pairings.MatchOutcome < pairings.WinIn3, 'GameLosses'] += 1
			pairings.loc[pairings.MatchOutcome < pairings.WinIn2, 'GameWins'] += 2
			pairings.loc[pairings.MatchOutcome > pairings.WinIn2, 'GameLosses'] += 2
			pairings.loc[pairings.MatchOutcome > pairings.LoseIn2, 'GameWins'] += 1

			# track match win probability
			pairings['WinProbability'] = (pairings.WinProbability**2) * (3 - 2 * pairings.WinProbability)
		
		pairings['FirstPlayerWins'] = (pairings.MatchOutcome < pairings.WinProbability).astype(int)
		pairings['CheckVictory'] = pairings.FirstPlayerWins.shift(1)

		# determine which outcome variable to check based on index
		pairings['EvenIndex'] = (pairings.index % 2 == 0).astype(int)
		pairings.loc[pairings.EvenIndex == 1, 'WonGame'] = pairings.FirstPlayerWins
		pairings.loc[pairings.EvenIndex == 0, 'WonGame'] = (pairings.CheckVictory != 1).astype(int)

		# increment wins/losess
		pairings.loc[pairings.WonGame == 0, 'CurrentLosses'] += 1
		pairings.loc[pairings.WonGame == 0, 'TotalLosses'] += 1
		pairings.loc[pairings.WonGame == 1, 'CurrentWins'] += 1
		pairings.loc[pairings.WonGame == 1, 'TotalWins'] += 1

		# in the first round, increment first round statistics
		if self.round == 1:
			pairings.loc[pairings.WonGame == 0, 'FirstRoundLosses'] += 1
			pairings.loc[pairings.WonGame == 1, 'FirstRoundWins'] += 1

		# append all players to the round holder, which will later recombine into a new players
		# dataframe
		self.round_holder.append(pairings)

	# simulates many drafts to generate statistics
	def play_n_drafts(self, n, verbose=False, timing=True):

		if timing:
			start = time.time()

		counter = 0
		while counter < n:
			self.play_draft(verbose=verbose)
			counter += 1

		if timing:
			end = time.time()
			print('Executed {} drafts in {} seconds, with {} players...'.format(
				n, round(end - start, 4), len(self.players)
			))