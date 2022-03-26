'''
Script which creates all output tables & figures used in the article
'''

__author__ = 'Austin Herrick'

import pandas as pd
import numpy as np
import math
import os
from os.path import join
import matplotlib.pyplot as plt

from tournament_client import generate_player_distribution, TournamentClient, TournamentType

def figure_manager():
	'''
	Manages all graph creation
	'''

	# create output folder, if it doesn't exist
	if not os.path.exists('Figures'):
		os.makedirs('Figures')

	# create single-draft simulators with 10 million players, used for several below figures
	bo3_ten_million = generate_player_distribution('bo3', 'normal', 10000000, std_dev_scalar=1.5)
	bo3_tourney = TournamentClient(bo3_ten_million, TournamentType.BEST_OF_3)
	bo3_tourney.play_n_drafts(1)
	bo1_ten_million = generate_player_distribution('bo1', 'normal', 10000000, std_dev_scalar=1.5)
	bo1_tourney = TournamentClient(bo1_ten_million, TournamentType.BEST_OF_1)
	bo1_tourney.play_n_drafts(1)

	# create bo3 game vs match win graphs
	match_game_delta()

	# create elo vs implied winrate graph
	elo_expected_winrate(bo3_ten_million)

	# create elo distribution by winrate graphs
	find_outcome_elo_distribution(bo3_tourney.players, 'bo3')
	find_outcome_elo_distribution(bo1_tourney.players, 'bo1')

	# find likelihood of achieving maximum wins by elo & percentile skill
	handle_max_wins_plot(bo3_tourney.players, bo1_tourney.players, 'Elo')
	handle_max_wins_plot(bo3_tourney.players, bo1_tourney.players, 'EloPercentile')

	# load previously generated simulations
	draft_data_bo3 = pd.read_pickle(join('Data', 'bo3_draft_results.pkl'))
	draft_data_bo1 = pd.read_pickle(join('Data', 'bo1_draft_results.pkl'))
	draft_data_bo1['ImpliedWinrateMatch'] = draft_data_bo1.ImpliedWinrate
	draft_data_bo1['GameWins'] = draft_data_bo1.TotalWins
	draft_data_bo1['GameLosses'] = draft_data_bo1.TotalLosses

	# compare expected and actual win rates
	summary_bo3 = compare_realized_winrates(draft_data_bo3, 'bo3')
	summary_bo1 = compare_realized_winrates(draft_data_bo1, 'bo1')
	compare_realized_winrates_delta(summary_bo3, summary_bo1)
	plot_games_played(draft_data_bo1)

	# find graphs on mean costs/packs per draft
	summary_bo1 = cost_per_draft(draft_data_bo1, 'ModeledWinrate')
	summary_bo3 = cost_per_draft(draft_data_bo3, 'ModeledWinrate')
	plot_prizing(summary_bo1, summary_bo3, 'Modeled Game Win Rate')
	summary_bo1 = cost_per_draft(draft_data_bo1, 'EloPercentile')
	summary_bo3 = cost_per_draft(draft_data_bo3, 'EloPercentile')
	plot_prizing(summary_bo1, summary_bo3, 'EloPercentile')

	# create plots related to set completion
	set_completion_plots(draft_data_bo1, draft_data_bo3)

	# compare winrate implied trophy rate and actual trophy rate
	summary_bo3 = expected_vs_actual_trophy_rate(draft_data_bo3, bo3_tourney.players, 'bo3')
	summary_bo1 = expected_vs_actual_trophy_rate(draft_data_bo1, bo1_tourney.players, 'bo1')
	trophy_summary = pd.merge(
		summary_bo1,
		summary_bo3,
		on='EloPercentile',
		how='outer',
		suffixes=(' Premier', ' Traditional')
	)
	trophy_summary.sort_values('EloPercentile').set_index('EloPercentile')[['Trophy Fraction Premier', 'Trophy Fraction Traditional']]\
		.plot(ylabel='Dynamic Model Trophies as a Fraction of Static Model Trophies')
	plt.savefig(join('Figures', 'TrophyRateFraction.png'), bbox_inches='tight')
	plt.close()

	# compare EV in Static vs Dynamic Models
	static_dynamic_model_comparison(draft_data_bo1, draft_data_bo3)

def match_game_delta():
	'''
	Plots differences between game and match win rate in bo3 drafts

	Given a game winrate of p, match win rate equals p^2(3 - 2p) via combinatorics
	'''

	# define function
	p = np.linspace(0, 1, 500)
	win_rate = p**2 * (3 - 2*p)
	
	# create plot showing match win rate bo1 vs bo3
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(p, win_rate, 'r', label='Best of 3 (Traditional Draft)')
	ax.plot(p, p, 'b', label='Best of 1 (Premier Draft)')
	ax.set_xlabel('Game Win Rate')
	ax.set_ylabel('Match Win Rate')
	fig.legend(bbox_to_anchor=(0.52,1), bbox_transform=ax.transAxes)
	plt.savefig(join('Figures', 'match_game_win_delta.png'), bbox_inches='tight')
	plt.close()

def elo_expected_winrate(input_df):
	'''
	Creates graphs of elo distribution and expected winrate distribution
	'''

	# avoid inplace modification
	df = input_df.copy()

	df['Elo'] = df.Elo.astype(int)
	df['ImpliedWinrate'] = 1 / (1 + 10**((1000 - df.Elo) / 400))
	df['ImpliedWinrate'] *= 1000
	df['ImpliedWinrate'] = df.ImpliedWinrate.astype(int)
	df['ImpliedWinrate'] /= 10
	df['EloConcentration'] = df.groupby('Elo').Drafts.transform('count')
	df['ImpliedWinrateConcentration'] = df.groupby('ImpliedWinrate').Drafts.transform('count')
	df['EloConcentration'] /= df.EloConcentration.sum()
	df['ImpliedWinrateConcentration'] /= df.ImpliedWinrateConcentration.sum()

	df = df[['Elo', 'ImpliedWinrate', 'EloConcentration', 'ImpliedWinrateConcentration']].drop_duplicates('Elo').sort_values('Elo')
	
	# create figure, with twinned x-axis labels
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.set_xlabel('Elo')
	ax1.set_ylabel('Concentration')
	ax2 = ax1.twiny()
	ax1.plot(df.Elo, df.EloConcentration, label='Elo')
	ax2.plot(df.ImpliedWinrate, df.ImpliedWinrateConcentration, label='Implied Win Rate', c='g')
	ax2.set_xlabel('Win Rate')
	fig.legend(bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
	plt.savefig(join('Figures', 'elo_versus_winrate_bo3.png'), bbox_inches='tight')
	plt.close()

def find_outcome_elo_distribution(input_players, tournament_type):
	'''
	Plots the distribution of elos for players at particular records
	'''

	# find elo distribution by record
	players = input_players.copy()
	players['Elo'] = players.Elo.round(-1)
	players['EloCount'] = players.groupby(['Elo', 'TotalWins']).Gems.transform('count')
	players['EloDenominator'] = players.groupby('TotalWins').Gems.transform('count')
	players['EloCount'] /= players.EloDenominator

	# pivot dataframe and graph
	summary = players[['Elo', 'TotalWins', 'EloCount']].drop_duplicates(['Elo', 'TotalWins']).sort_values(['Elo', 'TotalWins'])
	summary.rename(columns={'TotalWins': 'Number of Wins'}, inplace=True)
	summary = summary.pivot(index='Elo', columns='Number of Wins', values='EloCount')
	summary.plot(ylabel='Frequency')
	
	plt.savefig(join('Figures', tournament_type + "_elo_by_winrate.png"), bbox_inches='tight')
	plt.close()

def handle_max_wins_plot(bo3_input, bo1_input, xaxis):
	'''
	Creates a plot of likelihood of achieving maximum wins, according to a specified axis
	'''

	bo1_max_wins = maximum_wins_likelihood(bo3_input, xaxis)
	bo3_max_wins = maximum_wins_likelihood(bo1_input, xaxis)
	max_wins = pd.merge(
		bo1_max_wins,
		bo3_max_wins,
		left_index=True,
		right_index=True,
		how='outer',
		suffixes=('_Premier', '_Traditional'))
	max_wins.rename(columns={\
		'Max Wins Frequency_Premier': 'Premier Draft', 
		'Max Wins Frequency_Traditional': 'Traditional Draft'}, inplace=True)
	max_wins.plot(ylabel='Percent Chance of Winning Draft')
	plt.savefig(join('Figures', 'MaxWinFrequency' + xaxis + '.png'), bbox_inches='tight')
	plt.close()

def maximum_wins_likelihood(input_players, xaxis):
	'''
	Finds the frequency of achieving maximum wins by  a specified axis
	'''

	# identify frequency of achieving maximum wins, by elo
	players = input_players.copy()

	if xaxis == 'Elo':
		players[xaxis] = players[xaxis].round(-1)
	elif xaxis == 'EloPercentile':
		xaxis = 'Elo Percentile'
		players.rename(columns={'EloPercentile': xaxis}, inplace=True)

	players['MaxWins'] = (players.TotalWins == players.TotalWins.max()).astype(int)
	players['MaxWinsCount'] = players.groupby(xaxis).MaxWins.transform('sum')
	players['Count'] = players.groupby(xaxis).MaxWins.transform('count')
	players['Max Wins Frequency'] = players.MaxWinsCount / players.Count
	
	# plot frequency
	summary = players[players.Count > 100][[xaxis, 'Max Wins Frequency']]\
		.drop_duplicates(xaxis).sort_values(xaxis).set_index(xaxis)
	
	return summary

def compare_realized_winrates(data, draft_type):
	'''
	Creates a figure comparing implied win rate from Elo and realized win rate from actual
	matches played
	'''
	
	if draft_type == 'bo3':
		draft_label = '-- Traditional Draft'
	elif draft_type == 'bo1':
		draft_label = '-- Premier Draft'

	# copy input dataframe to avoid modifying in-place
	df = data.copy()

	# calculate actual overall winrate, and winrate in the first round (i.e, fully random pairings)
	df['ModeledWinrate'] = df.TotalWins / (df.TotalWins + df.TotalLosses)
	df['FirstRoundWinrate'] = df.FirstRoundWins / (df.FirstRoundWins + df.FirstRoundLosses)

	# find winrates by percentile 
	df['Implied Win Rate ' + draft_label] = df.groupby('EloPercentile').ImpliedWinrateMatch.transform('mean')
	df['Modeled Win Rate ' + draft_label] = df.groupby('EloPercentile').ModeledWinrate.transform('mean')

	df.rename(columns={'EloPercentile': 'Elo Percentile'}, inplace=True)
	summary = df[['Elo Percentile', 'Implied Win Rate ' + draft_label, 'Modeled Win Rate ' + draft_label]]\
		[df['Elo Percentile'].between(10, 90)]\
		.drop_duplicates('Elo Percentile').sort_values('Elo Percentile').set_index('Elo Percentile')

	summary[['Implied Win Rate ' + draft_label, 'Modeled Win Rate ' + draft_label]].plot(ylabel='Win Rate')
	plt.savefig(join('Figures', 'ImpliedVsModeledWinrate_' + draft_type + '.png'), bbox_inches='tight')
	plt.close()

	# calculate win rate delta (used for next graph)
	summary['Win Rate Difference'] = summary['Modeled Win Rate ' + draft_label] - summary['Implied Win Rate ' + draft_label]

	return summary

def compare_realized_winrates_delta(summary_bo3, summary_bo1):
	'''
	Compare difference in expected and actual win rate, by draft type
	'''

	summary = pd.merge(
		summary_bo3, 
		summary_bo1,
		left_index=True,
		right_index=True,
		how='outer',
		suffixes=(' -- Traditional', ' -- Premier')
	)
	summary['Equality'] = 0
	summary[['Win Rate Difference -- Premier', 'Win Rate Difference -- Traditional', 'Equality']].plot(ylabel='Modeled Win Rate Less Expected Win Rate')
	plt.savefig(join('Figures', 'ImpliedVsModeledWinrateDelta.png'), bbox_inches='tight')
	plt.close()

	# save plot of differences in implied win rate by format type
	summary[['Implied Win Rate -- Premier Draft', 'Implied Win Rate -- Traditional Draft']].plot()
	plt.savefig(join('Figures', 'ImpliedWinrateBo1vsBo3.png'))
	plt.close()

def plot_games_played(df_bo1):
	'''
	Creates a plot of the average number of games played, by elo percentile
	'''

	# avoid modifying input dataframe in place
	df = df_bo1.copy()
	df['GamesPerDraft'] = df.GamesPlayed / df.Drafts
	df['Average Games Played Per Draft'] = df.groupby('EloPercentile').GamesPerDraft.transform('mean')


	summary = df[['EloPercentile', 'Average Games Played Per Draft']]\
		.sort_values('EloPercentile').drop_duplicates('EloPercentile').set_index('EloPercentile')
	summary.plot(ylabel='Games Played')
	plt.savefig(join('Figures', 'GamesPerDraft.png'), bbox_inches='tight')
	plt.close()

def cost_per_draft(input_df, win_metric):
	'''
	Maps mean cost of a draft under various winrate metrics
	'''

	# back up dataframe to avoid inplace modification
	df = input_df.copy()
	if win_metric == 'ModeledWinrate':

		# for "Modeled Winrate" graphs, map game wins / game losses
		df['Modeled Game Win Rate'] = df.GameWins / (df.GameWins + df.GameLosses)
		win_metric = 'Modeled Game Win Rate'
	
	if win_metric == 'MatchWinrate':
		df['MatchWinrate'] = df.TotalWins / (df.TotalWins + df.TotalLosses)
	
	df[win_metric] = df[win_metric].round(3)


	# find mean draft cost, based on assumption of a pack's worth
	df['Gems20'] = df.Gems + df.Packs * 20
	df['Gems200'] = df.Gems + df.Packs * 200
	df['MeanDraftCost20'] = df.Gems20 / df.Drafts
	df['MeanDraftCost200'] = df.Gems200 / df.Drafts
	df['MeanPacks'] = df.Packs / df.Drafts

	# find draft cost distributed by chosen metric
	df['Cost20'] = df.groupby(win_metric).MeanDraftCost20.transform('mean')
	df['Cost200'] = df.groupby(win_metric).MeanDraftCost200.transform('mean')
	df['Packs per Draft'] = df.groupby(win_metric).MeanPacks.transform('mean')
	summary = df[[win_metric, 'Cost20', 'Cost200', 'Packs per Draft']]\
		.drop_duplicates(win_metric).sort_values(win_metric)

	return summary

def plot_prizing(summary_bo1, summary_bo3, win_metric):
	'''
	Creates plots of mean gem and mean pack distributions by win rate
	'''

	# merge summaries
	summary = pd.merge(
		summary_bo1,
		summary_bo3,
		on=win_metric,
		how='outer',
		suffixes=(' Premier', ' Traditional')
	)
	summary.sort_values(win_metric, inplace=True)
	summary.set_index(win_metric, inplace=True, drop=True)

	# create gem plot
	fig, ax = plt.subplots()
	ax.plot(summary.index, summary['Cost20 Premier'], \
		color='b', label='Cost per Draft (20 Gems/Pack) Premier')
	ax.plot(summary.index, summary['Cost200 Premier'], ':', \
		color='b', label='Cost per Draft (200 Gems/Pack) Premier')
	ax.plot(summary.index, summary['Cost20 Traditional'], \
		color='r', label='Cost per Draft (20 Gems/Pack) Traditional')
	ax.plot(summary.index, summary['Cost200 Traditional'], ':', \
		color='r', label='Cost per Draft (200 Gems/Pack) Traditional')
	ax.legend()
	ax.set_ylabel('Net Gems')
	ax.set_xlabel(win_metric)
	plt.savefig(join('Figures', 'NetGemsPerDraft_' + win_metric + '.png'), bbox_inches='tight')
	plt.close()

	# create packs plot (only happens on first execution)
	if win_metric == 'EloPercentile':
		summary[['Packs per Draft Premier', 'Packs per Draft Traditional']].plot(ylabel='Packs')
		plt.savefig(join('Figures', 'PacksPerDraft.png'), bbox_inches='tight')
		plt.close()

def set_completion_plots(df_bo1, df_bo3):
	'''
	Creates all plots related to set completion
	'''

	# compile plots by metric
	for win_metric in ['Modeled Game Win Rate', 'EloPercentile']:
		bo1_summary = set_completion(df_bo1, win_metric)
		bo3_summary = set_completion(df_bo3, win_metric)
		winrate_summary = pd.merge(
			bo1_summary,
			bo3_summary,
			on=win_metric,
			suffixes=(' -- Premier', ' -- Traditional')
		)
		
		winrate_summary.set_index(win_metric)\
			[['Average Drafts to Collect Rares -- Premier', 'Average Drafts to Collect Rares -- Traditional']].plot(ylabel='Drafts')
		plt.savefig(join('Figures', 'DraftsForRares_' + win_metric + '.png'), bbox_inches='tight')
		plt.close()

		winrate_summary.set_index(win_metric)\
			[['Average Gem Cost to Collect Rares -- Premier', 'Average Gem Cost to Collect Rares -- Traditional']].plot(ylabel='Net Gems')
		plt.savefig(join('Figures', 'GemsForRares_' + win_metric + '.png'), bbox_inches='tight')
		plt.close()

def set_completion(input_df, win_metric):
	'''
	Collapses metrics related to set completion
	'''

	# prevent inplace modification
	df = input_df.copy()

	# prepare ModeledWinrate if needed
	if win_metric == 'Modeled Game Win Rate':
		df['Modeled Game Win Rate'] = df.GameWins / (df.GameWins + df.GameLosses)
		df['Modeled Game Win Rate'] = df['Modeled Game Win Rate'].round(3)

	# collapse draft completion info
	df['Average Drafts to Collect Rares'] = df.groupby(win_metric).DraftsAtRareCompletion.transform('mean')
	df['Average Gem Cost to Collect Rares'] = df.groupby(win_metric).GemsAtRareCompletion.transform('mean')
	summary = df[[win_metric, 'Average Drafts to Collect Rares', 'Average Gem Cost to Collect Rares']]\
		.drop_duplicates(win_metric).sort_values(win_metric)

	return summary

def expected_vs_actual_trophy_rate(df_empirical, df_representative, draft_type):
	'''
	Computes the actual trophy rate compared to the trophy rate implied by win percentage
	'''

	# find aggregated actual win rate, without inplace modification
	df = df_empirical.copy()
	df['ModeledWinrate'] = df.TotalWins / (df.TotalWins + df.TotalLosses)
	df['ModeledWinrate'] = df.groupby('EloPercentile').ModeledWinrate.transform('mean')

	# best of 1 and best of 3 work differently
	if draft_type == 'bo3':
		df['Implied Trophy Rate'] = df.ModeledWinrate ** 3
	
	# for Best of 1 drafts, we use a summation of Bernoulli Trials to calculate odds of winning
	# 7, 8, or 9 games out of 9 games played
	elif draft_type == 'bo1':
		df['Implied Trophy Rate'] = (df.ModeledWinrate ** 9) + \
			((df.ModeledWinrate ** 8) * ((1 - df.ModeledWinrate) ** 1)) * ((math.factorial(9)) / (math.factorial(8) * math.factorial(9 - 8))) + \
			((df.ModeledWinrate ** 7) * ((1 - df.ModeledWinrate) ** 2)) * ((math.factorial(9)) / (math.factorial(7) * math.factorial(9 - 7)))

	# collapse data
	summary_empirical = df[['EloPercentile', 'ModeledWinrate', 'Implied Trophy Rate']].drop_duplicates('EloPercentile').sort_values('EloPercentile')

	# find representative trophy rate
	df_single = df_representative.copy()
	df_single['GotTrophy'] = (df_single.TotalWins == df_single.TotalWins.max()).astype(int)
	df_single['Modeled Trophy Rate'] = df_single.groupby('EloPercentile').GotTrophy.transform('mean')
	summary_representative = df_single[['EloPercentile', 'Modeled Trophy Rate']].drop_duplicates('EloPercentile').sort_values('EloPercentile')

	# merge & plot data
	summary = pd.merge(
		summary_empirical,
		summary_representative,
		on='EloPercentile',
		how='inner'
	)
	summary.rename(columns={'ModeledWinrate': 'Match Win Rate'}, inplace=True)
	summary.sort_values('Match Win Rate').set_index('Match Win Rate')[['Implied Trophy Rate', 'Modeled Trophy Rate']].plot()
	plt.savefig(join('Figures', 'ImpliedvsActualTrophy_' + draft_type + '.png'), bbox_inches='tight')
	plt.close()

	# collapse by nearest elo percentile, calculate trophy fraction
	summary['TrophyFraction'] = summary['Modeled Trophy Rate'] / summary['Implied Trophy Rate']
	summary['EloPercentile'] = summary.EloPercentile.round(0)
	summary['Trophy Fraction'] = summary.groupby('EloPercentile').TrophyFraction.transform('mean')
	summary.drop_duplicates('EloPercentile', inplace=True)

	return summary

def static_dynamic_model_comparison(draft_data_bo1, draft_data_bo3):
	'''
	Compares expected cost of drafting for both bo1 and bo3 using both the dynamic model
	and a simpler static model
	'''

	summary_bo3 = compare_model_draft_cost(draft_data_bo3, 'bo3', 'MatchWinrate')
	summary_bo1 = compare_model_draft_cost(draft_data_bo1, 'bo1', 'MatchWinrate')
	ev_delta = pd.merge(
		summary_bo3,
		summary_bo1,
		left_index=True,
		right_index=True,
		how='outer',
		suffixes=(' -- Traditional', ' -- Premier')
	)
	ev_delta.index.rename('Modeled Match Win Rate', inplace=True)
	ev_delta['Net Equality (For Visual Comparison)'] = 0
	ev_delta[[x for x in ev_delta.columns if 'Delta' in x] + ['Net Equality (For Visual Comparison)']]\
		.plot(ylabel='Dynamic Gems - Static Gems')
	plt.savefig(join('Figures', 'StaticDynamicDeltas.png'), bbox_inches='tight')

	summary_bo3 = compare_model_draft_cost(draft_data_bo3, 'bo3', 'ImpliedWinrateMatch')
	summary_bo1 = compare_model_draft_cost(draft_data_bo1, 'bo1', 'ImpliedWinrateMatch')
	ev_delta = pd.merge(
		summary_bo3,
		summary_bo1,
		left_index=True,
		right_index=True,
		how='outer',
		suffixes=(' -- Traditional', ' -- Premier')
	)
	ev_delta.index.rename('Implied Match Win Rate', inplace=True)
	ev_delta['Net Equality (For Visual Comparison)'] = 0
	ev_delta[[x for x in ev_delta.columns if 'Delta' in x] + ['Net Equality (For Visual Comparison)']]\
		.plot(ylabel='Dynamic Gems - Static Gems')
	plt.savefig(join('Figures', 'StaticDynamicDeltasImplied.png'), bbox_inches='tight')


def compare_model_draft_cost(df, draft_type, win_metric):
	'''
	Compares net draft cost by win rate, dynamic model vs static model
	'''

	# find prizing in dynamic vs static model
	summary = cost_per_draft(df, win_metric)
	summary = expected_net_cost_static_model(summary, draft_type, win_metric)

	summary['Static Model (20 Gems/Pack)'] = summary.StaticGems + 20*summary.StaticPacks - 1500
	summary['Static Model (200 Gems/Pack)'] = summary.StaticGems + 200*summary.StaticPacks - 1500
	summary.rename(columns={
		'Cost20': 'Dynamic Model (20 Gems/Pack)', 
		'Cost200': 'Dynamic Model (200 Gems/Pack)'}, inplace=True
	)

	summary['Delta (20 Gems/Pack)'] = summary['Dynamic Model (20 Gems/Pack)'] - summary['Static Model (20 Gems/Pack)']


	# create gem plot
	fig, ax = plt.subplots()
	summary.set_index(win_metric, inplace=True)
	ax.plot(summary.index, summary['Dynamic Model (20 Gems/Pack)'], \
		color='b', label='Dynamic Model (20 Gems/Pack)')
	ax.plot(summary.index, summary['Dynamic Model (200 Gems/Pack)'], ':', \
		color='b', label='Dynamic Model (200 Gems/Pack)')
	ax.plot(summary.index, summary['Static Model (20 Gems/Pack)'], \
		color='r', label='Static Model (20 Gems/Pack)')
	ax.plot(summary.index, summary['Static Model (200 Gems/Pack)'], ':', \
		color='r', label='Static Model (200 Gems/Pack)')		
	ax.legend()
	ax.set_ylabel('Net Gems')
	ax.set_xlabel('Match Win Rate')
	plt.savefig(join('Figures', 'StaticVsDynamicCost' + win_metric + '_' + draft_type + '.png'), bbox_inches='tight')
	plt.close()

	return summary

def expected_net_cost_static_model(df, draft_type, win_metric):
	'''
	Finds expected net cost of a draft by match winrate, for both bo1 and bo3, using a static model
	
	Note on the Bo1 Math:
	In order to achieve any bo1 record of X wins (below 7), a player needs to win X games
	and lose 2, in any order, and then lose a third game. This is represented by the
	following probability: 

	((x+2) choose x) * p^x * (1-p)^2, * (1-p), for x in [0, 6]

	Notably, ((x + 2) choose x) simplifies to (x+2)(x+1)/2
	'''

	# define outcome space for bo3 drafts
	if draft_type == 'bo3':

		# assign pack rewards for 0-1 wins (1 pack), 2 wins (4 packs), or 3 wins (6 packs)
		three_win_fraction = (df[win_metric] ** 3)
		two_win_fraction = (df[win_metric] ** 2) * (1 - df[win_metric]) * 3
		less_wins_fraction = (1 - df[win_metric])**3 + (1 - df[win_metric])**2 * (df[win_metric]) * 3
		df['StaticPacks'] = three_win_fraction * 6 + \
			two_win_fraction * 4 + \
			less_wins_fraction * 1
		df['StaticGems'] = three_win_fraction * 3000 + \
			two_win_fraction * 1500 + \
			less_wins_fraction * 0

	if draft_type == 'bo1':
		pack_prizes = {
			0: 1,
			1: 1,
			2: 2,
			3: 2,
			4: 3,
			5: 4,
			6: 5,
			7: 6
		}

		gem_prizes = {
			0: 50,
			1: 100,
			2: 250,
			3: 1000,
			4: 1400,
			5: 1600,
			6: 1800,
			7: 2200
		}
		
		df['StaticPacks'] = 0
		df['StaticGems'] = 0
		for i in range(7):
			df['StaticPacks'] += pack_prizes[i] * \
				(1 - df[win_metric])**3 * \
				(df[win_metric])**i * \
				((i + 2) * (i + 1) * (1/2))
			df['StaticGems'] += gem_prizes[i] * \
				(1 - df[win_metric])**3 * \
				(df[win_metric])**i * \
				((i + 2) * (i + 1) * (1/2))

		# separately calculate 7 win prizes, using formula discussed above		
		df['StaticPacks'] += pack_prizes[7] * ((df[win_metric] ** 9) + \
			((df[win_metric] ** 8) * ((1 - df[win_metric]) ** 1)) * ((math.factorial(9)) / (math.factorial(8) * math.factorial(9 - 8))) + \
			((df[win_metric] ** 7) * ((1 - df[win_metric]) ** 2)) * ((math.factorial(9)) / (math.factorial(7) * math.factorial(9 - 7))))
		df['StaticGems'] += gem_prizes[7] * ((df[win_metric] ** 9) + \
			((df[win_metric] ** 8) * ((1 - df[win_metric]) ** 1)) * ((math.factorial(9)) / (math.factorial(8) * math.factorial(9 - 8))) + \
			((df[win_metric] ** 7) * ((1 - df[win_metric]) ** 2)) * ((math.factorial(9)) / (math.factorial(7) * math.factorial(9 - 7))))

	return df