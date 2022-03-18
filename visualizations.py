'''
Script which creates all output tables & figures used in the article
'''

__author__ = 'Austin Herrick'

import pandas as pd
import math
import os
from os.path import join
import matplotlib.pyplot as plt

from tournament_client import TournamentClient
from tournament_client import TournamentType
from main import generate_player_distribution

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

	# create elo vs implied winrate graph
	elo_expected_winrate(bo3_ten_million)

	# create elo distribution by winrate graphs
	find_outcome_elo_distribution(bo3_tourney.players, 'bo3')
	find_outcome_elo_distribution(bo1_tourney.players, 'bo1')

	# find likelihood of achieving maximum wins by elo & percentile skill
	handle_max_wins_plot(bo3_tourney.players, bo1_tourney.players, 'Elo')
	handle_max_wins_plot(bo3_tourney.players, bo1_tourney.players, 'EloPercentile')

	# load previously generated simulations
	draft_data_bo3 = pd.read_pickle('bo3_draft_results.pkl')
	draft_data_bo1 = pd.read_pickle('bo1_draft_results.pkl')

	# compare expected and actual win rates
	compare_realized_winrates(draft_data_bo1, draft_data_bo3)
	plot_games_played(draft_data_bo1)

	# find graphs on mean costs/packs per draft
	summary_bo1 = cost_per_draft(draft_data_bo1, 'ModeledWinrate')
	summary_bo3 = cost_per_draft(draft_data_bo3, 'ModeledWinrate')
	plot_prizing(summary_bo1, summary_bo3, 'ModeledWinrate')
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
		on='ModeledWinrate',
		how='outer',
		suffixes=(' Premier', ' Traditional')
	)
	trophy_summary.set_index('ModeledWinrate')[['Trophy Fraction Premier', 'Trophy Fraction Traditional']]\
		.plot(ylabel='Modeled - Implied Trophy Rate')
	plt.savefig(join('Figures', 'TrophyRateFraction.png'), bbox_inches='tight')
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

def compare_realized_winrates(df_bo1, df_bo3):
	'''
	Creates a figure comparing implied win rate from Elo and realized win rate from actual
	matches played
	'''

	# copy input dataframe to avoid modifying in-place
	data_holder = []
	for data in [df_bo1, df_bo3]:
		df = data.copy()

		# calculate actual overall winrate, and winrate in the first round (i.e, fully random pairings)
		df['ModeledWinrate'] = df.TotalWins / (df.TotalWins + df.TotalLosses)
		df['FirstRoundWinrate'] = df.FirstRoundWins / (df.FirstRoundWins + df.FirstRoundLosses)

		# find winrates by percentile 
		df['Implied Win Rate'] = df.groupby('EloPercentile').ImpliedWinrate.transform('mean')
		df['Modeled Win Rate'] = df.groupby('EloPercentile').ModeledWinrate.transform('mean')

		df.rename(columns={'EloPercentile': 'Elo Percentile'}, inplace=True)
		summary = df[['Elo Percentile', 'Implied Win Rate', 'Modeled Win Rate']]\
			[df['Elo Percentile'].between(10, 90)]\
			.drop_duplicates('Elo Percentile').sort_values('Elo Percentile').set_index('Elo Percentile')
		data_holder.append(summary)
	
	# combine bo1 & bo3 data
	summary = pd.merge(
		data_holder[0], 
		data_holder[1],
		left_index=True,
		right_index=True,
		how='outer',
		suffixes=(' Premier', ' Traditional')
	)

	summary['Implied Win Rate'] = summary['Implied Win Rate Premier']
	summary[['Implied Win Rate', 'Modeled Win Rate Premier', 'Modeled Win Rate Traditional']].plot(ylabel='Win Rate')
	plt.savefig(join('Figures', 'ImpliedVsModeledWinrate.png'), bbox_inches='tight')
	plt.close()

	summary['Win Rate Difference -- Premier'] = summary['Modeled Win Rate Premier'] - summary['Implied Win Rate Premier']
	summary['Win Rate Difference -- Traditional'] = summary['Modeled Win Rate Traditional'] - summary['Implied Win Rate Traditional']
	summary['Equality'] = 0
	summary[['Win Rate Difference -- Premier', 'Win Rate Difference -- Traditional', 'Equality']].plot(ylabel='Modeled Win Rate Less Expected Win Rate')
	plt.savefig(join('Figures', 'ImpliedVsModeledWinrateDelta.png'), bbox_inches='tight')
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
		df['ModeledWinrate'] = df.TotalWins / (df.TotalWins + df.TotalLosses)
		df['ModeledWinrate'] = df.ModeledWinrate.round(3)

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
		color='b', label='Cost per Draft (20 Gem Packs) Premier')
	ax.plot(summary.index, summary['Cost200 Premier'], ':', \
		color='b', label='Cost per Draft (200 Gem Packs) Premier')
	ax.plot(summary.index, summary['Cost20 Traditional'], \
		color='r', label='Cost per Draft (20 Gem Packs) Traditional')
	ax.plot(summary.index, summary['Cost200 Traditional'], ':', \
		color='r', label='Cost per Draft (200 Gem Packs) Traditional')
	ax.legend()
	ax.set_ylabel('Net Gems')
	ax.set_xlabel(win_metric)
	plt.savefig(join('Figures', 'NetGemsPerDraft_' + win_metric + '.png'), bbox_inches='tight')
	plt.close()

	# create packs plot (only happens on first execution)
	if win_metric == 'ModeledWinrate':
		summary[['Packs per Draft Premier', 'Packs per Draft Traditional']].plot(ylabel='Packs')
		plt.savefig(join('Figures', 'PacksPerDraft.png'), bbox_inches='tight')
		plt.close()

def set_completion_plots(df_bo1, df_bo3):
	'''
	Creates all plots related to set completion
	'''

	# compile plots by metric
	for win_metric in ['ModeledWinrate', 'EloPercentile']:
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
	if win_metric == 'ModeledWinrate':
		df['ModeledWinrate'] = df.TotalWins / (df.TotalWins + df.TotalLosses)
		df['ModeledWinrate'] = df.ModeledWinrate.round(3)

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
	summary.sort_values('ModeledWinrate').set_index('ModeledWinrate')[['Implied Trophy Rate', 'Modeled Trophy Rate']].plot()
	plt.savefig(join('Figures', 'ImpliedvsActualTrophy_' + draft_type + '.png'), bbox_inches='tight')
	plt.close()

	# collapse by nearest winrate percents, calculate trophy fraction
	summary['TrophyFraction'] = summary['Modeled Trophy Rate'] / summary['Implied Trophy Rate']
	summary['ModeledWinrate'] = (summary.ModeledWinrate * 100).astype(int) / 100
	summary['Trophy Fraction'] = summary.groupby('ModeledWinrate').TrophyFraction.transform('mean')
	summary.drop_duplicates('ModeledWinrate', inplace=True)

	return summary


def find_elo_outcome_distribution(tournament_type, elo, sensitivity = 30):
	'''
	Plots the distribution of outcomes for players at various elos
	'''

	# create appropriate tournament type
	if tournament_type == TournamentType.BEST_OF_3:
		bo3_million = generate_player_distribution('bo3', 'normal', 1000000, std_dev_scalar=1.5)
		tourney = TournamentClient(bo3_million, TournamentType.BEST_OF_3)

	elif tournament_type == TournamentType.BEST_OF_1:
		bo1_million = generate_player_distribution('bo1', 'normal', 1000000, std_dev_scalar=1.5)
		tourney = TournamentClient(bo1_million, TournamentType.BEST_OF_1)

	# execute one draft with 1 million players
	tourney.play_n_drafts(1)

	# find record distribution surrounding the target elo
	players = tourney.players
	players = players[players.Elo.between(elo - sensitivity, elo + sensitivity)]
	players['WinFrequency'] = players.groupby('TotalWins').Gems.transform('count')
	players['WinFrequency'] /= players.WinFrequency.sum()


def compare_distributions(df_normal, df_logistic):
	'''
	Function to compare winrate spread under different distributions assumptions
	'''

	# summarize winrate statistics for easier comparison
	summary_normal = bucket_winrate(df_normal)
	summary_logistic = bucket_winrate(df_logistic)

	merged_summary = pd.merge(
		summary_normal, summary_logistic, 
		on='ImpliedWinrate', how='outer', 
		suffixes=('Normal', 'Logistic'))

	# graph merged data
	merged_summary.sort_values('ImpliedWinrate').set_index('ImpliedWinrate').plot()
	plt.show()

def bucket_winrate(input_df):
	'''
	Helper function to bucket winrate for easier graphing
	'''

	# copy passed dataframe to avoid modifying in place
	df = input_df.copy()
	df['ImpliedWinrate'] *= 100
	df['ImpliedWinrate'] = df.ImpliedWinrate.astype(int)
	df['WinrateFrequency'] = df.groupby('ImpliedWinrate').Elo.transform('count')

	return df[['ImpliedWinrate', 'WinrateFrequency']].drop_duplicates('ImpliedWinrate').sort_values('ImpliedWinrate')



def cost_per_draft_all_metrics(df_bo1, df_bo3, pack_worth=20):
	'''
	Find curve of draft cost by several plausible win metrics, for both bo1 and bo3
	drafts
	'''

	# iterate through each metric, and find mean draft costs
	metric_holder = []
	metric_list = ['ImpliedWinrate', 'FirstRoundWinrate', 'ModeledWinrate']
	for metric in metric_list:
		summary_bo1 = cost_per_draft(df_bo1, metric, pack_worth=pack_worth)
		summary_bo3 = cost_per_draft(df_bo3, metric, pack_worth=pack_worth)
		metric_summary = pd.merge(
			summary_bo1,
			summary_bo3,
			left_index=True,
			right_index=True,
			how='outer',
			suffixes=('BestOfOne', 'BestOfThree')
		)
		metric_holder.append(metric_summary)

	# merge all found metrics together
	full_summary = pd.merge(
		metric_holder[0],
		metric_holder[1],
		left_index=True,
		right_index=True,
		how='outer',
		suffixes=('_' + metric_list[0], '_' + metric_list[1])
	)
	full_summary = pd.merge(
		full_summary,
		metric_holder[2],
		left_index=True,
		right_index=True,
		how='outer'
	)
	full_summary.rename(columns={
		'NetGemsPerDraftBestOfOne': 'NetGemsPerDraftBestOfOne_ModeledWinrate',
		'NetGemsPerDraftBestOfThree': 'NetGemsPerDraftBestOfThree_ModeledWinrate'
	}, inplace=True)

	# plot results
	full_summary.plot()
	plt.show()

	return full_summary


'''
Track average elo delta between rounds
Track frequency of draft outcomes (7 wins, 3-0, 2-1, etc) at various elos,
by naive/sophisticated measurements

'''


