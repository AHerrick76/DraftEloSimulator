'''
Master script to handle all processing
'''

__author__ = 'Austin Herrick'

import argparse
from os.path import join

from distribution_finder import process_all_data
from visualizations import figure_manager
from tournament_client import generate_player_distribution, TournamentClient, TournamentType

def main(requests):
	'''
	Handles all processing. Depending on function call, does one of the following:
	
	- Find summary statistics from all 17lands data
	- Creates & caches large draft simulations
	- Creates all figures / results from draft simulations
	'''

	# loads 17lands data, finds summary statistics, and produces data-related charts
	if requests['process_data']:
		process_all_data()
	
	# creates 100k player tournament clients, runs 5000 drafts in each, and caches results
	if requests['run_simulator']:
		prepare_simulation('bo3')
		prepare_simulation('bo1')

	# creates all simulation-related charts necessary for article
	if requests['create_visualizations']:
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
	tourney.players.to_pickle(join('Data', draft_type + '_draft_results.pkl'))


def parse_input():
	"""
	Construct parser and fetch relevent flags for requested execution
	"""

	# construct dictionary to populate with selected flags
	requests = {
		'process_data': False,
		'run_simulator': False,
		'create_visualizations': False
	}

	parser = argparse.ArgumentParser()

	parser.add_argument(
		'--process_data',
		help="""Loads 17lands data, finds summary statistics, and produces data-related charts""",
		action = 'store_true'
	)

	parser.add_argument(
		'--run_simulator',
		help="""Creates 100k player tournament clients, runs 5000 drafts in each, and caches results""",
		action = 'store_true'
	)

	parser.add_argument(
		'--create_visualizations',
		help="""Creates all simulation-related charts necessary for article""",
		action = 'store_true'
	)

	args = parser.parse_args()

	if args.process_data:
		requests['process_data'] = True
	if args.run_simulator:
		requests['run_simulator'] = True
	if args.create_visualizations:
		requests['create_visualizations'] = True

	return requests

# execute model with all requested steps
if __name__ == '__main__':
	requests = parse_input()
	main(requests)

