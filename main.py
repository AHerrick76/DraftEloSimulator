'''
Master script to handle all processing
'''

__author__ = 'Austin Herrick'

import argparse
from os.path import join

from data_analysis import process_all_data
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
		prepare_simulation('bo3', requests['tourney_size'], requests['draft_count'])
		prepare_simulation('bo1', requests['tourney_size'], requests['draft_count'])

	# creates all simulation-related charts necessary for article
	if requests['create_visualizations']:
		figure_manager()	

def prepare_simulation(draft_type, tourney_size, draft_count):
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
	df = generate_player_distribution(draft_type, 'normal', tourney_size, std_dev_scalar=1.5)
	tourney = TournamentClient(df, t_type)
	tourney.play_n_drafts(draft_count)

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

	parser.add_argument(
		'--tourney_size', 
		help = """Chooses the number of players competing in the simulated tournament client.
		
		By default, 100,000 players are simulated.""",
		type = int,
	)

	parser.add_argument(
		'--draft_count', 
		help = """Chooses the number of drafts each player in the client will complete.
		
		By default, 5,000 drafts are simulated.""",
		type = int,
	)


	args = parser.parse_args()

	no_options = True
	if args.process_data:
		requests['process_data'] = True
		no_options = False
	if args.run_simulator:
		requests['run_simulator'] = True
		no_options = False
	if args.create_visualizations:
		requests['create_visualizations'] = True
		no_options = False
	if no_options:
		raise Exception("""
			Please provide at least one usage option when calling main.py.

			Options include:
				--process_data
				--run_simulator
				--create_visualizations
			
			Additionally, the size of the draft tournament (default 100,000) and the number of 
			drafts (default 5000) can be calibrated with the following options:
				--tourney_size
				--draft_count
		""")
	if args.tourney_size:
		requests['tourney_size'] = args.tourney_size
	else:
		requests['tourney_size'] = 100000
	if args.draft_count:
		requests['draft_count'] = args.draft_count
	else:
		requests['draft_count'] = 5000


	return requests

# execute model with all requested steps
if __name__ == '__main__':
	requests = parse_input()
	main(requests)

