# DraftEloSimulator

This repository contains the code used to produce the model and visualizations discussed in my paper, "Drafting Magic for Fun, and Maybe Profit -- A Dynamic Model, Elo Approach to Draft Analysis", linked here:

## Repository Structure

The code is organized into 4 primary scripts:

- `data_analysis.py`: This script uses draft microdata from 17Lands.com to construct a distribution of win rates for both Premier (best of one) and Traditional (best of three) draft players. The standard deviation of these distributions will be used to calibrate the elo scores of the model.
- `tournament_client.py`: This script constructs a dataframe of players with a distribution of elo scores calibrated against the 17Lands data analyzed before. It simulates large numbers of drafts using either tournament structure (3 rounds, or play until 7 wins/losses) by dynamically matching players against each other using current record.
- `visualizations.py`: This script constructs all the figures used throughout the paper, as well as any smaller analysis necessary to produce them. In particular, this means the construction of a static model that predicts outcomes based on combinatoric application of realized win rates.
- `main.py`: This script controls the program.

## Retrieving Data

The 17Lands data used to calibrate the model can be found online here: TODO URL. Prior to running the model, users should download draft data for any sets they are interested in analyzing, ideally to a folder labeled `Data` inside the repository. 

Note that this webpage cannot simply be scraped for the data directly, because all page content is displayed via Javascript and won't be retrieved with Python-based web parsers.

## Running the Simulation

To run the model locally with default parameters, users should call the `main.py` script, while supplying at least one run option. Options include the following:
  - `process_data` - Executes data analysis, creating statistical inputs needed for the model.
  - `run_simulator` - Creates two tournament clients, one for Premier Drafts and one for Traditional Drafts, each with a default of 100,000 players. Executes 5,000 results for each, and saves the results to file. Number of players can be adjusted with the `--tourney_size` flag, and number of drafts can be adjusted with the `--draft_count` flag.
  - `create_visualizations` - Uses the outputs of the simulator, along with brief other analysis, to create all figures used in the paper.

## Contact Me:

If you have any questions or feedback about the model or paper, or would like to extend the analysis in a new direction, please reach out to me via email at `atnherrick[aT]gmail.com`, or on Twitter at [@atnherrick](https://mobile.twitter.com/atnherrick).

