Xander Beberman
Hidden Markov Model for Computational Linguistics at UChicago Spring 2018


Contains:

src/
-----------------
hmm.py
---------------------------------
Run this to get expected output. Run in verbose mode to see all training states.

usage: hmm.py [-h] [-v] [-n N] [-i I] [-e E] [-t T T] [-u] INPUT

Initializes HMM and calculates forwards and backwards probabilities.
Outputs characters in alphabet associated with each state, along with probabilities.

positional arguments:
  INPUT          path to corpus to be fed into the HMM

optional arguments:
  -h, --help     show this help message and exit
  -v, --verbose  Run the HMM in verbose mode
  -n N           number of states in the HMM, default 2
  -i I           number of iterations over which to train, default 50
  -e E           minimum change in plog sum before stopping, default 1e-7
  -t T T         transition probabilities for 0->1, 1->0. Two arguments,
                 default random
  -u             use uniform initial emission probabilities, default random

Stop condition:
let S[i] be sum of plogs after iteration i. I generate a distance metric as

	| S[i] - S[i-1] | / S[i]

if this distance is below a certain epsilon value, stop training. I chose this method to account for different corpora having different plog sums. I chose default epsilon = 1E-7 since it tended to stop early around half the time, and would stop after 10-20 iterations.


make_corpus.py
---------------------------------
python script to generate fake corpus.


corpora/
-----------------
english1000.txt
---------------------------------
list of 1000 english words


bd.txt
---------------------------------
corpus of only "badi" and "dida"


fake.txt
---------------------------------
corpus of fake words made from alternating consonants and vowels.



