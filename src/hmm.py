'''
Xander Beberman
HW2 HMM part 3
Computational Linguistics 2018
python 3
'''

'''
nomenclature:
n - number of states
k - number of letters in alphabet

data stuctures:
initial state matrix (Pi) - n array
	pi[i] = p(starting in state i)
transition matrix (A) - n x n array
	transitions[i,j] = p(move to state j from state i)
emission matrix (B) - n x k array
	emissions[i,l] = p(emitting letter l from state i)
soft counts (SC) - k x n x n array
	soft_counts[l,i,j] = observed probability of emmitting letter l
	when moving from state i to state j
initial soft counts (ISC) - k x n x n array
	initial_soft_counts[l,i,j] = observed probability of emmitting letter l
	when moving from state i to state j in an initial time step
'''

import random
import numpy as np
import argparse

SEP_LEN = 33

def normalize(vec):
	'''
	normalizes a vector to sum to 1
	input:
		vec: vector to normalize
	'''
	s = sum(vec)
	if s == 0:
		return vec
	return vec / s

def print_in_box(text, n):
	'''
	prints text in a box of n dashes
	if text too long, print more dashes
	input:
		text: text to print
		n: number of dashes
	'''
	l = len(text)
	n = max(n, l+4)
	top = '-'*n
	mid = '- ' + text.upper() + ' '*(n-4-l) + ' -'
	print(top)
	print(mid)
	print(top)

def print_zipped_vec(text, vec):
	'''
	prints zipped vector along with text
	input:
		text: text with which to preface vec values
		vec: zipped vector to print
	'''
	for i in vec:
		print(text, str(i[0]), str(i[1]))

def clean(word):
	'''
	strips and adds # to word
	input:
		word: string to clean
	'''
	temp = word.strip().lower()
	if temp[-1] == '#':
		return temp
	return temp + '#'

def smooth(vec, epsilon=0.0000001):
	'''
	smooths a vector by adding a small epsilon to it and renormalizing
	input:
		vec: vector to smooth
		epsilon: value to add
	output:
		new smoothed vector
	'''
	return normalize(vec+epsilon)

class HMM:
	'''
	'''

	def get_alphabet(self, filename):
		'''
		opens file containing corpus and creates alphabet
		input:
			filename: file name of corpus
		output:
			list of unique alphabet characters
			num_words: number of words in corpus
		'''
		if self.verbose:
			print_in_box('INITIALIZING ALPHABET', SEP_LEN)
		alphabet = set('#')
		corpus = open(filename)
		num_words = 0
		for line in corpus:
			alphabet.update(clean(line))
			num_words += 1
		corpus.close()
		return sorted(list(alphabet)), num_words

	def get_wordlist(self, filename):
		'''
		opens file containing corpus and generates list of words
		input:
			filename: file name of corpus
		output:
			list of words in file
		'''
		corpus = open(filename)
		wordlist = [clean(line) for line in corpus]
		corpus.close()
		return wordlist

	def init_matrices(self, uniform=False):
		'''
		initializes random distributions for state transitions,
		initial states, and emissions
		input:
		output:

		'''
		# n vector
		# pi[i] = p(starting in state i)
		pi = normalize(np.random.rand(self.n))

		# n x n matrix
		# transitions[i,j] = p(move to state j from state i)
		transitions = np.random.rand(self.n, self.n)#, float)
		# transitions[0,1] += 2
		# transitions[1,0] += 1

		# n x k matrix
		# emissions[i,j] = p(emitting letter j from state i)
		if uniform:
			emissions = np.ones([self.n, self.k], float)
		else:
			emissions = np.random.rand(self.n, self.k)

		for i in range(self.n):
			transitions[i] = normalize(transitions[i])
			emissions[i] = normalize(emissions[i])

		return pi, transitions, emissions

	def __init__(self, n, filename, verbose, uniform=False):
		if verbose:
			print_in_box('INITIALIZATION', SEP_LEN)
			print('n = ', n)
		self.n = n
		self.verbose = verbose
		self.filename = filename

		self.alphabet, self.num_words = self.get_alphabet(filename)
		self.k = len(self.alphabet)
		self.lookup = {}
		self.lookup.update(zip(self.alphabet, range(self.k)))
		if self.verbose:
			print('ALPHABET: ', *self.alphabet)
			print('k = ', self.k, '\n')

		self.pi, self.transitions, self.emissions = self.init_matrices(uniform)
		# k x n x n matrix
		# soft_counts[l,i,j] = observed probability of emmitting letter l
		# when moving from state i to state j
		self.soft_counts = np.zeros((self.k, self.n, self.n))

		# k matrix
		# letter frequencies
		self.soft_freq = np.zeros(self.k)

		# k x n x n matrix
		# initial_soft_counts[l,i,j] = observed probability of emmitting letter l
		# when moving from state i to state j in an initial time step
		self.initial_soft_counts = np.zeros((self.k, self.n, self.n))

		# k matrix
		# letter frequencies
		self.initial_soft_freq = np.zeros(self.k)


		# if verbose:
		# 	print_in_box('INITIALIZING STATES', SEP_LEN)
		# 	print_states(t=False)
		# 	print('Initialization Complete\n')

	def print_states(self, t=True, e=True, p=True):
		'''
		just prints the states semi-nicely
		'''
		for i in range(self.n):
			print('STATE', i)
			print('-'*SEP_LEN)
			if t:
				print('Transitions')
				print_zipped_vec('\tTo state',
								 sorted(zip(range(self.n), self.transitions[i]),
										key = lambda x: x[1],
										reverse = True))
				print('\tTotal:', sum(self.transitions[i]))
			
			if e:
				print('\nEmissions')
				print_zipped_vec('\tLetter\t',
								 sorted(zip(self.alphabet, self.emissions[i]),
								 		key = lambda x: x[1],
								 		reverse = True))
				print('\tTotal:', sum(self.emissions[i]))
			print()
		if p:
			print('PI')
			print('-'*SEP_LEN)
			print_zipped_vec('State\t',
							 sorted(zip(range(self.n), self.pi),
							 		key = lambda x: x[1],
							 		reverse = True))
			print('\tTotal:', sum(self.pi))
		print()

	def alpha(self, word, verbose=False):
		'''
		calculates alphas for a word given current state of HMM
		alpha is forward probability
		input:
			word: word for which to calculate alpha
		output:
			alpha: table of beta values for each time and state
		'''
		if verbose:
			print_in_box('CALCULATING ALPHA', SEP_LEN)
			print('Word = {}'.format(word))
		l = len(word)
		# n x l array
		# alpha[i,j] = p(state i given output of first j-1 letters)
		alpha = np.zeros((self.n, l+1))
		alpha[:,0] = self.pi
		if verbose:
			print('Time 0 (Pi):')
			for i in range(self.n):
				print('\tState {}: {:.4e}'.format(i, alpha[i,0]))
			print()
		
		for t in range(1,l+1):
			letter = self.lookup[word[t-1]]
			# for i in range(self.n):
			# 	alpha[i,t] = sum(alpha[:,t-1]*self.transitions[:,i]*self.emissions[:,letter])
			# vec = previous alphas * emission probs
			# using matrix/vector operations makes this go faster
			vec = alpha[:,t-1] * self.emissions[:,letter]
			alpha[:,t] = np.dot(vec, self.transitions)
			
			if verbose:
				print('Time {}: \'{}\''.format(t, word[t-1]))
				for i in range(self.n):
					print('\tTo state {}: {:.4e}'.format(i, alpha[i,t]))
				print('Sum of alphas: {:.4e}\n'.format(sum(alpha[:,t])))
		
		# sum(alpha[:,l]) for P(word)
		if verbose:
			print('p({}) = {:.4e}\n'.format(word, sum(alpha[:,l])))
		return alpha

	def beta(self, word, verbose=False):
		'''
		calculates betas for a word given current state of HMM
		beta is backward probability
		input:
			word: word for which to calculate beta
		output:
			beta: table of beta values for each time and state
		'''
		if verbose:
			print_in_box('CALCULATING BETA', SEP_LEN)
			print('Word = {}'.format(word))
		l = len(word)
		# n x l array
		# alpha[i,j] = p(state i given output of first j-1 letters)
		beta = np.zeros((self.n, l+1))
		beta[:,l] = 1
		if verbose:
			print('Time {}:'.format(l))
			for i in range(self.n):
				print('\tState {}: {:.4e}'.format(i, beta[i,l]))
			print()

		for t in range(l-1,-1,-1):
			letter = self.lookup[word[t]]
			for i in range(self.n):
				beta[i,t] = sum(beta[:,t+1]*self.transitions[i,:]*self.emissions[i,letter])
			#print(word[t], letter)
			# vec = previous alphas * emission probs
			# vec = beta[:,t+1] * self.emissions[:,letter]
			# beta[:,t] = np.dot(vec, self.transitions)#)self.emissions[:,letter]*self.transitions[])

			if verbose:
				print('Time {}: \'{}\''.format(t, word[t]))
				for i in range(self.n):
					print('\tFrom state {}: {:.4e}'.format(i, beta[i,t]))
				print('Sum of betas: {:.4e}\n'.format(sum(beta[:,t])))
		
		# beta[:,0] = self.pi * beta[:,0]
		# sum beta[:,0] for P(word)
		if verbose:
			print('p({}) = {:.4e}\n'.format(word, sum(self.pi*beta[:,0])))
		return beta

	def print_plogs(self, wordlist=None, sum_only=True):
		'''
		OUTDATED
		'''
		'''
		prints plogs of words and sum of plogs
		input:
			wordlist: list of words to use. If None, reads whole corpus
			sum_only: if true, only print the sum
		output:
			plog_sum: sum of all plogs
		'''
		plog_sum = 0
		if not sum_only:
			print_in_box('CALCULATING PLOGS', SEP_LEN)
		
		if wordlist == None:
			corpus = open(self.filename)
			for line in corpus:
				word = clean(line)
				alpha = self.alpha(word)
				beta = self.beta(word)
				plog = -np.log2(alpha)
				
				if not sum_only:
					#print('word:', word, '\talpha:', alpha, '\tbeta:', beta, '\tplog:', plog)
					print('word: {}\talpha: {:.3e}\tbeta: {:.3e}\t\tplog: {:.3f}'.format(word, alpha, beta, plog))
				plog_sum += plog
			corpus.close()

		else:
			for word in wordlist:
				alpha = self.alpha(word)
				beta = self.beta(word)
				plog = -np.log2(alpha)
				
				if not sum_only:
					#print('word:', word, '\talpha:', alpha, '\tbeta:', beta, '\tplog:', plog)
					print('word: {}\talpha: {:.3e}\tbeta: {:.3e}\t\tplog: {:.3f}'.format(word, alpha, beta, plog))
				plog_sum += plog
		
		print('Sum of plogs:', plog_sum)
		return plog_sum

	def soft_count(self, word, verbose=True):
		'''
		calculates soft count for each time step in a word
		input:
			word: word to calculate with
			verbose: flag for printing output
		output:
			soft: soft counts for each letter (time) in word and pair of states
		'''
		if verbose:
			print_in_box('CALCULATING SOFT COUNTS', SEP_LEN)
			print('Word = {}'.format(word))

		# get alpha, beta
		alpha = self.alpha(word, False)
		beta = self.beta(word, False)

		# get length, P(O)
		l = len(word)
		total = sum(alpha[:,l])

		# soft count arrays p_t(i,j) = soft[i,j,t]
		soft = np.zeros((self.n, self.n, l))

		for t in range(l):
			letter = self.lookup[word[t]]
			for j in range(self.n):
				soft[:,j,t] = alpha[:,t]*self.transitions[:,j]*self.emissions[:,letter]*beta[j,t+1]/total

			if verbose:
				print('letter: {}'.format(word[t]))
				for i in range(self.n):
					print('\tfrom state {}'.format(i))
					for j in range(self.n):
						print('\t  to state {}: {:.3f}'.format(j, soft[i,j,t]))
				print('\tsums to: {:.3f}\n'.format(sum(sum(soft[:,:,t]))))

		return soft

	def update_soft_counts(self, word, verbose=True):
		'''
		updates table of soft counts and table of initial soft counts
		input:
			word: word to update with
			verbose: flag for printing
		'''
		soft = self.soft_count(word, verbose)
		
		letter = self.lookup[word[0]]
		self.initial_soft_counts[letter,:,:] += soft[:,:,0]
		self.initial_soft_freq[letter] += 1

		for t in range(len(word)):
			letter = self.lookup[word[t]]
			self.soft_counts[letter,:,:] += soft[:,:,0]
			self.soft_freq[letter] += 1

	def print_soft_counts(self, initial=False):
		'''
		prints table of soft counts
		input:
			initial: flag for printing initial soft counts
		'''

		if initial:
			print('Initial soft counts')
		else:
			print('Soft counts')

		print('letter\tfrom\tto\tcount\tfreq')
		for l in range(self.k):
			letter = self.alphabet[l]
			for i in range(self.n):
				for j in range(self.n):
					if initial:
						print('{}\t{}\t{}\t{:.3f}\t{}'.format(letter, i, j, self.initial_soft_counts[l,i,j], int(self.initial_soft_freq[l])))
					else:
						print('{}\t{}\t{}\t{:.3f}\t{}'.format(letter, i, j, self.soft_counts[l,i,j], int(self.soft_freq[l])))

		print()

	def update_all_soft_counts(self, wordlist=None, verbose=True):
		'''
		resets and updates soft counts for each word in the corpus
		input:
			wordlist: list of words to use. If None, goes through each word in corpus
			verbose: flag for output
		output:
			sum of plogs of all words in list
		'''
		# reinitialize
		self.soft_counts = np.zeros((self.k, self.n, self.n))
		self.soft_freq = np.zeros(self.k)
		self.initial_soft_counts = np.zeros((self.k, self.n, self.n))
		self.initial_soft_freq = np.zeros(self.k)

		# count
		plog_sum = 0
		if wordlist == None:
			corpus = open(self.filename)
			for line in corpus:
				word = clean(line)
				self.update_soft_counts(word, verbose)

			corpus.close()

		else:
			for word in wordlist:
				self.update_soft_counts(word, verbose)

		# checking smoothing just in case
		# self.initial_soft_counts += 0.0000001
		# for l in range(self.k):
		# 	self.initial_soft_counts[l,:,:] = self.initial_soft_counts[l,:,:]/np.sum(self.initial_soft_counts[l,:,:], axis=(0,1))

	def sum_plogs(self, wordlist=None):
		'''
		returns sum of all plogs of words
		input:
			wordlist: list of words to use. If None, reads whole corpus
		output:
			sum of plogs
		'''

		plog_sum = 0

		if wordlist == None:
			corpus = open(self.filename)
			for line in corpus:
				word = clean(line)

				l = len(word)
				prob = sum(self.alpha(word)[:,l])
				plog = -np.log2(alpha)
				plog_sum += plog

			corpus.close()

		else:
			for word in wordlist:
				l = len(word)
				prob = sum(self.alpha(word)[:,l])
				plog = -np.log2(prob)
				plog_sum += plog

		return plog_sum

	def update_transitions(self, verbose=True):
		'''
		updates transition matrix
		aij --> (expected number of transitions from state i to state j)
				/(expected number of transitions from state i)
		input:
			verbose: flag for output
		'''
		if verbose:
			print_in_box('Updating transition matrix', SEP_LEN)

		for i in range(self.n):
			# row[j] = sum over all letters l of SC[l,i,j]
			row = np.sum(self.soft_counts[:,i,:], axis=0)
			total = sum(row)
			# print(total)
			self.transitions[i,:] = row/total

			if verbose:
				print('i: {}'.format(i))
				print('\tsum of soft counts from i to j:\n\tj\tsum\tnormalized')
				for j in range(self.n):
					print('\t{}\t{:.3f}\t{:.3f}'.format(j, row[j], self.transitions[i,j]))
				print('\n\tsum of soft counts from i: {:.3f}'.format(total))
				print('\tsum of normalized row: {:.3f}\n'.format(sum(self.transitions[i,:])))

		if verbose:
			print()

	def update_emissions(self, verbose=True):
		'''
		updates emission matrix
		bij --> (expected number of transitions from state i to state j)
				/(expected number of transitions from state i)
		input:
			verbose: flag for output
		'''
		if verbose:
			print_in_box('Updating emission matrix', SEP_LEN)

		for i in range(self.n):
			# row[l] = sum over all states j of SC[l,i,j]
			row = np.sum(self.soft_counts[:,i,:], axis=1)
			total = sum(row)
			# print(total)
			self.emissions[i,:] = row/total

			if verbose:
				print('i: {}'.format(i))
				print('\tsum of soft counts for each letter i to j:\n\tl\tsum\tnormalized')
				for l in range(self.k):
					print('\t{}\t{:.3f}\t{:.3f}'.format(self.alphabet[l], row[l], self.emissions[i,l]))
				print('\n\tsum of soft counts from i: {:.3f}'.format(total))
				print('\tsum of normalized row: {:.3f}\n'.format(sum(self.emissions[i,:])))

		if verbose:
			print()

	def update_pi(self, verbose=True):
		'''
		updates emission matrix
		bij --> (expected number of transitions from state i to state j)
				/(expected number of transitions from state i)
		input:
			verbose: flag for output
		'''
		if verbose:
			print_in_box('Updating initial state matrix', SEP_LEN)

		# row[i] = sum of initial soft counts from i
		row = np.sum(self.initial_soft_counts[:,:,:], axis=(0,2))
		self.pi = row/self.num_words

		if verbose:
			print('number of words: {}\n'.format(self.num_words))
			print('sum of initial soft counts over letters and to states:')
			print('i\tsum\tnormalized')
			for i in range(self.n):
					print('{}\t{:.3f}\t{:.3f}'.format(i, row[i], self.pi[i]))
			print('sum of normalized row: {:.3f}\n'.format(sum(self.pi)))

	def train(self, wordlist=None, iterations=50, epsilon=0.005, verbose_train=True, verbose_step=False):
		'''
		loops through expectation and maximization steps.
		Stops after a certain number of iterations unless sum of plogs
		increases minimally
		input:
			wordlist: list of words to use. If none, reads whole corpus
			iterations: max number of iterations
			epsilon: minimum tolerance for improvement before
				training ends prematurely
			verbose_train: flag for output of training information
			verbose_step: flag for output of subroutine information
		output:
			array of sums of plogs per iteration
		'''
		if verbose_train:
			print_in_box('HMM TRAINING', SEP_LEN)
			print('Max iterations: {}\nEpsilon tolerance: {:.4f}'.format(iterations, epsilon))

		plogs = [0.0000001]

		for itr in range(iterations):
			if verbose_train:
				print('Iteration {}'.format(itr))
			# expectation
			self.update_all_soft_counts(wordlist=wordlist, verbose=False)

			# maximization
			self.update_pi(verbose_step)
			self.update_transitions(verbose_step)
			self.update_emissions(verbose_step)

			# plog check
			plog_sum = self.sum_plogs(wordlist)
			plogs.append(plog_sum)

			distance = abs(plog_sum-plogs[itr])/plogs[itr]
			if distance < epsilon:
				if verbose_train:
					print('Distance too low; stopping training\n')
				break

			if verbose_train:
				print('Sum of plogs: {:.3f}, distance: {:.4e}\n'.format(plog_sum, distance))
			if verbose_step:
				self.print_states()
				# print('plog_sum', plog_sum)
				# print('plog_sum', distance)

			
		if verbose_train:
			print()
		return plogs

	def print_log_ratios(self):
		'''
		prints log ratios of letters
		'''
		if self.n != 2:
			print('Error printing log ratios: n != 2')
			return None
		
		log_ratios = np.log2(self.emissions[0,:]/self.emissions[1,:])
		zipped = zip(self.alphabet, log_ratios)
		zplus = []
		zmin = []
		for l, z in zipped:
			if z >= 0:
				zplus.append((l,z))
			else:
				zmin.append((l,z))

		zplus = sorted(zplus,
					   key = lambda x: x[1],
					   reverse = True)
		zmin = sorted(zmin,
					   key = lambda x: x[1],
					   reverse = False)

		lplus = len(zplus)
		lmin = len(zmin)

		print('LOG RATIOS\n{}'.format('-'*SEP_LEN))
		print('positive\tnegative')
		print('(state 0)\t(state 1)')
		for i in range(max(lplus, lmin)):
			if i < lplus and i < lmin:
				print('{}\t{:.3f}\t{}\t{:.3f}'.format(zplus[i][0], zplus[i][1], zmin[i][0], zmin[i][1]))
			elif i < lplus and i >= lmin:
				print('{}\t{:.3f}'.format(zplus[i][0], zplus[i][1]))
			elif i >= lplus and i < lmin:
				print('\t\t{}\t{:.3f}'.format(zmin[i][0], zmin[i][1]))
		print()



def main():
	# parse 
	parser = argparse.ArgumentParser(description = 'Initializes HMM and calculates forwards and backwards probabilities')
	# parser.add_argument(['help', '-h', '--help'], action='help')
	parser.add_argument('input', action='store', metavar='INPUT',
						help='path to corpus to be fed into the HMM')
	parser.add_argument('-v', '--verbose', action='store_true', required=False,
						help='Run the HMM in verbose mode')
	parser.add_argument('-n', action='store', type=int, default=2,
						help='number of states in the HMM, default 2')
	parser.add_argument('-i', action='store', type=int, default=50,
						help='number of iterations over which to train, default 50')
	parser.add_argument('-e', action='store', type=float, default=0.0000001,
						help='minimum change in plog sum before stopping, default 1e-7')
	parser.add_argument('-t', action='store', type=float, default=[0.5,0.5], nargs=2,
						help='transition probabilities for 0->1, 1->0. Two arguments, default random')
	parser.add_argument('-u', action='store_true', required=False,
						help='use uniform initial emission probabilities, default random')
	
	args = vars(parser.parse_args())
	# print(args)

	hmm = HMM(args['n'], args['input'], args['verbose'], args['u'])
	if args['n'] == 2:
		i,j = args['t']
		hmm.transitions = np.array([[1-i, i], [j, 1-j]])
	
	# hmm.alpha('babi#', args['verbose'])
	# hmm.beta('babi#', args['verbose'])

	wordlist = hmm.get_wordlist(args['input'])
	# print(wordlist)

	plogs = hmm.train(wordlist=wordlist, iterations=args['i'], epsilon=args['e'], verbose_train=True, verbose_step=args['verbose'])
	
	if args['verbose']:
		hmm.print_states()
	print('# iterations: {}'.format(len(plogs)-1))
	print('Final sum of plogs: {:.3f}\n'.format(plogs[-1]))
	hmm.print_log_ratios()
	print('Number of words in corpus: {}\n'.format(hmm.num_words))
	
	# hmm.alpha('babi#', args['verbose'])
	# hmm.beta('babi#', args['verbose'])
	# hmm.alpha('dida#', args['verbose'])
	# hmm.beta('dida#', args['verbose'])

	# hmm.soft_count('babi#', args['verbose'])
	# hmm.soft_count('dida#', args['verbose'])

	# uncomment to test against assignment probabilities with bd.txt
	# hmm.pi = np.array([0.6814, 0.3186])
	# hmm.emissions[0] = [0.1680, 0.1896, 0.1571, 0.1109, 0.3744]
	# hmm.emissions[1] = [0.0633, 0.0916, 0.2787, 0.3272, 0.2393]


if __name__ == '__main__':
	main()