import numpy as np 
import os, sys, time, math, random
import csv
from googlesearch import hits
from tqdm import tqdm
import matplotlib.pyplot as plt
import requests as req


def word2vec_sim_score(w1,w2):
	'''
	Similarity score calculation between word1 and word2 using 
	word2vec_api. Model was google News vector
	'''

	host = 'http://localhost:5000/word2vec/similarity?'
	extension = 'w1='+w1+'&'+'w2='+w2
	r = req.get(host+extension)
	score = r.text
	return float(score)


def read_word_files():
	'''Extract the word pairs, unique words, and 
	pair ratings by humans'''

	words = []	# unique words
	word_pairs = []	# word pairs
	pair_ratings = []	# Human similarity ratings for the pairs

	'''Reading from combined csv file the above variables'''
	with open ('combined.csv') as file:
		csv_reader = csv.reader(file, delimiter = ',')
		for i, row in enumerate(csv_reader):
			if i ==0:
				continue
			if row[0] not in words:
				words.append(row[0])
			word_pairs.append([row[0], row[1]])	
			pair_ratings.append([row[2]])

	return words, word_pairs, pair_ratings


def ngd(w1, w2, pages):
	'''Calculating the NGD between the word1 and word2
	by requesting hits on these from google
	'''

	hw1 = hits(w1)
	time.sleep(2) 
	#sleep time of 2 secs is given so that ip doesn't get blocked

	hw2 = hits(w2)
	time.sleep(2)

	string_sep = '"' + w1 + '" "' + w2 + '"'
	hw1w2 = hits(string_sep)

	min_h = min(math.log(hw1), math.log(hw2))
	max_h = max(math.log(hw2), math.log(hw1))

	ngd = (max_h - math.log(hw1w2))/(math.log(1000*pages) - min_h)

	return ngd


def main():
	words, word_pairs, pair_ratings = read_word_files()
	ngd_word_pairs = []	# NGD values for the word pairs
	word_vec_similarity = []	# Similarity values based on word2Vec model
	pages = hits('the')


	pbar = tqdm(total = len(word_pairs))	# progress bar
	for i,wp in enumerate(word_pairs):
		w1 = wp[0]
		w2 = wp[1]
		ngd_word_pairs.append(ngd(w1, w2, pages))
		word_vec_similarity.append(word2vec_sim_score(w1, w2))
		pbar.update(1)
	pbar.close()

	'''Writing the computed scores to a file
	 because they take lot of time to run'''

	# Writing NGD values to file 
	with open('./ngd.txt', 'w') as f:
		for item in ngd_word_pairs:
			f.write('%s\n' %str(item))

	# Reading the NGD values from the file
	with open('./similarity_scores/ngd.txt', 'r') as f:
		for value in f:
			ngd_word_pairs.append(float(value))

	# Writing word2vec similarity values to file 
	with open('./word2vec.txt', 'w') as f:
		for item in word_vec_similarity:
			f.write('%s\n' %str(item))

	# Reading the word2vec similarity values
	with open('./similarity_scores/word2vec.txt', 'r') as f:
		for value in f:
			word_vec_similarity.append(float(value))


			
	# scaling NGD values between 0 and 10
	max_value = max(ngd_word_pairs)
	min_value = min(ngd_word_pairs)
	ngd_word_pairs = np.array(ngd_word_pairs)
	scaled_ngd_values = (ngd_word_pairs - min_value)/(max_value - min_value)* 10.0
	scaled_ngd_values = 10.0 - scaled_ngd_values


	# scaling word2vec similarity values between 0 and 10
	scaled_word2vec_sim_values = np.array(word_vec_similarity)*10.0

	# Plotting the scatter plots between the scaled NGD and human ratings
	plt.figure()
	plt.scatter(pair_ratings, scaled_ngd_values, marker = 'o')
	plt.xlabel('Human Similarity Ratings')
	plt.ylabel('Scaled NGD')
	plt.axis([0, 10, 0, 10])
	plt.title('Strong correlation between human ratings and NGD values')

	''' Plotting the sctter plots between the scaled word2vec similarity values
	 and human ratings'''
	plt.figure()
	plt.scatter(pair_ratings, scaled_word2vec_sim_values, marker = 'o')
	plt.xlabel('Human Similarity Ratings')
	plt.ylabel('Scaled Word2Vec Similarity Scores')
	plt.axis([0, 10, 0, 10])
	plt.title('Strong correlation between human ratings and word2vec similarity scores')

	plt.show()


if __name__ == '__main__':
	main()