import re
import matplotlib.pyplot as plt

def plot_ll(filename, plot_title, to_scale = False):
	likelihoods = []
	f = open(filename)
	lines = f.readlines()
	for line in lines:
		if re.match(r'^Iteration.*', line):
			likelihood = float(line.split()[6])
			likelihoods.append(likelihood)

	f.close()

	plt.figure()
	plt.plot(likelihoods)
	plt.xlabel('Iterations')
	plt.ylabel('Likelihood')
	if to_scale:
	  plt.ylim(-2e7,-1e7)
	#plt.title(plot_title)
	plt.savefig(plot_title + '.png')

def plot_AER(filename, plot_title):
	AERS = []
	f = open(filename)
	lines = f.readlines()
	for line in lines:
		if re.match(r'^Iteration.*', line):
			AER = float(line.split()[8])
			AERS.append(AER)

	f.close()

	plt.figure()
	plt.plot(AERS)
	plt.xlabel('Iterations')
	plt.ylabel('AER')
	plt.ylim(0,1)
	#plt.title(plot_title)
	plt.savefig(plot_title + '.png')

def get_best_AER(filename):
	f = open(filename)
	best_AER = 1.0
	lines = f.readlines()
	for line in lines:
		if re.match(r'^Iteration.*', line):
			AER = float(line.split()[8])
			if AER < best_AER:
				best_AER = float(AER)

	f.close()

	return best_AER


if __name__ == '__main__':
	plot_ll('ibm1.log', 'log_likelihood_ibm1_standard')
	plot_ll('ibm1_smooth.log', 'log_likelihood_ibm1_extension_smoothing_translation_counts')
	plot_ll('ibm1_add0.log', 'log_likelihood_ibm_extension_adding_null_words')
	plot_ll('ibm2_uniform.log', 'log_likelihood_ibm2_uniform_initialization', True)
	plot_ll('ibm2_random1.log', 'log_likelihood_ibm2_first_random_initialization', True)
	plot_ll('ibm2_random2.log', 'log_likelihood_ibm2_second_random_initialization', True)
	plot_ll('ibm2_random3.log', 'log_likelihood_ibm2_third_random_initialization', True)
	plot_ll('ibm2_ibm1.log', 'log_likelihood_ibm2_ibm1_initialization', True)

	plot_AER('ibm1.log', 'AER_ibm1_standard')
	plot_AER('ibm1_smooth.log', 'AER_ibm1_extension_smoothing_translation_counts')
	plot_AER('ibm1_add0.log', 'AER_ibm1_extension_adding_null_words')
	plot_AER('ibm2_uniform.log', 'AER_ibm2_uniform_initialization')
	plot_AER('ibm2_random1.log', 'AER_ibm2_first_random_initialization')
	plot_AER('ibm2_random2.log', 'AER_ibm2_second_random_initialization')
	plot_AER('ibm2_random3.log', 'AER_ibm2_third_random_initialization')
	plot_AER('ibm2_ibm1.log', 'AER_ibm2_ibm1_initialization')




	AER_IBM1 = get_best_AER('ibm1.log')
	AER_IBM1_S = get_best_AER('ibm1_smooth.log')
	AER_IBM1_0 = get_best_AER('ibm1_add0.log')
	AER_IBM2_U = get_best_AER('ibm2_uniform.log')
	AER_IBM2_R1 = get_best_AER('ibm2_random1.log')
	AER_IBM2_R2 = get_best_AER('ibm2_random2.log')
	AER_IBM2_R3 = get_best_AER('ibm2_random3.log')
	AER_IBM1_IBM2 = get_best_AER('ibm2_ibm1.log')

	print AER_IBM1, AER_IBM1_S, AER_IBM1_0, AER_IBM2_U, AER_IBM2_R1, AER_IBM2_R2, AER_IBM2_R3, AER_IBM1_IBM2

