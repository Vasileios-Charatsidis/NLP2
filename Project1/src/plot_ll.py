import re
import matplotlib.pyplot as plt

def plot_ll(filename, plot_title):
	likelihoods = []
	f = open(filename)
	lines = f.readlines()
	for line in lines:
		if re.match(r'^Iteration.*', line):
			likelihood = line.split()[6]
			likelihoods.append(likelihood)

	f.close()

	plt.figure()
	plt.plot(likelihoods)
	plt.xlabel('Number of runs')
	plt.ylabel('Likelihood')
	plt.title(plot_title)
	plt.savefig(plot_title + '.png')

if __name__ == '__main__':
	#plot_ll('ibm1.log', 'ibm1 standard')
	#plot_ll('ibm1_smooth.log', 'ibm1 extension: smoothing translation counts')
	#plot_ll('ibm1_add0.log', 'ibm1 extension: adding null words')
	plot_ll('ibm2_uniform.log', 'ibm2 uniform initialization')
	plot_ll('ibm2_random1.log', 'ibm2 first random initialization')
	#plot_ll('ibm2_random2.log', 'ibm2 second random initialization')
	#plot_ll('ibm2_random3.log', 'ibm2 third random initialization')
	#plot_ll('ibm2_ibm1.log', 'ibm2 ibm1 initialization')


