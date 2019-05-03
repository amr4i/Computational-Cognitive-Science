import numpy as np 

def drawFromADist(p):
	if np.sum(p) == 0:
		p = 0.05*np.ones((1,len(p)))
	p = p/np.sum(p)
	c = np.cumsum(p)
	idx = np.where((np.random.rand() - np.cumsum(p))<0)
	sample = np.min(idx)
	out = np.zeros(p.size)
	out[sample] = 1
	return out

if __name__ == "__main__":
	drawFromADist([0,0,0,0,0])