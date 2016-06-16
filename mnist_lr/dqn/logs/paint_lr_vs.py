import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys


fin = open('test_lr.log')

ar = []
for i in fin:
    ar.append(i)
	
ar = [float(x) for x in ar]
print(max(ar))

epoch = 20
episode = int(len(ar)/epoch)
print('episode = ' + str(episode))

##baseline VS last episode

t = range(0, epoch)
#bl = ar[0:epoch]
'''
bl = []
finbl = open('test_lr_baseline.log')
for i in finbl:
	bl.append(float(i))
bl = bl[0:20]
'''
bl = []
finbl = open('../../mnist/logs/test.log')
finbl.readline()
for i in finbl:
	bl.append(round(float(i[1:-6])*10, 2))

##
lastepisode = []
start = (episode-1) * epoch
#start = 8 * epoch
for i in xrange(start, start+epoch):
    lastepisode.append(ar[i])
print(bl)
print(lastepisode)
plt.title('Tuning Learning Rate')
plt.ylabel('Accuracy(%)')
plt.xlabel('Epoch')
plt.plot(t, bl, label='baseline')
plt.plot(t, lastepisode, label='last episode')
plt.legend(loc='lower right')

##


#plt.show()
plt.savefig('vslr.pdf')


