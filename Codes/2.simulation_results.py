from NTM import *
            
import sys
import datetime
from copy import deepcopy

#if (len(sys.argv) < 3):
#	print ("Usage: provide 2 arguments as N=ratio_sum, k=no_of_reagents, MAX_ITER(optional)")
#else:
dir_name = 'results'

for with_heuristic in [True, False]:
	for N in [16,64,256,1024]:
		for k in range(3,13) :	
			print ("Generating results for, N={}, K={}, With_Heuristic={}".format(N, k, with_heuristic))
			start_time = datetime.datetime.now()
	#		N = int(sys.argv[1])
	#		k = int(sys.argv[2])
			INIT_MAX = 10000	
			RANDOM_MAX = 50000
			random_count = INIT_MAX

			T_avg, C_avg, V_avg, A_avg = 0, 0, 0, 0
			Total = 0

			create_directory(dir_name)
			
			file_name = 'without_{}_{}.txt'.format(N, k)
			if with_heuristic:
				file_name = 'with_{}_{}.txt'.format(N, k)
			f = open(os.path.join(dir_name, file_name), 'w')

			file_name = 'common_without.txt'
			if with_heuristic:
				file_name = 'common_with.txt'
			f_common = open(os.path.join(dir_name, file_name), 'a')

			f.write ('Starting for Ratio Sum N={}, No. of Reagents={}\n'.format(N, k))
			for itr, ratio in enumerate(partitionfunc(N, k)):
				if random_count > RANDOM_MAX:
					f.write ("####### MAX_ITER={} reached...\n".format(RANDOM_MAX))
					break
				if itr > INIT_MAX and random.randint(1,100)%2 == 0:
					pass

				R = [('r{}'.format(idx+1), item) for idx, item in enumerate(ratio)]
				T = genMix(R, 4)
				
				if with_heuristic:
					T = hda(T)
				
				A = ntm(T, [0,0])

				timeCount = getTimeCount(A)
				cellCount = getCellCount(A)
				_, valveCount, actuationCount = getValveCount(A)
			
				T_avg += timeCount
				C_avg += cellCount
				V_avg += valveCount
				A_avg += actuationCount
				Total += 1

				if itr > INIT_MAX:
					random_count += 1

			
				f.write (str([timeCount, cellCount, valveCount, actuationCount, ratio])+'\n')
			end_time = datetime.datetime.now()
			f.write ('*'*10 + ' FINISHED in {} seconds...'.format(end_time - start_time) + '\n')
			T_avg = T_avg/float(Total)
			C_avg = C_avg/float(Total)
			V_avg = V_avg/float(Total)
			A_avg = A_avg/float(Total)
			f.write ('Total ratios found={}, Avg. Time={}, Avg. CellCount={}, Avg. ValveCount={}, Avg. ActuationCount={}\n'.format( Total, T_avg, C_avg, V_avg, A_avg ))
			time_diff = (end_time-start_time).seconds
			f_common.write( ', '.join(str(i) for i in [N, k, Total, time_diff, T_avg, C_avg, V_avg, A_avg]) + '\n')

			f.close()
			f_common.close()
