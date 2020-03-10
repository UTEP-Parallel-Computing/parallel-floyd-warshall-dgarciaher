from mpi4py import MPI
import numpy as np
import time
    
comm = MPI.COMM_WORLD

rank = comm.Get_rank()

# number of processes
thread_num = comm.Get_size()

if rank == 0:
    total_start = time.time()

file_name = "fwTest.txt"

matrix = np.loadtxt(file_name, dtype=int)

expected_output = np.loadtxt("fwTestResult.txt", dtype=int)

row_num = matrix.shape[0]

rows_per_thread = row_num/thread_num 
# threads_per_row = thread_num/row_num    don't think I am using this
start_row = int(rows_per_thread * rank)
end_row = int(rows_per_thread * (rank+1))

print("\n")

start= time.time()
for k in range(row_num):
    owner = int(thread_num/row_num*k)
    matrix[k] = comm.bcast(matrix[k], root = owner)
    for x in range(start_row, end_row):
        for y in range(row_num):
            if matrix[x,y] > matrix[x,k] + matrix[k,y]:
                matrix[x,y] = matrix[x,k] + matrix[k,y]

stop= time.time()
thread_time = stop-start
print("time for thread ", rank, ": ", thread_time)

if comm.Get_rank() == 0:
    for k in range(end_row, row_num):
        owner = int(thread_num/row_num*k)
        matrix[k] =  comm.recv(source = owner, tag = k)
        
    if np.array_equal(matrix,expected_output):
      print("----------------------------------------------")
      print("------Running program with ", thread_num, " threads-------")
      print("----------------------------------------------")
      print("\n\n")
      print("Successful execution!")
    else:
          print("Issues running")
    
else:
    for k in range(start_row,end_row):
        
if comm.Get_rank() == 0:
    total_stop = time.time()
    print("\n")
    print("Total program time: ", (total_stop-total_start))
    print("\n")
    print("------------subarray-------------")
    for row in range(1, 10):
            for col in range(1, 10):
                print("{" , matrix[row][col] , "}" , end="")
            print("")
    
    
    print("\n")
    print("------------expected output-------------")      
    for row in range(1, 10):
            for col in range(1, 10):
                print("{" , expected_output[row][col] , "}" , end="")
            print("")
    