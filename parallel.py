from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

rank = comm.Get_rank()

# number of processes
thread_num = comm.Get_size()

file_name = "fwTest.txt"

matrix = np.loadtxt(file_name, dtype=int)

expected_output = np.loadtxt("fwTestResult.txt", dtype=int)

row_num = matrix.shape[0]

rows_per_thread = row_num/thread_num 
threads_per_row = thread_num/row_num
start_row = int(rows_per_thread * rank)
end_row = int(rows_per_thread * (rank+1))

for k in range(row_num):
    owner = int(thread_num/row_num*k)
    matrix[k] = comm.bcast(matrix[k], root = owner)
    for x in range(start_row, end_row):
        for y in range(row_num):
            if matrix[x,y] > matrix[x,k] + matrix[k,y]:
                matrix[x,y] = matrix[x,k] + matrix[k,y]
                
if comm.Get_rank() is 0:
    for k in range(end_row, row_num):
        owner = int(thread_num/row_num*k)
        matrix[k] =  comm.recv(source = owner, tag = k)
        
    if np.array_equal(matrix,expected_output):
      print("Successful execution!")
    else:
          print("Issues running")
else:
    for k in range(start_row,end_row):
        comm.send(matrix[k], dest = 0, tag = k)