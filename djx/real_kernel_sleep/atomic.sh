nvcc -arch=native atomic_1.cu -o atomic_1
nvcc -arch=native atomic_2.cu -o atomic_2
nvcc -arch=native atomic_3.cu -o atomic_3
nvcc -arch=native atomic_4.cu -o atomic_4
nvprof ./atomic_1 3
nvprof ./atomic_2 3 
nvprof ./atomic_3 3 
nvprof ./atomic_4 3 