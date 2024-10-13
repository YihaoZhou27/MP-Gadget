#These variables are set to useful defaults, but may be overriden if needed
#MPICC=mpicc
#GSL_LIBS=
#GSL_INCL=
#This is a good optimized build default for gcc
OPTIMIZE =  -fopenmp -O3 -g -Wall -ffast-math -lstdc++
#This is a good non-optimized default for debugging
#OPTIMIZE =  -fopenmp -O0 -g -Wall

#--------------------------------------- Basic operation mode of code
#OPT += VALGRIND     # allow debugging with valgrind, disable the GADGET memory allocator.
#OPT += -DDEBUG      # print a lot of debugging messages
#Disable openmp locking. This means no threading.
#OPT += -DNO_OPENMP_SPINLOCK

#-------------------------------------------- Things for special behaviour
#OPT	+=  -DNO_ISEND_IRECV_IN_DOMAIN     #sparse MPI_Alltoallv do not use ISEND IRECV

#----------- 
#OPT += -DEXCUR_REION  # reionization with excursion set
