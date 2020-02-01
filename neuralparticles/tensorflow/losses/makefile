cudalib =  /usr/local/cuda-9.0/lib64
EIGEN_INC = thirt_party/eigen3

$(eval TF_CFLAGS := $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))'))
$(eval TF_LFLAGS := $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))'))


all: tf_approxmatch_so.so tf_approxmatch_g.cu.o tf_nndistance_so.so tf_nndistance_g.cu.o tf_auctionmatch_so.so tf_auctionmatch_g.cu.o


tf_approxmatch_so.so: tf_approxmatch_g.cu.o tf_approxmatch.cpp
	g++ -std=c++11 tf_approxmatch.cpp tf_approxmatch_g.cu.o -o tf_approxmatch_so.so -shared -fPIC $(TF_CFLAGS) $(TF_LFLAGS) -O2


tf_approxmatch_g.cu.o: tf_approxmatch_g.cu
	nvcc -std=c++11 -c -o tf_approxmatch_g.cu.o tf_approxmatch_g.cu $(TF_CFLAGS) $(TF_LFLAGS) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 -D_FORCE_INLINES


tf_nndistance_so.so: tf_nndistance_g.cu.o tf_nndistance.cpp
	g++ -std=c++11 tf_nndistance.cpp tf_nndistance_g.cu.o -o tf_nndistance_so.so -shared -fPIC $(TF_CFLAGS) $(TF_LFLAGS) -lcudart -L $(cudalib) -O2


tf_nndistance_g.cu.o: tf_nndistance_g.cu
	nvcc -std=c++11 -c -o tf_nndistance_g.cu.o tf_nndistance_g.cu -I $(EIGEN_INC) $(TF_CFLAGS) $(TF_LFLAGS) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 -D_FORCE_INLINES -D_MWAITXINTRIN_H_INCLUDED


tf_auctionmatch_so.so: tf_auctionmatch_g.cu.o tf_auctionmatch.cpp
	g++ -std=c++11 tf_auctionmatch.cpp tf_auctionmatch_g.cu.o -o tf_auctionmatch_so.so -shared -fPIC $(TF_CFLAGS) $(TF_LFLAGS) -O2


tf_auctionmatch_g.cu.o: tf_auctionmatch_g.cu
	nvcc -std=c++11 -c -o tf_auctionmatch_g.cu.o tf_auctionmatch_g.cu $(TF_CFLAGS) $(TF_LFLAGS) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 -D_FORCE_INLINES


clean:
	rm *.so
	rm  *.cu.o 
