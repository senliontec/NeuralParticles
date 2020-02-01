cudalib = /usr/local/cuda/lib64
cudainc = /usr/local/cuda/include

$(eval TF_CFLAGS := $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))'))
$(eval TF_LFLAGS := $(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))'))


all: losses tools
losses: losses/tf_approxmatch_so.so losses/tf_nndistance_so.so
tools: tools/tf_grouping_so.so tools/tf_interpolate_so.so tools/tf_sampling_so.so


losses/tf_approxmatch_so.so: losses/tf_approxmatch_g.cu.o losses/tf_approxmatch.cpp
	g++ -std=c++11 losses/tf_approxmatch.cpp losses/tf_approxmatch_g.cu.o -o losses/tf_approxmatch_so.so -shared -fPIC $(TF_CFLAGS) $(TF_LFLAGS) -O2


losses/tf_approxmatch_g.cu.o: losses/tf_approxmatch_g.cu
	nvcc -std=c++11 -c -o losses/tf_approxmatch_g.cu.o losses/tf_approxmatch_g.cu -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 -D_FORCE_INLINES


losses/tf_nndistance_so.so: losses/tf_nndistance_g.cu.o losses/tf_nndistance.cpp
	g++ -std=c++11 losses/tf_nndistance.cpp losses/tf_nndistance_g.cu.o -o losses/tf_nndistance_so.so -shared -fPIC $(TF_CFLAGS) $(TF_LFLAGS) -I$(cudainc) -lcudart -L$(cudalib) -O2


losses/tf_nndistance_g.cu.o: losses/tf_nndistance_g.cu
	nvcc -std=c++11 -c -o losses/tf_nndistance_g.cu.o losses/tf_nndistance_g.cu $(TF_CFLAGS) $(TF_LFLAGS) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 -D_FORCE_INLINES -D_MWAITXINTRIN_H_INCLUDED


tools/tf_grouping_so.so: tools/tf_grouping_g.cu.o tools/tf_grouping.cpp
	g++ -std=c++11 tools/tf_grouping.cpp tools/tf_grouping_g.cu.o -o tools/tf_grouping_so.so -shared -fPIC $(TF_CFLAGS) $(TF_LFLAGS) -I$(cudainc) -lcudart -L$(cudalib) -O2

tools/tf_grouping_g.cu.o: tools/tf_grouping_g.cu
	nvcc -std=c++11 -c -o tools/tf_grouping_g.cu.o tools/tf_grouping_g.cu -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 -D_FORCE_INLINES 

tools/tf_interpolate_so.so: tools/tf_interpolate.cpp
	g++ -std=c++11 tools/tf_interpolate.cpp -o tools/tf_interpolate_so.so -shared -fPIC $(TF_CFLAGS) $(TF_LFLAGS) -I$(cudainc) -lcudart -L$(cudalib) -O2

tools/tf_sampling_so.so: tools/tf_sampling_g.cu.o tools/tf_sampling.cpp
	g++ -std=c++11 tools/tf_sampling.cpp tools/tf_sampling_g.cu.o -o tools/tf_sampling_so.so -shared -fPIC $(TF_CFLAGS) $(TF_LFLAGS) -I$(cudainc) -lcudart -L$(cudalib) -O2

tools/tf_sampling_g.cu.o: tools/tf_sampling_g.cu
	nvcc -std=c++11 -c -o tools/tf_sampling_g.cu.o tools/tf_sampling_g.cu -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2 -D_FORCE_INLINES 


clean:
	rm losses/*.so
	rm losses/*.o
	rm tools/*.so
	rm tools/*.o
