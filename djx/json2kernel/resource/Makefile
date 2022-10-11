CC=hipcc

INCLUDES= -I/opt/rocm/hip/include

SUBDIRS = ${shell ls -d */ | sed 's/\///' -}


CURRENT_PATH = ${shell pwd}

CURRENT_DIR = ${shell basename $(CURRENT_PATH)}

ARCH=gfx906


%.raw.co: %.cu
	$(CC) $(INCLUDES) --save-temps --genco --offload-arch=${ARCH} $< -o $@
	rm *.bc *.cui *.o *.out *.txt