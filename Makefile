all: main.c
		 gcc -Wall -o3 -g -o Decoder main.c decoder.c -lrt -lm -std=gnu99

test: test.c
		 gcc -Wall -o3 -g -o Test test.c -lm -std=gnu99

openDecoder:	
	gcc main.c openDecoder.c -g -lm -lrt -std=gnu99 -O3 -lOpenCL -o openDecoder

clean:
	clear
	rm -rf *.o Decoder Test openDecoder
