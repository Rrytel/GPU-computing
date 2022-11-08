C = hipcc
CFLAGS = -lX11 -std=c++11 -lpthread -I /home/rrytel/Desktop/CImg/

all: hw3 

hw3: hw3.o mat.o

hw3.o: hw3.cpp
	$(CPP) $(CFLAGS) -c hw3.cpp -o hw3.o

mat.o: mat.cpp
	$(C) $(CFLAGS) -c mat.cpp -o mat.o
clean:
	rm -f hw3.exe *.dat *.o

