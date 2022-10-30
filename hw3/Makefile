C = hipcc
CFLAGS = -lX11 -std=c++11 -lpthread -I /home/rrytel/Desktop/CImg/

all: heat 

heat: heat.o mat.o

heat.o: heat.cpp
	$(CPP) $(CFLAGS) -c heat.cpp -o heat.o

mat.o: mat.cpp
	$(C) $(CFLAGS) -c mat.cpp -o mat.o
clean:
	rm -f heat.exe *.dat *.o

