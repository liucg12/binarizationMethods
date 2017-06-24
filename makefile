CC = g++
CFLAGS = -g -Wall
SRCS = bitPlaneSlicing.cpp
PROG = bitPlaneSlicing

OPENCV = `pkg-config --cflags --libs /usr/local/Cellar/opencv3/3.2.0/lib/pkgconfig/opencv.pc`
LIBS = $(OPENCV)

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)