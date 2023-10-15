# define variables
SHELL = /bin/sh

VPATH=./
SOURCE = $(wildcard ./*.cpp)
CPPFILES = $(notdir $(SOURCE))
OBJS = $(patsubst %.cpp, %.o, $(CPPFILES))
EXE = main.exe
CFLAGS = -Wall -fsycl-unnamed-lambda
CC = clang++
INCLUDES = -I ./
LIBS = 

$(EXE): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^

%.o: %.cpp
	${CC} ${CFLAGS} ${INCLUDES} -o $@ -c $^

.PHONY:clean

clean:
	-rm -f *.o *.exe

