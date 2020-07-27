CC=g++
CFLAGS= -g -Wall
DIR_LIB=lib
DIR_HEAD=include
ALL= tf_example

tf_example: obj/tf_functions.o obj/tf_example.o
	$(CC) $(CFLAGS) obj/tf_example.o obj/tf_functions.o -I$(DIR_HEAD) -L$(DIR_LIB) -ltensorflow -o tf_example

obj/tf_example.o: src/tf_example.cpp
	$(CC) $(CFLAGS) -I$(DIR_HEAD) -c src/tf_example.cpp -o obj/tf_example.o

obj/tf_functions.o: src/tf_functions.cpp include/tf_functions.hpp
	$(CC) $(CFLAGS) -I$(DIR_HEAD) -c src/tf_functions.cpp -o obj/tf_functions.o

clean:
	rm -f tf_example obj/*.o
