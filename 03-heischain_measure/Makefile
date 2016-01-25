CXX = pgCC
CXX = g++32
CXX = g++

RM = rm -f

HEISCHAIN_MEASURE_SOURCES = heischain_measure.cc
HEISCHAIN_MEASURE_OBJECTS = heischain_measure.o

# you may want to change this
LDFLAGS = -L/usr/lib64 -L/usr/lib/ -lblas -llapack -lm -lc 
#LDFLAGS = -framework veclib # for Mac OS only

# change by the actual location of your dmtk folder 
INCLUDES = -I./ -I/d/users/adrian/include

CXXCOMPILE = $(CXX) $(INCLUDES) -pthread -ftemplate-depth-150 -DWITH_LAPACK -DWITH_COMPLEX -funroll-loops -O3 -fstrict-aliasing -fno-gcse 

all: tutorial 

tutorial: heischain_measure

heischain_measure: $(HEISCHAIN_MEASURE_OBJECTS)  
	$(CXX) $(LDFLAGS) $(HEISCHAIN_MEASURE_OBJECTS) -o heischain_measure

%.o: %.cc
	$(CXXCOMPILE) -g -c $<

clean:
	$(RM) *.o heischain_measure vectors.dat out core 


