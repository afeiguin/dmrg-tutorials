CXX = pgCC
CXX = g++32
CXX = g++

RM = rm -f

HEISCHAIN_TIME_SOURCES = heischain_time.cc
HEISCHAIN_TIME_OBJECTS = heischain_time.o
HEISCHAIN_TIME0_SOURCES = heischain_time0.cc
HEISCHAIN_TIME0_OBJECTS = heischain_time0.o

# you may want to change this
LDFLAGS = -L/usr/lib64 -L/usr/lib/ -lblas -llapack -lm -lc 
#LDFLAGS = -framework veclib # for Mac OS only

# change by the actual location of your dmtk folder 
INCLUDES = -I./ -I/d/users/adrian/include/

CXXCOMPILE = $(CXX) $(INCLUDES) -pthread -ftemplate-depth-150 -DWITH_LAPACK -DWITH_COMPLEX -funroll-loops -O3 -fstrict-aliasing -fno-gcse 

all: tutorial 

tutorial: heischain_time heischain_time0

heischain_time: $(HEISCHAIN_TIME_OBJECTS)  
	$(CXX) $(LDFLAGS) $(HEISCHAIN_TIME_OBJECTS) -o heischain_time

heischain_time0: $(HEISCHAIN_TIME0_OBJECTS)  
	$(CXX) $(LDFLAGS) $(HEISCHAIN_TIME0_OBJECTS) -o heischain_time0

%.o: %.cc
	$(CXXCOMPILE) -g -c $<

clean:
	$(RM) *.o heischain_time heischain_time0 vectors.dat out core 


