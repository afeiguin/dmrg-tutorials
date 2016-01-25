CXX = pgCC
CXX = g++32
CXX = g++

RM = rm -f

HEISCHAIN_SOURCES = heischain.cc
HEISCHAIN_OBJECTS = heischain.o

# you may want to change this
LDFLAGS = -L/usr/lib64 -L/usr/lib/ -lblas -llapack -lm -lc 
#LDFLAGS = -framework veclib # for Mac OS only

# change by the actual location of your dmtk folder 
INCLUDES = -I./ -I/d/users/adrian/include

CXXCOMPILE = $(CXX) $(INCLUDES) -pthread -ftemplate-depth-150 -DWITH_LAPACK -DWITH_COMPLEX -funroll-loops -O3 -fstrict-aliasing -fno-gcse 

all: tutorial 

tutorial: heischain 

heischain: $(HEISCHAIN_OBJECTS)  
	$(CXX) $(LDFLAGS) $(HEISCHAIN_OBJECTS) -o heischain

%.o: %.cc
	$(CXXCOMPILE) -g -c $<

clean:
	$(RM) *.o heischain vectors.dat out core 


