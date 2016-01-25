CXX = pgCC
CXX = g++32
CXX = g++

RM = rm -f

HEISCHAIN_WF_SOURCES = heischain_wf.cc
HEISCHAIN_WF_OBJECTS = heischain_wf.o

# you may want to change this
LDFLAGS = -L/usr/lib64 -L/usr/lib/ -lblas -llapack -lm -lc 
#LDFLAGS = -framework veclib # for Mac OS only

# change by the actual location of your dmtk folder 
INCLUDES = -I./ -I/d/users/adrian/include

CXXCOMPILE = $(CXX) $(INCLUDES) -pthread -ftemplate-depth-150 -DWITH_LAPACK -DWITH_COMPLEX -funroll-loops -O3 -fstrict-aliasing -fno-gcse 

all: tutorial 

tutorial: heischain_wf

heischain_wf: $(HEISCHAIN_WF_OBJECTS)  
	$(CXX) $(LDFLAGS) $(HEISCHAIN_WF_OBJECTS) -o heischain_wf

%.o: %.cc
	$(CXXCOMPILE) -g -c $<

clean:
	$(RM) *.o heischain_wf vectors.dat out core 


