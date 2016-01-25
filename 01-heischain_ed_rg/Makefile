CXX = pgCC
CXX = g++32
CXX = g++

RM = rm -f

HEISCHAIN_ED_RG_SOURCES = heischain_ed_rg.cc
HEISCHAIN_ED_RG_OBJECTS = heischain_ed_rg.o

# you may want to change this
LDFLAGS = -L/usr/lib64 -L/usr/lib/ -lblas -llapack -lm -lc 
#LDFLAGS = -framework veclib # for Mac OS only

# change by the actual location of your dmtk folder 
INCLUDES = -I./ -I/d/users/adrian/include

CXXCOMPILE = $(CXX) $(INCLUDES) -pthread -ftemplate-depth-150 -DWITH_LAPACK -DWITH_COMPLEX -funroll-loops -O3 -fstrict-aliasing -fno-gcse 

all: tutorial 

tutorial: heischain_ed_rg

heischain_ed_rg: $(HEISCHAIN_ED_RG_OBJECTS)  
	$(CXX) $(LDFLAGS) $(HEISCHAIN_ED_RG_OBJECTS) -o heischain_ed_rg

%.o: %.cc
	$(CXXCOMPILE) -g -c $<

clean:
	$(RM) *.o heischain_ed_rg vectors.dat core out 


