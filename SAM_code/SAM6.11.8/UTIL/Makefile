# Makefile to compile conversion utilities. You need to set include and library paths for NETCDF
#

PLATFORM := $(shell uname -s)

# uncomment line for your platform or add your own:

#--------------------------------------------------------

ifeq ($(PLATFORM),Linux)

INC_NETCDF   := /burg/opt/netcdf-fortran-4.5.3/include
LIB_NETCDF   := /burg/opt/netcdf-fortran-4.5.3/lib

##FF = xlf90 -qsuffix=f=f -qfixed=132 -I$(INC_NETCDF)
#DFLAGS = -L${LIB_NETCDF} -lnetcdf 
#LDFLAGS = -L${LIB_NETCDF} -L/usr/lib -lnetcdf -lpthread
FF77 = /usr/mpi/gcc/openmpi-4.1.5a1/bin/mpif90 -c -ffixed-form -extend_source
FF90 = /usr/mpi/gcc/openmpi-4.1.5a1/bin/mpif90 -c
FF  = /usr/mpi/gcc/openmpi-4.1.5a1/bin/mpif90 -I$(INC_NETCDF)
CC = mpicc -c -DLINUX

FFLAGS = -O3 
#FFLAGS = -g -ftrapuv -check all

FFLAGS += -I${INC_MPI} -I${INC_NETCDF}
LD = /usr/mpi/gcc/openmpi-4.1.5a1/bin/mpif90
LDFLAGS = -L${LIB_NETCDF} -lnetcdff


endif
#---------------------------------------------------------

VPATH = ./SRC

all: bin2D2nc bin3D2nc 2Dbin2nc 2Dbin2nc_mean bin3D2nc_mean 2Dcom2nc 2Dcom2nc_mean com3D2nc com3D2nc_mean com2D2nc stat2nc isccp2nc modis2nc com3D2nc_sep 2Dbin2nc_sep 2Dcom_sep2one 2Dbin_sep2one com3D_sep2one bin3D_sep2one glue_movie_raw com3D2bin misr2nc

.f:   
	#$(FF77) -o $@ -I./SRC $< ./SRC/hbuf_lib.f ./SRC/cape.f ./SRC/cin.f $(LDFLAGS) 
	$(FF) -o $@ -I./SRC $< ./SRC/hbuf_lib.f ./SRC/cape.f ./SRC/cin.f $(LDFLAGS)

clean: 
	rm bin* com* stat* 2* isccp* modis* misr* *.o glue*

