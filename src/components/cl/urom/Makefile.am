#
# Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#

alltoall =              \
    alltoall/alltoall.h \
    alltoall/alltoall.c

allgather =              \
    allgather/allgather.h \
    allgather/allgather.c

alltoallv =               \
    alltoallv/alltoallv.h \
    alltoallv/alltoallv.c

allreduce =               \
    allreduce/allreduce.h \
    allreduce/allreduce.c

reduce_scatter =               \
    reduce_scatter/reduce_scatter.h \
    reduce_scatter/reduce_scatter.c

reduce_scatterv =               \
    reduce_scatterv/reduce_scatterv.h \
    reduce_scatterv/reduce_scatterv.c

sources =             \
	cl_urom.h         \
	cl_urom.c         \
	cl_urom_lib.c     \
	cl_urom_context.c \
	cl_urom_team.c    \
	cl_urom_coll.c    \
	$(alltoall)		  \
	$(allgather)      \
	$(alltoallv)      \
	$(reduce_scatter) \
	$(reduce_scatterv) \
	$(allreduce)

module_LTLIBRARIES         = libucc_cl_urom.la
libucc_cl_urom_la_SOURCES  = $(sources)
libucc_cl_urom_la_CPPFLAGS = $(AM_CPPFLAGS) $(BASE_CPPFLAGS) $(UROM_CPPFLAGS)
libucc_cl_urom_la_CFLAGS   = $(BASE_CFLAGS)
libucc_cl_urom_la_LDFLAGS  = -version-info $(SOVERSION) --as-needed $(UROM_LDFLAGS)
libucc_cl_urom_la_LIBADD   = $(UROM_LIBADD) $(UCC_TOP_BUILDDIR)/src/libucc.la

include $(top_srcdir)/config/module.am
