/**
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "cl_urom.h"
#include "utils/ucc_malloc.h"

#include <string.h>

#include "utils/arch/cuda_def.h"

ucc_status_t memcpy_init(ucc_cl_urom_lib_t *cl_lib)
{
#if 0 
    cudaStreamCreateWithFlags(&cl_lib->cuda_stream, cudaStreamNonBlocking);
#endif
    return UCC_OK;
}

ucc_status_t memcpy_nb(void *dst,
              void *src,
              ucc_memory_type_t src_mem_type,
              ucc_memory_type_t dst_mem_type,
              size_t n,
              ucc_cl_urom_lib_t *cl_lib)
{
#ifndef HAVE_CUDA
    if (src_mem_type == UCC_MEMORY_TYPE_CUDA ||
        dst_mem_type == UCC_MEMORY_TYPE_CUDA) {
        cl_error(cl_lib, "Unsupported operation. Did you build UCC with CUDA support?");
        return UCC_ERR_NOT_SUPPORTED;
    }
#else 
    /* copy from host to cuda */
    if (dst_mem_type == UCC_MEMORY_TYPE_CUDA) {
        printf("IM HOST->CUDA");
#if 0
        cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, cl_lib->cuda_stream);
#endif
        return UCC_OK;
    } else if (src_mem_type == UCC_MEMORY_TYPE_CUDA) {
        /* copy from cuda to host */
        printf("IM CUDA->HOST");
#if 0
        cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, cl_lib->cuda_stream);
#endif
        return UCC_OK;
    }
#endif
    else {
        /* copy from host to host */
        memcpy(dst, src, n);
    }
    return UCC_OK;
}

ucc_status_t memcpy_sync(ucc_cl_urom_lib_t *cl_lib)
{
#if 0
    cudaStreamSynchronize(cl_lib->cuda_stream);
#endif
    return UCC_OK;
}
