/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UROM_ALLGATHER_H_
#define UROM_ALLGATHER_H_
#include "../cl_urom_coll.h"
#include "../../../tl/ucp/tl_ucp.h"

enum
{
    UCC_CL_UROM_ALLGATHER_ALG_FULL,
    UCC_CL_UROM_ALLGATHER_ALG_LAST,
};

extern ucc_base_coll_alg_info_t
    ucc_cl_urom_allgather_algs[UCC_CL_UROM_ALLGATHER_ALG_LAST + 1];


static inline int ucc_cl_urom_allgather_alg_from_str(const char *str)
{
    int i;
    for (i = 0; i < UCC_CL_UROM_ALLGATHER_ALG_LAST; i++) {
        if (0 == strcasecmp(str, ucc_cl_urom_allgather_algs[i].name)) {
            break;
        }
    }
    return i;
}

#endif
