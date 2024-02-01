/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef ALLTOALLV_H_
#define ALLTOALLV_H_

#include "../tl_ucp.h"
#include "../tl_ucp_coll.h"

enum {
    UCC_TL_UCP_ALLTOALLV_ALG_PAIRWISE,
    UCC_TL_UCP_ALLTOALLV_ALG_HYBRID,
    UCC_TL_UCP_ALLTOALLV_ALG_NSARKA,
    UCC_TL_UCP_ALLTOALLV_ALG_LAST
};

#define UCC_TL_UCP_ALLTOALLV_DEFAULT_ALG_SELECT_STR                           \
    "alltoallv:host:[64-inf]:@hybrid"

extern ucc_base_coll_alg_info_t
             ucc_tl_ucp_alltoallv_algs[UCC_TL_UCP_ALLTOALLV_ALG_LAST + 1];

ucc_status_t ucc_tl_ucp_alltoallv_init(ucc_tl_ucp_task_t *task);

ucc_status_t ucc_tl_ucp_alltoallv_pairwise_init(ucc_base_coll_args_t *coll_args,
                                                ucc_base_team_t      *team,
                                                ucc_coll_task_t     **task_h);

ucc_status_t ucc_tl_ucp_alltoallv_hybrid_init(ucc_base_coll_args_t *coll_args,
                                              ucc_base_team_t      *team,
                                              ucc_coll_task_t     **task_h);

ucc_status_t ucc_tl_ucp_alltoallv_nsarka_init(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t *team,
                                               ucc_coll_task_t **task_h);

ucc_status_t
ucc_tl_ucp_alltoallv_nsarka_task_init(ucc_base_coll_args_t *coll_args,
                                     ucc_base_team_t *     team,
                                     ucc_tl_ucp_task_t *   task);

ucc_status_t
ucc_tl_ucp_alltoallv_nsarka_start(ucc_coll_task_t *coll_task);

void ucc_tl_ucp_alltoallv_nsarka_progress(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_alltoallv_nsarka_finalize(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_alltoallv_nsarka_allgather_info_finalize(
    ucc_service_coll_req_t *scoll_req, ucc_tl_ucp_task_t *sw_task);


ucc_status_t ucc_tl_ucp_alltoallv_pairwise_init_common(ucc_tl_ucp_task_t *task);

#define ALLTOALLV_CHECK_INPLACE(_args, _team)                                  \
    do {                                                                       \
        if (UCC_IS_INPLACE(_args)) {                                           \
            tl_error(UCC_TL_TEAM_LIB(_team),                                   \
                     "inplace alltoallv is not supported");                    \
            status = UCC_ERR_NOT_SUPPORTED;                                    \
            goto out;                                                          \
        }                                                                      \
    } while (0)

#define ALLTOALLV_CHECK_USERDEFINED_DT(_args, _team)                           \
    do {                                                                       \
        if (!ucc_coll_args_is_predefined_dt(&(_args), UCC_RANK_INVALID)) {     \
            tl_error(UCC_TL_TEAM_LIB(_team),                                   \
                     "user defined datatype is not supported");                \
            status = UCC_ERR_NOT_SUPPORTED;                                    \
            goto out;                                                          \
        }                                                                      \
    } while (0)

#define ALLTOALLV_TASK_CHECK(_args, _team)                                     \
    ALLTOALLV_CHECK_INPLACE((_args), (_team));                                 \
    ALLTOALLV_CHECK_USERDEFINED_DT((_args), (_team));


static inline int ucc_tl_ucp_alltoallv_alg_from_str(const char *str)
{
    int i;
    for (i = 0; i < UCC_TL_UCP_ALLTOALLV_ALG_LAST; i++) {
        if (0 == strcasecmp(str, ucc_tl_ucp_alltoallv_algs[i].name)) {
            break;
        }
    }
    return i;
}

#define ALLTOALLV_PACKED_KEY_MAX_LEN 1024

typedef struct ucc_tl_ucp_alltoallv_nsarka_global_work_buf_info {
    void *packed_src_memh;
    void *packed_dst_memh;
} ucc_tl_ucp_alltoallv_nsarka_global_work_buf_info;

struct ucc_tl_ucp_alltoallv_nsarka_export_buf {
    ucp_context_h ucp_context;
    ucp_mem_h     memh;
    void *        packed_memh;
    size_t        packed_memh_len;
    void *        packed_key;
    size_t        packed_key_len;
    uint64_t      memh_id;
};

typedef struct ucc_tl_ucp_alltoallv_nsarka_host_allgather {
    void *src_buf;
    void *dst_buf;
    char  packed_src_key[ALLTOALLV_PACKED_KEY_MAX_LEN];
    char  packed_dst_key[ALLTOALLV_PACKED_KEY_MAX_LEN];
} ucc_tl_ucp_alltoallv_nsarka_host_allgather;

#endif
