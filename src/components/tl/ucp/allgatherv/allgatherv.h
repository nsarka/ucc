/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef ALLGATHERV_H_
#define ALLGATHERV_H_

#include "../tl_ucp.h"
#include "../tl_ucp_coll.h"

enum {
    UCC_TL_UCP_ALLGATHERV_ALG_RING,
    UCC_TL_UCP_ALLGATHERV_ALG_NSARKA,
    UCC_TL_UCP_ALLGATHERV_ALG_LAST
};

extern ucc_base_coll_alg_info_t
             ucc_tl_ucp_allgatherv_algs[UCC_TL_UCP_ALLGATHERV_ALG_LAST + 1];

static inline int ucc_tl_ucp_allgatherv_alg_from_str(const char *str)
{
    int i;
    for (i = 0; i < UCC_TL_UCP_ALLGATHERV_ALG_LAST; i++) {
        if (0 == strcasecmp(str, ucc_tl_ucp_allgatherv_algs[i].name)) {
            break;
        }
    }
    return i;
}

ucc_status_t ucc_tl_ucp_allgatherv_ring_init_common(ucc_tl_ucp_task_t *task);

ucc_status_t ucc_tl_ucp_allgatherv_ring_init(ucc_base_coll_args_t *coll_args,
                                            ucc_base_team_t *     team,
                                            ucc_coll_task_t **    task_h);

ucc_status_t ucc_tl_ucp_allgatherv_init(ucc_tl_ucp_task_t *task);

ucc_status_t ucc_tl_ucp_allgatherv_nsarka_init(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t *team,
                                               ucc_coll_task_t **task_h);

ucc_status_t
ucc_tl_ucp_allgatherv_nsarka_task_init(ucc_base_coll_args_t *coll_args,
                                     ucc_base_team_t *     team,
                                     ucc_tl_ucp_task_t *   task);

ucc_status_t
ucc_tl_ucp_allgatherv_nsarka_start(ucc_coll_task_t *coll_task);

void ucc_tl_ucp_allgatherv_nsarka_progress(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_allgatherv_nsarka_finalize(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_allgatherv_nsarka_allgather_info_finalize(
    ucc_service_coll_req_t *scoll_req, ucc_tl_ucp_task_t *sw_task);

#define ALLGATHERV_PACKED_KEY_MAX_LEN 1024

typedef struct ucc_tl_ucp_allgatherv_nsarka_global_work_buf_info {
    void *packed_src_memh;
    void *packed_dst_memh;
} ucc_tl_ucp_allgatherv_nsarka_global_work_buf_info;

struct ucc_tl_ucp_allgatherv_nsarka_export_buf {
    ucp_context_h ucp_context;
    ucp_mem_h     memh;
    void *        packed_memh;
    size_t        packed_memh_len;
    void *        packed_key;
    size_t        packed_key_len;
    uint64_t      memh_id;
};

typedef struct ucc_tl_ucp_allgatherv_nsarka_host_allgather {
    void *src_buf;
    void *dst_buf;
    char  packed_src_key[ALLGATHERV_PACKED_KEY_MAX_LEN];
    char  packed_dst_key[ALLGATHERV_PACKED_KEY_MAX_LEN];
} ucc_tl_ucp_allgatherv_nsarka_host_allgather;

#endif
