/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#ifndef REDUCE_SCATTERV_H_
#define REDUCE_SCATTERV_H_
#include "tl_ucp_coll.h"

enum
{
    UCC_TL_UCP_REDUCE_SCATTERV_ALG_RING,
    UCC_TL_UCP_REDUCE_SCATTERV_ALG_NSARKA,
    UCC_TL_UCP_REDUCE_SCATTERV_ALG_LAST
};

extern ucc_base_coll_alg_info_t
    ucc_tl_ucp_reduce_scatterv_algs[UCC_TL_UCP_REDUCE_SCATTERV_ALG_LAST + 1];

#define UCC_TL_UCP_REDUCE_SCATTERV_DEFAULT_ALG_SELECT_STR                      \
    "reduce_scatterv:@ring"

static inline int ucc_tl_ucp_reduce_scatterv_alg_from_str(const char *str)
{
    int i;
    for (i = 0; i < UCC_TL_UCP_REDUCE_SCATTERV_ALG_LAST; i++) {
        if (0 == strcasecmp(str, ucc_tl_ucp_reduce_scatterv_algs[i].name)) {
            break;
        }
    }
    return i;
}

ucc_status_t
ucc_tl_ucp_reduce_scatterv_ring_init(ucc_base_coll_args_t *coll_args,
                                     ucc_base_team_t *     team,
                                     ucc_coll_task_t **    task_h);


ucc_status_t
ucc_tl_ucp_reduce_scatterv_nsarka_alloc_pipe(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t *     team,
                                               ucc_tl_ucp_task_t *   task);

ucc_status_t ucc_tl_ucp_reduce_scatterv_nsarka_init(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t *team,
                                               ucc_coll_task_t **task_h);

ucc_status_t
ucc_tl_ucp_reduce_scatterv_nsarka_task_init(ucc_base_coll_args_t *coll_args,
                                     ucc_base_team_t *     team,
                                     ucc_tl_ucp_task_t *   task);

ucc_status_t
ucc_tl_ucp_reduce_scatterv_nsarka_start(ucc_coll_task_t *coll_task);

void ucc_tl_ucp_reduce_scatterv_nsarka_progress(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_reduce_scatterv_nsarka_finalize(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_reduce_scatterv_nsarka_free_gwbi(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_reduce_scatterv_nsarka_allgather_info_finalize(
    ucc_service_coll_req_t *scoll_req, ucc_tl_ucp_task_t *sw_task);


#define REDUCE_SCATTER_PACKED_KEY_MAX_LEN 1024

typedef struct ucc_tl_ucp_reduce_scatterv_nsarka_global_work_buf_info {
    void *packed_src_memh;
    void *packed_dst_memh;
} ucc_tl_ucp_reduce_scatterv_nsarka_global_work_buf_info;

struct ucc_tl_ucp_reduce_scatterv_nsarka_export_buf {
    ucp_context_h ucp_context;
    ucp_mem_h     memh;
    void *        packed_memh;
    size_t        packed_memh_len;
    void *        packed_key;
    size_t        packed_key_len;
    uint64_t      memh_id;
};

typedef struct ucc_tl_ucp_reduce_scatterv_nsarka_buf {
    void *                            buf;
    ucc_tl_ucp_allreduce_sw_buf_state state;
    ucs_status_ptr_t                  ucp_req;
    size_t                            count;
    size_t                            bytes;
} ucc_tl_ucp_reduce_scatterv_nsarka_buf;

typedef struct ucc_tl_ucp_reduce_scatterv_nsarka_pipeline {
    ucc_tl_ucp_reduce_scatterv_nsarka_buf  accbuf;
    ucc_tl_ucp_reduce_scatterv_nsarka_buf *getbuf;
    ucs_status_ptr_t *           put_requests;
    size_t                       buffer_size;
    size_t                       num_buffers;
    size_t                       avail_buffs;
    size_t                       my_count;
    size_t                       my_offset;
    size_t                       count_issued;
    size_t                       count_received;
    size_t                       count_reduced;
    size_t                       count_serviced;
    size_t                       get_idx;
    size_t                       red_idx;
    ucc_rank_t                   src_rank;
    ucc_rank_t                   dst_rank;
    int                          done_get;
    int                          done_red;
    int                          done_put;
    int                          posted_put;
} ucc_tl_ucp_reduce_scatterv_nsarka_pipeline;

typedef struct ucc_tl_ucp_reduce_scatterv_nsarka_host_allgather {
    void *src_buf;
    void *dst_buf;
    char  packed_src_key[REDUCE_SCATTER_PACKED_KEY_MAX_LEN];
    char  packed_dst_key[REDUCE_SCATTER_PACKED_KEY_MAX_LEN];
} ucc_tl_ucp_reduce_scatterv_nsarka_host_allgather;

#endif
