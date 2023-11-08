/**
 * Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#ifndef UCC_CL_UROM_COLL_H_
#define UCC_CL_UROM_COLL_H_

#include "cl_urom.h"
#include "schedule/ucc_schedule_pipelined.h"
#include "components/mc/ucc_mc.h"
#include "alltoall/alltoall.h"

#define UCC_CL_UROM_N_DEFAULT_ALG_SELECT_STR 2

extern const char
    *ucc_cl_urom_default_alg_select_str[UCC_CL_UROM_N_DEFAULT_ALG_SELECT_STR];

struct export_buf {
    ucp_context_h ucp_context;
    ucp_mem_h     memh;
    void         *packed_memh;
    size_t        packed_memh_len;
    void         *packed_key;
    size_t        packed_key_len;
    uint64_t      memh_id;
};

typedef struct ucc_cl_urom_schedule_t {
    ucc_schedule_pipelined_t super;
    ucc_mc_buffer_header_t  *scratch;
    struct export_buf src_ebuf;
    struct export_buf dst_ebuf;
} ucc_cl_urom_schedule_t;

static inline ucc_cl_urom_schedule_t *
ucc_cl_urom_get_schedule(ucc_cl_urom_team_t *team)
{
    ucc_cl_urom_context_t  *ctx      = UCC_CL_UROM_TEAM_CTX(team);
    ucc_cl_urom_schedule_t *schedule = ucc_mpool_get(&ctx->sched_mp);

    return schedule;
}

static inline void ucc_cl_urom_put_schedule(ucc_schedule_t *schedule)
{
    ucc_mpool_put(schedule);
}

ucc_status_t ucc_cl_urom_alg_id_to_init(int alg_id, const char *alg_id_str,
                                        ucc_coll_type_t   coll_type,
                                        ucc_memory_type_t mem_type, //NOLINT
                                        ucc_base_coll_init_fn_t *init);
#endif
