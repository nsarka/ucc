/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "allgatherv.h"
#include "utils/ucc_coll_utils.h"

ucc_base_coll_alg_info_t
    ucc_tl_ucp_allgatherv_algs[UCC_TL_UCP_ALLGATHERV_ALG_LAST + 1] = {
        [UCC_TL_UCP_ALLGATHERV_ALG_RING] =
            {.id   = UCC_TL_UCP_ALLGATHERV_ALG_RING,
             .name = "ring",
             .desc = "O(N) Ring"},
        [UCC_TL_UCP_ALLGATHERV_ALG_NSARKA] =
            {.id   = UCC_TL_UCP_ALLGATHERV_ALG_NSARKA,
             .name = "nsarka",
             .desc = "nsarka xgvmi allgatherv"},
        [UCC_TL_UCP_ALLGATHERV_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

ucc_status_t ucc_tl_ucp_allgatherv_init(ucc_tl_ucp_task_t *task)
{
    if (!ucc_coll_args_is_predefined_dt(&TASK_ARGS(task), UCC_RANK_INVALID)) {
        tl_error(UCC_TASK_LIB(task), "user defined datatype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }

    return ucc_tl_ucp_allgatherv_ring_init_common(task);
}

ucc_status_t ucc_tl_ucp_allgatherv_ring_init(ucc_base_coll_args_t *coll_args,
                                            ucc_base_team_t *     team,
                                            ucc_coll_task_t **    task_h)
{
    ucc_tl_ucp_task_t *task;
    ucc_status_t status;

    task = ucc_tl_ucp_init_task(coll_args, team);
    status = ucc_tl_ucp_allgatherv_ring_init_common(task);
    if (status != UCC_OK) {
        ucc_tl_ucp_put_task(task);
        return status;
    }
    *task_h = &task->super;
    return UCC_OK;
}

ucc_status_t
ucc_tl_ucp_allgatherv_nsarka_init(ucc_base_coll_args_t *coll_args,
                                         ucc_base_team_t *     team,
                                         ucc_coll_task_t **    task_h)
{
    ucc_status_t             status  = UCC_OK;
    ucc_tl_ucp_task_t *      task;

    task = ucc_tl_ucp_init_task(coll_args, team);
    if (ucc_unlikely(!task)) {
        ucc_error("couldnt allocate task");
        return UCC_ERR_NO_MEMORY;
    }
    *task_h              = &task->super;
    task->super.post     = ucc_tl_ucp_allgatherv_nsarka_start;
    task->super.progress = ucc_tl_ucp_allgatherv_nsarka_progress;
    task->super.finalize = ucc_tl_ucp_allgatherv_nsarka_finalize;

    ucc_tl_ucp_allgatherv_nsarka_task_init(coll_args, team, task);

    return status;
}
