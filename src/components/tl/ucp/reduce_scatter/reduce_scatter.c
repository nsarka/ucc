/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_ucp.h"
#include "reduce_scatter.h"
#include "utils/ucc_coll_utils.h"

ucc_base_coll_alg_info_t
    ucc_tl_ucp_reduce_scatter_algs[UCC_TL_UCP_REDUCE_SCATTER_ALG_LAST + 1] = {
        [UCC_TL_UCP_REDUCE_SCATTER_ALG_RING] =
            {.id   = UCC_TL_UCP_REDUCE_SCATTER_ALG_RING,
             .name = "ring",
             .desc = "O(N) ring"},
        [UCC_TL_UCP_REDUCE_SCATTER_ALG_NSARKA] =
            {.id   = UCC_TL_UCP_REDUCE_SCATTER_ALG_NSARKA,
             .name = "nsarka",
             .desc = "nsarka redscat alg"},
        [UCC_TL_UCP_REDUCE_SCATTER_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

ucc_status_t
ucc_tl_ucp_reduce_scatter_nsarka_init(ucc_base_coll_args_t *coll_args,
                                         ucc_base_team_t *     team,
                                         ucc_coll_task_t **    task_h)
{
    ucc_status_t             status  = UCC_OK;
    //ucc_tl_ucp_team_t *      tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t *      task;
    ucc_ee_executor_params_t params;

    //ALLTOALLV_TASK_CHECK(coll_args->args, tl_team);

    task = ucc_tl_ucp_init_task(coll_args, team);
    if (ucc_unlikely(!task)) {
        ucc_error("couldnt allocate task");
        return UCC_ERR_NO_MEMORY;
    }
    *task_h              = &task->super;
    task->super.post     = ucc_tl_ucp_reduce_scatter_nsarka_start;
    task->super.progress = ucc_tl_ucp_reduce_scatter_nsarka_progress;
    task->super.finalize = ucc_tl_ucp_reduce_scatter_nsarka_finalize;

    ucc_tl_ucp_reduce_scatter_nsarka_task_init(coll_args, team, task);

    params.mask    = UCC_EE_EXECUTOR_PARAM_FIELD_TYPE;
    params.ee_type = UCC_EE_CPU_THREAD;
    status =
        ucc_ee_executor_init(&params, &task->reduce_scatter_nsarka.executor);

    if (UCC_OK != status) {
        ucc_error("failed to init executor: %s", ucc_status_string(status));
    }

    return status;
}
