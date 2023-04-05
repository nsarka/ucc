/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "cl_urom.h"
#include "utils/ucc_coll_utils.h"

#include <urom.h>
#include <urom_ucc.h>

static ucc_status_t ucc_cl_hier_alltoallv_start(ucc_coll_task_t *task)
{
    UCC_CL_HIER_PROFILE_REQUEST_EVENT(task, "cl_hier_alltoallv_start", 0);
    return ucc_schedule_start(task);
}

static ucc_status_t ucc_cl_urom_coll_finalize(ucc_coll_task_t *task)
{
    ucc_cl_hier_schedule_t *schedule =
        ucc_derived_of(task, ucc_cl_hier_schedule_t);
    ucc_status_t status;

    UCC_CL_HIER_PROFILE_REQUEST_EVENT(task, "cl_hier_alltoallv_finalize", 0);
    ucc_assert(schedule->super.super.n_tasks == 1 ||
               schedule->super.super.n_tasks == 2);
    if (schedule->scratch) {
        ucc_mc_free(schedule->scratch);
    }
    status = ucc_schedule_finalize(task);
    ucc_cl_hier_put_schedule(&schedule->super.super);
    return status;
}

ucc_status_t ucc_cl_urom_coll_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t      *team,
                                   ucc_coll_task_t     **task)
{
    ucc_cl_urom_team_t *cl_team = ucc_derived_of(team, ucc_cl_urom_team_t);
    ucc_status_t        ucc_status;
    urom_status_t       urom_status;
    urom_worker_cmd_t   coll_cmd = {
        .cmd_type = UROM_WORKER_CMD_UCC,
        .ucc.dpu_worker_id = UCC_CL_TEAM_RANK(cl_team),
        .ucc.cmd_type      = UROM_WORKER_CMD_UCC_COLL,
        .ucc.coll_cmd.coll_args = coll_args,
        .ucc.coll_cmd.team = team,
        .ucc.coll_cmd.use_xgvmi = 0,
    };

    /* F: I dont think this is right */
    urom_status = urom_worker_push_cmdq(UCC_CL_TEAM_LIB(cl_team)->urom_worker, 0, &coll_cmd);
    if (UROM_OK != status) {
        return urom_to_ucc_status(status);
    }

#if 0    
    status = ucc_coll_init(cl_team->score_map, coll_args, task);
    if (UCC_ERR_NOT_FOUND == status) {
        cl_warn(UCC_CL_TEAM_LIB(cl_team),
                "no TL supporting given coll args is available");
        return UCC_ERR_NOT_SUPPORTED;
    }
#endif
    
    return status;
}
