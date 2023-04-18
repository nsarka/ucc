/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "alltoall.h"

ucc_base_coll_alg_info_t
    ucc_cl_urom_alltoall_algs[UCC_CL_UROM_ALLTOALL_ALG_LAST + 1] = {
        [UCC_CL_UROM_ALLTOALL_ALG_FULL] =
            {.id   = UCC_CL_UROM_ALLTOALL_ALG_FULL,
             .name = "urom_full_offload",
             .desc = "full offload of alltoall"},
        [UCC_CL_UROM_ALLTOALL_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

static ucc_status_t ucc_cl_urom_alltoall_full_start(ucc_coll_task_t *task)
{
    ucc_cl_urom_team_t     *cl_team = ucc_derived_of(task->team, ucc_cl_urom_team_t);
    ucc_cl_urom_context_t *ctx  = UCC_CL_UROM_TEAM_CTX(cl_team);
    ucc_cl_urom_lib_t *cl_lib = ucc_derived_of(ctx->super.super.lib, ucc_cl_urom_lib_t);
    ucc_coll_args_t        *coll_args = &task->bargs.args;
    //ucc_status_t        ucc_status;
    urom_status_t       urom_status;
    urom_worker_cmd_t   coll_cmd = {
        .cmd_type = UROM_WORKER_CMD_UCC,
        .ucc.dpu_worker_id = UCC_CL_TEAM_RANK(cl_team),
        .ucc.cmd_type      = UROM_WORKER_CMD_UCC_COLL,
        .ucc.coll_cmd.coll_args = coll_args,
        .ucc.coll_cmd.team = cl_team->teams[0],
        //.ucc.coll_cmd.use_xgvmi = 1,
    };

    urom_status = urom_worker_push_cmdq(cl_lib->urom_worker, 0, &coll_cmd);
    if (UROM_OK != urom_status) {
        return UCC_ERR_NO_MESSAGE;
//        return urom_to_ucc_status(status);
    }

    printf("pushed the collective to urom\n");
    return ucc_progress_queue_enqueue(ctx->super.super.ucc_context->pq, task);
}

static ucc_status_t ucc_cl_urom_alltoall_full_finalize(ucc_coll_task_t *task)
{
    ucc_cl_urom_schedule_t *schedule =
        ucc_derived_of(task, ucc_cl_urom_schedule_t);
    ucc_status_t status;

    status = ucc_schedule_finalize(task);
    ucc_cl_urom_put_schedule(&schedule->super.super);
    return status;
}

static void ucc_cl_urom_alltoall_full_progress(ucc_coll_task_t *ctask)
{
    ucc_cl_urom_team_t     *cl_team = ucc_derived_of(ctask->team, ucc_cl_urom_team_t);
    ucc_cl_urom_context_t *ctx  = UCC_CL_UROM_TEAM_CTX(cl_team);
    ucc_cl_urom_lib_t *cl_lib = ucc_derived_of(ctx->super.super.lib, ucc_cl_urom_lib_t);
    urom_status_t           urom_status;
    urom_worker_notify_t   *notif;

    urom_status = urom_worker_pop_notifyq(cl_lib->urom_worker, 0, &notif);
    if (UROM_ERR_QUEUE_EMPTY == urom_status) {
        return;
    }

    if (urom_status != UROM_ERR_QUEUE_EMPTY && urom_status < 0) {
        cl_error(cl_lib, "Error in UROM");
        ctask->status = UCC_ERR_NO_MESSAGE;
        //ctask->status = urom_to_ucc_status(urom_status);
        return;
    }

    ctask->status = notif->ucc.status;
}  

ucc_status_t ucc_cl_urom_alltoall_full_init(
                         ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
                         ucc_coll_task_t **task)
{
    ucc_cl_urom_team_t     *cl_team = ucc_derived_of(team, ucc_cl_urom_team_t);
    //ucc_cl_urom_lib_t      *cl_lib  = UCC_CL_UROM_TEAM_LIB(cl_team);
    ucc_cl_urom_schedule_t *cl_schedule;
    ucc_base_coll_args_t    args;
    ucc_schedule_t         *schedule;
    ucc_status_t            status;

    cl_schedule = ucc_cl_urom_get_schedule(cl_team);
    if (ucc_unlikely(!cl_schedule)) {
        return UCC_ERR_NO_MEMORY;
    }
    schedule = &cl_schedule->super.super;
    memcpy(&args, coll_args, sizeof(args));
    status = ucc_schedule_init(schedule, &args, team); 
    if (UCC_OK != status) {
        printf("FAILED to put schedule\n");
        ucc_cl_urom_put_schedule(schedule);
        return status;
    }

    printf("schedule up!\n");
    schedule->super.post           = ucc_cl_urom_alltoall_full_start;
    schedule->super.progress       = ucc_cl_urom_alltoall_full_progress;
    schedule->super.finalize       = ucc_cl_urom_alltoall_full_finalize;
    *task = &schedule->super;
    return UCC_OK;
}
