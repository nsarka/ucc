/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "tl_ucp.h"
#include "allreduce.h"
#include "utils/ucc_coll_utils.h"

ucc_base_coll_alg_info_t
    ucc_tl_ucp_allreduce_algs[UCC_TL_UCP_ALLREDUCE_ALG_LAST + 1] = {
        [UCC_TL_UCP_ALLREDUCE_ALG_KNOMIAL] =
            {.id   = UCC_TL_UCP_ALLREDUCE_ALG_KNOMIAL,
             .name = "knomial",
             .desc =
                 "recursive knomial with arbitrary radix (optimized for latency)"},
        [UCC_TL_UCP_ALLREDUCE_ALG_SRA_KNOMIAL] =
            {.id   = UCC_TL_UCP_ALLREDUCE_ALG_SRA_KNOMIAL,
             .name = "sra_knomial",
             .desc = "recursive knomial scatter-reduce followed by knomial "
                     "allgather (optimized for BW)"},
        [UCC_TL_UCP_ALLREDUCE_ALG_SLIDING_WINDOW] =
            {.id   = UCC_TL_UCP_ALLREDUCE_ALG_SLIDING_WINDOW,
             .name = "sliding_window",
             .desc = "sliding window allreduce (DPU based)"},
        [UCC_TL_UCP_ALLREDUCE_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

ucc_status_t ucc_tl_ucp_allreduce_init(ucc_tl_ucp_task_t *task)
{
    ucc_status_t status;
    ALLREDUCE_TASK_CHECK(TASK_ARGS(task), TASK_TEAM(task));
    status = ucc_tl_ucp_allreduce_knomial_init_common(task);
out:
    return status;
}

ucc_status_t ucc_tl_ucp_allreduce_knomial_init(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t *     team,
                                               ucc_coll_task_t **    task_h)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t *task;
    ucc_status_t       status;
    ALLREDUCE_TASK_CHECK(coll_args->args, tl_team);
    task                 = ucc_tl_ucp_init_task(coll_args, team);
    *task_h              = &task->super;
    status = ucc_tl_ucp_allreduce_knomial_init_common(task);
out:
    return status;
}

ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_init(ucc_base_coll_args_t *coll_args,
                                         ucc_base_team_t      *team,
                                         ucc_coll_task_t     **task_h)
{
    ucc_tl_ucp_team_t       *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_rank_t               size    = (ucc_rank_t)team->params.size;
    ucc_status_t             status  = UCC_OK;
    ucc_tl_ucp_task_t       *task;
    ucc_ee_executor_params_t params;

    ALLREDUCE_TASK_CHECK(coll_args->args, tl_team);

    if (!(coll_args->args.mask & UCC_COLL_ARGS_FIELD_GLOBAL_WORK_BUFFER)) {
        tl_error(UCC_TL_TEAM_LIB(tl_team),
                 "global work buffer not provided nor associated with team");
        status = UCC_ERR_NOT_SUPPORTED;
        goto out;
    }
    if (coll_args->args.mask & UCC_COLL_ARGS_FIELD_FLAGS) {
        if (!(coll_args->args.flags & UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS)) {
            tl_error(UCC_TL_TEAM_LIB(tl_team),
                     "non memory mapped buffers are not supported");
            status = UCC_ERR_NOT_SUPPORTED;
            goto out;
        }
    }
    task                 = ucc_tl_ucp_init_task(coll_args, team);
    *task_h              = &task->super;
    task->super.post     = ucc_tl_ucp_allreduce_sliding_window_start;
    task->super.progress = ucc_tl_ucp_allreduce_sliding_window_progress;
    task->super.finalize = ucc_tl_ucp_allreduce_sliding_window_finalize;

    ucc_tl_ucp_allreduce_sw_global_work_buf_info *gwbi_p =
        (ucc_tl_ucp_allreduce_sw_global_work_buf_info *)TASK_ARGS(task).global_work_buffer;

    task->allreduce_sliding_window.src_rkeys    = &gwbi_p->src_rkeys[size * gwbi_p->tid];
    task->allreduce_sliding_window.dst_rkeys    = &gwbi_p->dst_rkeys[size * gwbi_p->tid];
    task->allreduce_sliding_window.host_eps     = &gwbi_p->host_eps[size * gwbi_p->tid];
    task->allreduce_sliding_window.worker       = gwbi_p->ucp_thread_workers[gwbi_p->tid];
    task->allreduce_sliding_window.sbufs        = gwbi_p->sbufs;
    task->allreduce_sliding_window.rbufs        = gwbi_p->rbufs;
    task->allreduce_sliding_window.pipe         = &gwbi_p->pipes[gwbi_p->tid];
    task->allreduce_sliding_window.num_get_bufs = gwbi_p->num_bufs - 1;
    task->allreduce_sliding_window.window_size  = gwbi_p->window_size;
    task->allreduce_sliding_window.tid          = gwbi_p->tid;
    task->allreduce_sliding_window.nthreads     = gwbi_p->nthreads;
    task->allreduce_sliding_window.put_requests = gwbi_p->pipes[gwbi_p->tid].put_requests;

    params.mask    = UCC_EE_EXECUTOR_PARAM_FIELD_TYPE;
    params.ee_type = UCC_EE_CPU_THREAD;
    status         = ucc_ee_executor_init(&params,
                                          &task->allreduce_sliding_window.executor);

    if (UCC_OK != status) {
        ucc_error("failed to init executor: %s", ucc_status_string(status));
    }

out:
    return status;
}
