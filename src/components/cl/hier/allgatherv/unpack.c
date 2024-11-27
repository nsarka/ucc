/**
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "unpack.h"

ucc_status_t ucc_cl_hier_allgatherv_unpack_finalize(ucc_coll_task_t *task)
{
    ucc_schedule_t *schedule = ucc_derived_of(task, ucc_schedule_t);
    ucc_cl_hier_schedule_t *cl_schedule = ucc_derived_of(schedule, ucc_cl_hier_schedule_t);

    ucc_mc_free(cl_schedule->scratch);

    return UCC_OK;
}

void ucc_cl_hier_allgatherv_unpack_progress(ucc_coll_task_t *task)
{
    ucc_schedule_t *schedule = ucc_derived_of(task, ucc_schedule_t);
    ucc_cl_hier_team_t  *cl_team = ucc_derived_of(task->team, ucc_cl_hier_team_t);
    ucc_rank_t           team_size = UCC_CL_TEAM_SIZE(cl_team);
    ucc_cl_hier_schedule_t *cl_schedule = ucc_derived_of(schedule, ucc_cl_hier_schedule_t);
    ucc_ee_executor_task_t **tasks = cl_schedule->scratch->addr;
    ucc_status_t st;
    ucc_rank_t i;

    for (i = 0; i < team_size; i++) {
        ucc_ee_executor_task_t *etask = tasks[i];
        if (etask != NULL) {
            st = ucc_ee_executor_task_test(etask);
            if (st == UCC_OK) {
                ucc_ee_executor_task_finalize(etask);
                tasks[i] = NULL;
            } else {
                if (ucc_likely(st > 0)) {
                    st = UCC_INPROGRESS;
                }
                goto out;
            }
        }
    }

out:
    schedule->super.status       = st;
    schedule->super.super.status = st;
}

ucc_status_t ucc_cl_hier_allgatherv_unpack_start(ucc_coll_task_t *task)
{
    ucc_schedule_t *schedule = ucc_derived_of(task, ucc_schedule_t);
    ucc_cl_hier_team_t  *cl_team = ucc_derived_of(task->team, ucc_cl_hier_team_t);
    ucc_rank_t           team_size = UCC_CL_TEAM_SIZE(cl_team);
    ucc_coll_args_t       *args = &task->bargs.args;
    ucc_ee_executor_task_args_t eargs = {0};
    ucc_cl_hier_schedule_t *cl_schedule = ucc_derived_of(schedule, ucc_cl_hier_schedule_t);
    ucc_ee_executor_task_t **tasks = cl_schedule->scratch->addr;
    ucc_rank_t n_tasks = 0;
    size_t dt_size = ucc_dt_size(args->dst.info_v.datatype);
    ucc_ee_executor_t *exec;
    ucc_status_t status;
    int i;

    UCC_CHECK_GOTO(ucc_coll_task_get_executor(&schedule->super, &exec), out, status);
    eargs.task_type = UCC_EE_EXECUTOR_TASK_COPY;
    size_t disp_counter = ucc_coll_args_get_total_count(args, args->dst.info_v.counts, team_size);

    for (i = team_size - 1; i >= 0; i--) {
        size_t this_rank_count = ucc_coll_args_get_count(args, args->dst.info_v.counts, i);
        disp_counter -= this_rank_count;
        eargs.copy.src  = PTR_OFFSET(args->dst.info_v.buffer, disp_counter * dt_size);
        eargs.copy.dst  = PTR_OFFSET(args->dst.info_v.buffer, ucc_coll_args_get_displacement(args, args->dst.info_v.displacements, i) * dt_size);
        eargs.copy.len  = this_rank_count * dt_size;
        UCC_CHECK_GOTO(ucc_ee_executor_task_post(exec, &eargs, &tasks[n_tasks]), out, status);
        n_tasks++;
    }

    schedule->super.status       = UCC_INPROGRESS;
    schedule->super.super.status = UCC_INPROGRESS;

    ucc_progress_queue_enqueue(cl_team->super.super.context->ucc_context->pq, task);

    return UCC_OK;
out:
    return status;
}

ucc_status_t ucc_cl_hier_allgatherv_unpack_init(ucc_base_coll_args_t *coll_args,
                                                ucc_base_team_t      *team,
                                                ucc_coll_task_t     **task_h)
{
    ucc_cl_hier_team_t  *cl_team = ucc_derived_of(team, ucc_cl_hier_team_t);
    ucc_rank_t           team_size = UCC_CL_TEAM_SIZE(cl_team);
    ucc_status_t         status;
    ucc_schedule_t      *schedule;
    ucc_cl_hier_schedule_t *cl_schedule;
    size_t scratch_size;
    

    schedule = &ucc_cl_hier_get_schedule(cl_team)->super.super;
    if (ucc_unlikely(!schedule)) {
        return UCC_ERR_NO_MEMORY;
    }
    cl_schedule = ucc_derived_of(schedule, ucc_cl_hier_schedule_t);

    UCC_CHECK_GOTO(ucc_schedule_init(schedule, coll_args, team), out, status);

    scratch_size = team_size * sizeof(ucc_ee_executor_task_t*);
    UCC_CHECK_GOTO(ucc_mc_alloc(&cl_schedule->scratch, scratch_size, UCC_MEMORY_TYPE_HOST), out, status);

    schedule->super.flags |= UCC_COLL_TASK_FLAG_EXECUTOR;
    schedule->super.post     = ucc_cl_hier_allgatherv_unpack_start;
    schedule->super.progress = ucc_cl_hier_allgatherv_unpack_progress;
    schedule->super.finalize = ucc_cl_hier_allgatherv_unpack_finalize;

    *task_h = &schedule->super;
    return UCC_OK;
out:
    ucc_cl_hier_put_schedule(schedule);
    return status;
}
