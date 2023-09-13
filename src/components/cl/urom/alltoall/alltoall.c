/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "alltoall.h"
//#include "utils/arch/cuda_def.h"

ucc_base_coll_alg_info_t
    ucc_cl_urom_alltoall_algs[UCC_CL_UROM_ALLTOALL_ALG_LAST + 1] = {
        [UCC_CL_UROM_ALLTOALL_ALG_FULL] =
            {.id   = UCC_CL_UROM_ALLTOALL_ALG_FULL,
             .name = "urom_full_offload",
             .desc = "full offload of alltoall"},
        [UCC_CL_UROM_ALLTOALL_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

ucc_status_t ucc_cl_urom_alltoall_triggered_post_setup(ucc_coll_task_t *task)
{
    return UCC_OK;
}

static ucc_status_t ucc_cl_urom_alltoall_full_start(ucc_coll_task_t *task)
{
    ucc_cl_urom_team_t     *cl_team = ucc_derived_of(task->team, ucc_cl_urom_team_t);
    ucc_cl_urom_context_t *ctx  = UCC_CL_UROM_TEAM_CTX(cl_team);
    ucc_cl_urom_lib_t *cl_lib = ucc_derived_of(ctx->super.super.lib, ucc_cl_urom_lib_t);
    ucc_coll_args_t        *coll_args = &task->bargs.args;
    urom_status_t       urom_status;
    urom_worker_cmd_t   coll_cmd = {
        .cmd_type = UROM_WORKER_CMD_UCC,
        .ucc.dpu_worker_id = UCC_CL_TEAM_RANK(cl_team),
        .ucc.cmd_type      = UROM_WORKER_CMD_UCC_COLL,
        .ucc.coll_cmd.coll_args = coll_args,
        .ucc.coll_cmd.team = cl_team->teams[0],
        .ucc.coll_cmd.use_xgvmi = cl_lib->xgvmi_enabled,
    };

    if (coll_args->src.info.mem_type == UCC_MEMORY_TYPE_CUDA) {
        // FIXME: a better way is to tweak args in urom 
        cudaStreamSynchronize(cl_lib->cuda_stream);

        coll_args->src.info.mem_type = UCC_MEMORY_TYPE_HOST;
        coll_args->dst.info.mem_type = UCC_MEMORY_TYPE_HOST;
        urom_status = urom_worker_push_cmdq(cl_lib->urom_worker, 0, &coll_cmd);
        if (UROM_OK != urom_status) {
            cl_debug(&cl_lib->super, "failed to push collective to urom");
            return UCC_ERR_NO_MESSAGE;
        }
        coll_args->src.info.mem_type = UCC_MEMORY_TYPE_CUDA;
        coll_args->dst.info.mem_type = UCC_MEMORY_TYPE_CUDA;
    } else {
        urom_status = urom_worker_push_cmdq(cl_lib->urom_worker, 0, &coll_cmd);
        if (UROM_OK != urom_status) {
            cl_debug(&cl_lib->super, "failed to push collective to urom");
            return UCC_ERR_NO_MESSAGE;
        }
    }
    task->status = UCC_INPROGRESS;
    cl_debug(&cl_lib->super, "pushed the collective to urom");
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
    urom_status_t           urom_status = 0;
    urom_worker_notify_t   *notif;

    urom_status = urom_worker_pop_notifyq(cl_lib->urom_worker, 0, &notif);
    if (UROM_ERR_QUEUE_EMPTY == urom_status) {
        return;
    }

    if (urom_status < 0) {
        cl_error(cl_lib, "Error in UROM");
        ctask->status = UCC_ERR_NO_MESSAGE;
        return;
    }

    if (notif->notify_type != UROM_WORKER_NOTIFY_UCC) {
        cl_debug(cl_lib, "WRONG NOTIFICATION (%ld != %d)", notif->notify_type, UROM_WORKER_NOTIFY_UCC);
        return;
    }

    if (1 || cl_lib->xgvmi_enabled) {
        size_t size_mod = 8;

        switch(ctask->bargs.args.src.info.datatype) {
            case UCC_DT_INT8:
            case UCC_DT_UINT8:
                size_mod = sizeof(char);
                break;
            case UCC_DT_INT32:
            case UCC_DT_UINT32:
            case UCC_DT_FLOAT32:
                size_mod = sizeof(int);
                break;
            case UCC_DT_INT64:
            case UCC_DT_UINT64:
            case UCC_DT_FLOAT64:
                size_mod = sizeof(uint64_t);
                break;
            case UCC_DT_INT128:
            case UCC_DT_UINT128:
            case UCC_DT_FLOAT128:
                size_mod = sizeof(__int128_t);
                break;
            default:
                printf("**** SCALAR UNKNOWN: %ld\n", ctask->bargs.args.src.info.datatype);
                break;
        }

        if (ctask->bargs.args.dst.info.mem_type == UCC_MEMORY_TYPE_CUDA) {
            cudaMemcpyAsync(cl_lib->old_dest, ctask->bargs.args.dst.info.buffer, ctask->bargs.args.src.info.count * size_mod , cudaMemcpyHostToDevice, cl_lib->cuda_stream);
            cudaStreamSynchronize(cl_lib->cuda_stream);
        } else {
            memcpy(cl_lib->old_dest, ctask->bargs.args.dst.info.buffer, ctask->bargs.args.src.info.count * size_mod);
        }

        ctask->bargs.args.dst.info.buffer = cl_lib->old_dest;
        ctask->bargs.args.src.info.buffer = cl_lib->old_src;
    }
    cl_debug(&cl_lib->super, "completed the collective from urom");

    ctask->status = (ucc_status_t) notif->ucc.status;
}  

ucc_status_t ucc_cl_urom_alltoall_full_init(
                         ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
                         ucc_coll_task_t **task)
{
    ucc_cl_urom_team_t     *cl_team = ucc_derived_of(team, ucc_cl_urom_team_t);
    ucc_cl_urom_context_t *ctx  = UCC_CL_UROM_TEAM_CTX(cl_team);
    ucc_cl_urom_lib_t *cl_lib = ucc_derived_of(ctx->super.super.lib, ucc_cl_urom_lib_t);
    ucc_cl_urom_schedule_t *cl_schedule;
    ucc_base_coll_args_t    args;
    ucc_schedule_t         *schedule;
    ucc_status_t            status;

    cl_schedule = ucc_cl_urom_get_schedule(cl_team);
    if (ucc_unlikely(!cl_schedule)) {
        return UCC_ERR_NO_MEMORY;
    }
    schedule = &cl_schedule->super.super;
    if (1 || cl_lib->xgvmi_enabled) {
        size_t size_mod = 8;
        switch(coll_args->args.src.info.datatype) {
            case UCC_DT_INT8:
            case UCC_DT_UINT8:
                size_mod = sizeof(char);
                break;
            case UCC_DT_INT32:
            case UCC_DT_UINT32:
            case UCC_DT_FLOAT32:
                size_mod = sizeof(int);
                break;
            case UCC_DT_INT64:
            case UCC_DT_UINT64:
            case UCC_DT_FLOAT64:
                size_mod = sizeof(uint64_t);
                break;
            case UCC_DT_INT128:
            case UCC_DT_UINT128:
            case UCC_DT_FLOAT128:
                size_mod = sizeof(__int128_t);
                break;
            default:
                printf("**** SCALAR UNKNOWN: %ld\n", coll_args->args.src.info.datatype);
                break;
        }
        //memcpy args to xgvmi buffer
        void * ptr = cl_lib->xgvmi_buffer + (cl_lib->cfg.xgvmi_buffer_size * (schedule->super.seq_num % cl_lib->cfg.num_buffers));
        if (coll_args->args.src.info.mem_type == UCC_MEMORY_TYPE_CUDA) {
            cudaMemcpyAsync(ptr, coll_args->args.src.info.buffer, coll_args->args.src.info.count * size_mod, cudaMemcpyDeviceToHost, cl_lib->cuda_stream);
        } else {
            memcpy(ptr, coll_args->args.src.info.buffer, coll_args->args.src.info.count * size_mod);
        }

        cl_lib->old_src = coll_args->args.src.info.buffer;
        coll_args->args.src.info.buffer = ptr;
        cl_lib->old_dest = coll_args->args.dst.info.buffer;
        coll_args->args.dst.info.buffer = ptr + coll_args->args.src.info.count * size_mod;
    } 
    memcpy(&args, coll_args, sizeof(args));
    status = ucc_schedule_init(schedule, &args, team); 
    if (UCC_OK != status) {
        ucc_cl_urom_put_schedule(schedule);
        return status;
    }

    schedule->super.post           = ucc_cl_urom_alltoall_full_start;
    schedule->super.progress       = ucc_cl_urom_alltoall_full_progress;
    schedule->super.finalize       = ucc_cl_urom_alltoall_full_finalize;
    schedule->super.triggered_post = ucc_triggered_post;
    schedule->super.triggered_post_setup =
        ucc_cl_urom_alltoall_triggered_post_setup;

    *task = &schedule->super;
    return UCC_OK;
}
