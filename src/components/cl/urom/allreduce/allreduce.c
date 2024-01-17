/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "allreduce.h"
#include "../allreduce/allreduce.h"

ucc_base_coll_alg_info_t
    ucc_cl_urom_allreduce_algs[UCC_CL_UROM_ALLREDUCE_ALG_LAST + 1] = {
        [UCC_CL_UROM_ALLREDUCE_ALG_FULL] =
            {.id   = UCC_CL_UROM_ALLREDUCE_ALG_FULL,
             .name = "urom_full_offload",
             .desc = "full offload of allreduce"},
        [UCC_CL_UROM_ALLREDUCE_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};


static int buffer_export_ucc(ucp_context_h ucp_context, void *buf, size_t len,
                      struct export_buf *ebuf)
{
    ucs_status_t           ucs_status;
    ucp_mem_map_params_t   params;
    ucp_memh_pack_params_t pack_params;

    ebuf->ucp_context = ucp_context;

    params.field_mask =
        UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    params.address = buf;
    params.length  = len;

    ucs_status = ucp_mem_map(ucp_context, &params, &ebuf->memh);
    assert(ucs_status == UCS_OK);

    pack_params.field_mask = UCP_MEMH_PACK_PARAM_FIELD_FLAGS;
    pack_params.flags      = UCP_MEMH_PACK_FLAG_EXPORT;

    ucs_status = ucp_memh_pack(ebuf->memh, &pack_params, &ebuf->packed_memh,
                               &ebuf->packed_memh_len);
    if (ucs_status != UCS_OK) {
        printf("ucp_memh_pack() returned error: %s\n",
               ucs_status_string(ucs_status));
        ebuf->packed_memh     = NULL;
        ebuf->packed_memh_len = 0;
    }
    ucs_status = ucp_rkey_pack(ucp_context, ebuf->memh, &ebuf->packed_key,
                               &ebuf->packed_key_len);
    if (UCS_OK != ucs_status) {
        printf("ucp_rkey_pack() returned error: %s\n",
               ucs_status_string(ucs_status));
        return UROM_ERR_NO_RESOURCE;
    }

    return 0;
}

ucc_status_t ucc_cl_urom_allreduce_triggered_post_setup(ucc_coll_task_t *task)
{
    return UCC_OK;
}

static size_t dt_size(ucc_datatype_t ucc_dt)
{
    size_t size_mod = 8;

    switch(ucc_dt) {
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
            break;
    }

    return size_mod;
}

static ucc_status_t ucc_cl_urom_allreduce_full_start(ucc_coll_task_t *task)
{
    ucc_cl_urom_team_t     *cl_team = ucc_derived_of(task->team, ucc_cl_urom_team_t);
    ucc_cl_urom_context_t *ctx  = UCC_CL_UROM_TEAM_CTX(cl_team);
    ucc_cl_urom_lib_t *cl_lib = ucc_derived_of(ctx->super.super.lib, ucc_cl_urom_lib_t);
    ucc_coll_args_t        *coll_args = &task->bargs.args;
    urom_status_t       urom_status;
    int ucp_index = cl_lib->tl_ucp_index;
    ucc_tl_ucp_context_t *tl_ctx = ucc_derived_of(ctx->super.tl_ctxs[ucp_index], ucc_tl_ucp_context_t);
    urom_worker_cmd_t   coll_cmd = {
        .cmd_type = UROM_WORKER_CMD_UCC,
        .ucc.dpu_worker_id = UCC_CL_TEAM_RANK(cl_team),
        .ucc.cmd_type      = UROM_WORKER_CMD_UCC_COLL,
        .ucc.coll_cmd.coll_args = coll_args,
        .ucc.coll_cmd.team = cl_team->teams[0],
        //.ucc.coll_cmd.use_xgvmi = 0,
        //.ucc.coll_cmd.use_sliding_window_allreduce = 1,
    };
    ucc_memory_type_t prev_src, prev_dst;
    ucc_cl_urom_schedule_t *schedule =
        ucc_derived_of(task, ucc_cl_urom_schedule_t);
    struct export_buf *src_ebuf = &schedule->src_ebuf;
    struct export_buf *dst_ebuf = &schedule->dst_ebuf;

    prev_src = coll_args->src.info.mem_type;
    prev_dst = coll_args->dst.info.mem_type;
    coll_args->src.info.mem_type = UCC_MEMORY_TYPE_HOST;
    coll_args->dst.info.mem_type = UCC_MEMORY_TYPE_HOST;

    buffer_export_ucc(tl_ctx->worker.ucp_context, coll_args->src.info.buffer, coll_args->src.info.count * dt_size(coll_args->src.info.datatype), src_ebuf);
    buffer_export_ucc(tl_ctx->worker.ucp_context, coll_args->dst.info.buffer, coll_args->dst.info.count * dt_size(coll_args->dst.info.datatype), dst_ebuf);

    coll_cmd.ucc.coll_cmd.src_memh_packed = src_ebuf->packed_memh;
    coll_cmd.ucc.coll_cmd.src_memh_packed_len = src_ebuf->packed_memh_len;

    coll_cmd.ucc.coll_cmd.dst_memh_packed = dst_ebuf->packed_memh;
    coll_cmd.ucc.coll_cmd.dst_memh_packed_len = dst_ebuf->packed_memh_len;

    urom_status = urom_worker_push_cmdq(cl_lib->urom_ctx.urom_worker, 0, &coll_cmd);
    if (UROM_OK != urom_status) {
        cl_debug(&cl_lib->super, "failed to push collective to urom");
        return UCC_ERR_NO_MESSAGE;
    }
    coll_args->src.info.mem_type = prev_src;
    coll_args->dst.info.mem_type = prev_dst;

/*
    if (coll_args->src.info.mem_type != UCC_MEMORY_TYPE_CUDA) {
        urom_status = urom_worker_push_cmdq(cl_lib->urom_ctx.urom_worker, 0, &coll_cmd);
        if (UROM_OK != urom_status) {
            cl_debug(&cl_lib->super, "failed to push collective to urom");
            return UCC_ERR_NO_MESSAGE;
        }
    } else {
        coll_args->src.info.mem_type = UCC_MEMORY_TYPE_HOST;
        coll_args->dst.info.mem_type = UCC_MEMORY_TYPE_HOST;
        urom_status = urom_worker_push_cmdq(cl_lib->urom_ctx.urom_worker, 0, &coll_cmd);
        if (UROM_OK != urom_status) {
            cl_debug(&cl_lib->super, "failed to push collective to urom");
            return UCC_ERR_NO_MESSAGE;
        }
        coll_args->src.info.mem_type = UCC_MEMORY_TYPE_CUDA;
        coll_args->dst.info.mem_type = UCC_MEMORY_TYPE_CUDA;
    }
*/
    task->status = UCC_INPROGRESS;
    cl_debug(&cl_lib->super, "pushed the collective to urom");
    return ucc_progress_queue_enqueue(ctx->super.super.ucc_context->pq, task);
}

static ucc_status_t ucc_cl_urom_allreduce_full_finalize(ucc_coll_task_t *task)
{
    ucc_cl_urom_team_t     *cl_team = ucc_derived_of(task->team, ucc_cl_urom_team_t);
    ucc_cl_urom_context_t *ctx  = UCC_CL_UROM_TEAM_CTX(cl_team);
    ucc_cl_urom_lib_t *cl_lib = ucc_derived_of(ctx->super.super.lib, ucc_cl_urom_lib_t);
    int ucp_index = cl_lib->tl_ucp_index;
    ucc_tl_ucp_context_t *tl_ctx = ucc_derived_of(ctx->super.tl_ctxs[ucp_index], ucc_tl_ucp_context_t);
    ucc_status_t status;
    ucc_cl_urom_schedule_t *schedule =
        ucc_derived_of(task, ucc_cl_urom_schedule_t);
    struct export_buf *src_ebuf = &schedule->src_ebuf;
    struct export_buf *dst_ebuf = &schedule->dst_ebuf;

    ucp_mem_unmap(tl_ctx->worker.ucp_context, src_ebuf->memh);
    ucp_mem_unmap(tl_ctx->worker.ucp_context, dst_ebuf->memh);

    status = ucc_schedule_finalize(task);
    ucc_cl_urom_put_schedule(&schedule->super.super);
    return status;
}

static void ucc_cl_urom_allreduce_full_progress(ucc_coll_task_t *ctask)
{
    ucc_cl_urom_team_t     *cl_team = ucc_derived_of(ctask->team, ucc_cl_urom_team_t);
    ucc_cl_urom_context_t *ctx  = UCC_CL_UROM_TEAM_CTX(cl_team);
    ucc_cl_urom_lib_t *cl_lib = ucc_derived_of(ctx->super.super.lib, ucc_cl_urom_lib_t);
    urom_status_t           urom_status = 0;
    urom_worker_notify_t   *notif;

    urom_status = urom_worker_pop_notifyq(cl_lib->urom_ctx.urom_worker, 0, &notif);
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

    if (ctx->req_mc) {
        size_t size_mod = dt_size(ctask->bargs.args.dst.info.datatype);

        if ((ucc_status_t) notif->ucc.status == UCC_OK) {
            ucc_mc_memcpy(ctx->old_dest, ctask->bargs.args.dst.info.buffer, ctask->bargs.args.dst.info.count * size_mod, ctask->bargs.args.dst.info.mem_type, UCC_MEMORY_TYPE_HOST);
            ctask->bargs.args.dst.info.buffer = ctx->old_dest;
            ctask->bargs.args.src.info.buffer = ctx->old_src;
        }
    }
    cl_debug(&cl_lib->super, "completed the collective from urom");

    ctask->status = (ucc_status_t) notif->ucc.status;
}  

ucc_status_t ucc_cl_urom_allreduce_full_init(
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
    if (ctx->req_mc) {
        size_t size_mod = dt_size(coll_args->args.src.info.datatype);
        size_t count = coll_args->args.src.info.count * size_mod;
        //memcpy args to xgvmi buffer
        printf("nick memcpy args to xgvmi buffer\n");
        void * ptr = ctx->xgvmi.xgvmi_buffer + (cl_lib->cfg.xgvmi_buffer_size * (schedule->super.seq_num % cl_lib->cfg.num_buffers));
        ucc_mc_memcpy(ptr, coll_args->args.src.info.buffer, count, UCC_MEMORY_TYPE_HOST, coll_args->args.src.info.mem_type);

        ctx->old_src = coll_args->args.src.info.buffer;
        coll_args->args.src.info.buffer = ptr;
        ctx->old_dest = coll_args->args.dst.info.buffer;
        coll_args->args.dst.info.buffer = ptr + count;
    } 
    memcpy(&args, coll_args, sizeof(args));
    status = ucc_schedule_init(schedule, &args, team); 
    if (UCC_OK != status) {
        ucc_cl_urom_put_schedule(schedule);
        return status;
    }

    schedule->super.post           = ucc_cl_urom_allreduce_full_start;
    schedule->super.progress       = ucc_cl_urom_allreduce_full_progress;
    schedule->super.finalize       = ucc_cl_urom_allreduce_full_finalize;
    schedule->super.triggered_post = ucc_triggered_post;
    schedule->super.triggered_post_setup =
        ucc_cl_urom_allreduce_triggered_post_setup;

    *task = &schedule->super;
    cl_debug(cl_lib, "urom coll init'd");
    return UCC_OK;
}
