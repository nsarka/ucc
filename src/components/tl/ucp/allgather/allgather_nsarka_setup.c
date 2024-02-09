/**
 * Copyright(c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "allgather.h"
#include "../allgather/allgather.h"
#include "utils/ucc_dt_reduce.h"
#include "tl_ucp_ep.h"

static int ucc_tl_ucp_allgather_nsarka_register(
    ucp_context_h ucp_context, ucc_tl_ucp_team_t *tl_team,
    struct ucc_tl_ucp_allgather_nsarka_export_buf *ebuf, void *packed_memh)
{
    ucs_status_t         ucs_status;
    ucp_mem_map_params_t params = {0};

    ebuf->ucp_context = ucp_context;

    params.field_mask           = UCP_MEM_MAP_PARAM_FIELD_EXPORTED_MEMH_BUFFER;
    params.exported_memh_buffer = packed_memh;

    ucs_status = ucp_mem_map(ucp_context, &params, &ebuf->memh);
    if (UCS_OK != ucs_status) {
        tl_error(UCC_TL_TEAM_LIB(tl_team),
                 "import using ucp_mem_map() returned error: %s\n",
                 ucs_status_string(ucs_status));
        return 0;
    }

    ucs_status = ucp_rkey_pack(ucp_context, ebuf->memh, &ebuf->packed_key,
                               &ebuf->packed_key_len);
    if (UCS_OK != ucs_status) {
        tl_error(UCC_TL_TEAM_LIB(tl_team),
                 "ucp_rkey_pack() returned error: %s\n",
                 ucs_status_string(ucs_status));
        return 0;
    }

    return 0;
}

ucc_status_t
ucc_tl_ucp_allgather_nsarka_task_init(ucc_base_coll_args_t *coll_args,
                                              ucc_base_team_t *     team,
                                              ucc_tl_ucp_task_t *   task)
{
    ucc_status_t          status    = UCC_OK;
    void *                src_buf   = coll_args->args.src.info.buffer;
    void *                dst_buf   = coll_args->args.dst.info.buffer;
    ucc_rank_t            team_size = (ucc_rank_t)team->params.size;
    ucc_tl_ucp_team_t *   tl_team   = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_context_t *tl_ctx    = UCC_TL_UCP_TEAM_CTX(tl_team);
    ucc_tl_ucp_allgather_nsarka_global_work_buf_info *gwbi_p = NULL;
    size_t allgather_size = sizeof(ucc_tl_ucp_allgather_nsarka_host_allgather);
    ucc_tl_ucp_allgather_nsarka_host_allgather *allgather_data;

    allgather_data = ucc_malloc(allgather_size * (team_size + 1));

    gwbi_p = coll_args->args.global_work_buffer;
    task->super.bargs.args.global_work_buffer = gwbi_p;

    task->allgather_nsarka.inplace = UCC_IS_INPLACE(coll_args->args);

    task->allgather_nsarka.barrier_task = NULL;

    if (!task->allgather_nsarka.inplace) {
        task->allgather_nsarka.sbufs =
            ucc_malloc(sizeof(void *) * team_size);
        task->allgather_nsarka.src_rkeys =
            ucc_malloc(sizeof(ucp_rkey_h) * team_size);
    }

    task->allgather_nsarka.rbufs =
        ucc_malloc(sizeof(void *) * team_size);
    task->allgather_nsarka.dst_rkeys =
        ucc_malloc(sizeof(ucp_rkey_h) * team_size);
    task->allgather_nsarka.eps =
        ucc_malloc(sizeof(ucp_ep_h) * team_size);


    if (!task->allgather_nsarka.inplace) {
        task->allgather_nsarka.src_ebuf =
            ucc_malloc(sizeof(struct ucc_tl_ucp_allgather_nsarka_export_buf));
    } else {
        task->allgather_nsarka.src_ebuf = NULL;
    }

    task->allgather_nsarka.dst_ebuf =
        ucc_malloc(sizeof(struct ucc_tl_ucp_allgather_nsarka_export_buf));

    if (!task->allgather_nsarka.inplace)
        allgather_data->src_buf = src_buf;

    allgather_data->dst_buf = dst_buf;

    // Register the src and dst bufs
    ucc_tl_ucp_allgather_nsarka_register(
        tl_ctx->worker.ucp_context, tl_team,
        task->allgather_nsarka.dst_ebuf, gwbi_p->packed_dst_memh);
    memcpy(allgather_data->packed_dst_key,
           task->allgather_nsarka.dst_ebuf->packed_key,
           task->allgather_nsarka.dst_ebuf->packed_key_len);
    
    if (!task->allgather_nsarka.inplace) {
        ucc_tl_ucp_allgather_nsarka_register(
            tl_ctx->worker.ucp_context, tl_team,
            task->allgather_nsarka.src_ebuf, gwbi_p->packed_src_memh);
        memcpy(allgather_data->packed_src_key,
               task->allgather_nsarka.src_ebuf->packed_key,
               task->allgather_nsarka.src_ebuf->packed_key_len);
    }

    task->allgather_nsarka.allgather_data      = allgather_data;
    task->allgather_nsarka.allgather_scoll_req = NULL;

    return status;
}

ucc_status_t ucc_tl_ucp_allgather_nsarka_allgather_info_finalize(
    ucc_service_coll_req_t *scoll_req, ucc_tl_ucp_task_t *task)
{
    ucc_status_t       status;
    ucc_rank_t         i;
    ucc_base_team_t *  base_team = task->super.team;
    ucc_tl_ucp_team_t *tl_team   = ucc_derived_of(base_team, ucc_tl_ucp_team_t);
    ucc_rank_t         team_size = base_team->params.size;

    size_t allgather_size = sizeof(ucc_tl_ucp_allgather_nsarka_host_allgather);
    ucc_tl_ucp_allgather_nsarka_host_allgather *all_host_allgather =
        PTR_OFFSET(scoll_req->data, allgather_size);

    for (i = 0; i < team_size; i++) {
        ucs_status_t ucs_status = UCS_OK;
        ucp_rkey_h   src_unpacked, dst_unpacked;
        ucp_ep_h     ep;

        status = ucc_tl_ucp_get_ep(tl_team, i, &ep);
        if (ucc_unlikely(UCC_OK != status)) {
            return status;
        }

        ucs_status = ucp_ep_rkey_unpack(
            ep, all_host_allgather[i].packed_dst_key, &dst_unpacked);
        if (UCS_OK != ucs_status) {
            tl_error(UCC_TL_TEAM_LIB(tl_team), "dst rkey unpack failed\n");
            return UCC_ERR_NO_RESOURCE;
        }

        task->allgather_nsarka.rbufs[i] =
            all_host_allgather[i].dst_buf;
        task->allgather_nsarka.dst_rkeys[i] = dst_unpacked;

        if (!task->allgather_nsarka.inplace) {
            ucs_status = ucp_ep_rkey_unpack(
                ep, all_host_allgather[i].packed_src_key, &src_unpacked);
            if (UCS_OK != ucs_status) {
                tl_error(UCC_TL_TEAM_LIB(tl_team), "src rkey unpack failed\n");
                return UCC_ERR_NO_RESOURCE;
            }

            task->allgather_nsarka.sbufs[i] =
                all_host_allgather[i].src_buf;
            task->allgather_nsarka.src_rkeys[i] = src_unpacked;
        } else {
            task->allgather_nsarka.sbufs =
                task->allgather_nsarka.rbufs;
            task->allgather_nsarka.src_rkeys =
                task->allgather_nsarka.dst_rkeys;
        }

        task->allgather_nsarka.eps[i] = ep;
    }

    return status;
}

ucc_status_t
ucc_tl_ucp_allgather_nsarka_free_gwbi(ucc_coll_task_t *coll_task)
{
    ucc_base_team_t *     team    = coll_task->team;
    ucc_tl_ucp_team_t *   tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t *   task   = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_context_t *tl_ctx = UCC_TL_UCP_TEAM_CTX(tl_team);

    if (!task->allgather_nsarka.inplace)
        ucc_free(task->allgather_nsarka.sbufs);

    ucc_free(task->allgather_nsarka.rbufs);
    ucc_free(task->allgather_nsarka.eps);
    ucc_free(task->allgather_nsarka.allgather_data);

    if (!task->allgather_nsarka.inplace) {
        ucp_mem_unmap(tl_ctx->worker.ucp_context,
                      task->allgather_nsarka.src_ebuf->memh);
        ucc_free(task->allgather_nsarka.src_ebuf);
        ucc_free(task->allgather_nsarka.src_rkeys);
    }

    ucp_mem_unmap(tl_ctx->worker.ucp_context,
                  task->allgather_nsarka.dst_ebuf->memh);
    ucc_free(task->allgather_nsarka.dst_ebuf);
    ucc_free(task->allgather_nsarka.dst_rkeys);

    return UCC_OK;
}

