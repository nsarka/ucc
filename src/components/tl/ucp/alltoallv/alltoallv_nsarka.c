/**
 * Copyright(c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "alltoallv.h"
#include "../allgather/allgather.h"
#include "../barrier/barrier.h"
#include "utils/ucc_dt_reduce.h"
#include "tl_ucp_ep.h"


ucc_status_t
ucc_tl_ucp_alltoallv_nsarka_start(ucc_coll_task_t *coll_task)
{
    ucc_status_t       status = UCC_OK;
    ucc_tl_ucp_task_t *task   = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team   = TASK_TEAM(task);
    ucc_rank_t         size   = (ucc_rank_t)task->subset.map.ep_num;
    ucc_tl_ucp_alltoallv_nsarka_host_allgather *allgather_data =
        task->alltoallv_nsarka.allgather_data;
    size_t allgather_size = sizeof(ucc_tl_ucp_alltoallv_nsarka_host_allgather);
    ucc_service_coll_req_t *scoll_req;
    int i;

    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);

    task->alltoallv_nsarka.barrier_task       = NULL;

    task->alltoallv_nsarka.gets_posted    = 0;
    task->alltoallv_nsarka.gets_completed = 0;

    task->alltoallv_nsarka.requests = ucc_malloc(sizeof(ucs_status_ptr_t) * size);

    for(i = 0; i < size; i++) {
        task->alltoallv_nsarka.requests[i] = NULL;
    }

    UCC_CHECK_GOTO(
        ucc_service_allgather(
            UCC_TL_CORE_TEAM(team), allgather_data,
            PTR_OFFSET(allgather_data, allgather_size), allgather_size,
            ucc_sbgp_to_subset(ucc_topo_get_sbgp(team->topo, UCC_SBGP_FULL)),
            &scoll_req),
        out, status);

    scoll_req->data                                    = allgather_data;
    task->alltoallv_nsarka.allgather_scoll_req = scoll_req;

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
out:
    return status;
}

ucc_status_t
ucc_tl_ucp_alltoallv_nsarka_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_status_t       st = UCC_OK;

    ucc_free(task->alltoallv_nsarka.requests);

    st = ucc_tl_ucp_coll_finalize(&task->super);

    if (ucc_unlikely(st != UCC_OK)) {
        tl_error(UCC_TASK_LIB(task), "failed to finalize collective");
    }

    return st;
}

static inline ucc_status_t
ucc_tl_ucp_alltoallv_nsarka_req_test(ucs_status_ptr_t   request,
                                             ucc_tl_ucp_task_t *task)
{
    if (request == NULL) {
        return UCC_OK;
    } else if (UCS_PTR_IS_ERR(request)) {
        tl_error(UCC_TASK_LIB(task), "unable to complete UCX request=%p: %d\n",
                 request, UCS_PTR_STATUS(request));
        return ucs_status_to_ucc_status(UCS_PTR_STATUS(request));
    } else {
        return ucs_status_to_ucc_status(ucp_request_check_status(request));
    }
}

static inline void ucc_tl_ucp_alltoallv_nsarka_allgather_info_test(
    ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *     task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_service_coll_req_t *allgather_scoll_req =
        task->alltoallv_nsarka.allgather_scoll_req;
    ucc_status_t status = ucc_service_coll_test(allgather_scoll_req);
    if (status < 0) {
        tl_error(coll_task->team->context->lib,
                 "failure during service coll exchange: %s",
                 ucc_status_string(status));
        ucc_service_coll_finalize(allgather_scoll_req);
        task->super.status = status;
        return;
    }
    if (UCC_INPROGRESS == status) {
        return;
    }
    ucc_assert(status == UCC_OK);

    // copy from allgather recvbuf into gwbi
    ucc_tl_ucp_alltoallv_nsarka_allgather_info_finalize(
        allgather_scoll_req, task);

    ucc_service_coll_finalize(
        task->alltoallv_nsarka.allgather_scoll_req);
    task->alltoallv_nsarka.allgather_scoll_req = NULL;
}

static inline void ucc_tl_ucp_alltoallv_nsarka_allgather_free_rkeys(
    ucc_coll_task_t *coll_task)
{
    int                i;
    ucc_base_team_t *  team      = coll_task->team;
    ucc_rank_t         team_size = (ucc_rank_t)team->params.size;
    ucc_tl_ucp_task_t *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);

    for (i = 0; i < team_size; i++) {
        if (!task->alltoallv_nsarka.inplace)
            ucp_rkey_destroy(task->alltoallv_nsarka.src_rkeys[i]);
        ucp_rkey_destroy(task->alltoallv_nsarka.dst_rkeys[i]);
    }
}

static inline void
ucc_tl_ucp_alltoallv_nsarka_barrier(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_base_team_t *  team = coll_task->team;
    ucc_status_t       status;

    ucc_base_coll_args_t coll_args = {
        .team           = coll_task->team->params.team,
        .args.coll_type = UCC_COLL_TYPE_BARRIER,
    };

    status = ucc_tl_ucp_coll_init(&coll_args, team,
                                  &task->alltoallv_nsarka.barrier_task);
    if (status < 0) {
        tl_error(coll_task->team->context->lib,
                 "failure during sliding window barrier init: %s",
                 ucc_status_string(status));
        task->super.status = status;
        return;
    }

    status = ucc_tl_ucp_barrier_knomial_start(
        task->alltoallv_nsarka.barrier_task);
    if (status < 0) {
        tl_error(coll_task->team->context->lib,
                 "failure during sliding window barrier start: %s",
                 ucc_status_string(status));
        task->super.status = status;
    }
}

void ucc_tl_ucp_alltoallv_nsarka_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task    = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_rank_t         size    = (ucc_rank_t)task->subset.map.ep_num;
    ucc_datatype_t     dtype   = TASK_ARGS(task).dst.info_v.datatype;
    size_t             dt_size = ucc_dt_size(dtype);
    /*ucc_count_t       *src_counts  = coll_task->bargs.args.src.info_v.counts;
    ucc_aint_t        *src_displacements = coll_task->bargs.args.src.info_v.displacements;*/ // nick: for now these arent passed through the coll_cmd. if needed, update urom pack/unpack funcs
    ucc_count_t       *dst_counts  = coll_task->bargs.args.dst.info_v.counts;
    ucc_aint_t        *dst_displacements = coll_task->bargs.args.dst.info_v.displacements;
    uint32_t           host_team_size = size;
    ucc_base_team_t *  base_team      = coll_task->team;
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(base_team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_context_t *            tl_ctx = UCC_TL_UCP_TEAM_CTX(tl_team);
    ucp_request_param_t          req_param = {0};
    int                          i         = 0;
    ucc_service_coll_req_t *     allgather_scoll_req =
        task->alltoallv_nsarka.allgather_scoll_req;
    ucc_coll_task_t *barrier_task = task->alltoallv_nsarka.barrier_task;
    ucc_rank_t       rank = UCC_TL_TEAM_RANK(tl_team);
    void *           src_addr;
    void *           dst_addr;
    ucp_rkey_h       rkey;
    ucp_ep_h         ep;

    ucs_status_ptr_t *requests = task->alltoallv_nsarka.requests;
    int *posted = &task->alltoallv_nsarka.gets_posted;
    int *completed = &task->alltoallv_nsarka.gets_completed;

    if (barrier_task != NULL) {
        // mark sliding window task complete once barrier finishes
        if (barrier_task->super.status == UCC_OK) {
            ucc_tl_ucp_put_task(
                ucc_derived_of(task->alltoallv_nsarka.barrier_task,
                               ucc_tl_ucp_task_t));
            task->alltoallv_nsarka.barrier_task = NULL;
            task->super.status = UCC_OK;
        }

        ucc_assert(barrier_task->super.status >= 0);
        return;
    }

    if (allgather_scoll_req != NULL) {
        ucc_tl_ucp_alltoallv_nsarka_allgather_info_test(coll_task);
        return;
    }

    req_param.op_attr_mask |= UCP_OP_ATTR_FIELD_MEMH;

    for (i = *posted; i < host_team_size; i++) {
        size_t data_size = dst_counts[i] * dt_size; // / host_team_size;
        req_param.memh = task->alltoallv_nsarka.dst_ebuf->memh;
        src_addr = dst_displacements[rank] + task->alltoallv_nsarka.sbufs[i];
        dst_addr = dst_displacements[i] + task->alltoallv_nsarka.rbufs[rank];
        rkey = task->alltoallv_nsarka.src_rkeys[i];
        ep = task->alltoallv_nsarka.eps[i];

        printf("nick posting get: data_size=%ld, src_addr: %p, dst_addr: %p, rkey: %p, ep: %p\n", data_size, src_addr, dst_addr, rkey, ep);
        requests[i] = ucp_get_nbx(
                ep, dst_addr,
                data_size, (uint64_t)src_addr,
                rkey, &req_param);

        *posted += 1;
    }

    ucp_worker_progress(tl_ctx->worker.ucp_worker);

    for (i = *completed; i < *posted; i++) {
        if (ucc_tl_ucp_alltoallv_nsarka_req_test(requests[i], task) == UCC_OK) {
            if (requests[i]) ucp_request_free(requests[i]);
            *completed += 1;
        } else {
            break;
        }
    }

    if (*completed == host_team_size)
        task->super.status = UCC_OK;
}
