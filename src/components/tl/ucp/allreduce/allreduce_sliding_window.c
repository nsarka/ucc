/**
 * Copyright(c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "allreduce.h"
#include "utils/ucc_dt_reduce.h"

static inline void
ucc_tl_ucp_allreduce_sliding_window_reset_buf(ucc_tl_ucp_allreduce_sw_buf_t *buf)
{
    buf->state   = FREE;
    buf->count   = 0;
    buf->bytes   = 0;
    buf->ucp_req = NULL;
}

static inline void
ucc_tl_ucp_allreduce_sliding_window_reset_pipeline(ucc_tl_ucp_allreduce_sw_pipeline_t *pipe,
                                                   ucc_rank_t rank)
{
    pipe->avail_buffs   = pipe->num_buffers;
    pipe->src_rank      = pipe->dst_rank       = rank;
    pipe->my_count      = pipe->my_offset      = 0;
    pipe->get_idx       = pipe->red_idx        = 0;
    pipe->done_get      = pipe->done_red       = 0;
    pipe->done_put      = pipe->posted_put     = 0;
    pipe->count_issued  = pipe->count_received = 0;
    pipe->count_reduced = pipe->count_serviced = 0;

    ucc_tl_ucp_allreduce_sliding_window_reset_buf(&pipe->accbuf);
    for (int i=0; i<pipe->num_buffers; i++) {
        ucc_tl_ucp_allreduce_sliding_window_reset_buf(&pipe->getbuf[i]);
    }
}

ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task        = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team        = TASK_TEAM(task);
    ucc_rank_t         size        = coll_task->team->params.size;
    ucc_rank_t         rank        = UCC_TL_TEAM_RANK(team);
    ucc_datatype_t     dtype       = TASK_ARGS(task).dst.info.datatype;
    size_t             dt_size     = ucc_dt_size(dtype);
    uint32_t           count_total = coll_task->bargs.args.src.info.count;
    int                i           = 0;

    ucc_tl_ucp_allreduce_sw_pipeline_t *pipe
                                   = task->allreduce_sliding_window.pipe;
    
    ucc_tl_ucp_allreduce_sliding_window_reset_pipeline(pipe, rank);

    pipe->my_count  = count_total / size;
    pipe->my_offset = pipe->my_count * dt_size * rank;
    if (rank == size - 1) {
        pipe->my_count += count_total % size;
    }

    /* Adjust count and offset for thread id */
    pipe->my_count  /= task->allreduce_sliding_window.nthreads;
    pipe->my_offset += pipe->my_count * dt_size
                        * task->allreduce_sliding_window.tid;

    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);

    for(i = 0; i < task->allreduce_sliding_window.window_size; i++) {
        task->allreduce_sliding_window.put_requests[i] = NULL;
    }

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_status_t       st        = UCC_OK;
    ucc_status_t       global_st = ucc_ee_executor_finalize(
                                    task->allreduce_sliding_window.executor);

    st = ucc_tl_ucp_coll_finalize(&task->super);
    if (ucc_unlikely(st != UCC_OK)) {
        tl_error(UCC_TASK_LIB(task), "failed finalize collective");
        global_st = st;
    }

    return global_st;
}

static inline void
ucc_tl_ucp_allreduce_sliding_window_reduction(ucc_coll_task_t *coll_task,
                                              ucc_tl_ucp_allreduce_sw_buf_t *accbuf,
                                              ucc_tl_ucp_allreduce_sw_buf_t *getbuf)
{
    ucc_status_t       status = UCC_OK;
    ucc_tl_ucp_task_t *task   = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t   *args   = &TASK_ARGS(task);
    ucc_datatype_t     dt     = TASK_ARGS(task).dst.info.datatype;

    status = ucc_dt_reduce(accbuf->buf,
                           getbuf->buf,
                           accbuf->buf,
                           accbuf->count, dt,
                           args, 0,
                           0, task->allreduce_sliding_window.executor,
                           &task->allreduce_sliding_window.etask);

    if (ucc_unlikely(UCC_OK != status)) {
        tl_error(UCC_TASK_LIB(task), "failed to perform dt reduction");
        task->super.status = status;
        return;
    }

    #define SAVE_STATE(_phase) do { } while(0) 
    EXEC_TASK_TEST(1, "failed to perform dt reduction",
                   task->allreduce_sliding_window.etask);
}

static inline ucs_status_t
ucc_tl_ucp_allreduce_sliding_window_req_test(ucs_status_ptr_t request,
                                             ucc_tl_ucp_task_t *task)
{
    if (request == NULL) {
        return UCS_OK;
    } else if (UCS_PTR_IS_ERR(request)) {
        tl_error(UCC_TASK_LIB(task), "unable to complete UCX request=%p: %d\n",
                 request,
                 UCS_PTR_STATUS(request));
        return UCS_PTR_STATUS(request);
    } else {
        return ucp_request_check_status(request);
    }
}

void ucc_tl_ucp_allreduce_sliding_window_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t                  *task           = ucc_derived_of(coll_task,
                                                                        ucc_tl_ucp_task_t);
    ucc_rank_t                          size           = (ucc_rank_t)task->subset.map.ep_num;
    ucc_datatype_t                      dtype          = TASK_ARGS(task).dst.info.datatype;
    size_t                              dt_size        = ucc_dt_size(dtype);
    uint32_t                            host_team_size = size;
    ucc_tl_ucp_allreduce_sw_pipeline_t *pipe           = task->allreduce_sliding_window.pipe;
    ucc_tl_ucp_allreduce_sw_buf_t      *accbuf         = &pipe->accbuf;
    ucp_request_param_t                 req_param      = {0};
    int                                 i              = 0;

    size_t                         remaining_elems;
    size_t                         get_idx;
    size_t                         count;
    size_t                         get_offset;
    size_t                         data_size;
    int                            src_rank;
    void                          *src_addr;
    void                          *dst_addr;
    ucs_status_ptr_t               request;
    size_t                         red_idx;
    ucc_tl_ucp_allreduce_sw_buf_t *redbuf;
    ucc_tl_ucp_allreduce_sw_buf_t *getbuf;
    size_t                         put_offset;
    int                            window;
    int                            put_idx;
    int                            dst_rank;

    if (pipe->count_serviced < pipe->my_count) {
        if ((pipe->count_received < pipe->my_count)
            && (pipe->done_get < host_team_size)
            && (pipe->avail_buffs > 0)
            && (accbuf->state != REDUCED && accbuf->state != SENDING))
        {
            remaining_elems = pipe->my_count - pipe->count_received;
            get_idx         = pipe->get_idx % pipe->num_buffers;
            count           = ucc_min(pipe->buffer_size/dt_size,
                                      remaining_elems);
            get_offset      = pipe->count_received * dt_size
                                + pipe->my_offset;
            data_size       = count * dt_size;
            src_rank        = pipe->src_rank;
            getbuf          = accbuf->state == FREE ? 
                                accbuf : &pipe->getbuf[get_idx];
            src_addr        = task->allreduce_sliding_window.sbufs[src_rank]
                                + get_offset;
            dst_addr        = getbuf->buf;

            assert(getbuf->state == FREE);

            ucp_worker_fence(task->allreduce_sliding_window.worker);

            getbuf->state = RECVING;
            getbuf->count = count;
            getbuf->bytes = data_size;
            getbuf->ucp_req =
                ucp_get_nbx(task->allreduce_sliding_window.host_eps[src_rank],
                            dst_addr, data_size, (uint64_t)src_addr,
                            task->allreduce_sliding_window.src_rkeys[src_rank],
                            &req_param);

            pipe->src_rank = (src_rank + 1) % host_team_size;

            if (getbuf != accbuf) {
                pipe->avail_buffs--;
                pipe->get_idx++;
            }

            pipe->done_get++;
            if (pipe->done_get == host_team_size) {
                pipe->count_received += count;
            }
        }

        if (accbuf->state == RECVING) {
            request = accbuf->ucp_req;
            if (ucc_tl_ucp_allreduce_sliding_window_req_test(request, task)
                == UCS_OK) {
                if (request) ucp_request_free(request);
                accbuf->state = REDUCING;
                accbuf->ucp_req = NULL;
            }
        }

        red_idx = pipe->red_idx % pipe->num_buffers;
        redbuf = &pipe->getbuf[red_idx];
        if (accbuf->state == REDUCING && redbuf->state == RECVING) {
            request = redbuf->ucp_req;
            if (ucc_tl_ucp_allreduce_sliding_window_req_test(request, task)
                == UCS_OK) {
                if (request) ucp_request_free(request);
                redbuf->state = REDUCING;
                redbuf->ucp_req = NULL;

                ucc_tl_ucp_allreduce_sliding_window_reduction(coll_task,
                                                              accbuf, redbuf);

                redbuf->state = FREE;
                pipe->avail_buffs++;
                pipe->red_idx++;
                pipe->done_red++;

                if (pipe->done_red == host_team_size-1) {
                    accbuf->state = REDUCED;
                    pipe->count_reduced += accbuf->count;
                }
            }
        }

        if ((pipe->count_serviced < pipe->count_reduced)
            && (accbuf->state == REDUCED))
        {
            data_size = accbuf->bytes;
            put_offset = pipe->count_serviced * dt_size
                                + pipe->my_offset;

            window = ucc_min(task->allreduce_sliding_window.window_size,
                                 host_team_size - pipe->posted_put);

            for (i = 0; i < window; i++) {
                dst_rank = pipe->dst_rank;
                src_addr = accbuf->buf;
                dst_addr = task->allreduce_sliding_window.rbufs[dst_rank]
                            + put_offset;
                put_idx = pipe->posted_put
                            % task->allreduce_sliding_window.window_size;

                if(task->allreduce_sliding_window.put_requests[put_idx] != NULL) {
                    // We've already posted a put at this index that didn't yet
                    // complete, left this function and came back. Skip to check
                    // whether this request finished instead of overwriting it
                    // with another put
                    break;
                }

                ucp_worker_fence(task->allreduce_sliding_window.worker);
                task->allreduce_sliding_window.put_requests[put_idx] = 
                    ucp_put_nbx(task->allreduce_sliding_window.host_eps[dst_rank],
                                 src_addr, data_size, (uint64_t)dst_addr,
                                 task->allreduce_sliding_window.dst_rkeys[dst_rank],
                                 &req_param);
                pipe->posted_put++;
                pipe->dst_rank = (dst_rank + 1) % host_team_size;
            }

            for (i = pipe->done_put; i < pipe->posted_put; i++) {
                put_idx = i % task->allreduce_sliding_window.window_size;
                request = task->allreduce_sliding_window.put_requests[put_idx];

                // These are fenced, so if the first fails, the proceding will too
                if (ucc_tl_ucp_allreduce_sliding_window_req_test(request, task)
                    != UCS_OK)
                    break;

                if (request != NULL) ucp_request_free(request);
                task->allreduce_sliding_window.put_requests[put_idx] = NULL;
                pipe->done_put++;
            }

            if (pipe->done_put == host_team_size) {
                assert(pipe->avail_buffs == pipe->num_buffers);
                assert(pipe->done_get == host_team_size);
                assert(pipe->done_red == host_team_size-1);
                assert(pipe->done_put == host_team_size);

                pipe->count_serviced += accbuf->count;
                accbuf->state         = FREE;
                accbuf->count         = accbuf->bytes = 0;
                accbuf->ucp_req       = NULL;
                pipe->done_get        = 0;
                pipe->done_red        = pipe->done_put = pipe->posted_put = 0;

                for (i = 0; i < task->allreduce_sliding_window.window_size; i++) {
                    task->allreduce_sliding_window.put_requests[i] = NULL;
                }
            }
        }

        while(ucp_worker_progress(task->allreduce_sliding_window.worker));
    }

    if (pipe->count_serviced == pipe->my_count)
        task->super.status = UCC_OK;

    return;
}
