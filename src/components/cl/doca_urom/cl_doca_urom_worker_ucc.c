/*
 * Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES, ALL RIGHTS RESERVED.
 *
 * This software product is a proprietary product of NVIDIA CORPORATION &
 * AFFILIATES (the "Company") and all right, title, and interest in and to the
 * software product, including all associated intellectual property rights, are
 * and shall remain exclusively with the Company.
 *
 * This software product is governed by the End User License Agreement
 * provided with the software product.
 *
 */

#include <stdint.h>
#include <stdlib.h>

#include <doca_buf.h>
#include <doca_pe.h>

#include "cl_doca_urom_worker_ucc.h"

DOCA_LOG_REGISTER(UCC::DOCA_CL : WORKER_UCC);

static uint64_t ucc_id;             /* UCC plugin id, id is generated by UROM lib and
                                     * will be updated in init function */
static uint64_t ucc_version = 0x01; /* UCC plugin host version */

/* UCC task metadata */
struct ucc_cl_doca_urom_task_data {
    union doca_data cookie;                                   /* User cookie */
    union {
        ucc_cl_doca_urom_lib_create_finished_cb  lib_create;  /* User lib create task callback */
        ucc_cl_doca_urom_lib_destroy_finished_cb lib_destroy; /* User lib destroy task callback */
        ucc_cl_doca_urom_ctx_create_finished_cb  ctx_create;  /* User context create task callback */
        ucc_cl_doca_urom_ctx_destroy_finished_cb ctx_destroy; /* User context destroy task callback */
        ucc_cl_doca_urom_team_create_finished_cb team_create; /* User UCC team create task callback */
        ucc_cl_doca_urom_collective_finished_cb  collective;  /* User UCC collective task callback */
        ucc_cl_doca_urom_pd_channel_finished_cb  pd_channel;  /* User passive data channel task callback */
    };
};

/*
 * UCC notification unpack function
 *
 * @packed_notif [in]: packed UCC notification buffer
 * @ucc_notif [out]: set unpacked UCC notification
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t notif_unpack(void                           *packed_notif,
                                 struct urom_worker_notify_ucc **ucc_notif)
{
    *ucc_notif = packed_notif;
    return DOCA_SUCCESS;
}

/*
 * UCC common command's completion callback function
 *
 * @task [in]: UROM worker task
 * @type [in]: UCC task type
 */
static void completion(struct doca_urom_worker_cmd_task *task,
                       enum urom_worker_ucc_notify_type  type)
{
    struct urom_worker_notify_ucc      notify_error = {0};
    struct urom_worker_notify_ucc     *ucc_notify   = &notify_error;
    struct ucc_cl_doca_urom_task_data *task_data;
    struct doca_buf                   *response;
    doca_error_t                       result;
    size_t                             data_len;

    notify_error.notify_type = type;

    task_data = (struct ucc_cl_doca_urom_task_data *)
                    doca_urom_worker_cmd_task_get_user_data(task);
    if (task_data == NULL) {
        DOCA_LOG_ERR("Failed to get task data buffer");
        goto task_release;
    }

    response = doca_urom_worker_cmd_task_get_response(task);
    if (response == NULL) {
        DOCA_LOG_ERR("Failed to get task response buffer");
        result = DOCA_ERROR_INVALID_VALUE;
        goto error_exit;
    }

    result = doca_buf_get_data(response, (void **)&ucc_notify);
    if (result != DOCA_SUCCESS)
        goto error_exit;

    result = notif_unpack((void *)ucc_notify, &ucc_notify);
    if (result != DOCA_SUCCESS)
        goto error_exit;

    result = doca_buf_get_data_len(response, &data_len);
    if (result != DOCA_SUCCESS) {
        DOCA_LOG_ERR("Failed to get response data length");
        goto error_exit;
    }

    result = doca_task_get_status(doca_urom_worker_cmd_task_as_task(task));
    if (result != DOCA_SUCCESS)
        goto error_exit;

    if (data_len != sizeof(*ucc_notify)) {
        DOCA_LOG_ERR("Task response data length is different"
                     "from notification expected length");
        result = DOCA_ERROR_INVALID_VALUE;
        goto error_exit;
    }

error_exit:
    switch (ucc_notify->notify_type) {
    case UROM_WORKER_NOTIFY_UCC_LIB_CREATE_COMPLETE:
        (task_data->lib_create)(result, task_data->cookie,
                                ucc_notify->dpu_worker_id);
        break;
    case UROM_WORKER_NOTIFY_UCC_LIB_DESTROY_COMPLETE:
        (task_data->lib_destroy)(result, task_data->cookie,
                                 ucc_notify->dpu_worker_id);
        break;
    case UROM_WORKER_NOTIFY_UCC_CONTEXT_CREATE_COMPLETE:
        (task_data->ctx_create)(result,
                    task_data->cookie,
                    ucc_notify->dpu_worker_id,
                    ucc_notify->context_create_nqe.context);
        break;
    case UROM_WORKER_NOTIFY_UCC_CONTEXT_DESTROY_COMPLETE:
        (task_data->ctx_destroy)(result, task_data->cookie,
                                 ucc_notify->dpu_worker_id);
        break;
    case UROM_WORKER_NOTIFY_UCC_TEAM_CREATE_COMPLETE:
        (task_data->team_create)(result,
                     task_data->cookie,
                     ucc_notify->dpu_worker_id,
                     ucc_notify->team_create_nqe.team);
        break;
    case UROM_WORKER_NOTIFY_UCC_COLLECTIVE_COMPLETE:
        (task_data->collective)(result,
                    task_data->cookie,
                    ucc_notify->dpu_worker_id,
                    ucc_notify->coll_nqe.status);
        break;
    case UROM_WORKER_NOTIFY_UCC_PASSIVE_DATA_CHANNEL_COMPLETE:
        (task_data->pd_channel)(result,
                    task_data->cookie,
                    ucc_notify->dpu_worker_id,
                    ucc_notify->pass_dc_nqe.status);
        break;
    default:
        DOCA_LOG_ERR("Invalid UCC notification type %u",
                     ucc_notify->notify_type);
        break;
    }

task_release:
    result = doca_urom_worker_cmd_task_release(task);
    if (result != DOCA_SUCCESS)
        DOCA_LOG_ERR("Failed to release worker command task %s",
                      doca_error_get_descr(result));
}

/*
 * Pack UCC command
 *
 * @ucc_cmd [in]: ucc command
 * @packed_cmd_len [in/out]: packed command buffer size
 * @packed_cmd [out]: packed command buffer
 * @return: DOCA_SUCCESS on success and DOCA_ERROR otherwise
 */
static doca_error_t cmd_pack(struct urom_worker_ucc_cmd *ucc_cmd,
                             size_t                     *packed_cmd_len,
                             void                       *packed_cmd)
{
    void            *pack_tail = packed_cmd;
    void            *pack_head;
    size_t           pack_len;
    size_t           team_size;
    size_t           disp_pack_size;
    size_t           count_pack_size;
    ucc_coll_args_t *coll_args;
    int is_count_64, is_disp_64;

    pack_len = sizeof(struct urom_worker_ucc_cmd);
    if (pack_len > *packed_cmd_len)
        return DOCA_ERROR_INITIALIZATION;

    /* Pack base command */
    pack_head = urom_ucc_serialize_next_raw(&pack_tail, void, pack_len);
    memcpy(pack_head, ucc_cmd, pack_len);
    *packed_cmd_len = pack_len;

    switch (ucc_cmd->cmd_type) {
    case UROM_WORKER_CMD_UCC_LIB_CREATE:
        pack_len = sizeof(ucc_lib_params_t);
        pack_head = urom_ucc_serialize_next_raw(&pack_tail, void, pack_len);
        memcpy(pack_head, ucc_cmd->lib_create_cmd.params, pack_len);
        *packed_cmd_len += pack_len;
        break;
    case UROM_WORKER_CMD_UCC_CONTEXT_CREATE:
        if (ucc_cmd->context_create_cmd.stride <= 0) {
            pack_len = sizeof(int64_t) * ucc_cmd->context_create_cmd.size;
            pack_head = urom_ucc_serialize_next_raw(&pack_tail, void, pack_len);
            memcpy(pack_head, ucc_cmd->context_create_cmd.array, pack_len);
            *packed_cmd_len += pack_len;
        }
        break;
    case UROM_WORKER_CMD_UCC_COLL:
        coll_args = ucc_cmd->coll_cmd.coll_args;
        pack_len = sizeof(ucc_coll_args_t);
        pack_head = urom_ucc_serialize_next_raw(&pack_tail, void, pack_len);
        memcpy(pack_head, ucc_cmd->coll_cmd.coll_args, pack_len);
        *packed_cmd_len += pack_len;
        pack_len = ucc_cmd->coll_cmd.work_buffer_size;
        if (pack_len > 0 && ucc_cmd->coll_cmd.work_buffer) {
            pack_head = urom_ucc_serialize_next_raw(&pack_tail, void, pack_len);
            memcpy(pack_head, ucc_cmd->coll_cmd.work_buffer, pack_len);
            *packed_cmd_len += pack_len;
        }

        if (coll_args->coll_type == UCC_COLL_TYPE_ALLTOALLV       ||
            coll_args->coll_type == UCC_COLL_TYPE_ALLGATHERV      ||
            coll_args->coll_type == UCC_COLL_TYPE_GATHERV         ||
            coll_args->coll_type == UCC_COLL_TYPE_REDUCE_SCATTERV ||
            coll_args->coll_type == UCC_COLL_TYPE_SCATTERV) {
            team_size = ucc_cmd->coll_cmd.team_size;
            is_count_64 = ((coll_args->mask & UCC_COLL_ARGS_FIELD_FLAGS) &&
                       (coll_args->flags & UCC_COLL_ARGS_FLAG_COUNT_64BIT));
            is_disp_64 = ((coll_args->mask & UCC_COLL_ARGS_FIELD_FLAGS) &&
                      (coll_args->flags & UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT));
            count_pack_size = team_size * ((is_count_64) ? sizeof(uint64_t) : sizeof(uint32_t));
            disp_pack_size = team_size * ((is_disp_64) ? sizeof(uint64_t) : sizeof(uint32_t));
            pack_len = count_pack_size;
            pack_head = urom_ucc_serialize_next_raw(&pack_tail, void, pack_len);
            memcpy(pack_head, coll_args->src.info_v.counts, pack_len);
            *packed_cmd_len += pack_len;
            pack_head = urom_ucc_serialize_next_raw(&pack_tail, void, pack_len);
            memcpy(pack_head, coll_args->dst.info_v.counts, pack_len);
            *packed_cmd_len += pack_len;

            pack_len = disp_pack_size;
            pack_head = urom_ucc_serialize_next_raw(&pack_tail, void, pack_len);
            memcpy(pack_head, coll_args->src.info_v.displacements, pack_len);
            *packed_cmd_len += pack_len;
            pack_head = urom_ucc_serialize_next_raw(&pack_tail, void, pack_len);
            memcpy(pack_head, coll_args->dst.info_v.displacements, pack_len);
            *packed_cmd_len += pack_len;
        }
        break;
    case UROM_WORKER_CMD_UCC_CREATE_PASSIVE_DATA_CHANNEL:
        pack_len = ucc_cmd->pass_dc_create_cmd.addr_len;
        pack_head = urom_ucc_serialize_next_raw(&pack_tail, void, pack_len);
        memcpy(pack_head, ucc_cmd->pass_dc_create_cmd.ucp_addr, pack_len);
        *packed_cmd_len += pack_len;
        break;
    default:
        DOCA_LOG_ERR("Invalid UCC cmd type %u",
                      ucc_cmd->cmd_type);
        break;
    }
    return DOCA_SUCCESS;
}

/*
 * UCC library create command completion callback function, user callback will be called inside the function
 *
 * @task [in]: UROM worker task
 * @task_user_data [in]: task user data
 * @ctx_user_data [in]: worker context user data
 */
static void lib_create_completed(struct doca_urom_worker_cmd_task *task,
                                 union doca_data                   task_user_data,
                                 union doca_data                   ctx_user_data)
{
    (void)task_user_data;
    (void)ctx_user_data;
    completion(task, UROM_WORKER_NOTIFY_UCC_LIB_CREATE_COMPLETE);
}

doca_error_t ucc_cl_doca_urom_task_lib_create(
                struct doca_urom_worker                *worker_ctx,
                union doca_data                         cookie,
                uint64_t                                dpu_worker_id,
                void                                   *params,
                ucc_cl_doca_urom_lib_create_finished_cb cb)
{
    size_t                             pack_len = 0;
    struct doca_buf                   *payload;
    struct doca_urom_worker_cmd_task  *task;
    struct ucc_cl_doca_urom_task_data *task_data;
    struct urom_worker_ucc_cmd        *ucc_cmd;
    doca_error_t                       result;

    /* Allocate task */
    result = doca_urom_worker_cmd_task_allocate_init(worker_ctx, ucc_id, &task);
    if (result != DOCA_SUCCESS)
        return result;

    payload = doca_urom_worker_cmd_task_get_payload(task);
    result = doca_buf_get_data(payload, (void **)&ucc_cmd);
    if (result != DOCA_SUCCESS)
        goto task_destroy;

    result = doca_buf_get_data_len(payload, &pack_len);
    if (result != DOCA_SUCCESS)
        goto task_destroy;

    /* Populate commands attributes */
    ucc_cmd->cmd_type = UROM_WORKER_CMD_UCC_LIB_CREATE;
    ucc_cmd->dpu_worker_id = dpu_worker_id;
    ucc_cmd->lib_create_cmd.params = params;

    result = cmd_pack(ucc_cmd, &pack_len, (void *)ucc_cmd);
    if (result != DOCA_SUCCESS)
        goto task_destroy;

    result = doca_buf_set_data(payload, ucc_cmd, pack_len);
    if (result != DOCA_SUCCESS)
        goto task_destroy;

    task_data = (struct ucc_cl_doca_urom_task_data *)
                    doca_urom_worker_cmd_task_get_user_data(task);
    task_data->lib_create = cb;
    task_data->cookie = cookie;

    doca_urom_worker_cmd_task_set_cb(task, lib_create_completed);

    result = doca_task_submit(doca_urom_worker_cmd_task_as_task(task));
    if (result != DOCA_SUCCESS)
        goto task_destroy;

    return DOCA_SUCCESS;

task_destroy:
    doca_urom_worker_cmd_task_release(task);
    return result;
}

/*
 * UCC library destroy command completion callback function,
 * user callback will be called inside the function
 *
 * @task [in]: UROM worker task
 * @task_user_data [in]: task user data
 * @ctx_user_data [in]: worker context user data
 */
static void lib_destroy_completed(struct doca_urom_worker_cmd_task *task,
                                  union doca_data task_user_data,
                                  union doca_data ctx_user_data)
{
    (void)task_user_data;
    (void)ctx_user_data;
    completion(task, UROM_WORKER_NOTIFY_UCC_LIB_DESTROY_COMPLETE);
}

doca_error_t doca_urom_ucc_task_lib_destroy(
                struct doca_urom_worker                 *worker_ctx,
                union doca_data                          cookie,
                uint64_t                                 dpu_worker_id,
                ucc_cl_doca_urom_lib_destroy_finished_cb cb)
{
    size_t                             pack_len = 0;
    struct doca_buf                   *payload;
    struct doca_urom_worker_cmd_task  *task;
    struct ucc_cl_doca_urom_task_data *task_data;
    struct urom_worker_ucc_cmd        *ucc_cmd;
    doca_error_t                       result;

    /* Allocate task */
    result = doca_urom_worker_cmd_task_allocate_init(worker_ctx, ucc_id, &task);
    if (result != DOCA_SUCCESS)
        return result;

    payload = doca_urom_worker_cmd_task_get_payload(task);
    result = doca_buf_get_data(payload, (void **)&ucc_cmd);
    if (result != DOCA_SUCCESS)
        goto task_destroy;

    result = doca_buf_get_data_len(payload, &pack_len);
    if (result != DOCA_SUCCESS)
        goto task_destroy;

    /* Populate commands attributes */
    ucc_cmd->cmd_type = UROM_WORKER_CMD_UCC_LIB_DESTROY;
    ucc_cmd->dpu_worker_id = dpu_worker_id;

    result = cmd_pack(ucc_cmd, &pack_len, (void *)ucc_cmd);
    if (result != DOCA_SUCCESS)
        goto task_destroy;

    result = doca_buf_set_data(payload, ucc_cmd, pack_len);
    if (result != DOCA_SUCCESS)
        goto task_destroy;

    task_data = (struct ucc_cl_doca_urom_task_data *)
                    doca_urom_worker_cmd_task_get_user_data(task);
    task_data->lib_destroy = cb;
    task_data->cookie = cookie;

    doca_urom_worker_cmd_task_set_cb(task, lib_destroy_completed);

    result = doca_task_submit(doca_urom_worker_cmd_task_as_task(task));
    if (result != DOCA_SUCCESS)
        goto task_destroy;

    return DOCA_SUCCESS;

task_destroy:
    doca_urom_worker_cmd_task_release(task);
    return result;
}

/*
 * UCC context create command completion callback function,
 * user callback will be called inside the function
 *
 * @task [in]: UROM worker task
 * @task_user_data [in]: task user data
 * @ctx_user_data [in]: worker context user data
 */
static void ctx_create_completed(struct doca_urom_worker_cmd_task *task,
                                 union doca_data task_user_data,
                                 union doca_data ctx_user_data)
{
    (void)task_user_data;
    (void)ctx_user_data;
    completion(task, UROM_WORKER_NOTIFY_UCC_CONTEXT_CREATE_COMPLETE);
}

doca_error_t ucc_cl_doca_urom_task_ctx_create(
                struct doca_urom_worker                *worker_ctx,
                union doca_data                         cookie,
                uint64_t                                dpu_worker_id,
                int64_t                                 start,
                int64_t                                *array,
                int64_t                                 stride,
                int64_t                                 size,
                void                                   *base_va,
                uint64_t                                len,
                ucc_cl_doca_urom_ctx_create_finished_cb cb)
{
    size_t                             pack_len = 0;
    struct ucc_cl_doca_urom_task_data *task_data;
    struct doca_urom_worker_cmd_task  *task;
    struct urom_worker_ucc_cmd        *ucc_cmd;
    struct doca_buf                   *payload;
    doca_error_t                       result;

    /* Allocate task */
    result = doca_urom_worker_cmd_task_allocate_init(worker_ctx, ucc_id, &task);
    if (result != DOCA_SUCCESS)
        return result;

    payload = doca_urom_worker_cmd_task_get_payload(task);
    result = doca_buf_get_data(payload, (void **)&ucc_cmd);
    if (result != DOCA_SUCCESS)
        goto task_destroy;

    result = doca_buf_get_data_len(payload, &pack_len);
    if (result != DOCA_SUCCESS)
        goto task_destroy;

    /* Populate commands attributes */
    ucc_cmd->cmd_type = UROM_WORKER_CMD_UCC_CONTEXT_CREATE;
    ucc_cmd->dpu_worker_id = dpu_worker_id;
    if (array == NULL) {
        ucc_cmd->context_create_cmd.start = start;
    } else {
        ucc_cmd->context_create_cmd.array = array;
    }

    ucc_cmd->context_create_cmd.stride  = stride;
    ucc_cmd->context_create_cmd.size    = size;
    ucc_cmd->context_create_cmd.base_va = base_va;
    ucc_cmd->context_create_cmd.len     = len;

    result = cmd_pack(ucc_cmd, &pack_len, (void *)ucc_cmd);
    if (result != DOCA_SUCCESS)
        goto task_destroy;

    result = doca_buf_set_data(payload, ucc_cmd, pack_len);
    if (result != DOCA_SUCCESS)
        goto task_destroy;

    task_data             = (struct ucc_cl_doca_urom_task_data *)
                                doca_urom_worker_cmd_task_get_user_data(task);
    task_data->ctx_create = cb;
    task_data->cookie     = cookie;

    doca_urom_worker_cmd_task_set_cb(task, ctx_create_completed);

    result = doca_task_submit(doca_urom_worker_cmd_task_as_task(task));
    if (result != DOCA_SUCCESS)
        goto task_destroy;

    return DOCA_SUCCESS;

task_destroy:
    doca_urom_worker_cmd_task_release(task);
    return result;
}

/*
 * UCC context destroy command completion callback function,
 * user callback will be called inside the function
 *
 * @task [in]: UROM worker task
 * @task_user_data [in]: task user data
 * @ctx_user_data [in]: worker context user data
 */
static void ctx_destroy_completed(struct doca_urom_worker_cmd_task *task,
                                  union doca_data task_user_data,
                                  union doca_data ctx_user_data)
{
    (void)task_user_data;
    (void)ctx_user_data;
    completion(task, UROM_WORKER_NOTIFY_UCC_CONTEXT_DESTROY_COMPLETE);
}

doca_error_t ucc_cl_doca_urom_task_ctx_destroy(
                struct doca_urom_worker                 *worker_ctx,
                union doca_data                          cookie,
                uint64_t                                 dpu_worker_id,
                void                                    *context,
                ucc_cl_doca_urom_ctx_destroy_finished_cb cb)
{
    size_t                             pack_len = 0;
    struct ucc_cl_doca_urom_task_data *task_data;
    struct doca_urom_worker_cmd_task  *task;
    struct urom_worker_ucc_cmd        *ucc_cmd;
    struct doca_buf                   *payload;
    doca_error_t                       result;

    /* Allocate task */
    result = doca_urom_worker_cmd_task_allocate_init(worker_ctx, ucc_id, &task);
    if (result != DOCA_SUCCESS)
        return result;

    payload = doca_urom_worker_cmd_task_get_payload(task);
    result = doca_buf_get_data(payload, (void **)&ucc_cmd);
    if (result != DOCA_SUCCESS)
        goto task_destroy;

    result = doca_buf_get_data_len(payload, &pack_len);
    if (result != DOCA_SUCCESS)
        goto task_destroy;

    /* Populate commands attributes */
    ucc_cmd->cmd_type = UROM_WORKER_CMD_UCC_CONTEXT_DESTROY;
    ucc_cmd->dpu_worker_id = dpu_worker_id;
    ucc_cmd->context_destroy_cmd.context_h = context;

    result = cmd_pack(ucc_cmd, &pack_len, (void *)ucc_cmd);
    if (result != DOCA_SUCCESS)
        goto task_destroy;

    result = doca_buf_set_data(payload, ucc_cmd, pack_len);
    if (result != DOCA_SUCCESS)
        goto task_destroy;

    task_data              = (struct ucc_cl_doca_urom_task_data *)
                                doca_urom_worker_cmd_task_get_user_data(task);
    task_data->ctx_destroy = cb;
    task_data->cookie      = cookie;

    doca_urom_worker_cmd_task_set_cb(task, ctx_destroy_completed);

    result = doca_task_submit(doca_urom_worker_cmd_task_as_task(task));
    if (result != DOCA_SUCCESS)
        goto task_destroy;

    return DOCA_SUCCESS;

task_destroy:
    doca_urom_worker_cmd_task_release(task);
    return result;
}

/*
 * UCC team create command completion callback function, user callback will be called inside the function
 *
 * @task [in]: UROM worker task
 * @task_user_data [in]: task user data
 * @ctx_user_data [in]: worker context user data
 */
static void team_create_completed(struct doca_urom_worker_cmd_task *task,
                                  union doca_data task_user_data,
                                  union doca_data ctx_user_data)
{
    (void)task_user_data;
    (void)ctx_user_data;
    completion(task, UROM_WORKER_NOTIFY_UCC_TEAM_CREATE_COMPLETE);
}

doca_error_t ucc_cl_doca_urom_task_team_create(
                struct doca_urom_worker                *worker_ctx,
                union doca_data                         cookie,
                uint64_t                                dpu_worker_id,
                int64_t                                 start,
                int64_t                                 stride,
                int64_t                                 size,
                void                                    *context,
                ucc_cl_doca_urom_team_create_finished_cb cb)
{
    size_t                             pack_len = 0;
    struct ucc_cl_doca_urom_task_data *task_data;
    struct doca_urom_worker_cmd_task  *task;
    struct urom_worker_ucc_cmd        *ucc_cmd;
    struct doca_buf                   *payload;
    doca_error_t                       result;

    /* Allocate task */
    result = doca_urom_worker_cmd_task_allocate_init(worker_ctx, ucc_id, &task);
    if (result != DOCA_SUCCESS)
        return result;

    payload = doca_urom_worker_cmd_task_get_payload(task);
    result = doca_buf_get_data(payload, (void **)&ucc_cmd);
    if (result != DOCA_SUCCESS)
        goto task_destroy;

    result = doca_buf_get_data_len(payload, &pack_len);
    if (result != DOCA_SUCCESS)
        goto task_destroy;

    /* Populate commands attributes */
    ucc_cmd->cmd_type = UROM_WORKER_CMD_UCC_TEAM_CREATE;
    ucc_cmd->dpu_worker_id = dpu_worker_id;
    ucc_cmd->team_create_cmd.start = start;
    ucc_cmd->team_create_cmd.stride = stride;
    ucc_cmd->team_create_cmd.size = size;
    ucc_cmd->team_create_cmd.context_h = context;

    result = cmd_pack(ucc_cmd, &pack_len, (void *)ucc_cmd);
    if (result != DOCA_SUCCESS)
        goto task_destroy;

    result = doca_buf_set_data(payload, ucc_cmd, pack_len);
    if (result != DOCA_SUCCESS)
        goto task_destroy;

    task_data              = (struct ucc_cl_doca_urom_task_data *)
                                doca_urom_worker_cmd_task_get_user_data(task);
    task_data->team_create = cb;
    task_data->cookie      = cookie;

    doca_urom_worker_cmd_task_set_cb(task, team_create_completed);

    result = doca_task_submit(doca_urom_worker_cmd_task_as_task(task));
    if (result != DOCA_SUCCESS)
        goto task_destroy;

    return DOCA_SUCCESS;

task_destroy:
    doca_urom_worker_cmd_task_release(task);
    return result;
}

/*
 * UCC collective command completion callback function,
 * user callback will be called inside the function
 *
 * @task [in]: UROM worker task
 * @task_user_data [in]: task user data
 * @ctx_user_data [in]: worker context user data
 */
static void collective_completed(struct doca_urom_worker_cmd_task *task,
                union doca_data task_user_data,
                union doca_data ctx_user_data)
{
    (void)task_user_data;
    (void)ctx_user_data;
    completion(task, UROM_WORKER_NOTIFY_UCC_COLLECTIVE_COMPLETE);
}

doca_error_t ucc_cl_doca_urom_task_collective(
                struct doca_urom_worker                *worker_ctx,
                union doca_data                         cookie,
                uint64_t                                dpu_worker_id,
                void                                   *coll_args,
                void                                   *team,
                int                                     use_xgvmi,
                void                                   *work_buffer,
                size_t                                  work_buffer_size,
                size_t                                  team_size,
                ucc_cl_doca_urom_collective_finished_cb cb)
{
    size_t                             pack_len = 0;
    struct ucc_cl_doca_urom_task_data *task_data;
    struct doca_urom_worker_cmd_task  *task;
    struct urom_worker_ucc_cmd        *ucc_cmd;
    struct doca_buf                   *payload;
    doca_error_t                       result;

    /* Allocate task */
    result = doca_urom_worker_cmd_task_allocate_init(worker_ctx, ucc_id, &task);
    if (result != DOCA_SUCCESS)
        return result;

    payload = doca_urom_worker_cmd_task_get_payload(task);
    result = doca_buf_get_data(payload, (void **)&ucc_cmd);
    if (result != DOCA_SUCCESS)
        goto task_destroy;

    result = doca_buf_get_data_len(payload, &pack_len);
    if (result != DOCA_SUCCESS)
        goto task_destroy;

    /* Populate commands attributes */
    ucc_cmd->cmd_type = UROM_WORKER_CMD_UCC_COLL;
    ucc_cmd->dpu_worker_id = dpu_worker_id;
    ucc_cmd->coll_cmd.coll_args = coll_args;
    ucc_cmd->coll_cmd.team = team;
    ucc_cmd->coll_cmd.use_xgvmi = use_xgvmi;
    ucc_cmd->coll_cmd.work_buffer = work_buffer;
    ucc_cmd->coll_cmd.work_buffer_size = work_buffer_size;
    ucc_cmd->coll_cmd.team_size = team_size;

    result = cmd_pack(ucc_cmd, &pack_len, (void *)ucc_cmd);
    if (result != DOCA_SUCCESS)
        goto task_destroy;

    result = doca_buf_set_data(payload, ucc_cmd, pack_len);
    if (result != DOCA_SUCCESS)
        goto task_destroy;

    task_data             = (struct ucc_cl_doca_urom_task_data *)
                                doca_urom_worker_cmd_task_get_user_data(task);
    task_data->collective = cb;
    task_data->cookie     = cookie;

    doca_urom_worker_cmd_task_set_cb(task, collective_completed);

    result = doca_task_submit(doca_urom_worker_cmd_task_as_task(task));
    if (result != DOCA_SUCCESS)
        goto task_destroy;

    return DOCA_SUCCESS;

task_destroy:
    doca_urom_worker_cmd_task_release(task);
    return result;
}

/*
 * UCC passive data channel command completion callback function,
 * user callback will be called inside the function
 *
 * @task [in]: UROM worker task
 * @task_user_data [in]: task user data
 * @ctx_user_data [in]: worker context user data
 */
static void pd_channel_completed(struct doca_urom_worker_cmd_task *task,
                union doca_data task_user_data,
                union doca_data ctx_user_data)
{
    (void)task_user_data;
    (void)ctx_user_data;
    completion(task, UROM_WORKER_NOTIFY_UCC_PASSIVE_DATA_CHANNEL_COMPLETE);
}

doca_error_t ucc_cl_doca_urom_task_pd_channel(
                struct doca_urom_worker                *worker_ctx,
                union doca_data                         cookie,
                uint64_t                                dpu_worker_id,
                void                                   *ucp_addr,
                size_t                                  addr_len,
                ucc_cl_doca_urom_pd_channel_finished_cb cb)
{
    size_t                             pack_len = 0;
    struct ucc_cl_doca_urom_task_data *task_data;
    struct doca_urom_worker_cmd_task  *task;
    struct urom_worker_ucc_cmd        *ucc_cmd;
    struct doca_buf                   *payload;
    doca_error_t                       result;

    /* Allocate task */
    result = doca_urom_worker_cmd_task_allocate_init(worker_ctx, ucc_id, &task);
    if (result != DOCA_SUCCESS)
        return result;

    payload = doca_urom_worker_cmd_task_get_payload(task);
    result = doca_buf_get_data(payload, (void **)&ucc_cmd);
    if (result != DOCA_SUCCESS)
        goto task_destroy;

    result = doca_buf_get_data_len(payload, &pack_len);
    if (result != DOCA_SUCCESS)
        goto task_destroy;

    /* Populate commands attributes */
    ucc_cmd->cmd_type                    =
        UROM_WORKER_CMD_UCC_CREATE_PASSIVE_DATA_CHANNEL;
    ucc_cmd->dpu_worker_id               = dpu_worker_id;
    ucc_cmd->pass_dc_create_cmd.ucp_addr = ucp_addr;
    ucc_cmd->pass_dc_create_cmd.addr_len = addr_len;

    result = cmd_pack(ucc_cmd, &pack_len, (void *)ucc_cmd);
    if (result != DOCA_SUCCESS)
        goto task_destroy;

    result = doca_buf_set_data(payload, ucc_cmd, pack_len);
    if (result != DOCA_SUCCESS)
        goto task_destroy;

    task_data             = (struct ucc_cl_doca_urom_task_data *)
                                doca_urom_worker_cmd_task_get_user_data(task);
    task_data->pd_channel = cb;
    task_data->cookie     = cookie;

    doca_urom_worker_cmd_task_set_cb(task, pd_channel_completed);

    result = doca_task_submit(doca_urom_worker_cmd_task_as_task(task));
    if (result != DOCA_SUCCESS)
        goto task_destroy;

    return DOCA_SUCCESS;

task_destroy:
    doca_urom_worker_cmd_task_release(task);
    return result;
}

doca_error_t ucc_cl_doca_urom_save_plugin_id(uint64_t plugin_id,
                                             uint64_t version)
{
    if (version != ucc_version) {
        return DOCA_ERROR_UNSUPPORTED_VERSION;
    }

    ucc_id = plugin_id;
    return DOCA_SUCCESS;
}

/*
 * UCC lib create callback
 *
 * @result [in]: task result
 * @cookie [in]: program cookie
 * @dpu_worker_id [in]: UROM DPU worker id
 */
void ucc_cl_doca_urom_lib_create_finished(
        doca_error_t result, union doca_data cookie, uint64_t dpu_worker_id)
{
    struct ucc_cl_doca_urom_result *res =
        (struct ucc_cl_doca_urom_result *)cookie.ptr;
    if (res == NULL) {
        return;
    }

    res->dpu_worker_id = dpu_worker_id;
    res->result        = result;
}

/*
 * UCC passive data channel callback
 *
 * @result [in]: task result
 * @cookie [in]: program cookie
 * @dpu_worker_id [in]: UROM DPU worker id
 * @status [in]: channel creation status
 */
void ucc_cl_doca_urom_pss_dc_finished(
        doca_error_t result, union doca_data cookie,
        uint64_t dpu_worker_id, ucc_status_t status)
{
    struct ucc_cl_doca_urom_result *res =
        (struct ucc_cl_doca_urom_result *)cookie.ptr;
    if (res == NULL) {
        return;
    }

    res->dpu_worker_id  = dpu_worker_id;
    res->result         = result;
    res->pass_dc.status = status;
}

/*
 * UCC lib destroy callback
 *
 * @result [in]: task result
 * @cookie [in]: program cookie
 * @dpu_worker_id [in]: UROM DPU worker id
 */
void ucc_cl_doca_urom_lib_destroy_finished(
        doca_error_t result, union doca_data cookie,
        uint64_t dpu_worker_id)
{
    struct ucc_cl_doca_urom_result *res =
        (struct ucc_cl_doca_urom_result *)cookie.ptr;
    if (res == NULL) {
        return;
    }

    res->dpu_worker_id = dpu_worker_id;
    res->result        = result;
}

/*
 * UCC context create callback
 *
 * @result [in]: task result
 * @cookie [in]: program cookie
 * @dpu_worker_id [in]: UROM DPU worker id
 * @context [in]: pointer to UCC context
 */
void ucc_cl_doca_urom_ctx_create_finished(
        doca_error_t result, union doca_data cookie,
        uint64_t dpu_worker_id, void *context)
{
    struct ucc_cl_doca_urom_result *res =
        (struct ucc_cl_doca_urom_result *)cookie.ptr;
    if (res == NULL) {
        return;
    }

    res->dpu_worker_id          = dpu_worker_id;
    res->result                 = result;
    res->context_create.context = context;
}

/*
 * UCC collective callback
 *
 * @result [in]: task result
 * @cookie [in]: program cookie
 * @dpu_worker_id [in]: UROM DPU worker id
 * @status [in]: collective status
 */
void ucc_cl_doca_urom_collective_finished(
        doca_error_t result, union doca_data cookie,
        uint64_t dpu_worker_id, ucc_status_t status)
{
    struct ucc_cl_doca_urom_result *res =
        (struct ucc_cl_doca_urom_result *)cookie.ptr;
    if (res == NULL) {
        return;
    }

    res->dpu_worker_id     = dpu_worker_id;
    res->result            = result;
    res->collective.status = status;
}

/*
 * UCC team create callback
 *
 * @result [in]: task result
 * @cookie [in]: program cookie
 * @dpu_worker_id [in]: UROM DPU worker id
 * @team [in]: pointer to UCC team
 */
void ucc_cl_doca_urom_team_create_finished(
        doca_error_t result, union doca_data cookie,
        uint64_t dpu_worker_id, void *team)
{
    struct ucc_cl_doca_urom_result *res =
        (struct ucc_cl_doca_urom_result *)cookie.ptr;
    if (res == NULL) {
        return;
    }

    res->dpu_worker_id      = dpu_worker_id;
    res->result             = result;
    res->team_create.team   = team;
    res->team_create.status = 2; // set done
}

ucc_status_t ucc_cl_doca_urom_buffer_export_ucc(
        ucp_context_h ucp_context, void *buf,
        size_t len, struct export_buf *ebuf)
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
        return UCC_ERR_NO_RESOURCE;
    }
    ucs_status = ucp_rkey_pack(ucp_context, ebuf->memh, &ebuf->packed_key,
                               &ebuf->packed_key_len);
    if (UCS_OK != ucs_status) {
        printf("ucp_rkey_pack() returned error: %s\n",
               ucs_status_string(ucs_status));
        return UCC_ERR_NO_RESOURCE;
    }

    return UCC_OK;
}
