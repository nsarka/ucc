/**
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "cl_urom.h"
#include "cl_urom_coll.h"
#include "utils/ucc_malloc.h"

#include "components/tl/ucp/tl_ucp.h"

#if 0
static urom_status_t oob_allgather(void *src, void *dst, size_t size,
                                   void *ag_info, void **request)
{
    ucc_context_oob_coll_t *coll_info = (ucc_context_oob_coll_t *)ag_info;
    urom_status_t urom_status;
    ucc_status_t ucc_status;
    void *req;

    ucc_status = coll_info->allgather(src, dst, size, coll_info->coll_info, &req);
    if (UCC_OK != ucc_status) {
        return UROM_ERR_NO_MESSAGE;
    }

    *request = req;
    return UROM_OK;
}

static urom_status_t req_test(void *request)
{
    ucc_status_t ucc_status;

    ucc_status = req_test(
#endif
UCC_CLASS_INIT_FUNC(ucc_cl_urom_context_t,
                    const ucc_base_context_params_t *params,
                    const ucc_base_config_t *config)
{
    const ucc_cl_context_config_t *cl_config =
        ucc_derived_of(config, ucc_cl_context_config_t);
    ucc_config_names_array_t *tls = &cl_config->cl_lib->tls.array;
    ucc_cl_urom_lib_t *urom_lib = ucc_derived_of(cl_config->cl_lib, ucc_cl_urom_lib_t);
    ucc_status_t status;
    urom_status_t urom_status;
    int          i;
    urom_mem_map_t *domain_mem_map;
    urom_domain_params_t urom_domain_params;
    urom_worker_notify_t *notif;
    ucc_mem_map_params_t ucc_mem_params = params->params.mem_params; 

    ucc_lib_params_t lib_params = {
        .mask = UCC_LIB_PARAM_FIELD_THREAD_MODE,
        .thread_mode = UCC_THREAD_SINGLE,
    };
    urom_worker_cmd_t init_cmd = {
        .cmd_type = UROM_WORKER_CMD_UCC,
        .ucc.cmd_type = UROM_WORKER_CMD_UCC_LIB_CREATE,
        .ucc.lib_create_cmd.params = &lib_params,
    };
    urom_worker_params_t worker_params;
#if 0
    urom_worker_cmd_t pass_dc_cmd = {
        .cmd_type = UROM_WORKER_CMD_UCC,
        .ucc.cmd_type = UROM_WORKER_CMD_CREATE_PASSIVE_DATA_CHANNEL,
    };
#endif

    /* TODO: i need a passive dc */
    urom_worker_cmd_t ctx_cmd = {
        .cmd_type = UROM_WORKER_CMD_UCC,
        .ucc.dpu_worker_id = params->params.oob.oob_ep,
        .ucc.cmd_type = UROM_WORKER_CMD_UCC_CONTEXT_CREATE,
        .ucc.context_create_cmd = 
        {
            .start = 0,
            .stride = 1,
            .size = params->params.oob.n_oob_eps,
        },
    };

    UCC_CLASS_CALL_SUPER_INIT(ucc_cl_context_t, cl_config,
                              params->context);
    if (tls->count == 1 && !strcmp(tls->names[0], "all")) {
        tls = &params->context->all_tls;
    }

    self->super.tl_ctxs = ucc_malloc(sizeof(ucc_tl_context_t*) * tls->count,
                               "cl_urom_tl_ctxs");
    if (!self->super.tl_ctxs) {
        cl_error(cl_config->cl_lib, "failed to allocate %zd bytes for tl_ctxs",
                 sizeof(ucc_tl_context_t**) * tls->count);
        return UCC_ERR_NO_MEMORY;
    }
    self->super.n_tl_ctxs = 0;
    for (i = 0; i < tls->count; i++) {
        status = ucc_tl_context_get(params->context, tls->names[i],
                                   &self->super.tl_ctxs[self->super.n_tl_ctxs]);
        if (UCC_OK != status) {
            cl_debug(cl_config->cl_lib,
                     "TL %s context is not available, skipping", tls->names[i]);
        } else {
            self->super.n_tl_ctxs++;
        }
    }
    if (0 == self->super.n_tl_ctxs) {
        cl_error(cl_config->cl_lib, "no TL contexts are available");
        ucc_free(self->super.tl_ctxs);
        self->super.tl_ctxs = NULL;
        return UCC_ERR_NOT_FOUND;
    }
#if 1
    ucc_tl_ucp_context_t *tl_ctx = ucc_derived_of(self->super.tl_ctxs[1], ucc_tl_ucp_context_t);
    printf("rkeys: %p\n", tl_ctx->rkeys);
#endif
#if 1
    urom_lib->worker_id = params->params.oob.oob_ep;
    /* TODO: rather than UROM_WORKER_TYPE_UCC, create a value of OR'd types */
    printf("urom_lib: %p urom_service: %p, worker addr: %p\n", urom_lib, urom_lib->urom_service, urom_lib->urom_worker_addr);
    urom_status = urom_worker_spawn(
        urom_lib->urom_service, UROM_WORKER_TYPE_UCC, urom_lib->urom_worker_addr,
        &urom_lib->urom_worker_len, &urom_lib->worker_id);
    if (UROM_OK != urom_status) {
        cl_error(&urom_lib->super, "failed to connect to urom worker");
        return UCC_ERR_NO_MESSAGE;
    }

    worker_params.serviceh        = urom_lib->urom_service;
//    worker_params.worker_id       = urom_lib->worker_id;
    worker_params.addr            = urom_lib->urom_worker_addr;
    worker_params.addr_len        = urom_lib->urom_worker_len;
    worker_params.num_cmd_notifyq = 16;

    urom_status = urom_worker_connect(&worker_params, &urom_lib->urom_worker);
    if (UROM_OK != urom_status) {
        cl_error(&urom_lib->super, "failed to perform urom_worker_connect() with error: %s",
                 urom_status_string(urom_status));
        return UCC_ERR_NO_MESSAGE;
    }
    printf("DONE!\n");
#endif

    urom_domain_params.flags = UROM_DOMAIN_WORKER_ADDR;
    urom_domain_params.mask = UROM_DOMAIN_PARAM_FIELD_OOB |
                              UROM_DOMAIN_PARAM_FIELD_WORKER |
                              UROM_DOMAIN_PARAM_FIELD_WORKER_ID; 
    urom_domain_params.oob.allgather = (urom_status_t (*)(void *, void *, size_t, void *, void **))params->params.oob.allgather;
    urom_domain_params.oob.req_test = (urom_status_t (*)(void *))params->params.oob.req_test;
    urom_domain_params.oob.req_free = (urom_status_t (*)(void *))params->params.oob.req_free;
    urom_domain_params.oob.coll_info = params->params.oob.coll_info;
    urom_domain_params.oob.n_oob_indexes = params->params.oob.n_oob_eps;
    urom_domain_params.oob.oob_index = params->params.oob.oob_ep;

    printf("coll_info: %p, size: %d, pe #: %d\n", params->params.oob.coll_info, params->params.oob.n_oob_eps, params->params.oob.oob_ep);

    urom_domain_params.domain_worker_id = params->params.oob.oob_ep;
    urom_domain_params.workers = &urom_lib->urom_worker;
    urom_domain_params.num_workers = 1,
    urom_domain_params.domain_size = params->params.oob.n_oob_eps;

    if (params->context->params.mask & UCC_CONTEXT_PARAM_FIELD_OOB &&
        params->context->params.mask & UCC_CONTEXT_PARAM_FIELD_MEM_PARAMS) {
    //    ucc_mem_map_params_t ucc_mem_params = params->params.mem_params; 
        domain_mem_map = ucc_calloc(ucc_mem_params.n_segments, sizeof(urom_mem_map_t),
                                    "urom_domain_mem_map");
        if (!domain_mem_map) {
            cl_error(&urom_lib->super.super, "Failed to allocate urom_mem_map");
            return UCC_ERR_NO_MEMORY;
        }

        for (i = 0; i < ucc_mem_params.n_segments; i++) {
            /* add in memh if added to UCC */
            domain_mem_map[i].mask    = UROM_WORKER_MEM_MAP_FIELD_BASE_VA | UROM_WORKER_MEM_MAP_FIELD_MKEY;
            domain_mem_map[i].base_va = (uint64_t)ucc_mem_params.segments[i].address;
            domain_mem_map[i].len  = ucc_mem_params.segments[i].len;
//            domain_mem_map[i].mask |= UROM_WORKER_MEM_MAP_FIELD_MKEY;
            domain_mem_map[i].mkey = tl_ctx->remote_info[i].packed_key;
            domain_mem_map[i].mkey_len = tl_ctx->remote_info[i].packed_key_len;
            printf("ADDING %lx/%p to domain mem map with key %p and len %lu\n", domain_mem_map[i].base_va, tl_ctx->remote_info[i].va_base, domain_mem_map[i].mkey, domain_mem_map[i].mkey_len);
        }

        urom_domain_params.mask |= UROM_DOMAIN_PARAM_FIELD_MEM_MAP;
        urom_domain_params.mem_map.segments = domain_mem_map;
        urom_domain_params.mem_map.n_segments = ucc_mem_params.n_segments;
    }

#if 0
    pass_dc_cmd.ucc.pass_dc_create_cmd.ucp_addr = tl_ctx->worker.worker_address;
    pass_dc_cmd.ucc.pass_dc_create_cmd.addr_len = tl_ctx->worker.ucp_addrlen;

    printf("WORKER ADDRESS: %p, WORKER LEN: %lu\n",tl_ctx->worker.worker_address, tl_ctx->worker.ucp_addrlen); 
#endif
#if 0
        urom_domain_params.flags = UROM_DOMAIN_WORKER_ADDR;
        urom_domain_params.mask = UROM_DOMAIN_PARAM_FIELD_OOB |
                                  UROM_DOMAIN_PARAM_FIELD_WORKER |
                                  UROM_DOMAIN_PARAM_FIELD_WORKER_ID |
                                  UROM_DOMAIN_PARAM_FIELD_MEM_MAP;
        urom_domain_params.oob.allgather = params->params.oob.oob_allgather;
        urom_domain_params.oob.req_test = params->params.oob.req_test;
        urom_domain_params.oob.req_free = params->params.oob.req_free;
        urom_domain_params.oob.coll_info = params->params.oob.coll_info;
        urom_domain_params.oob.n_oob_indexes = params->params.oob.n_oob_eps;
        urom_domain_params.oob.oob_index = params->params.oob.oob_ep;

        urom_domain_params.domain_worker_id = params->params.oob.oob_ep;
        urom_domain_params.workers = &urom_lib->urom_worker;
        urom_domain_params.num_workers = 1,
        urom_domain_params.domain_size = params->params.oob.n_oob_eps;
        urom_domain_params.mem_map.segments = domain_mem_map;
        urom_domain_params.mem_map.n_segments = ucc_mem_params->n_segments;
#endif
    urom_status = urom_domain_create_post(&urom_domain_params, &self->urom_domain);
    if (urom_status < UROM_OK) {
        cl_error(&urom_lib->super.super, "failed to post urom domain: %s", urom_status_string(status));
        return UCC_ERR_NO_MESSAGE;
    }

    while (UROM_INPROGRESS == (urom_status = urom_domain_create_test(self->urom_domain)));
    if (urom_status < UROM_OK) {
        cl_error(&urom_lib->super.super, "failed to create urom domain: %s", urom_status_string(status));
        return UCC_ERR_NO_MESSAGE;
    }

    urom_worker_push_cmdq(urom_lib->urom_worker, 0, &init_cmd);
    while (UROM_ERR_QUEUE_EMPTY ==
           (urom_status = urom_worker_pop_notifyq(urom_lib->urom_worker, 0, &notif))) {
        sched_yield();
    }
    if ((ucc_status_t) notif->ucc.status != UCC_OK) {
        printf("debug: lib create notif->status: %d\n", notif->ucc.status);
        return notif->ucc.status;
    }
#if 0
        urom_worker_push_cmdq(urom_lib->urom_worker, 0, &pass_dc_cmd);
        while (UROM_ERR_QUEUE_EMPTY ==
               (urom_status = urom_worker_pop_notifyq(urom_lib->urom_worker, 0, &notif))) {
            sched_yield();
        }
        if ((ucc_status_t) notif->ucc.status != UCC_OK) {
            printf("debug: pass dc create notif->status: %d\n", notif->ucc.status);
            return notif->ucc.status;
        }
#endif
    ctx_cmd.ucc.context_create_cmd.base_va = ucc_mem_params.segments[0].address;
    ctx_cmd.ucc.context_create_cmd.len = ucc_mem_params.segments[0].len;
    urom_worker_push_cmdq(urom_lib->urom_worker, 0, &ctx_cmd);
    while (UROM_ERR_QUEUE_EMPTY ==
           (urom_status = urom_worker_pop_notifyq(urom_lib->urom_worker, 0, &notif))) {
        sched_yield();
    }
    if ((ucc_status_t) notif->ucc.status != UCC_OK) {
        printf("debug: ctx create notif->status: %d, ucc_context: %p\n",
           notif->ucc.status, notif->ucc.context_create_nqe.context);
        return notif->ucc.status;
    }

    status = ucc_mpool_init(&self->sched_mp, 0, sizeof(ucc_cl_urom_schedule_t),
                            0, UCC_CACHE_LINE_SIZE, 2, UINT_MAX,
                            &ucc_coll_task_mpool_ops, params->thread_mode,
                            "cl_urom_sched_mp");
    if (UCC_OK != status) {
        cl_error(cl_config->cl_lib, "failed to initialize cl_urom_sched mpool");
        return UCC_ERR_NO_MESSAGE;
    }



    cl_debug(cl_config->cl_lib, "initialized cl context: %p", self);
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_cl_urom_context_t)
{
    int i;
    cl_debug(self->super.super.lib, "finalizing cl context: %p", self);
    for (i = 0; i < self->super.n_tl_ctxs; i++) {
        ucc_tl_context_put(self->super.tl_ctxs[i]);
    }
    ucc_free(self->super.tl_ctxs);
}

UCC_CLASS_DEFINE(ucc_cl_urom_context_t, ucc_cl_context_t);

ucc_status_t
ucc_cl_urom_get_context_attr(const ucc_base_context_t *context,
                              ucc_base_ctx_attr_t      *attr)
{
    if (attr->attr.mask & UCC_CONTEXT_ATTR_FIELD_CTX_ADDR_LEN) {
        attr->attr.ctx_addr_len = 0;
    }

    return UCC_OK;
}
