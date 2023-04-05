/**
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "cl_urom.h"
#include "utils/ucc_malloc.h"

UCC_CLASS_INIT_FUNC(ucc_cl_urom_context_t,
                    const ucc_base_context_params_t *params,
                    const ucc_base_config_t *config)
{
    const ucc_cl_context_config_t *cl_config =
        ucc_derived_of(config, ucc_cl_context_config_t);
    ucc_config_names_array_t *tls = &cl_config->cl_lib->tls.array;
    ucc_cl_urom_lib_t *urom_lib = ucc_derived_of(self->super.super.lib, ucc_cl_urom_lib_t);
    ucc_status_t status;
    int          i;
    urom_mem_map_t *domain_mem_map;
    urom_domain_params_t urom_domain_params;
    ucc_lib_params_t lib_params = {
        .mask = UCC_LIB_PARAM_FIELD_THREAD_MODE,
        .thread_mode = UCC_THREAD_SINGLE,
    };
    urom_worker_cmd_t init_cmd = {
        .cmd_type = UROM_WORKER_CMD_UCC,
        .ucc.cmd_type = UROM_WORKER_CMD_UCC_LIB_CREATE,
        .ucc.lib_create_cmd.params = &lib_params,
    };

    /* TODO: i need a passive dc */
    urom_worker_cmd_t ctx_cmd = {
        .start = 0,
        .stride = 1,
        .size = params->params.oob.n_oob_eps,
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

    urom_domain_params.flags = UROM_DOMAIN_WORKER_ADDR;
    urom_domain_params.mask = UROM_DOMAIN_PARAM_FIELD_OOB |
                              UROM_DOMAIN_PARAM_FIELD_WORKER |
                              UROM_DOMAIN_PARAM_FIELD_WORKER_ID; 
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

    if (params->context->params.mask & UCC_CONTEXT_PARAM_FIELD_OOB &&
        params->context->params.mask & UCC_CONTEXT_PARAM_MEM_PARAMS) {
        ucc_mem_map_params_t *ucc_mem_params = params->params.mem_params; 
        domain_mem_map = ucc_calloc(sizeof(urom_mem_map_t) * ucc_mem_params->n_segments, 
                                    "urom_domain_mem_map");
        if (!domain_mem_map) {
            cl_error("Failed to allocate urom_mem_map");
            return UCC_ERR_OUT_OF_MEMORY;
        }

        for (i = 0; i < ucc_mem_params->n_segments; i++) {
            /* add in memh if added to UCC */
            domain_mem_map[i].mask    = 0;
            domain_mem_map[i].va_base = ucc_mem_params->segments[i].address;
            domain_mem_map[i].va_len  = ucc_mem_params->segments[i].len;
        }

        urom_domain_params.mask |= UROM_DOMAIN_PARAM_FIELD_MEM_MAP;
        urom_domain_params.mem_map.segments = domain_mem_map;
        urom_domain_params.mem_map.n_segments = ucc_mem_params->n_segments;
    }
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
        cl_error("failed to post urom domain: %s", urom_status_string(status));
        return UCC_ERR_NO_MESSAGE;
    }

    while (UROM_INPROGRESS == (urom_status = urom_domain_create_test(self->urom_domain)));
    if (urom_status < UROM_OK) {
        cl_error("failed to create urom domain: %s", urom_status_string(status));
        return UCC_ERR_NO_MESSAGE;
    }

    urom_worker_push_cmdq(worker, 0, &init_cmd);
    while (UROM_ERR_QUEUE_EMPTY ==
           (status = urom_worker_pop_notifyq(worker, 0, &notif))) {
        sched_yield();
    }
    if (notif->ucc.status != UCC_OK) {
        printf("debug: lib create notif->status: %d\n", notif->ucc.status);
        return notif->ucc.status;
    }
/*
        urom_worker_push_cmdq(worker, 0, &pass_dc_cmd);
        while (UROM_ERR_QUEUE_EMPTY ==
               (status = urom_worker_pop_notifyq(worker, 0, &notif))) {
            sched_yield();
        }
        if (notif->ucc.status != UCC_OK) {
            printf("debug: pass dc create notif->status: %d\n", notif->ucc.status);
            return notif->ucc.status;
        }
*/
    urom_worker_push_cmdq(worker, 0, &ctx_cmd);
    while (UROM_ERR_QUEUE_EMPTY ==
           (status = urom_worker_pop_notifyq(worker, 0, &notif))) {
        sched_yield();
    }
    if (notif->ucc.status != UCC_OK) {
        printf("debug: ctx create notif->status: %d, ucc_context: %p\n",
           notif->ucc.status, notif->ucc.context_create_nqe.context);
        return notif->ucc.status;
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
    const ucc_cl_urom_context_t *ctx =
        ucc_derived_of(context, ucc_cl_urom_context_t);
    ucc_base_ctx_attr_t tl_attr;
    ucc_status_t        status;
    int                 i;

    if (attr->attr.mask & UCC_CONTEXT_ATTR_FIELD_CTX_ADDR_LEN) {
        attr->attr.ctx_addr_len = 0;
    }

    return UCC_OK;
}
