/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "cl_urom.h"
#include "utils/ucc_coll_utils.h"

#include "components/tl/ucp/tl_ucp.h"

#include <urom/api/urom.h>
#include <urom/api/urom_ucc.h>

ucc_status_t ucc_cl_urom_alltoall_full_init(
                         ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
                         ucc_coll_task_t **task);

#if 0
static ucc_status_t ucc_cl_urom_alltoallv_start(ucc_coll_task_t *task)
{
    return ucc_schedule_start(task);
}

static ucc_status_t ucc_cl_urom_coll_finalize(ucc_coll_task_t *task)
{
    ucc_cl_urom_schedule_t *schedule =
        ucc_derived_of(task, ucc_cl_urom_schedule_t);
    ucc_status_t status;

    if (schedule->scratch) {
        ucc_mc_free(schedule->scratch);
    }
    status = ucc_schedule_finalize(task);
    ucc_cl_urom_put_schedule(&schedule->super.super);
    return status;
}
#endif

ucc_status_t ucc_cl_urom_coll_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t      *team,
                                   ucc_coll_task_t     **task)
{
    ucc_cl_urom_team_t *cl_team = ucc_derived_of(team, ucc_cl_urom_team_t);
    ucc_cl_urom_context_t *ctx  = UCC_CL_UROM_TEAM_CTX(cl_team);
    ucc_cl_urom_lib_t *urom_lib = ucc_derived_of(ctx->super.super.lib, ucc_cl_urom_lib_t);
    ucc_tl_ucp_context_t *tl_ctx = ucc_derived_of(ctx->super.tl_ctxs[1], ucc_tl_ucp_context_t);
    urom_status_t urom_status;
//    printf("rkeys: %p\n", tl_ctx->rkeys);

    if (!urom_lib->pass_dc_exist) {
        urom_worker_cmd_t pass_dc_cmd = {
            .cmd_type = UROM_WORKER_CMD_UCC,
            .ucc.cmd_type = UROM_WORKER_CMD_CREATE_PASSIVE_DATA_CHANNEL,
            .ucc.pass_dc_create_cmd.ucp_addr = tl_ctx->worker.worker_address,
            .ucc.pass_dc_create_cmd.addr_len = tl_ctx->worker.ucp_addrlen,
        };
        urom_worker_notify_t *notif;

        urom_worker_push_cmdq(urom_lib->urom_worker, 0, &pass_dc_cmd);
        while (UROM_ERR_QUEUE_EMPTY ==
               (urom_status = urom_worker_pop_notifyq(urom_lib->urom_worker, 0, &notif))) {
            sched_yield();
        }
        if ((ucc_status_t) notif->ucc.status != UCC_OK) {
            printf("debug: pass dc create notif->status: %d\n", notif->ucc.status);
            return notif->ucc.status;
        }
        urom_lib->pass_dc_exist = 1;
    }

    switch (coll_args->args.coll_type) {
        case UCC_COLL_TYPE_ALLTOALL:
            return ucc_cl_urom_alltoall_full_init(coll_args, team, task);
        default:
            cl_error(urom_lib, "coll_type %s is not supported", ucc_coll_type_str(coll_args->args.coll_type));
            break;
    }
    return UCC_ERR_NOT_SUPPORTED;
}
#if 0

    ucc_status_t        ucc_status = UCC_OK;
    urom_status_t       urom_status;
    urom_worker_cmd_t   coll_cmd = {
        .cmd_type = UROM_WORKER_CMD_UCC,
        .ucc.dpu_worker_id = UCC_CL_TEAM_RANK(cl_team),
        .ucc.cmd_type      = UROM_WORKER_CMD_UCC_COLL,
        .ucc.coll_cmd.coll_args = coll_args,
        .ucc.coll_cmd.team = team,
//        .ucc.coll_cmd.use_xgvmi = 0,
    };

    printf("CALLED!\n");
    /* F: I dont think this is right */
    urom_status = urom_worker_push_cmdq(urom_lib->urom_worker, 0, &coll_cmd);
    if (UROM_OK != urom_status) {
        printf("RETURNING ERROR!\n");
        return UCC_ERR_NO_MESSAGE;
//        return urom_to_ucc_status(status);
    }

#if 0    
    status = ucc_coll_init(cl_team->score_map, coll_args, task);
    if (UCC_ERR_NOT_FOUND == status) {
        cl_warn(UCC_CL_TEAM_LIB(cl_team),
                "no TL supporting given coll args is available");
        return UCC_ERR_NOT_SUPPORTED;
    }
#endif
    printf("returnning %d\n", ucc_status);
    return ucc_status;
}
#endif
