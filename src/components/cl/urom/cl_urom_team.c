/**
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "cl_urom.h"
#include "utils/ucc_malloc.h"
#include "core/ucc_team.h"

UCC_CLASS_INIT_FUNC(ucc_cl_urom_team_t, ucc_base_context_t *cl_context,
                    const ucc_base_team_params_t *params)
{
    ucc_cl_urom_context_t *ctx       =
        ucc_derived_of(cl_context, ucc_cl_urom_context_t);
    ucc_cl_urom_lib_t *urom_lib = ucc_derived_of(ctx->super.super.lib, ucc_cl_urom_lib_t);
//    ucc_cl_urom_lib_t *lib = ctx->super.super.lib;
    urom_status_t   urom_status;
    //unsigned                n_tl_ctxs = ctx->super.n_tl_ctxs;
//    int                     i;
    ucc_status_t            status;
    urom_worker_cmd_t team_cmd = {
        .cmd_type = UROM_WORKER_CMD_UCC,
        .ucc.dpu_worker_id = params->rank,
        .ucc.cmd_type = UROM_WORKER_CMD_UCC_TEAM_CREATE,
        /* FIXME: proper way: use ec map.. for now assume linear */
        .ucc.team_create_cmd = 
        {
            .start = 0,
            .stride = 1,
            .size = params->size,
        },
    };

    UCC_CLASS_CALL_SUPER_INIT(ucc_cl_team_t, &ctx->super, params);
    self->teams = (ucc_team_h **)ucc_malloc(sizeof(ucc_team_h *) * 16);
    if (!self->teams) {
        cl_error(cl_context->lib, "failed to allocate %zd bytes for urom teams", sizeof(ucc_team_h *) * 16);
        status = UCC_ERR_NO_MEMORY;
        return status;
//        goto err;
    }
    self->n_teams = 0;
   
    urom_status = urom_worker_push_cmdq(urom_lib->urom_worker, 0, &team_cmd);
    if (UROM_OK != urom_status) {
        // error
    }
#if 0
    self->tl_teams = ucc_malloc(sizeof(ucc_tl_team_t *) * n_tl_ctxs,
                                "cl_urom_tl_teams");
    if (!self->tl_teams) {
        cl_error(cl_context->lib, "failed to allocate %zd bytes for tl_teams",
                 sizeof(ucc_tl_team_t *) * n_tl_ctxs);
        status = UCC_ERR_NO_MEMORY;
        goto err;
    }
    self->n_tl_teams = 0;
    status           = ucc_team_multiple_req_alloc(&self->team_create_req,
                                                   n_tl_ctxs);
    if (UCC_OK != status) {
        cl_error(cl_context->lib, "failed to allocate team req multiple");
        goto err;
    }
    for (i = 0; i < n_tl_ctxs; i++) {
        memcpy(&self->team_create_req->descs[i].param, params,
               sizeof(ucc_base_team_params_t));
        self->team_create_req->descs[i].ctx            = ctx->super.tl_ctxs[i];
        self->team_create_req->descs[i].param.scope    = UCC_CL_UROM;
        self->team_create_req->descs[i].param.scope_id = 0;
    }
    self->team_create_req->n_teams = n_tl_ctxs;

    status = ucc_tl_team_create_multiple(self->team_create_req);
    if (status < 0) {
        cl_error(cl_context->lib, "failed to post tl team create (%d)",
                 status);
        goto err;
    }
#endif
    cl_debug(cl_context->lib, "posted cl team: %p", self);
    return UCC_OK;
}
/*err:
    ucc_free(self->tl_teams);
    return status;
}
*/

UCC_CLASS_CLEANUP_FUNC(ucc_cl_urom_team_t)
{
    cl_debug(self->super.super.context->lib, "finalizing cl team: %p", self);
}

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_cl_urom_team_t, ucc_base_team_t);
UCC_CLASS_DEFINE(ucc_cl_urom_team_t, ucc_cl_team_t);

ucc_status_t ucc_cl_urom_team_destroy(ucc_base_team_t *cl_team)
{
#if 0
    ucc_cl_urom_team_t    *team    = ucc_derived_of(cl_team, ucc_cl_urom_team_t);
    ucc_cl_urom_context_t *ctx     = UCC_CL_UROM_TEAM_CTX(team);
    ucc_status_t            status  = UCC_OK;
    int                     i;
    if (NULL == team->team_create_req) {
        status = ucc_team_multiple_req_alloc(&team->team_create_req,
                                             team->n_tl_teams);
        if (UCC_OK != status) {
            cl_error(ctx->super.super.lib, "failed to allocate team req multiple");
            return status;
        }
        team->team_create_req->n_teams       = team->n_tl_teams;
        for (i = 0; i < team->n_tl_teams; i++) {
            team->team_create_req->descs[i].team = team->tl_teams[i];
        }
    }
    status = ucc_tl_team_destroy_multiple(team->team_create_req);
    if (UCC_INPROGRESS == status) {
        return status;
    }
    for (i = 0; i < team->n_tl_teams; i++) {
        if (team->team_create_req->descs[i].status != UCC_OK) {
            cl_error(ctx->super.super.lib, "tl team destroy failed (%d)",
                     status);
            status = team->team_create_req->descs[i].status;
        }
    }
    ucc_team_multiple_req_free(team->team_create_req);
    if (team->score_map) {
        ucc_coll_score_free_map(team->score_map);
    }
    ucc_free(team->tl_teams);
    UCC_CLASS_DELETE_FUNC_NAME(ucc_cl_urom_team_t)(cl_team);
#endif
//    return status;
    return UCC_OK;    
}

ucc_status_t ucc_cl_urom_team_create_test(ucc_base_team_t *cl_team)
{
    ucc_cl_urom_team_t    *team = ucc_derived_of(cl_team, ucc_cl_urom_team_t);
    ucc_cl_urom_context_t *ctx  = UCC_CL_UROM_TEAM_CTX(team);
    ucc_cl_urom_lib_t *urom_lib = ucc_derived_of(ctx->super.super.lib, ucc_cl_urom_lib_t);
//    int                     i;
//    ucc_coll_score_t       *score, *score_next, *score_merge;
    urom_status_t           urom_status;
    urom_worker_notify_t   *notif;

    urom_status = urom_worker_pop_notifyq(urom_lib->urom_worker, 0, &notif);
    if (UROM_ERR_QUEUE_EMPTY != urom_status) {
        if (urom_status == UROM_OK) {
            if (notif->ucc.status == (urom_status_t)UCC_OK) {
                team->teams[team->n_teams] = notif->ucc.team_create_nqe.team;
                ++team->n_teams;
                return UCC_OK;
            }
        }
        return UCC_ERR_NO_MESSAGE;
    }
    return UCC_INPROGRESS;
}

#if 0
    status = ucc_tl_team_create_multiple(team->team_create_req);
    if (status == UCC_OK) {
        for (i = 0; i < ctx->super.n_tl_ctxs; i++) {
            if (team->team_create_req->descs[i].status == UCC_OK) {
                team->tl_teams[team->n_tl_teams++] =
                    team->team_create_req->descs[i].team;
                cl_debug(ctx->super.super.lib, "initialized tl %s team",
                         UCC_TL_CTX_IFACE(team->team_create_req->descs[i].ctx)->
                         super.name);
            } else {
                cl_debug(ctx->super.super.lib, "failed to create tl %s team: (%d)",
                         UCC_TL_CTX_IFACE(team->team_create_req->descs[i].ctx)->
                         super.name, team->team_create_req->descs[i].status);
            }
        }
        ucc_team_multiple_req_free(team->team_create_req);
        team->team_create_req = NULL;
        if (0 == team->n_tl_teams) {
            cl_error(ctx->super.super.lib, "no tl teams were created");
            return UCC_ERR_NOT_FOUND;
        }
        status =
            UCC_TL_TEAM_IFACE(team->tl_teams[0])
                ->team.get_scores(&team->tl_teams[0]->super, &score);
        if (UCC_OK != status) {
            cl_error(ctx->super.super.lib, "failed to get tl %s scores",
                     UCC_TL_TEAM_IFACE(team->tl_teams[0])->super.name);
            return status;
        }
        for (i = 1; i < team->n_tl_teams; i++) {
            status =
                UCC_TL_TEAM_IFACE(team->tl_teams[i])
                ->team.get_scores(&team->tl_teams[i]->super, &score_next);
            if (UCC_OK != status) {
                cl_error(ctx->super.super.lib, "failed to get tl %s scores",
                         UCC_TL_TEAM_IFACE(team->tl_teams[i])->super.name);
                return status;
            }
            status =
                ucc_coll_score_merge(score, score_next, &score_merge, 1);
            if (UCC_OK != status) {
                cl_error(ctx->super.super.lib, "failed to merge scores");
                return status;
            }
            score = score_merge;
        }
        status = ucc_coll_score_build_map(score, &team->score_map);
        if (UCC_OK != status) {
            cl_error(ctx->super.super.lib, "failed to build score map");
        }
        team->score = score;
        ucc_coll_score_set(team->score, UCC_CL_UROM_DEFAULT_SCORE);
    }
    return status;
}
#endif

ucc_status_t ucc_cl_urom_team_get_scores(ucc_base_team_t   *cl_team,
                                          ucc_coll_score_t **score)
{
/*oob_
    ucc_cl_urom_team_t *team = ucc_derived_of(cl_team, ucc_cl_urom_team_t);
    ucc_base_context_t  *ctx  = UCC_CL_TEAM_CTX(team);
    ucc_status_t         status;
*/
#if 0
    status = ucc_coll_score_dup(team->score, score);
    if (UCC_OK != status) {
        return status;
    }

    if (strlen(ctx->score_str) > 0) {
        status = ucc_coll_score_update_from_str(
            ctx->score_str, *score, UCC_CL_TEAM_SIZE(team), NULL, cl_team,
            UCC_CL_UROM_DEFAULT_SCORE, NULL, NULL, 0);

        /* If INVALID_PARAM - User provided incorrect input - try to proceed */
        if ((status < 0) && (status != UCC_ERR_INVALID_PARAM) &&
            (status != UCC_ERR_NOT_SUPPORTED)) {
            goto err;
        }
    }
#endif
    return UCC_OK;
}
//err:
//    ucc_coll_score_free(*score);
//    *score = NULL;
//    return status;
//}
