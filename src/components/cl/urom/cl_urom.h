/**
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_CL_UROM_H_
#define UCC_CL_UROM_H_
#include "components/cl/ucc_cl.h"
#include "components/cl/ucc_cl_log.h"
#include "components/tl/ucc_tl.h"
#include "coll_score/ucc_coll_score.h"
#include "utils/ucc_mpool.h"


#include <urom/api/urom.h>
#include <urom/api/urom_ucc.h>

#ifndef UCC_CL_UROM_DEFAULT_SCORE
#define UCC_CL_UROM_DEFAULT_SCORE 20
#endif

typedef struct ucc_cl_urom_iface {
    ucc_cl_iface_t super;
} ucc_cl_urom_iface_t;
/* Extern iface should follow the pattern: ucc_cl_<cl_name> */
extern ucc_cl_urom_iface_t ucc_cl_urom;

typedef struct ucc_cl_urom_lib_config {
    ucc_cl_lib_config_t super;
} ucc_cl_urom_lib_config_t;

typedef struct ucc_cl_urom_context_config {
    ucc_cl_context_config_t super;
} ucc_cl_urom_context_config_t;

typedef struct ucc_cl_urom_lib {
    ucc_cl_lib_t             super;
    urom_service_h           urom_service;
    urom_worker_h            urom_worker;
    void                    *urom_worker_addr;
    size_t                   urom_worker_len;
    uint64_t                 worker_id;
    int                      pass_dc_exist;
} ucc_cl_urom_lib_t;
UCC_CLASS_DECLARE(ucc_cl_urom_lib_t, const ucc_base_lib_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_cl_urom_context {
    ucc_cl_context_t   super;
    urom_domain_h      urom_domain;
    ucc_mpool_t        sched_mp;
} ucc_cl_urom_context_t;
UCC_CLASS_DECLARE(ucc_cl_urom_context_t, const ucc_base_context_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_cl_urom_team {
    ucc_cl_team_t            super;
    int                      team_posted;
    ucc_team_h             **teams;
    unsigned                 n_teams;
    ucc_coll_score_t        *score;
    ucc_score_map_t         *score_map;
} ucc_cl_urom_team_t;
UCC_CLASS_DECLARE(ucc_cl_urom_team_t, ucc_base_context_t *,
                  const ucc_base_team_params_t *);

ucc_status_t ucc_cl_urom_coll_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t      *team,
                                   ucc_coll_task_t     **task);

#define UCC_CL_UROM_TEAM_CTX(_team)                                           \
    (ucc_derived_of((_team)->super.super.context, ucc_cl_urom_context_t))

#endif
