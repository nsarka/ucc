/**
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "allgatherv.h"
#include "unpack.h"
#include "../cl_hier_coll.h"
#include "core/ucc_team.h"

#define MAX_ALLGATHERV_TASKS 4

ucc_base_coll_alg_info_t
    ucc_cl_hier_allgatherv_algs[UCC_CL_HIER_ALLGATHERV_ALG_LAST + 1] = {
        [UCC_CL_HIER_ALLGATHERV_ALG_GAB] =
            {.id   = UCC_CL_HIER_ALLGATHERV_ALG_GAB,
             .name = "gab",
             .desc = "gatherv + allgatherv + bcast"},
        [UCC_CL_HIER_ALLGATHERV_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

static ucc_status_t ucc_cl_hier_allgatherv_start(ucc_coll_task_t *task)
{
    UCC_CL_HIER_PROFILE_REQUEST_EVENT(task, "cl_hier_allgatherv_start", 0);
    return ucc_schedule_start(task);
}

static ucc_status_t ucc_cl_hier_allgatherv_finalize(ucc_coll_task_t *task)
{
    ucc_schedule_t *schedule = ucc_derived_of(task, ucc_schedule_t);
    ucc_cl_hier_schedule_t *cl_schedule =
        ucc_derived_of(task, ucc_cl_hier_schedule_t);
    ucc_status_t    status;

    ucc_mc_free(cl_schedule->scratch);

    UCC_CL_HIER_PROFILE_REQUEST_EVENT(task, "cl_hier_allgatherv_finalize",
                                      0);
    status = ucc_schedule_finalize(task);
    ucc_cl_hier_put_schedule(schedule);
    return status;
}

static inline int is_leader(ucc_base_team_t *team, ucc_rank_t rank)
{
    ucc_cl_hier_team_t  *cl_team = ucc_derived_of(team, ucc_cl_hier_team_t);
    ucc_rank_t           leader_sbgp_size = SBGP_SIZE(cl_team, NODE_LEADERS);
    ucc_rank_t i;
    for (i = 0; i < leader_sbgp_size; i++) {
        if (ucc_ep_map_eval(SBGP_MAP(cl_team, NODE_LEADERS), i) == rank) {
            return 1;
        }
    }
    return 0;
}

/* TODO: is there a better way to do this? */
static inline ucc_rank_t find_leader_rank(ucc_base_team_t *team, ucc_rank_t team_rank)
{
    ucc_cl_hier_team_t  *cl_team = ucc_derived_of(team, ucc_cl_hier_team_t);
    ucc_team_t         *core_team = team->params.team;
    ucc_rank_t i;

    for (i = 0; i < UCC_CL_TEAM_SIZE(cl_team); i++) {
        if (ucc_team_ranks_on_same_node(i, team_rank, core_team) && is_leader(team, i)) {
            return i;
        }
    }

    return UCC_RANK_INVALID;
}

UCC_CL_HIER_PROFILE_FUNC(ucc_status_t, ucc_cl_hier_allgatherv_init,
                         (coll_args, team, task),
                         ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
                         ucc_coll_task_t **task)
{
    ucc_cl_hier_team_t  *cl_team = ucc_derived_of(team, ucc_cl_hier_team_t);
    ucc_coll_task_t     *tasks[MAX_ALLGATHERV_TASKS] = {NULL};
    ucc_rank_t           rank = UCC_CL_TEAM_RANK(cl_team);
    ucc_rank_t           node_sbgp_size = SBGP_SIZE(cl_team, NODE);
    ucc_rank_t           leader_sbgp_size = SBGP_SIZE(cl_team, NODE_LEADERS);
    ucc_rank_t           team_size = UCC_CL_TEAM_SIZE(cl_team);
    ucc_aint_t *node_disps = NULL;
    ucc_count_t *node_counts = NULL;
    ucc_aint_t *leader_disps = NULL;
    ucc_count_t *leader_counts = NULL;
    size_t dt_size = ucc_dt_size(coll_args->args.dst.info_v.datatype);
    int in_place = 0;
    int is_contig = 1;
    ucc_schedule_t      *schedule;
    ucc_cl_hier_schedule_t *cl_schedule;
    ucc_status_t         status;
    ucc_base_coll_args_t args, args_old;
    int                  n_tasks, i;
    size_t scratch_size;
    size_t node_counts_size;
    size_t node_disps_size;
    size_t leader_counts_size;
    size_t leader_disps_size;
    size_t total_count;
    void *node_gathered_data;

    schedule = &ucc_cl_hier_get_schedule(cl_team)->super.super;
    if (ucc_unlikely(!schedule)) {
        return UCC_ERR_NO_MEMORY;
    }
    cl_schedule = ucc_derived_of(schedule, ucc_cl_hier_schedule_t);

    memcpy(&args, coll_args, sizeof(args));
    memcpy(&args_old, coll_args, sizeof(args));
    in_place = UCC_IS_INPLACE(args.args);
    is_contig = UCC_COLL_IS_DST_CONTIG(&args.args);
    n_tasks        = 0;
    UCC_CHECK_GOTO(ucc_schedule_init(schedule, &args, team), out, status);

    node_counts_size = node_sbgp_size * sizeof(ucc_count_t);
    node_disps_size = node_sbgp_size * sizeof(ucc_aint_t);
    leader_counts_size = leader_sbgp_size * sizeof(ucc_count_t);
    leader_disps_size = leader_sbgp_size * sizeof(ucc_aint_t);
    total_count = ucc_coll_args_get_total_count(&args.args, args.args.dst.info_v.counts, team_size);
    scratch_size = node_counts_size + node_disps_size + leader_counts_size + leader_disps_size + (total_count * dt_size);
    UCC_CHECK_GOTO(ucc_mc_alloc(&cl_schedule->scratch, scratch_size, UCC_MEMORY_TYPE_HOST), out, status);
    memset(cl_schedule->scratch->addr, 0, scratch_size);
    node_counts = PTR_OFFSET(cl_schedule->scratch->addr, 0);
    node_disps = PTR_OFFSET(node_counts, node_counts_size);
    leader_counts = PTR_OFFSET(node_disps, node_disps_size);
    leader_disps = PTR_OFFSET(leader_counts, leader_counts_size);
    node_gathered_data = PTR_OFFSET(leader_disps, leader_disps_size);

    if (SBGP_ENABLED(cl_team, NODE)) {
        ucc_assert(n_tasks == 0);
        if (cl_team->top_sbgp == UCC_HIER_SBGP_NODE) {
            args.args.coll_type = UCC_COLL_TYPE_ALLGATHERV;
        } else {
            size_t disp_counter = 0;
            for (i = 0; i < node_sbgp_size; i++) {
                ucc_rank_t team_rank =
                    ucc_ep_map_eval(SBGP_MAP(cl_team, NODE), i);
                ucc_coll_args_set_count(&args.args, node_counts, i, ucc_coll_args_get_count(&args.args, args.args.dst.info_v.counts, team_rank));
                ucc_coll_args_set_displacement(&args.args, node_disps, i, disp_counter);
                disp_counter += ucc_coll_args_get_count(&args.args, node_counts, i);
            }

            if (in_place) {
                args.args.src.info.buffer = PTR_OFFSET(args.args.dst.info_v.buffer, dt_size * ucc_coll_args_get_displacement(&args.args, args.args.dst.info_v.displacements, rank));
                args.args.src.info.count = ucc_coll_args_get_count(&args.args, args.args.dst.info_v.counts, rank);
                args.args.src.info.datatype = args.args.dst.info_v.datatype;
                args.args.src.info.mem_type = args.args.dst.info_v.mem_type;
            }

            args.args.coll_type = UCC_COLL_TYPE_GATHERV;
            args.args.root = 0;
            args.args.flags &= ~UCC_COLL_ARGS_FLAG_IN_PLACE;
            args.args.dst.info_v.displacements = node_disps;
            args.args.dst.info_v.counts = node_counts;
            args.args.dst.info_v.buffer = node_gathered_data;
        }
        UCC_CHECK_GOTO(
            ucc_coll_init(SCORE_MAP(cl_team, NODE), &args, &tasks[n_tasks]),
            out, status);
        n_tasks++;
    }

    args = args_old;

    if (SBGP_ENABLED(cl_team, NODE_LEADERS)) {
        ucc_assert(cl_team->top_sbgp == UCC_HIER_SBGP_NODE_LEADERS);
        size_t disp_counter = 0;

        /* Sum up the counts on each node to get the count for each node leader */
        for (i = 0; i < team_size; i++) {
            ucc_rank_t leader_team_rank = find_leader_rank(team, i);
            size_t leader_old_count = ucc_coll_args_get_count(&args.args, leader_counts, ucc_ep_map_local_rank(SBGP_MAP(cl_team, NODE_LEADERS), leader_team_rank));
            size_t add_count = ucc_coll_args_get_count(&args.args, args.args.dst.info_v.counts, i);
            size_t new_count = add_count + leader_old_count;
            ucc_coll_args_set_count(&args.args, leader_counts, ucc_ep_map_local_rank(SBGP_MAP(cl_team, NODE_LEADERS), leader_team_rank), new_count);
        }

        /*
           Need to order leader displacements by their team rank, not their leader sbgp rank.
           The reason is leaders are not always in the same order as they are in the team
           e.g., 2n2ppn
           team ranks = 0 1 2 3 with 0 and 2 as leaders
           leader sbgp ranks can be 2 0 wrt their team ranks
        */
        for (i = 0; i < team_size; i++) {
            if (is_leader(team, i)) {
                ucc_rank_t leader_sgbp_rank = ucc_ep_map_local_rank(SBGP_MAP(cl_team, NODE_LEADERS), i);
                ucc_coll_args_set_displacement(&args.args, leader_disps, leader_sgbp_rank, disp_counter);
                disp_counter += ucc_coll_args_get_count(&args.args, leader_counts, leader_sgbp_rank);
            }
        }
        args.args.coll_type = UCC_COLL_TYPE_ALLGATHERV;
        args.args.flags &= ~UCC_COLL_ARGS_FLAG_IN_PLACE;
        args.args.src.info.buffer = node_gathered_data;
        args.args.src.info.count = ucc_coll_args_get_total_count(&args.args, node_counts, node_sbgp_size);
        args.args.src.info.datatype = args.args.dst.info_v.datatype;
        args.args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;
        args.args.dst.info_v.displacements = leader_disps;
        args.args.dst.info_v.counts = leader_counts;
        args.args.dst.info_v.buffer = args_old.args.dst.info_v.buffer;
        UCC_CHECK_GOTO(ucc_coll_init(SCORE_MAP(cl_team, NODE_LEADERS), &args,
                                     &tasks[n_tasks]),
                       out, status);
        n_tasks++;
    }

    if (SBGP_ENABLED(cl_team, NODE) &&
        cl_team->top_sbgp != UCC_HIER_SBGP_NODE) {
        args = args_old;
        args.args.coll_type = UCC_COLL_TYPE_BCAST;
        args.args.mask |= UCC_COLL_ARGS_FIELD_FLAGS;
        args.args.flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;
        args.args.root = 0;
        args.args.src.info.buffer = args_old.args.dst.info_v.buffer;
        args.args.src.info.count = total_count;
        args.args.src.info.datatype = args_old.args.dst.info_v.datatype;
        args.args.src.info.mem_type = args_old.args.dst.info_v.mem_type;
        UCC_CHECK_GOTO(
            ucc_coll_init(SCORE_MAP(cl_team, NODE), &args, &tasks[n_tasks]),
            out, status);
        n_tasks++;

        if (!is_contig) {
            args = args_old;
            UCC_CHECK_GOTO(ucc_cl_hier_allgatherv_unpack_init(&args, team, &tasks[n_tasks]), out, status);
            n_tasks++;
        }
    }

    UCC_CHECK_GOTO(ucc_event_manager_subscribe(
                       &schedule->super, UCC_EVENT_SCHEDULE_STARTED, tasks[0],
                       ucc_task_start_handler),
                   out, status);
    UCC_CHECK_GOTO(ucc_schedule_add_task(schedule, tasks[0]), out, status);
    for (i = 1; i < n_tasks; i++) {
        UCC_CHECK_GOTO(
            ucc_event_manager_subscribe(tasks[i - 1], UCC_EVENT_COMPLETED,
                                        tasks[i], ucc_task_start_handler),
            out, status);
        UCC_CHECK_GOTO(ucc_schedule_add_task(schedule, tasks[i]), out, status);
    }

    schedule->super.flags |= UCC_COLL_TASK_FLAG_EXECUTOR;
    schedule->super.post     = ucc_cl_hier_allgatherv_start;
    schedule->super.finalize = ucc_cl_hier_allgatherv_finalize;
    *task                    = &schedule->super;
    return UCC_OK;

out:
    for (i = 0; i < n_tasks; i++) {
        tasks[i]->finalize(tasks[i]);
    }
    ucc_cl_hier_put_schedule(schedule);
    return status;
}
