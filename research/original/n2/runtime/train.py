"""Prepare a frozen run, or execute one explicitly scheduled cloud training arm."""
import argparse
import copy
import json
from pathlib import Path

import n2_runtime as rt


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--order", required=True, help="Existing common frozen order JSON, under runtime/")
    parser.add_argument("--run-dir", required=True, help="New run directory under runtime/")
    parser.add_argument("--branch", choices=("R", "C"), required=True)
    parser.add_argument("--execute", action="store_true", help="Explicit opt-in after GPU scheduling")
    parser.add_argument("--resume", help="Resume same frozen queue, optimizer, scheduler, and RNG")
    parser.add_argument("--continue-from", help="Preserve optimizer moments but begin explicit new queue/horizon")
    parser.add_argument("--mix-config", help="Required explicit CE/mix contract for continuation")
    parser.add_argument("--replay-ids", help="Explicit JSON list of previously exposed IDs allowed in new queue")
    parser.add_argument("--scheduler-horizon", type=int, help="Actual updates in this queue; required for continuation")
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--token-chunk-size", type=int, default=32)
    parser.add_argument("--without-activation-checkpointing", action="store_true")
    parser.add_argument("--stage-token-budget", type=int, default=None)
    return parser.parse_args()


def read_runtime_json(path):
    with rt.cloud_output(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def main():
    args = arguments()
    if args.resume and args.continue_from:
        raise ValueError("Choose same-queue resume or continuation")
    if args.continue_from and (not args.mix_config or args.scheduler_horizon is None):
        raise ValueError("Continuation requires an explicit mix and new scheduler horizon")
    if args.mix_config and not args.continue_from and not args.resume:
        raise ValueError("The initial small-stage R/C recipe has fixed CE/mix")
    run_dir = rt.cloud_output(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    mix = rt.validate_mix(read_runtime_json(args.mix_config)) if args.mix_config else copy.deepcopy(rt.DEFAULT_MIX)
    rows = rt.load_rows(args.rows, args.data_sha256)
    tokenizer = rt.load_tokenizer()
    examples = [rt.encode(row, tokenizer) for row in rows]
    order_path = rt.cloud_output(args.order)
    if order_path.exists():
        order_value = read_runtime_json(order_path)
        if isinstance(order_value, dict):
            if order_value["data_sha256"] != args.data_sha256 or order_value["mix"] != mix:
                raise ValueError("Common order belongs to different data or objective")
            order = order_value["groups"]
        else:
            raise ValueError("Order must include frozen data hash and explicit mix")
    else:
        if args.resume:
            raise FileNotFoundError("A resume must retain its common frozen order")
        order = rt.build_order(examples, mix)
        rt.write_json(order_path, {"data_sha256": args.data_sha256, "mix": mix,
                                   "seed": rt.SEED, "groups": order,
                                   "order_sha256": rt.object_sha(order)})
    by_id = rt.validate_order(examples, order, mix)
    summary = rt.corpus_summary(examples, order, mix)
    horizon = args.scheduler_horizon if args.scheduler_horizon is not None else len(order)
    if horizon != len(order):
        raise ValueError("Scheduler horizon must equal the actual queue update count")
    continuation = bool(args.continue_from)
    if not args.resume and not continuation:
        if mix != rt.DEFAULT_MIX or args.stage_token_budget not in (None, 300000):
            raise ValueError("Initial small-stage contract cannot be silently changed")
        if len(rows) > 1600 or sum(x["count"] for x in examples) > 300000:
            raise ValueError("Initial stage exceeds the 1600-row / 300k-token arm budget")
    if continuation:
        if args.stage_token_budget is None:
            raise ValueError("Continuation needs a separately measured explicit token budget")
        if sum(x["count"] for x in examples) > args.stage_token_budget:
            raise ValueError("Additional queue exceeds explicit continuation token budget")
    prepared = {"utc": rt.utc(), "status": "PREPARED_NOT_EXECUTED",
                "canon_commit": rt.CANON_COMMIT, "branch": args.branch,
                "data_path": str(rt.training_input(args.rows)), "data_sha256": args.data_sha256,
                "order_path": str(order_path), "order_sha256": rt.object_sha(order),
                "summary": summary, "scheduler_horizon": horizon, "warmup": args.warmup,
                "mix": mix, "token_chunk_size": args.token_chunk_size,
                "activation_checkpointing": not args.without_activation_checkpointing,
                "stage_token_budget": args.stage_token_budget if continuation else 300000,
                "continuation_parent": args.continue_from, "resume": args.resume,
                "scientific_outcome": "NOT_ASSESSED"}
    prepared_path = run_dir / "prepared.json"
    if prepared_path.exists():
        previous = read_runtime_json(prepared_path)
        for key in ("data_sha256", "order_sha256", "branch", "mix", "scheduler_horizon"):
            if previous[key] != prepared[key]:
                raise ValueError(("Run directory already prepared for a different run", key))
    else:
        rt.write_json(prepared_path, prepared)
    if not args.execute:
        print(json.dumps(prepared, indent=2))
        return

    gpu = rt.assert_cloud_cuda()
    rt.seed_all()
    parent_hashes_before = rt.verify_parent_files()
    activation = not args.without_activation_checkpointing
    if args.resume or args.continue_from:
        student, tokenizer, optimizer, scheduler, state = rt.restore_training(
            args.resume or args.continue_from, activation_checkpointing=activation)
        if state["branch"] != args.branch:
            raise ValueError("Branch cannot change on resume/continuation")
        if args.resume:
            for key, value in (("data_sha256", args.data_sha256),
                               ("order_sha256", rt.object_sha(order)), ("mix", mix)):
                if state[key] != value:
                    raise ValueError(("Resume frozen contract mismatch", key))
            if state["order"] != order or state["scheduler_contract"]["horizon"] != horizon:
                raise ValueError("Resume order/horizon mismatch")
            start = state["next_data_group"]
        else:
            seen = {key for record in state["exposure"] for key in record["ids"]}
            replay = set(read_runtime_json(args.replay_ids)) if args.replay_ids else set()
            repeated = seen & {x["id"] for x in examples}
            if repeated != replay:
                raise ValueError(("Explicit replay IDs must exactly match repeated prior exposures",
                                  sorted(repeated), sorted(replay)))
            previous_parent = str(rt.cloud_output(args.continue_from))
            previous_contract = copy.deepcopy(state["scheduler_contract"])
            previous_updates = state["completed_updates"]
            scheduler, contract = rt.make_scheduler(optimizer, horizon, args.warmup)
            state.update(stage="continuation", parent=previous_parent, data_sha256=args.data_sha256,
                         order=order, order_sha256=rt.object_sha(order), mix=mix,
                         next_data_group=0, stage_start_completed_updates=previous_updates,
                         scheduler_contract=contract, continuation_previous_scheduler=previous_contract,
                         explicit_replay_ids=sorted(replay))
            start = 0
    else:
        student, tokenizer = rt.load_student(args.branch, activation_checkpointing=activation)
        optimizer = rt.make_optimizer(student)
        scheduler, contract = rt.make_scheduler(optimizer, horizon, args.warmup)
        state = {"canon_commit": rt.CANON_COMMIT, "branch": args.branch, "stage": "small_R_C",
                 "parent": str(rt.K500 if args.branch == "R" else rt.B0),
                 "data_sha256": args.data_sha256, "order_sha256": rt.object_sha(order),
                 "order": order, "mix": mix, "scheduler_contract": contract,
                 "completed_updates": 0, "stage_start_completed_updates": 0,
                 "next_data_group": 0, "exposure": [], "seed": rt.SEED,
                 "token_chunk_size": args.token_chunk_size, "technical_only": False}
        start = 0
    if not 0 <= start <= len(order):
        raise ValueError("Invalid next data group")
    if state.get("technical_only"):
        raise ValueError("A technical pilot checkpoint cannot initialize a scientific training arm")
    frozen_before = rt.parameter_hashes(student, "frozen")
    references = rt.ReferencePool()
    midpoint = max(1, len(order) // 2)
    checkpoints = run_dir / "checkpoints"
    initial = checkpoints / f"group{start:04d}"
    if not initial.exists():
        rt.save_checkpoint(student, tokenizer, optimizer, scheduler, initial, state)
    elif not args.resume:
        raise FileExistsError("Existing checkpoint in a fresh run directory")
    for index in range(start, len(order)):
        group = [by_id[key] for key in order[index]]
        metrics = rt.synchronize_timed_update(student, optimizer, scheduler, group, references,
                                              mix, args.token_chunk_size)
        state["completed_updates"] += 1
        state["next_data_group"] = index + 1
        state["exposure"].append(rt.exposure_record(group, metrics, state["completed_updates"]))
        record = {"utc": rt.utc(), "branch": args.branch, "stage": state["stage"],
                  "completed_updates": state["completed_updates"], "data_group": index + 1,
                  "ids": [x["id"] for x in group], "technical_only": False, **metrics}
        rt.append_jsonl(run_dir / "updates.jsonl", record)
        if index + 1 in {midpoint, len(order)}:
            if rt.parameter_hashes(student, "frozen") != frozen_before:
                raise AssertionError("Frozen student weights changed")
            references.assert_frozen()
            rt.save_checkpoint(student, tokenizer, optimizer, scheduler,
                               checkpoints / f"group{index + 1:04d}", state)
    if rt.verify_parent_files() != parent_hashes_before:
        raise AssertionError("Parent files changed")
    rt.write_json(run_dir / "completed.json", {
        "utc": rt.utc(), "status": "TRAINING_COMPLETE_QUALITY_NOT_ASSESSED",
        "branch": args.branch, "stage": state["stage"], "gpu": gpu,
        "completed_updates": state["completed_updates"],
        "stage_updates": len(order) - start, "actual_exposure_updates": len(state["exposure"]),
        "actual_row_presentations": sum(len(x["ids"]) for x in state["exposure"]),
        "actual_active_tokens": sum(x["active_tokens"] for x in state["exposure"]),
        "data_sha256": args.data_sha256, "order_sha256": rt.object_sha(order),
        "frozen_student_unchanged": True, "parent_files_unchanged": True,
        "reference_forward_calls": references.forward_calls,
        "final_checkpoint": str(checkpoints / f"group{len(order):04d}"),
        "scientific_gain_claimed": False})
    print("Training queue complete; model quality is not assessed by this runtime.")


if __name__ == "__main__":
    main()
