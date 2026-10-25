"""Fixed061 development current-input ledger; no model or future-state reads."""

from pathlib import Path

from s2a.clean_training_runtime import reference, require, validate_current_request


def case_order():
    return [
        dict(
            case_index=i,
            case_id=f"task{task}_init0_s{seed}_{variant}",
            task_index=task,
            record_name=f"task{task}_init0",
            sampling_seed=seed,
            variant=variant,
            expected_requests=75,
        )
        for i, (task, seed, variant) in enumerate(
            (task, seed, variant) for task in range(3) for seed in (42, 43) for variant in ("raw", "terminal058")
        )
    ]


def case_directory(root, case):
    return (
        Path(root)
        / "development/cases"
        / case["case_id"]
        / "simulation/8400_masked_position_hold"
        / (f"{case['record_name']}_s{case['sampling_seed']}")
    )


class DevelopmentPolicy:
    """One irreversible ordered ledger; no model or future-data dependency."""

    def __init__(self, prepared):
        self.root = Path(prepared["root"]).resolve()
        self.order = case_order()
        self.templates = {e["request"]["prompt"]: e["request"] for e in prepared["zero_cases"]}
        self.tasks = {}
        require(
            len(prepared["development_tasks"]) == 3
            and all(set(item) == {"task_index", "task"} for item in prepared["development_tasks"]),
            "Exactly three explicit task descriptors required",
        )
        for anchor in prepared["development_tasks"]:
            task = anchor["task_index"]
            require(task not in self.tasks or self.tasks[task] == anchor["task"], "task metadata is inconsistent")
            self.tasks[task] = anchor["task"]
        require(
            set(self.tasks) == {0, 1, 2} and set(self.templates) == set(self.tasks.values()),
            "three fixed tasks required",
        )
        self.case_index = -1
        self.case_requests = 0
        self.requests = []
        self.pending = None
        self.poisoned = False
        self.stopped = False
        self.finish_requested = False
        self.first_raw = {}

    @property
    def current(self):
        return None if self.case_index < 0 else self.order[self.case_index]

    def bind(self, body):
        require(
            not self.poisoned and not self.stopped and self.pending is None, "service is failed, closed or occupied"
        )
        require(
            self.case_index < 11 and (self.case_index < 0 or self.case_requests == 75), "preceding case is incomplete"
        )
        expected = self.order[self.case_index + 1]
        require(body == expected, "case order/identity differs from the complete fixed12 matrix")
        self.case_index += 1
        self.case_requests = 0
        return self.current

    def reserve(self, request):
        require(
            not self.poisoned and not self.stopped and self.pending is None, "service is failed, closed or occupied"
        )
        case = self.current
        require(
            case is not None and self.case_requests < 75 and len(self.requests) < 900, "case/request budget exhausted"
        )
        directory = case_directory(self.root, case)
        image = directory / f"input_{self.case_requests:04d}.png"
        require(
            image.resolve().is_relative_to(self.root) and request.get("vision_path") == str(image),
            "current image is outside the bound case directory",
        )
        image_ref = reference(image)
        task = self.tasks[case["task_index"]]
        validate_current_request(
            request,
            name=f"s2a_{directory.name}_{self.case_requests}",
            seed=case["sampling_seed"],
            image_ref=image_ref,
            task=task,
            template=self.templates[task],
        )
        key = (case["task_index"], case["sampling_seed"])
        if case["variant"] == "terminal058" and self.case_requests == 0:
            require(
                key in self.first_raw and image_ref["sha256"] == self.first_raw[key]["image_ref"]["sha256"],
                "raw/terminal058 first current PNG bytes differ",
            )
        self.pending = dict(
            case_index=self.case_index,
            case_id=case["case_id"],
            request_index=self.case_requests,
            global_request_index=len(self.requests),
            variant=case["variant"],
            sampling_seed=case["sampling_seed"],
            image_ref=image_ref,
        )
        return dict(self.pending)

    def complete(self, record):
        require(self.pending is not None and not self.poisoned and not self.stopped, "no admitted pending request")
        require(all(record.get(k) == v for k, v in self.pending.items()), "completed request identity changed")
        require(record.get("complete") is True, "partial model request cannot be counted")
        case = self.current
        if self.case_requests == 0 and case["variant"] == "raw":
            self.first_raw[(case["task_index"], case["sampling_seed"])] = dict(record)
        self.requests.append(record)
        self.case_requests += 1
        self.pending = None

    def finish(self, body):
        require(body == dict(completed_cases=12, completed_requests=900), "finish body differs from fixed budget")
        require(
            not self.poisoned
            and self.pending is None
            and self.case_index == 11
            and self.case_requests == 75
            and len(self.requests) == 900,
            "complete12/900 ledger required",
        )
        self.finish_requested = self.stopped = True

    def abort(self, body):
        require(
            isinstance(body, dict)
            and set(body) == {"case_id", "reason"}
            and body["case_id"] == (None if self.current is None else self.current["case_id"])
            and isinstance(body["reason"], str)
            and 0 < len(body["reason"]) <= 1024,
            "invalid owned abort",
        )
        self.poisoned = self.stopped = True
        return dict(body)
