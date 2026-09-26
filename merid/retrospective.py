"""Evidence-guided review: the Planner, Engineer and Reviewer roles report to the Chair, who decides to accept or revise."""
from __future__ import annotations

import json
import os
import re

_INVESTIGATE = (
    "\nBEFORE REPORTING YOU MAY INVESTIGATE. Emit a single line of the form\n"
    "    CHECK: <tool>(<argument>)\n"
    "and you will be shown the result, then you may check again or report. Available tools:\n"
    "%s\n"
    "You have %d checks. Use them to settle a suspicion you would otherwise have to hedge about — a "
    "width that looks wrong, a column you are not sure exists, a number you are recalling rather than "
    "reading. When you are ready, answer with your report and no CHECK line. Every tool reads the "
    "selection data only; none of them can reach the test set, so do not ask.\n"
)

_FORWARD = (
    "\nTHEN FINISH WITH ONE LINE BEGINNING 'NEXT:'. Name the one direction this search has not "
    "tried that you would spend the remaining budget on, drawn from %s. The review exists to make the "
    "NEXT round of candidates better, not only to catch mistakes, and a review that can only report "
    "that nothing is wrong has done half its job. You are shown what the search already covered: "
    "which model families were scored and how each did, which allowed families were never reached, "
    "which sources appear in the candidates that scored well, and whether the last round moved the "
    "incumbent. Be concrete - what to add - and give the number in your evidence that makes you "
    "expect it to beat the incumbent. The addition is scored on the same venue under the same "
    "promotion rule, so being wrong costs evaluations and retains the incumbent; the thing to avoid "
    "is not a wrong guess but an ungrounded one. If nothing in the evidence supports a direction, "
    "write 'NEXT: none' and say why. Never propose reading held-out data, changing the metric, the "
    "labels, the partition, or restoring an excluded source."
)

ROLE_SYSTEM = {
    "planner": (
        "You are the planner in a post-run review of an automated ML pipeline, and the decisions under "
        "review are your own: which metric this corpus is judged by, which feature sources exist and "
        "how they were grouped, which model families were worth trying, whether a custom method was "
        "needed. You are shown those decisions and the reasoning you gave for them.\n"
        "Report, in at most 8 short lines: what you assumed, and which of those assumptions the run "
        "could have invalidated. Be specific about anything you flagged at the time and want checked — "
        "a source you suspected of sitting upstream of the label, a modality you expected to matter. "
        "Do not state a fact about the data from memory when you can check it."
        + _FORWARD % ("the allowed sources and how they are grouped into recipes - a source "
                      "combination absent from the candidates that scored well, a grouping you did "
                      "not think of when you wrote the space")
    ),
    "engineer": (
        "You are the data engineer in a post-run review of an automated ML pipeline. You are shown what "
        "happened when the plan met the actual files: how many attempts the loader needed and what each "
        "failed on, which sources you refused and why, and the feature width every recipe returned.\n"
        "Report, in at most 8 short lines: which parts of the plan turned out to be unimplementable, "
        "and which require investigation against the declared representation contract: constant "
        "columns, a source that loads but produces nothing that varies, or an operation whose measured "
        "output contradicts its documented semantics. Establish whether inputs are temporal sequences "
        "or already aggregated per-patient/tabular vectors before judging an aggregation setting. "
        "Identical widths are not evidence of identical values or a broken loader. A "
        "suspicion you could have checked and did not is worth less than one you checked."
        + _FORWARD % ("the representations and operations the loaded data actually supports - an "
                      "aggregation or transform that is implementable on these arrays and was never "
                      "scored. Do not propose one the adapter cannot execute")
    ),
    "reviewer": (
        "You are the reviewer in a post-run review of an automated ML pipeline. You are shown the search "
        "results measured on the SELECTION data only — the test set has not been read.\n"
        "Report, in at most 8 short lines, what the numbers say about the SEARCH rather than about the "
        "corpus: the measured score differences and the winner's difference from the supplied "
        "training-fitted trivial reference on the same venue. Discuss statistical uncertainty only "
        "when its computation and results are supplied. Investigate "
        "whether a near-perfect score suggests the winning source is not evidence about the label but "
        "the instrument the label was defined with, and whether the winner's predictions collapse onto "
        "one class. A winner that rests on a verified target-defining source is disqualified, not a "
        "winner: report the best candidate WITHOUT it as the finding. Name the specific numbers you rely on."
        + _FORWARD % ("the model families and hyper-parameter ranges in the allowed list - a family "
                      "in that list that was never scored, or a range the top candidates sit at the "
                      "edge of, which is the usual sign a search stopped at its own boundary")
    ),
}

CHAIR_SYSTEM = (
    "You chair a post-run review of an automated ML pipeline. Three participants have reported, each "
    "from evidence the others did not see: the planner (what was decided and why), the engineer (what "
    "happened when that plan met the files, including any checks it ran), the reviewer (what the "
    "finished search looks like on the selection data). The test set has NOT been read and you are not "
    "shown it.\n"
    "Your job is to investigate contradictions between the accounts against the supplied run contract. "
    "A contradiction can expose a mistaken assumption, but does not itself identify an implementation "
    "error. A suspected target-defining source requires checking provenance and task semantics; its "
    "score alone does not establish leakage or its absence.\n"
    "WHAT TARGET-DEFINING MEANS, EXACTLY. A source is target-defining when the label was COMPUTED from "
    "it: the questionnaire or rating scale whose total or items the label thresholds (PHQ-8/9, HAM-D, "
    "GAD-7 when it defines the label), a clinician diagnosis the label copies, a column derived from the "
    "label. What a participant said, sounded like, looked like or how their brain or body signalled - "
    "interview transcripts, speech, video, EEG, wearables - is BEHAVIOURAL EVIDENCE, even when clinicians "
    "consult it and even when it predicts well. It is upstream of the diagnosis the way symptoms are; it "
    "is not the instrument. On E-DAIC the transcript is evidence and PHQ-8 is the label; on MODMA the "
    "PHQ-9 column is the label and the recordings are evidence. Calling evidence target-defining removes "
    "the modality the study exists to test, so name a source only when you can say which instrument, "
    "item or formula produced the label from it.\n"
    "WHERE THE CALLER SUPPLIES THE LOADER'S FIELD LIST, IT SETTLES WHAT A SOURCE CONTRIBUTES. The "
    "constraints name every literal field the executed loader reads. A record may also contain scale "
    "totals or the label itself; that is not evidence the loader read them, and it is not grounds to "
    "exclude. If you object to a source, name the field in that list you object to. If the field you "
    "would object to is absent from the list, the objection does not apply to this run and you must "
    "say so rather than exclude on the possibility.\n"
    "WHERE THE CALLER SUPPLIES A MEASURED LABEL PROVENANCE, IT OUTRANKS IMPRESSION. That line reports "
    "how much of the label a threshold rule on a single column, or on a simple total, reproduces on "
    "training rows. If it names a column and cut points, the instrument and the formula are established "
    "and exclusion is mandatory. If it reports that no column or total reproduces the label, you may "
    "NOT write that the label is thresholded from that source - to exclude anyway you must name a "
    "mechanism the measurement does not cover (a subset of items, a nonlinear scoring rule, a "
    "documented derivation someone actually read) and say which participant established it.\n"
    "ONCE VERIFIED, A TARGET-DEFINING SOURCE IS DISQUALIFYING, NOT CONFIRMING. If the provenance or "
    "task semantics establish that a source is the instrument, items or criterion the label was "
    "derived from, the verdict is REVISE: exclude_source naming that source - whatever it scores, "
    "and however wide its margin. Such a source is not a strong signal about the label; it is the "
    "label, and a pipeline that wins with it has measured nothing. A verdict of ACCEPT that describes "
    "the winning source as target-defining is a contradiction; do not write one. This holds whether or "
    "not the source sits in the current winner: while it remains in the space it can enter any "
    "candidate, so exclude it as long as it is present - 'it never entered the winner' is not a reason "
    "to keep it.\n"
    "If earlier rounds are shown, say what changed since and whether the revision worked; do not repeat "
    "a revision that has already been tried and did not help.\n"
    "Answer on the FIRST line, exactly one of:\n"
    "  ACCEPT — <why the decisions stand>\n"
    "  REVISE: <one of: exclude_source | eval_metric | recipe_space | adapter | clf_space> — <what is wrong>\n"
    "exclude_source is the SMALL step: name the source and it is removed from the frozen space while the "
    "loader is kept - no re-proposal, no new adapter. Prefer it whenever the problem is one source. "
    "recipe_space re-proposes the whole space from scratch and forces a new loader; use it only when "
    "the space as a whole is wrong. Write the exclusion as: REVISE: exclude_source — <source name>: <why>\n"
    "Revisions trigger measured comparisons. TWO KINDS OF REVISION ARE PERMITTED AND THEY ARE HELD "
    "TO DIFFERENT STANDARDS.\n"
    "  A DEFECT revision - exclude_source, adapter, eval_metric - requires verified evidence: an "
    "established target-defining source, a verified operation-contract violation. A close score "
    "difference, equal feature widths or byte-identical arrays alone is not a verified "
    "implementation failure, and a suspicion is not a defect.\n"
    "  AN OPPORTUNITY revision - clf_space or recipe_space - does NOT require a defect. Nothing "
    "being wrong and the search being incomplete are different findings, and one run can be both. "
    "Order one when the participants' NEXT lines name a concrete unexplored direction and you can "
    "point to the number that makes it plausible: a model family in the allowed list that was never "
    "scored, a hyper-parameter range the top candidates sit at the edge of, a source combination "
    "absent from the candidates that scored well. It is additive and safe to be wrong about - the "
    "new candidates are scored on the same venue under the same promotion rule, so a poor direction "
    "costs evaluations and retains the incumbent; it cannot lower what this run reports. Prefer "
    "clf_space, which the existing loader scores directly. Use recipe_space only when the direction "
    "is a source combination the current recipes do not contain, and keep it to sources the loader "
    "already reads. Do NOT order one when the remaining evaluation budget shown in the constraints "
    "leaves no room, when no participant could name a grounded direction, or when an earlier round "
    "already tried it.\n"
    "A run where nothing is broken and budget remains is the ORDINARY case for an opportunity "
    "revision, not a reason to ACCEPT. Reserve ACCEPT for a search whose remaining directions are "
    "exhausted, unaffordable, already tried, or named by nobody - and say which of those it was. "
    "Apply the caller's actual acceptance "
    "policy; do not introduce a different tie rule, metric or threshold.\n"
    "DIAGNOSTIC HYPOTHESES, NOT ESTABLISHED CAUSES. A source scoring below a trivial predictor "
    "can reflect sampling noise, class imbalance, model mismatch, distribution changes or an "
    "extraction error. Check these possibilities; the score alone does not identify a cause. "
    "Before discussing an unused modality, check whether it is deliberately excluded, unavailable, "
    "unmatched to patients, or outside prediction-time evidence. Its absence does not establish a "
    "space or loader defect. Do not ask to restore deliberately excluded sources. "
    "After that line, give the note to pass to whichever step is revised: what it should do "
    "differently, in its own terms. For an opportunity revision that note IS the next candidate: "
    "name the family, range or source combination to add and the observed number behind it, so the "
    "step receiving it proposes that rather than re-deriving a space from the same profile. Keep the whole answer under 220 words. Your LAST line must be exactly "
    "    TARGET-DEFINING SOURCES IN SPACE: <comma-separated source names, or none>\n"
    "listing every source still in the space that any participant established as the instrument, items "
    "or criterion the label was derived from. Whether such a source won, lost, was capped or never ran "
    "makes no difference to that line: it is about what is in the space, and a named source is removed."
)

SOLO_SYSTEM = (
    "You are the sole reviewer in a post-run review of an automated ML pipeline, and you hold every "
    "piece of evidence the run produced: the decisions that were taken and the reasoning given for "
    "them, what happened when that plan met the actual files, and what the finished search looks "
    "like on the selection data. The test set has NOT been read and you are not shown it.\n"
    "Do the whole job yourself. Report what you assumed and what the run could have invalidated; "
    "which parts of the plan turned out unimplementable or contradict the declared representation "
    "contract; and what the numbers say about the SEARCH rather than about the corpus - the measured "
    "score differences and the winner\'s difference from the supplied training-fitted trivial "
    "reference on the same venue. Investigate whether a near-perfect score means the winning source "
    "is not evidence about the label but the instrument the label was defined with, and whether the "
    "winner\'s predictions collapse onto one class. Name the specific numbers you rely on, and do "
    "not state a fact about the data from memory when you can check it.\n"
    "Then decide, in the same grammar the chair uses, and under the same rules.\n"
)


def solo_system():
    return (SOLO_SYSTEM
            + _FORWARD % ("anything in the evidence you hold - an unscored model family, a "
                          "hyper-parameter range the top candidates sit at the edge of, a source "
                          "combination absent from the candidates that scored well")
            + chr(10) + CHAIR_SYSTEM)


_SELECTION_CONTRACT = (
    "\nSUPPLIED SELECTION CONTRACT GOVERNS THIS REVIEW. Follow the supplied selection/revision "
    "policy and immutable source, task and grouping constraints. When it specifies strict improvement "
    "with ties retaining the incumbent, use that policy. Apply a one-standard-error rule only when "
    "the supplied contract explicitly specifies it AND computed standard-error evidence is supplied. "
    "If no selection rule or uncertainty computation is supplied, say it is unavailable; do not "
    "assume one was used. Do not invent standard errors, confidence intervals, statistical ties or "
    "significance from aggregate score differences.\n"
    "REPRESENTATION SEMANTICS MATTER. Temporal sequences and already aggregated/tabular vectors "
    "are different input contracts. Temporal aggregation settings may intentionally have no effect "
    "on raw tabular vectors; do not demand synthetic temporal variation. Equal widths do not imply "
    "equal arrays. Measured byte equivalence establishes equality on the inspected train/dev arrays "
    "only; it does not identify a loader error, its cause or equivalence on future inputs. Check "
    "the declared source representation and operation semantics before alleging a violation. "
    "Distinguish an intentional equivalent candidate from a verified ignored operation and from "
    "an unresolved hypothesis. Respect exclusions as experimental conditions."
)
ROLE_SYSTEM = {role: system + _SELECTION_CONTRACT for role, system in ROLE_SYSTEM.items()}
CHAIR_SYSTEM += _SELECTION_CONTRACT
ROLE_SYSTEM["solo"] = solo_system()

REVISION_REQUIRED = (
    "THIS IS A FIXED-HORIZON SELF-IMPROVEMENT RUN AND REVISION BUDGET REMAINS: %s.\n"
    "In this run ACCEPT does not end the search and is not an available verdict while budget "
    "remains: every round tries the most promising revision the evidence supports. Choose the single "
    "best opportunity revision - or exclude_source, if a verified target-defining source is still in "
    "the space - and state it in the required shape: the FIRST line exactly 'REVISE: <clf_space|"
    "recipe_space|exclude_source> — <what to add or remove, and the observed number behind it>'; then "
    "the note; the LAST line 'TARGET-DEFINING SOURCES IN SPACE: <names, or none>'. The incumbent is "
    "replaced only by a measured strict improvement on the same venue, so a direction that does not "
    "help costs evaluations and cannot lower what this run reports. Do not repeat a revision an "
    "earlier round already tried.")
_REVISE_STEPS = ("exclude_source", "eval_metric", "recipe_space", "adapter", "clf_space")


def _revise_step(directive):
    if directive is None:
        return None
    m = re.match(r"\s*REVISE:\s*([a-z_]+)(?=$|\s|[-—:])", directive[1], re.I)
    return m.group(1).lower() if m and m.group(1).lower() in _REVISE_STEPS else None


def _demand_revision(proposer, system, packet, previous, require_revision):
    demand = (packet + "\n\nYOUR PREVIOUS REPLY:\n" + previous[:1500] + "\n\n"
              + REVISION_REQUIRED % require_revision)
    try:
        return (proposer.complete(system, demand, max_tokens=700) or "").strip()
    except Exception as exc:
        return "REVIEW_FAILED — the revision demand could not be answered (%s)" % type(exc).__name__

_CHECK_RE = re.compile(r"^\s*CHECK:\s*([a-z_]+)\s*\((.*?)\)\s*$", re.I | re.M)


def assert_no_test(evidence) -> None:
    blob = json.dumps(evidence, default=str).lower()
    for bad in ("test_pred", "test_score", "test_acc", "test_result", '"test_x"', "y_test"):
        if bad in blob:
            raise AssertionError(
                "retrospective evidence contains %r — the review may only see the selection venue. "
                "Revising a decision after seeing test performance is test-set selection." % bad)


def _run_tool(tools, name, arg):
    fn = tools.get(name)
    if fn is None:
        return "no such tool %r; available: %s" % (name, ", ".join(sorted(tools)))
    try:
        out = fn(arg.strip().strip("'\""))
    except Exception as exc:
        return "%s raised %s: %s" % (name, type(exc).__name__, str(exc)[:200])
    return json.dumps(out, default=str, ensure_ascii=False)[:2000]


def _speak(proposer, role, packet, tools=None, budget=5, traj=None, ctx=""):
    system = ROLE_SYSTEM[role]
    if tools:
        menu = "\n".join("    %s(%s)" % (n, d) for n, (d, _f) in sorted(tools.items()))
        system += _INVESTIGATE % (menu, budget)
    calls = {n: f for n, (_d, f) in (tools or {}).items()}
    body = json.dumps(packet, indent=1, default=str, ensure_ascii=False)[:6000]
    convo = "YOUR EVIDENCE:\n" + body
    used, spent, txt = [], 0, ""
    for _turn in range(2 * budget + 2):
        try:
            txt = proposer.complete(system, convo, max_tokens=900) or ""
        except Exception as exc:
            return "(%s could not report: %s: %s)" % (role, type(exc).__name__, str(exc)[:120]), used
        m = _CHECK_RE.search(txt)
        if not m:
            break
        if spent >= budget:
            convo += (chr(10) + chr(10) + "No checks remain. Write your report now from what you "
                      "have, and say plainly which suspicions you could not settle rather than "
                      "leaving them mid-sentence.")
            try:
                txt = proposer.complete(system, convo, max_tokens=900) or txt
            except Exception:
                pass
            break
        name, arg = m.group(1).lower(), m.group(2)
        res = _run_tool(calls, name, arg)
        ok = not res.startswith("no such tool") and " raised " not in res[:60]
        if ok:
            spent += 1
            used.append({"tool": name, "arg": arg, "result": res[:400]})
        print("    [%s checks%s] %s(%s) -> %s"
              % (role, "" if ok else "/malformed", name, arg[:40], res[:110]), flush=True)
        convo += (chr(10) + chr(10) + "YOU ASKED: CHECK: %s(%s)" + chr(10) + "RESULT:" + chr(10)
                  + "%s" + chr(10) + chr(10) + "%s") % (
            name, arg, res,
            ("That call was not well formed; re-read the tool description and try again — it did not "
             "cost you a check.") if not ok else
            ("Checks left: %d. Continue: check again, or write your report." % (budget - spent)))
    txt = _CHECK_RE.sub("", txt).strip()

    if traj is not None:
        traj.log("retro_%s" % role, dataset=ctx, reasoning=txt, observation=packet, checks=used)
    return txt.strip(), used


def _hold_single_agent(proposer, evidence, tools=None, prior_rounds=None, budget=5, traj=None,
                       ctx="", return_status=False, require_revision=None):
    merged, merged_tools = {}, {}
    for role in ("planner", "engineer", "reviewer"):
        for key, value in (evidence.get(role) or {}).items():
            if key not in merged:
                merged[key] = value
        for name, entry in ((tools or {}).get(role) or {}).items():
            merged_tools.setdefault(name, entry)
    if prior_rounds:
        merged["earlier_rounds_of_this_review"] = prior_rounds
    statement, used = _speak(proposer, "solo", merged, tools=merged_tools, budget=3 * budget,
                             traj=traj, ctx=ctx)
    print("  [review/solo] %s" % statement.replace(chr(10), " ")[:160], flush=True)
    lines, directive, headline = _parse_chair(statement)
    minutes = "== SOLE REVIEWER ==" + chr(10)
    if used:
        minutes += "(checks run: %s)%s" % ("; ".join("%s(%s)" % (c["tool"], c["arg"][:30])
                                                     for c in used), chr(10))
    minutes += statement
    failed = not statement or statement.startswith("(solo could not report:")
    revision_demanded = False
    if require_revision and not failed and _revise_step(directive) is None:
        revision_demanded = True
        body = "YOUR EVIDENCE:\n" + json.dumps(merged, indent=1, default=str, ensure_ascii=False)[:6000]
        reply = _demand_revision(proposer, ROLE_SYSTEM["solo"], body, statement, require_revision)
        minutes += "\n\n== SOLE REVIEWER (revision required: budget remains) ==\n" + reply
        lines3, directive3, headline3 = _parse_chair(reply)
        if _revise_step(directive3) is not None:
            statement, lines, directive, headline = reply, lines3, directive3, headline3
    print("  [review/chair] %s" % headline[:200], flush=True)
    if traj is not None:
        traj.log("retro_decision", dataset=ctx, reasoning=statement, choice=headline[:200])
    verdict, state = None, "review_failed" if failed else "invalid_proposal"
    if not failed and directive is not None:
        index, line = directive
        m = re.match(r"\s*REVISE:\s*([a-z_]+)(?=$|\s|[-\u2014:])\s*[-\u2014:]?\s*(.*)", line, re.I)
        if m and m.group(1).lower() in ("exclude_source", "eval_metric", "recipe_space", "adapter",
                                        "clf_space"):
            note = (m.group(2).strip() + chr(10)
                    + chr(10).join(lines[:index] + lines[index + 1:])).strip()
            verdict, state = {"step": m.group(1).lower(), "note": note}, "revision_proposed"
        elif re.match(r"^\s*ACCEPT(?:\s|$|[-\u2014:])", line, re.I):
            state = "accepted"
    named = []
    for line in lines:
        mm = re.match(r"^\s*TARGET-DEFINING SOURCES IN SPACE\s*[:\uff1a]\s*(.*)$", line, re.I)
        if mm:
            named = [t for t in re.findall(r"[A-Za-z0-9_]+", mm.group(1))
                     if t.lower() not in ("none", "no", "n", "a", "nil", "null")]
    if named and not failed and (verdict is None or verdict["step"] != "exclude_source"):
        print("  [review/chair] OVERRIDE -> REVISE: exclude_source \u2014 %s (named target-defining "
              "but not excluded)" % named[0], flush=True)
        verdict = {"step": "exclude_source",
                   "note": "%s: listed by the sole reviewer as target-defining; exclusion is "
                           "mandatory. First line: %s" % (named[0], headline[:200])}
        state = "revision_proposed"
    status = dict(state=state, complete=state in ("accepted", "revision_proposed"),
                  failed_roles=["solo"] if failed else [], chair_failed=failed,
                  chair_retried=False, review_mode="single_agent")
    if require_revision:
        status.update(revision_required=True, revision_demanded=revision_demanded)
    if traj is not None:
        traj.log("retro_status", dataset=ctx, observation=status)
    return (verdict, minutes, status) if return_status else (verdict, minutes)


def hold_review(proposer, evidence, tools=None, prior_rounds=None, budget=5, traj=None, ctx="",
                return_status=False, mode="roles", require_revision=None):
    assert_no_test(evidence)
    if prior_rounds:
        assert_no_test(prior_rounds)
    if mode == "single_agent":
        return _hold_single_agent(proposer, evidence, tools=tools, prior_rounds=prior_rounds,
                                  budget=budget, traj=traj, ctx=ctx, return_status=return_status,
                                  require_revision=require_revision)
    statements, checks = {}, {}
    for role in ("planner", "engineer", "reviewer"):
        pkt = dict(evidence.get(role, {}))
        if prior_rounds:
            pkt["earlier_rounds_of_this_review"] = prior_rounds
        statements[role], checks[role] = _speak(
            proposer, role, pkt, tools=(tools or {}).get(role), budget=budget, traj=traj, ctx=ctx)
        print("  [review/%s] %s" % (role, statements[role].replace("\n", " ")[:160]), flush=True)

    parts = []
    for r in ("planner", "engineer", "reviewer"):
        head = "== %s ==" % r.upper()
        if checks[r]:
            head += "\n(checks run: %s)" % "; ".join("%s(%s)" % (c["tool"], c["arg"][:30])
                                                     for c in checks[r])
        parts.append(head + "\n" + statements[r])
    packet = "\n\n".join(parts)
    if prior_rounds:
        packet = ("== EARLIER ROUNDS ==\n" + json.dumps(prior_rounds, default=str,
                                                        ensure_ascii=False)[:2500] + "\n\n" + packet)
    chair_failed = False
    try:
        decision = (proposer.complete(CHAIR_SYSTEM, packet, max_tokens=700) or "").strip()
    except Exception as exc:
        chair_failed = True
        decision = "REVIEW_FAILED — the chair could not be reached (%s)" % type(exc).__name__
    lines, directive, headline = _parse_chair(decision)
    minutes = packet + "\n\n== CHAIR ==\n" + decision
    chair_retried = False
    if not chair_failed and directive is None:
        chair_retried = True
        reminder = (packet + "\n\nYOUR PREVIOUS REPLY:\n" + decision[:1500] +
                    "\n\nThat reply carried no valid verdict at a response boundary, so it cannot be applied. "
                    "State the SAME judgement again in the required shape and nothing else: the FIRST line is "
                    "exactly one of 'ACCEPT — <why>' or 'REVISE: <exclude_source|eval_metric|recipe_space|adapter|"
                    "clf_space> — <what is wrong>'; then the note; the LAST line is "
                    "'TARGET-DEFINING SOURCES IN SPACE: <names, or none>'.")
        try:
            decision2 = (proposer.complete(CHAIR_SYSTEM, reminder, max_tokens=700) or "").strip()
        except Exception as exc:
            decision2 = "REVIEW_FAILED — the chair could not be reached on retry (%s)" % type(exc).__name__
        lines2, directive2, headline2 = _parse_chair(decision2)
        minutes += "\n\n== CHAIR (format retry) ==\n" + decision2
        if directive2 is not None:
            decision, lines, directive, headline = decision2, lines2, directive2, headline2
    failed_roles = [r for r, statement in statements.items()
                    if not statement or statement.startswith('(%s could not report:' % r)]
    revision_demanded = False
    if require_revision and not chair_failed and not failed_roles and _revise_step(directive) is None:
        revision_demanded = True
        decision3 = _demand_revision(proposer, CHAIR_SYSTEM, packet, decision, require_revision)
        lines3, directive3, headline3 = _parse_chair(decision3)
        minutes += "\n\n== CHAIR (revision required: budget remains) ==\n" + decision3
        if _revise_step(directive3) is not None:
            decision, lines, directive, headline = decision3, lines3, directive3, headline3
    print("  [review/chair] %s%s%s" % (headline[:200], " (after format retry)" if chair_retried else "",
                                       " (after revision demand)" if revision_demanded else ""), flush=True)
    if traj is not None:
        traj.log("retro_decision", dataset=ctx, reasoning=decision, choice=headline[:200])

    verdict = None
    state = 'review_failed' if chair_failed or failed_roles else 'invalid_proposal'
    if not chair_failed and not failed_roles and directive is not None:
        index, line = directive
        m = re.match(r"\s*REVISE:\s*([a-z_]+)(?=$|\s|[-—:])\s*[-—:]?\s*(.*)", line, re.I)
        if m and m.group(1).lower() in ('exclude_source', 'eval_metric', 'recipe_space', 'adapter', 'clf_space'):
            note = (m.group(2).strip() + '\n' + '\n'.join(lines[:index] + lines[index + 1:])).strip()
            verdict = {'step': m.group(1).lower(), 'note': note}
            state = 'revision_proposed'
        elif re.match(r'^\s*ACCEPT(?:\s|$|[-—:])', line, re.I):
            state = 'accepted'
    named = []
    for line in lines:
        mm = re.match(r'^\s*TARGET-DEFINING SOURCES IN SPACE\s*[:：]\s*(.*)$', line, re.I)
        if mm:
            named = [t for t in re.findall(r'[A-Za-z0-9_]+', mm.group(1))
                     if t.lower() not in ('none', 'no', 'n', 'a', 'nil', 'null')]
    if named and not chair_failed and not failed_roles and (verdict is None or verdict['step'] != 'exclude_source'):
        print("  [review/chair] OVERRIDE -> REVISE: exclude_source — %s (the chair listed it as "
              "target-defining but did not exclude it; its first line was: %s)" % (named[0], headline[:90]),
              flush=True)
        verdict = {'step': 'exclude_source',
                   'note': '%s: listed by the chair as target-defining (the instrument or criterion the label '
                           'was derived from); exclusion is mandatory. Chair first line: %s' % (named[0], headline[:200])}
        state = 'revision_proposed'
    status = dict(state=state, complete=state in ('accepted', 'revision_proposed'),
                  failed_roles=failed_roles, chair_failed=chair_failed, chair_retried=chair_retried)
    if require_revision:
        status.update(revision_required=True, revision_demanded=revision_demanded)
    if traj is not None:
        traj.log('retro_status', dataset=ctx, observation=status)
    return (verdict, minutes, status) if return_status else (verdict, minutes)


_TARGET_LINE = re.compile(r'^\s*TARGET-DEFINING SOURCES IN SPACE\s*[:：]', re.I)


def _parse_chair(decision):
    lines = [line.strip() for line in decision.splitlines() if line.strip()]
    directives, fence = [], None
    for i, line in enumerate(lines):
        marker = re.match(r'^(`{3,}|~{3,})', line)
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None
            continue
        if fence is None and re.match(r'^(?:ACCEPT(?:\s|$|[-—:])|REVISE:)', line, re.I):
            directives.append((i, line))
    body = [i for i, line in enumerate(lines) if not _TARGET_LINE.match(line)]
    boundaries = {body[0], body[-1]} if body else set()
    directive = (directives[0] if len(directives) == 1 and directives[0][0] in boundaries else None)
    headline = directive[1] if directive else (lines[0] if lines else
                                              'INVALID_PROPOSAL — empty chair response')
    return lines, directive, headline


def write_minutes(freeze_dir, ctx, task, round_no, minutes):
    os.makedirs(freeze_dir, exist_ok=True)
    safe = ctx.replace("/", "_")
    fp = os.path.join(freeze_dir, "review_%s_%s_r%d.md" % (safe, task, round_no))
    with open(fp, "w", encoding="utf-8") as fh:
        fh.write("# Post-run review — %s / %s, round %d\n\n" % (ctx, task, round_no))
        fh.write("Held on selection-venue evidence only. This does not attest to file reads inside frozen adapters.\n\n")
        fh.write(minutes + "\n")
    print("  [review] minutes -> %s" % fp, flush=True)
    return fp
