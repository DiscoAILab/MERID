"""Research memory: distills lessons after each review round and reconciles them with stored entries."""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import tempfile

from merid.core.evidence_guided_evolution.retrospective import assert_no_test


DISTILL_SYSTEM = """You distill research experience after a development experiment.
Return only JSON: {"drafts": [{"id": "d1", "condition": "...", "lesson": "...",
"next_action": "...", "evidence_ids": ["r1:outcome"]}]}.
Produce at most 4 concise drafts, or an empty list when no useful lesson is supported.
Use only supplied evidence IDs, including at least one current-round ID per draft.
State the applicable recipe/model/data conditions. Separate observed outcomes from hypotheses.
Treat failed execution, a valid nonpromoting candidate, and incumbent replacement as different outcomes.
A rejected candidate can suggest another experiment. It does not prove its component is useless.
Aggregate scores do not establish causal effects or clinical validity. A promotion is a development
selection outcome, not independent final performance. Suggest a discriminating next experiment.
The fixed run constraints and executed numeric records take precedence over prior lessons or reports.
Do not include participant data, invent measurements, or change labels, splits, or metric.
Each text field must have at most 600 characters. Evidence IDs must be copied exactly.
"""

RECONCILE_SYSTEM = """You reconcile a persistent research memory with newly distilled experience.
Return only JSON: {"operations": [{"action": "add", "draft": "d1"},
{"action": "revise", "target": "m0001", "draft": "d2"},
{"action": "retire", "target": "m0002", "draft": "d3"}]}.
Use at most 8 operations. Return an empty list for a no-op. These are schema examples, not requests.
Add a distinct useful lesson. Revise an existing lesson to qualify its conditions, integrate new
evidence, or correct a contradiction. Retire a superseded or inapplicable lesson with draft evidence.
Prefer revising a related lesson over accumulating duplicates. Review existing entries for conflicts.
Use only supplied draft IDs and active memory IDs. Do not rewrite drafts or numeric evidence.
Each target may occur only once. Keep the active entry count within the supplied capacity.
Memory lessons are conditional model syntheses, subordinate to executed evidence and run constraints.
"""

MEMORY_USE = ("Use active research memory to choose the next experiment and construct its candidates. "
              "Check each lesson's conditions against current evidence. State which relevant lesson "
              "motivates, redirects, or rules out a proposal. Resolve conflicts using the authoritative "
              "measured records and current run constraints. A lesson is a model synthesis, not a "
              "verified causal rule. An unconfirmed hypothesis calls for a discriminating comparison.")


def _json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 600:
        raise ValueError(name + ' must be nonempty text of at most 600 characters')
    return value.strip()


def _parse(raw, key, limit):
    if not isinstance(raw, str) or len(raw) > 24000:
        raise ValueError('memory response must be bounded JSON text')
    raw = raw.strip()
    if raw.startswith('```json\n') and raw.endswith('```'):
        raw = raw[8:-3].strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find('{'), raw.rfind('}')
        if start < 0 or end <= start:
            raise
        data = json.loads(raw[start:end + 1])
    if not isinstance(data, dict) or set(data) != {key}:
        raise ValueError('expected JSON object containing only ' + key)
    items = data[key]
    if not isinstance(items, list) or len(items) > limit:
        raise ValueError(key + ' exceeds the permitted number of items')
    return items


def _round_evidence(record, candidates, incumbent, constraints, diagnostics):
    n = record['round']
    current = [c for c in candidates if c.get('round') == n]
    valid = sorted((c for c in current if c.get('valid')), key=lambda c: -float(c['score']))
    failed = [c for c in current if not c.get('valid')]
    wanted = {record.get('incumbent'), (incumbent or {}).get('id')}
    anchors = [c for c in candidates if c.get('id') in wanted]
    chosen = anchors + valid[:8] + valid[-4:] + failed[:4]
    event = {k: record[k] for k in ('round', 'status', 'decision', 'incumbent', 'outcome',
                                   'comparison', 'error', 'held_axis', 'search_mode') if k in record}
    event.update(development_incumbent={k: incumbent[k] for k in ('id', 'score') if k in (incumbent or {})},
                 current_round_evaluations=len(current), valid_evaluations=len(valid),
                 failed_evaluations=len(failed), constraints=copy.deepcopy(constraints))
    evidence = {'r%d:outcome' % n: event}
    for c in chosen:
        item = {k: copy.deepcopy(c[k]) for k in
                ('id', 'round', 'recipe', 'config', 'valid', 'n_features', 'error') if k in c}
        if c.get('valid'):
            score = float(c['score'])
            if not math.isfinite(score):
                raise ValueError('a valid evidence score must be finite')
            item['score'] = score
        evidence['r%d:c:%s' % (n, c['id'])] = item
    if diagnostics is not None:
        evidence['r%d:diagnostics' % n] = copy.deepcopy(diagnostics)
    assert_no_test(evidence)
    _json(evidence)
    return evidence


class ResearchMemory:
    def __init__(self, output, scope, max_entries=16):
        if type(max_entries) is not int or not 1 <= max_entries <= 32:
            raise ValueError('memory capacity must be between 1 and 32')
        assert_no_test(scope)
        self.path = Path(output) / 'research_memory.json'
        if self.path.exists():
            raise FileExistsError('research memory requires a fresh run directory: ' + str(self.path))
        self.max_entries = max_entries
        self._state = dict(schema_version=1, scope=copy.deepcopy(scope), version=0,
                           entries=[], evidence={}, events=[])

    def snapshot(self):
        return copy.deepcopy(self._state)

    def context(self):
        entries = [{k: copy.deepcopy(e[k]) for k in
                    ('id', 'condition', 'lesson', 'next_action', 'evidence_ids', 'interpretation')}
                   for e in self._state['entries'] if e['status'] == 'active']
        for entry in entries:
            entry['archived_evidence_count'] = max(0, len(entry['evidence_ids']) - 3)
            entry['evidence_ids'] = entry['evidence_ids'][-3:]
        ids = {i for e in entries for i in e['evidence_ids']}
        evidence = {i: copy.deepcopy(self._state['evidence'][i]) for i in sorted(ids)}
        return dict(version=self._state['version'], scope=copy.deepcopy(self._state['scope']),
                    entries=entries, evidence=evidence)

    def _commit(self, state):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix='.research_memory_', suffix='.tmp', dir=self.path.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(_json(state))
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self._state = state

    def learn(self, proposer, record, candidates, *, incumbent, constraints, diagnostics=None, traj=None):
        assert_no_test(record)
        n = record['round']
        if type(n) is not int or n < 1 or any(e['round'] == n for e in self._state['events']):
            raise ValueError('memory learning requires a new positive round number')
        evidence = _round_evidence(record, candidates, incumbent, constraints, diagnostics)
        state = self.snapshot()
        state['evidence'].update(evidence)
        visible_ids = set(self.context()['evidence']) | set(evidence)
        event = dict(round=n, version_before=state['version'], llm_calls=0)
        try:
            if traj:
                traj.add_system_prompt('experience_distillation', DISTILL_SYSTEM)
                traj.add_system_prompt('memory_reconciliation', RECONCILE_SYSTEM)
            prompt = _json(dict(memory=self.context(), current_evidence=evidence))
            event['llm_calls'] += 1
            raw = proposer.complete(DISTILL_SYSTEM, prompt, max_tokens=2200)
            event['distillation_response'] = str(raw)[:24000]
            drafts = _parse(raw, 'drafts', 4)
            assert_no_test(drafts)
            draft_map = {}
            for d in drafts:
                if not isinstance(d, dict) or set(d) != {'id', 'condition', 'lesson', 'next_action', 'evidence_ids'}:
                    raise ValueError('invalid experience draft fields')
                for field in ('id', 'condition', 'lesson', 'next_action'):
                    d[field] = _text(d[field], field)
                refs = d['evidence_ids']
                if (not isinstance(refs, list) or not 1 <= len(refs) <= 6
                        or not all(isinstance(i, str) and i in visible_ids for i in refs)
                        or not any(i in evidence for i in refs)):
                    raise ValueError('draft requires known evidence IDs and current-round evidence')
                if d['id'] in draft_map:
                    raise ValueError('duplicate draft ID')
                draft_map[d['id']] = d
            prompt = _json(dict(memory=self.context(), drafts=drafts,
                                capacity=self.max_entries, current_evidence=evidence))
            event['llm_calls'] += 1
            raw = proposer.complete(RECONCILE_SYSTEM, prompt, max_tokens=1600)
            event['reconciliation_response'] = str(raw)[:24000]
            ops = _parse(raw, 'operations', 8)
            proposed = copy.deepcopy(state['entries'])
            used = set()
            for op in ops:
                if not isinstance(op, dict) or op.get('action') not in ('add', 'revise', 'retire'):
                    raise ValueError('invalid memory operation')
                action = op['action']
                expected = {'action', 'draft'} if action == 'add' else {'action', 'draft', 'target'}
                if set(op) != expected or not isinstance(op['draft'], str) or op['draft'] not in draft_map:
                    raise ValueError('operation requires a known draft and exact fields')
                d = draft_map[op['draft']]
                if action == 'add':
                    target = dict(id='m%04d' % (len(proposed) + 1), versions=[], evidence_ids=[])
                    proposed.append(target)
                else:
                    target = next((e for e in proposed if e['id'] == op['target'] and e['status'] == 'active'), None)
                    if target is None or op['target'] in used:
                        raise ValueError('operation requires an untouched active memory entry')
                used.add(target['id'])
                update = dict(condition=d['condition'], lesson=d['lesson'], next_action=d['next_action'],
                              evidence_ids=list(dict.fromkeys(target['evidence_ids'] + d['evidence_ids'])),
                              interpretation='model_synthesis', round=n,
                              status='retired' if action == 'retire' else 'active')
                target.update(update)
                target['versions'].append(copy.deepcopy(update))
            if sum(e['status'] == 'active' for e in proposed) > self.max_entries:
                raise ValueError('active memory capacity exceeded; revise or retire existing entries')
            state['entries'] = proposed
            state['version'] += bool(ops)
            event.update(state='updated' if ops else 'unchanged', operations=ops)
        except Exception as exc:
            # Memory updates are atomic: a malformed update is logged and the current bank stays in force.
            event.update(state='failed', error=type(exc).__name__ + ': ' + str(exc)[:400])
        event['version_after'] = state['version']
        state['events'].append(event)
        self._commit(state)
        if traj:
            traj.log('research_memory_update', observation=event)
        return {k: event[k] for k in ('state', 'round', 'llm_calls', 'version_after', 'error') if k in event}
