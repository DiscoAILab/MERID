"""Web search that grounds proposals in cited references."""

RETRIEVE_AGENT_SYSTEM = (
    "You are a research agent grounding a modelling decision. You are given the DECISION to be made and "
    "the OBSERVED DATA REGIME — what was actually measured in this dataset: which feature extractors "
    "exist per modality, their shapes and value statistics, the sample size, the feature dimension, and "
    "the class balance.\n"
    "FORMULATE YOUR OWN SEARCH QUERIES FROM WHAT YOU OBSERVE. Do not search for the dataset name alone. "
    "Let the specifics drive the search: the named extractors, the modality combination actually "
    "available, the n-vs-p regime, the imbalance, and any anomaly in the statistics (e.g. an extractor "
    "whose values span an absurd numeric range is a real finding — search for how practitioners handle "
    "that extractor, not for whether it is popular).\n"
    "Ground EVERY claim in a page you actually retrieved this session and cite its URL. Never answer from "
    "memory; never invent a paper, repo, API, or number. If the web does not answer part of the need, say "
    "so plainly rather than filling the gap.\n"
    "Prefer primary sources: a method's official implementation, its paper, its docs. You are grounding "
    "HOW methods work (features, architecture, training setup) and which are appropriate for THIS regime "
    "— not what score anything achieved. Do not seek leaderboards, result tables, or ground-truth/answer "
    "data: the decision does not need them and they would contaminate it.\n"
    "Finish with 3-6 CONCRETE findings that would change this decision, each with its source URL, each "
    "tied to something specific you observed in the regime."
)


OPTIONAL_SEARCH_CLAUSE = (
    "\n\nYou may search the web (web_search / web_fetch), but SEARCHING IS OPTIONAL AND USUALLY "
    "UNNECESSARY. The observed statistics in front of you often settle the question outright — when they "
    "do, DECIDE FROM THE DATA and do not search. Search only where the observation genuinely leaves you "
    "uncertain: something you cannot read off the numbers (e.g. how practitioners handle an extractor "
    "whose values span this range, or whether a method family suits this n-vs-p regime). Never search to "
    "confirm what you can already see, and never let a search override what the data plainly shows — a "
    "popular recommendation does not beat a measured statistic from THIS dataset. "
    "Begin your reply with one line: `SEARCHED: yes — <what you needed and could not observe>` or "
    "`SEARCHED: no — <what in the observation settled it>`. If you searched, cite the URLs."
)


def decide_with_optional_search(proposer, system, user, max_web_uses=4, max_steps=5, traj=None,
                                ctx="", tag="decide"):
    tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": max_web_uses},
             {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": max_web_uses}]
    client = proposer._client_lazy()
    messages = [{"role": "user", "content": user}]
    web_calls, cited = 0, []
    for _ in range(max_steps):
        try:
            resp = client.messages.create(model=proposer.model, max_tokens=4000,
                                          system=system + OPTIONAL_SEARCH_CLAUSE, tools=tools,
                                          messages=messages)
        except Exception as exc:
            print(f"  [{tag}] LLM error: {type(exc).__name__}: {str(exc)[:150]}", flush=True)
            return ""
        if traj is not None:
            u = getattr(resp, "usage", None)
            it = getattr(u, "input_tokens", 0) or 0
            ot = getattr(u, "output_tokens", 0) or 0
            traj.record_call(it, ot, cost=proposer._usd(it, ot) if hasattr(proposer, "_usd")
                             else it / 1e6 * 5 + ot / 1e6 * 25)
        messages.append({"role": "assistant", "content": resp.content})
        for b in resp.content:
            bt = getattr(b, "type", None)
            if bt == "server_tool_use":
                web_calls += 1
                _inp = getattr(b, 'input', '') or {}
                _q = (_inp.get('query') or _inp.get('url') or _inp.get('command') or _inp.get('code') or str(_inp)
                      ) if isinstance(_inp, dict) else str(_inp)
                print(f"  [{tag}] search: {str(_q)[:90]}", flush=True)
            elif bt in ("web_search_tool_result", "web_fetch_tool_result"):
                c = getattr(b, "content", None)
                if isinstance(c, list):
                    cited += [u for u in (getattr(r, "url", None) for r in c[:5]) if u]
        if resp.stop_reason == "pause_turn":
            continue
        text = "\n".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
        if text:
            first = text.splitlines()[0][:110] if text else ""
            print(f"  [{tag}] {first}", flush=True)
            if traj is not None:
                traj.log(f"{tag}_decision", dataset=ctx, reasoning=text[:4000],
                         searched=web_calls > 0, web_calls=web_calls, urls=sorted(set(cited))[:8])
            return text
    return ""


def agentic_retrieve(proposer, need, profile="", max_web_uses=6, max_steps=6, traj=None, ctx="",
                     **_ignored):
    tools = [{"type": "web_search_20260209", "name": "web_search", "max_uses": max_web_uses},
             {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": max_web_uses}]
    client = proposer._client_lazy()
    messages = [{"role": "user", "content":
                 f"DECISION TO GROUND: {need}\n\nOBSERVED DATA REGIME (this is what was actually measured "
                 f"in the data — derive your searches from it):\n{profile or '(none supplied)'}\n\n"
                 f"Decide what to look up, search, and report your sourced findings."}]
    web_calls, cited = 0, []
    for _ in range(max_steps):
        try:
            resp = client.messages.create(model=proposer.model, max_tokens=4000,
                                          system=RETRIEVE_AGENT_SYSTEM, tools=tools, messages=messages)
        except Exception as exc:
            print(f"  [retrieve] LLM error: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
            return ""
        messages.append({"role": "assistant", "content": resp.content})

        for b in resp.content:
            bt = getattr(b, "type", None)
            if bt == "server_tool_use":
                web_calls += 1
                print(f"  [retrieve] {getattr(b, 'name', '?')}: {str(getattr(b, 'input', ''))[:90]}",
                      flush=True)
            elif bt in ("web_search_tool_result", "web_fetch_tool_result"):
                c = getattr(b, "content", None)
                if isinstance(c, list):
                    cited += [u for u in (getattr(r, "url", None) for r in c[:5]) if u]
                elif getattr(c, "url", None):
                    cited.append(c.url)
                else:
                    print(f"  [retrieve] tool error: {getattr(c, 'error_code', c)}", flush=True)

        if resp.stop_reason == "pause_turn":
            continue

        text = "\n".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
        if text:
            urls = sorted(set(cited))
            print(f"  [retrieve] grounded via {web_calls} web calls, {len(urls)} sources", flush=True)
            if traj is not None:
                traj.log("retrieve_agent", dataset=ctx, reasoning=text[:4000],
                         observation={"web_calls": web_calls, "urls": urls[:10]})
            return text
    return ""
