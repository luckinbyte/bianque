"""A2UI v0.9 adapter: translates the agent loop's event stream into a stream of
A2UI envelopes (``createSurface`` / ``updateComponents`` / ``updateDataModel``)
plus a few native passthroughs (the context meter, the ask_user clarification,
and terminal signals).

Implemented as an ``emit`` wrapper. The agent loop calls an ``A2UIAdapter``
instance exactly as it would call the raw queue emit, so :func:`run_turn` and
:func:`run_subagent` are unchanged — the sub-agent and all existing tests are
unaffected. One adapter instance = one turn = one A2UI surface.

Catalog: a small extension of A2UI's basic catalog (Column / Row / Card / Text /
Tabs / Divider / Chips) with ``tone`` / ``variant`` / ``status`` props that the
bianque renderer understands. ``catalogId`` = ``https://bianque.local/a2ui/v1``.

Design rules followed:
- Structure vs content split: ``updateComponents`` declares a block's component
  tree once (low frequency); ``updateDataModel`` streams volatile content
  (reasoning text, args, results, status) over JSON-Pointer-bound paths.
- Heterogeneous blocks are managed here (stable component ids + root children
  list). The renderer only handles the static ``children: [ids]`` array form.
- ``{call}`` (client-side functions) is never emitted — strings are pre-formatted
  server-side, so the renderer only resolves literals and ``{path}`` bindings.
"""
from __future__ import annotations

import json
import secrets
from typing import Any, Awaitable, Callable

from app.agent.redact import redact_event, redact_tool_summary, scrub_text

CATALOG_ID = "https://bianque.local/a2ui/v1"
VERSION = "v0.9"

Emit = Callable[[dict[str, Any]], Awaitable[None]]

# Event types forwarded to the client untouched (app chrome, not agent output).
# NOTE: ``clarification`` is NOT here — its agent-authored question text must be
# scrubbed (the model may quote source into it), so it gets a dedicated branch.
_PASSTHROUGH = {"context"}


def _args_label(args: dict[str, Any]) -> str:
    """Render a tool's args as a compact human-readable subtitle string."""
    if not args:
        return ""
    parts: list[str] = []
    for k, v in args.items():
        sv = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        if len(sv) > 60:
            sv = sv[:57] + "…"
        parts.append(f"{k}: {sv}")
    return "  ".join(parts)


class A2UIAdapter:
    """An ``emit`` wrapper that turns agent events into an A2UI envelope stream.

    Call ``await adapter(event)`` exactly like the raw emit. Each turn should
    use a fresh instance (one surface per turn, so native user bubbles and A2UI
    agent output interleave correctly).
    """

    def __init__(
        self,
        emit: Emit,
        *,
        surface_id: str | None = None,
        catalog_id: str = CATALOG_ID,
    ) -> None:
        self._emit = emit
        self._surface_id = surface_id or f"turn-{secrets.token_urlsafe(8)}"
        self._catalog_id = catalog_id
        self._created = False
        self._n = 0
        self._children: list[str] = []          # root Column's block ids, in order
        self._open_reasoning: str | None = None
        self._reasoning_text: dict[str, str] = {}
        self._call_to_block: dict[str, str] = {}   # tool call_id -> block id
        self._call_to_tool: dict[str, tuple[str, dict]] = {}  # call_id -> (tool name, args) for redaction
        self._explore_calls: set[str] = set()       # call_ids whose block is `explore`
        self._journey_children: dict[str, list[str]] = {}  # explore block id -> journey child ids
        self._step_index: dict[str, dict[str, str]] = {}   # explore block id -> {sub_call_id: step id}
        self._sub_args: dict[str, dict[str, tuple[str, dict]]] = {}  # block id -> {sub_call_id -> (tool, args)}
        self._open_seg: dict[str, str | None] = {}  # explore block id -> open reasoning segment id
        self._seg_text: dict[str, str] = {}         # reasoning segment id -> accumulated raw text

    # ----- low-level envelope emitters -----

    async def _send(self, key: str, payload: dict[str, Any]) -> None:
        await self._emit({"version": VERSION, key: payload})

    async def _components(self, components: list[dict[str, Any]]) -> None:
        await self._send("updateComponents", {"surfaceId": self._surface_id, "components": components})

    async def _data(self, path: str, value: Any) -> None:
        await self._send("updateDataModel", {"surfaceId": self._surface_id, "path": path, "value": value})

    async def _ensure_surface(self) -> None:
        if self._created:
            return
        self._created = True
        await self._send("createSurface", {"surfaceId": self._surface_id, "catalogId": self._catalog_id})
        await self._components([{"id": "root", "component": "Column", "children": []}])

    def _new_id(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}{self._n}"

    async def _append_block(self, block_id: str, subtree: list[dict[str, Any]]) -> None:
        """Emit a block's component subtree and grow root's children to include it."""
        self._children.append(block_id)
        root = {"id": "root", "component": "Column", "children": list(self._children)}
        await self._components([*subtree, root])

    async def _close_reasoning(self) -> None:
        self._open_reasoning = None

    # ----- the emit interface the agent loop calls -----

    async def __call__(self, event: dict[str, Any]) -> None:
        t = event.get("type")
        if t in _PASSTHROUGH:
            # App chrome: forward untouched, no surface needed.
            await self._emit(event)
            return
        if t == "clarification":
            # App chrome, but its question text is agent-authored and may quote
            # source — scrub it on the wire, without spawning a surface.
            await self._emit(redact_event(event))
            return
        await self._ensure_surface()
        if t == "step":
            await self._on_step(event)
        elif t == "tool_call":
            await self._on_tool_call(event)
        elif t == "tool_result":
            await self._on_tool_result(event)
        elif t == "subagent_started":
            await self._on_subagent(event, status="exploring")
        elif t == "subagent_step":
            await self._on_subagent_step(event)
        elif t == "subagent_tool_call":
            await self._on_subagent_tool_call(event)
        elif t == "subagent_tool_result":
            await self._on_subagent_tool_result(event)
        elif t == "subagent_finished":
            await self._on_subagent(event, status="done" if event.get("ok") else "failed")
        elif t == "answer":
            await self._on_answer(event)
        elif t == "error":
            await self._on_terminal_block(event, "误  " + scrub_text(str(event.get("message", ""))), "error")
        elif t == "cancelled":
            await self._on_terminal_block(event, "已停止", "muted")
        else:
            await self._emit(redact_event(event))  # unknown: forward, defensively scrubbed

    # ----- block handlers -----

    async def _on_step(self, event: dict[str, Any]) -> None:
        rid = self._open_reasoning
        if rid is None:
            rid = self._new_id("r")
            self._open_reasoning = rid
            self._reasoning_text[rid] = ""
            subtree = [{
                "id": rid, "component": "Text",
                "text": {"path": f"/blocks/{rid}/text"},
                "variant": "reasoning", "tone": "muted",
            }]
            await self._append_block(rid, subtree)
        self._reasoning_text[rid] += str(event.get("delta", ""))
        # Scrub the accumulated text on send (not per-delta) so a fenced block
        # that streams across deltas is removed coherently, with no stray fence.
        await self._data(f"/blocks/{rid}/text", scrub_text(self._reasoning_text[rid]))

    async def _on_tool_call(self, event: dict[str, Any]) -> None:
        await self._close_reasoning()
        call_id = str(event.get("call_id"))
        name = str(event.get("tool", "?"))
        args = event.get("args") or {}
        self._call_to_tool[call_id] = (name, args)
        if name == "explore":
            await self._create_explore(call_id, args)
        else:
            await self._create_tool(call_id, name, args)

    async def _create_tool(self, call_id: str, name: str, args: dict[str, Any]) -> None:
        bid = self._new_id("t")
        body = f"{bid}_body"
        self._call_to_block[call_id] = bid
        subtree = [
            {
                "id": bid, "component": "Card", "icon": "🔧", "title": name,
                "subtitle": _args_label(args),
                "status": {"path": f"/blocks/{bid}/status"},
                "tone": "tool", "collapsible": True, "child": body,
            },
            {
                "id": body, "component": "Text",
                "text": {"path": f"/blocks/{bid}/result"},
                "variant": "result", "tone": "muted",
            },
        ]
        await self._append_block(bid, subtree)
        await self._data(f"/blocks/{bid}", {"status": "running", "result": ""})

    async def _create_explore(self, call_id: str, args: dict[str, Any]) -> None:
        bid = self._new_id("e")
        journey = f"{bid}_journey"
        task = scrub_text(str(args.get("task", "")))
        self._call_to_block[call_id] = bid
        self._explore_calls.add(call_id)
        self._journey_children[bid] = []
        self._step_index[bid] = {}
        self._open_seg[bid] = None
        # One unified view: the card body is the journey Column. Each sub-agent
        # turn's reasoning is appended as its own segment in arrival order, so
        # the chronologically-last turn — the conclusion — streams at the BOTTOM
        # of the journey, not pinned to the top. A tool call ends the current
        # turn; the next turn opens a fresh segment after the step. The final
        # segment is promoted to the conclusion in place (see _on_tool_result),
        # so the conclusion is shown once, at the end.
        subtree = [
            {
                "id": bid, "component": "Card", "icon": "🔍", "title": "望诊",
                "subtitle": task, "status": {"path": f"/blocks/{bid}/status"},
                "tone": "explore", "collapsible": True, "collapsed": False, "child": journey,
            },
            {
                "id": journey, "component": "Column",
                "children": [], "tone": "journey",
            },
        ]
        await self._append_block(bid, subtree)
        await self._data(f"/blocks/{bid}/status", "exploring")

    async def _on_tool_result(self, event: dict[str, Any]) -> None:
        call_id = str(event.get("call_id"))
        bid = self._call_to_block.get(call_id)
        if bid is None:
            return
        raw = str(event.get("summary", ""))
        if call_id in self._explore_calls:
            # The explore tool's own result is the sub-agent's free-text
            # conclusion. The final turn was streamed live as the bottom-most
            # reasoning segment; promote it in place to the conclusion (body
            # styling, stable /conclusion path). The loop returns the final
            # turn verbatim, so the streamed text == the conclusion — this is a
            # style swap, not a second copy. If no segment streamed (empty
            # final turn / nothing produced), append a dedicated block instead.
            conclusion = scrub_text(raw)
            seg = self._open_seg.get(bid)
            children = self._journey_children.get(bid, [])
            if seg is not None and seg in children:
                await self._data(f"/blocks/{bid}/conclusion", conclusion)
                await self._components([{
                    "id": seg, "component": "Text",
                    "text": {"path": f"/blocks/{bid}/conclusion"},
                    "variant": "body",
                }])
            else:
                concl_id = f"{bid}_concl"
                self._journey_children[bid].append(concl_id)
                await self._components([
                    {"id": concl_id, "component": "Text",
                     "text": {"path": f"/blocks/{bid}/conclusion"}, "variant": "body"},
                    {"id": f"{bid}_journey", "component": "Column",
                     "children": list(self._journey_children[bid]), "tone": "journey"},
                ])
                await self._data(f"/blocks/{bid}/conclusion", conclusion)
            self._open_seg[bid] = None
        else:
            tool, args = self._call_to_tool.get(call_id, ("", {}))
            # Rebuild read_file/grep results as file:line references; passthrough
            # list_dir/find_files; defensively scrub anything else.
            summary = redact_tool_summary(tool, raw, args)
            await self._data(f"/blocks/{bid}/result", summary)
            await self._data(f"/blocks/{bid}/status", "done" if event.get("ok") else "failed")

    async def _on_subagent(self, event: dict[str, Any], *, status: str) -> None:
        bid = self._call_to_block.get(str(event.get("call_id")))
        if bid is not None:
            await self._data(f"/blocks/{bid}/status", status)

    async def _on_subagent_step(self, event: dict[str, Any]) -> None:
        bid = self._call_to_block.get(str(event.get("call_id")))
        if bid is None:
            return
        delta = str(event.get("delta", ""))
        seg = self._open_seg.get(bid)
        if seg is None:
            # A new sub-agent turn: open a fresh reasoning segment and append it
            # at the bottom of the journey (after any prior steps). This is what
            # keeps the final turn — the conclusion — streaming at the end
            # rather than accumulating at the top.
            seg = self._new_id(f"{bid}_r")
            self._open_seg[bid] = seg
            self._seg_text[seg] = ""
            self._journey_children[bid].append(seg)
            subtree = [{
                "id": seg, "component": "Text",
                "text": {"path": f"/blocks/{bid}/seg/{seg}"},
                "variant": "reasoning", "tone": "muted",
            }]
            journey = {"id": f"{bid}_journey", "component": "Column",
                       "children": list(self._journey_children[bid]), "tone": "journey"}
            await self._components([*subtree, journey])
        self._seg_text[seg] += delta
        await self._data(f"/blocks/{bid}/seg/{seg}", scrub_text(self._seg_text[seg]))

    async def _on_subagent_tool_call(self, event: dict[str, Any]) -> None:
        bid = self._call_to_block.get(str(event.get("call_id")))
        if bid is None:
            return
        # A tool call ends this sub-agent turn; the next reasoning delta opens a
        # new segment (appended after the step below), so turns stay chronological.
        self._open_seg[bid] = None
        sub_call_id = str(event.get("sub_call_id"))
        name = str(event.get("tool", "?"))
        args = event.get("args") or {}
        sid = self._new_id(f"{bid}_s")
        body = f"{sid}_body"
        self._step_index[bid][sub_call_id] = sid
        self._sub_args.setdefault(bid, {})[sub_call_id] = (name, args)
        subtree = [
            {
                "id": sid, "component": "Card", "icon": "↳", "title": name,
                "subtitle": _args_label(args),
                "status": {"path": f"/blocks/{bid}/steps/{sid}/status"},
                "tone": "substep", "collapsible": True, "child": body,
            },
            {
                "id": body, "component": "Text",
                "text": {"path": f"/blocks/{bid}/steps/{sid}/result"},
                "variant": "result", "tone": "muted",
            },
        ]
        self._journey_children[bid].append(sid)
        journey = {"id": f"{bid}_journey", "component": "Column",
                   "children": list(self._journey_children[bid]), "tone": "journey"}
        await self._components([*subtree, journey])
        await self._data(f"/blocks/{bid}/steps/{sid}", {"status": "running", "result": ""})

    async def _on_subagent_tool_result(self, event: dict[str, Any]) -> None:
        bid = self._call_to_block.get(str(event.get("call_id")))
        if bid is None:
            return
        idx = self._step_index.get(bid, {})
        sid = idx.get(str(event.get("sub_call_id")))
        if sid is None:
            return
        raw = str(event.get("summary", ""))
        tool, args = self._sub_args.get(bid, {}).get(str(event.get("sub_call_id")), ("", {}))
        await self._data(f"/blocks/{bid}/steps/{sid}/result", redact_tool_summary(tool, raw, args))
        await self._data(f"/blocks/{bid}/steps/{sid}/status", "done" if event.get("ok") else "failed")

    async def _on_answer(self, event: dict[str, Any]) -> None:
        # The open reasoning block was fed by the model's final turn, whose text
        # IS the answer — so it would duplicate the answer card we're about to
        # emit. If that block's accumulated text matches the answer verbatim
        # (the normal case), drop it from the root before appending the answer
        # block, so a single rebuild both removes the dup and adds the answer.
        dup_rid = self._open_reasoning
        dup_text = self._reasoning_text.get(dup_rid, "") if dup_rid else ""
        ans_text = str(event.get("text", ""))
        await self._close_reasoning()
        if dup_rid is not None and dup_rid in self._children \
                and dup_text.rstrip() == ans_text.rstrip():
            self._children.remove(dup_rid)
        bid = self._new_id("a")
        text_id = f"{bid}_text"
        chips_id = f"{bid}_chips"
        evidence = event.get("evidence") or []
        chips = [f"{e.get('file')}:{e.get('line')}" for e in evidence if e.get("file")]
        subtree = [
            {"id": bid, "component": "Card", "icon": "💡", "title": "论治",
             "status": "毕", "tone": "answer", "collapsible": False, "child": f"{bid}_col"},
            {"id": f"{bid}_col", "component": "Column", "children": [text_id, chips_id]},
            {"id": text_id, "component": "Text",
             "text": {"path": f"/blocks/{bid}/answer"}, "variant": "body", "tone": "answer"},
            {"id": chips_id, "component": "Chips", "items": chips,
             "label": "引经" if chips else ""},
        ]
        await self._append_block(bid, subtree)
        await self._data(f"/blocks/{bid}/answer", scrub_text(str(event.get("text", ""))))
        await self._emit(redact_event(event))  # native terminal signal (closes the SSE stream)

    async def _on_terminal_block(self, event: dict[str, Any], text: str, tone: str) -> None:
        await self._close_reasoning()
        bid = self._new_id("x")
        subtree = [{
            "id": bid, "component": "Text", "text": text, "variant": "body", "tone": tone,
        }]
        await self._append_block(bid, subtree)
        await self._emit(redact_event(event))  # native terminal signal
