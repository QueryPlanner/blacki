"""Mini App behaviour shared by the HTTP routes and Blacki's two tools."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from blacki.container import AppContainer, get_container

from .catalog import AGENT_HOME_PAGE, BUILTIN_PAGES, TARGET_KINDS, reference_text
from .compiler import PAGE_ID, CompileError, compile_program
from .lang import LangError, Program, parse
from .sources import SOURCES, MiniAppUser, SourceContext, resolve
from .store import BUILTIN_TITLES, MiniAppStore, StaleVersionError, builtin_source

logger = logging.getLogger(__name__)

PromptHandler = Callable[[int, str], Awaitable[None]]
Authorizer = Callable[[int], Awaitable[bool]]


@dataclass
class Hooks:
    """Callbacks the Telegram bot registers when it starts."""

    authorize: Authorizer | None = None
    send_prompt: PromptHandler | None = None


hooks = Hooks()


class ActionError(ValueError):
    """An action the user asked for could not be done."""


@dataclass
class MiniAppService:
    container: AppContainer
    store: MiniAppStore
    _builtin_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    _prompt_times: dict[int, list[float]] = field(default_factory=dict)

    # ------------------------------------------------------------ pages

    def builtin_ir(self, page_id: str) -> dict[str, Any]:
        if page_id not in self._builtin_cache:
            self._builtin_cache[page_id] = compile_program(
                parse(builtin_source(page_id))
            )
        return self._builtin_cache[page_id]

    async def agent_page(
        self, user: MiniAppUser, page_id: str
    ) -> dict[str, Any] | None:
        page = await self.store.get_page(user.chat_user_id, page_id)
        if page is None and page_id == AGENT_HOME_PAGE:
            source = builtin_source(AGENT_HOME_PAGE)
            ir = compile_program(parse(source))
            return {
                "id": page_id,
                "title": ir["title"],
                "kind": "agent",
                "version": 0,
                "source": source,
                "ir": ir,
            }
        return page

    async def load(self, user: MiniAppUser, page_id: str) -> dict[str, Any]:
        """Return a page's IR plus its resolved data for this user."""
        if page_id in BUILTIN_PAGES:
            ir = self.builtin_ir(page_id)
            meta = {
                "id": page_id,
                "title": BUILTIN_TITLES[page_id],
                "kind": "builtin",
                "version": 0,
            }
        else:
            page = await self.agent_page(user, page_id)
            if page is None:
                raise LookupError("page not found")
            ir = page["ir"]
            meta = {k: page[k] for k in ("id", "title", "kind", "version")}
        return {**meta, "ir": ir, "data": await self._resolve_all(user, ir, page_id)}

    async def load_card(self, user: MiniAppUser, card_id: str) -> dict[str, Any]:
        card = await self.store.get_card(user.chat_user_id, card_id)
        if card is None:
            raise LookupError("card not found")
        return {
            "id": card["id"],
            "title": card["title"],
            "kind": "card",
            "pinned": card["pinned"],
            "expires_at": card["expires_at"],
            "ir": card["ir"],
            "data": await self._resolve_all(user, card["ir"], f"card:{card_id}"),
        }

    async def _resolve_all(
        self, user: MiniAppUser, ir: dict[str, Any], page_id: str
    ) -> dict[str, Any]:
        ctx = SourceContext(user, self.container, self.store, page_id)
        out: dict[str, Any] = {}
        for name, query in ir.get("queries", {}).items():
            out[name] = await resolve(ctx, query["source"], query["params"])
        if any(q["source"] == "cards.pinned" for q in ir.get("queries", {}).values()):
            out["_pinned_cards"] = await self._pinned_cards(user)
        return out

    async def _pinned_cards(self, user: MiniAppUser) -> list[dict[str, Any]]:
        cards = []
        for summary in await self.store.list_cards(user.chat_user_id, pinned_only=True):
            card = await self.store.get_card(user.chat_user_id, summary["id"])
            if card is None:
                continue
            data = await self._resolve_all(user, card["ir"], f"card:{card['id']}")
            cards.append(
                {
                    "id": card["id"],
                    "title": card["title"],
                    "ir": card["ir"],
                    "data": data,
                }
            )
        return cards[:6]

    async def nav(self, user: MiniAppUser) -> list[dict[str, Any]]:
        return [
            {k: p[k] for k in ("id", "title", "kind")}
            for p in await self.store.list_pages(user.chat_user_id)
        ]

    # ------------------------------------------------------------ actions

    async def run_action(
        self, user: MiniAppUser, body: dict[str, Any]
    ) -> dict[str, Any]:
        """Run one tapped action. Saving actions only arrive from a user's tap."""
        action = str(body.get("action", ""))
        raw_args = body.get("args")
        args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
        page_id = str(body.get("page", ""))
        builtin = page_id in BUILTIN_PAGES
        if action == "SaveTarget":
            kind, value = args.get("kind"), args.get("value")
            if (
                kind not in TARGET_KINDS
                or not isinstance(value, (int, float))
                or isinstance(value, bool)
            ):
                raise ActionError("invalid target")
            if not 0 <= value <= 100000:
                raise ActionError("target is out of range")
            targets = await self.store.get_pref(user.chat_user_id, "targets") or {}
            targets[kind] = round(float(value), 1)
            await self.store.set_pref(user.chat_user_id, "targets", targets)
            return {
                "status": "ok",
                "message": f"Saved {kind.replace('_', ' ')}: {targets[kind]:g}",
            }
        if action == "Log":
            return await self._log_tracker(user, args)
        if action == "AskBlacki":
            return await self._ask_blacki(user, args)
        if action == "Pin":
            card_id = str(body.get("card", ""))
            pinned = bool(args.get("pinned", True))
            if not await self.store.set_card_pinned(user.chat_user_id, card_id, pinned):
                raise ActionError("card not found")
            return {
                "status": "ok",
                "message": "Pinned to Today" if pinned else "Unpinned",
            }
        if action == "card.delete":
            if not await self.store.delete_card(
                user.chat_user_id, str(args.get("id", ""))
            ):
                raise ActionError("card not found")
            return {"status": "ok", "message": "Card removed"}
        # Built-in widgets that change records only run on locked built-in pages, so a
        # page Blacki wrote can never put a misleading label on a destructive action.
        if (
            action in {"meal.edit", "meal.delete", "file.delete", "workout.save"}
            and not builtin
        ):
            raise ActionError("this action is only available on built-in pages")
        if action == "meal.edit":
            return await self._edit_meal(user, args)
        if action == "meal.delete":
            return await self._delete_meal(user, args)
        if action == "file.delete":
            return await self._delete_file(user, args)
        if action == "workout.save":
            return await self._save_workout(user, args)
        if action == "prefs.theme":
            theme = args.get("theme")
            if not isinstance(theme, dict) or len(json.dumps(theme)) > 4000:
                raise ActionError("invalid theme")
            await self.store.set_pref(user.chat_user_id, "theme", theme)
            return {"status": "ok"}
        raise ActionError("unknown action")

    async def _log_tracker(
        self, user: MiniAppUser, args: dict[str, Any]
    ) -> dict[str, Any]:
        template = str(args.get("template", ""))
        values = args.get("values")
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", template) or not isinstance(
            values, dict
        ):
            raise ActionError("invalid tracker entry")
        if len(values) > 12 or any(
            not isinstance(v, (int, float, str, bool))
            or (isinstance(v, str) and len(v) > 500)
            for v in values.values()
        ):
            raise ActionError("tracker values are too large")
        storage = self.container.declarative_db_storage
        await storage.initialize()
        async with self.container.conn.execute(
            "SELECT query_type FROM saved_query_templates WHERE user_id = ? AND template_name = ?",  # noqa: E501
            (user.chat_user_id, template),
        ) as cursor:
            row = await cursor.fetchone()
        # Only INSERT templates: a page must never run a stored UPDATE or DELETE.
        if row is None or str(row[0]).upper() != "INSERT":
            raise ActionError("that tracker can't be logged from the app")
        try:
            await storage.execute_query_template(user.chat_user_id, template, values)
        except ValueError as exc:
            raise ActionError(str(exc)) from exc
        return {"status": "ok", "message": "Logged"}

    async def _ask_blacki(
        self, user: MiniAppUser, args: dict[str, Any]
    ) -> dict[str, Any]:
        text = str(args.get("text", "")).strip()
        if not 1 <= len(text) <= 300:
            raise ActionError("message is empty or too long")
        if hooks.send_prompt is None:
            raise ActionError("Blacki's Telegram chat isn't running")
        now = time.monotonic()
        recent = [
            t for t in self._prompt_times.get(user.telegram_user_id, []) if now - t < 60
        ]
        if len(recent) >= 3 or (recent and now - recent[-1] < 10):
            raise ActionError("please wait a moment before asking again")
        self._prompt_times[user.telegram_user_id] = [*recent, now]
        await hooks.send_prompt(user.telegram_user_id, text)
        return {"status": "ok", "message": "Sent to Blacki in Telegram"}

    async def _owned_meal(self, user: MiniAppUser, meal_id: Any) -> tuple[int, str]:
        if not isinstance(meal_id, int) or meal_id <= 0:
            raise ActionError("invalid meal")
        async with self.container.conn.execute(
            "SELECT user_id FROM calorie_logs WHERE id = ? AND (user_id = ? OR user_id LIKE ?)",  # noqa: E501
            (meal_id, user.chat_user_id, f"{user.chat_user_id}-thread-%"),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise ActionError("meal not found")
        return meal_id, str(row[0])

    async def _edit_meal(
        self, user: MiniAppUser, args: dict[str, Any]
    ) -> dict[str, Any]:
        from blacki.calories.service import get_meal_service

        meal_id, owner = await self._owned_meal(user, args.get("id"))
        updates: dict[str, Any] = {}
        raw_changes = args.get("changes")
        changes: dict[str, Any] = raw_changes if isinstance(raw_changes, dict) else {}
        for key, column, cast in (
            ("kcal", "calories", int),
            ("protein", "protein_g", float),
            ("carbs", "carbs_g", float),
            ("fat", "fat_g", float),
        ):
            if key in changes:
                value = changes[key]
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not 0 <= value <= 10000
                ):
                    raise ActionError(f"{key} is out of range")
                updates[column] = cast(value)
        if "name" in changes:
            name = str(changes["name"]).strip()
            if not 1 <= len(name) <= 300:
                raise ActionError("meal name is empty or too long")
            updates["description"] = name
        if not updates:
            raise ActionError("nothing to save")
        try:
            _, status = await get_meal_service().mutate(
                owner, private=True, entry_id=meal_id, updates=updates
            )
        except ValueError as exc:
            raise ActionError(str(exc)) from exc
        return {"status": "ok", "message": "Meal saved", "sync": status}

    async def _delete_meal(
        self, user: MiniAppUser, args: dict[str, Any]
    ) -> dict[str, Any]:
        from blacki.calories.service import get_meal_service

        meal_id, owner = await self._owned_meal(user, args.get("id"))
        try:
            _, status = await get_meal_service().mutate(
                owner, private=True, entry_id=meal_id
            )
        except ValueError as exc:
            raise ActionError(str(exc)) from exc
        return {"status": "ok", "message": "Meal deleted", "sync": status}

    async def _delete_file(
        self, user: MiniAppUser, args: dict[str, Any]
    ) -> dict[str, Any]:
        from blacki.user_files.service import get_user_file_service

        object_id = str(args.get("id", ""))
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", object_id):
            raise ActionError("invalid file")
        try:
            deleted = await get_user_file_service().delete(user.sender_id, object_id)
        except RuntimeError as exc:
            raise ActionError(str(exc)) from exc
        except Exception as exc:
            logger.exception("Mini App file delete failed")
            raise ActionError(
                "the file may be only partly deleted; check Files again in a moment"
            ) from exc
        if not deleted:
            raise ActionError("file not found")
        return {"status": "ok", "message": "File deleted"}

    async def _save_workout(
        self, user: MiniAppUser, args: dict[str, Any]
    ) -> dict[str, Any]:
        from blacki.tools.workouts import log_training_for_user

        name = str(args.get("exercise", "")).strip()
        sets = args.get("sets")
        if (
            not 1 <= len(name) <= 60
            or not isinstance(sets, list)
            or not 1 <= len(sets) <= 10
        ):
            raise ActionError("invalid workout")
        clean = []
        for item in sets:
            if not isinstance(item, dict):
                raise ActionError("invalid set")
            kg, reps = item.get("kg"), item.get("reps")
            if (
                not isinstance(kg, (int, float))
                or not isinstance(reps, int)
                or not 0 <= kg <= 500
                or not 1 <= reps <= 100
            ):
                raise ActionError("sets need kg 0..500 and reps 1..100")
            clean.append({"weight_kg": float(kg), "reps": reps})
        result = await log_training_for_user(
            user.chat_user_id,
            session_type="resistance",
            exercises=[{"name": name.lower(), "sets": clean}],
            notes="Logged from the Mini App",
        )
        if result.get("status") != "success":
            raise ActionError(str(result.get("message", "could not save the workout")))
        return {"status": "ok", "message": f"Saved {len(clean)} sets of {name}"}

    # ------------------------------------------------------------ Blacki's tools

    async def tool_read(self, user: MiniAppUser, code: str) -> dict[str, Any]:
        """Answer one read request from Blacki."""
        text = code.strip()
        if text in {"", "help"}:
            return {
                "status": "ok",
                "commands": [
                    "pages",
                    "page <id>",
                    "history <id>",
                    "version <id> <n>",
                    "cards",
                    "card <id>",
                    "sources",
                    "catalog",
                    'Query("health.sleep", {days: 7})',
                ],
            }
        if text == "catalog":
            return {"status": "ok", "catalog": reference_text()}
        if text == "sources":
            return {
                "status": "ok",
                "sources": [
                    {
                        "name": s.name,
                        "doc": s.doc,
                        "fields": list(s.fields or ["(from the template)"]),
                        "settings": {k: _param_text(p) for k, p in s.params.items()},
                    }
                    for s in SOURCES.values()
                    if s.public
                ],
            }
        if text == "pages":
            return {
                "status": "ok",
                "pages": await self.store.list_pages(user.chat_user_id),
            }
        if text == "cards":
            return {
                "status": "ok",
                "cards": await self.store.list_cards(user.chat_user_id),
            }
        match = re.fullmatch(r"(page|history|card)\s+([A-Za-z0-9_-]{1,40})", text)
        if match:
            verb, ident = match.groups()
            if verb == "card":
                card = await self.store.get_card(user.chat_user_id, ident)
                if card is None:
                    return {"status": "error", "error": "card not found"}
                return {
                    "status": "ok",
                    "card": {k: card[k] for k in ("id", "title", "source", "pinned")},
                }
            if verb == "history":
                return {
                    "status": "ok",
                    "versions": await self.store.history(user.chat_user_id, ident),
                }
            if ident in BUILTIN_PAGES:
                return {
                    "status": "ok",
                    "page": {
                        "id": ident,
                        "kind": "builtin",
                        "locked": True,
                        "source": builtin_source(ident),
                    },
                }
            page = await self.agent_page(user, ident)
            if page is None:
                return {"status": "error", "error": "page not found"}
            return {
                "status": "ok",
                "page": {
                    k: page[k] for k in ("id", "title", "kind", "version", "source")
                },
            }
        match = re.fullmatch(r"version\s+([a-z][a-z0-9-]{1,31})\s+(\d{1,4})", text)
        if match:
            row = await self.store.version_source(
                user.chat_user_id, match.group(1), int(match.group(2))
            )
            return (
                {"status": "ok", "version": row}
                if row
                else {"status": "error", "error": "version not found"}
            )
        if text.startswith("Query("):
            return await self._peek(user, text)
        return {"status": "error", "error": "unknown read command; try  help"}

    async def _peek(self, user: MiniAppUser, text: str) -> dict[str, Any]:
        try:
            program = parse(f"peek = {text}")
            ir = compile_program(_wrap_peek(program), keep_all_queries=True)
        except (LangError, CompileError) as exc:
            return {"status": "error", "error": str(exc)}
        query = ir["queries"]["peek"]
        result = await resolve(
            SourceContext(user, self.container, self.store, "peek"),
            query["source"],
            query["params"],
        )
        rows = result.get("rows", [])
        return {**result, "rows": rows[-20:], "row_count": len(rows)}

    async def tool_write(
        self, user: MiniAppUser, code: str, base_version: int | None
    ) -> dict[str, Any]:
        """Apply one write script from Blacki and report what happened."""
        try:
            blocks = _split_blocks(code)
        except LangError as exc:
            return {"status": "error", "errors": [str(exc)]}
        results: list[dict[str, Any]] = []
        for block in blocks:
            try:
                results.append(await self._run_block(user, block, base_version))
            except (LangError, CompileError) as exc:
                errors = exc.errors if isinstance(exc, CompileError) else [exc]
                lines = [_shift(e, block.get("offset", 0)) for e in errors]
                results.append(
                    {"status": "error", "target": block.get("target"), "errors": lines}
                )
                break
            except (StaleVersionError, ValueError, LookupError) as exc:
                results.append(
                    {
                        "status": "error",
                        "target": block.get("target"),
                        "errors": [str(exc)],
                    }
                )
                break
        ok = all(r["status"] == "ok" for r in results)
        return {"status": "ok" if ok else "error", "results": results}

    async def _run_block(
        self, user: MiniAppUser, block: dict[str, Any], base_version: int | None
    ) -> dict[str, Any]:
        owner = user.chat_user_id
        command = block["command"]
        if command == "delete":
            page_id = block["target"]
            if page_id in BUILTIN_PAGES or page_id == AGENT_HOME_PAGE:
                raise ValueError(f"{page_id} can't be deleted")
            if not await self.store.delete_page(owner, page_id):
                raise LookupError(f"page {page_id} not found")
            return {"status": "ok", "target": page_id, "message": "deleted"}
        if command == "move":
            if not await self.store.move_page(
                owner, block["target"], block["position"]
            ):
                raise LookupError(f"page {block['target']} not found")
            return {"status": "ok", "target": block["target"], "message": "moved"}
        if command == "revert":
            page_id, version = block["target"], block["version"]
            row = await self.store.version_source(owner, page_id, version)
            if row is None:
                raise LookupError(f"{page_id} has no version {version}")
            ir = compile_program(parse(row["source"]))
            new_version = await self.store.save_page(
                owner,
                page_id,
                title=ir["title"],
                source=row["source"],
                ir=ir,
                base_version=base_version,
                created_by="blacki",
                note=f"revert to {version}",
            )
            return {"status": "ok", "target": page_id, "version": new_version}
        if command == "card":
            program = parse(block["body"])
            if block.get("title"):
                _retitle(program, block["title"])
            ir = compile_program(program, kind="card")
            await self._check_templates(user, ir)
            source = program.to_source()
            card_id = await self.store.create_card(
                owner, title=ir["title"], source=source, ir=ir
            )
            card_result: dict[str, Any] = {
                "status": "ok",
                "target": "card",
                "card_id": card_id,
                "title": ir["title"],
            }
            if ir.get("unused"):
                card_result["warning"] = "not shown on the card: " + ", ".join(
                    ir["unused"]
                )
            return card_result
        # page create or edit
        page_id = block["target"]
        if page_id in BUILTIN_PAGES:
            raise ValueError(f"{page_id} is a built-in page and can't be edited")
        existing = await self.agent_page(user, page_id)
        if existing is not None and existing["version"] > 0 and base_version is None:
            raise ValueError(
                f"page {page_id} exists at version {existing['version']}; read it and pass base_version"  # noqa: E501
            )
        program = (
            Program()
            if existing is None or block["replace"]
            else parse(existing["source"])
        )
        incoming = parse(block["body"]) if block["body"].strip() else Program()
        for key in incoming.order:
            program.set(incoming.statements[key])
        for name in block["remove"]:
            if not program.remove(name):
                raise ValueError(f"there is no line named {name} to remove")
        if block.get("title"):
            _retitle(program, block["title"])
        source = program.to_source()
        ir = compile_program(parse(source))
        await self._check_templates(user, ir)
        version = await self.store.save_page(
            owner,
            page_id,
            title=ir["title"],
            source=source,
            ir=ir,
            base_version=(existing["version"] if existing else 0)
            if base_version is None
            else base_version,
            created_by="blacki",
            note=block.get("note"),
        )
        page_result: dict[str, Any] = {
            "status": "ok",
            "target": page_id,
            "version": version,
            "title": ir["title"],
        }
        if ir.get("unused"):
            page_result["warning"] = "not shown on the page: " + ", ".join(ir["unused"])
        return page_result

    async def _check_templates(self, user: MiniAppUser, ir: dict[str, Any]) -> None:
        """@Log needs an existing INSERT template and tracker reads a SELECT one."""
        needed: list[tuple[str, str]] = []
        for q in ir.get("queries", {}).values():
            if q["source"] == "tracker":
                needed.append((q["params"]["template"], "SELECT"))
        for action in _walk_actions(ir.get("root")):
            if action.get("a") == "Log":
                needed.append((action["args"]["template"], "INSERT"))
        if not needed:
            return
        await self.container.declarative_db_storage.initialize()
        for name, kind in needed:
            async with self.container.conn.execute(
                "SELECT query_type FROM saved_query_templates WHERE user_id = ? AND template_name = ?",  # noqa: E501
                (user.chat_user_id, name),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None or str(row[0]).upper() != kind:
                raise ValueError(
                    f"template {name!r} must exist and be a {kind} template"
                )


def _param_text(p: Any) -> str:
    if p.kind == "enum":
        return "one of " + ", ".join(p.options)
    if p.kind == "int":
        return f"whole number {p.min}..{p.max}, default {p.default}"
    return "lowercase name"


def _wrap_peek(program: Program) -> Program:
    """Wrap a lone Query so it compiles: root = Page("peek", [])."""
    wrapper = parse('root = Page("peek", [])')
    for key in program.order:
        wrapper.set(program.statements[key])
    return wrapper


def _walk_actions(node: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if "a" in node and "args" in node:
            found.append(node)
        for value in node.values():
            found += _walk_actions(value)
    elif isinstance(node, list):
        for item in node:
            found += _walk_actions(item)
    return found


def _retitle(program: Program, title: str) -> None:
    from .lang import Call, Literal, Statement

    root = program.statements.get("root")
    if root is None or not isinstance(root.expr, Call) or root.expr.name != "Page":
        return
    args = list(root.expr.args)
    if args:
        args[0] = Literal(root.line, title)
    program.set(
        Statement(
            "root", Call(root.line, "Page", tuple(args), root.expr.kwargs), root.line
        )
    )


def _shift(error: LangError, offset: int) -> str:
    return (
        f"line {error.line + offset}: {error.message}" if error.line else error.message
    )


_COMMAND = re.compile(
    r"^(?:(page)\s+([a-z][a-z0-9-]{1,31})(?:\s+\"([^\"]{1,60})\")?(\s+replace)?"
    r"|(card)(?:\s+\"([^\"]{1,60})\")?"
    r"|(remove)\s+([A-Za-z_][A-Za-z0-9_\s,$]*)"
    r"|(delete)\s+page\s+([a-z][a-z0-9-]{1,31})"
    r"|(revert)\s+([a-z][a-z0-9-]{1,31})\s+to\s+(\d{1,4})"
    r"|(move)\s+([a-z][a-z0-9-]{1,31})\s+to\s+(\d{1,3})"
    r"|(note)\s+\"([^\"]{1,120})\")\s*$"
)


def _split_blocks(code: str) -> list[dict[str, Any]]:
    """Split a write script into command blocks."""
    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    body: list[str] = []

    def close() -> None:
        if current is not None:
            current["body"] = "\n".join(body)
            blocks.append(current)

    for index, raw in enumerate(code.splitlines()):
        line = raw.strip()
        match = (
            _COMMAND.match(line) if line and "=" not in line.split("#", 1)[0] else None
        )
        if match is None:
            if current is None:
                if line and not line.startswith("#"):
                    raise LangError(
                        'start with a command such as  page recovery "Recovery"  or  card "Title"',  # noqa: E501
                        index + 1,
                    )
                continue
            body.append(raw)
            continue
        g = match.groups()
        if g[0]:  # page
            close()
            current, body = (
                {
                    "command": "page",
                    "target": g[1],
                    "title": g[2],
                    "replace": bool(g[3]),
                    "remove": [],
                    "offset": index + 1,
                },
                [],
            )
            if not PAGE_ID.match(g[1]):
                raise LangError(
                    "page ids are lowercase letters, digits and dashes", index + 1
                )
        elif g[4]:  # card
            close()
            current, body = (
                {
                    "command": "card",
                    "target": "card",
                    "title": g[5],
                    "remove": [],
                    "offset": index + 1,
                },
                [],
            )
        elif g[6]:  # remove
            if current is None or current["command"] != "page":
                raise LangError("remove belongs inside a page block", index + 1)
            names = [n for n in re.split(r"[\s,]+", g[7]) if n]
            current["remove"].extend(names)
        elif g[8]:
            close()
            current, body = None, []
            blocks.append({"command": "delete", "target": g[9]})
        elif g[10]:
            close()
            current, body = None, []
            blocks.append({"command": "revert", "target": g[11], "version": int(g[12])})
        elif g[13]:
            close()
            current, body = None, []
            blocks.append({"command": "move", "target": g[14], "position": int(g[15])})
        elif g[16]:
            if current is None:
                raise LangError("note belongs inside a page block", index + 1)
            current["note"] = g[17]
    close()
    if not blocks:
        raise LangError(
            'nothing to do; start with  page <id> "Title"  or  card "Title"'
        )
    return blocks


_service: MiniAppService | None = None


def get_miniapp_service() -> MiniAppService:
    global _service
    if _service is None:
        container = get_container()
        _service = MiniAppService(container, container.miniapp_store)
    return _service


def reset_miniapp_service() -> None:
    global _service
    _service = None
