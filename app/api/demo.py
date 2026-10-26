"""The demo account is read-only.

One shared account (users.is_demo, created by `python -m scripts.seed_demo`)
lets visitors try DocuMind without registering. Its password is public in
effect, so everything that changes or destroys shared state is refused:

- uploading, deleting or reprocessing documents;
- creating, renaming or deleting collections (a rename would show every
  visitor whatever the last one typed);
- renaming or deleting conversations (they are shared by all visitors);
- changing the password (it would lock every other visitor out) and
  deleting the account (it would delete the demo for everyone).

Asking, searching and reading stay allowed, and the demo account's daily
quotas and rate limits still apply: the global LLM budget is shared with
real users, so the demo can't spend it all.
"""

from fastapi import Depends

from app.api.deps import CurrentUser
from app.core.errors import ApiError

DEMO_READ_ONLY = (
    "The demo account is read-only: it can ask, search and read. "
    "Create your own account to upload and change documents."
)


async def refuse_demo(user: CurrentUser) -> None:
    if user.is_demo:
        raise ApiError(403, DEMO_READ_ONLY, code="demo_read_only")


not_demo = Depends(refuse_demo)
