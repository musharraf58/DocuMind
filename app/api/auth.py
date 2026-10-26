from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from redis.asyncio import Redis

from app.api.deps import CurrentUser, DbSession, StorageDep
from app.api.limits import (
    Limiter,
    enforce_login_email,
    limit_account,
    limit_default,
    limit_login,
    limit_register,
)
from app.core.client_ip import client_ip
from app.core.errors import ApiError
from app.core.redis import get_redis
from app.core.security import create_access_token, verify_password
from app.schemas.auth import Token
from app.schemas.user import ChangePassword, DeleteAccount, UserCreate, UserRead
from app.services import users as user_service
from app.services.login_guard import LoginGuard

router = APIRouter(prefix="/auth", tags=["auth"])

INCORRECT_LOGIN = "Incorrect email or password"


def get_login_guard(redis: Annotated[Redis, Depends(get_redis)]) -> LoginGuard:
    return LoginGuard(redis)


Guard = Annotated[LoginGuard, Depends(get_login_guard)]


@router.post(
    "/register",
    response_model=UserRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[limit_register],
)
async def register(payload: UserCreate, db: DbSession):
    user_service.check_registration_allowed(payload.invite_code)
    return await user_service.register_user(db, payload.email, payload.password)


@router.post("/login", response_model=Token, dependencies=[limit_login])
async def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    request: Request,
    db: DbSession,
    limiter: Limiter,
    guard: Guard,
    response: Response,
):
    # OAuth2 form fields are named "username"/"password"; username holds the email.
    email, ip = user_service.normalize_email(form.username), client_ip(request)
    await enforce_login_email(limiter, email, response)
    # Before the (deliberately slow) password check: a blocked attempt costs
    # us almost nothing.
    await guard.check(email, ip)
    user = await user_service.authenticate(db, email, form.password)
    if user is None:
        await guard.failed(email, ip)
        # One message for both "no such email" and "wrong password".
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=INCORRECT_LOGIN,
            headers={"WWW-Authenticate": "Bearer"},
        )
    await guard.succeeded(email, ip)
    return Token(access_token=create_access_token(user.id, user.token_version))


@router.get("/me", response_model=UserRead, dependencies=[limit_default])
async def me(current_user: CurrentUser):
    return current_user


@router.post("/change-password", response_model=Token, dependencies=[limit_account])
async def change_password(
    body: ChangePassword, request: Request, user: CurrentUser, db: DbSession, guard: Guard
):
    """Change your password. Every token issued before this (on any device)
    stops working; the response carries a fresh one for this client."""
    # A stolen token must not become a way to guess the password: same
    # progressive delay as login.
    ip = client_ip(request)
    await guard.check(user.email, ip)
    if not verify_password(body.current_password, user.hashed_password):
        await guard.failed(user.email, ip)
        raise ApiError(403, "Current password is incorrect", code="wrong_password")
    await guard.succeeded(user.email, ip)
    version = await user_service.change_password(db, user, body.new_password)
    return Token(access_token=create_access_token(user.id, version))


@router.delete("/me", status_code=status.HTTP_204_NO_CONTENT, dependencies=[limit_account])
async def delete_me(
    body: DeleteAccount,
    user: CurrentUser,
    db: DbSession,
    storage: StorageDep,
    redis: Annotated[Redis, Depends(get_redis)],
):
    """Delete your account and everything stored for it: collections,
    documents (rows AND files), conversations, usage history and rate-limit
    counters. Requires your password. Cannot be undone."""
    await user_service.delete_account(db, storage, redis, user, body.password)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
